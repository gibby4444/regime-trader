"""Walk-forward allocation backtester.

Each window trains a fresh HMM (with BIC model selection) on ``train_window``
in-sample bars, ranks its regimes by volatility, then walks the next
``test_window`` out-of-sample bars one at a time:

1. fill any order scheduled for this bar at its open (``fill_delay_bars`` after the signal);
2. mark the portfolio to the close;
3. update the risk manager with the marked equity;
4. advance the filtered HMM forward algorithm with this bar's features, ask the
   strategy for a target allocation, apply risk rules, and schedule a rebalance
   only if the target differs from the current allocation by more than
   ``rebalance_threshold``.

Windows advance by ``step_size``; the portfolio carries across windows so the
out-of-sample segments form one continuous equity curve. Features are computed
with trailing windows only (verified prefix-invariant in ``tests/test_look_ahead.py``),
so they may use history before the in-sample window for warm-up.

Allocation math per rebalance, for each symbol with target weight ``w``::

    equity        = cash + sum(shares * price)
    target_shares = int(equity * w / price)
    delta         = target_shares - shares
    cash         -= delta * price + |delta| * price * slippage_pct + commission
    shares        = target_shares

``price`` is the fill bar's open. Weights above 1 (leverage) drive cash negative
(margin); the equity identity still holds.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
import pandas as pd

from core.hmm_engine import HMMConfig, HMMEngine, RegimeState
from core.regime_strategies import StrategyConfig, StrategyOrchestrator
from core.risk_manager import RiskConfig, RiskManager, TradingState
from data.feature_engineering import FeatureConfig, FeatureEngineer, log_returns, validate_bars

logger = logging.getLogger("regime_trader.backtest")


@dataclass
class BacktestConfig:
    """Parameters mirroring the ``backtest`` section of settings.yaml.

    ``min_train_bars`` overrides ``hmm.min_train_bars`` for walk-forward fits so the
    in-sample window can be shorter than the live training minimum.
    ``rebalance_threshold`` defaults to ``strategy.rebalance_threshold``.
    """

    slippage_pct: float = 0.0005
    initial_capital: float = 100_000.0
    train_window: int = 252
    test_window: int = 126
    step_size: int = 126
    risk_free_rate: float = 0.045
    min_train_bars: int = 252
    rebalance_threshold: float = 0.10
    fill_delay_bars: int = 1
    commission_per_trade: float = 0.0

    def __post_init__(self) -> None:
        if self.train_window < self.min_train_bars:
            raise ValueError(
                f"train_window ({self.train_window}) must be >= min_train_bars ({self.min_train_bars})"
            )
        if min(self.test_window, self.step_size, self.fill_delay_bars) < 1:
            raise ValueError("test_window, step_size and fill_delay_bars must be >= 1")

    @classmethod
    def from_dict(
        cls, cfg: dict[str, Any] | None, strategy_cfg: dict[str, Any] | None = None
    ) -> BacktestConfig:
        """Build a config from the parsed ``backtest`` settings section."""
        params = dict(cfg or {})
        unknown = set(params) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown backtest config keys: {sorted(unknown)}")
        if strategy_cfg and "rebalance_threshold" in strategy_cfg:
            params.setdefault("rebalance_threshold", strategy_cfg["rebalance_threshold"])
        return cls(**params)


@dataclass
class Fold:
    """Boundaries and fitted-model summary of one walk-forward window."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int = 0
    n_test: int = 0
    n_states: int | None = None
    bic: float | None = None
    candidate_bic: dict[int, float] = field(default_factory=dict)


@dataclass
class TargetSignal:
    """Output of a signal source for one bar.

    ``weights`` maps symbol -> target fraction of equity (None = no opinion, hold).
    ``meta`` is recorded verbatim in the per-bar history.
    """

    weights: dict[str, float] | None
    meta: dict[str, Any] = field(default_factory=dict)


SignalFn = Callable[[int, pd.Timestamp], TargetSignal]


@dataclass
class SimulationResult:
    """Raw output of :func:`run_simulation`."""

    equity: pd.Series
    allocation: pd.Series
    target_allocation: pd.Series
    risk_state: pd.Series
    weights: pd.DataFrame
    history: pd.DataFrame
    fills: pd.DataFrame
    trades: pd.DataFrame


@dataclass
class BacktestResult:
    """Stitched out-of-sample results.

    ``regimes`` is the confirmed HMM label per bar; ``regime_history`` holds the
    full per-bar record (regime id, vol regime, probability, allocation, risk state).
    ``trades`` are holding segments between consecutive rebalances.
    """

    equity_curve: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    regimes: pd.Series
    trades: pd.DataFrame
    folds: list[Fold] = field(default_factory=list)
    fills: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocation: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    initial_capital: float = 100_000.0
    engines: list[HMMEngine] = field(default_factory=list, repr=False)


class Portfolio:
    """Cash plus integer share holdings."""

    def __init__(self, cash: float, symbols: list[str]) -> None:
        self.cash = float(cash)
        self.shares: dict[str, int] = {s: 0 for s in symbols}

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(n * prices[s] for s, n in self.shares.items())

    def exposure(self, prices: dict[str, float]) -> float:
        """Gross long exposure as a fraction of equity."""
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        return sum(n * prices[s] for s, n in self.shares.items()) / eq

    def rebalance(
        self,
        prices: dict[str, float],
        weights: dict[str, float],
        slippage_pct: float = 0.0,
        commission: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Trade to ``weights`` at ``prices``; return one record per symbol traded."""
        equity = self.equity(prices)
        records = []
        for sym in self.shares:
            w = weights.get(sym, 0.0)
            price = prices[sym]
            target_shares = int(equity * w / price) if equity > 0 else 0
            delta = target_shares - self.shares[sym]
            if delta == 0:
                continue
            slip = abs(delta) * price * slippage_pct
            self.cash -= delta * price + slip + commission
            records.append({
                "symbol": sym,
                "shares_before": self.shares[sym],
                "shares_after": target_shares,
                "delta": delta,
                "price": price,
                "slippage_cost": slip,
                "commission": commission,
                "target_weight": w,
            })
            self.shares[sym] = target_shares
        return records


def apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    """Effective fill price after adverse slippage (``side`` = 'buy' | 'sell')."""
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    return price * (1.0 + slippage_pct) if side == "buy" else price * (1.0 - slippage_pct)


def align_bars(bars_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Validate each symbol's bars and restrict all to their common timestamps."""
    if not bars_by_symbol:
        raise ValueError("bars_by_symbol is empty")
    clean = {s: validate_bars(b) for s, b in bars_by_symbol.items()}
    common = None
    for b in clean.values():
        common = b.index if common is None else common.intersection(b.index)
    assert common is not None
    return {s: b.loc[common] for s, b in clean.items()}


def _apply_risk(
    weights: dict[str, float] | None,
    state: TradingState | None,
    current: float,
    risk_config: RiskConfig | None,
) -> dict[str, float] | None:
    if weights is None or risk_config is None or state is None:
        return weights
    if state is TradingState.KILLED:
        return {s: 0.0 for s in weights}
    out = dict(weights)
    if state is TradingState.REDUCED:
        out = {s: w * risk_config.reduce_size_mult for s, w in out.items()}
    gross = sum(out.values())
    if gross > risk_config.max_leverage:
        out = {s: w * risk_config.max_leverage / gross for s, w in out.items()}
        gross = risk_config.max_leverage
    if state is TradingState.HALTED and gross > current:
        return None
    return out


def run_simulation(
    bars_by_symbol: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    signal_fn: SignalFn,
    config: BacktestConfig,
    risk_config: RiskConfig | None = None,
    rearm_at: set[pd.Timestamp] | frozenset[pd.Timestamp] = frozenset(),
) -> SimulationResult:
    """Walk ``index`` bar by bar, trading on ``signal_fn`` targets.

    The signal at bar ``t`` (computed after the close) fills at the open of bar
    ``t + fill_delay_bars``. ``risk_config=None`` disables the risk overlay.
    ``rearm_at`` lists bars where a tripped kill-switch is cleared.
    """
    symbols = list(bars_by_symbol)
    opens = {s: bars_by_symbol[s]["open"].reindex(index).to_numpy(dtype=float) for s in symbols}
    closes = {s: bars_by_symbol[s]["close"].reindex(index).to_numpy(dtype=float) for s in symbols}
    if any(np.isnan(a).any() for a in (*opens.values(), *closes.values())):
        raise ValueError("bars missing for some simulation timestamps")

    portfolio = Portfolio(config.initial_capital, symbols)
    risk = RiskManager(risk_config) if risk_config is not None else None
    pending: list[tuple[int, dict[str, float], pd.Timestamp]] = []
    n = len(index)
    equity = np.empty(n)
    alloc = np.empty(n)
    target_alloc = np.full(n, np.nan)
    weights_rows: list[dict[str, float]] = []
    states: list[str] = []
    history: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []

    for pos, ts in enumerate(index):
        open_px = {s: opens[s][pos] for s in symbols}
        close_px = {s: closes[s][pos] for s in symbols}

        while pending and pending[0][0] == pos:
            _, w, sig_ts = pending.pop(0)
            eq_before = portfolio.equity(open_px)
            records = portfolio.rebalance(open_px, w, config.slippage_pct, config.commission_per_trade)
            eq_after = portfolio.equity(open_px)
            for r in records:
                fills.append({
                    "timestamp": ts, "signal_timestamp": sig_ts, "bar": pos,
                    "equity_before": eq_before, "equity_after": eq_after,
                    "exposure_after": portfolio.exposure(open_px), **r,
                })

        eq = portfolio.equity(close_px)
        cur = portfolio.exposure(close_px)
        equity[pos], alloc[pos] = eq, cur
        weights_rows.append(
            {s: (portfolio.shares[s] * close_px[s] / eq if eq > 0 else 0.0) for s in symbols}
        )

        state = None
        if risk is not None:
            risk.update_equity(eq, ts)
            if ts in rearm_at:
                risk.rearm(eq)
            state = risk.evaluate_state(eq)
        states.append(state.value if state is not None else "off")

        sig = signal_fn(pos, ts)
        target = _apply_risk(sig.weights, state, cur, risk_config)
        if eq <= 0 and target is not None:
            target = {s: 0.0 for s in target}
        history.append({"timestamp": ts, **sig.meta})
        if target is None:
            continue
        gross = sum(target.values())
        target_alloc[pos] = gross
        reference = sum(pending[-1][1].values()) if pending else cur
        force_flat = state is TradingState.KILLED and any(portfolio.shares.values()) and not pending
        exec_pos = pos + config.fill_delay_bars
        if exec_pos < n and (abs(gross - reference) > config.rebalance_threshold + 1e-9 or force_flat):
            pending.append((exec_pos, target, ts))

    equity_s = pd.Series(equity, index=index, name="equity")
    fills_df = pd.DataFrame(fills)
    hist_df = pd.DataFrame(history).set_index("timestamp") if history else pd.DataFrame(index=index)
    return SimulationResult(
        equity=equity_s,
        allocation=pd.Series(alloc, index=index, name="allocation"),
        target_allocation=pd.Series(target_alloc, index=index, name="target_allocation").ffill(),
        risk_state=pd.Series(states, index=index, name="risk_state"),
        weights=pd.DataFrame(weights_rows, index=index),
        history=hist_df,
        fills=fills_df,
        trades=build_trades(fills_df, equity_s, hist_df),
    )


def build_trades(fills: pd.DataFrame, equity: pd.Series, history: pd.DataFrame) -> pd.DataFrame:
    """Holding segments between consecutive rebalances.

    Segment ``i`` runs from rebalance ``i`` to rebalance ``i + 1`` (or the final
    bar). P&L = equity just before the next rebalance minus equity just before this
    one, so it includes this rebalance's slippage and segment P&Ls sum to the total.
    Regime and confidence are taken from the signal bar that triggered the rebalance.
    """
    cols = [
        "entry_time", "exit_time", "signal_time", "holding_bars", "allocation",
        "pnl", "return", "regime", "vol_regime", "confidence", "strategy", "is_flat",
    ]
    if fills.empty:
        return pd.DataFrame(columns=cols)
    events = fills.groupby("bar", sort=True).agg(
        timestamp=("timestamp", "first"),
        signal_timestamp=("signal_timestamp", "first"),
        equity_before=("equity_before", "first"),
        exposure=("exposure_after", "last"),
    )
    bars = events.index.to_numpy()
    eq_before = events["equity_before"].to_numpy()
    n = len(equity)
    rows = []
    for i, bar in enumerate(bars):
        last = i + 1 >= len(bars)
        exit_bar = n - 1 if last else bars[i + 1]
        end_eq = float(equity.iloc[-1]) if last else eq_before[i + 1]
        pnl = end_eq - eq_before[i]
        sig_ts = events["signal_timestamp"].iloc[i]
        meta = history.loc[sig_ts] if sig_ts in history.index else pd.Series(dtype=object)
        expo = float(events["exposure"].iloc[i])
        rows.append({
            "entry_time": events["timestamp"].iloc[i],
            "exit_time": equity.index[exit_bar],
            "signal_time": sig_ts,
            "holding_bars": int(exit_bar - bar + (1 if last else 0)),
            "allocation": expo,
            "pnl": pnl,
            "return": pnl / eq_before[i] if eq_before[i] else np.nan,
            "regime": meta.get("regime"),
            "vol_regime": meta.get("vol_regime"),
            "confidence": meta.get("probability", np.nan),
            "strategy": meta.get("strategy"),
            "is_flat": abs(expo) < 1e-9,
        })
    return pd.DataFrame(rows, columns=cols)


class RegimeSignalSource:
    """Bar-by-bar HMM regime + strategy targets for the walk-forward windows.

    On the first out-of-sample bar of each window it fits (or reuses) that window's
    engine on the in-sample features, warms the forward filter through them, and
    rebuilds the orchestrator's volatility ranking from ``regime_info``. Each call
    then advances the filter by exactly one bar.
    """

    def __init__(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        features: pd.DataFrame,
        returns: pd.Series,
        folds: list[Fold],
        hmm_config: HMMConfig,
        strategy_config: StrategyConfig,
        engines: list[HMMEngine] | None = None,
        state_transform: Callable[[int, pd.Timestamp, RegimeState, HMMEngine], RegimeState] | None = None,
    ) -> None:
        self.bars = bars_by_symbol
        self.symbols = list(bars_by_symbol)
        self.features = features
        self.returns = returns
        self.folds = folds
        self.hmm_config = hmm_config
        self.strategy_config = strategy_config
        self.engines: list[HMMEngine | None] = list(engines) if engines is not None else [None] * len(folds)
        if len(self.engines) != len(folds):
            raise ValueError("engines must align 1:1 with folds")
        self.state_transform = state_transform
        primary = bars_by_symbol[self.symbols[0]]
        self._bar_pos = pd.Series(np.arange(len(primary)), index=primary.index)
        self._fold_of: dict[pd.Timestamp, int] = {}
        for k, f in enumerate(folds):
            for ts in features.loc[f.test_start:f.test_end].index:
                self._fold_of[ts] = k
        self._active = -1
        self._engine: HMMEngine | None = None
        self._orch: StrategyOrchestrator | None = None

    def _activate(self, k: int) -> None:
        f = self.folds[k]
        train = self.features.loc[f.train_start:f.train_end]
        engine = self.engines[k]
        if engine is None:
            engine = HMMEngine(self.hmm_config).fit(
                train, self.returns.reindex(train.index), training_date=f.train_end
            )
            self.engines[k] = engine
            logger.info(
                "Window %d: trained on %d bars %s..%s -> n_states=%d",
                k, len(train), f.train_start.date(), f.train_end.date(), engine.n_states,
            )
        f.n_train, f.n_states = len(train), engine.n_states
        f.bic = engine.metadata.get("bic")
        f.candidate_bic = dict(engine.metadata.get("candidate_bic", {}))
        engine.run_filter(train)
        self._engine, self._active = engine, k
        self._orch = StrategyOrchestrator.from_engine(self.strategy_config, engine)

    def __call__(self, pos: int, ts: pd.Timestamp) -> TargetSignal:
        k = self._fold_of[ts]
        if k != self._active:
            self._activate(k)
        engine, orch = self._engine, self._orch
        assert engine is not None and orch is not None
        state = engine.update(self.features.loc[ts], ts, log_events=False)
        if self.state_transform is not None:
            state = self.state_transform(k, ts, state, engine)
        flicker = engine.is_flickering()
        loc = int(self._bar_pos[ts])
        bars = {s: self.bars[s].iloc[: loc + 1] for s in self.symbols}
        signals = orch.generate_signals(self.symbols, bars, state, is_flickering=flicker)
        meta = {
            "window": k,
            "n_states": engine.n_states,
            "regime": state.label,
            "regime_id": state.state_id,
            "raw_regime_id": state.raw_state_id,
            "vol_regime": orch.vol_regime[state.state_id].value,
            "vol_rank": orch.vol_rank[state.state_id],
            "probability": state.probability,
            "in_transition": state.in_transition,
            "flickering": flicker,
            "uncertain": orch.is_uncertain(state, flicker),
        }
        if not signals:
            return TargetSignal(None, {**meta, "strategy": None, "signal_allocation": np.nan})
        weights = {s.symbol: s.metadata["target_weight"] for s in signals}
        meta.update(strategy=signals[0].strategy_name, signal_allocation=sum(weights.values()))
        return TargetSignal(weights, meta)


class WalkForwardBacktester:
    """Runs walk-forward simulations of the full signal pipeline."""

    def __init__(
        self,
        config: BacktestConfig,
        hmm_config: HMMConfig,
        strategy_config: StrategyConfig,
        risk_config: RiskConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        """Wire in component configs; ``hmm_config.min_train_bars`` is replaced by ``config.min_train_bars``."""
        self.config = config
        self.hmm_config = replace(hmm_config, min_train_bars=config.min_train_bars)
        self.strategy_config = strategy_config
        self.risk_config = risk_config
        self.features = FeatureEngineer(feature_config)

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> WalkForwardBacktester:
        """Build from a parsed settings.yaml dict."""
        return cls(
            BacktestConfig.from_dict(settings.get("backtest"), settings.get("strategy")),
            HMMConfig.from_dict(settings.get("hmm")),
            StrategyConfig.from_dict(settings.get("strategy"), settings.get("hmm")),
            RiskConfig.from_dict(settings.get("risk")),
            FeatureConfig.from_dict(settings.get("features")),
        )

    def make_folds(self, index: pd.DatetimeIndex) -> list[Fold]:
        """Rolling (train, test) windows over ``index``; the last test window may be partial."""
        c = self.config
        folds = []
        start = 0
        while start + c.train_window < len(index):
            train = index[start : start + c.train_window]
            test = index[start + c.train_window : start + c.train_window + c.test_window]
            folds.append(Fold(train[0], train[-1], test[0], test[-1], len(train), len(test)))
            start += c.step_size
        return folds

    def prepare(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.Series, list[Fold]]:
        """Align bars, build causal features for the first symbol, and cut folds.

        Bars before ``start`` are used only for feature warm-up; the first
        in-sample window begins at the first bar >= ``start`` with valid features.
        """
        bars = align_bars(bars_by_symbol)
        if end is not None:
            bars = {s: b.loc[: pd.Timestamp(end)] for s, b in bars.items()}
        primary = bars[next(iter(bars))]
        feats = self.features.build_features(primary)
        if start is not None:
            feats = feats.loc[pd.Timestamp(start):]
        returns = log_returns(primary["close"].astype(float), 1).reindex(feats.index)
        folds = self.make_folds(feats.index)
        if not folds:
            raise ValueError(
                f"Not enough bars: {len(feats)} feature rows after warm-up, "
                f"need > train_window ({self.config.train_window})"
            )
        return bars, feats, returns, folds

    def run(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        engines: list[HMMEngine] | None = None,
        state_transform: Callable[[int, pd.Timestamp, RegimeState, HMMEngine], RegimeState] | None = None,
        use_risk: bool = True,
    ) -> BacktestResult:
        """Execute all windows and return stitched out-of-sample results.

        Pass ``engines`` (from a previous result) to reuse fitted models instead of
        retraining, e.g. for stress tests on perturbed prices.
        """
        bars, feats, returns, folds = self.prepare(bars_by_symbol, start, end)
        source = RegimeSignalSource(
            bars, feats, returns, folds, self.hmm_config, self.strategy_config, engines, state_transform
        )
        sim_index = feats.loc[folds[0].test_start : folds[-1].test_end].index
        sim = run_simulation(
            bars, sim_index, source, self.config,
            self.risk_config if use_risk else None,
            rearm_at=frozenset(f.test_start for f in folds[1:]),
        )
        return self._to_result(sim, folds, [e for e in source.engines if e is not None])

    def _to_result(self, sim: SimulationResult, folds: list[Fold], engines: list[HMMEngine]) -> BacktestResult:
        hist = sim.history.copy()
        hist["equity"] = sim.equity
        hist["allocation"] = sim.allocation
        hist["target_allocation"] = sim.target_allocation
        hist["risk_state"] = sim.risk_state
        returns = sim.equity.pct_change()
        returns.iloc[0] = sim.equity.iloc[0] / self.config.initial_capital - 1.0
        return BacktestResult(
            equity_curve=sim.equity,
            returns=returns.rename("returns"),
            weights=sim.weights,
            regimes=hist["regime"].rename("regime") if "regime" in hist else pd.Series(dtype=object),
            trades=sim.trades,
            folds=folds,
            fills=sim.fills,
            regime_history=hist,
            allocation=sim.allocation,
            initial_capital=self.config.initial_capital,
            engines=engines,
        )

    def apply_slippage(self, price: float, side: str) -> float:
        """Adjust a fill price adversely by ``slippage_pct``."""
        return apply_slippage(price, side, self.config.slippage_pct)
