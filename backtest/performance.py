"""Performance analytics: risk-adjusted returns, drawdowns, regime/confidence breakdowns,
benchmarks, worst-case statistics, and report output (rich tables + CSVs).

Conventions: periodic returns are simple bar returns; annualisation uses
``TRADING_DAYS`` bars per year. A *trade* is a holding segment between two
rebalances (see :func:`backtest.backtester.build_trades`); flat (zero-allocation)
segments are excluded from trade statistics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.backtester import (
    BacktestConfig,
    BacktestResult,
    TargetSignal,
    align_bars,
    run_simulation,
)
from core.risk_manager import RiskConfig

TRADING_DAYS = 252
CONFIDENCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<50%", 0.0, 0.50),
    ("50-60%", 0.50, 0.60),
    ("60-70%", 0.60, 0.70),
    ("70%+", 0.70, 1.0 + 1e-9),
)


def _nan_if_degenerate(num: float, den: float) -> float:
    return float(num / den) if den and np.isfinite(den) and den > 1e-12 else float("nan")


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.045) -> float:
    """Annualised Sharpe ratio of periodic ``returns`` vs annual ``risk_free_rate``."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_rate / TRADING_DAYS
    return _nan_if_degenerate(excess.mean() * np.sqrt(TRADING_DAYS), excess.std(ddof=1))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.045) -> float:
    """Annualised Sortino ratio using downside deviation (root mean square of negative excess)."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_rate / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    return _nan_if_degenerate(excess.mean() * np.sqrt(TRADING_DAYS), downside)


def total_return(equity: pd.Series, initial: float | None = None) -> float:
    """Final over initial equity minus one (initial defaults to the first mark)."""
    start = float(initial if initial is not None else equity.iloc[0])
    return float(equity.iloc[-1] / start - 1.0)


def cagr(equity: pd.Series, initial: float | None = None) -> float:
    """Compound annual growth rate of an equity curve (``len / TRADING_DAYS`` years)."""
    years = len(equity) / TRADING_DAYS
    growth = 1.0 + total_return(equity, initial)
    if years <= 0 or growth <= 0:
        return float("nan") if years <= 0 else -1.0
    return float(growth ** (1.0 / years) - 1.0)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown from running peak at each bar (non-positive fractions)."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction."""
    return float(-drawdown_series(equity).min()) if len(equity) else 0.0


def underwater_periods(equity: pd.Series) -> pd.DataFrame:
    """Each drawdown episode: start (peak bar), trough, end (recovery or last bar), depth, bars."""
    dd = drawdown_series(equity).to_numpy()
    idx = equity.index
    rows = []
    i, n = 0, len(dd)
    while i < n:
        if dd[i] < 0:
            start = i - 1 if i > 0 else 0
            j = i
            while j < n and dd[j] < 0:
                j += 1
            trough = i + int(np.argmin(dd[i:j]))
            rows.append({
                "start": idx[start], "trough": idx[trough], "end": idx[min(j, n - 1)],
                "depth": float(-dd[trough]), "bars": int(j - start), "recovered": j < n,
            })
            i = j
        else:
            i += 1
    return pd.DataFrame(rows, columns=["start", "trough", "end", "depth", "bars", "recovered"])


def max_drawdown_duration(equity: pd.Series) -> int:
    """Bars from the peak preceding the maximum drawdown until recovery (or the end)."""
    periods = underwater_periods(equity)
    return int(periods.loc[periods["depth"].idxmax(), "bars"]) if len(periods) else 0


def longest_underwater(equity: pd.Series) -> int:
    """Longest drawdown episode in bars, regardless of depth."""
    periods = underwater_periods(equity)
    return int(periods["bars"].max()) if len(periods) else 0


def calmar_ratio(equity: pd.Series, initial: float | None = None) -> float:
    """CAGR divided by max drawdown."""
    return _nan_if_degenerate(cagr(equity, initial), max_drawdown(equity))


def _active(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=getattr(trades, "columns", []))
    return trades[~trades["is_flat"].astype(bool)]


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of non-flat trades with positive P&L."""
    t = _active(trades)
    return float((t["pnl"] > 0).mean()) if len(t) else float("nan")


def profit_factor(trades: pd.DataFrame) -> float:
    """Gross profit over gross loss of non-flat trades (inf if no losses)."""
    t = _active(trades)
    if not len(t):
        return float("nan")
    gains, losses = t.loc[t["pnl"] > 0, "pnl"].sum(), -t.loc[t["pnl"] < 0, "pnl"].sum()
    return float(gains / losses) if losses > 0 else float("inf")


def max_consecutive(mask: pd.Series | np.ndarray) -> int:
    """Longest run of True values."""
    best = cur = 0
    for v in np.asarray(mask, dtype=bool):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def regime_breakdown(
    returns: pd.Series,
    regimes: pd.Series,
    trades: pd.DataFrame | None = None,
    trade_regime_col: str = "vol_regime",
    risk_free_rate: float = 0.045,
) -> pd.DataFrame:
    """Regime | % Time In | Return Contribution | Avg Trade P&L | Win Rate | Sharpe.

    Return contribution is the regime's share of total log return (sums to the
    total log return over all regimes). Trade columns use the regime at the
    trade's signal bar.
    """
    r = returns.reindex(regimes.index)
    log_r = np.log1p(r.fillna(0.0))
    trades_active = _active(trades) if trades is not None else pd.DataFrame()
    rows = []
    for name, grp in r.groupby(regimes, sort=True):
        t = trades_active[trades_active[trade_regime_col] == name] if len(trades_active) else trades_active
        rows.append({
            "Regime": name,
            "% Time In": len(grp) / len(r),
            "Return Contribution": float(log_r.loc[grp.index].sum()),
            "Trades": len(t),
            "Avg Trade P&L": float(t["pnl"].mean()) if len(t) else float("nan"),
            "Win Rate": float((t["pnl"] > 0).mean()) if len(t) else float("nan"),
            "Sharpe": sharpe_ratio(grp, risk_free_rate),
        })
    return pd.DataFrame(rows).set_index("Regime") if rows else pd.DataFrame()


def confidence_breakdown(
    trades: pd.DataFrame, returns: pd.Series, confidence: pd.Series, risk_free_rate: float = 0.045
) -> pd.DataFrame:
    """Confidence bucket | Trades | Sharpe | Win Rate | Avg P&L.

    Trades are bucketed by confidence at their signal bar; Sharpe uses bar returns
    bucketed by the confidence of the regime estimate on the previous bar (the one
    that set the position held through that bar).
    """
    t = _active(trades)
    conf_prev = confidence.shift(1).reindex(returns.index)
    rows = []
    for label, lo, hi in CONFIDENCE_BUCKETS:
        tb = t[(t["confidence"] >= lo) & (t["confidence"] < hi)] if len(t) else t
        rb = returns[(conf_prev >= lo) & (conf_prev < hi)]
        rows.append({
            "Confidence": label,
            "Trades": len(tb),
            "Sharpe": sharpe_ratio(rb, risk_free_rate),
            "Win Rate": float((tb["pnl"] > 0).mean()) if len(tb) else float("nan"),
            "Avg P&L": float(tb["pnl"].mean()) if len(tb) else float("nan"),
        })
    return pd.DataFrame(rows).set_index("Confidence")


def worst_case(equity: pd.Series, returns: pd.Series, trades: pd.DataFrame | None = None) -> dict[str, float]:
    """Worst day/week/month, max consecutive losing days/trades, longest underwater (bars)."""
    r = returns.dropna()
    out = {"worst_day": float(r.min()) if len(r) else float("nan")}
    if isinstance(equity.index, pd.DatetimeIndex):
        for key, rule in (("worst_week", "W-FRI"), ("worst_month", "ME")):
            period = (1.0 + r).resample(rule).prod() - 1.0
            out[key] = float(period.min()) if len(period) else float("nan")
    out["max_consecutive_losing_days"] = max_consecutive(r < 0)
    t = _active(trades) if trades is not None else pd.DataFrame()
    out["max_consecutive_losing_trades"] = max_consecutive(t["pnl"] < 0) if len(t) else 0
    out["longest_underwater_bars"] = longest_underwater(equity)
    return out


def benchmark_comparison(
    returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.045
) -> dict[str, float]:
    """Alpha (annualised), beta, correlation, tracking error, information ratio vs a benchmark."""
    df = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(df) < 2:
        return {k: float("nan") for k in ("alpha", "beta", "correlation", "tracking_error", "information_ratio")}
    s, b = df.iloc[:, 0], df.iloc[:, 1]
    rf = risk_free_rate / TRADING_DAYS
    var_b = b.var(ddof=1)
    beta = float(s.cov(b) / var_b) if var_b > 0 else float("nan")
    alpha = float(((s - rf).mean() - beta * (b - rf).mean()) * TRADING_DAYS)
    active = s - b
    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {
        "alpha": alpha,
        "beta": beta,
        "correlation": float(s.corr(b)),
        "tracking_error": te,
        "information_ratio": _nan_if_degenerate(active.mean() * TRADING_DAYS, te),
    }


# ---------------------------------------------------------------- benchmarks


def _constant_signal(symbols: list[str], fn: Callable[[int, pd.Timestamp], float | None]) -> Callable:
    def signal(pos: int, ts: pd.Timestamp) -> TargetSignal:
        a = fn(pos, ts)
        return TargetSignal(None if a is None else {s: a / len(symbols) for s in symbols})
    return signal


def _summary(equity: pd.Series, initial: float, rf: float) -> dict[str, float]:
    rets = equity.pct_change().fillna(equity.iloc[0] / initial - 1.0)
    return {
        "total_return": total_return(equity, initial),
        "cagr": cagr(equity, initial),
        "sharpe": sharpe_ratio(rets, rf),
        "max_drawdown": max_drawdown(pd.concat([pd.Series([initial]), equity.reset_index(drop=True)])),
    }


def buy_and_hold(
    bars_by_symbol: dict[str, pd.DataFrame], index: pd.DatetimeIndex, config: BacktestConfig
) -> pd.Series:
    """Equal-weight 100% invested from the first bar's signal (fills next open)."""
    sym = list(bars_by_symbol)
    return run_simulation(bars_by_symbol, index, _constant_signal(sym, lambda p, t: 1.0), config).equity


def sma_trend(
    bars_by_symbol: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    config: BacktestConfig,
    window: int = 200,
) -> pd.Series:
    """100% invested when the first symbol closes above its ``window`` SMA, else flat."""
    sym = list(bars_by_symbol)
    close = bars_by_symbol[sym[0]]["close"].astype(float)
    above = (close > close.rolling(window, min_periods=window).mean()).reindex(index)
    return run_simulation(
        bars_by_symbol, index, _constant_signal(sym, lambda p, t: 1.0 if above.iloc[p] else 0.0), config
    ).equity


def random_allocation(
    bars_by_symbol: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    config: BacktestConfig,
    risk_config: RiskConfig | None,
    allocations: np.ndarray,
    change_prob: float,
    seed: int,
    rearm_at: frozenset[pd.Timestamp] = frozenset(),
) -> pd.Series:
    """Random target changes (prob ``change_prob`` per bar, values drawn from ``allocations``)
    under the same threshold, slippage, fill delay and risk rules as the strategy."""
    sym = list(bars_by_symbol)
    rng = np.random.default_rng(seed)
    draws = rng.random(len(index)) < change_prob
    values = rng.choice(allocations, size=len(index))
    path = np.empty(len(index))
    cur = values[0]
    for i in range(len(index)):
        if draws[i]:
            cur = values[i]
        path[i] = cur
    return run_simulation(
        bars_by_symbol, index, _constant_signal(sym, lambda p, t: float(path[p])), config, risk_config, rearm_at
    ).equity


def compare_benchmarks(
    result: BacktestResult,
    bars_by_symbol: dict[str, pd.DataFrame],
    config: BacktestConfig,
    risk_config: RiskConfig | None,
    n_random: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Strategy vs buy-and-hold, 200-SMA trend, and random allocation (mean/std over seeds)."""
    index = result.equity_curve.index
    bars = align_bars(bars_by_symbol)
    init, rf = config.initial_capital, config.risk_free_rate
    rows = {"Strategy": _summary(result.equity_curve, init, rf)}
    rows["Buy & Hold"] = _summary(buy_and_hold(bars, index, config), init, rf)
    rows["200 SMA Trend"] = _summary(sma_trend(bars, index, config), init, rf)

    hist = result.regime_history
    target = hist["target_allocation"].dropna() if "target_allocation" in hist else pd.Series(dtype=float)
    allocations = np.unique(np.round(target.to_numpy(), 4)) if len(target) else np.array([0.6, 0.95, 1.1875])
    allocations = allocations[allocations > 0] if (allocations > 0).any() else np.array([1.0])
    change_prob = float((target.diff().abs() > 1e-9).mean()) if len(target) > 1 else 0.05
    change_prob = max(change_prob, 1.0 / len(index))
    rearm = frozenset(f.test_start for f in result.folds[1:])
    sims = pd.DataFrame([
        _summary(random_allocation(bars, index, config, risk_config, allocations, change_prob, seed + i, rearm), init, rf)
        for i in range(n_random)
    ])
    rows[f"Random (mean, n={n_random})"] = sims.mean().to_dict()
    rows[f"Random (std, n={n_random})"] = sims.std(ddof=1).to_dict() if n_random > 1 else {k: float("nan") for k in sims}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- report


@dataclass
class PerformanceReport:
    """Aggregated performance summary for a backtest run."""

    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    max_drawdown_duration: int = 0
    win_rate: float = float("nan")
    profit_factor: float = float("nan")
    total_trades: int = 0
    avg_holding_period: float = float("nan")
    regime_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    confidence_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    worst_case: dict[str, float] = field(default_factory=dict)
    benchmark: dict[str, float] = field(default_factory=dict)
    benchmarks: pd.DataFrame | None = None
    monthly_returns: pd.DataFrame | None = None

    @classmethod
    def from_backtest(
        cls,
        equity: pd.Series,
        regimes: pd.Series,
        benchmark_equity: pd.Series | None = None,
        risk_free_rate: float = 0.045,
        trades: pd.DataFrame | None = None,
        confidence: pd.Series | None = None,
        initial_capital: float | None = None,
    ) -> PerformanceReport:
        """Compute all metrics from an equity curve, regime labels and (optionally) trades."""
        init = float(initial_capital if initial_capital is not None else equity.iloc[0])
        returns = equity.pct_change()
        returns.iloc[0] = equity.iloc[0] / init - 1.0
        curve = pd.concat([pd.Series([init], index=[equity.index[0] - pd.Timedelta(1, "D")]), equity]) \
            if isinstance(equity.index, pd.DatetimeIndex) else equity
        trades = trades if trades is not None else pd.DataFrame(columns=["pnl", "is_flat", "holding_bars"])
        active = _active(trades)
        bench = {}
        if benchmark_equity is not None:
            bench = benchmark_comparison(returns, benchmark_equity.pct_change(), risk_free_rate)
        monthly = None
        if isinstance(equity.index, pd.DatetimeIndex):
            m = (1.0 + returns).resample("ME").prod() - 1.0
            monthly = pd.DataFrame({"year": m.index.year, "month": m.index.month, "return": m.to_numpy()}) \
                .pivot(index="year", columns="month", values="return")
        return cls(
            total_return=total_return(equity, init),
            cagr=cagr(equity, init),
            sharpe=sharpe_ratio(returns, risk_free_rate),
            sortino=sortino_ratio(returns, risk_free_rate),
            max_drawdown=max_drawdown(curve),
            calmar=_nan_if_degenerate(cagr(equity, init), max_drawdown(curve)),
            max_drawdown_duration=max_drawdown_duration(curve),
            win_rate=win_rate(trades),
            profit_factor=profit_factor(trades),
            total_trades=len(active),
            avg_holding_period=float(active["holding_bars"].mean()) if len(active) else float("nan"),
            regime_stats=regime_breakdown(returns, regimes, trades, risk_free_rate=risk_free_rate)
            if len(regimes) else pd.DataFrame(),
            confidence_stats=confidence_breakdown(trades, returns, confidence, risk_free_rate)
            if confidence is not None else pd.DataFrame(),
            worst_case=worst_case(curve, returns, trades),
            benchmark=bench,
            monthly_returns=monthly,
        )

    @classmethod
    def from_result(cls, result: BacktestResult, risk_free_rate: float = 0.045) -> PerformanceReport:
        """Report for a :class:`BacktestResult`, grouping regimes by volatility regime."""
        hist = result.regime_history
        regimes = hist["vol_regime"] if "vol_regime" in hist else result.regimes
        return cls.from_backtest(
            result.equity_curve, regimes, risk_free_rate=risk_free_rate, trades=result.trades,
            confidence=hist["probability"] if "probability" in hist else None,
            initial_capital=result.initial_capital,
        )

    def to_dict(self) -> dict[str, float | np.floating]:
        """Scalar metrics as a flat dict (for logging / JSON)."""
        out: dict[str, Any] = {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_bars": self.max_drawdown_duration,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_holding_period_bars": self.avg_holding_period,
        }
        out.update(self.worst_case)
        out.update({f"bench_{k}": v for k, v in self.benchmark.items()})
        return out

    def print_summary(self, console: Any | None = None) -> None:
        """Render ``rich`` tables: headline metrics, regimes, confidence, worst case, benchmarks."""
        from rich.console import Console

        console = console or Console()
        pct = {"total_return", "cagr", "max_drawdown", "win_rate", "worst_day", "worst_week", "worst_month"}
        rows = [(k, _fmt(v, k in pct)) for k, v in self.to_dict().items() if not k.startswith("bench_")]
        console.print(_table("Performance", ["Metric", "Value"], rows))
        if not self.regime_stats.empty:
            console.print(_frame_table("Regime Breakdown", self.regime_stats,
                                       {"% Time In", "Return Contribution", "Win Rate"}))
        if not self.confidence_stats.empty:
            console.print(_frame_table("Confidence Buckets", self.confidence_stats, {"Win Rate"}))
        if self.benchmarks is not None:
            console.print(_frame_table("Benchmarks", self.benchmarks, {"total_return", "cagr", "max_drawdown"}))

    def to_csv(self, result: BacktestResult, out_dir: str | Path) -> dict[str, Path]:
        """Write equity curve, trade log, regime history (and summary tables) as CSVs."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "equity_curve": out / "equity_curve.csv",
            "trade_log": out / "trade_log.csv",
            "fills": out / "fills.csv",
            "regime_history": out / "regime_history.csv",
            "summary": out / "summary.csv",
        }
        eq = pd.DataFrame({
            "equity": result.equity_curve,
            "returns": result.returns,
            "drawdown": drawdown_series(result.equity_curve),
            "allocation": result.allocation,
        })
        eq.to_csv(paths["equity_curve"], index_label="timestamp")
        result.trades.to_csv(paths["trade_log"], index=False)
        result.fills.to_csv(paths["fills"], index=False)
        result.regime_history.to_csv(paths["regime_history"], index_label="timestamp")
        pd.Series(self.to_dict(), name="value").to_csv(paths["summary"], index_label="metric")
        for name, frame in (("regime_stats", self.regime_stats), ("confidence_stats", self.confidence_stats),
                            ("benchmarks", self.benchmarks)):
            if frame is not None and not frame.empty:
                paths[name] = out / f"{name}.csv"
                frame.to_csv(paths[name])
        return paths


def _fmt(v: Any, as_pct: bool = False) -> str:
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "inf" if f == np.inf else "n/a"
    return f"{f:.2%}" if as_pct else f"{f:,.3f}"


def _table(title: str, headers: list[str], rows: list[tuple[str, ...]]) -> Any:
    from rich.table import Table

    t = Table(title=title)
    for i, h in enumerate(headers):
        t.add_column(h, justify="left" if i == 0 else "right")
    for r in rows:
        t.add_row(*[str(c) for c in r])
    return t


def _frame_table(title: str, frame: pd.DataFrame, pct_cols: set[str]) -> Any:
    headers = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
    ints = {c for c in frame.columns if pd.api.types.is_integer_dtype(frame[c]) or c in ("Trades", "sims")}
    rows = [
        (str(idx), *[str(int(v)) if c in ints and pd.notna(v) else _fmt(v, c in pct_cols)
                     for c, v in zip(frame.columns, vals)])
        for idx, vals in zip(frame.index, frame.itertuples(index=False))
    ]
    return _table(title, headers, rows)
