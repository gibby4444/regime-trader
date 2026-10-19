"""Stress testing: crash injection, overnight gap risk, and regime misclassification.

Each Monte Carlo simulation perturbs the historical bars (or the regime labels)
and re-walks the out-of-sample period with the *frozen* HMMs fitted in the
baseline run. Retraining 10+ HMMs per simulation would make 100-sim runs
impractical, and the question here is how execution and the risk overlay respond
to shocks the models did not see, not how training adapts to them. Features and
filtered regimes are recomputed from the perturbed prices, so the HMM still
reacts to the shock bar by bar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from backtest.backtester import BacktestResult, WalkForwardBacktester, align_bars
from backtest.performance import max_drawdown, total_return
from core.hmm_engine import HMMEngine, RegimeState
from data.feature_engineering import atr

PRICE_COLUMNS = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class StressScenario:
    """Description of one synthetic shock family."""

    name: str
    shock_pct: float
    duration_bars: int = 1
    start: pd.Timestamp | None = None
    gap: bool = False


@dataclass
class StressOutcome:
    """Per-simulation metrics for one scenario plus the shocks applied."""

    name: str
    sims: pd.DataFrame
    events: list[pd.DataFrame] = field(default_factory=list)


def kill_events(result: BacktestResult) -> int:
    """Number of times the peak-drawdown breaker (trading lock) tripped."""
    state = result.regime_history.get("risk_state")
    if state is None:
        return 0
    killed = state == "locked"
    return int((killed & ~killed.shift(1, fill_value=False)).sum())


def window_drawdowns(result: BacktestResult) -> pd.Series:
    """Max drawdown within each walk-forward window (peak reset at each window start)."""
    eq = result.equity_curve
    out = {}
    for k, f in enumerate(result.folds):
        seg = eq.loc[f.test_start:f.test_end]
        prior = eq.loc[: f.test_start].iloc[:-1]
        start_eq = prior.iloc[-1] if len(prior) else result.initial_capital
        out[k] = max_drawdown(pd.concat([pd.Series([start_eq]), seg.reset_index(drop=True)]))
    return pd.Series(out, name="window_max_drawdown")


def sim_metrics(result: BacktestResult) -> dict[str, float]:
    curve = pd.concat([pd.Series([result.initial_capital]), result.equity_curve.reset_index(drop=True)])
    return {
        "total_return": total_return(result.equity_curve, result.initial_capital),
        "max_drawdown": max_drawdown(curve),
        "worst_window_drawdown": float(window_drawdowns(result).max()),
        "min_equity": float(result.equity_curve.min()),
        "kill_events": kill_events(result),
    }


class StressTester:
    """Applies shocks to price data and evaluates strategy behaviour."""

    def __init__(
        self,
        backtester: WalkForwardBacktester,
        seed: int | None = None,
        n_sims: int = 100,
        n_events: int = 10,
        crash_range: tuple[float, float] = (-0.15, -0.05),
        gap_atr_range: tuple[float, float] = (2.0, 5.0),
    ) -> None:
        """Bind the backtester and RNG seed for reproducible random shocks."""
        self.backtester = backtester
        self.rng = np.random.default_rng(seed)
        self.n_sims = n_sims
        self.n_events = n_events
        self.crash_range = crash_range
        self.gap_atr_range = gap_atr_range

    # ------------------------------------------------------------ perturbations

    def inject_crash(
        self, bars: pd.DataFrame, shock_pct: float, start: pd.Timestamp, duration_bars: int = 1
    ) -> pd.DataFrame:
        """Scale prices down by ``shock_pct`` spread over ``duration_bars`` from ``start``.

        With ``duration_bars=1`` this is an overnight gap: bar ``start`` opens
        ``shock_pct`` below where it otherwise would, and every later bar is shifted
        by the same factor (a permanent level shift). Volume is untouched.
        """
        out = bars.copy()
        pos = out.index.get_loc(start)
        factors = np.ones(len(out))
        steps = np.arange(1, duration_bars + 1)
        factors[pos : pos + duration_bars] = (1.0 + shock_pct) ** (steps[: len(out) - pos] / duration_bars)
        factors[pos + duration_bars :] = 1.0 + shock_pct
        out[PRICE_COLUMNS] = out[PRICE_COLUMNS].mul(factors, axis=0)
        return out

    def simulate_gap(self, bars: pd.DataFrame, gap_pct: float, at: pd.Timestamp) -> pd.DataFrame:
        """Apply an overnight open gap of ``gap_pct`` at ``at`` (prices thereafter shift with it)."""
        return self.inject_crash(bars, gap_pct, at, 1)

    def _event_dates(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        k = min(self.n_events, len(index) - 1)
        return pd.DatetimeIndex(np.sort(self.rng.choice(index[1:], size=k, replace=False)))

    def random_crashes(
        self, bars_by_symbol: dict[str, pd.DataFrame], index: pd.DatetimeIndex
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """``n_events`` gaps of U(crash_range) at random bars of ``index``, same shock for all symbols."""
        dates = self._event_dates(index)
        shocks = self.rng.uniform(*self.crash_range, size=len(dates))
        out = dict(bars_by_symbol)
        for d, s in zip(dates, shocks):
            out = {sym: self.inject_crash(b, float(s), d) for sym, b in out.items()}
        return out, pd.DataFrame({"timestamp": dates, "shock_pct": shocks})

    def random_gaps(
        self, bars_by_symbol: dict[str, pd.DataFrame], index: pd.DatetimeIndex, atr_window: int = 14
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """``n_events`` downward overnight gaps of U(gap_atr_range) x prior-bar ATR."""
        dates = self._event_dates(index)
        mults = self.rng.uniform(*self.gap_atr_range, size=len(dates))
        out = dict(bars_by_symbol)
        rows = []
        for d, m in zip(dates, mults):
            for sym, b in list(out.items()):
                pos = b.index.get_loc(d)
                a = atr(b["high"], b["low"], b["close"], atr_window).iloc[pos - 1]
                prev_close = float(b["close"].iloc[pos - 1])
                gap = -float(m) * float(a) / prev_close
                out[sym] = self.simulate_gap(b, gap, d)
                rows.append({"timestamp": d, "symbol": sym, "atr_mult": float(m), "shock_pct": gap})
        return out, pd.DataFrame(rows)

    def shuffled_regimes(
        self, baseline: BacktestResult
    ) -> Callable[[int, pd.Timestamp, RegimeState, HMMEngine], RegimeState]:
        """State transform that permutes each window's confirmed regime sequence in time.

        The regime's reported probability is kept, so the strategy acts on a
        confidently wrong label (no uncertainty discount).
        """
        hist = baseline.regime_history
        mapping: dict[pd.Timestamp, int] = {}
        for _, grp in hist.groupby("window"):
            ids = grp["regime_id"].to_numpy()
            mapping.update(zip(grp.index, self.rng.permutation(ids)))

        def transform(k: int, ts: pd.Timestamp, state: RegimeState, engine: HMMEngine) -> RegimeState:
            sid = int(mapping.get(ts, state.state_id))
            return replace(state, state_id=sid, label=engine.labels[sid])

        return transform

    # ---------------------------------------------------------------- runners

    def _run(self, bars, baseline: BacktestResult, start, end, **kw) -> BacktestResult:
        return self.backtester.run(bars, start=start, end=end, engines=baseline.engines, **kw)

    def run_monte_carlo(
        self,
        kind: str,
        bars_by_symbol: dict[str, pd.DataFrame],
        baseline: BacktestResult,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        n_sims: int | None = None,
    ) -> StressOutcome:
        """``kind`` = 'crash' | 'gap': perturb prices ``n_sims`` times and re-walk."""
        perturb = {"crash": self.random_crashes, "gap": self.random_gaps}[kind]
        bars = align_bars(bars_by_symbol)
        index = baseline.equity_curve.index
        rows, events = [], []
        for _ in range(n_sims or self.n_sims):
            shocked, ev = perturb(bars, index)
            rows.append(sim_metrics(self._run(shocked, baseline, start, end)))
            events.append(ev)
        return StressOutcome(kind, pd.DataFrame(rows), events)

    def run_misclassification(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        baseline: BacktestResult,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        n_sims: int | None = None,
    ) -> StressOutcome:
        """Shuffle regime labels; run each permutation with and without the risk overlay."""
        rows = []
        for _ in range(n_sims or self.n_sims):
            transform = self.shuffled_regimes(baseline)
            on = sim_metrics(self._run(bars_by_symbol, baseline, start, end, state_transform=transform))
            off = sim_metrics(self._run(bars_by_symbol, baseline, start, end, state_transform=transform,
                                        use_risk=False))
            rows.append({**on, **{f"{k}_no_risk": v for k, v in off.items()}})
        return StressOutcome("misclassification", pd.DataFrame(rows))

    def default_scenarios(self) -> list[StressScenario]:
        """Scenario families run by :meth:`run_all`."""
        lo, hi = self.crash_range
        return [
            StressScenario("crash", shock_pct=(lo + hi) / 2, gap=True),
            StressScenario("gap", shock_pct=float("nan"), gap=True),
            StressScenario("misclassification", shock_pct=0.0),
        ]

    def run_scenario(
        self,
        scenario: StressScenario,
        bars_by_symbol: dict[str, pd.DataFrame],
        baseline: BacktestResult | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> BacktestResult:
        """Apply one fixed shock (``scenario.start`` or the middle OOS bar) to all symbols and run."""
        baseline = baseline or self.backtester.run(bars_by_symbol, start, end)
        index = baseline.equity_curve.index
        at = scenario.start if scenario.start is not None else index[len(index) // 2]
        bars = {
            s: self.inject_crash(b, scenario.shock_pct, at, scenario.duration_bars)
            for s, b in align_bars(bars_by_symbol).items()
        }
        return self._run(bars, baseline, start, end)

    def run_all(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        baseline: BacktestResult | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> dict[str, StressOutcome]:
        """Crash, gap and misclassification Monte Carlo runs; keyed by scenario name."""
        baseline = baseline or self.backtester.run(bars_by_symbol, start, end)
        return {
            "crash": self.run_monte_carlo("crash", bars_by_symbol, baseline, start, end),
            "gap": self.run_monte_carlo("gap", bars_by_symbol, baseline, start, end),
            "misclassification": self.run_misclassification(bars_by_symbol, baseline, start, end),
        }


def summarize(outcomes: dict[str, StressOutcome], baseline: BacktestResult) -> pd.DataFrame:
    """One row per scenario: return/drawdown distribution and kill-switch activity."""
    base = sim_metrics(baseline)
    rows = {"baseline": {
        "sims": 1, "mean_return": base["total_return"], "p5_return": base["total_return"],
        "mean_max_dd": base["max_drawdown"], "worst_max_dd": base["max_drawdown"],
        "worst_window_dd": base["worst_window_drawdown"], "kill_rate": float(base["kill_events"] > 0),
    }}
    for name, o in outcomes.items():
        s = o.sims
        rows[name] = {
            "sims": len(s),
            "mean_return": s["total_return"].mean(),
            "p5_return": s["total_return"].quantile(0.05),
            "mean_max_dd": s["max_drawdown"].mean(),
            "worst_max_dd": s["max_drawdown"].max(),
            "worst_window_dd": s["worst_window_drawdown"].max(),
            "kill_rate": float((s["kill_events"] > 0).mean()),
        }
        if "max_drawdown_no_risk" in s:
            rows[f"{name} (no risk)"] = {
                "sims": len(s),
                "mean_return": s["total_return_no_risk"].mean(),
                "p5_return": s["total_return_no_risk"].quantile(0.05),
                "mean_max_dd": s["max_drawdown_no_risk"].mean(),
                "worst_max_dd": s["max_drawdown_no_risk"].max(),
                "worst_window_dd": s["worst_window_drawdown_no_risk"].max(),
                "kill_rate": 0.0,
            }
    return pd.DataFrame(rows).T
