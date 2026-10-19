"""Stress testing: synthetic crash injection and overnight gap simulation.

Perturbs historical bars and re-runs the backtester to verify that risk limits
(drawdown halts, kill switch) engage and losses stay bounded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.backtester import BacktestResult, WalkForwardBacktester


@dataclass(frozen=True)
class StressScenario:
    """Description of one synthetic shock."""

    name: str
    shock_pct: float
    duration_bars: int = 1
    start: pd.Timestamp | None = None
    gap: bool = False


class StressTester:
    """Applies shocks to price data and evaluates strategy behaviour."""

    def __init__(self, backtester: WalkForwardBacktester, seed: int | None = None) -> None:
        """Bind the backtester and RNG seed for reproducible random shocks."""
        self.backtester = backtester
        self.rng = np.random.default_rng(seed)

    def inject_crash(
        self, bars: pd.DataFrame, shock_pct: float, start: pd.Timestamp, duration_bars: int
    ) -> pd.DataFrame:
        """Scale prices down by ``shock_pct`` spread over ``duration_bars`` from ``start``."""
        raise NotImplementedError

    def simulate_gap(
        self, bars: pd.DataFrame, gap_pct: float, at: pd.Timestamp
    ) -> pd.DataFrame:
        """Apply an overnight open gap at ``at`` (stops fill at the gapped open)."""
        raise NotImplementedError

    def default_scenarios(self) -> list[StressScenario]:
        """Return standard scenarios (e.g. -10%/-20% crash, -7% gap, 2020-style)."""
        raise NotImplementedError

    def run_scenario(
        self, scenario: StressScenario, bars_by_symbol: dict[str, pd.DataFrame]
    ) -> BacktestResult:
        """Apply ``scenario`` to all symbols and run the backtest."""
        raise NotImplementedError

    def run_all(
        self, bars_by_symbol: dict[str, pd.DataFrame]
    ) -> dict[str, BacktestResult]:
        """Run every default scenario; keyed by scenario name."""
        raise NotImplementedError
