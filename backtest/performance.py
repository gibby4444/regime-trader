"""Performance analytics: risk-adjusted returns, drawdowns, regime breakdown, benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.045) -> float:
    """Annualised Sharpe ratio of periodic ``returns`` vs annual ``risk_free_rate``."""
    raise NotImplementedError


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.045) -> float:
    """Annualised Sortino ratio using downside deviation."""
    raise NotImplementedError


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate of an equity curve."""
    raise NotImplementedError


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown from running peak at each bar (non-positive fractions)."""
    raise NotImplementedError


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction."""
    raise NotImplementedError


def calmar_ratio(equity: pd.Series) -> float:
    """CAGR divided by max drawdown."""
    raise NotImplementedError


def regime_breakdown(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Per-regime stats: bars, % time, mean/vol of returns, Sharpe, hit rate."""
    raise NotImplementedError


def benchmark_comparison(
    returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.045
) -> dict[str, float]:
    """Alpha, beta, correlation, tracking error, information ratio vs benchmark (e.g. SPY buy-and-hold)."""
    raise NotImplementedError


@dataclass
class PerformanceReport:
    """Aggregated performance summary for a backtest run."""

    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    regime_stats: pd.DataFrame
    benchmark: dict[str, float]
    monthly_returns: pd.DataFrame | None = None

    @classmethod
    def from_backtest(
        cls,
        equity: pd.Series,
        regimes: pd.Series,
        benchmark_equity: pd.Series | None = None,
        risk_free_rate: float = 0.045,
    ) -> PerformanceReport:
        """Compute all metrics from an equity curve and regime labels."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, float | np.floating]:
        """Scalar metrics as a flat dict (for logging / JSON)."""
        raise NotImplementedError

    def print_summary(self) -> None:
        """Render a ``rich`` table of the headline metrics."""
        raise NotImplementedError
