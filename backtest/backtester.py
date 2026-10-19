"""Walk-forward allocation backtester.

For each fold: fit the HMM on ``train_window`` bars, then step bar-by-bar through
the next ``test_window`` bars producing signals using only data available at each
bar, executing at the next bar's open with ``slippage_pct``. Advance by
``step_size`` and repeat. Out-of-sample segments are stitched into one equity curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from core.risk_manager import RiskManager
from data.feature_engineering import FeatureEngineer


@dataclass
class BacktestConfig:
    """Parameters mirroring the ``backtest`` section of settings.yaml."""

    slippage_pct: float = 0.0005
    initial_capital: float = 100_000.0
    train_window: int = 504
    test_window: int = 126
    step_size: int = 126
    risk_free_rate: float = 0.045

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> BacktestConfig:
        """Build a config from the parsed ``backtest`` settings section."""
        raise NotImplementedError


@dataclass
class Fold:
    """Index boundaries of one walk-forward fold."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class BacktestResult:
    """Stitched out-of-sample results."""

    equity_curve: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    regimes: pd.Series
    trades: pd.DataFrame
    folds: list[Fold] = field(default_factory=list)


class WalkForwardBacktester:
    """Runs walk-forward simulations of the full signal pipeline."""

    def __init__(
        self,
        config: BacktestConfig,
        hmm: HMMEngine,
        strategy: StrategyOrchestrator,
        risk: RiskManager,
        features: FeatureEngineer,
    ) -> None:
        """Wire in the components to simulate."""
        self.config = config
        self.hmm = hmm
        self.strategy = strategy
        self.risk = risk
        self.features = features

    def make_folds(self, index: pd.DatetimeIndex) -> list[Fold]:
        """Split ``index`` into rolling train/test folds per config."""
        raise NotImplementedError

    def run(self, bars_by_symbol: dict[str, pd.DataFrame]) -> BacktestResult:
        """Execute all folds and return stitched out-of-sample results."""
        raise NotImplementedError

    def run_fold(
        self, fold: Fold, bars_by_symbol: dict[str, pd.DataFrame], equity: float
    ) -> BacktestResult:
        """Fit on the train window, then simulate the test window bar by bar."""
        raise NotImplementedError

    def apply_slippage(self, price: float, side: str) -> float:
        """Adjust a fill price adversely by ``slippage_pct``."""
        raise NotImplementedError
