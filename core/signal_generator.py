"""Signal generation: combines HMM regime, allocation strategy, and risk gate.

Produces per-symbol :class:`~core.regime_strategies.Signal` objects consumed by
the broker layer or the backtester.
"""

from __future__ import annotations

import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from core.regime_strategies import AllocationTarget, Signal, StrategyOrchestrator
from core.risk_manager import RiskDecision, RiskManager

__all__ = ["Signal", "SignalGenerator"]


class SignalGenerator:
    """Orchestrates regime -> strategy signals -> risk -> final signals."""

    def __init__(
        self,
        hmm: HMMEngine,
        strategy: StrategyOrchestrator,
        risk: RiskManager,
    ) -> None:
        """Wire together the core components."""
        self.hmm = hmm
        self.strategy = strategy
        self.risk = risk

    def generate(
        self,
        features: pd.DataFrame,
        bars: dict[str, pd.DataFrame],
        current_allocation: float,
        equity: float,
    ) -> list[Signal]:
        """Produce signals for the latest bar using only data up to that bar.

        Steps: ``hmm.update`` on the latest feature row -> orchestrator signals
        (uncertainty/transition scaling, rebalance threshold) -> risk limits.
        """
        raise NotImplementedError

    def build_target(self, signals: list[Signal]) -> AllocationTarget | None:
        """Portfolio-level summary of the orchestrator's signals."""
        raise NotImplementedError

    def apply_risk(
        self,
        signals: list[Signal],
        decision: RiskDecision,
        regime: RegimeState,
    ) -> list[Signal]:
        """Scale/clip signals per the risk decision."""
        raise NotImplementedError
