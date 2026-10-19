"""Risk management: position sizing, leverage/exposure caps, and drawdown limits.

Every proposed allocation passes through :class:`RiskManager`, which clips it to
the ``risk`` settings and can reduce or halt trading on daily, weekly, or
peak-to-trough drawdown breaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.regime_strategies import AllocationTarget


class TradingState(str, Enum):
    """Risk-driven trading permission level."""

    NORMAL = "normal"
    REDUCED = "reduced"
    HALTED = "halted"
    KILLED = "killed"


@dataclass(frozen=True)
class RiskDecision:
    """Outcome of a risk check.

    Attributes:
        approved: Whether any trading may proceed.
        state: Current trading permission level.
        adjusted_weights: Per-symbol weights after all caps are applied.
        size_multiplier: Multiplier applied due to drawdown reductions.
        reasons: Human-readable list of constraints that were triggered.
    """

    approved: bool
    state: TradingState
    adjusted_weights: dict[str, float]
    size_multiplier: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class RiskConfig:
    """Risk parameters mirroring the ``risk`` section of settings.yaml."""

    max_risk_per_trade: float = 0.01
    max_exposure: float = 0.80
    max_leverage: float = 1.25
    max_single_position: float = 0.15
    max_concurrent: int = 5
    max_daily_trades: int = 20
    daily_dd_reduce: float = 0.02
    daily_dd_halt: float = 0.03
    weekly_dd_reduce: float = 0.05
    weekly_dd_halt: float = 0.07
    max_dd_from_peak: float = 0.10

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> RiskConfig:
        """Build a config from the parsed ``risk`` settings section."""
        raise NotImplementedError


class RiskManager:
    """Stateful risk gate tracking equity history and trade counts."""

    def __init__(self, config: RiskConfig) -> None:
        """Initialise with config; equity marks are recorded via :meth:`update_equity`."""
        self.config = config
        self.peak_equity: float | None = None
        self.day_start_equity: float | None = None
        self.week_start_equity: float | None = None
        self.trades_today: int = 0
        self.state: TradingState = TradingState.NORMAL

    def update_equity(self, equity: float, timestamp: datetime) -> None:
        """Record an equity mark, rolling day/week anchors and the running peak."""
        raise NotImplementedError

    def daily_drawdown(self, equity: float) -> float:
        """Return drawdown from the start-of-day equity as a positive fraction."""
        raise NotImplementedError

    def weekly_drawdown(self, equity: float) -> float:
        """Return drawdown from the start-of-week equity as a positive fraction."""
        raise NotImplementedError

    def drawdown_from_peak(self, equity: float) -> float:
        """Return drawdown from the all-time equity peak as a positive fraction."""
        raise NotImplementedError

    def evaluate_state(self, equity: float) -> TradingState:
        """Derive the trading state from drawdown thresholds (reduce/halt/kill)."""
        raise NotImplementedError

    def position_size(
        self, equity: float, entry_price: float, stop_price: float
    ) -> int:
        """Shares to buy so loss at ``stop_price`` <= ``max_risk_per_trade`` * equity."""
        raise NotImplementedError

    def apply_limits(
        self, target: AllocationTarget, equity: float, open_positions: int
    ) -> RiskDecision:
        """Clip a target allocation to exposure, leverage, per-position, and count caps."""
        raise NotImplementedError

    def can_trade(self) -> bool:
        """True if trading isn't halted and ``max_daily_trades`` isn't reached."""
        raise NotImplementedError

    def record_trade(self) -> None:
        """Increment today's trade counter."""
        raise NotImplementedError

    def reset_daily(self) -> None:
        """Reset per-day counters and anchors at the session open."""
        raise NotImplementedError
