"""Risk management: position sizing, leverage/exposure caps, and drawdown limits.

Every proposed allocation passes through :class:`RiskManager`, which clips it to
the ``risk`` settings and can reduce or halt trading on daily, weekly, or
peak-to-trough drawdown breaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

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
    reduce_size_mult: float = 0.50

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> RiskConfig:
        """Build a config from the parsed ``risk`` settings section."""
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown risk config keys: {sorted(unknown)}")
        return cls(**cfg)


def _drawdown(anchor: float | None, equity: float) -> float:
    if anchor is None or anchor <= 0:
        return 0.0
    return max(0.0, 1.0 - equity / anchor)


class RiskManager:
    """Stateful risk gate tracking equity history and trade counts.

    Day/week anchors are the last equity mark before the new day/week, so on
    daily bars the daily drawdown is that bar's close-to-close loss. KILLED is
    sticky until :meth:`rearm`.
    """

    def __init__(self, config: RiskConfig) -> None:
        """Initialise with config; equity marks are recorded via :meth:`update_equity`."""
        self.config = config
        self.peak_equity: float | None = None
        self.day_start_equity: float | None = None
        self.week_start_equity: float | None = None
        self.trades_today: int = 0
        self.state: TradingState = TradingState.NORMAL
        self._last_equity: float | None = None
        self._last_day: Any = None
        self._last_week: tuple[int, int] | None = None

    def update_equity(self, equity: float, timestamp: datetime) -> None:
        """Record an equity mark, rolling day/week anchors and the running peak."""
        ts = pd.Timestamp(timestamp)
        day, week = ts.date(), tuple(ts.isocalendar())[:2]
        if self._last_equity is None:
            self.day_start_equity = self.week_start_equity = self.peak_equity = equity
        else:
            if day != self._last_day:
                self.day_start_equity = self._last_equity
                self.trades_today = 0
            if week != self._last_week:
                self.week_start_equity = self._last_equity
        self.peak_equity = max(self.peak_equity or equity, equity)
        self._last_equity, self._last_day, self._last_week = equity, day, week

    def daily_drawdown(self, equity: float) -> float:
        """Return drawdown from the start-of-day equity as a positive fraction."""
        return _drawdown(self.day_start_equity, equity)

    def weekly_drawdown(self, equity: float) -> float:
        """Return drawdown from the start-of-week equity as a positive fraction."""
        return _drawdown(self.week_start_equity, equity)

    def drawdown_from_peak(self, equity: float) -> float:
        """Return drawdown from the all-time equity peak as a positive fraction."""
        return _drawdown(self.peak_equity, equity)

    def evaluate_state(self, equity: float) -> TradingState:
        """Derive the trading state from drawdown thresholds (reduce/halt/kill)."""
        c = self.config
        daily, weekly = self.daily_drawdown(equity), self.weekly_drawdown(equity)
        if self.state is TradingState.KILLED or self.drawdown_from_peak(equity) >= c.max_dd_from_peak:
            self.state = TradingState.KILLED
        elif daily >= c.daily_dd_halt or weekly >= c.weekly_dd_halt:
            self.state = TradingState.HALTED
        elif daily >= c.daily_dd_reduce or weekly >= c.weekly_dd_reduce:
            self.state = TradingState.REDUCED
        else:
            self.state = TradingState.NORMAL
        return self.state

    def rearm(self, equity: float) -> None:
        """Clear a kill-switch and restart the peak at ``equity`` (e.g. after a model retrain)."""
        self.state = TradingState.NORMAL
        self.peak_equity = equity

    def position_size(
        self, equity: float, entry_price: float, stop_price: float
    ) -> int:
        """Shares to buy so loss at ``stop_price`` <= ``max_risk_per_trade`` * equity."""
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0 or equity <= 0:
            return 0
        return int(np.floor(self.config.max_risk_per_trade * equity / risk_per_share))

    def apply_limits(
        self, target: AllocationTarget, equity: float, open_positions: int
    ) -> RiskDecision:
        """Clip a target allocation to exposure, leverage, per-position, and count caps."""
        raise NotImplementedError

    def can_trade(self) -> bool:
        """True if trading isn't halted and ``max_daily_trades`` isn't reached."""
        if self.state in (TradingState.HALTED, TradingState.KILLED):
            return False
        return self.trades_today < self.config.max_daily_trades

    def record_trade(self) -> None:
        """Increment today's trade counter."""
        self.trades_today += 1

    def reset_daily(self) -> None:
        """Reset per-day counters and anchors at the session open."""
        self.trades_today = 0
        if self._last_equity is not None:
            self.day_start_equity = self._last_equity
