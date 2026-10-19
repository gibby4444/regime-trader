"""Terminal live dashboard built on ``rich``.

Shows current regime and confidence, target vs actual allocation, positions and
P&L, risk state/drawdowns, and recent orders/alerts. Refreshes every
``monitoring.dashboard_refresh_seconds``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table


@dataclass
class DashboardState:
    """Snapshot of everything the dashboard renders."""

    regime: str = "unknown"
    confidence: float = 0.0
    equity: float = 0.0
    daily_pnl: float = 0.0
    drawdown: float = 0.0
    trading_state: str = "normal"
    positions: list[dict[str, Any]] = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)
    recent_events: list[str] = field(default_factory=list)


class Dashboard:
    """Renders :class:`DashboardState` in a live-updating terminal layout."""

    def __init__(self, refresh_seconds: float = 5.0, console: Console | None = None) -> None:
        """Configure refresh cadence and output console."""
        self.refresh_seconds = refresh_seconds
        self.console = console or Console()
        self.state = DashboardState()
        self._live: Live | None = None

    def build_layout(self) -> Layout:
        """Compose header, regime, positions, risk, and events panels."""
        raise NotImplementedError

    def positions_table(self) -> Table:
        """Render open positions with qty, price, value, and unrealised P&L."""
        raise NotImplementedError

    def update(self, state: DashboardState) -> None:
        """Replace the current state and trigger a re-render."""
        raise NotImplementedError

    def start(self) -> None:
        """Begin the ``rich.live.Live`` rendering loop."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop live rendering and restore the terminal."""
        raise NotImplementedError
