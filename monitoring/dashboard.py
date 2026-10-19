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
        layout = Layout()
        header = Table(title=f"Regime: {self.state.regime} ({self.state.confidence:.1%})")
        header.add_column('Equity')
        header.add_column('Daily P&L')
        header.add_column('Risk state')
        header.add_row(f'{self.state.equity:,.2f}', f'{self.state.daily_pnl:,.2f}', self.state.trading_state)
        layout.split_column(Layout(header, size=5), Layout(self.positions_table()))
        return layout

    def positions_table(self) -> Table:
        """Render open positions with qty, price, value, and unrealised P&L."""
        table = Table(title='Positions')
        for name in ('symbol', 'qty', 'current_price', 'unrealized_pnl', 'stop'):
            table.add_column(name)
        for position in self.state.positions:
            table.add_row(*(str(position.get(k, '')) for k in
                            ('symbol', 'qty', 'current_price', 'unrealized_pnl', 'stop')))
        return table

    def update(self, state: DashboardState) -> None:
        """Replace the current state and trigger a re-render."""
        self.state = state
        if self._live is not None:
            self._live.update(self.build_layout())

    def start(self) -> None:
        """Begin the ``rich.live.Live`` rendering loop."""
        self._live = Live(self.build_layout(), console=self.console, auto_refresh=False)
        self._live.start()

    def stop(self) -> None:
        """Stop live rendering and restore the terminal."""
        if self._live is not None:
            self._live.stop()
            self._live = None
