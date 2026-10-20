"""Five-second Rich monitoring dashboard."""
from dataclasses import dataclass, field
from typing import Any
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class DashboardState:
    regime: str = 'unknown'
    confidence: float = 0
    equity: float = 0
    daily_pnl: float = 0
    drawdown: float = 0
    trading_state: str = 'normal'
    positions: list[dict[str, Any]] = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)
    recent_events: list[str] = field(default_factory=list)
    stability: int = 0
    flicker: int = 0
    flicker_window: int = 20
    allocation: float = 0
    leverage: float = 0
    daily_drawdown: float = 0
    daily_limit: float = .03
    peak_limit: float = .10
    data_ok: bool = True
    api_ok: bool = True
    api_latency_ms: float = 0
    hmm_age_days: float | None = None
    paper: bool = True


def risk_color(value, limit):
    ratio = value/limit if limit > 0 else 1
    return 'red' if ratio >= 1 else 'yellow' if ratio >= .7 else 'green'


def risk_bar(label, value, limit):
    color = risk_color(value, limit)
    filled = min(10, max(0, int(value/limit*10))) if limit > 0 else 10
    return f'{label}: [{color}]{"█"*filled}{"░"*(10-filled)} {value:.1%}/{limit:.0%}[/{color}]'


class Dashboard:
    def __init__(self, refresh_seconds=5, console=None):
        self.refresh_seconds = refresh_seconds
        self.console = console or Console()
        self.state = DashboardState()
        self._live = None

    def build_layout(self):
        s = self.state
        base = s.equity-s.daily_pnl
        daily_pct = s.daily_pnl/base if base else 0
        age = 'unknown' if s.hmm_age_days is None else f'{s.hmm_age_days:.1f}d ago'
        panels = [
            ('REGIME', f'{s.regime} ({s.confidence:.0%}) │ Stability: {s.stability} bars │ Flicker: {s.flicker}/{s.flicker_window}', 3),
            ('PORTFOLIO', f'Equity: ${s.equity:,.0f} │ Daily: {s.daily_pnl:+,.0f} ({daily_pct:+.2%})\nAllocation: {s.allocation:.0%} │ Leverage: {s.leverage:.2f}x', 4),
            ('POSITIONS', self.positions_table(), max(6, len(s.positions)+6)),
            ('RECENT SIGNALS', '\n'.join(s.recent_events[-5:]) or 'None', 7),
            ('RISK STATUS', Text.from_markup(risk_bar('Daily DD', s.daily_drawdown, s.daily_limit)+' │ '+risk_bar('From Peak', s.drawdown, s.peak_limit)+'\n'+s.trading_state), 4),
            ('SYSTEM', f'Data: {"✅" if s.data_ok else "❌"} │ API: {"✅" if s.api_ok else "❌"} {s.api_latency_ms:.0f}ms │ HMM: {age} │ {"PAPER" if s.paper else "LIVE"}', 3)]
        layout = Layout()
        layout.split_column(*(Layout(Panel(body, title=title), name=title, size=size) for title, body, size in panels))
        return layout

    def positions_table(self):
        table = Table(expand=True)
        for name in ('Symbol', 'Side', 'Price', 'P&L', 'Stop', 'Age'):
            table.add_column(name)
        for p in self.state.positions:
            cost = abs(p.get('qty', 0)*p.get('entry_price', 0))
            pct = p.get('unrealized_pnl', 0)/cost if cost else 0
            table.add_row(str(p.get('symbol', '')), 'LONG' if p.get('qty', 0)>0 else 'SHORT',
                          f'${p.get("current_price", 0):.2f}', f'{pct:+.1%}',
                          str(p.get('stop', '')), str(p.get('holding_period', '')))
        return table

    def update(self, state):
        self.state = state
        if self._live:
            self._live.update(self.build_layout())

    def start(self):
        if self._live is None:
            self._live = Live(self.build_layout(), console=self.console, refresh_per_second=1/self.refresh_seconds)
            self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()
            self._live = None
