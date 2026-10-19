"""Open-position and P&L tracking, reconciled against the broker."""

from __future__ import annotations

from dataclasses import dataclass

from broker.alpaca_client import AlpacaClient


@dataclass
class Position:
    """A single open position."""

    symbol: str
    qty: float
    avg_entry_price: float
    market_price: float

    @property
    def market_value(self) -> float:
        """Current market value (qty * market_price)."""
        raise NotImplementedError

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L versus average entry price."""
        raise NotImplementedError


class PositionTracker:
    """Maintains a local view of positions and realised/unrealised P&L."""

    def __init__(self, client: AlpacaClient) -> None:
        """Bind the broker client; positions are populated by :meth:`sync`."""
        self.client = client
        self.positions: dict[str, Position] = {}
        self.realized_pnl: float = 0.0

    def sync(self) -> None:
        """Refresh positions from the broker, replacing local state."""
        raise NotImplementedError

    def update_prices(self, prices: dict[str, float]) -> None:
        """Mark positions to the supplied latest prices."""
        raise NotImplementedError

    def on_fill(self, symbol: str, qty: float, price: float) -> None:
        """Apply a fill (signed qty), updating avg price and realised P&L."""
        raise NotImplementedError

    def weights(self, equity: float) -> dict[str, float]:
        """Return each position's market value as a fraction of ``equity``."""
        raise NotImplementedError

    def gross_exposure(self) -> float:
        """Sum of absolute market values."""
        raise NotImplementedError

    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across positions."""
        raise NotImplementedError

    def count(self) -> int:
        """Number of open positions."""
        raise NotImplementedError
