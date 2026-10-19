"""Order placement, modification, and cancellation.

Translates :class:`core.regime_strategies.Signal` objects into broker orders,
enforces idempotency (client order IDs), and reports fills back to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from broker.alpaca_client import AlpacaClient

if TYPE_CHECKING:
    from core.regime_strategies import Signal
    from core.risk_manager import RiskManager


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass(frozen=True)
class OrderRequest:
    """Broker-agnostic order intent."""

    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "day"
    client_order_id: str | None = None


@dataclass(frozen=True)
class OrderResult:
    """Broker response for a submitted order."""

    order_id: str
    client_order_id: str
    symbol: str
    status: str
    filled_qty: float
    filled_avg_price: float | None


class OrderExecutor:
    """Submits and manages orders through :class:`AlpacaClient`."""

    def __init__(self, client: AlpacaClient, risk: RiskManager) -> None:
        """Bind the broker client and risk manager (for trade-count gating)."""
        self.client = client
        self.risk = risk

    def signals_to_orders(
        self, signals: list[Signal], equity: float, prices: dict[str, float]
    ) -> list[OrderRequest]:
        """Convert weight-based signals into share-quantity order requests.

        Sells are ordered before buys to free buying power.
        """
        raise NotImplementedError

    def submit(self, order: OrderRequest) -> OrderResult:
        """Submit a single order after checking ``risk.can_trade()``."""
        raise NotImplementedError

    def submit_batch(self, orders: list[OrderRequest]) -> list[OrderResult]:
        """Submit orders sequentially, stopping if the risk gate closes."""
        raise NotImplementedError

    def modify(
        self,
        order_id: str,
        qty: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> OrderResult:
        """Replace an open order's quantity and/or prices."""
        raise NotImplementedError

    def cancel(self, order_id: str) -> None:
        """Cancel a single open order."""
        raise NotImplementedError

    def cancel_all(self) -> None:
        """Cancel every open order (used on halt / shutdown)."""
        raise NotImplementedError

    def close_all_positions(self) -> list[OrderResult]:
        """Liquidate all positions (kill switch)."""
        raise NotImplementedError

    def wait_for_fill(self, order_id: str, timeout_seconds: float = 30.0) -> OrderResult:
        """Poll until the order is filled, cancelled, or ``timeout_seconds`` elapses."""
        raise NotImplementedError
