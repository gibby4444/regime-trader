"""Tests for broker.order_executor (against a mocked AlpacaClient)."""

from __future__ import annotations

import pytest

from broker.order_executor import OrderExecutor, OrderRequest, OrderSide  # noqa: F401

pytestmark = pytest.mark.skip(reason="TODO: implement once OrderExecutor is implemented")


def test_signals_to_orders_sells_before_buys() -> None:
    """Sell orders are emitted before buy orders."""


def test_submit_blocked_when_risk_halted() -> None:
    """submit() refuses when risk.can_trade() is False."""


def test_modify_order() -> None:
    """modify() replaces qty/limit on an open order."""


def test_cancel_and_cancel_all() -> None:
    """cancel() and cancel_all() call the broker cancel endpoints."""


def test_client_order_id_idempotent() -> None:
    """Resubmitting with the same client_order_id doesn't duplicate orders."""
