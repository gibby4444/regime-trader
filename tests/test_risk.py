"""Tests for core.risk_manager."""

from __future__ import annotations

import pytest

from core.risk_manager import RiskConfig, RiskManager, TradingState  # noqa: F401

pytestmark = pytest.mark.skip(reason="TODO: implement once RiskManager is implemented")


def test_position_size_respects_max_risk_per_trade() -> None:
    """Loss at stop never exceeds max_risk_per_trade * equity."""


def test_caps_single_position_and_exposure() -> None:
    """Weights are clipped to max_single_position and max_exposure * max_leverage."""


def test_max_concurrent_positions() -> None:
    """No more than max_concurrent positions are allowed."""


def test_daily_drawdown_reduce_and_halt() -> None:
    """State goes REDUCED at daily_dd_reduce and HALTED at daily_dd_halt."""


def test_weekly_drawdown_reduce_and_halt() -> None:
    """State goes REDUCED at weekly_dd_reduce and HALTED at weekly_dd_halt."""


def test_kill_switch_on_peak_drawdown() -> None:
    """State goes KILLED at max_dd_from_peak."""


def test_max_daily_trades() -> None:
    """can_trade() is False after max_daily_trades trades."""
