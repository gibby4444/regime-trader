"""Tests for core.risk_manager."""

from __future__ import annotations

import pandas as pd
import pytest

from core.risk_manager import RiskConfig, RiskManager, TradingState


def marks(rm: RiskManager, values: list[float], start: str = "2024-01-01") -> TradingState:
    state = TradingState.NORMAL
    for ts, eq in zip(pd.bdate_range(start, periods=len(values)), values):
        rm.update_equity(eq, ts)
        state = rm.evaluate_state(eq)
    return state


def test_from_dict_rejects_unknown_keys() -> None:
    assert RiskConfig.from_dict({"max_leverage": 1.5}).max_leverage == 1.5
    with pytest.raises(ValueError, match="Unknown"):
        RiskConfig.from_dict({"bogus": 1})


def test_position_size_respects_max_risk_per_trade() -> None:
    """Loss at stop never exceeds max_risk_per_trade * equity."""
    rm = RiskManager(RiskConfig(max_risk_per_trade=0.01))
    shares = rm.position_size(100_000, 50.0, 48.0)
    assert shares == 500 and shares * 2.0 <= 1_000
    assert rm.position_size(100_000, 50.0, 51.0) == 0


@pytest.mark.skip(reason="TODO: apply_limits not implemented (backtest applies leverage cap + drawdown states only)")
def test_caps_single_position_and_exposure() -> None:
    """Weights are clipped to max_single_position and max_exposure * max_leverage."""


@pytest.mark.skip(reason="TODO: apply_limits not implemented")
def test_max_concurrent_positions() -> None:
    """No more than max_concurrent positions are allowed."""


def test_daily_drawdown_reduce_and_halt() -> None:
    """State goes REDUCED at daily_dd_reduce and HALTED at daily_dd_halt."""
    cfg = RiskConfig(weekly_dd_reduce=1, weekly_dd_halt=1, max_dd_from_peak=1)
    assert marks(RiskManager(cfg), [100, 101, 98.9]) is TradingState.REDUCED
    assert marks(RiskManager(cfg), [100, 101, 97.9]) is TradingState.HALTED
    assert marks(RiskManager(cfg), [100, 101, 97.9, 98.0]) is TradingState.NORMAL


def test_weekly_drawdown_reduce_and_halt() -> None:
    """State goes REDUCED at weekly_dd_reduce and HALTED at weekly_dd_halt."""
    cfg = RiskConfig(daily_dd_reduce=1, daily_dd_halt=1, max_dd_from_peak=1)
    # 2024-01-05 is a Friday; the following Monday starts a new week anchored at 100.
    week = [90, 95, 98, 99, 100]
    assert marks(RiskManager(cfg), week + [99, 98, 96, 95]) is TradingState.REDUCED
    assert marks(RiskManager(cfg), week + [99, 97, 95, 92.9]) is TradingState.HALTED
    rm = RiskManager(cfg)
    marks(rm, week + [99])
    assert rm.week_start_equity == 100


def test_kill_switch_on_peak_drawdown() -> None:
    """State goes KILLED at max_dd_from_peak, stays killed, and rearm() clears it."""
    cfg = RiskConfig(daily_dd_reduce=1, daily_dd_halt=1, weekly_dd_reduce=1, weekly_dd_halt=1)
    rm = RiskManager(cfg)
    assert marks(rm, [100, 110, 105, 99.5]) is TradingState.NORMAL
    assert marks(rm, [98.9], start="2024-01-05") is TradingState.KILLED
    assert marks(rm, [120], start="2024-01-08") is TradingState.KILLED
    rm.rearm(99)
    assert rm.peak_equity == 99 and rm.evaluate_state(99) is TradingState.NORMAL


def test_max_daily_trades() -> None:
    """can_trade() is False after max_daily_trades trades."""
    rm = RiskManager(RiskConfig(max_daily_trades=2))
    rm.update_equity(100, pd.Timestamp("2024-01-02"))
    rm.record_trade()
    assert rm.can_trade()
    rm.record_trade()
    assert not rm.can_trade()
    rm.update_equity(100, pd.Timestamp("2024-01-03"))
    assert rm.can_trade(), "counter resets on a new day"
