"""Tests for core.risk_manager: breakers, sizing, leverage, validation, correlation, caps."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from core.regime_strategies import Direction, Signal
from core.risk_manager import (
    BreakerType,
    CircuitBreaker,
    PortfolioState,
    PositionInfo,
    RiskConfig,
    RiskDecision,
    RiskManager,
    TradingState,
)

EQ = 100_000.0
SETTINGS = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def sig(
    symbol: str = "SPY",
    weight: float = 0.15,
    entry: float = 100.0,
    stop: float | None = 99.0,
    vol: str = "mid",
    leverage: float = 1.0,
    uncertain: bool = False,
    direction: Direction = Direction.LONG,
) -> Signal:
    return Signal(
        symbol=symbol, direction=direction, confidence=0.9, entry_price=entry, stop_loss=stop,  # type: ignore[arg-type]
        take_profit=None, position_size_pct=weight / leverage, leverage=leverage, regime_id=0,
        regime_name=f"{vol}_vol", regime_probability=0.9, timestamp=None, reasoning="test",
        strategy_name="Test", metadata={"target_weight": weight, "vol_regime": vol, "uncertain": uncertain},
    )


def pos(symbol: str, weight: float, equity: float = EQ, price: float = 100.0) -> PositionInfo:
    return PositionInfo(symbol, weight * equity / price, price, price * 0.95)


def state(equity: float = EQ, positions: dict[str, PositionInfo] | None = None, **kw: object) -> PortfolioState:
    params: dict[str, object] = dict(equity=equity, cash=equity, buying_power=equity * 2, positions=positions or {})
    params.update(kw)
    return PortfolioState(**params)  # type: ignore[arg-type]


def held(*items: tuple[str, float]) -> dict[str, PositionInfo]:
    return {s: pos(s, w) for s, w in items}


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(RiskConfig(), lock_path=None)


def weight(d: RiskDecision) -> float:
    assert d.approved and d.modified_signal is not None, d
    return d.modified_signal.metadata["target_weight"]


def marks(rm: RiskManager, values: list[float], dates: list[str] | pd.DatetimeIndex) -> list[TradingState]:
    out = []
    for ts, eq in zip(pd.DatetimeIndex(dates), values):
        rm.update_equity(eq, ts)
        out.append(rm.evaluate_state(eq))
    return out


# ------------------------------------------------------------------ config


def test_settings_yaml_drives_every_threshold() -> None:
    cfg = RiskConfig.from_dict(yaml.safe_load(SETTINGS.read_text())["risk"])
    assert (cfg.max_exposure, cfg.max_single_position, cfg.max_sector_exposure) == (0.80, 0.15, 0.30)
    assert (cfg.max_concurrent, cfg.max_daily_trades, cfg.max_leverage, cfg.default_leverage) == (5, 20, 1.25, 1.0)
    assert (cfg.daily_dd_reduce, cfg.daily_dd_halt, cfg.weekly_dd_reduce, cfg.weekly_dd_halt) == (0.02, 0.03, 0.05, 0.07)
    assert cfg.max_dd_from_peak == 0.10 and cfg.lock_file == "trading_halted.lock"
    assert (cfg.max_risk_per_trade, cfg.min_position_value, cfg.gap_stop_multiple, cfg.max_gap_loss) == (0.01, 100, 3, 0.02)
    assert (cfg.max_spread_pct, cfg.duplicate_window_seconds) == (0.005, 60)
    assert (cfg.correlation_window, cfg.correlation_reduce, cfg.correlation_reject) == (60, 0.70, 0.85)
    assert cfg.sector("spy") == "broad_market" and cfg.sector("NVDA") == "technology"
    assert cfg.sector("XYZ") == "unclassified:XYZ", "unmapped symbols form their own sector"


def test_from_dict_rejects_unknown_keys() -> None:
    assert RiskConfig.from_dict({"max_leverage": 1.5}).max_leverage == 1.5
    with pytest.raises(ValueError, match="Unknown"):
        RiskConfig.from_dict({"bogus": 1})


# ---------------------------------------------------------- circuit breakers


def pnl(equity: float, day_start: float = EQ, week_start: float = EQ, peak: float = EQ, **kw: object) -> PortfolioState:
    return state(
        equity, positions=held(("SPY", 0.5), ("AAPL", 0.1)), daily_pnl=equity - day_start,
        weekly_pnl=equity - week_start, day_start_equity=day_start, week_start_equity=week_start,
        peak_equity=peak, drawdown=max(0.0, 1 - equity / peak), regime="bear", **kw,
    )


def test_daily_breakers_reduce_then_halt_and_reset() -> None:
    cb = CircuitBreaker(RiskConfig())
    assert cb.update(pnl(98_000)) == [], "exactly 2% does not trigger (strictly greater)"
    assert cb.check().state is TradingState.NORMAL and cb.check().size_multiplier == 1.0

    (ev,) = cb.update(pnl(97_900))
    st = cb.check()
    assert ev["breaker"] == "daily_reduce" and ev["positions_closed"] == []
    assert st.state is TradingState.REDUCED and st.size_multiplier == 0.5 and not st.close_all
    assert cb.update(pnl(97_800)) == [], "an active breaker is logged once"

    (ev,) = cb.update(pnl(96_900))
    st = cb.check()
    assert ev["breaker"] == "daily_halt" and ev["positions_closed"] == ["SPY", "AAPL"]
    assert st.state is TradingState.HALTED and st.close_all and st.size_multiplier == 0.0
    assert st.active == (BreakerType.DAILY_REDUCE, BreakerType.DAILY_HALT)

    cb.reset_weekly()
    assert cb.check().state is TradingState.HALTED, "weekly reset leaves daily breakers alone"
    cb.reset_daily()
    assert cb.check().state is TradingState.NORMAL


def test_weekly_breakers_survive_daily_reset() -> None:
    cb = CircuitBreaker(RiskConfig())
    cb.update(pnl(94_500, day_start=95_000))
    assert cb.check().state is TradingState.REDUCED
    cb.reset_daily()
    assert cb.check().state is TradingState.REDUCED
    (ev,) = cb.update(pnl(92_900, day_start=94_000))
    assert ev["breaker"] == "weekly_halt" and cb.check().close_all
    cb.reset_daily()
    assert cb.check().state is TradingState.HALTED
    cb.reset_weekly()
    assert cb.check().state is TradingState.NORMAL


def test_peak_breaker_writes_lock_file_that_must_be_deleted(tmp_path: Path) -> None:
    lock = tmp_path / "trading_halted.lock"
    cb = CircuitBreaker(RiskConfig(), lock)
    events = cb.update(pnl(89_000, day_start=89_500, week_start=89_500))
    assert [e["breaker"] for e in events] == ["peak_halt"]
    assert cb.check().state is TradingState.LOCKED and cb.check().close_all
    payload = json.loads(lock.read_text())
    assert payload["breaker"] == "peak_halt" and "Delete this file" in payload["message"]

    cb.reset_daily()
    cb.reset_weekly()
    assert cb.check().state is TradingState.LOCKED, "only deleting the file clears the lock"
    assert CircuitBreaker(RiskConfig(), lock).check().state is TradingState.LOCKED, "survives restarts"

    lock.unlink()
    assert cb.check().state is TradingState.NORMAL


def test_breaker_history_and_structured_log(caplog: pytest.LogCaptureFixture) -> None:
    cb = CircuitBreaker(RiskConfig())
    ts = pd.Timestamp("2024-03-05 15:30")
    with caplog.at_level(logging.WARNING, logger="regime_trader.risk"):
        cb.update(pnl(96_500, timestamp=ts))
    hist = cb.get_history()
    assert [h["breaker"] for h in hist] == ["daily_halt", "daily_reduce"]
    h = hist[0]
    assert h["drawdown"] == pytest.approx(0.035) and h["threshold"] == 0.03
    assert h["equity"] == 96_500 and h["regime"] == "bear" and h["timestamp"] == ts
    assert h["positions_closed"] == ["SPY", "AAPL"]
    logged = [json.loads(r.getMessage().split(": ", 1)[1]) for r in caplog.records]
    assert [e["breaker"] for e in logged] == ["daily_halt", "daily_reduce"]
    hist.clear()
    assert len(cb.get_history()) == 2, "get_history returns a copy"


def test_risk_manager_rolls_daily_and_weekly_anchors(rm: RiskManager) -> None:
    states = marks(rm, [100_000, 97_500, 97_500], ["2024-01-02", "2024-01-03", "2024-01-04"])
    assert states == [TradingState.NORMAL, TradingState.REDUCED, TradingState.NORMAL]

    rm2 = RiskManager(RiskConfig(), lock_path=None)
    week = pd.bdate_range("2024-01-08", periods=6)  # Mon..Fri, next Mon
    states = marks(rm2, [100_000, 98_500, 97_000, 95_500, 94_000, 94_000], week)
    assert states[:4] == [TradingState.NORMAL] * 4, "each day only ~1.5% down"
    assert states[4] is TradingState.REDUCED, "6% down on the week"
    assert rm2.weekly_drawdown(94_000) == 0.0 and states[5] is TradingState.NORMAL, "new week"


def test_risk_manager_lock_resumes_after_manual_delete_with_reset_peak(tmp_path: Path) -> None:
    lock = tmp_path / "trading_halted.lock"
    rm = RiskManager(RiskConfig(), lock_path=lock)
    days = pd.bdate_range("2024-01-04", periods=4)  # Thu, Fri, Mon, Tue
    assert marks(rm, [100_000, 89_000, 89_000], days[:3])[1:] == [TradingState.LOCKED] * 2
    assert lock.exists() and not rm.can_trade()
    assert rm.validate_signal(sig(), rm.build_state(89_000, 89_000)).rejection_reason == "circuit_breaker_halt"
    lock.unlink()
    assert marks(rm, [89_000], days[3:]) == [TradingState.NORMAL]
    assert rm.peak_equity == 89_000 and rm.drawdown_from_peak(89_000) == 0.0


# ------------------------------------------------------------------ sizing


def test_one_percent_and_gap_risk_sizing(rm: RiskManager) -> None:
    assert rm.position_size(EQ, 100.0, 95.0) == 200, "1% of 100k / $5 stop distance"
    assert rm.gap_position_size(EQ, 100.0, 95.0) == 133, "a 3x stop gap ($15) loses at most 2%"
    assert rm.position_size(EQ, 100.0, 100.0) == 0 and rm.gap_position_size(0, 100.0, 95.0) == 0

    d = rm.validate_signal(sig(weight=0.5, stop=95.0), state())
    assert weight(d) == pytest.approx(0.133)
    assert any("gap-risk" in m for m in d.modifications)

    d = rm.validate_signal(sig(weight=0.5, stop=90.0), state())
    assert weight(d) == pytest.approx(0.066), "min(normal=$10k, gap=$6.6k)"


def test_regime_and_single_position_caps(rm: RiskManager) -> None:
    d = rm.validate_signal(sig(weight=0.95), state())
    assert weight(d) == pytest.approx(0.15) and any("max position" in m for m in d.modifications)
    assert weight(rm.validate_signal(sig(weight=0.95, vol="high"), state())) == pytest.approx(0.10)

    rm_loose = RiskManager(RiskConfig(regime_max_position={"low": 0.5, "mid": 0.5, "high": 0.5}), lock_path=None)
    assert weight(rm_loose.validate_signal(sig(weight=0.95), state())) == pytest.approx(0.15), "15% hard cap"


def test_stop_loss_required(rm: RiskManager) -> None:
    for stop in (None, float("nan"), 0.0, 100.0, 101.0):
        d = rm.validate_signal(sig(stop=stop), state())
        assert not d.approved and d.rejection_reason == "missing_stop_loss" and d.modified_signal is None


def test_minimum_position_value(rm: RiskManager) -> None:
    d = rm.validate_signal(sig(weight=0.01), state(5_000.0))
    assert d.rejection_reason == "below_min_position"
    assert weight(rm.validate_signal(sig(weight=0.02), state(5_000.0))) == pytest.approx(0.02)


# ------------------------------------------------------------------ leverage


def test_leverage_only_for_calm_low_vol(rm: RiskManager) -> None:
    calm = state()
    assert rm.allowed_leverage(sig(vol="low", leverage=1.25), calm) == (1.25, [])
    assert rm.allowed_leverage(sig(vol="low", leverage=2.0), calm)[0] == 1.25, "capped at max_leverage"
    assert rm.allowed_leverage(sig(vol="mid", leverage=1.25), calm) == (1.0, [])

    forced = {
        "uncertain": (sig(vol="low", leverage=1.25, uncertain=True), calm),
        "positions open": (sig(vol="low", leverage=1.25), state(positions=held(("AAPL", .1), ("GOOGL", .1), ("XOM", .1)))),
        "flicker": (sig(vol="low", leverage=1.25), state(flicker_rate=5)),
    }
    for why, (s, st) in forced.items():
        lev, mods = rm.allowed_leverage(s, st)
        assert lev == 1.0 and why in mods[0], why
    assert rm.allowed_leverage(sig(vol="low", leverage=1.25), state(flicker_rate=4))[0] == 1.25

    rm.circuit_breaker.update(pnl(97_500))
    lev, mods = rm.allowed_leverage(sig(vol="low", leverage=1.25), calm)
    assert lev == 1.0 and "circuit breaker active" in mods[0]


def test_gross_cap_scales_with_allowed_leverage(rm: RiskManager) -> None:
    book = held(("AAPL", 0.45), ("GOOGL", 0.45))
    d = rm.validate_signal(sig(vol="low", leverage=1.25), state(positions=book))
    assert weight(d) == pytest.approx(0.10), "80% x 1.25 = 100% gross, 90% used"
    assert d.modified_signal.leverage == 1.25
    d = rm.validate_signal(sig(vol="mid"), state(positions=book))
    assert d.rejection_reason == "no_capacity", "80% x 1.0 gross already exceeded"


def test_gross_exposure_cap_80pct(rm: RiskManager) -> None:
    book = held(("AAPL", 0.175), ("GOOGL", 0.175), ("AMZN", 0.175), ("XOM", 0.175))
    d = rm.validate_signal(sig(), state(positions=book))
    assert weight(d) == pytest.approx(0.10) and any("gross exposure" in m for m in d.modifications)


# ---------------------------------------------------------- portfolio limits


def test_sector_cap(rm: RiskManager) -> None:
    d = rm.validate_signal(sig("MSFT"), state(positions=held(("AAPL", 0.20))))
    assert weight(d) == pytest.approx(0.10) and any("sector technology" in m for m in d.modifications)
    assert rm.validate_signal(sig("NVDA"), state(positions=held(("AAPL", 0.15), ("MSFT", 0.15)))).rejection_reason \
        == "no_capacity"
    assert weight(rm.validate_signal(sig("XYZ"), state(positions=held(("ABC", 0.30))))) == pytest.approx(0.15)


def test_max_concurrent_positions(rm: RiskManager) -> None:
    book = held(("AAPL", .05), ("GOOGL", .05), ("AMZN", .05), ("XOM", .05), ("JPM", .05))
    assert rm.validate_signal(sig("SPY"), state(positions=book)).rejection_reason == "max_concurrent_positions"
    assert weight(rm.validate_signal(sig("AAPL", weight=0.10), state(positions=book))) == pytest.approx(0.10), \
        "adding to an existing position is not a new position"


def test_max_daily_trades(rm: RiskManager) -> None:
    for i in range(20):
        rm.record_order(f"S{i}", "buy", when=0.0)
    assert not rm.can_trade()
    assert rm.validate_signal(sig(), state(), now=1_000.0).rejection_reason == "max_daily_trades"
    rm.reset_daily()
    assert rm.can_trade() and rm.validate_signal(sig(), state(), now=1_000.0).approved


# ---------------------------------------------------------- order validation


def test_duplicate_buy_blocked_within_60s(rm: RiskManager) -> None:
    rm.record_order("SPY", "buy", when=1_000.0)
    rm.record_order("QQQ", "sell", when=1_000.0)
    assert rm.validate_signal(sig(), state(), now=1_059.0).rejection_reason == "duplicate_order"
    assert rm.validate_signal(sig(), state(), now=1_061.0).approved
    assert rm.validate_signal(sig("QQQ"), state(), now=1_001.0).approved, "a sell doesn't block a buy"
    rm.record_order("AAPL", "buy", when=pd.Timestamp("2024-01-02 10:00:00"))
    ts = pd.Timestamp("2024-01-02 10:00:30")
    assert rm.validate_signal(sig("AAPL"), state(timestamp=ts)).rejection_reason == "duplicate_order"


def test_spread_and_tradeable_checks(rm: RiskManager) -> None:
    assert rm.validate_signal(sig(), state(quotes={"SPY": (99.7, 100.3)})).rejection_reason == "spread_too_wide"
    assert rm.validate_signal(sig(), state(quotes={"SPY": (99.9, 100.1)})).approved
    assert rm.validate_signal(sig(), state(tradeable={"SPY": False})).rejection_reason == "not_tradeable"
    assert rm.validate_signal(sig(), state(tradeable={"SPY": True})).approved


def test_buying_power(rm: RiskManager) -> None:
    d = rm.validate_signal(sig(), state(buying_power=5_000.0))
    assert weight(d) == pytest.approx(0.05) and any("buying power" in m for m in d.modifications)
    assert rm.validate_signal(sig(), state(buying_power=50.0)).rejection_reason == "below_min_position"
    assert rm.validate_signal(sig(), state(buying_power=0.0)).rejection_reason == "no_capacity"


def test_rejections_are_logged_as_structured_json(rm: RiskManager, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="regime_trader.risk"):
        rm.validate_signal(sig(stop=None), state(timestamp=pd.Timestamp("2024-01-02")))
    rec = json.loads(caplog.records[-1].getMessage().split(": ", 1)[1])
    assert rec["event"] == "risk_rejection" and rec["reason"] == "missing_stop_loss"
    assert rec["symbol"] == "SPY" and rec["regime"] == "mid_vol" and rec["equity"] == EQ
    assert rm.rejections[-1]["reason"] == "missing_stop_loss"


# ---------------------------------------------------------- breaker effects


def test_halt_rejects_new_risk_but_approves_reductions(rm: RiskManager) -> None:
    rm.circuit_breaker.update(pnl(96_500))
    book = held(("SPY", 0.20))
    assert rm.validate_signal(sig("AAPL"), state(positions=book)).rejection_reason == "circuit_breaker_halt"
    d = rm.validate_signal(sig("SPY", weight=0.05, stop=None), state(positions=book))
    assert weight(d) == pytest.approx(0.05) and d.modifications == ["risk-reducing order"]
    flat = rm.validate_signal(sig("SPY", direction=Direction.FLAT), state(positions=book))
    assert weight(flat) == 0.0


def test_reduce_breaker_halves_size(rm: RiskManager) -> None:
    rm.circuit_breaker.update(pnl(97_500))
    d = rm.validate_signal(sig(), state())
    assert weight(d) == pytest.approx(0.075) and any("reduce x0.5" in m for m in d.modifications)


# ------------------------------------------------------------------ correlation


def returns_frame(rho: float, n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"AAPL": a, "SPY": rho * a + np.sqrt(1 - rho**2) * b})


def test_correlation_reduce_and_reject(rm: RiskManager) -> None:
    book = held(("AAPL", 0.10))
    mid = returns_frame(0.78)
    assert 0.70 < mid.corr().iloc[0, 1] < 0.85
    d = rm.validate_signal(sig(), state(positions=book, returns=mid))
    assert weight(d) == pytest.approx(0.075) and any("correlation" in m for m in d.modifications)

    d = rm.validate_signal(sig(), state(positions=book, returns=returns_frame(0.95)))
    assert d.rejection_reason == "correlation_too_high"

    low = returns_frame(0.1)
    assert weight(rm.validate_signal(sig(), state(positions=book, returns=low))) == pytest.approx(0.15)


def test_correlation_uses_rolling_60_bar_window(rm: RiskManager) -> None:
    old = returns_frame(0.0, n=200, seed=1)
    recent = returns_frame(1.0, n=60, seed=2)
    frame = pd.concat([old, recent], ignore_index=True)
    assert frame.corr().iloc[0, 1] < 0.7, "full history looks uncorrelated"
    d = rm.validate_signal(sig(), state(positions=held(("AAPL", 0.10)), returns=frame))
    assert d.rejection_reason == "correlation_too_high"


# ------------------------------------------------------------------ batch


def test_validate_signals_processes_reductions_first(rm: RiskManager) -> None:
    book = held(("AAPL", 0.45), ("GOOGL", 0.35))
    signals = [sig("SPY"), sig("AAPL", weight=0.10)]
    assert rm.validate_signal(signals[0], state(positions=book)).rejection_reason == "no_capacity"
    spy, aapl = rm.validate_signals(signals, state(positions=book))
    assert weight(aapl) == pytest.approx(0.10)
    assert weight(spy) == pytest.approx(0.15), "freed gross capacity is visible to later signals"
    assert book["AAPL"].qty == 450, "caller's state is not mutated"


def test_validate_signals_shares_capacity(rm: RiskManager) -> None:
    decisions = rm.validate_signals([sig("AAPL"), sig("MSFT"), sig("NVDA")], state())
    assert [weight(d) for d in decisions[:2]] == pytest.approx([0.15, 0.15])
    assert decisions[2].rejection_reason == "no_capacity", "technology sector full at 30%"
