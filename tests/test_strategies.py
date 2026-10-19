"""Tests for core.regime_strategies."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from core.hmm_engine import REGIME_LABELS, HMMEngine, RegimeInfo, RegimeState, regime_labels, volatility_bucket
from core.regime_strategies import (
    LABEL_TO_STRATEGY,
    BaseStrategy,
    BearTrendStrategy,
    BullTrendStrategy,
    CrashDefensiveStrategy,
    Direction,
    EuphoriaCautiousStrategy,
    HighVolDefensiveStrategy,
    LowVolBullStrategy,
    MeanReversionStrategy,
    MidVolCautiousStrategy,
    StrategyConfig,
    StrategyOrchestrator,
    VolRegime,
    vol_regime_for_rank,
)
from data.feature_engineering import true_range

SETTINGS = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
SYMBOLS = ["SPY", "QQQ", "AAPL"]
L, M, H = VolRegime.LOW, VolRegime.MID, VolRegime.HIGH


def make_bars(n: int = 150, drift: float = 0.004, vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = drift + vol * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(r))
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) * (1 + vol * np.abs(rng.standard_normal(n)))
    low = np.minimum(open_, close) * (1 - vol * np.abs(rng.standard_normal(n)))
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)


def ref_indicators(bars: pd.DataFrame) -> tuple[float, float, float]:
    """Independent EMA50 / Wilder ATR14 reference."""
    close = bars["close"]
    ema = close.ewm(span=50, adjust=False).mean().iloc[-1]
    tr = true_range(bars["high"], bars["low"], close).to_numpy()
    a = tr[0]
    for x in tr[1:]:
        a = a + (x - a) / 14
    return float(close.iloc[-1]), float(ema), float(a)


def make_infos(vols: list[float]) -> list[RegimeInfo]:
    """Regimes in return order (ids 0..n-1) with the given volatilities."""
    labels = regime_labels(len(vols))
    return [
        RegimeInfo(i, labels[i], -0.002 + 0.001 * i, v, "", 1.0, 0.15, 0.55)
        for i, v in enumerate(vols)
    ]


def make_state(sid: int, label: str, prob: float = 0.9, in_transition: bool = False, n: int = 4) -> RegimeState:
    probs = np.full(n, (1 - prob) / (n - 1))
    probs[sid] = prob
    return RegimeState(label, sid, prob, probs, None, not in_transition, 5, sid, in_transition, 0)


@pytest.fixture()
def up_bars() -> dict[str, pd.DataFrame]:
    return {s: make_bars(drift=0.004, seed=i) for i, s in enumerate(SYMBOLS)}


@pytest.fixture()
def down_bars() -> dict[str, pd.DataFrame]:
    return {s: make_bars(drift=-0.004, seed=i) for i, s in enumerate(SYMBOLS)}


# 4 regimes in return order CRASH, BEAR, BULL, EUPHORIA; vol order BULL < BEAR < EUPHORIA < CRASH
VOLS_4 = [0.030, 0.012, 0.006, 0.020]


@pytest.fixture()
def orch() -> StrategyOrchestrator:
    return StrategyOrchestrator(StrategyConfig(), make_infos(VOLS_4))


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (3, [L, M, H]),
        (4, [L, M, M, H]),
        (5, [L, L, M, H, H]),
        (6, [L, L, M, M, H, H]),
        (7, [L, L, M, M, M, H, H]),
    ],
)
def test_vol_rank_mapping(n: int, expected: list[VolRegime]) -> None:
    assert [vol_regime_for_rank(r, n) for r in range(n)] == expected
    buckets = {L: "calm", M: "moderate", H: "turbulent"}
    assert [volatility_bucket(r, n) for r in range(n)] == [buckets[v] for v in expected]


def test_orchestrator_uses_vol_rank_not_labels(orch: StrategyOrchestrator) -> None:
    assert orch.volatility_ranking == [2, 1, 3, 0]
    assert orch.vol_rank == {2: 0, 1: 1, 3: 2, 0: 3}
    assert isinstance(orch.get_strategy(2), LowVolBullStrategy)  # BULL, calmest
    assert isinstance(orch.get_strategy(1), MidVolCautiousStrategy)  # BEAR
    assert isinstance(orch.get_strategy(3), MidVolCautiousStrategy)  # EUPHORIA
    assert isinstance(orch.get_strategy(0), HighVolDefensiveStrategy)  # CRASH
    # A calm "CRASH"-labelled regime still gets the low-vol strategy.
    flipped = StrategyOrchestrator(StrategyConfig(), make_infos([0.005, 0.01, 0.02, 0.04]))
    assert isinstance(flipped.get_strategy(0), LowVolBullStrategy)
    assert isinstance(flipped.get_strategy(3), HighVolDefensiveStrategy)


def test_unknown_regime_raises(orch: StrategyOrchestrator) -> None:
    with pytest.raises(ValueError):
        orch.get_strategy(9)


def test_low_vol_signal(up_bars: dict[str, pd.DataFrame]) -> None:
    bars = up_bars["SPY"]
    sig = LowVolBullStrategy().generate_signal("SPY", bars, make_state(2, "BULL"))
    assert sig is not None
    price, ema, a = ref_indicators(bars)
    assert sig.direction is Direction.LONG
    assert sig.position_size_pct == 0.95 and sig.leverage == 1.25
    assert sig.entry_price == pytest.approx(price)
    assert sig.stop_loss == pytest.approx(min(max(price - 3 * a, ema - 0.5 * a), price - 0.5 * a))
    assert sig.take_profit is None
    assert sig.strategy_name == "LowVolBullStrategy"
    assert sig.timestamp == bars.index[-1]
    assert (sig.regime_id, sig.regime_name, sig.regime_probability, sig.confidence) == (2, "BULL", 0.9, 0.9)


def test_low_vol_stop_takes_tighter_of_atr_and_ema() -> None:
    bars = make_bars(drift=0.002, vol=0.02, seed=3)
    price, ema, a = ref_indicators(bars)
    sig = LowVolBullStrategy().generate_signal("X", bars, make_state(2, "BULL"))
    assert sig is not None
    assert sig.stop_loss >= price - 3 * a - 1e-9
    assert sig.stop_loss >= min(ema - 0.5 * a, price - 0.5 * a) - 1e-9


def test_mid_vol_trend_intact(up_bars: dict[str, pd.DataFrame]) -> None:
    bars = up_bars["SPY"]
    price, ema, a = ref_indicators(bars)
    assert price > ema
    sig = MidVolCautiousStrategy().generate_signal("SPY", bars, make_state(1, "BEAR"))
    assert sig is not None
    assert (sig.position_size_pct, sig.leverage) == (0.95, 1.0)
    assert sig.stop_loss == pytest.approx(min(ema - 0.5 * a, price - 0.5 * a))
    assert sig.metadata["trend_intact"] is True


def test_mid_vol_trend_broken_reduces_and_stays_long(down_bars: dict[str, pd.DataFrame]) -> None:
    bars = down_bars["SPY"]
    price, ema, a = ref_indicators(bars)
    assert price < ema
    sig = MidVolCautiousStrategy().generate_signal("SPY", bars, make_state(1, "BEAR"))
    assert sig is not None
    assert sig.direction is Direction.LONG
    assert (sig.position_size_pct, sig.leverage) == (0.60, 1.0)
    assert sig.stop_loss < sig.entry_price
    assert sig.stop_loss == pytest.approx(min(ema - 0.5 * a, price - 0.5 * a))


def test_high_vol_defensive(down_bars: dict[str, pd.DataFrame]) -> None:
    bars = down_bars["QQQ"]
    price, ema, a = ref_indicators(bars)
    sig = HighVolDefensiveStrategy().generate_signal("QQQ", bars, make_state(0, "CRASH"))
    assert sig is not None
    assert (sig.direction, sig.position_size_pct, sig.leverage) == (Direction.LONG, 0.60, 1.0)
    assert sig.stop_loss == pytest.approx(min(ema - 1.0 * a, price - 0.5 * a))


@pytest.mark.parametrize("drift", [-0.01, -0.003, 0.0, 0.003, 0.01])
@pytest.mark.parametrize("cls", [LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy])
def test_always_long_never_short(cls: type[BaseStrategy], drift: float) -> None:
    for seed in range(5):
        sig = cls().generate_signal("X", make_bars(drift=drift, vol=0.03, seed=seed), make_state(0, "CRASH"))
        assert sig is not None
        assert sig.direction is Direction.LONG
        assert 0.60 <= sig.position_size_pct <= 0.95
        assert sig.leverage in (1.0, 1.25)
        assert 0 < sig.stop_loss < sig.entry_price


def test_insufficient_bars_returns_none_and_is_skipped(orch: StrategyOrchestrator) -> None:
    short = make_bars(n=30)
    assert LowVolBullStrategy().generate_signal("X", short, make_state(2, "BULL")) is None
    sigs = orch.generate_signals(["X", "MISSING", "SPY"], {"X": short, "SPY": make_bars()}, make_state(2, "BULL"))
    assert [s.symbol for s in sigs] == ["SPY"]


def test_generate_signals_confident(orch: StrategyOrchestrator, up_bars: dict[str, pd.DataFrame]) -> None:
    sigs = orch.generate_signals(SYMBOLS, up_bars, make_state(2, "BULL", prob=0.9))
    assert [s.symbol for s in sigs] == SYMBOLS
    for s in sigs:
        assert (s.position_size_pct, s.leverage) == (0.95, 1.25)
        assert s.metadata["vol_regime"] == "low" and s.metadata["vol_rank"] == 0
        assert s.metadata["size_multiplier"] == 1.0
        assert s.metadata["target_weight"] == pytest.approx(0.95 * 1.25 / 3)
    target = orch.target_allocation(sigs)
    assert target is not None
    assert target.gross_exposure == pytest.approx(0.95 * 1.25)
    assert target.strategy_name == "LowVolBullStrategy" and target.vol_regime is L


def test_low_confidence_halves_size_and_forces_1x(orch: StrategyOrchestrator, up_bars: dict[str, pd.DataFrame]) -> None:
    sigs = orch.generate_signals(SYMBOLS, up_bars, make_state(2, "BULL", prob=0.50))
    for s in sigs:
        assert s.position_size_pct == pytest.approx(0.475)
        assert s.leverage == 1.0
        assert s.metadata["uncertain"] is True and "uncertain" in s.reasoning


def test_confidence_threshold_boundary(orch: StrategyOrchestrator, up_bars: dict[str, pd.DataFrame]) -> None:
    sigs = orch.generate_signals(SYMBOLS, up_bars, make_state(2, "BULL", prob=0.55))
    assert all(s.metadata["uncertain"] is False and s.leverage == 1.25 for s in sigs)


def test_flickering_halves_size(orch: StrategyOrchestrator, up_bars: dict[str, pd.DataFrame]) -> None:
    sigs = orch.generate_signals(SYMBOLS, up_bars, make_state(0, "CRASH", prob=0.99), is_flickering=True)
    for s in sigs:
        assert s.position_size_pct == pytest.approx(0.30) and s.leverage == 1.0


def test_transition_multiplier_and_combination(orch: StrategyOrchestrator, up_bars: dict[str, pd.DataFrame]) -> None:
    pending = orch.generate_signals(SYMBOLS, up_bars, make_state(2, "BULL", prob=0.9, in_transition=True))
    assert all(s.position_size_pct == pytest.approx(0.95 * 0.75) and s.leverage == 1.0 for s in pending)
    both = orch.generate_signals(SYMBOLS, up_bars, make_state(2, "BULL", prob=0.4, in_transition=True))
    assert all(s.position_size_pct == pytest.approx(0.95 * 0.5 * 0.75) for s in both)
    assert orch.size_multiplier(make_state(2, "BULL", prob=0.4, in_transition=True), True) == pytest.approx(0.375)


def test_rebalance_threshold(
    orch: StrategyOrchestrator, up_bars: dict[str, pd.DataFrame], caplog: pytest.LogCaptureFixture
) -> None:
    state = make_state(2, "BULL")
    target = 0.95 * 1.25
    assert orch.generate_signals(SYMBOLS, up_bars, state, current_allocation=target - 0.05) == []
    assert orch.generate_signals(SYMBOLS, up_bars, state, current_allocation=target - 0.10) == []
    assert len(orch.generate_signals(SYMBOLS, up_bars, state, current_allocation=target - 0.11)) == 3
    assert len(orch.generate_signals(SYMBOLS, up_bars, state, current_allocation=0.0)) == 3
    assert len(orch.generate_signals(SYMBOLS, up_bars, state)) == 3
    assert orch.needs_rebalance(0.60, 0.95) and not orch.needs_rebalance(0.90, 0.95)


def test_aliases_and_label_map() -> None:
    assert CrashDefensiveStrategy is HighVolDefensiveStrategy
    assert BearTrendStrategy is HighVolDefensiveStrategy
    assert MeanReversionStrategy is MidVolCautiousStrategy
    assert BullTrendStrategy is LowVolBullStrategy
    assert EuphoriaCautiousStrategy is LowVolBullStrategy
    all_labels = {label for labels in REGIME_LABELS.values() for label in labels}
    assert set(LABEL_TO_STRATEGY) == all_labels
    assert all(issubclass(cls, BaseStrategy) for cls in LABEL_TO_STRATEGY.values())


def test_base_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseStrategy()  # type: ignore[abstract]


def test_config_from_settings_yaml() -> None:
    settings = yaml.safe_load(SETTINGS.read_text())
    cfg = StrategyConfig.from_dict(settings["strategy"], settings["hmm"])
    assert cfg.low_vol_allocation == 0.95 and cfg.low_vol_leverage == 1.25
    assert cfg.mid_vol_allocation_no_trend == 0.60 and cfg.high_vol_allocation == 0.60
    assert cfg.rebalance_threshold == 0.10 and cfg.uncertainty_size_mult == 0.50
    assert cfg.min_confidence == 0.55 and cfg.transition_size_mult == 0.75
    assert cfg.ema_window == 50 and cfg.atr_window == 14
    with pytest.raises(ValueError):
        StrategyConfig.from_dict({"nope": 1})


def test_from_engine_integration(
    trained_engine: HMMEngine,
    features: pd.DataFrame,
    synthetic: tuple[pd.DataFrame, pd.Series],
    caplog: pytest.LogCaptureFixture,
) -> None:
    orch = StrategyOrchestrator.from_engine(StrategyConfig(), trained_engine)
    assert orch.volatility_ranking == trained_engine.get_volatility_ranking()
    for rid, info in orch.regime_infos.items():
        assert info.recommended_strategy_type == orch.get_strategy(rid).name

    state = trained_engine.run_filter(features)[-1]
    bars = synthetic[0].loc[: features.index[-1]]
    caplog.set_level(logging.INFO)
    sigs = orch.generate_signals(["SYN"], {"SYN": bars}, state, trained_engine.is_flickering())
    assert len(sigs) == 1
    s = sigs[0]
    assert s.direction is Direction.LONG and s.regime_id == state.state_id
    assert s.strategy_name == orch.get_strategy(state.state_id).name
    assert s.leverage in (1.0, 1.25) and 0 < s.stop_loss < s.entry_price
