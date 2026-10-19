"""Tests for backtest.stress_test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.stress_test import StressScenario, StressTester, kill_events, summarize, window_drawdowns
from core.hmm_engine import HMMEngine
from data.feature_engineering import atr


@pytest.fixture()
def tester(wf_backtester) -> StressTester:
    return StressTester(wf_backtester, seed=11, n_sims=2)


def test_inject_crash_is_overnight_level_shift(tester: StressTester, synthetic) -> None:
    bars = synthetic[0]
    at = bars.index[500]
    out = tester.inject_crash(bars, -0.10, at)
    pd.testing.assert_frame_equal(out.iloc[:500], bars.iloc[:500])
    cols = ["open", "high", "low", "close"]
    np.testing.assert_allclose(out[cols].iloc[500:], bars[cols].iloc[500:] * 0.9)
    pd.testing.assert_series_equal(out["volume"], bars["volume"])
    gap = out["open"].iloc[500] / out["close"].iloc[499] - 1
    assert gap == pytest.approx((bars["open"].iloc[500] / bars["close"].iloc[499]) * 0.9 - 1)


def test_inject_crash_spread_over_duration(tester: StressTester, synthetic) -> None:
    bars = synthetic[0]
    out = tester.inject_crash(bars, -0.20, bars.index[100], duration_bars=4)
    ratio = (out["close"] / bars["close"]).iloc[99:106].to_numpy()
    np.testing.assert_allclose(ratio, [1, 0.8 ** 0.25, 0.8 ** 0.5, 0.8 ** 0.75, 0.8, 0.8, 0.8])


def test_random_crashes_count_and_range(tester: StressTester, synthetic, wf_result) -> None:
    index = wf_result.equity_curve.index
    shocked, events = tester.random_crashes({"SPY": synthetic[0]}, index)
    assert len(events) == 10 and events["timestamp"].isin(index).all()
    assert events["shock_pct"].between(-0.15, -0.05).all()
    total = np.prod(1 + events["shock_pct"])
    assert shocked["SPY"]["close"].iloc[-1] == pytest.approx(synthetic[0]["close"].iloc[-1] * total)


def test_random_gaps_scale_with_atr(tester: StressTester, synthetic, wf_result) -> None:
    bars = synthetic[0]
    shocked, events = tester.random_gaps({"SPY": bars}, wf_result.equity_curve.index)
    assert len(events) == 10 and events["atr_mult"].between(2, 5).all()
    first = events.iloc[0]
    pos = bars.index.get_loc(first["timestamp"])
    a = atr(bars["high"], bars["low"], bars["close"], 14).iloc[pos - 1]
    assert first["shock_pct"] == pytest.approx(-first["atr_mult"] * a / bars["close"].iloc[pos - 1])
    assert (events["shock_pct"] < 0).all()


def test_monte_carlo_reuses_frozen_models(
    tester: StressTester, synthetic, wf_result, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(HMMEngine, "fit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrained")))
    out = tester.run_monte_carlo("crash", {"SPY": synthetic[0]}, wf_result)
    assert len(out.sims) == 2 and len(out.events) == 2
    assert {"total_return", "max_drawdown", "worst_window_drawdown", "kill_events"} <= set(out.sims.columns)
    gap = tester.run_monte_carlo("gap", {"SPY": synthetic[0]}, wf_result, n_sims=1)
    assert len(gap.sims) == 1


def test_crashes_hurt_relative_to_baseline(tester: StressTester, synthetic, wf_result) -> None:
    out = tester.run_monte_carlo("crash", {"SPY": synthetic[0]}, wf_result, n_sims=3)
    base = wf_result.equity_curve.iloc[-1] / 100_000 - 1
    assert out.sims["total_return"].mean() < base


def test_misclassification_contained_by_risk(tester: StressTester, synthetic, wf_result, wf_backtester) -> None:
    out = tester.run_misclassification({"SPY": synthetic[0]}, wf_result, n_sims=2)
    s = out.sims
    assert {"max_drawdown", "max_drawdown_no_risk", "worst_window_drawdown_no_risk"} <= set(s.columns)
    limit = wf_backtester.risk_config.max_dd_from_peak
    assert (s["worst_window_drawdown"] <= limit + 0.05).all(), "kill switch bounds intra-window drawdown"


def test_shuffled_regimes_permute_within_window(tester: StressTester, wf_result) -> None:
    transform = tester.shuffled_regimes(wf_result)
    hist = wf_result.regime_history
    engine = wf_result.engines[0]
    from core.hmm_engine import RegimeState

    w0 = hist[hist["window"] == 0]
    new_ids = []
    for ts in w0.index:
        st = RegimeState("x", 0, 0.9, np.ones(engine.n_states) / engine.n_states, ts, True, 5)
        new = transform(0, ts, st, engine)
        assert new.probability == 0.9 and new.label == engine.labels[new.state_id]
        new_ids.append(new.state_id)
    assert sorted(new_ids) == sorted(w0["regime_id"].tolist())


def test_window_drawdowns_and_kill_events(wf_result) -> None:
    dd = window_drawdowns(wf_result)
    assert len(dd) == len(wf_result.folds) and (dd >= 0).all()
    assert kill_events(wf_result) >= 0


def test_run_scenario_and_summary(tester: StressTester, synthetic, wf_result) -> None:
    res = tester.run_scenario(StressScenario("crash20", -0.20), {"SPY": synthetic[0]}, wf_result)
    assert res.equity_curve.index.equals(wf_result.equity_curve.index)
    outcomes = {
        "crash": tester.run_monte_carlo("crash", {"SPY": synthetic[0]}, wf_result, n_sims=1),
        "misclassification": tester.run_misclassification({"SPY": synthetic[0]}, wf_result, n_sims=1),
    }
    table = summarize(outcomes, wf_result)
    assert list(table.index) == ["baseline", "crash", "misclassification", "misclassification (no risk)"]
    assert [s.name for s in tester.default_scenarios()] == ["crash", "gap", "misclassification"]
