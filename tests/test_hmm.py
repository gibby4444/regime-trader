"""Tests for core.hmm_engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import fast_config

from core.hmm_engine import (
    REGIME_LABELS,
    HMMConfig,
    HMMEngine,
    RegimeStabilityFilter,
    bic_score,
    n_free_parameters,
    regime_labels,
    volatility_bucket,
)

SETTINGS = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def test_regime_labels_for_each_n() -> None:
    assert regime_labels(3) == ("BEAR", "NEUTRAL", "BULL")
    assert regime_labels(4) == ("CRASH", "BEAR", "BULL", "EUPHORIA")
    assert regime_labels(7)[3] == "NEUTRAL"
    for n, labels in REGIME_LABELS.items():
        assert len(labels) == n
    with pytest.raises(ValueError):
        regime_labels(2)


def test_n_free_parameters_full_covariance() -> None:
    # K=3, d=2: start 2 + trans 6 + means 6 + covs 3*3 = 23
    assert n_free_parameters(3, 2, "full") == 23
    assert n_free_parameters(4, 14, "full") == 3 + 12 + 56 + 4 * 105


def test_bic_formula() -> None:
    assert bic_score(-100.0, 10, 100) == pytest.approx(200.0 + 10 * np.log(100))


def test_volatility_bucket() -> None:
    assert [volatility_bucket(i, 3) for i in range(3)] == ["calm", "moderate", "turbulent"]
    assert volatility_bucket(0, 7) == "calm" and volatility_bucket(6, 7) == "turbulent"


def test_fit_requires_min_train_bars(features: pd.DataFrame) -> None:
    engine = HMMEngine(fast_config())
    with pytest.raises(ValueError, match="at least 504"):
        engine.fit(features.iloc[:503])


def test_unfitted_engine_raises(features: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError):
        HMMEngine().predict_regime_proba(features)


def test_model_selection_logs_all_candidates_and_picks_min_bic(
    features: pd.DataFrame, returns: pd.Series, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="regime_trader.hmm")
    engine = HMMEngine(fast_config(n_init=2)).fit(features, returns)

    assert [c.n_states for c in engine.candidates] == [3, 4]
    best = min(engine.candidates, key=lambda c: c.bic)
    assert engine.n_states == best.n_states
    for c in engine.candidates:
        expected = bic_score(c.log_likelihood, n_free_parameters(c.n_states, features.shape[1]), len(features))
        assert c.bic == pytest.approx(expected)

    text = caplog.text
    assert "n_states=3" in text and "n_states=4" in text
    assert f"selected n_states={best.n_states}" in text
    assert "converged=" in text and "iter=" in text


def test_n_init_keeps_best_likelihood(features: pd.DataFrame) -> None:
    single = [HMMEngine(fast_config(n_candidates=[3], n_init=1, random_state=s)).fit(features) for s in (0, 1, 2)]
    multi = HMMEngine(fast_config(n_candidates=[3], n_init=3, random_state=0)).fit(features)
    best_single = max(e.candidates[0].log_likelihood for e in single)
    assert multi.candidates[0].log_likelihood == pytest.approx(best_single)


def test_regimes_sorted_by_expected_return(trained_engine: HMMEngine) -> None:
    rets = [ri.expected_return for ri in trained_engine.regime_info]
    assert rets == sorted(rets)
    assert [ri.regime_name for ri in trained_engine.regime_info] == list(regime_labels(trained_engine.n_states))


def test_volatility_exposed_for_strategy_layer(trained_engine: HMMEngine) -> None:
    vols = trained_engine.get_regime_volatilities()
    assert set(vols) == set(range(trained_engine.n_states))
    assert all(v > 0 for v in vols.values())
    ranking = trained_engine.get_volatility_ranking()
    assert [vols[i] for i in ranking] == sorted(vols.values())
    calmest, wildest = trained_engine.regime_info[ranking[0]], trained_engine.regime_info[ranking[-1]]
    assert calmest.max_leverage_allowed >= wildest.max_leverage_allowed
    assert calmest.max_position_size_pct >= wildest.max_position_size_pct


def test_recovers_true_volatility_structure(
    trained_engine: HMMEngine, features: pd.DataFrame, synthetic: tuple[pd.DataFrame, pd.Series]
) -> None:
    true_vol = pd.Series([0.006, 0.012, 0.028])[synthetic[1].reindex(features.index).to_numpy()].to_numpy()
    probs = trained_engine.predict_regime_proba(features).to_numpy()
    est_vol = probs @ np.array([ri.expected_volatility for ri in trained_engine.regime_info])
    assert np.corrcoef(true_vol, est_vol)[0, 1] > 0.5


def test_transition_matrix(trained_engine: HMMEngine) -> None:
    tm = trained_engine.get_transition_matrix()
    assert list(tm.index) == list(trained_engine.labels) == list(tm.columns)
    np.testing.assert_allclose(tm.sum(axis=1), 1.0)


def test_forward_loglik_matches_hmmlearn_score(trained_engine: HMMEngine, features: pd.DataFrame) -> None:
    assert trained_engine.model is not None
    assert trained_engine.forward_log_likelihood(features) == pytest.approx(
        trained_engine.model.score(features.to_numpy()), rel=1e-8
    )


def test_filtered_probabilities_are_distributions(trained_engine: HMMEngine, features: pd.DataFrame) -> None:
    probs = trained_engine.predict_regime_proba(features)
    assert probs.index.equals(features.index)
    assert (probs.to_numpy() >= 0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_array_equal(trained_engine.predict_regime_filtered(features), probs.to_numpy().argmax(axis=1))


def test_zero_likelihood_bar_keeps_prior_instead_of_nan(
    trained_engine: HMMEngine, features: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = features.iloc[:30]
    real = HMMEngine.emission_logprob

    def impossible_bar_10(self: HMMEngine, arr: np.ndarray) -> np.ndarray:
        out = real(self, arr)
        if len(out) > 10:
            out[10] = -np.inf
        return out

    baseline = trained_engine.predict_regime_proba(x).to_numpy()
    monkeypatch.setattr(HMMEngine, "emission_logprob", impossible_bar_10)
    probs = trained_engine.predict_regime_proba(x).to_numpy()
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(probs[:10], baseline[:10])
    expected_prior = baseline[9] @ trained_engine.transmat
    np.testing.assert_allclose(probs[10], expected_prior, atol=1e-12)


def test_incremental_update_matches_batch(trained_engine: HMMEngine, features: pd.DataFrame) -> None:
    batch = trained_engine.predict_regime_proba(features).to_numpy()
    states = trained_engine.run_filter(features)
    inc = np.vstack([s.state_probabilities for s in states])
    np.testing.assert_allclose(inc, batch, atol=1e-10)
    assert states[-1].timestamp == features.index[-1]
    assert trained_engine.get_current_state() == states[-1]


def test_stability_filter_confirms_after_n_bars() -> None:
    f = RegimeStabilityFilter(stability_bars=3)
    assert f.update(0) == (False, False)
    f.update(0)
    f.update(0)
    assert f.is_confirmed and f.consecutive_bars == 3
    assert f.update(1) == (True, False)
    assert f.confirmed_regime == 0 and f.in_transition
    assert f.update(1) == (False, False)
    assert f.confirmed_regime == 0
    assert f.update(1) == (False, True)
    assert f.confirmed_regime == 1 and not f.in_transition and f.consecutive_bars == 3


def test_stability_filter_aborted_transition_keeps_regime() -> None:
    f = RegimeStabilityFilter(stability_bars=3)
    for r in [0, 0, 0, 1, 1, 0]:
        f.update(r)
    assert f.confirmed_regime == 0 and not f.in_transition and not f.last_change_confirmed


def test_flicker_detection() -> None:
    f = RegimeStabilityFilter(stability_bars=3, flicker_window=20, flicker_threshold=4)
    for r in [0] * 10 + [1, 0, 1, 0]:
        f.update(r)
    assert f.flicker_rate == 4 and not f.is_flickering
    f.update(1)
    assert f.flicker_rate == 5 and f.is_flickering
    for _ in range(20):
        f.update(1)
    assert f.flicker_rate == 0 and not f.is_flickering


def _drive(engine: HMMEngine, raw: list[int]) -> list[bool]:
    engine.reset_filter()
    out = []
    for r in raw:
        engine.stability.update(r)
        out.append(engine.detect_regime_change())
    return out


def test_detect_regime_change_and_size_multiplier(trained_engine: HMMEngine) -> None:
    changes = _drive(trained_engine, [0, 0, 0, 1])
    assert changes == [False, False, False, False]
    assert trained_engine.get_size_multiplier() == pytest.approx(0.75)
    changes = _drive(trained_engine, [0, 0, 0, 1, 1, 1, 1])
    assert changes == [False, False, False, False, False, True, False]
    assert trained_engine.get_size_multiplier() == 1.0
    assert trained_engine.get_regime_stability() == 4


def test_update_logs_change_warning_and_confirmation_info(
    trained_engine: HMMEngine, features: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="regime_trader.hmm")
    states = trained_engine.run_filter(features, log_events=True)
    raw = [s.raw_state_id for s in states]
    if len(set(raw)) < 2:
        pytest.skip("synthetic path produced a single regime")
    levels = {(r.levelno, r.getMessage().split(" at ")[0]) for r in caplog.records}
    assert (logging.WARNING, "Regime change detected") in levels
    assert (logging.INFO, "Regime change confirmed") in levels


def test_uncertainty_mode(trained_engine: HMMEngine, features: pd.DataFrame) -> None:
    trained_engine.reset_filter()
    assert trained_engine.is_uncertain()
    state = trained_engine.run_filter(features)[-1]
    expected = trained_engine.is_flickering() or state.probability < trained_engine.config.min_confidence
    assert trained_engine.is_uncertain() == expected


def test_retrain_interval(features: pd.DataFrame, returns: pd.Series) -> None:
    engine = HMMEngine(fast_config(n_candidates=[3], n_init=1, retrain_interval_bars=5))
    train = features.iloc[:600]
    engine.retrain(train, returns.iloc[:600])
    assert not engine.should_retrain()
    for ts, row in features.iloc[600:604].iterrows():
        engine.update(row, ts)
    assert not engine.maybe_retrain(features.iloc[:604], returns.iloc[:604])
    engine.update(features.iloc[604], features.index[604])
    assert engine.maybe_retrain(features.iloc[:605], returns.iloc[:605])
    assert engine.metadata["n_samples"] == 605
    assert not engine.should_retrain()


def test_save_load_roundtrip(trained_engine: HMMEngine, features: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "hmm.pkl"
    trained_engine.save(path)
    meta = json.loads((tmp_path / "hmm.pkl.json").read_text())
    for key in ("n_regimes", "bic", "training_date", "labels"):
        assert key in meta
    loaded = HMMEngine.load(path)
    assert loaded.labels == trained_engine.labels
    assert loaded.regime_info == trained_engine.regime_info
    pd.testing.assert_frame_equal(loaded.predict_regime_proba(features), trained_engine.predict_regime_proba(features))


def test_config_from_settings_yaml() -> None:
    settings = yaml.safe_load(SETTINGS.read_text())
    cfg = HMMConfig.from_dict(settings["hmm"])
    assert cfg.min_train_bars == 504
    assert cfg.transition_size_mult == 0.75
    assert cfg.n_candidates == [3, 4, 5, 6, 7] and cfg.n_init == 10
    assert cfg.covariance_type == "full"
    with pytest.raises(ValueError):
        HMMConfig.from_dict({"bogus": 1})
