"""Look-ahead bias tests.

Core invariant: any output at bar t must be identical whether computed on data
truncated at t or on a longer history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMEngine
from data.feature_engineering import build_features, compute_raw_features, rolling_zscore


def test_no_look_ahead_bias(trained_engine: HMMEngine, features: pd.DataFrame) -> None:
    """Regime at bar 399 must not change when bars 400..499 are added."""
    assert len(features) >= 500
    short = features.iloc[0:400]
    long = features.iloc[0:500]

    regime_short = trained_engine.predict_regime_filtered(short)[-1]
    regime_long = trained_engine.predict_regime_filtered(long)[400 - 1]
    assert regime_short == regime_long, "LOOK-AHEAD BIAS DETECTED"

    proba_short = trained_engine.predict_regime_proba(short)
    proba_long = trained_engine.predict_regime_proba(long)
    np.testing.assert_allclose(
        proba_short.to_numpy(), proba_long.iloc[:400].to_numpy(), rtol=0, atol=1e-12,
        err_msg="LOOK-AHEAD BIAS DETECTED",
    )


@pytest.mark.parametrize("cut", [520, 650, 800])
def test_filtered_regimes_prefix_invariant(trained_engine: HMMEngine, features: pd.DataFrame, cut: int) -> None:
    full = trained_engine.predict_regime_filtered(features)
    prefix = trained_engine.predict_regime_filtered(features.iloc[:cut])
    np.testing.assert_array_equal(prefix, full[:cut], err_msg="LOOK-AHEAD BIAS DETECTED")


def test_future_perturbation_does_not_change_past(trained_engine: HMMEngine, features: pd.DataFrame) -> None:
    perturbed = features.copy()
    perturbed.iloc[400:] = perturbed.iloc[400:] * -3.0 + 5.0
    a = trained_engine.predict_regime_proba(features).iloc[:400]
    b = trained_engine.predict_regime_proba(perturbed).iloc[:400]
    pd.testing.assert_frame_equal(a, b)


def test_stability_filtered_states_prefix_invariant(trained_engine: HMMEngine, features: pd.DataFrame) -> None:
    full = trained_engine.run_filter(features)
    prefix = trained_engine.run_filter(features.iloc[:450])
    for s_full, s_pre in zip(full[:450], prefix):
        assert (s_full.state_id, s_full.is_confirmed, s_full.consecutive_bars) == (
            s_pre.state_id, s_pre.is_confirmed, s_pre.consecutive_bars,
        )
        np.testing.assert_allclose(s_full.state_probabilities, s_pre.state_probabilities, atol=1e-12)


def test_inference_never_uses_viterbi_or_smoothing(
    trained_engine: HMMEngine, features: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("smoothed/Viterbi inference called")

    model = trained_engine.model
    for name in ("predict", "predict_proba", "decode", "score_samples", "score"):
        monkeypatch.setattr(model, name, forbidden)
    trained_engine.predict_regime_proba(features)
    trained_engine.predict_regime_filtered(features)
    trained_engine.run_filter(features.iloc[:50])


@pytest.mark.parametrize("cut", [500, 700, 1000])
def test_feature_engineering_is_causal(synthetic: tuple[pd.DataFrame, pd.Series], cut: int) -> None:
    bars = synthetic[0]
    full = build_features(bars, dropna=False)
    prefix = build_features(bars.iloc[:cut], dropna=False)
    pd.testing.assert_frame_equal(prefix, full.iloc[:cut], check_exact=False, rtol=1e-9, atol=1e-12)


def test_feature_future_perturbation_does_not_change_past(synthetic: tuple[pd.DataFrame, pd.Series]) -> None:
    bars = synthetic[0]
    shocked = bars.copy()
    shocked.iloc[900:, :4] *= 0.5
    shocked.iloc[900:, 4] *= 10.0
    a = build_features(bars, dropna=False).iloc[:900]
    b = build_features(shocked, dropna=False).iloc[:900]
    pd.testing.assert_frame_equal(a, b)
    assert not build_features(bars, dropna=False).iloc[900:].equals(build_features(shocked, dropna=False).iloc[900:])


def test_raw_features_complete_and_finite_after_warmup(synthetic: tuple[pd.DataFrame, pd.Series]) -> None:
    raw = compute_raw_features(synthetic[0])
    assert raw.shape[1] == 14
    assert np.isfinite(raw.iloc[260:].to_numpy()).all()
    feats = build_features(synthetic[0])
    assert list(feats.columns) == list(raw.columns)
    assert np.isfinite(feats.to_numpy()).all()


def test_rolling_zscore_backward_only() -> None:
    s = pd.Series(np.arange(10, dtype=float) ** 2)
    z = rolling_zscore(s, window=4)
    assert z.iloc[:3].isna().all()
    window = s.iloc[2:6]
    assert z.iloc[5] == pytest.approx((s.iloc[5] - window.mean()) / window.std())
