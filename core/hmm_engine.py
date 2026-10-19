"""Hidden Markov Model regime detection engine.

The HMM is a *volatility classifier*: it groups bars into regimes with distinct
feature distributions (calm / moderate / turbulent). It does not forecast
direction. Regimes are labelled by mean return purely for readability; the
strategy layer should rank regimes by ``expected_volatility``.

Inference is causal only. ``model.predict`` / Viterbi / ``predict_proba``
(forward-backward smoothing) are never used; a manual log-space forward filter
yields P(state_t | x_1..x_t).
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

logger = logging.getLogger("regime_trader.hmm")

FeatureInput = pd.DataFrame | np.ndarray

REGIME_LABELS: dict[int, tuple[str, ...]] = {
    3: ("BEAR", "NEUTRAL", "BULL"),
    4: ("CRASH", "BEAR", "BULL", "EUPHORIA"),
    5: ("CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"),
    6: ("CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"),
    7: ("CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"),
}

VOL_BUCKETS: tuple[str, ...] = ("calm", "moderate", "turbulent")

# Default per-bucket guidance attached to RegimeInfo; the strategy/risk layers own
# the final numbers via settings.yaml.
VOL_BUCKET_PROFILES: dict[str, dict[str, Any]] = {
    "calm": {
        "recommended_strategy_type": "LowVolBullStrategy",
        "max_leverage_allowed": 1.25,
        "max_position_size_pct": 0.15,
        "min_confidence_to_act": 0.55,
    },
    "moderate": {
        "recommended_strategy_type": "MidVolCautiousStrategy",
        "max_leverage_allowed": 1.0,
        "max_position_size_pct": 0.15,
        "min_confidence_to_act": 0.55,
    },
    "turbulent": {
        "recommended_strategy_type": "HighVolDefensiveStrategy",
        "max_leverage_allowed": 1.0,
        "max_position_size_pct": 0.10,
        "min_confidence_to_act": 0.55,
    },
}


def regime_labels(n_states: int) -> tuple[str, ...]:
    """Labels for ``n_states`` regimes ordered by ascending mean return."""
    if n_states not in REGIME_LABELS:
        raise ValueError(f"No label set for n_states={n_states}; supported: {sorted(REGIME_LABELS)}")
    return REGIME_LABELS[n_states]


def n_free_parameters(n_states: int, n_features: int, covariance_type: str = "full") -> int:
    """Free parameters of a Gaussian HMM (start probs, transitions, means, covariances)."""
    k, d = n_states, n_features
    cov = {
        "full": k * d * (d + 1) // 2,
        "diag": k * d,
        "spherical": k,
        "tied": d * (d + 1) // 2,
    }
    if covariance_type not in cov:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")
    return (k - 1) + k * (k - 1) + k * d + cov[covariance_type]


def bic_score(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """BIC = -2 * logL + n_params * ln(n_samples)."""
    return float(-2.0 * log_likelihood + n_params * np.log(n_samples))


def volatility_bucket(
    vol_rank: int, n_states: int, low_cutoff: float = 0.33, high_cutoff: float = 0.67
) -> str:
    """Map a 0-based volatility rank to calm/moderate/turbulent.

    position = rank / (n - 1); <= low_cutoff -> calm, >= high_cutoff -> turbulent.
    Same rule as ``core.regime_strategies.vol_regime_for_rank``.
    """
    position = vol_rank / (n_states - 1) if n_states > 1 else 0.0
    if position <= low_cutoff:
        return "calm"
    if position >= high_cutoff:
        return "turbulent"
    return "moderate"


@dataclass(frozen=True)
class RegimeInfo:
    """Static description of one fitted regime."""

    regime_id: int
    regime_name: str
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass(frozen=True)
class RegimeState:
    """Stability-filtered regime estimate for one bar.

    ``label``/``state_id``/``probability`` describe the *confirmed* regime, which
    is held through transitions until a new regime persists ``stability_bars``.
    """

    label: str
    state_id: int
    probability: float
    state_probabilities: np.ndarray
    timestamp: pd.Timestamp | None
    is_confirmed: bool
    consecutive_bars: int
    raw_state_id: int = -1
    in_transition: bool = False
    flicker_rate: int = 0


@dataclass
class CandidateResult:
    """Best-of-``n_init`` fit for one candidate state count."""

    n_states: int
    log_likelihood: float
    bic: float
    n_params: int
    converged: bool
    n_iter: int


@dataclass
class HMMConfig:
    """HMM parameters mirroring the ``hmm`` section of settings.yaml."""

    n_candidates: list[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])
    n_init: int = 10
    covariance_type: str = "full"
    n_iter: int = 100
    tol: float = 1e-4
    min_covar: float = 1e-3
    random_state: int | None = 42
    min_train_bars: int = 504
    retrain_interval_bars: int = 21
    stability_bars: int = 3
    flicker_window: int = 20
    flicker_threshold: int = 4
    min_confidence: float = 0.55
    transition_size_mult: float = 0.75

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> HMMConfig:
        """Build a config from the parsed ``hmm`` settings section."""
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown hmm config keys: {sorted(unknown)}")
        return cls(**cfg)


class RegimeStabilityFilter:
    """Debounces raw per-bar regimes and tracks flicker.

    A different raw regime becomes *pending*; it is confirmed only after
    ``stability_bars`` consecutive bars. Until then the previous confirmed regime
    is kept. Flicker rate = raw regime changes over the last ``flicker_window`` bars.
    """

    def __init__(self, stability_bars: int = 3, flicker_window: int = 20, flicker_threshold: int = 4) -> None:
        self.stability_bars = stability_bars
        self.flicker_window = flicker_window
        self.flicker_threshold = flicker_threshold
        self.reset()

    def reset(self) -> None:
        """Clear all state."""
        self.confirmed_regime: int | None = None
        self.pending_regime: int | None = None
        self.pending_count: int = 0
        self.consecutive_bars: int = 0
        self.last_change_confirmed: bool = False
        self._raw_history: deque[int] = deque(maxlen=self.flicker_window + 1)

    def update(self, raw_regime: int) -> tuple[bool, bool]:
        """Feed one raw regime; return ``(change_detected, change_confirmed)``."""
        self._raw_history.append(raw_regime)
        detected = confirmed = False
        if self.confirmed_regime is None:
            self.confirmed_regime = raw_regime
            self.consecutive_bars = 1
        elif raw_regime == self.confirmed_regime:
            self.pending_regime = None
            self.pending_count = 0
            self.consecutive_bars += 1
        else:
            if raw_regime == self.pending_regime:
                self.pending_count += 1
            else:
                self.pending_regime = raw_regime
                self.pending_count = 1
                detected = True
            if self.pending_count >= self.stability_bars:
                self.confirmed_regime = raw_regime
                self.consecutive_bars = self.pending_count
                self.pending_regime = None
                self.pending_count = 0
                confirmed = True
        self.last_change_confirmed = confirmed
        return detected, confirmed

    @property
    def in_transition(self) -> bool:
        return self.pending_regime is not None

    @property
    def is_confirmed(self) -> bool:
        return not self.in_transition and self.consecutive_bars >= self.stability_bars

    @property
    def flicker_rate(self) -> int:
        h = list(self._raw_history)
        return sum(1 for a, b in zip(h, h[1:]) if a != b)

    @property
    def is_flickering(self) -> bool:
        return self.flicker_rate > self.flicker_threshold


def _as_array(features: FeatureInput) -> np.ndarray:
    x = np.asarray(features.values if isinstance(features, pd.DataFrame) else features, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("features contain NaN or inf")
    return x


class HMMEngine:
    """Trains a Gaussian HMM with BIC model selection and runs causal regime inference."""

    def __init__(self, config: HMMConfig | None = None) -> None:
        self.config = config or HMMConfig()
        self.model: GaussianHMM | None = None
        self.n_states: int | None = None
        self.labels: tuple[str, ...] = ()
        self.regime_info: list[RegimeInfo] = []
        self.candidates: list[CandidateResult] = []
        self.metadata: dict[str, Any] = {}
        self.feature_names: list[str] | None = None
        self.startprob: np.ndarray | None = None
        self.transmat: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.covars: np.ndarray | None = None
        self.stability = RegimeStabilityFilter(
            self.config.stability_bars, self.config.flicker_window, self.config.flicker_threshold
        )
        self._log_alpha: np.ndarray | None = None
        self._last_state: RegimeState | None = None
        self._bars_since_train: int = 0

    # ------------------------------------------------------------------ training

    def _fit_candidate(self, x: np.ndarray, n_states: int) -> tuple[GaussianHMM | None, CandidateResult | None]:
        cfg = self.config
        best_model: GaussianHMM | None = None
        best_ll = -np.inf
        for i in range(cfg.n_init):
            seed = None if cfg.random_state is None else cfg.random_state + i
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=cfg.covariance_type,
                n_iter=cfg.n_iter,
                tol=cfg.tol,
                min_covar=cfg.min_covar,
                random_state=seed,
            )
            try:
                model.fit(x)
                ll = float(model.score(x))
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.debug("n_states=%d init=%d failed: %s", n_states, i, exc)
                continue
            logger.debug(
                "n_states=%d init=%d logL=%.3f converged=%s iter=%d",
                n_states, i, ll, model.monitor_.converged, model.monitor_.iter,
            )
            if np.isfinite(ll) and ll > best_ll:
                best_ll, best_model = ll, model
        if best_model is None:
            return None, None
        n_params = n_free_parameters(n_states, x.shape[1], cfg.covariance_type)
        result = CandidateResult(
            n_states=n_states,
            log_likelihood=best_ll,
            bic=bic_score(best_ll, n_params, x.shape[0]),
            n_params=n_params,
            converged=bool(best_model.monitor_.converged),
            n_iter=int(best_model.monitor_.iter),
        )
        return best_model, result

    def fit(
        self,
        features: FeatureInput,
        returns: pd.Series | np.ndarray | None = None,
        training_date: datetime | pd.Timestamp | None = None,
    ) -> HMMEngine:
        """Fit every candidate state count, keep the lowest-BIC model, and label regimes.

        Args:
            features: Standardized feature matrix (rows = bars).
            returns: Raw 1-bar returns aligned with ``features`` used for per-regime
                expected return/volatility. Defaults to the ``ret_1`` column (or
                the first column) of ``features``.
            training_date: Timestamp recorded in metadata; defaults to now (UTC).

        Raises:
            ValueError: If fewer than ``min_train_bars`` rows are supplied.
            RuntimeError: If no candidate could be fit.
        """
        x = _as_array(features)
        if x.shape[0] < self.config.min_train_bars:
            raise ValueError(
                f"Need at least {self.config.min_train_bars} bars to train, got {x.shape[0]}"
            )
        if isinstance(features, pd.DataFrame):
            self.feature_names = [str(c) for c in features.columns]
        r = self._resolve_returns(features, x, returns)

        fitted: dict[int, GaussianHMM] = {}
        self.candidates = []
        for n in self.config.n_candidates:
            model, result = self._fit_candidate(x, n)
            if model is None or result is None:
                logger.warning("HMM candidate n_states=%d failed for all %d inits", n, self.config.n_init)
                continue
            fitted[n] = model
            self.candidates.append(result)
            logger.info(
                "HMM candidate n_states=%d logL=%.3f n_params=%d BIC=%.3f converged=%s iter=%d",
                n, result.log_likelihood, result.n_params, result.bic, result.converged, result.n_iter,
            )
        if not self.candidates:
            raise RuntimeError("No HMM candidate could be fit")

        best = min(self.candidates, key=lambda c: c.bic)
        logger.info(
            "HMM model selection: BIC scores %s -> selected n_states=%d (BIC=%.3f)",
            {c.n_states: round(c.bic, 3) for c in self.candidates}, best.n_states, best.bic,
        )
        self._install_model(fitted[best.n_states], x, r)
        self.metadata = {
            "n_regimes": best.n_states,
            "bic": best.bic,
            "log_likelihood": best.log_likelihood,
            "converged": best.converged,
            "n_iter": best.n_iter,
            "training_date": pd.Timestamp(training_date or datetime.now(timezone.utc)).isoformat(),
            "labels": list(self.labels),
            "n_samples": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "feature_names": self.feature_names,
            "candidate_bic": {c.n_states: c.bic for c in self.candidates},
        }
        logger.info(
            "HMM trained: n_regimes=%d logL=%.3f BIC=%.3f converged=%s iter=%d samples=%d",
            best.n_states, best.log_likelihood, best.bic, best.converged, best.n_iter, x.shape[0],
        )
        self.reset_filter()
        self._bars_since_train = 0
        return self

    @staticmethod
    def _resolve_returns(
        features: FeatureInput, x: np.ndarray, returns: pd.Series | np.ndarray | None
    ) -> np.ndarray:
        if returns is None:
            if isinstance(features, pd.DataFrame) and "ret_1" in features.columns:
                return features["ret_1"].to_numpy(dtype=float)
            return x[:, 0]
        if isinstance(returns, pd.Series) and isinstance(features, pd.DataFrame):
            returns = returns.reindex(features.index)
        r = np.asarray(returns, dtype=float)
        if r.shape != (x.shape[0],) or not np.all(np.isfinite(r)):
            raise ValueError("returns must be finite and aligned 1:1 with features")
        return r

    def _install_model(self, model: GaussianHMM, x: np.ndarray, returns: np.ndarray) -> None:
        """Store model params re-ordered by ascending expected return and build RegimeInfo."""
        self.model = model
        self.startprob = np.asarray(model.startprob_, dtype=float)
        self.transmat = np.asarray(model.transmat_, dtype=float)
        self.means = np.asarray(model.means_, dtype=float)
        self.covars = np.asarray(model.covars_, dtype=float)
        k = self.startprob.shape[0]

        # Filtered (causal) responsibilities weight the per-regime return stats.
        w = self._forward(x)[0]
        w = np.exp(w)
        mass = w.sum(axis=0) + 1e-12
        mu = (w * returns[:, None]).sum(axis=0) / mass
        sigma = np.sqrt((w * (returns[:, None] - mu) ** 2).sum(axis=0) / mass)

        order = np.argsort(mu, kind="stable")
        self.startprob = self.startprob[order]
        self.transmat = self.transmat[np.ix_(order, order)]
        self.means = self.means[order]
        self.covars = self.covars[order]
        mu, sigma = mu[order], sigma[order]

        self.n_states = k
        self.labels = regime_labels(k)
        vol_rank = np.empty(k, dtype=int)
        vol_rank[np.argsort(sigma, kind="stable")] = np.arange(k)
        self.regime_info = [
            RegimeInfo(
                regime_id=i,
                regime_name=self.labels[i],
                expected_return=float(mu[i]),
                expected_volatility=float(sigma[i]),
                **VOL_BUCKET_PROFILES[volatility_bucket(int(vol_rank[i]), k)],
            )
            for i in range(k)
        ]

    # ----------------------------------------------------------------- inference

    def _require_fitted(self) -> None:
        if self.startprob is None:
            raise RuntimeError("HMMEngine is not fitted")

    def emission_logprob(self, x: np.ndarray) -> np.ndarray:
        """log N(x_t | mean_k, cov_k) for every bar and state, shape (T, K)."""
        self._require_fitted()
        assert self.means is not None and self.covars is not None
        return np.column_stack(
            [
                np.atleast_1d(multivariate_normal.logpdf(x, mean=m, cov=c, allow_singular=True))
                for m, c in zip(self.means, self.covars)
            ]
        )

    def _forward(
        self, x: np.ndarray, prev_log_alpha: np.ndarray | None = None
    ) -> tuple[np.ndarray, float]:
        """Normalized log-space forward pass.

        Returns ``(log_alpha, log_likelihood)`` where ``log_alpha[t]`` is
        log P(state_t | x_1..x_t) (continuing from ``prev_log_alpha`` if given).
        """
        assert self.startprob is not None and self.transmat is not None
        log_b = self.emission_logprob(x)
        with np.errstate(divide="ignore"):
            log_pi = np.log(self.startprob)
            log_a = np.log(self.transmat)
        out = np.empty_like(log_b)
        loglik = 0.0
        prev = prev_log_alpha
        for t in range(log_b.shape[0]):
            prior = log_pi if prev is None else logsumexp(prev[:, None] + log_a, axis=0)
            la = prior + log_b[t]
            norm = logsumexp(la)
            loglik += float(norm)
            if not np.isfinite(norm):
                # Observation has zero density under every state (e.g. outside a singular
                # covariance's support after an extreme shock): treat it as uninformative.
                logger.warning("Bar %d has zero likelihood under all states; keeping the prior", t)
                la = prior
                norm = logsumexp(la)
            prev = la - norm
            out[t] = prev
        return out, loglik

    def forward_log_likelihood(self, features: FeatureInput) -> float:
        """Total log-likelihood log P(x_1..x_T) from the forward filter."""
        self._require_fitted()
        return self._forward(_as_array(features))[1]

    def predict_regime_proba(self, features: FeatureInput) -> pd.DataFrame:
        """Filtered probabilities P(state_t | x_1..x_t) per bar (columns = labels).

        Stateless: does not touch the incremental cache or stability filter.
        """
        self._require_fitted()
        x = _as_array(features)
        probs = np.exp(self._forward(x)[0])
        index = features.index if isinstance(features, pd.DataFrame) else pd.RangeIndex(len(x))
        return pd.DataFrame(probs, index=index, columns=list(self.labels))

    def predict_regime_filtered(self, features: FeatureInput) -> np.ndarray:
        """Most likely regime id per bar under the filtered distribution."""
        return self.predict_regime_proba(features).to_numpy().argmax(axis=1)

    # ------------------------------------------------------- incremental / live

    def reset_filter(self) -> None:
        """Drop the cached forward state and stability history."""
        self._log_alpha = None
        self._last_state = None
        self.stability.reset()

    def update(
        self,
        observation: pd.Series | np.ndarray,
        timestamp: pd.Timestamp | None = None,
        log_events: bool = True,
    ) -> RegimeState:
        """Advance the forward filter one bar using the cached alpha and apply stability rules."""
        self._require_fitted()
        if timestamp is None and isinstance(observation, pd.Series):
            ts = observation.name
            timestamp = pd.Timestamp(ts) if isinstance(ts, (pd.Timestamp, datetime, str)) else None
        x = _as_array(np.asarray(observation, dtype=float).reshape(1, -1))
        log_alpha, _ = self._forward(x, self._log_alpha)
        self._log_alpha = log_alpha[-1]
        probs = np.exp(self._log_alpha)
        raw = int(probs.argmax())

        prev_confirmed = self.stability.confirmed_regime
        detected, confirmed = self.stability.update(raw)
        if log_events and detected and prev_confirmed is not None:
            logger.warning(
                "Regime change detected at %s: %s -> %s (p=%.3f, pending %d/%d bars)",
                timestamp, self.labels[prev_confirmed], self.labels[raw], probs[raw],
                self.stability.pending_count, self.config.stability_bars,
            )
        if log_events and confirmed:
            logger.info(
                "Regime change confirmed at %s: %s -> %s after %d bars",
                timestamp, self.labels[prev_confirmed] if prev_confirmed is not None else None,
                self.labels[raw], self.config.stability_bars,
            )

        sid = int(self.stability.confirmed_regime)  # type: ignore[arg-type]
        self._last_state = RegimeState(
            label=self.labels[sid],
            state_id=sid,
            probability=float(probs[sid]),
            state_probabilities=probs,
            timestamp=timestamp,
            is_confirmed=self.stability.is_confirmed,
            consecutive_bars=self.stability.consecutive_bars,
            raw_state_id=raw,
            in_transition=self.stability.in_transition,
            flicker_rate=self.stability.flicker_rate,
        )
        self._bars_since_train += 1
        return self._last_state

    def run_filter(self, features: pd.DataFrame | np.ndarray, log_events: bool = False) -> list[RegimeState]:
        """Reset, then :meth:`update` through every row (backtest / warm start)."""
        self.reset_filter()
        x = _as_array(features)
        index = features.index if isinstance(features, pd.DataFrame) else [None] * len(x)
        states = [self.update(row, ts, log_events=log_events) for row, ts in zip(x, index)]
        self._bars_since_train = 0
        return states

    def get_current_state(self) -> RegimeState | None:
        """Most recent stability-filtered state, or None before any update."""
        return self._last_state

    def get_regime_stability(self) -> int:
        """Consecutive bars the confirmed regime has persisted."""
        return self.stability.consecutive_bars

    def get_transition_matrix(self) -> pd.DataFrame:
        """Transition matrix with label index/columns (rows = from, columns = to)."""
        self._require_fitted()
        return pd.DataFrame(self.transmat, index=list(self.labels), columns=list(self.labels))

    def detect_regime_change(self) -> bool:
        """True only if the latest update confirmed a regime change."""
        return self.stability.last_change_confirmed

    def get_regime_flicker_rate(self) -> int:
        """Raw regime changes over the last ``flicker_window`` bars."""
        return self.stability.flicker_rate

    def is_flickering(self) -> bool:
        """True if the flicker rate exceeds ``flicker_threshold``."""
        return self.stability.is_flickering

    def is_uncertain(self) -> bool:
        """Uncertainty mode: no state yet, flickering, or confidence below ``min_confidence``."""
        s = self._last_state
        return s is None or self.is_flickering() or s.probability < self.config.min_confidence

    def get_size_multiplier(self) -> float:
        """``transition_size_mult`` while a regime change is pending, else 1.0."""
        return self.config.transition_size_mult if self.stability.in_transition else 1.0

    def get_regime_info(self, regime_id: int) -> RegimeInfo:
        """Static info for ``regime_id``."""
        self._require_fitted()
        return self.regime_info[regime_id]

    def get_regime_volatilities(self) -> dict[int, float]:
        """Expected per-bar return volatility per regime id."""
        return {ri.regime_id: ri.expected_volatility for ri in self.regime_info}

    def get_volatility_ranking(self) -> list[int]:
        """Regime ids ordered from calmest to most turbulent."""
        vols = self.get_regime_volatilities()
        return sorted(vols, key=vols.__getitem__)

    # ----------------------------------------------------------------- retraining

    def should_retrain(self) -> bool:
        """True once ``retrain_interval_bars`` updates have elapsed since training."""
        return self._bars_since_train >= self.config.retrain_interval_bars

    def retrain(
        self,
        features: pd.DataFrame | np.ndarray,
        returns: pd.Series | np.ndarray | None = None,
        training_date: datetime | pd.Timestamp | None = None,
    ) -> list[RegimeState]:
        """Expanding-window retrain on all history in ``features``, then rebuild filter state."""
        self.fit(features, returns, training_date)
        return self.run_filter(features)

    def maybe_retrain(
        self,
        features: pd.DataFrame | np.ndarray,
        returns: pd.Series | np.ndarray | None = None,
        training_date: datetime | pd.Timestamp | None = None,
    ) -> bool:
        """Retrain if :meth:`should_retrain`; return whether it did."""
        if not self.should_retrain():
            return False
        self.retrain(features, returns, training_date)
        return True

    # ---------------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        """Pickle the engine state and write ``<path>.json`` metadata alongside."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "model": self.model,
            "startprob": self.startprob,
            "transmat": self.transmat,
            "means": self.means,
            "covars": self.covars,
            "labels": self.labels,
            "regime_info": [asdict(ri) for ri in self.regime_info],
            "candidates": [asdict(c) for c in self.candidates],
            "feature_names": self.feature_names,
            "metadata": self.metadata,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(self.metadata, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> HMMEngine:
        """Load an engine written by :meth:`save` (filter state starts fresh)."""
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        engine = cls(HMMConfig(**payload["config"]))
        engine.model = payload["model"]
        engine.startprob = payload["startprob"]
        engine.transmat = payload["transmat"]
        engine.means = payload["means"]
        engine.covars = payload["covars"]
        engine.labels = tuple(payload["labels"])
        engine.n_states = len(engine.labels)
        engine.regime_info = [RegimeInfo(**ri) for ri in payload["regime_info"]]
        engine.candidates = [CandidateResult(**c) for c in payload["candidates"]]
        engine.feature_names = payload["feature_names"]
        engine.metadata = payload["metadata"]
        return engine
