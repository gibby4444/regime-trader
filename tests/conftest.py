"""Shared synthetic-data fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from core.hmm_engine import HMMConfig, HMMEngine
from data.feature_engineering import build_features, log_returns
from data.synthetic import TRUE_DRIFTS, TRUE_TRANSMAT, TRUE_VOLS, make_synthetic_bars  # noqa: F401


def fast_config(**overrides: object) -> HMMConfig:
    """Spec-shaped config with fewer candidates/inits so tests run quickly."""
    params: dict[str, object] = {"n_candidates": [3, 4], "n_init": 3, "n_iter": 50}
    params.update(overrides)
    return HMMConfig(**params)  # type: ignore[arg-type]


def fast_backtester(risk: bool = True, **bt_overrides: object):
    """Walk-forward backtester with spec windows (IS 252 / OOS 126) and a fast HMM."""
    from backtest.backtester import BacktestConfig, WalkForwardBacktester
    from core.regime_strategies import StrategyConfig
    from core.risk_manager import RiskConfig

    return WalkForwardBacktester(
        BacktestConfig(**bt_overrides),  # type: ignore[arg-type]
        fast_config(n_candidates=[3, 4], n_init=2, n_iter=30, min_train_bars=504),
        StrategyConfig(),
        RiskConfig() if risk else None,
    )


@pytest.fixture(scope="session")
def wf_backtester():
    return fast_backtester()


@pytest.fixture(scope="session")
def wf_result(wf_backtester, synthetic: tuple[pd.DataFrame, pd.Series]):
    return wf_backtester.run({"SPY": synthetic[0]})


@pytest.fixture(scope="session")
def synthetic() -> tuple[pd.DataFrame, pd.Series]:
    return make_synthetic_bars()


@pytest.fixture(scope="session")
def features(synthetic: tuple[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    return build_features(synthetic[0])


@pytest.fixture(scope="session")
def returns(synthetic: tuple[pd.DataFrame, pd.Series], features: pd.DataFrame) -> pd.Series:
    return log_returns(synthetic[0]["close"], 1).reindex(features.index)


@pytest.fixture(scope="session")
def trained_engine(features: pd.DataFrame, returns: pd.Series) -> HMMEngine:
    return HMMEngine(fast_config()).fit(features, returns)
