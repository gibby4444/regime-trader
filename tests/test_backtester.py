"""Tests for backtest.backtester: allocation math, execution timing, risk overlay, walk-forward."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import fast_backtester

from backtest.backtester import (
    BacktestConfig,
    Portfolio,
    TargetSignal,
    _apply_risk,
    apply_slippage,
    run_simulation,
)
from core.hmm_engine import HMMEngine
from core.risk_manager import RiskConfig, TradingState


def flat_bars(prices: list[float] | np.ndarray, opens: list[float] | np.ndarray | None = None) -> pd.DataFrame:
    close = np.asarray(prices, dtype=float)
    open_ = np.asarray(opens if opens is not None else prices, dtype=float)
    idx = pd.bdate_range("2022-01-03", periods=len(close), name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999,
         "close": close, "volume": 1e6},
        index=idx,
    )


def schedule(targets: list[float | None]):
    def fn(pos: int, ts: pd.Timestamp) -> TargetSignal:
        t = targets[pos] if pos < len(targets) else targets[-1]
        return TargetSignal(None if t is None else {"A": t}, {"probability": 0.9})
    return fn


# ----------------------------------------------------------- allocation math


def test_portfolio_allocation_math_with_leverage_and_slippage() -> None:
    p = Portfolio(100_000.0, ["A"])
    fills = p.rebalance({"A": 100.0}, {"A": 1.25}, slippage_pct=0.0005)
    assert p.shares["A"] == 1250 and fills[0]["delta"] == 1250
    assert p.cash == pytest.approx(100_000 - 125_000 - 62.5)
    assert p.cash < 0, "leverage > 1 must borrow (negative cash)"
    assert p.equity({"A": 100.0}) == pytest.approx(99_937.5)
    assert p.equity({"A": 110.0}) == pytest.approx(p.cash + 1250 * 110.0)

    eq = p.equity({"A": 110.0})
    p.rebalance({"A": 110.0}, {"A": 0.6}, slippage_pct=0.0005)
    expected_shares = int(eq * 0.6 / 110.0)
    assert p.shares["A"] == expected_shares == 613
    assert p.cash == pytest.approx(-25_062.5 + 637 * 110.0 - 637 * 110.0 * 0.0005)
    assert p.exposure({"A": 110.0}) == pytest.approx(613 * 110.0 / p.equity({"A": 110.0}))


def test_portfolio_commission_and_zero_weight() -> None:
    p = Portfolio(10_000.0, ["A", "B"])
    p.rebalance({"A": 50.0, "B": 20.0}, {"A": 0.5, "B": 0.5}, commission=1.0)
    assert p.shares == {"A": 100, "B": 250}
    assert p.cash == pytest.approx(10_000 - 5_000 - 5_000 - 2.0)
    p.rebalance({"A": 50.0, "B": 20.0}, {"A": 0.0}, commission=1.0)
    assert p.shares == {"A": 0, "B": 0}


def test_apply_slippage_is_adverse() -> None:
    assert apply_slippage(100.0, "buy", 0.0005) == pytest.approx(100.05)
    assert apply_slippage(100.0, "sell", 0.0005) == pytest.approx(99.95)
    with pytest.raises(ValueError):
        apply_slippage(100.0, "short", 0.0005)


# ------------------------------------------------------------ execution timing


def test_signal_at_close_fills_next_open() -> None:
    bars = flat_bars(np.linspace(100, 110, 10), opens=np.linspace(99, 109, 10))
    sim = run_simulation({"A": bars}, bars.index, schedule([None, None, 1.0]), BacktestConfig(slippage_pct=0.0))
    first = sim.fills.iloc[0]
    assert first["signal_timestamp"] == bars.index[2]
    assert first["timestamp"] == bars.index[3]
    assert first["price"] == pytest.approx(bars["open"].iloc[3])
    assert sim.allocation.iloc[2] == 0.0 and sim.allocation.iloc[3] > 0.9


def test_fill_delay_configurable() -> None:
    bars = flat_bars([100.0] * 10)
    sim = run_simulation({"A": bars}, bars.index, schedule([1.0]), BacktestConfig(fill_delay_bars=3))
    assert sim.fills.iloc[0]["timestamp"] == bars.index[3]


def test_last_bar_signal_never_fills() -> None:
    bars = flat_bars([100.0] * 5)
    sim = run_simulation({"A": bars}, bars.index, schedule([None] * 4 + [1.0]), BacktestConfig())
    assert sim.fills.empty


def test_rebalance_threshold() -> None:
    bars = flat_bars([100.0] * 8)
    sim = run_simulation(
        {"A": bars}, bars.index, schedule([1.0, 1.0, 0.95, 0.95, 0.91, 0.6, 0.6, 0.6]), BacktestConfig()
    )
    assert list(sim.fills["signal_timestamp"]) == [bars.index[0], bars.index[5]]


def test_equity_identity_holds_every_bar() -> None:
    rng = np.random.default_rng(1)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
    bars = flat_bars(prices, opens=prices * (1 + rng.normal(0, 0.005, 60)))
    targets = list(rng.choice([0.0, 0.6, 0.95, 1.1875], size=60))
    sim = run_simulation({"A": bars}, bars.index, schedule(targets), BacktestConfig(commission_per_trade=1.0))
    cash, shares = 100_000.0, 0
    fills = sim.fills.set_index("bar")
    for pos in range(60):
        if pos in fills.index:
            f = fills.loc[[pos]].iloc[0]
            cash -= f["delta"] * f["price"] + f["slippage_cost"] + f["commission"]
            shares = f["shares_after"]
        assert sim.equity.iloc[pos] == pytest.approx(cash + shares * bars["close"].iloc[pos])


def test_trades_pnl_sums_to_total() -> None:
    rng = np.random.default_rng(2)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))
    bars = flat_bars(prices)
    targets = list(rng.choice([0.0, 0.6, 1.1875], size=80))
    sim = run_simulation({"A": bars}, bars.index, schedule(targets), BacktestConfig())
    assert sim.trades["pnl"].sum() == pytest.approx(sim.equity.iloc[-1] - 100_000.0)
    assert (sim.trades["holding_bars"] >= 1).all()


# ------------------------------------------------------------------ risk rules


def test_apply_risk_states() -> None:
    rc = RiskConfig()
    w = {"A": 1.1875}
    assert _apply_risk(w, TradingState.NORMAL, 0.0, rc) == w
    assert _apply_risk(w, TradingState.REDUCED, 0.0, rc)["A"] == pytest.approx(1.1875 * 0.5)
    assert _apply_risk(w, TradingState.KILLED, 1.0, rc) == {"A": 0.0}
    assert _apply_risk(w, TradingState.HALTED, 0.6, rc) is None, "halt blocks exposure increases"
    assert _apply_risk({"A": 0.3}, TradingState.HALTED, 0.6, rc) == {"A": 0.3}, "halt allows de-risking"
    assert _apply_risk({"A": 2.0}, TradingState.NORMAL, 0.0, rc)["A"] == pytest.approx(rc.max_leverage)
    assert _apply_risk(w, None, 0.0, None) == w


def test_kill_switch_flattens_and_rearms() -> None:
    prices = np.r_[np.full(5, 100.0), np.linspace(100, 80, 15), np.full(10, 80.0)]
    bars = flat_bars(prices)
    rearm = bars.index[24]
    sim = run_simulation(
        {"A": bars}, bars.index, schedule([1.0]), BacktestConfig(slippage_pct=0.0),
        RiskConfig(daily_dd_reduce=1, daily_dd_halt=1, weekly_dd_reduce=1, weekly_dd_halt=1),
        rearm_at={rearm},
    )
    killed = sim.risk_state[sim.risk_state == "killed"]
    assert len(killed) > 0
    first_kill = bars.index.get_loc(killed.index[0])
    assert sim.allocation.iloc[first_kill + 1] == 0.0, "position flattened at the next open"
    assert sim.risk_state.loc[rearm] == "normal"
    assert sim.allocation.iloc[-1] > 0.9, "trading resumes after re-arm"


def test_risk_disabled_keeps_full_exposure() -> None:
    bars = flat_bars(np.linspace(100, 70, 30))
    sim = run_simulation({"A": bars}, bars.index, schedule([1.0]), BacktestConfig(), None)
    assert (sim.risk_state == "off").all()
    assert sim.allocation.iloc[-1] > 0.9


# --------------------------------------------------------------- walk-forward


def test_backtest_config_from_dict_and_validation() -> None:
    cfg = BacktestConfig.from_dict({"train_window": 252, "min_train_bars": 252}, {"rebalance_threshold": 0.2})
    assert cfg.rebalance_threshold == 0.2 and cfg.fill_delay_bars == 1
    with pytest.raises(ValueError, match="min_train_bars"):
        BacktestConfig(train_window=200, min_train_bars=252)
    with pytest.raises(ValueError, match="Unknown"):
        BacktestConfig.from_dict({"bogus": 1})


def test_make_folds_spec_windows(wf_backtester, features: pd.DataFrame) -> None:
    folds = wf_backtester.make_folds(features.index)
    assert len(folds) == 5
    pos = {ts: i for i, ts in enumerate(features.index)}
    for k, f in enumerate(folds):
        assert pos[f.train_start] == 126 * k
        assert pos[f.train_end] - pos[f.train_start] + 1 == 252
        assert pos[f.test_start] == pos[f.train_end] + 1
    assert all(f.n_test == 126 for f in folds[:-1])
    assert folds[-1].n_test == len(features) - (126 * 4 + 252)


def test_walk_forward_trains_on_252_bars_despite_hmm_minimum(wf_backtester, wf_result) -> None:
    assert wf_backtester.hmm_config.min_train_bars == 252
    assert len(wf_result.engines) == len(wf_result.folds) == 5
    for f, e in zip(wf_result.folds, wf_result.engines):
        assert f.n_train == 252 and e.metadata["n_samples"] == 252
        assert f.n_states == e.n_states and set(f.candidate_bic) == {3, 4}
        assert e.metadata["training_date"].startswith(str(f.train_end.date()))


def test_walk_forward_outputs(wf_result, features: pd.DataFrame) -> None:
    folds = wf_result.folds
    expected = features.loc[folds[0].test_start:folds[-1].test_end].index
    assert wf_result.equity_curve.index.equals(expected)
    assert np.isfinite(wf_result.equity_curve).all()
    hist = wf_result.regime_history
    for col in ("regime", "regime_id", "vol_regime", "probability", "window", "equity", "allocation", "risk_state"):
        assert col in hist
    assert hist["vol_regime"].isin(["low", "mid", "high"]).all()
    assert len(wf_result.fills) > 0
    nxt = {ts: wf_result.equity_curve.index[i + 1] for i, ts in enumerate(wf_result.equity_curve.index[:-1])}
    assert all(nxt[s] == t for s, t in zip(wf_result.fills["signal_timestamp"], wf_result.fills["timestamp"]))
    assert wf_result.trades["pnl"].sum() == pytest.approx(wf_result.equity_curve.iloc[-1] - 100_000.0)


def test_walk_forward_vol_ranking_drives_strategy(wf_result) -> None:
    hist = wf_result.regime_history.dropna(subset=["strategy"])
    mapping = {"low": "LowVolBullStrategy", "mid": "MidVolCautiousStrategy", "high": "HighVolDefensiveStrategy"}
    assert (hist["strategy"] == hist["vol_regime"].map(mapping)).all()


def test_walk_forward_has_no_look_ahead(synthetic: tuple[pd.DataFrame, pd.Series], wf_result) -> None:
    """Dropping future bars must not change any earlier equity, allocation or regime."""
    bars = synthetic[0]
    short = fast_backtester().run({"SPY": bars.iloc[:-60]})
    common = short.equity_curve.index
    assert len(common) > 400
    pd.testing.assert_series_equal(short.equity_curve, wf_result.equity_curve.loc[common])
    pd.testing.assert_series_equal(short.regimes, wf_result.regimes.loc[common])
    pd.testing.assert_series_equal(short.allocation, wf_result.allocation.loc[common])


def test_reusing_engines_skips_training(
    wf_backtester, wf_result, synthetic: tuple[pd.DataFrame, pd.Series], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(HMMEngine, "fit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrained")))
    again = wf_backtester.run({"SPY": synthetic[0]}, engines=wf_result.engines)
    pd.testing.assert_series_equal(again.equity_curve, wf_result.equity_curve)


def test_start_end_and_warmup_history(synthetic: tuple[pd.DataFrame, pd.Series]) -> None:
    bars = synthetic[0]
    bt = fast_backtester()
    _, feats, _, folds = bt.prepare({"SPY": bars}, start="2021-01-04", end="2023-12-29")
    assert feats.index[0] >= pd.Timestamp("2021-01-04")
    assert feats.index[-1] <= pd.Timestamp("2023-12-29")
    assert folds[0].train_start == feats.index[0], "warm-up history before start feeds features only"
    with pytest.raises(ValueError, match="Not enough bars"):
        bt.prepare({"SPY": bars.iloc[:600]})


def test_multi_symbol_equal_weight(synthetic: tuple[pd.DataFrame, pd.Series]) -> None:
    bars = synthetic[0]
    other = bars * 1.0
    other[["open", "high", "low", "close"]] *= 0.5
    res = fast_backtester().run({"SPY": bars, "QQQ": other}, start="2021-06-01")
    w = res.weights[res.allocation > 0.5]
    assert set(res.fills["symbol"]) == {"SPY", "QQQ"}
    np.testing.assert_allclose(w["SPY"], w["QQQ"], atol=0.02)
