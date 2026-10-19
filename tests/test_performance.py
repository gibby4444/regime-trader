"""Tests for backtest.performance."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from backtest.backtester import BacktestConfig
from backtest.performance import (
    TRADING_DAYS,
    PerformanceReport,
    benchmark_comparison,
    buy_and_hold,
    cagr,
    calmar_ratio,
    compare_benchmarks,
    confidence_breakdown,
    drawdown_series,
    longest_underwater,
    max_consecutive,
    max_drawdown,
    max_drawdown_duration,
    profit_factor,
    regime_breakdown,
    sharpe_ratio,
    sortino_ratio,
    sma_trend,
    total_return,
    win_rate,
    worst_case,
)

IDX = pd.bdate_range("2022-01-03", periods=6)


def trades_frame(pnls: list[float], conf: list[float] | None = None, regimes: list[str] | None = None) -> pd.DataFrame:
    n = len(pnls)
    return pd.DataFrame({
        "pnl": pnls,
        "is_flat": [False] * n,
        "holding_bars": [2] * n,
        "confidence": conf or [0.9] * n,
        "vol_regime": regimes or ["low"] * n,
    })


def test_ratios_known_values() -> None:
    r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    excess = r - 0.045 / TRADING_DAYS
    assert sharpe_ratio(r) == pytest.approx(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    assert sortino_ratio(r) == pytest.approx(excess.mean() * np.sqrt(TRADING_DAYS) / downside)
    assert np.isnan(sharpe_ratio(pd.Series([0.001] * 10)))
    assert np.isnan(sharpe_ratio(pd.Series([0.01])))


def test_cagr_total_return_and_calmar() -> None:
    eq = pd.Series(np.linspace(100, 200, TRADING_DAYS))
    assert total_return(eq) == pytest.approx(1.0)
    assert cagr(eq) == pytest.approx(1.0)
    assert cagr(pd.Series(np.full(2 * TRADING_DAYS, 100.0)), initial=25.0) == pytest.approx(1.0)
    assert np.isnan(calmar_ratio(eq))


def test_drawdown_depth_and_duration() -> None:
    eq = pd.Series([100, 120, 90, 100, 130, 117, 120], index=pd.bdate_range("2022-01-03", periods=7), dtype=float)
    assert drawdown_series(eq).min() == pytest.approx(-0.25)
    assert max_drawdown(eq) == pytest.approx(0.25)
    assert max_drawdown_duration(eq) == 3, "peak bar 1 -> recovery bar 4"
    assert longest_underwater(eq) == 3
    assert calmar_ratio(eq) == pytest.approx(cagr(eq) / 0.25)


def test_trade_statistics() -> None:
    t = trades_frame([100, -50, 200, -25, 0])
    flat = pd.DataFrame({"pnl": [1e6], "is_flat": [True], "holding_bars": [5], "confidence": [0.9], "vol_regime": ["low"]})
    t = pd.concat([t, flat], ignore_index=True)
    assert win_rate(t) == pytest.approx(2 / 5)
    assert profit_factor(t) == pytest.approx(300 / 75)
    assert profit_factor(trades_frame([10, 20])) == np.inf
    assert max_consecutive([True, True, False, True, True, True, False]) == 3


def test_regime_breakdown_contributions_sum_to_total() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=300)
    r = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
    regimes = pd.Series(rng.choice(["low", "mid", "high"], 300), index=idx)
    t = trades_frame([10, -5, 7], regimes=["low", "low", "high"])
    table = regime_breakdown(r, regimes, t)
    assert list(table.columns) == ["% Time In", "Return Contribution", "Trades", "Avg Trade P&L", "Win Rate", "Sharpe"]
    assert table["% Time In"].sum() == pytest.approx(1.0)
    assert table["Return Contribution"].sum() == pytest.approx(np.log1p(r).sum())
    assert table.loc["low", "Trades"] == 2 and table.loc["low", "Avg Trade P&L"] == pytest.approx(2.5)
    assert table.loc["high", "Win Rate"] == 1.0 and table.loc["mid", "Trades"] == 0


def test_confidence_buckets() -> None:
    t = trades_frame([10, -5, 3, 8, -2], conf=[0.3, 0.55, 0.65, 0.95, 0.70])
    idx = pd.bdate_range("2022-01-03", periods=20)
    r = pd.Series(np.linspace(-0.01, 0.01, 20), index=idx)
    conf = pd.Series(np.linspace(0.3, 0.95, 20), index=idx)
    table = confidence_breakdown(t, r, conf)
    assert list(table.index) == ["<50%", "50-60%", "60-70%", "70%+"]
    assert list(table["Trades"]) == [1, 1, 1, 2]
    assert table.loc["70%+", "Avg P&L"] == pytest.approx(3.0)
    assert table.loc["70%+", "Win Rate"] == pytest.approx(0.5)


def test_worst_case() -> None:
    idx = pd.bdate_range("2022-01-03", periods=40)
    r = pd.Series(0.001, index=idx)
    r.iloc[10:13] = -0.02
    r.iloc[20] = -0.05
    eq = 100 * (1 + r).cumprod()
    wc = worst_case(eq, r, trades_frame([1, -1, -1, -1, 2]))
    assert wc["worst_day"] == pytest.approx(-0.05)
    assert wc["worst_week"] <= -0.05 and wc["worst_month"] < wc["worst_week"]
    assert wc["max_consecutive_losing_days"] == 3
    assert wc["max_consecutive_losing_trades"] == 3
    assert wc["longest_underwater_bars"] == longest_underwater(eq)


def test_benchmark_comparison_beta() -> None:
    rng = np.random.default_rng(3)
    b = pd.Series(rng.normal(0, 0.01, 500))
    stats = benchmark_comparison(2 * b, b)
    assert stats["beta"] == pytest.approx(2.0)
    assert stats["correlation"] == pytest.approx(1.0)


def test_buy_and_hold_and_sma_trend(synthetic: tuple[pd.DataFrame, pd.Series]) -> None:
    bars = synthetic[0]
    index = bars.index[300:500]
    cfg = BacktestConfig()
    eq = buy_and_hold({"SPY": bars}, index, cfg)
    open1 = bars["open"].loc[index[1]]
    shares = int(100_000 / open1)
    cash = 100_000 - shares * open1 * (1 + cfg.slippage_pct)
    assert eq.iloc[-1] == pytest.approx(cash + shares * bars["close"].loc[index[-1]])


def test_sma_trend_exits_below_sma() -> None:
    close = np.r_[np.linspace(100, 150, 260), np.linspace(150, 90, 140)]
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)
    index = idx[220:]
    eq = sma_trend({"SPY": bars}, index, BacktestConfig())
    sma = bars["close"].rolling(200).mean()
    cross = (bars["close"] <= sma).loc[index].idxmax()
    after = eq.loc[cross:].iloc[2:]
    assert eq.iloc[5] > eq.iloc[1], "invested while above the SMA"
    assert after.nunique() == 1, "flat (constant equity) after closing below the SMA"


def test_compare_benchmarks(wf_result, synthetic: tuple[pd.DataFrame, pd.Series]) -> None:
    from core.risk_manager import RiskConfig

    table = compare_benchmarks(wf_result, {"SPY": synthetic[0]}, BacktestConfig(), RiskConfig(), n_random=3)
    assert {"Strategy", "Buy & Hold", "200 SMA Trend", "Random (mean, n=3)", "Random (std, n=3)"} <= set(table.index)
    assert list(table.columns) == ["total_return", "cagr", "sharpe", "max_drawdown"]
    assert table.loc["Strategy", "total_return"] == pytest.approx(wf_result.equity_curve.iloc[-1] / 100_000 - 1)
    assert np.isfinite(table.loc["Random (std, n=3)"]).all()


def test_report_from_result_outputs(wf_result, tmp_path) -> None:
    rep = PerformanceReport.from_result(wf_result)
    d = rep.to_dict()
    for key in ("total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown", "max_drawdown_duration_bars",
                "win_rate", "profit_factor", "total_trades", "avg_holding_period_bars", "worst_day", "worst_week",
                "worst_month", "max_consecutive_losing_days", "longest_underwater_bars"):
        assert key in d
    assert rep.total_trades == int((~wf_result.trades["is_flat"]).sum())
    assert set(rep.regime_stats.index) <= {"low", "mid", "high"}
    assert rep.confidence_stats["Trades"].sum() == rep.total_trades

    buf = io.StringIO()
    rep.print_summary(Console(file=buf, width=140))
    text = buf.getvalue()
    assert "Regime Breakdown" in text and "Confidence Buckets" in text

    paths = rep.to_csv(wf_result, tmp_path)
    for name in ("equity_curve", "trade_log", "regime_history", "summary"):
        assert paths[name].exists()
    eq = pd.read_csv(paths["equity_curve"], index_col=0, parse_dates=True)
    assert len(eq) == len(wf_result.equity_curve)
    hist = pd.read_csv(paths["regime_history"])
    assert {"regime", "probability", "vol_regime"} <= set(hist.columns)
