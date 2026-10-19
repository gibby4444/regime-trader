"""End-to-end tests for the ``main.py backtest`` CLI on offline data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

import main
from data.synthetic import make_synthetic_bars


@pytest.fixture()
def fast_settings(tmp_path: Path) -> Path:
    cfg = yaml.safe_load(main.DEFAULT_CONFIG.read_text())
    cfg["hmm"].update(n_candidates=[3, 4], n_init=2, n_iter=30)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture()
def spy_csv(tmp_path: Path) -> Path:
    bars, _ = make_synthetic_bars(1200, seed=3, start="2017-06-01")
    path = tmp_path / "SPY.csv"
    bars.to_csv(path, index_label="date")
    return path


def test_parse_args_backtest_flags() -> None:
    args = main.parse_args(["backtest", "--symbols", "SPY", "--start", "2019-01-01", "--end", "2024-12-31",
                            "--compare", "--stress-test"])
    assert args.mode == "backtest" and args.symbols == ["SPY"] and args.compare and args.stress_test
    assert main.parse_args(["stress"]).stress_test


def test_load_config_overlays_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "abc")
    cfg = main.load_config()
    assert cfg["backtest"]["train_window"] == 252
    assert cfg["backtest"]["min_train_bars"] == 252
    assert cfg["hmm"]["min_train_bars"] == 504, "live HMM minimum unchanged"
    assert cfg["credentials"]["alpaca_api_key"] == "abc"


def test_cli_backtest_csv_compare(fast_settings: Path, spy_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    code = main.main(["--config", str(fast_settings), "backtest", "--symbols", "SPY", "--csv", str(spy_csv),
                      "--start", "2019-03-01", "--end", "2021-12-31", "--compare", "--n-sims", "2",
                      "--output-dir", str(out)])
    assert code == 0
    for name in ("equity_curve", "trade_log", "regime_history", "summary", "benchmarks", "regime_stats"):
        assert (out / f"{name}.csv").exists(), name
    eq = pd.read_csv(out / "equity_curve.csv", index_col=0, parse_dates=True)
    assert eq.index[0] > pd.Timestamp("2019-03-01") and eq.index[-1] <= pd.Timestamp("2021-12-31")
    bench = pd.read_csv(out / "benchmarks.csv", index_col=0)
    assert "Buy & Hold" in bench.index and "200 SMA Trend" in bench.index


def test_cli_stress_synthetic(fast_settings: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    res = main.run_backtest(main.load_config(fast_settings), main.parse_args(
        ["backtest", "--synthetic", "--synthetic-bars", "1100", "--stress-test", "--n-sims", "1",
         "--output-dir", str(out)]))
    assert set(res["stress"].index) >= {"baseline", "crash", "gap", "misclassification",
                                        "misclassification (no risk)"}
    assert (out / "stress_summary.csv").exists() and (out / "stress_crash_sims.csv").exists()


def test_cli_requires_a_data_source(monkeypatch: pytest.MonkeyPatch, fast_settings: Path) -> None:
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(SystemExit, match="No data source"):
        main.main(["--config", str(fast_settings), "backtest"])
