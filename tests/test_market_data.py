"""Tests for data.market_data (CSV loading, validation) and Alpaca historical bars (mocked)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from broker.alpaca_client import AlpacaClient, parse_timeframe
from data.market_data import MarketDataFeed, load_bars_csv


def ohlcv(n: int = 5, start: str = "2023-01-02", scale: float = 1.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, name="timestamp")
    c = scale * (100 + np.arange(n, dtype=float))
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1000.0}, index=idx)


def test_load_single_csv(tmp_path) -> None:
    df = ohlcv().rename(columns=str.title)
    df.index.name = "Date"
    path = tmp_path / "spy.csv"
    df.to_csv(path)
    bars = load_bars_csv(path, ["spy"])
    assert list(bars) == ["SPY"]
    pd.testing.assert_frame_equal(bars["SPY"], ohlcv(), check_names=False, check_freq=False)
    with pytest.raises(ValueError, match="no 'symbol' column"):
        load_bars_csv(path, ["SPY", "QQQ"])


def test_load_long_csv_with_symbol_column(tmp_path) -> None:
    a, b = ohlcv().assign(symbol="SPY"), ohlcv(scale=2).assign(symbol="qqq")
    path = tmp_path / "all.csv"
    pd.concat([a, b]).to_csv(path)
    bars = load_bars_csv(path, ["QQQ", "SPY"])
    assert list(bars) == ["QQQ", "SPY"]
    assert bars["QQQ"]["close"].iloc[0] == 200.0
    with pytest.raises(ValueError, match="not found"):
        load_bars_csv(path, ["IWM"])


def test_load_directory(tmp_path) -> None:
    ohlcv().to_csv(tmp_path / "SPY.csv")
    ohlcv(scale=3).to_csv(tmp_path / "AAPL.csv")
    bars = load_bars_csv(tmp_path, ["aapl"])
    assert list(bars) == ["AAPL"] and bars["AAPL"]["close"].iloc[0] == 300.0
    assert set(load_bars_csv(tmp_path)) == {"SPY", "AAPL"}
    with pytest.raises(FileNotFoundError):
        load_bars_csv(tmp_path, ["MSFT"])


def test_validate_sorts_dedupes_and_drops_bad_rows() -> None:
    df = ohlcv(6)
    df.iloc[2, df.columns.get_loc("close")] = np.nan
    df.iloc[3, df.columns.get_loc("low")] = 0.0
    messy = pd.concat([df.iloc[::-1], df.iloc[[0]]])
    out = MarketDataFeed.validate(messy)
    assert out.index.is_monotonic_increasing and not out.index.has_duplicates
    assert len(out) == 4
    with pytest.raises(ValueError, match="missing"):
        MarketDataFeed.validate(df.drop(columns="volume"))


def test_parse_timeframe() -> None:
    tf = parse_timeframe("1Day")
    assert tf.amount == 1 and tf.unit.value == "Day"
    assert parse_timeframe("15Min").amount == 15
    with pytest.raises(ValueError):
        parse_timeframe("daily")


def test_from_env_and_credentials_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert not AlpacaClient.has_env_credentials()
    with pytest.raises(RuntimeError):
        AlpacaClient.from_env()
    monkeypatch.setenv("ALPACA_API_KEY", "your_key_here")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "your_secret_here")
    assert not AlpacaClient.has_env_credentials(), "placeholders from .env.example are not credentials"
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_PAPER", "false")
    c = AlpacaClient.from_env()
    assert (c.api_key, c.secret_key, c.paper) == ("k", "s", False)


class FakeDataClient:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.requests: list = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        parts = []
        for sym in request.symbol_or_symbols:
            f = self.frames[sym].copy()
            f.index = pd.MultiIndex.from_arrays(
                [[sym] * len(f), f.index.tz_localize("America/New_York").tz_convert("UTC")],
                names=["symbol", "timestamp"],
            )
            parts.append(f.assign(trade_count=1, vwap=f["close"]))
        return SimpleNamespace(df=pd.concat(parts))


def test_get_bars_and_history_via_alpaca(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
    client = AlpacaClient("k", "s")
    fake = FakeDataClient({"SPY": ohlcv(), "QQQ": ohlcv(scale=2)})
    client.data = fake  # type: ignore[assignment]
    raw = client.get_bars(["SPY", "QQQ"], "1Day", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01"))
    assert list(raw.columns) == ["open", "high", "low", "close", "volume"]
    req = fake.requests[0]
    assert req.adjustment.value == "all" and req.timeframe.amount == 1

    feed = MarketDataFeed(client, ["SPY", "QQQ"], "1Day", cache_dir=str(tmp_path))
    hist = feed.get_history(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01"))
    assert list(hist) == ["SPY", "QQQ"]
    assert hist["SPY"].index.tz is None
    assert hist["SPY"].index[0] == pd.Timestamp("2023-01-02"), "daily bars indexed by session date"
    assert len(list(tmp_path.glob("*.csv"))) == 2

    n_requests = len(fake.requests)
    cached = feed.get_history(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01"))
    assert len(fake.requests) == n_requests, "served from cache"
    pd.testing.assert_frame_equal(cached["QQQ"], hist["QQQ"], check_freq=False, check_names=False)
    panel = feed.get_close_panel(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01"))
    assert list(panel.columns) == ["SPY", "QQQ"]
