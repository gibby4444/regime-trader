"""Thin wrapper around the Alpaca trading and market-data APIs (alpaca-py).

Centralises authentication (from env / credentials file), paper vs live endpoint
selection, retries, and conversion of SDK objects to plain Python/pandas types.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient

_TIMEFRAME_UNITS = {
    "min": TimeFrameUnit.Minute,
    "minute": TimeFrameUnit.Minute,
    "hour": TimeFrameUnit.Hour,
    "day": TimeFrameUnit.Day,
    "week": TimeFrameUnit.Week,
    "month": TimeFrameUnit.Month,
}


def parse_timeframe(spec: str) -> TimeFrame:
    """'1Day', '15Min', '1Hour' ... -> alpaca ``TimeFrame``."""
    m = re.fullmatch(r"(\d+)\s*([A-Za-z]+)", spec.strip())
    unit = _TIMEFRAME_UNITS.get(m.group(2).lower()) if m else None
    if m is None or unit is None:
        raise ValueError(f"Unsupported timeframe: {spec!r}")
    return TimeFrame(int(m.group(1)), unit)


class AlpacaClient:
    """Authenticated access to Alpaca trading and data endpoints."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        """Store credentials; SDK clients are created lazily by :meth:`connect`."""
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.trading: TradingClient | None = None
        self.data: StockHistoricalDataClient | None = None

    @classmethod
    def from_env(cls) -> AlpacaClient:
        """Build a client from ``ALPACA_API_KEY``, ``ALPACA_SECRET_KEY``, ``ALPACA_PAPER``."""
        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        paper = os.environ.get("ALPACA_PAPER", "true").strip().lower() not in ("0", "false", "no")
        return cls(key, secret, paper)

    @staticmethod
    def has_env_credentials() -> bool:
        """True if both Alpaca key variables are set and non-placeholder."""
        key, secret = os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_SECRET_KEY", "")
        return bool(key and secret and "your_" not in key and "your_" not in secret)

    def data_client(self) -> StockHistoricalDataClient:
        """Historical market-data client (created on first use; no account call)."""
        if self.data is None:
            self.data = StockHistoricalDataClient(self.api_key, self.secret_key)
        return self.data

    def connect(self) -> None:
        """Instantiate SDK clients and verify credentials via an account call."""
        raise NotImplementedError

    def get_account(self) -> dict[str, Any]:
        """Return account details (equity, cash, buying power, status)."""
        raise NotImplementedError

    def get_equity(self) -> float:
        """Return current account equity."""
        raise NotImplementedError

    def is_market_open(self) -> bool:
        """Return True if the market clock reports the market is open."""
        raise NotImplementedError

    def get_clock(self) -> dict[str, Any]:
        """Return market clock info (is_open, next_open, next_close)."""
        raise NotImplementedError

    def get_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch split/dividend-adjusted OHLCV bars as a (symbol, timestamp)-indexed DataFrame.

        ``ALPACA_DATA_FEED`` (e.g. ``iex``) selects the feed; default is the
        account's default (SIP for historical bars older than 15 minutes).
        """
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=parse_timeframe(timeframe),
            start=pd.Timestamp(start).to_pydatetime(),
            end=pd.Timestamp(end).to_pydatetime() if end is not None else None,
            adjustment=Adjustment.ALL,
            feed=DataFeed(os.environ["ALPACA_DATA_FEED"].lower()) if os.environ.get("ALPACA_DATA_FEED") else None,
        )
        df = self.data_client().get_stock_bars(request).df
        if df.empty:
            return df
        cols = ["open", "high", "low", "close", "volume"]
        return df[cols].sort_index()

    def get_latest_quotes(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """Return latest bid/ask per symbol."""
        raise NotImplementedError

    def get_positions(self) -> list[dict[str, Any]]:
        """Return raw open positions from the broker."""
        raise NotImplementedError

    def get_orders(self, status: str = "open") -> list[dict[str, Any]]:
        """Return orders filtered by ``status``."""
        raise NotImplementedError
