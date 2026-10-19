"""Thin wrapper around the Alpaca trading and market-data APIs (alpaca-py).

Centralises authentication (from env / credentials file), paper vs live endpoint
selection, retries, and conversion of SDK objects to plain Python/pandas types.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient


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
        raise NotImplementedError

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
        """Fetch historical OHLCV bars as a (symbol, timestamp)-indexed DataFrame."""
        raise NotImplementedError

    def get_latest_quotes(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """Return latest bid/ask per symbol."""
        raise NotImplementedError

    def get_positions(self) -> list[dict[str, Any]]:
        """Return raw open positions from the broker."""
        raise NotImplementedError

    def get_orders(self, status: str = "open") -> list[dict[str, Any]]:
        """Return orders filtered by ``status``."""
        raise NotImplementedError
