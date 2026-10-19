"""Real-time and historical market data fetching.

Historical bars come from Alpaca's REST data API (with optional on-disk cache);
real-time bars/quotes arrive over Alpaca's websocket stream and are buffered
into the same DataFrame schema.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from broker.alpaca_client import AlpacaClient

BarCallback = Callable[[str, pd.Series], None]


class MarketDataFeed:
    """Unified access to historical and streaming OHLCV data."""

    def __init__(
        self,
        client: AlpacaClient,
        symbols: list[str],
        timeframe: str = "1Day",
        cache_dir: str | None = None,
    ) -> None:
        """Configure the symbol universe, timeframe, and optional cache directory."""
        self.client = client
        self.symbols = symbols
        self.timeframe = timeframe
        self.cache_dir = cache_dir
        self._callbacks: list[BarCallback] = []

    def get_history(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Return per-symbol OHLCV DataFrames indexed by timestamp (UTC)."""
        raise NotImplementedError

    def get_close_panel(
        self, start: pd.Timestamp, end: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Return a wide DataFrame of close prices (columns = symbols)."""
        raise NotImplementedError

    def get_latest_bars(self, lookback: int) -> dict[str, pd.DataFrame]:
        """Return the most recent ``lookback`` completed bars per symbol."""
        raise NotImplementedError

    def subscribe(self, callback: BarCallback) -> None:
        """Register a callback invoked on each new completed bar."""
        raise NotImplementedError

    def start_stream(self) -> None:
        """Open the websocket stream and dispatch bars to subscribers."""
        raise NotImplementedError

    def stop_stream(self) -> None:
        """Close the websocket stream."""
        raise NotImplementedError

    @staticmethod
    def validate(bars: pd.DataFrame) -> pd.DataFrame:
        """Check monotonic index, drop duplicates, flag gaps/NaNs."""
        raise NotImplementedError
