"""Real-time and historical market data fetching.

Historical bars come from Alpaca's REST data API (with optional on-disk cache);
real-time bars/quotes arrive over Alpaca's websocket stream and are buffered
into the same DataFrame schema.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger("regime_trader.data")

OHLCV = ["open", "high", "low", "close", "volume"]

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

    def _cache_path(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp | None) -> Path | None:
        if not self.cache_dir:
            return None
        end_s = pd.Timestamp(end).date() if end is not None else "latest"
        return Path(self.cache_dir) / f"{symbol}_{self.timeframe}_{pd.Timestamp(start).date()}_{end_s}.csv"

    def get_history(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Return per-symbol OHLCV DataFrames indexed by timestamp.

        Daily bars are indexed by tz-naive session date (America/New_York);
        intraday bars keep UTC timestamps. Cached as CSV under ``cache_dir`` when
        an explicit ``end`` is given.
        """
        out: dict[str, pd.DataFrame] = {}
        missing = []
        for sym in self.symbols:
            path = self._cache_path(sym, start, end)
            if use_cache and end is not None and path is not None and path.exists():
                out[sym] = self.validate(_read_ohlcv_csv(path))
            else:
                missing.append(sym)
        if missing:
            raw = self.client.get_bars(missing, self.timeframe, pd.Timestamp(start), end)
            for sym in missing:
                if raw.empty or sym not in raw.index.get_level_values(0):
                    raise ValueError(f"No bars returned for {sym}")
                bars = raw.xs(sym, level=0)
                if self.timeframe.lower().endswith("day"):
                    idx = pd.DatetimeIndex(bars.index)
                    idx = idx.tz_convert("America/New_York").tz_localize(None) if idx.tz is not None else idx
                    bars = bars.set_axis(idx.normalize().rename("timestamp"))
                bars = self.validate(bars)
                out[sym] = bars
                path = self._cache_path(sym, start, end)
                if use_cache and end is not None and path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    bars.to_csv(path, index_label="timestamp")
        return {sym: out[sym] for sym in self.symbols}

    def get_close_panel(
        self, start: pd.Timestamp, end: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Return a wide DataFrame of close prices (columns = symbols)."""
        return pd.DataFrame({s: b["close"] for s, b in self.get_history(start, end).items()})

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
        """Sort, drop duplicate timestamps and rows with missing/non-positive prices."""
        b = bars.rename(columns=str.lower)
        missing = [c for c in OHLCV if c not in b.columns]
        if missing:
            raise ValueError(f"bars missing columns: {missing}")
        b = b[OHLCV].astype(float).sort_index()
        b = b[~b.index.duplicated(keep="last")]
        bad = b[["open", "high", "low", "close"]].isna().any(axis=1) | (b[["open", "high", "low", "close"]] <= 0).any(axis=1)
        if bad.any():
            logger.warning("Dropping %d bars with missing/non-positive prices", int(bad.sum()))
            b = b[~bad]
        b["volume"] = b["volume"].fillna(0.0)
        return b


def _read_ohlcv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    ts_col = next((c for c in ("timestamp", "date", "datetime", "time") if c in df.columns), df.columns[0])
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False)
    if getattr(df[ts_col].dt, "tz", None) is not None:
        df[ts_col] = df[ts_col].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df.rename(columns={ts_col: "timestamp", "adj close": "adj_close"}).set_index("timestamp")
    return df


def load_bars_csv(path: str | Path, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load OHLCV bars for offline backtests.

    ``path`` may be:
      - a directory of ``<SYMBOL>.csv`` files;
      - one CSV with a ``symbol`` column (long format);
      - one CSV without a ``symbol`` column, used for the single requested symbol.

    Columns are case-insensitive: timestamp|date, open, high, low, close, volume.
    """
    p = Path(path)
    if p.is_dir():
        files = {f.stem.upper(): f for f in p.glob("*.csv")}
        wanted = [s.upper() for s in symbols] if symbols else sorted(files)
        absent = [s for s in wanted if s not in files]
        if absent:
            raise FileNotFoundError(f"No CSV for {absent} in {p}")
        return {s: MarketDataFeed.validate(_read_ohlcv_csv(files[s])) for s in wanted}
    df = _read_ohlcv_csv(p)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
        wanted = [s.upper() for s in symbols] if symbols else sorted(df["symbol"].unique())
        absent = [s for s in wanted if s not in set(df["symbol"])]
        if absent:
            raise ValueError(f"Symbols {absent} not found in {p}")
        return {s: MarketDataFeed.validate(df[df["symbol"] == s]) for s in wanted}
    if symbols and len(symbols) > 1:
        raise ValueError(f"{p} has no 'symbol' column but {len(symbols)} symbols were requested")
    name = symbols[0].upper() if symbols else p.stem.upper()
    return {name: MarketDataFeed.validate(df)}
