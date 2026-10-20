"""Alpaca history and streams; optional yfinance remains lazy."""
from __future__ import annotations

import inspect
import os
import re
from threading import Thread

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.requests import (StockBarsRequest, StockLatestBarRequest,
                                 StockLatestQuoteRequest, StockSnapshotRequest)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def data_feed():
    """Market data feed; IEX by default (free plan cannot query recent SIP)."""
    value = (os.environ.get('ALPACA_DATA_FEED') or '').strip().lower()
    if not value:
        return DataFeed.IEX
    try:
        return DataFeed(value)
    except ValueError as exc:
        raise ValueError(f'Invalid ALPACA_DATA_FEED {value!r}; use iex or sip') from exc

def plain(value):
    return value.model_dump(mode='python') if hasattr(value, 'model_dump') else value if isinstance(value, dict) else vars(value)



def timeframe(value):
    if isinstance(value, TimeFrame):
        return value
    match = re.fullmatch(r'(\d+)(Min|Hour|Day|Week|Month)', value)
    if not match:
        raise ValueError('Unsupported timeframe')
    return TimeFrame(int(match[1]), TimeFrameUnit(match[2]))


class MarketDataFeed:
    def __init__(self, client, symbols, timeframe='1Day', cache_dir=None):
        self.client, self.symbols, self.timeframe = client, symbols, timeframe
        self.cache_dir = cache_dir
        self._callbacks = []
        self._bars = {}
        self.stream = self._thread = None
        self.gaps = []

    def _data(self):
        if self.client.data is None:
            self.client.connect()
        return self.client.data

    def _historical(self, symbols, period, start, end=None):
        response = self._data().get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbols, timeframe=timeframe(period), start=start, end=end,
            feed=data_feed()))
        return response.df

    def get_historical_bars(self, symbol, timeframe, start, end=None):
        frame = self._historical([symbol], timeframe, start, end)
        if frame.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'],
                                index=pd.DatetimeIndex([], tz='UTC', name='timestamp'))
        if isinstance(frame.index, pd.MultiIndex):
            if symbol not in frame.index.get_level_values(0):
                return pd.DataFrame(columns=frame.columns, index=pd.DatetimeIndex([], tz='UTC'))
            frame = frame.xs(symbol)
        return self.validate(frame)

    def get_history(self, start, end=None, use_cache=True):
        return {s: self.get_historical_bars(s, self.timeframe, start, end) for s in self.symbols}

    def get_close_panel(self, start, end=None):
        return pd.DataFrame({s: b['close'] for s, b in self.get_history(start, end).items()})

    def get_latest_bars(self, lookback):
        return {s: b.tail(lookback) for s, b in self._bars.items()}

    def get_latest_bar(self, symbol):
        return self._data().get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=[symbol], feed=data_feed())).get(symbol)

    def get_latest_quote(self, symbol):
        return self._data().get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=[symbol], feed=data_feed())).get(symbol)

    def get_snapshot(self, symbol):
        return self._data().get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=[symbol], feed=data_feed())).get(symbol)

    def _stream(self):
        self._data()
        if self.stream is None:
            self.stream = StockDataStream(self.client._api_key, self.client._secret_key,
                                          feed=data_feed())
        return self.stream

    def subscribe(self, callback):
        self._callbacks.append(callback)

    def subscribe_bars(self, symbols, timeframe, callback):
        period = globals()['timeframe'](timeframe)
        # Alpaca provides minute and daily bar channels. Aggregate completed
        # minute bars into larger intraday intervals, without inventing gap bars.
        if period.unit not in (TimeFrameUnit.Minute, TimeFrameUnit.Hour, TimeFrameUnit.Day):
            raise ValueError('Streaming supports minute, hour, and daily bars')
        if period.unit == TimeFrameUnit.Day and period.amount != 1:
            raise ValueError('Streaming daily bars supports 1Day')
        pending = {}
        last_seen = {}

        async def dispatch(raw):
            raw = plain(raw)
            symbol = raw['symbol']
            stamp = pd.Timestamp(raw['timestamp'])
            stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
            if symbol in last_seen and stamp <= last_seen[symbol]:
                return
            last_seen[symbol] = stamp
            row = {k: raw[k] for k in ('open', 'high', 'low', 'close', 'volume')}
            if period.unit != TimeFrameUnit.Day:
                minutes = period.amount*(60 if period.unit == TimeFrameUnit.Hour else 1)
                bucket = stamp.floor(f'{minutes}min')
                if minutes > 1:
                    prev = pending.get(symbol)
                    if prev and bucket < prev[0]:
                        return
                    if prev and bucket == prev[0]:
                        agg = prev[1]
                        agg.update(high=max(agg['high'], row['high']), low=min(agg['low'], row['low']),
                                   close=row['close'], volume=agg['volume']+row['volume'])
                    else:
                        pending[symbol] = (bucket, row)
                    if stamp+pd.Timedelta(minutes=1) < bucket+pd.Timedelta(minutes=minutes):
                        return
                    stamp, row = pending.pop(symbol)
            frame = self._bars.get(symbol)
            if frame is not None and not frame.empty:
                if stamp <= frame.index[-1]:
                    return
                expected = pd.Timedelta(days=1) if period.unit == TimeFrameUnit.Day else pd.Timedelta(minutes=minutes)
                if stamp-frame.index[-1] > expected:
                    self.gaps.append((symbol, frame.index[-1], stamp))
            new = pd.DataFrame([row], index=pd.DatetimeIndex([stamp], name='timestamp'))
            self._bars[symbol] = self.validate(pd.concat([frame, new]) if frame is not None else new)
            value = pd.Series(row, name=stamp)
            for fn in [callback, *self._callbacks]:
                answer = fn(symbol, value)
                if inspect.isawaitable(answer):
                    await answer

        stream = self._stream()
        if period.unit == TimeFrameUnit.Day:
            stream.subscribe_daily_bars(dispatch, *symbols)
        else:
            stream.subscribe_bars(dispatch, *symbols)

    def subscribe_quotes(self, symbols, callback):
        async def dispatch(quote):
            answer = callback(quote)
            if inspect.isawaitable(answer):
                await answer
        self._stream().subscribe_quotes(dispatch, *symbols)

    def start_stream(self):
        stream = self._stream()
        self._thread = Thread(target=stream.run, daemon=True)
        self._thread.start()

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
        if self._thread:
            self._thread.join(timeout=5)

    @staticmethod
    def validate(bars):
        bars = bars.copy()
        bars.index = pd.to_datetime(bars.index, utc=True)
        bars = bars.loc[~bars.index.duplicated(keep='last')].sort_index()
        # Never forward-fill prices through market closures, holidays, or halts.
        return bars.dropna(subset=['open', 'high', 'low', 'close', 'volume'])


def load_yfinance_history(symbols, start, end=None):
    """Optional historical loader. Import and network access happen only on invocation."""
    import yfinance as yf
    from data.feature_engineering import validate_bars
    result = {}
    for symbol in symbols:
        bars = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)
        if bars.empty:
            raise ValueError(f"No historical data for {symbol}")
        result[symbol] = validate_bars(bars)
    return result
