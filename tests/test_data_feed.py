from unittest.mock import MagicMock, patch

import pytest
from alpaca.data.enums import DataFeed

from data import market_data
from data.market_data import MarketDataFeed, data_feed


def test_default_is_iex(monkeypatch):
    monkeypatch.delenv('ALPACA_DATA_FEED', raising=False)
    assert data_feed() == DataFeed.IEX


def test_env_override_sip(monkeypatch):
    monkeypatch.setenv('ALPACA_DATA_FEED', 'SIP')
    assert data_feed() == DataFeed.SIP


def test_invalid_value(monkeypatch):
    monkeypatch.setenv('ALPACA_DATA_FEED', 'bogus')
    with pytest.raises(ValueError):
        data_feed()


def _feed():
    client = MagicMock()
    return MarketDataFeed(client, ['SPY'], '1Day'), client.data


def test_requests_use_iex(monkeypatch):
    monkeypatch.delenv('ALPACA_DATA_FEED', raising=False)
    feed, data = _feed()
    feed._historical(['SPY'], '1Day', '2024-01-01')
    feed.get_latest_bar('SPY'); feed.get_latest_quote('SPY'); feed.get_snapshot('SPY')
    for call in (data.get_stock_bars, data.get_stock_latest_bar,
                 data.get_stock_latest_quote, data.get_stock_snapshot):
        assert call.call_args[0][0].feed == DataFeed.IEX


def test_stream_uses_iex(monkeypatch):
    monkeypatch.delenv('ALPACA_DATA_FEED', raising=False)
    feed, _ = _feed()
    with patch.object(market_data, 'StockDataStream') as stream:
        feed._stream()
    assert stream.call_args.kwargs['feed'] == DataFeed.IEX
