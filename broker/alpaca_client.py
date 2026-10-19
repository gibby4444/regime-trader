"""Lazy Alpaca SDK access. Credentials are read only when explicitly connecting."""
from __future__ import annotations

import os
import time
from typing import Any

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from dotenv import load_dotenv


def plain(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='python')
    if isinstance(value, dict):
        return value
    return vars(value)


class AlpacaClient:
    PAPER_URL = 'https://paper-api.alpaca.markets'
    LIVE_URL = 'https://api.alpaca.markets'
    LIVE_PROMPT = "⚠️ LIVE TRADING MODE. Type 'YES I UNDERSTAND THE RISKS' to confirm:"

    def __init__(self, config_dict=None, *, input_fn=None, sleep_fn=time.sleep):
        self.config = dict(config_dict or {})
        if any(k in self.config for k in ('api_key', 'secret_key', 'ALPACA_API_KEY', 'ALPACA_SECRET_KEY')):
            raise ValueError('Credentials must come from the environment or .env')
        self.paper = True
        self.trading = self.data = None
        self._api_key = self._secret_key = None
        self._input = input_fn or input
        self._sleep = sleep_fn
        self._live_confirmed = False
        self.trade_context: dict[str, dict] = {}

    def __repr__(self):
        return f'AlpacaClient(paper={self.paper}, credentials=***)'

    @classmethod
    def from_env(cls, **kwargs):
        return cls(**kwargs)

    @property
    def base_url(self):
        return self.PAPER_URL if self.paper else self.LIVE_URL

    def connect(self):
        load_dotenv(override=False)
        setting = self.config.get('paper_trading', os.getenv('ALPACA_PAPER', 'true'))
        if isinstance(setting, str):
            if setting.lower() not in ('true', 'false'):
                raise ValueError('ALPACA_PAPER must be true or false')
            setting = setting.lower() == 'true'
        if not isinstance(setting, bool):
            raise ValueError('paper_trading must be boolean')
        self.paper = setting
        if not self.paper and not self._live_confirmed:
            if self._input(self.LIVE_PROMPT) != 'YES I UNDERSTAND THE RISKS':
                raise PermissionError('Live trading confirmation refused')
            self._live_confirmed = True
        self._api_key = os.getenv('ALPACA_API_KEY')
        self._secret_key = os.getenv('ALPACA_SECRET_KEY')
        if not self._api_key or not self._secret_key:
            raise ValueError('Missing Alpaca environment credentials')
        for attempt in range(self.config.get('reconnect_attempts', 3)):
            try:
                trading = TradingClient(self._api_key, self._secret_key, paper=self.paper,
                                        url_override=self.base_url)
                trading.get_account()  # startup health check
                self.data = StockHistoricalDataClient(self._api_key, self._secret_key)
                self.trading = trading
                return
            except Exception:
                self.trading = self.data = None
                if attempt + 1 == self.config.get('reconnect_attempts', 3):
                    raise ConnectionError('Alpaca connection failed') from None
                self._sleep(min(2 ** attempt, 30))

    def disconnect(self):
        self.trading = self.data = None
        self._api_key = self._secret_key = None

    def call(self, method, *args, **kwargs):
        """Retry reads only; never replay an ambiguous order submission."""
        if self.trading is None:
            self.connect()
        try:
            return getattr(self.trading, method)(*args, **kwargs)
        except (ConnectionError, TimeoutError, OSError):
            self.connect()
            return getattr(self.trading, method)(*args, **kwargs)

    def health_check(self):
        self.get_account()
        return True

    def get_account(self) -> dict[str, Any]:
        return plain(self.call('get_account'))

    def get_equity(self):
        return float(self.get_account()['equity'])

    def get_available_margin(self):
        return float(self.get_account()['buying_power'])

    def get_positions(self):
        return [plain(p) for p in self.call('get_all_positions')]

    def get_orders(self, status='open'):
        return [plain(o) for o in self.call('get_orders', filter=GetOrdersRequest(status=status, nested=True))]

    def get_order_history(self):
        return self.get_orders('all')

    def get_clock(self):
        return plain(self.call('get_clock'))

    def is_market_open(self):
        return bool(self.get_clock()['is_open'])

    def get_bars(self, symbols, timeframe, start, end=None):
        from data.market_data import MarketDataFeed
        return MarketDataFeed(self, symbols, timeframe)._historical(symbols, timeframe, start, end)

    def get_latest_quotes(self, symbols):
        from data.market_data import MarketDataFeed
        return {s: plain(MarketDataFeed(self, symbols).get_latest_quote(s)) for s in symbols}
