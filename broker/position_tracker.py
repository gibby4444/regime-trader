"""Reconciled positions, fill accounting, and portfolio circuit breakers."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Thread

from alpaca.trading.stream import TradingStream
from broker.alpaca_client import plain
from core.risk_manager import PortfolioState, RiskManager


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_price: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stop: float | None = None
    regime_at_entry: str | None = None
    current_regime: str | None = None
    trade_id: str | None = None

    @property
    def market_value(self):
        return self.qty*self.market_price

    @property
    def unrealized_pnl(self):
        return self.qty*(self.market_price-self.avg_entry_price)

    @property
    def holding_period(self):
        return datetime.now(timezone.utc)-self.entry_time


class PositionTracker:
    def __init__(self, client, risk=None, portfolio=None):
        self.client = client
        self.risk = risk or RiskManager()
        self.portfolio = portfolio or PortfolioState(0, 0, 0)
        self.positions = {}
        self.realized_pnl = 0.
        self.stream = self._thread = None
        self._seen = set()
        self._filled = {}

    def sync(self):
        actual = {}
        for raw in self.client.get_positions():
            symbol = raw['symbol']
            qty = float(raw['qty'])
            if str(getattr(raw.get('side'), 'value', raw.get('side'))) == 'short':
                qty = -abs(qty)
            old = self.positions.get(symbol)
            actual[symbol] = Position(symbol, qty, float(raw['avg_entry_price']), float(raw['current_price']),
                                     entry_time=old.entry_time if old else datetime.now(timezone.utc),
                                     stop=old.stop if old else None,
                                     regime_at_entry=old.regime_at_entry if old else None,
                                     current_regime=old.current_regime if old else None,
                                     trade_id=old.trade_id if old else None)
        # Recover protective stops from broker orders, including bracket legs.
        for parent in self.client.get_orders():
            for raw_order in [parent, *(parent.get('legs') or [])]:
                order = plain(raw_order)
                symbol = order.get('symbol')
                if symbol in actual and order.get('stop_price'):
                    actual[symbol].stop = float(order['stop_price'])
        for context in self.client.trade_context.values():
            signal = context['signal']
            if signal is not None and signal.symbol in actual:
                position = actual[signal.symbol]
                position.regime_at_entry = position.regime_at_entry or signal.regime_name
                position.current_regime = position.current_regime or signal.regime_name
                position.trade_id = position.trade_id or context['trade_id']
        self.positions = actual
        account = self.client.get_account()
        self.portfolio.equity = float(account['equity'])
        self.portfolio.cash = float(account['cash'])
        self.portfolio.buying_power = float(account['buying_power'])
        self._publish()

    def _publish(self):
        p = self.portfolio
        p.positions = {s: dict(qty=v.qty, market_value=v.market_value, entry_price=v.avg_entry_price,
                               current_price=v.market_price, unrealized_pnl=v.unrealized_pnl,
                               stop=v.stop, entry_time=v.entry_time, holding_period=v.holding_period,
                               regime_at_entry=v.regime_at_entry, current_regime=v.current_regime,
                               trade_id=v.trade_id) for s, v in self.positions.items()}
        p.timestamp = datetime.now(timezone.utc)
        p.peak_equity = max(p.peak_equity, p.equity)
        self.risk.update_equity(p.equity, p.timestamp)
        self.risk.evaluate_state(p.equity, p.positions)
        state = self.risk.breaker.update(p)
        self.risk.state = state
        p.circuit_breaker_status = state.value

    def on_fill(self, symbol, qty, price, *, trade_id=None, stop=None, regime=None, timestamp=None):
        if not qty:
            return
        old = self.positions.get(symbol)
        before = self.unrealized_pnl()+self.realized_pnl
        remaining = qty+(old.qty if old else 0)
        entry = price
        if old:
            if old.qty*qty > 0:
                entry = (abs(old.qty)*old.avg_entry_price+abs(qty)*price)/abs(remaining)
            else:
                self.realized_pnl += min(abs(qty), abs(old.qty))*(price-old.avg_entry_price)*(1 if old.qty > 0 else -1)
                if remaining*old.qty > 0:
                    entry = old.avg_entry_price
        if remaining:
            same = old is not None and old.qty*remaining > 0
            self.positions[symbol] = Position(symbol, remaining, entry, price,
                entry_time=old.entry_time if same else timestamp or datetime.now(timezone.utc),
                stop=stop if stop is not None else old.stop if same else None,
                regime_at_entry=old.regime_at_entry if same else regime,
                current_regime=regime if regime is not None else old.current_regime if same else None,
                trade_id=trade_id or (old.trade_id if same else None))
        else:
            self.positions.pop(symbol, None)
        change = self.unrealized_pnl()+self.realized_pnl-before
        self.portfolio.equity += change
        self.portfolio.cash -= qty*price
        self.portfolio.buying_power = max(0., self.portfolio.buying_power-qty*price)
        self.portfolio.daily_pnl += change
        self.portfolio.weekly_pnl += change
        self._publish()

    async def handle_trade_update(self, event):
        event = plain(event)
        kind = str(getattr(event['event'], 'value', event['event']))
        if kind not in ('fill', 'partial_fill'):
            return
        order = plain(event['order'])
        order_id = str(order['id'])
        cumulative = float(order.get('filled_qty') or 0)
        key = (order_id, cumulative)
        if key in self._seen:
            return
        delta = cumulative-self._filled.get(order_id, 0.)
        if delta <= 0:
            return
        context = self.client.trade_context.get(order.get('client_order_id'), self.client.trade_context.get(order_id, {}))
        if not context:
            for c in self.client.trade_context.values():
                if c.get('signal') is not None and c['signal'].symbol == order['symbol']:
                    context = c
        signal = context.get('signal')
        side = str(getattr(order['side'], 'value', order['side']))
        self.on_fill(order['symbol'], delta if side == 'buy' else -delta, float(event['price']),
                     trade_id=context.get('trade_id'), stop=signal.stop_loss if signal else None,
                     regime=signal.regime_name if signal else None, timestamp=event.get('timestamp'))
        if order_id not in self._filled:
            self.risk.record_trade()
            self.portfolio.daily_trades += 1
        self._filled[order_id] = cumulative
        self._seen.add(key)

    def start(self):
        self.sync()
        self.stream = TradingStream(self.client._api_key, self.client._secret_key, paper=self.client.paper)
        self.stream.subscribe_trade_updates(self.handle_trade_update)
        self._thread = Thread(target=self.stream.run, daemon=True)
        self._thread.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
        if self._thread:
            self._thread.join(timeout=5)

    def update_prices(self, prices, regime=None):
        before = self.unrealized_pnl()
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].market_price = price
                if regime is not None:
                    self.positions[symbol].current_regime = regime
        change = self.unrealized_pnl()-before
        self.portfolio.equity += change
        self.portfolio.daily_pnl += change
        self.portfolio.weekly_pnl += change
        self._publish()

    def weights(self, equity):
        return {s: p.market_value/equity for s, p in self.positions.items()} if equity else {}

    def gross_exposure(self):
        return sum(abs(p.market_value) for p in self.positions.values())

    def unrealized_pnl(self):
        return sum(p.unrealized_pnl for p in self.positions.values())

    def count(self):
        return len(self.positions)
