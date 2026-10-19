"""Risk-validated order lifecycle with stable signal-to-fill trade identifiers."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from enum import Enum
from uuid import uuid4

from alpaca.trading.requests import (LimitOrderRequest, MarketOrderRequest,
    ReplaceOrderRequest, StopLossRequest, TakeProfitRequest)
from broker.alpaca_client import plain
from core.regime_strategies import Signal, Direction
from core.risk_manager import PortfolioState, RiskManager


class OrderSide(str, Enum):
    BUY = 'buy'
    SELL = 'sell'


class OrderType(str, Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.LIMIT
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = 'day'
    client_order_id: str | None = None
    signal: Signal | None = None


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    client_order_id: str
    symbol: str
    status: str
    filled_qty: float
    filled_avg_price: float | None
    trade_id: str = ''


def result(order, trade_id=''):
    o = plain(order)
    return OrderResult(str(o['id']), o.get('client_order_id', ''), o['symbol'],
                       str(getattr(o['status'], 'value', o['status'])),
                       float(o.get('filled_qty') or 0),
                       float(o['filled_avg_price']) if o.get('filled_avg_price') else None,
                       trade_id or o.get('client_order_id', ''))


class OrderExecutor:
    def __init__(self, client, risk, portfolio=None, *, sleep_fn=time.sleep,
                 monotonic_fn=time.monotonic, market_retry=False):
        self.client, self.risk, self.portfolio = client, risk, portfolio
        self._sleep, self._clock = sleep_fn, monotonic_fn
        self.market_retry = market_retry
        self._results = {}
        self._intents = {}

    def _portfolio(self):
        if self.portfolio is not None:
            return self.portfolio
        a = self.client.get_account()
        return PortfolioState(float(a['equity']), float(a['cash']), float(a['buying_power']),
                              {p['symbol']: {**p, 'market_value': float(p['market_value'])}
                               for p in self.client.get_positions()})

    def _validate(self, signal, qty=None):
        if signal.stop_loss is None or not math.isfinite(signal.stop_loss) or signal.stop_loss <= 0:
            raise ValueError('A valid stop loss is required')
        # Revalidate lifecycle changes without treating the same trade as a
        # second strategy signal. All sizing and breaker checks still run.
        linked = signal.metadata.get('parent_trade_id') or signal.metadata.get('trade_id')
        own_trade = any(c['trade_id'] == linked for c in self.client.trade_context.values())
        key = (signal.symbol, signal.direction)
        prior = None
        if own_trade and isinstance(self.risk, RiskManager):
            prior = self.risk._duplicates.pop(key, None)
        try:
            decision = self.risk.validate_signal(signal, self._portfolio())
        finally:
            if prior is not None:
                self.risk._duplicates[key] = prior
        if not decision.approved or not self.risk.can_trade():
            raise PermissionError(decision.rejection_reason or 'Risk manager rejected order')
        size = decision.quantity if qty is None else min(qty, decision.quantity)
        if not math.isfinite(size) or size <= 0:
            raise ValueError('Risk-approved quantity must be positive')
        return decision, size

    def _send(self, signal, *, bracket=False, qty=None, side='buy', market=False,
              trade_id=None, client_order_id=None):
        trade_id = trade_id or signal.metadata.get('trade_id') or uuid4().hex
        client_order_id = client_order_id or trade_id
        if client_order_id in self._results:
            return self._results[client_order_id]
        signal = replace(signal, metadata={**signal.metadata, 'trade_id': trade_id})
        decision, size = self._validate(signal, qty)
        decision.trade_id = trade_id
        if bracket and (signal.take_profit is None or signal.take_profit <= signal.entry_price):
            raise ValueError('Bracket requires take profit above entry')
        if side == 'buy' and signal.stop_loss >= signal.entry_price:
            raise ValueError('Long stop must be below entry')
        args = dict(symbol=signal.symbol, qty=size, side=side, time_in_force='day',
                    client_order_id=client_order_id)
        if bracket:
            args.update(order_class='bracket', stop_loss=StopLossRequest(stop_price=signal.stop_loss),
                        take_profit=TakeProfitRequest(limit_price=signal.take_profit))
        request = (MarketOrderRequest(**args) if market else
                   LimitOrderRequest(**args, limit_price=round(signal.entry_price*(1.001 if side == 'buy' else .999), 2)))
        self.client.trade_context[client_order_id] = dict(trade_id=trade_id, signal=signal, risk_decision=decision)
        # No automatic retry: a timeout can mean the broker accepted the order.
        response = result(self.client.trading.submit_order(order_data=request), trade_id)
        self._results[client_order_id] = response
        self._intents[response.order_id] = (signal, side, size)
        self.client.trade_context[response.order_id] = self.client.trade_context[client_order_id]
        return response

    def submit_order(self, signal, *, market_retry=None):
        order = self._send(signal)
        final = self.wait_for_fill(order.order_id)
        retry = self.market_retry if market_retry is None else market_retry
        if retry and final.status == 'canceled':
            original = self._intents[order.order_id][2]
            remaining = original-final.filled_qty
            if remaining > 0:
                return self._send(signal, qty=remaining, market=True, trade_id=order.trade_id,
                                  client_order_id=order.trade_id+'-retry')
        return final

    def submit_bracket_order(self, signal):
        order = self._send(signal, bracket=True)
        return self.wait_for_fill(order.order_id)

    def signals_to_orders(self, signals, equity, prices):
        orders = []
        portfolio = self._portfolio()
        for signal in signals:
            signal = replace(signal, entry_price=prices[signal.symbol])
            target = signal.position_size_pct*signal.leverage*equity/signal.entry_price
            held = float(portfolio.positions.get(signal.symbol, {}).get('qty', 0))
            delta = (0 if signal.direction == Direction.FLAT else target)-held
            if delta:
                orders.append(OrderRequest(signal.symbol, OrderSide.BUY if delta > 0 else OrderSide.SELL,
                                           abs(delta), stop_price=signal.stop_loss, signal=signal))
        return sorted(orders, key=lambda o: o.side == OrderSide.BUY)

    def submit(self, order):
        if order.signal is None or order.stop_price is None:
            raise ValueError('Order requires its source signal and stop')
        signal = replace(order.signal, stop_loss=order.stop_price)
        return self._send(signal, qty=order.qty, side=order.side.value,
                          market=order.order_type == OrderType.MARKET,
                          client_order_id=order.client_order_id)

    def submit_batch(self, orders):
        return [self.submit(o) for o in orders]

    def modify(self, order_id, qty=None, limit_price=None, stop_price=None):
        old = plain(self.client.call('get_order_by_id', order_id))
        context = self.client.trade_context.get(order_id)
        if context is None:
            raise ValueError('Cannot replace an order without its risk context')
        previous = float(old.get('stop_price') or context['signal'].stop_loss)
        side = str(getattr(old['side'], 'value', old['side']))
        if stop_price is not None and ((side == 'sell' and stop_price < previous) or
                                       (side == 'buy' and stop_price > previous)):
            raise ValueError('Stop may only tighten')
        signal = replace(context['signal'], entry_price=limit_price or context['signal'].entry_price,
                         stop_loss=stop_price or context['signal'].stop_loss)
        _, approved = self._validate(signal, qty or float(old['qty']))
        if qty is not None and approved < qty:
            raise PermissionError('Replacement exceeds approved quantity')
        response = result(self.client.trading.replace_order_by_id(order_id,
                          order_data=ReplaceOrderRequest(qty=qty, limit_price=limit_price, stop_price=stop_price)),
                          context['trade_id'])
        self.client.trade_context[response.order_id] = {**context, 'signal': signal}
        return response

    def modify_stop(self, symbol, new_stop):
        for parent in self.client.get_orders():
            for order in [parent, *(parent.get('legs') or [])]:
                order = plain(order)
                if order['symbol'] == symbol and order.get('stop_price'):
                    key = str(order['id'])
                    if key not in self.client.trade_context:
                        context = self.client.trade_context.get(str(parent['id']))
                        if context:
                            self.client.trade_context[key] = context
                    return self.modify(key, stop_price=new_stop)
        raise ValueError('No tracked stop order for symbol')

    def cancel_order(self, order_id):
        self.client.trading.cancel_order_by_id(order_id)

    cancel = cancel_order

    def cancel_all(self):
        self.client.trading.cancel_orders()

    def close_position(self, symbol, signal=None):
        position = next(p for p in self.client.get_positions() if p['symbol'] == symbol)
        # Exit sizing comes from the broker position, bypassing all entry vetoes.
        qty = float(position['qty'])
        if str(getattr(position.get('side'), 'value', position.get('side'))) == 'short':
            qty = -abs(qty)
        trade_id = uuid4().hex
        if signal is not None:
            metadata = {**signal.metadata, 'trade_id': trade_id}
            if signal.metadata.get('trade_id'):
                metadata['parent_trade_id'] = signal.metadata['trade_id']
            signal = replace(signal, metadata=metadata)
        request = MarketOrderRequest(symbol=symbol, qty=abs(qty),
                                     side='sell' if qty > 0 else 'buy',
                                     time_in_force='day', client_order_id=trade_id)
        context = dict(trade_id=trade_id, signal=signal, risk_decision=None)
        self.client.trade_context[trade_id] = context
        # A submission timeout must not trigger an automatic retry.
        response = result(self.client.trading.submit_order(order_data=request), trade_id)
        self._results[trade_id] = response
        self.client.trade_context[response.order_id] = context
        return response

    def close_all_positions(self):
        return [self.close_position(p['symbol']) for p in self.client.get_positions()]

    def wait_for_fill(self, order_id, timeout_seconds=30.):
        deadline = self._clock()+timeout_seconds
        context = self.client.trade_context.get(order_id, {})
        while True:
            order = result(self.client.call('get_order_by_id', order_id), context.get('trade_id', ''))
            if order.status in ('filled', 'canceled', 'expired', 'rejected'):
                return order
            if self._clock() >= deadline:
                self.cancel_order(order_id)
                # Confirm cancellation before any retry; account for late/partial fills.
                canceled = result(self.client.call('get_order_by_id', order_id), order.trade_id)
                return canceled
            self._sleep(min(1., max(0., deadline-self._clock())))
