"""All broker clients and streams are mocked; no credentials or network needed."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor, OrderRequest, OrderSide
from broker.position_tracker import PositionTracker
from core.regime_strategies import Signal, Direction
from core.risk_manager import PortfolioState, RiskDecision, RiskManager, RiskConfig, TradingState
from data.market_data import MarketDataFeed


@pytest.fixture(autouse=True)
def sdk(monkeypatch):
    """Replace every SDK transport before any test can construct it."""
    for path in ('broker.alpaca_client.TradingClient', 'broker.alpaca_client.StockHistoricalDataClient',
                 'broker.position_tracker.TradingStream', 'data.market_data.StockDataStream'):
        monkeypatch.setattr(path, MagicMock())
    monkeypatch.setattr('broker.alpaca_client.load_dotenv', MagicMock())
    monkeypatch.setenv('ALPACA_API_KEY', 'test-key')
    monkeypatch.setenv('ALPACA_SECRET_KEY', 'test-secret')
    monkeypatch.setenv('ALPACA_PAPER', 'true')


@pytest.fixture
def client():
    c = AlpacaClient()
    c.connect()
    c.trading.get_account.return_value = dict(equity='100000', cash='100000', buying_power='100000')
    c.trading.get_all_positions.return_value = []
    return c


@pytest.fixture
def signal():
    return Signal('SPY', Direction.LONG, .9, 100., 95., 110., .1, 1., 0, 'low', .9,
                  pd.Timestamp('2026-01-02', tz='UTC'), 'test', 'test')


def order(status='filled', filled=10, **kwargs):
    return dict(id='order-1', client_order_id='trade-1', symbol='SPY', status=status,
                filled_qty=filled, filled_avg_price=100 if filled else None, qty=10,
                side='buy', **kwargs)


@pytest.fixture
def executor(client):
    risk = MagicMock()
    risk.validate_signal.return_value = RiskDecision(True, quantity=10)
    risk.can_trade.return_value = True
    client.trading.submit_order.return_value = order()
    client.trading.get_order_by_id.return_value = order()
    return OrderExecutor(client, risk, PortfolioState(100000, 100000, 100000), sleep_fn=lambda _: None)


def test_signals_to_orders_sells_before_buys(executor, signal):
    executor.portfolio.positions = {'QQQ': dict(qty=10, market_value=1000)}
    signals = [signal, replace(signal, symbol='QQQ', direction=Direction.FLAT)]
    orders = executor.signals_to_orders(signals, 100000, {'SPY': 100, 'QQQ': 100})
    assert [o.side for o in orders] == [OrderSide.SELL, OrderSide.BUY]
    assert all(o.signal and o.stop_price for o in orders)


def test_submit_blocked_when_risk_halted(executor, signal):
    executor.risk.can_trade.return_value = False
    with pytest.raises(PermissionError):
        executor.submit_order(signal)
    executor.client.trading.submit_order.assert_not_called()


def test_modify_order(executor, signal):
    executor._send(signal)
    executor.client.trading.replace_order_by_id.return_value = order()
    executor.modify('order-1', qty=5, limit_price=101)
    req = executor.client.trading.replace_order_by_id.call_args.kwargs['order_data']
    assert req.qty == 5 and req.limit_price == 101
    assert executor.risk.validate_signal.call_count == 2


def test_cancel_and_cancel_all(executor):
    executor.cancel('order-1')
    executor.cancel_all()
    executor.client.trading.cancel_order_by_id.assert_called_once_with('order-1')
    executor.client.trading.cancel_orders.assert_called_once()


def test_client_order_id_idempotent(executor, signal):
    req = OrderRequest('SPY', OrderSide.BUY, 10, stop_price=95, signal=signal, client_order_id='same')
    assert executor.submit(req) == executor.submit(req)
    executor.client.trading.submit_order.assert_called_once()


@pytest.mark.parametrize('stop', [None, 0, float('nan')])
def test_missing_stop_refuses(executor, signal, stop):
    with pytest.raises(ValueError):
        executor.submit_order(replace(signal, stop_loss=stop))
    executor.client.trading.submit_order.assert_not_called()


def test_rejection_refuses(executor, signal):
    executor.risk.validate_signal.return_value = RiskDecision(False, rejection_reason='test')
    with pytest.raises(PermissionError):
        executor.submit_order(signal)
    executor.client.trading.submit_order.assert_not_called()


def test_limit_and_trade_link(executor, signal):
    response = executor.submit_order(signal)
    req = executor.client.trading.submit_order.call_args.kwargs['order_data']
    assert req.limit_price == 100.1 and req.qty == 10
    context = executor.client.trade_context['order-1']
    assert context['signal'].metadata['trade_id'] == response.trade_id
    assert context['risk_decision'].approved


def test_bracket(executor, signal):
    executor.submit_bracket_order(signal)
    req = executor.client.trading.submit_order.call_args.kwargs['order_data']
    assert req.order_class.value == 'bracket'
    assert req.stop_loss.stop_price == 95 and req.take_profit.limit_price == 110


def test_timeout_partial_retry(executor, signal):
    clock = [0.]
    executor._clock = lambda: clock[0]
    executor._sleep = lambda seconds: clock.__setitem__(0, clock[0]+seconds)
    executor.client.trading.get_order_by_id.side_effect = [order('partially_filled', 3)]*31+[order('canceled', 4)]
    response = executor.submit_order(signal, market_retry=True)
    assert clock[0] == 30
    assert executor.client.trading.submit_order.call_count == 2
    req = executor.client.trading.submit_order.call_args.kwargs['order_data']
    assert req.qty == 6 and req.type.value == 'market'
    assert req.client_order_id == response.trade_id+'-retry'
    executor.client.trading.cancel_order_by_id.assert_called_once()


def test_no_retry_until_cancellation_confirmed(executor, signal):
    executor._clock = iter([0, 31]).__next__
    executor.client.trading.get_order_by_id.return_value = order('pending_cancel', 3)
    executor.submit_order(signal, market_retry=True)
    executor.client.trading.submit_order.assert_called_once()


def test_stop_only_tightens(executor, signal):
    executor._send(signal)
    executor.client.trading.get_order_by_id.return_value = order(stop_price=95) | {'side': 'sell'}
    executor.client.trading.replace_order_by_id.return_value = order()
    with pytest.raises(ValueError):
        executor.modify('order-1', stop_price=94)
    executor.modify('order-1', stop_price=96)


def test_close_positions_bypasses_validation(executor, signal):
    executor.client.trading.get_all_positions.return_value = [dict(symbol='SPY', qty='5')]
    executor.close_position('SPY', replace(signal, stop_loss=None))
    req = executor.client.trading.submit_order.call_args.kwargs['order_data']
    assert req.side.value == 'sell' and req.qty == 5
    executor.risk.validate_signal.assert_not_called()


@pytest.mark.parametrize('answer', ['', 'yes', ' YES I UNDERSTAND THE RISKS', 'YES I UNDERSTAND THE RISKS\n'])
def test_live_requires_exact_confirmation(answer):
    c = AlpacaClient({'paper_trading': False}, input_fn=lambda prompt: answer)
    with pytest.raises(PermissionError):
        c.connect()
    assert c.trading is None


def test_client_endpoints_and_masking(monkeypatch):
    from broker import alpaca_client as module
    prompt = MagicMock(return_value='YES I UNDERSTAND THE RISKS')
    c = AlpacaClient({'paper_trading': False}, input_fn=prompt)
    c.connect()
    assert c.base_url == 'https://api.alpaca.markets'
    prompt.assert_called_once_with(AlpacaClient.LIVE_PROMPT)
    assert module.TradingClient.call_args.kwargs['paper'] is False
    assert 'test-key' not in repr(c) and 'test-secret' not in repr(c)
    c.disconnect()
    assert c.trading is None and c.data is None


def test_paper_default_and_reads(client):
    assert client.base_url == 'https://paper-api.alpaca.markets'
    assert client.get_equity() == 100000 and client.get_available_margin() == 100000
    client.trading.get_clock.return_value = {'is_open': True}
    assert client.is_market_open()
    client.trading.get_orders.return_value = [order()]
    assert client.get_order_history()[0]['id'] == 'order-1'
    assert client.trading.get_orders.call_args.kwargs['filter'].status.value == 'all'


def test_backoff_and_reconnect(monkeypatch):
    from broker import alpaca_client as module
    trading = module.TradingClient.return_value
    trading.get_account.side_effect = [ConnectionError(), ConnectionError(), {'equity': 1}, ConnectionError(), {'equity': 1}, {'equity': 1}]
    sleep = MagicMock()
    c = AlpacaClient(sleep_fn=sleep)
    c.connect()
    assert [x.args[0] for x in sleep.call_args_list] == [1, 2]
    assert c.get_account()['equity'] == 1


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv('ALPACA_API_KEY')
    with pytest.raises(ValueError):
        AlpacaClient().connect()


def test_fill_accounting_and_dedup(client, tmp_path, signal):
    risk = RiskManager(RiskConfig(lock_path=str(tmp_path/'halt.lock')))
    tracker = PositionTracker(client, risk)
    tracker.sync()
    client.trade_context['trade-1'] = dict(signal=signal, trade_id='trade-1')
    event = dict(event='partial_fill', order=order('partially_filled', 3), price=100)
    asyncio.run(tracker.handle_trade_update(event))
    asyncio.run(tracker.handle_trade_update(event))
    assert tracker.positions['SPY'].qty == 3 and risk.trades_today == 1
    event['order'] = order('filled', 10)
    asyncio.run(tracker.handle_trade_update(event))
    assert tracker.positions['SPY'].qty == 10
    assert tracker.portfolio.positions['SPY']['trade_id'] == 'trade-1'
    tracker.update_prices({'SPY': 105}, regime='high')
    assert tracker.unrealized_pnl() == 50
    assert tracker.positions['SPY'].regime_at_entry == 'low'
    assert tracker.positions['SPY'].current_regime == 'high'
    tracker.on_fill('SPY', -10, 105)
    assert tracker.realized_pnl == 50 and tracker.count() == 0
    assert tracker.portfolio.equity == 100050


def test_reconcile_and_stream(client, tmp_path):
    risk = RiskManager(RiskConfig(lock_path=str(tmp_path/'halt.lock')))
    tracker = PositionTracker(client, risk)
    client.trading.get_all_positions.return_value = [dict(symbol='QQQ', qty='2', avg_entry_price='100', current_price='90')]
    tracker.start()
    tracker.stop()
    tracker.stream.subscribe_trade_updates.assert_called_once()
    assert tracker.positions['QQQ'].unrealized_pnl == -20
    assert tracker.positions['QQQ'].holding_period.total_seconds() >= 0
    client.trading.get_all_positions.return_value = []
    tracker.sync()
    assert tracker.count() == 0


def test_fill_updates_breaker(client, tmp_path):
    tracker = PositionTracker(client, RiskManager(RiskConfig(lock_path=str(tmp_path/'halt.lock'))))
    tracker.sync()
    tracker.on_fill('SPY', 1000, 100)
    tracker.update_prices({'SPY': 96})
    assert tracker.portfolio.circuit_breaker_status == 'halted'


def test_data_history_and_gaps(client):
    feed = MarketDataFeed(client, ['SPY'])
    idx = pd.to_datetime(['2026-01-05', '2026-01-02', '2026-01-02'], utc=True)
    frame = pd.DataFrame({k: [100, 99, 98] for k in ['open', 'high', 'low', 'close', 'volume']}, index=idx)
    client.data.get_stock_bars.return_value = SimpleNamespace(df=frame)
    data = feed.get_historical_bars('SPY', '1Day', pd.Timestamp('2026-01-01'))
    assert len(data) == 2 and data.index.is_monotonic_increasing
    assert data.iloc[0]['close'] == 98
    client.data.get_stock_bars.return_value = SimpleNamespace(df=pd.DataFrame())
    assert feed.get_historical_bars('SPY', '1Day', pd.Timestamp('2026-01-01')).empty
    client.data.get_stock_latest_bar.return_value = {'SPY': 'bar'}
    client.data.get_stock_latest_quote.return_value = {'SPY': 'quote'}
    client.data.get_stock_snapshot.return_value = {'SPY': 'snapshot'}
    assert feed.get_latest_bar('SPY') == 'bar'
    assert feed.get_latest_quote('SPY') == 'quote'
    assert feed.get_snapshot('SPY') == 'snapshot'


def test_stream_bars_quotes(client):
    feed = MarketDataFeed(client, ['SPY'])
    callback = MagicMock()
    feed.subscribe_bars(['SPY'], '1Min', callback)
    handler = feed.stream.subscribe_bars.call_args.args[0]
    raw = dict(symbol='SPY', timestamp=pd.Timestamp('2026-01-02T15:00Z'), open=100, high=101, low=99, close=100, volume=10)
    asyncio.run(handler(raw))
    asyncio.run(handler(raw))
    asyncio.run(handler(raw | {'timestamp': pd.Timestamp('2026-01-05T15:00Z')}))
    assert callback.call_count == 2 and len(feed.gaps) == 1
    assert len(feed.get_latest_bars(10)['SPY']) == 2
    feed.subscribe_quotes(['SPY'], callback)
    asyncio.run(feed.stream.subscribe_quotes.call_args.args[0]({'bid_price': 99}))
    assert callback.call_count == 3
    feed.start_stream()
    feed.stop_stream()
    feed.stream.stop.assert_called_once()


def test_real_risk_retry_and_replace(client, signal, tmp_path):
    risk = RiskManager(RiskConfig(lock_path=str(tmp_path/'halt.lock')))
    executor = OrderExecutor(client, risk, PortfolioState(100000, 100000, 100000))
    client.trading.submit_order.return_value = order()
    first = executor._send(signal)
    second = executor._send(signal, qty=3, market=True, trade_id=first.trade_id,
                            client_order_id=first.trade_id+'-retry')
    assert first.trade_id == second.trade_id
    assert client.trading.submit_order.call_count == 2
    client.trading.get_order_by_id.return_value = order()
    client.trading.replace_order_by_id.return_value = order()
    executor.modify('order-1', qty=2, limit_price=100.2)


def test_submission_timeout_is_never_replayed(executor, signal):
    executor.client.trading.submit_order.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        executor._send(signal)
    executor.client.trading.submit_order.assert_called_once()


def test_reconnect_exhaustion_hides_sdk_error(monkeypatch):
    from broker import alpaca_client as module
    module.TradingClient.side_effect = RuntimeError('test-secret')
    c = AlpacaClient(sleep_fn=lambda _: None)
    with pytest.raises(ConnectionError, match='Alpaca connection failed') as error:
        c.connect()
    assert 'test-secret' not in str(error.value)
    assert c.trading is None


def test_env_paper_and_config_precedence(monkeypatch):
    monkeypatch.setenv('ALPACA_PAPER', 'false')
    c = AlpacaClient({'paper_trading': True}, input_fn=lambda _: pytest.fail('Unexpected live prompt'))
    c.connect()
    assert c.paper is True
    with pytest.raises(ValueError):
        AlpacaClient({'api_key': 'test-key'})


def test_short_fill_and_reversal(client, tmp_path):
    tracker = PositionTracker(client, RiskManager(RiskConfig(lock_path=str(tmp_path/'halt.lock'))))
    tracker.sync()
    tracker.on_fill('SPY', -5, 100)
    tracker.on_fill('SPY', 2, 90)
    assert tracker.positions['SPY'].qty == -3 and tracker.realized_pnl == 20
    tracker.on_fill('SPY', 5, 95)
    assert tracker.positions['SPY'].qty == 2
    assert tracker.positions['SPY'].avg_entry_price == 95
    assert tracker.realized_pnl == 35


def test_intraday_aggregation_and_daily_channel(client):
    feed = MarketDataFeed(client, ['SPY'])
    cb = MagicMock()
    feed.subscribe_bars(['SPY'], '5Min', cb)
    handler = feed.stream.subscribe_bars.call_args.args[0]
    for minute in range(5):
        raw = dict(symbol='SPY', timestamp=pd.Timestamp('2026-01-02T15:00Z')+pd.Timedelta(minutes=minute),
                   open=100, high=101, low=99, close=100+minute, volume=10)
        asyncio.run(handler(raw))
        asyncio.run(handler(raw))
    assert cb.call_count == 1
    assert cb.call_args.args[1]['volume'] == 50
    assert cb.call_args.args[1]['close'] == 104
    feed.subscribe_bars(['SPY'], '1Day', cb)
    feed.stream.subscribe_daily_bars.assert_called_once()
    with pytest.raises(ValueError):
        feed.subscribe_bars(['SPY'], '1Week', cb)


def test_lazy_yfinance_loader(monkeypatch):
    import sys
    from data.market_data import load_yfinance_history
    index = pd.date_range('2026-01-01', periods=3)
    bars = pd.DataFrame(dict(Open=[100]*3, High=[101]*3, Low=[99]*3, Close=[100]*3, Volume=[1000]*3), index=index)
    fake = MagicMock()
    fake.download.return_value = bars
    monkeypatch.setitem(sys.modules, 'yfinance', fake)
    assert len(load_yfinance_history(['SPY'], '2026-01-01')['SPY']) == 3
    fake.download.assert_called_once()


def test_public_modify_stop_bracket_leg(executor, signal):
    executor._send(signal, bracket=True)
    leg = order(stop_price=95) | {'id': 'stop-1', 'side': 'sell'}
    executor.client.trading.get_orders.return_value = [order(legs=[leg])]
    executor.client.trading.get_order_by_id.return_value = leg
    executor.client.trading.replace_order_by_id.return_value = leg
    executor.modify_stop('SPY', 96)
    assert executor.client.trading.replace_order_by_id.call_args.args[0] == 'stop-1'
    with pytest.raises(ValueError):
        executor.modify_stop('SPY', 94)


def test_sell_limit_and_close_all(executor, signal):
    req = OrderRequest('SPY', OrderSide.SELL, 5, stop_price=95, signal=signal)
    executor.submit(req)
    submitted = executor.client.trading.submit_order.call_args.kwargs['order_data']
    assert submitted.limit_price == 99.9
    executor.client.trading.get_all_positions.return_value = [dict(symbol='SPY', qty='5')]
    results = executor.close_all_positions()
    assert len(results) == 1
    assert executor.client.trading.submit_order.call_count == 2


def test_source_less_order_refuses(executor):
    with pytest.raises(ValueError):
        executor.submit(OrderRequest('SPY', OrderSide.BUY, 5))
    executor.client.trading.submit_order.assert_not_called()


@pytest.mark.parametrize('blocked', ['halted', 'killed', 'lock', 'daily_cap'])
def test_close_without_signal_while_blocked(client, signal, tmp_path, blocked):
    lock = tmp_path / 'trading_halted.lock'
    risk = RiskManager(RiskConfig(lock_path=str(lock)))
    if blocked == 'lock':
        lock.touch()
    elif blocked == 'daily_cap':
        risk.trades_today = risk.config.max_daily_trades
    else:
        risk.breaker.active.add('daily_halt' if blocked == 'halted' else 'peak_halt')
        risk.state = TradingState.HALTED if blocked == 'halted' else TradingState.KILLED
    executor = OrderExecutor(client, risk, PortfolioState(100000, 100000, 100000))
    if blocked == 'halted':
        executor.portfolio.daily_pnl = -5000
    if blocked == 'daily_cap':
        executor.portfolio.daily_trades = risk.config.max_daily_trades
    client.trading.get_all_positions.return_value = [dict(symbol='SPY', qty='125')]
    client.trading.submit_order.return_value = order()
    assert not risk.can_trade()
    with pytest.raises(PermissionError):
        executor.submit_order(signal)
    client.trading.submit_order.assert_not_called()
    response = executor.close_position('SPY')
    req = client.trading.submit_order.call_args.kwargs['order_data']
    assert req.type.value == 'market' and req.side.value == 'sell' and req.qty == 125
    assert response.trade_id == req.client_order_id
    assert client.trade_context[req.client_order_id]['trade_id'] == response.trade_id
    assert client.trade_context[response.order_id]['signal'] is None


def test_close_all_without_signals(executor):
    executor.risk.validate_signal.side_effect = AssertionError('Entry validation on exit')
    executor.risk.can_trade.return_value = False
    executor.client.trading.get_all_positions.return_value = [
        dict(symbol='SPY', qty='125'), dict(symbol='QQQ', qty='7', side='short'),
        dict(symbol='IWM', qty='-3')]
    def fill(*, order_data):
        return order() | dict(id='close-' + order_data.symbol, symbol=order_data.symbol,
                             client_order_id=order_data.client_order_id)
    executor.client.trading.submit_order.side_effect = fill
    results = executor.close_all_positions()
    requests = [c.kwargs['order_data'] for c in executor.client.trading.submit_order.call_args_list]
    assert [(r.symbol, r.qty, r.side.value) for r in requests] == [
        ('SPY', 125, 'sell'), ('QQQ', 7, 'buy'), ('IWM', 3, 'buy')]
    assert all(r.type.value == 'market' for r in requests)
    assert len({r.trade_id for r in results}) == 3
    for response in results:
        assert executor.client.trade_context[response.order_id]['trade_id'] == response.trade_id
    executor.risk.validate_signal.assert_not_called()
    executor.risk.can_trade.assert_not_called()
