"""Step 7: all broker/stream/clock/sleep boundaries are fakes; no keys or sockets."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import json
import os

import numpy as np
import pandas as pd
import pytest

import main
from core.hmm_engine import HMMEngine, RegimeState, RegimeStabilityFilter
from core.regime_strategies import Signal, Direction
from core.risk_manager import PortfolioState, RiskConfig, RiskManager, RiskDecision, TradingState
from core.trading_loop import TradingLoop, retry_api


@pytest.fixture(autouse=True)
def forbid_network_and_secrets(monkeypatch):
    import socket
    from broker import alpaca_client
    def forbidden(*args, **kwargs):
        pytest.fail('Loop tests must never access network or credentials')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(alpaca_client, 'load_dotenv', forbidden)
    monkeypatch.setattr(alpaca_client, 'TradingClient', forbidden)
    monkeypatch.setattr(alpaca_client, 'StockHistoricalDataClient', forbidden)


@pytest.fixture
def loop(tmp_path):
    now = datetime(2026, 10, 19, 16, tzinfo=timezone.utc)
    client, feed, tracker = MagicMock(), MagicMock(), MagicMock()
    client.get_account.return_value = dict(equity=100000, cash=90000, buying_power=90000)
    client.get_clock.return_value = dict(is_open=True, next_open='tomorrow')
    tracker.portfolio = PortfolioState(100000, 90000, 90000, timestamp=now)
    tracker.positions = {}
    risk = RiskManager(RiskConfig(lock_path=str(tmp_path/'trading_halted.lock')))
    engine = MagicMock()
    engine.metadata = {}
    engine.config.flicker_window = 20
    engine._log_alpha = np.array([0.])
    engine.stability = RegimeStabilityFilter()
    engine.regime_info = []
    regime = RegimeState('calm', 0, .9, np.array([1.]), pd.Timestamp(now), True, 3)
    engine.update.return_value = regime
    engine.is_flickering.return_value = False
    strategy, features = MagicMock(), MagicMock()
    sig = Signal('SPY', Direction.LONG, .9, 100, 95, None, .1, 1, 0, 'calm', .9,
                 pd.Timestamp(now), 'test', 'test')
    strategy.generate_signals.return_value = [sig]
    features.build_features.side_effect = lambda b: b[['close']]
    obj = TradingLoop(dict(broker=dict(symbols=['SPY'])), client=client, feed=feed,
                      engine=engine, features=features, strategy=strategy, risk=risk,
                      tracker=tracker, executor=MagicMock(), dashboard=MagicMock(),
                      alerts=MagicMock(), train=MagicMock(), now=lambda: now,
                      sleep=MagicMock(), model_path=tmp_path/'hmm.pkl',
                      snapshot_path=tmp_path/'state_snapshot.json', output=MagicMock())
    return obj


def bar(loop, minutes=0):
    return pd.Series(dict(open=100, high=101, low=99, close=100, volume=1000),
                     name=pd.Timestamp(loop.now()) + pd.Timedelta(minutes=minutes))


def test_startup_order(loop):
    calls = MagicMock()
    for name, mock in [('connect',loop.client.connect), ('account',loop.client.get_account),
                       ('clock',loop.client.get_clock), ('train',loop.train),
                       ('equity',loop.risk), ('sync',loop.tracker.sync),
                       ('stream',loop.feed.start_stream), ('dashboard',loop.dashboard.start)]:
        calls.attach_mock(mock if isinstance(mock, MagicMock) else MagicMock(),name)
    original = loop.restore_snapshot
    loop.restore_snapshot = MagicMock(side_effect=original)
    calls.attach_mock(loop.restore_snapshot,'restore')
    assert loop.startup()
    names = [c[0] for c in calls.mock_calls]
    assert names.index('connect') < names.index('account') < names.index('clock') < names.index('train')
    assert names.index('train') < names.index('sync') < names.index('restore') < names.index('stream') < names.index('dashboard')
    loop.feed.subscribe_bars.assert_called_once_with(['SPY'], '5Min', loop.enqueue_bar)


@pytest.mark.parametrize('age,retrain', [(None,True),(6,False),(7,False),(8,True)])
def test_model_age(loop,monkeypatch,age,retrain):
    if age is not None:
        loop.model_path.touch()
        stamp = (loop.now()-timedelta(days=age)).timestamp()
        os.utime(loop.model_path,(stamp,stamp))
    load = MagicMock(return_value=loop.engine)
    monkeypatch.setattr(HMMEngine,'load',load)
    loop.ensure_model()
    assert loop.train.called == retrain
    if retrain:
        loop.engine.save.assert_called_once()
    else:
        load.assert_called_once()


def test_weekly_retrain(loop):
    loop.trained_at = loop.now()-timedelta(days=8)
    loop.process_bar('SPY',bar(loop))
    loop.train.assert_called_once()


def test_snapshot_roundtrip(loop):
    loop.trained_at = loop.now()
    loop.risk.update_equity(100000,loop.now())
    loop.risk.peak_equity = 120000
    loop.risk.breaker.active.add('daily_halt')
    loop.risk.state = TradingState.HALTED
    loop.last_regime = loop.engine.update.return_value
    loop.last_prediction = pd.Timestamp(loop.now())
    loop.save_snapshot()
    loop.risk.peak_equity = 1
    loop.risk.breaker.active.clear()
    loop.last_regime = None
    loop.restore_snapshot()
    assert loop.risk.peak_equity == 120000
    assert loop.risk.state == TradingState.HALTED
    assert loop.risk.breaker.active == {'daily_halt'}
    assert loop.last_regime.label == 'calm'
    assert loop.tracker.positions == {}


@pytest.mark.parametrize('status', ['approved','modified','rejected'])
def test_risk_paths(loop,status,caplog):
    proposed = loop.strategy.generate_signals.return_value[0]
    changed = replace(proposed,position_size_pct=.05)
    decision = RiskDecision(status != 'rejected', quantity=10,
                            modified_signal=changed if status == 'modified' else None,
                            modifications=['cap'] if status == 'modified' else [],
                            rejection_reason='spread' if status == 'rejected' else None)
    loop.risk.validate_signal = MagicMock(return_value=decision)
    with caplog.at_level('INFO'):
        loop.process_bar('SPY',bar(loop))
    assert loop.counts[status] == 1
    assert status in caplog.text
    if status == 'rejected':
        loop.executor.submit_order.assert_not_called()
    else:
        assert loop.executor.submit_order.call_args.args[0] == (changed if status == 'modified' else proposed)


def test_dry_run_full_pipeline_no_mutations(loop):
    loop.dry_run = True
    loop.tracker.positions['SPY'] = SimpleNamespace(stop=90,qty=10)
    loop.process_bar('SPY',bar(loop))
    loop.features.build_features.assert_called_once()
    loop.engine.update.assert_called_once()
    loop.strategy.generate_signals.assert_called()
    loop.dashboard.update.assert_called_once()
    assert loop.counts['modified'] + loop.counts['approved'] == 1
    assert loop.executor.mock_calls == []


@pytest.mark.parametrize('mid_session',[False,True])
def test_lock_blocks_entries(loop,mid_session):
    if mid_session:
        loop.process_bar('SPY',bar(loop))
        loop.executor.reset_mock()
    Path(loop.risk.config.lock_path).touch()
    if not mid_session:
        loop.startup()
        assert not loop.entries_allowed()
    loop.process_bar('SPY',bar(loop,5))
    loop.executor.submit_order.assert_not_called()


@pytest.mark.parametrize('state',[TradingState.HALTED,TradingState.KILLED])
def test_breaker_exits_bypass_entry_veto(loop,state):
    loop.risk.update_equity(100000,loop.now())
    loop.risk.breaker.active.add('daily_halt' if state == TradingState.HALTED else 'peak_halt')
    loop.risk.state = state
    loop.portfolio.positions = {'SPY':dict(qty=10,market_value=1000)}
    loop.process_bar('SPY',bar(loop))
    loop.executor.submit_order.assert_not_called()
    loop.executor.close_position.assert_called_once_with('SPY')


@pytest.mark.parametrize('sig',[2,15])
def test_shutdown_no_orders_and_snapshot(loop,sig):
    loop.startup()
    loop.executor.reset_mock()
    loop.request_stop(sig)
    loop.shutdown()
    assert loop.executor.mock_calls == []
    assert loop.snapshot_path.exists()
    loop.feed.stop_stream.assert_called_once()
    loop.tracker.stop.assert_called_once()
    loop.process_bar('SPY',bar(loop))
    assert loop.executor.mock_calls == []


def test_feed_drop_pauses_signals_keeps_stops(loop):
    loop.last_regime = loop.engine.update.return_value
    loop.tracker.positions['SPY'] = SimpleNamespace(stop=90,qty=10)
    loop.feed_dropped()
    loop.process_bar('SPY',bar(loop))
    loop.executor.submit_order.assert_not_called()
    loop.engine.update.assert_not_called()
    loop.executor.modify_stop.assert_called_once_with('SPY',95)
    assert loop.tracker.positions['SPY'].stop == 95
    loop.process_bar('SPY',bar(loop,5))
    assert loop.executor.modify_stop.call_count == 1


def test_hmm_error_holds_regime(loop,caplog):
    held = loop.engine.update.return_value
    loop.last_regime = held
    loop.engine.update.side_effect = ValueError('bad HMM')
    loop.process_bar('SPY',bar(loop))
    assert loop.last_regime is held
    assert 'holding current regime' in caplog.text
    assert loop.strategy.generate_signals.call_args.args[2] is held


def test_retry_backoff(loop):
    op = MagicMock(side_effect=[TimeoutError(),ConnectionError(),42])
    assert retry_api(op,loop.sleep) == 42
    assert [c.args[0] for c in loop.sleep.call_args_list] == [1,2]
    op = MagicMock(side_effect=RuntimeError('fail'))
    with pytest.raises(RuntimeError):
        retry_api(op,loop.sleep)
    assert op.call_count == 3


def test_market_closed(loop):
    loop.client.get_clock.return_value = dict(is_open=False,next_open='2026-10-20 09:30 ET')
    loop.run(install_signals=False)
    loop.train.assert_not_called()
    loop.feed.start_stream.assert_not_called()
    assert '2026-10-20' in loop.output.call_args_list[0].args[0]
    assert loop.executor.mock_calls == []


def test_unhandled_saves_alerts_traceback(loop,caplog):
    loop.features.build_features.side_effect = RuntimeError('unexpected')
    loop.queue.put(('bar','SPY',bar(loop)))
    with pytest.raises(RuntimeError,match='unexpected'):
        loop.run(install_signals=False)
    assert json.loads(loop.snapshot_path.read_text())['version'] == 1
    assert [c.args[0].key for c in loop.alerts.send.call_args_list].count('loop_error') == 1
    assert any(r.exc_info for r in caplog.records)
    loop.feed.stop_stream.assert_called_once()


@pytest.mark.parametrize('argv', [['live','--dry-run'],['train-only'],['backtest'],['--dry-run'],
                                 ['--train-only'],['--backtest'],['--stress-test'],['--compare']])
def test_cli_flags(argv):
    assert main.parse_args(argv)


@pytest.mark.parametrize('argv,expected', [(['live','--dry-run'],'run'),(['--train-only'],'train'),
                                          (['--backtest'],'backtest'),(['--stress-test'],'backtest'),
                                          (['--compare'],'backtest')])
def test_cli_dispatch(monkeypatch,argv,expected):
    live, back = MagicMock(), MagicMock()
    factory = MagicMock(return_value=live)
    monkeypatch.setattr(main,'build_live_loop',factory)
    monkeypatch.setattr(main,'run_backtest',back)
    assert main.main(argv) == 0
    if expected == 'run':
        live.run.assert_called_once()
        assert factory.call_args.kwargs['dry_run']
    elif expected == 'train':
        live.train.assert_called_once()
        live.run.assert_not_called()
    else:
        back.assert_called_once()


def test_alpaca_read_retries_without_connecting_or_credentials():
    from broker.alpaca_client import AlpacaClient
    sleep = MagicMock()
    client = AlpacaClient(sleep_fn=sleep)
    client.trading = MagicMock()
    client.trading.get_clock.side_effect = [TimeoutError(), OSError(), {'is_open': True}]
    assert client.get_clock()['is_open']
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2]
    client.trading.get_clock.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        client.get_clock()
    assert client.trading.get_clock.call_count == 6


def test_inflight_shutdown_does_not_cancel_or_retry(loop):
    from broker.order_executor import OrderExecutor
    raw = dict(id='one', client_order_id='one', symbol='SPY', status='new', filled_qty=0)
    loop.client.call.return_value = raw
    loop.client.trade_context = {}
    executor = OrderExecutor(loop.client,loop.risk,loop.portfolio, sleep_fn=loop.sleep,
                             monotonic_fn=lambda: 0, should_stop=lambda: loop.stopping)
    loop.request_stop()
    assert executor.wait_for_fill('one').status == 'new'
    executor.cancel_order('one')
    executor.cancel_all()
    assert loop.client.trading.mock_calls == []


def test_approved_target_submits_delta_and_protective_stop(loop):
    from broker.order_executor import OrderExecutor
    loop.client.trade_context = {}
    loop.client.trading.submit_order.return_value = dict(
        id='one',client_order_id='one',symbol='SPY',status='filled',filled_qty=5)
    loop.client.call.return_value = dict(
        id='one',client_order_id='one',symbol='SPY',status='filled',filled_qty=5)
    loop.portfolio.positions = {'SPY':dict(qty=5,market_value=500)}
    executor = OrderExecutor(loop.client,loop.risk,loop.portfolio,sleep_fn=loop.sleep,monotonic_fn=lambda: 0)
    sig = loop.strategy.generate_signals.return_value[0]
    executor.submit_order(sig,risk_decision=RiskDecision(True,quantity=10))
    request = loop.client.trading.submit_order.call_args.kwargs['order_data']
    assert request.qty == 5
    assert request.order_class.value == 'oto'
    assert request.stop_loss.stop_price == 95
    assert request.time_in_force.value == 'gtc'


def test_halted_stop_updates_bypass_entry_risk(loop):
    from broker.order_executor import OrderExecutor
    loop.client.trade_context = {}
    loop.client.get_orders.return_value = [dict(id='stop',symbol='SPY',stop_price=90,side='sell')]
    loop.client.trading.replace_order_by_id.return_value = dict(
        id='stop',client_order_id='stop',symbol='SPY',status='new',filled_qty=0)
    loop.risk.state = TradingState.KILLED
    executor = OrderExecutor(loop.client,loop.risk,loop.portfolio,sleep_fn=loop.sleep,monotonic_fn=lambda: 0)
    executor.modify_stop('SPY',95)
    assert loop.client.trading.replace_order_by_id.call_count == 1
    with pytest.raises(ValueError,match='tighten'):
        executor.modify_stop('SPY',89)
    assert loop.client.trading.replace_order_by_id.call_count == 1


def test_builder_is_lazy_and_does_not_read_credentials(monkeypatch, tmp_path):
    from broker import alpaca_client
    monkeypatch.setattr(alpaca_client,'load_dotenv',lambda **kw: pytest.fail('No secrets'))
    monkeypatch.setattr(alpaca_client,'TradingClient',lambda *a,**kw: pytest.fail('No network'))
    config = main.load_config()
    config['monitoring']['log_dir'] = str(tmp_path)
    obj = main.build_live_loop(config,dry_run=True)
    from monitoring.logger import setup_logging
    setup_logging(json_file=False)
    assert obj.client.trading is None
    assert obj.config['broker']['paper_trading'] is True
    assert obj.config['risk']['max_exposure'] == .80


def test_idle_drop_and_closed_exit_use_only_fake_clock_and_sleep(loop):
    loop.client.get_clock.side_effect = [dict(is_open=True),dict(is_open=True),
                                       dict(is_open=False,next_open='tomorrow')]
    loop.feed._thread.is_alive.return_value = False
    loop.run(install_signals=False)
    assert not loop.feed_ok
    loop.sleep.assert_called_once_with(1)
    loop.executor.submit_order.assert_not_called()


def test_market_closes_with_queued_bar(loop):
    loop.client.get_clock.side_effect = [dict(is_open=True),dict(is_open=False,next_open='tomorrow')]
    loop.enqueue_bar('SPY',bar(loop))
    loop.run(install_signals=False)
    assert loop.counts['bars'] == 0
    loop.executor.submit_order.assert_not_called()
    assert loop.snapshot_path.exists()


def test_raising_status_store_cannot_interrupt_trading(loop):
    class RaisingStore:
        def __getattr__(self, name):
            def fail(*args, **kwargs):
                raise RuntimeError('monitor unavailable')
            return fail
    loop.status_store = RaisingStore()
    loop.startup()
    loop.process_bar('SPY', bar(loop))
    loop.refresh()
    assert loop.counts['bars'] == 1
    assert sum(loop.counts[key] for key in ('approved', 'modified', 'rejected')) == 1
    loop.shutdown()


def test_status_snapshot_contains_only_public_position_fields(loop):
    loop.status_store = MagicMock()
    loop.portfolio.positions = {'SPY': dict(qty=2, market_value=200, entry_price=90, internal='private')}
    loop.online = True
    loop.refresh()
    data = loop.status_store.update_status.call_args.args[0]
    assert data['positions'] == [dict(symbol='SPY', qty=2, market_value=200)]
    assert data['running'] is True
    assert data['counts'] == loop.counts
    assert data['started_at'] == loop.started_at
    assert data['target_weights'] == loop.target_weights


def test_status_events_and_final_counts(loop):
    loop.status_store = MagicMock()
    loop.client.get_clock.return_value = dict(is_open=False, next_open='tomorrow')
    loop.run(install_signals=False)
    events = loop.status_store.record_event.call_args_list
    assert [call.args[0] for call in events] == ['start', 'market_closed', 'stop']
    data = loop.status_store.update_status.call_args.args[0]
    assert data['running'] is False
    assert data['counts'] == loop.counts
    assert loop.status_store.update_status.call_args.kwargs['force'] is True
    loop.status_store.close.assert_called_once()


@pytest.mark.parametrize('exception', [ValueError('bad input'), InterruptedError('unexpected interruption')])
def test_unhandled_errors_record_status(loop, exception):
    loop.status_store = MagicMock()
    loop.client.connect.side_effect = exception
    with pytest.raises(type(exception)):
        loop.run(install_signals=False)
    errors = [call.args for call in loop.status_store.record_event.call_args_list if call.args[0] == 'error']
    assert errors == [('error', f'{type(exception).__name__}: {exception}')]
    loop.status_store.close.assert_called_once()


@pytest.mark.parametrize('status,held,target,expected_side,original,final', [
    ('approved', 0, 100, 'buy', 100, 100),
    ('modified', 0, 40, 'buy', 100, 40),
    ('rejected', 0, 0, 'buy', 100, 0),
    ('approved', 150, 100, 'sell', 50, 50),
])
def test_dry_run_intended_orders(loop, status, held, target, expected_side, original, final):
    loop.dry_run = True
    loop.status_store = MagicMock()
    loop.portfolio.positions = {'SPY': dict(qty=held, market_value=held * 100)}
    proposed = loop.strategy.generate_signals.return_value[0]
    changed = replace(proposed, position_size_pct=.04, stop_loss=96)
    loop.risk.validate_signal = MagicMock(return_value=RiskDecision(
        status != 'rejected', quantity=target,
        modified_signal=changed if status == 'modified' else None,
        modifications=['weight cap'] if status == 'modified' else [],
        rejection_reason='spread' if status == 'rejected' else None))
    loop.process_bar('SPY', bar(loop))
    row = loop.status_store.record_intended_order.call_args.kwargs
    assert row['side'] == expected_side
    assert row['original_qty'] == original
    assert row['final_qty'] == final
    assert row['price'] == (100.1 if expected_side == 'buy' else 99.9)
    assert row['stop_price'] == (96 if status == 'modified' else 95)
    assert row['outcome'] == ('shrunk' if status == 'modified' else status)
    assert row['notional'] == (final or original) * row['price']
    assert row['dry_run'] and not row['sent']
    loop.executor.submit_order.assert_not_called()


def test_halted_intended_order(loop):
    loop.dry_run = True
    loop.status_store = MagicMock()
    Path(loop.risk.config.lock_path).touch()
    loop.process_bar('SPY', bar(loop))
    row = loop.status_store.record_intended_order.call_args.kwargs
    assert row['outcome'] == 'rejected'
    assert row['reason'] == 'trading halted'
    assert row['original_qty'] == 100
    assert row['final_qty'] == 0
    loop.executor.submit_order.assert_not_called()


def test_intended_order_failure_is_harmless(loop):
    loop.dry_run = True
    loop.status_store = MagicMock()
    loop.status_store.record_intended_order.side_effect = RuntimeError('unavailable')
    loop.process_bar('SPY', bar(loop))
    assert loop.counts['bars'] == 1
    loop.status_store.record_intended_order.assert_called_once()
    loop.executor.submit_order.assert_not_called()
