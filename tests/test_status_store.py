"""Status persistence never requires a database or exposes connection details."""
import json
import sys
import threading
from types import SimpleNamespace

from monitoring.status_store import NullStatusStore, StatusStore


def exercise(store):
    store.update_status({'equity': 100})
    store.record_decision('SPY', 'approved', 'risk approved', 'bull', .9)
    store.record_regime('bull', .9)
    store.record_event('start', 'Started')
    store.record_intended_order(symbol='SPY', side='buy', original_qty=10, final_qty=5, reason='cap')


def test_disabled(monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    store = StatusStore()
    assert not store.enabled
    exercise(store)
    store.close()
    exercise(NullStatusStore())


def test_import_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, 'psycopg', None)
    store = StatusStore('test-only')
    assert not store.enabled
    exercise(store)
    store.close()


def queued_store(monkeypatch, clock=lambda: 0):
    monkeypatch.setattr(threading.Thread, 'start', lambda self: None)
    return StatusStore('test-only', clock=clock)


def test_throttling_and_serialization(monkeypatch):
    now = [0]
    store = queued_store(monkeypatch, lambda: now[0])
    store.update_status({'value': object()})
    store.update_status({})
    assert store._queue.qsize() == 1
    payload = json.loads(store._queue.queue[0][1][0])
    assert payload['heartbeat'].endswith('+00:00')
    assert isinstance(payload['value'], str)
    now[0] = 30
    store.update_status({})
    store.update_status({}, force=True)
    assert store._queue.qsize() == 3
    store.record_regime('bull', .9)
    store.record_regime('bull', .8)
    store.record_regime('bear', .9)
    assert store._queue.qsize() == 5


def test_full_queue_drops(monkeypatch):
    store = queued_store(monkeypatch)
    for _ in range(501):
        store.record_event('start', 'test')
    assert store._queue.maxsize == 500
    assert store._queue.qsize() == 500
    exercise(store)
    assert store._queue.qsize() == 500


def test_connect_failure_backs_off_and_never_raises(monkeypatch, caplog):
    store = queued_store(monkeypatch)
    attempts = []
    def fail(*args, **kwargs):
        attempts.append(kwargs)
        raise RuntimeError('test-only-private-connection')
    monkeypatch.setattr(store._driver, 'connect', fail)
    exercise(store)
    store._stop.set()
    store._worker()
    assert len(attempts) == 1
    assert attempts[0] == dict(connect_timeout=5, autocommit=True, options='-c statement_timeout=5000')
    assert 'test-only-private-connection' not in caplog.text
    assert 'test-only' not in caplog.text
    store.close()


def test_fake_connection_sql_and_cleanup(monkeypatch):
    store = queued_store(monkeypatch)
    calls = []
    connection = SimpleNamespace(execute=lambda *args: calls.append(args), close=lambda: calls.append(('CLOSE',)))
    monkeypatch.setattr(store._driver, 'connect', lambda *args, **kwargs: connection)
    exercise(store)
    store._stop.set()
    store._worker()
    sql = [call[0] for call in calls]
    assert sum(s.startswith('CREATE TABLE IF NOT EXISTS') for s in sql) == 5
    assert sum(s.startswith('DELETE FROM') for s in sql) == 4
    assert any('ON CONFLICT (id)' in s for s in sql)
    assert any('INSERT INTO bot_decisions' in s for s in sql)
    assert any('INSERT INTO bot_regimes' in s for s in sql)
    assert any('INSERT INTO bot_events' in s for s in sql)
    assert any('INSERT INTO bot_intended_orders' in s for s in sql)
    assert any("DELETE FROM bot_intended_orders" in s for s in sql)
    insert = next(call for call in calls if call[0].startswith('INSERT INTO bot_intended_orders'))
    assert insert[1][:4] == ('SPY', 'buy', 10, 5)
    assert sql[-1] == 'CLOSE'
    assert store._queue.unfinished_tasks == 0


def test_database_execute_failure_closes(monkeypatch):
    store = queued_store(monkeypatch)
    closed = []
    def fail(*args):
        raise RuntimeError('unavailable')
    monkeypatch.setattr(store._driver, 'connect', lambda *args, **kwargs: SimpleNamespace(execute=fail, close=lambda: closed.append(True)))
    exercise(store)
    store._stop.set()
    store._worker()
    assert closed == [True]


def test_daemon_worker_drains_on_close(monkeypatch):
    import psycopg
    calls = []
    connection = SimpleNamespace(execute=lambda *args: calls.append(args), close=lambda: calls.append(('CLOSE',)))
    monkeypatch.setattr(psycopg, 'connect', lambda *args, **kwargs: connection)
    store = StatusStore('test-only')
    assert store._thread.daemon
    exercise(store)
    store.close()
    assert not store._thread.is_alive()
    assert any('INSERT INTO bot_events' in call[0] for call in calls)
    assert calls[-1] == ('CLOSE',)
