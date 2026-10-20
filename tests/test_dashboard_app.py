import asyncio
import selectors
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from dashboard import app as module

class PollingSelector(selectors.DefaultSelector):
    # This sandbox denies socketpair.send(), including asyncio's thread wakeups.
    # Polling lets TestClient exercise the real ASGI app without network access.
    def select(self, timeout=None):
        return super().select(.01 if timeout is None else min(timeout, .01))


def make_client():
    return TestClient(module.app, backend_options={
        'loop_factory': lambda: asyncio.SelectorEventLoop(selector=PollingSelector())})

NOW = datetime(2026, 10, 20, 14, 35, tzinfo=timezone.utc)


def view(data=None, decisions=None, regimes=None, events=None, now=NOW):
    return module.build_view({'data': data} if data is not None else None, decisions or [], regimes or [], events or [], now)


def test_running_badges_and_metrics():
    result = view(dict(running=True, heartbeat=NOW.isoformat(), dry_run=True, paper=True,
                       regime='strong bear', confidence=1, target_weights={'SPY': .2, 'QQQ': .3},
                       equity=1000, daily_pnl=100, drawdown=.04, peak_limit=.1, trading_state='normal'))
    assert result['state'] == 'Running'
    assert result['next_run'] is None
    assert result['badges'] == ['Practice mode (no real orders)', 'Paper account']
    assert result['mood'] == 'Strong downtrend'
    assert result['confidence'] == 'Very sure (100%)'
    assert result['target_allocation'] == '50.0%'
    assert result['equity'] == '$1,000.00'
    assert result['daily_pnl_percent'] == '+11.11%'
    assert result['circuit_breaker'] == 'OK – trading allowed'


@pytest.mark.parametrize('data', [None, {'running': False}, {'running': True},
                                    {'running': True, 'heartbeat': (NOW-timedelta(minutes=11)).isoformat()}])
def test_idle(data):
    assert view(data)['state'] == 'Idle'
    assert view(data)['next_run'] == 'Wed Oct 21, 7:35 AM MT'


@pytest.mark.parametrize('now,expected', [
    (datetime(2026, 10, 23, 14, tzinfo=timezone.utc), 'Mon Oct 26, 7:35 AM MT'),
    (datetime(2026, 10, 24, 12, tzinfo=timezone.utc), 'Mon Oct 26, 7:35 AM MT'),
    (datetime(2026, 10, 26, 12, tzinfo=timezone.utc), 'Mon Oct 26, 7:35 AM MT'),
    (datetime(2026, 10, 30, 15, tzinfo=timezone.utc), 'Mon Nov 02, 7:35 AM MT'),
])
def test_weekend_and_dst(now, expected):
    assert view(now=now)['next_run'] == expected


def test_error_and_later_restart():
    events = [dict(kind='error', ts=NOW, message='Failure'), dict(kind='start', ts=NOW-timedelta(hours=1))]
    assert view(events=events)['state'] == 'Stopped with an error'
    assert view(events=events)['last_error'] == dict(message='Failure', time='Tue Oct 20, 8:35 AM MT')
    events.append(dict(kind='start', ts=NOW+timedelta(seconds=1)))
    assert view(events=events)['state'] == 'Idle'


@pytest.mark.parametrize('label,expected', [('weak bull', 'Mild uptrend'), ('bull', 'Uptrend'), ('bear', 'Downtrend'), ('neutral', 'Sideways'), ('sideways', 'Sideways'), ('crash', 'Sharp selloff'), ('euphoria', 'Strong rally'), ('new_label', 'New Label')])
def test_moods(label, expected):
    assert module.mood(label) == expected


@pytest.mark.parametrize('value,expected', [(1, 'Very sure (100%)'), (.7, 'Fairly sure (70%)'), (.55, 'Somewhat sure (55%)'), (.54, 'Unsure (54%)')])
def test_confidence(value, expected):
    assert module.confidence_text(value) == expected


@pytest.mark.parametrize('status,detail,expected', [
    ('rejected', 'stop not below entry', 'Skipped: stop-loss would be above the price'),
    ('rejected', 'Stop above market', 'Skipped: stop-loss would be above the price'),
    ('rejected', 'trading halted', 'Skipped: trading paused by safety limits'),
    ('approved', 'risk approved', 'Approved'),
    ('modified', 'weight cap', 'Approved, size reduced: to stay within limits'),
    ('modified', 'volatility adjustment', 'Approved, size reduced: volatility adjustment'),
    ('rejected', 'duplicate signal', 'Skipped: duplicate signal'),
])
def test_reasons(status, detail, expected):
    assert module.reason(status, detail) == expected


def test_latest_decision_and_regime_history():
    decisions = [dict(symbol='SPY', status='rejected', detail='old', ts=NOW-timedelta(hours=1)),
                 dict(symbol='SPY', status='approved', detail='risk approved', ts=NOW)]
    regimes = [dict(regime='bull', confidence=.9, ts=NOW-timedelta(minutes=i)) for i in range(25)]
    result = view(decisions=decisions, regimes=regimes)
    assert len(result['decisions']) == 1
    assert result['decisions'][0]['reason'] == 'Approved'
    assert result['decisions'][0]['time'] == 'Tue Oct 20, 8:35 AM MT'
    assert len(result['regimes']) == 20


@pytest.mark.parametrize('breaker,expected', [('reduced', 'Caution – trading at half size'), ('halted', 'Paused for the day'), ('killed', 'Shut off – needs a manual reset'), ('other', 'Status unknown')])
def test_breaker(breaker, expected):
    assert view({'trading_state': breaker})['circuit_breaker'] == expected


def test_routes_database_unavailable(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', 'test-only')
    import psycopg
    def fail(*args, **kwargs):
        raise RuntimeError('private connection details')
    monkeypatch.setattr(psycopg, 'connect', fail)
    client = make_client()
    for path in ('/', '/api/status', '/healthz'):
        response = client.get(path)
        assert response.status_code == 200
        assert 'ALPACA' not in response.text
        assert 'private connection' not in response.text
    assert 'No data from the bot yet' in client.get('/').text
    assert all(route.methods == {'GET'} for route in module.app.routes)
    for method in ('post', 'put', 'delete', 'patch'):
        assert getattr(client, method)('/api/status').status_code == 405


def test_routes_with_data_and_escaping(monkeypatch):
    data = dict(running=True, heartbeat=NOW.isoformat(), regime='<script>alert(1)</script>', confidence=.9)
    monkeypatch.setattr(module, 'load_data', lambda: ({'data': data}, [], [], [dict(kind='error', ts=NOW, message='<script>bad()</script>')]))
    client = make_client()
    response = client.get('/')
    assert response.status_code == 200
    assert '<script>' not in response.text
    assert '&lt;script&gt;' in response.text
    assert 'View only. This page cannot place trades.' in response.text
    response = client.get('/api/status')
    assert response.status_code == 200
    assert response.json()['state'] == 'Stopped with an error'
    assert 'ALPACA' not in response.text


def test_loader_is_read_only_and_missing_tables_are_safe(monkeypatch):
    import psycopg
    from types import SimpleNamespace
    queries, options = [], []
    class Connection:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, sql):
            queries.append(sql)
            return SimpleNamespace(fetchone=lambda: {'data': {'running': False}}, fetchall=lambda: [])
    def connect(*args, **kwargs):
        options.append(kwargs)
        return Connection()
    monkeypatch.setenv('DATABASE_URL', 'test-only')
    monkeypatch.setattr(psycopg, 'connect', connect)
    assert module.load_data() == ({'data': {'running': False}}, [], [], [])
    assert options[0]['connect_timeout'] == 5
    assert 'default_transaction_read_only=on' in options[0]['options']
    assert all(sql.startswith('SELECT ') for sql in queries)
    def missing(*args):
        raise RuntimeError('table missing')
    monkeypatch.setattr(Connection, 'execute', missing)
    assert module.load_data() == (None, [], [], [])
