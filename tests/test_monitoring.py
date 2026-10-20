"""Monitoring checks use temporary logs, captured terminals and mocked transports."""
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import MagicMock

import pytest
from rich.console import Console

import main
from monitoring.alerts import Alert, AlertManager, AlertSeverity
from monitoring.dashboard import Dashboard, DashboardState, risk_color
from monitoring.logger import FIELDS, RetentionHandler, get_logger, log_event, setup_logging
from tests.test_main_loop import loop, bar


@pytest.fixture(autouse=True)
def isolated_monitoring(tmp_path, monkeypatch):
    smtp, http = MagicMock(), MagicMock()
    monkeypatch.setattr('smtplib.SMTP', smtp)
    monkeypatch.setattr('urllib.request.urlopen', http)
    setup_logging(log_dir=tmp_path)
    yield tmp_path, smtp, http
    for handler in list(logging.getLogger().handlers):
        if getattr(handler, '_monitoring', False):
            logging.getLogger().removeHandler(handler)
            handler.close()


def alert(key='regime_change'):
    return Alert(key, AlertSeverity.WARNING, 'Test', 'Event', datetime.now(timezone.utc))


def test_log_files_fields_rotation_retention(isolated_monitoring):
    path, _, _ = isolated_monitoring
    context = dict(regime='calm', probability=.9, equity=100, positions={'SPY': {}}, daily_pnl=1)
    for category in ('main', 'trades', 'alerts', 'regime'):
        log_event(get_logger(category), 'event', **context)
    for category in ('main', 'trades', 'alerts', 'regime'):
        entries = [json.loads(line) for line in (path/f'{category}.log').read_text().splitlines()]
        assert entries
        assert all(set(FIELDS) | {'timestamp'} <= e.keys() for e in entries)
        assert entries[-1]['equity'] == 100
    handlers = [h for h in logging.getLogger().handlers if isinstance(h, RetentionHandler)]
    assert len(handlers) == 4
    assert all(h.maxBytes == 10*1024*1024 and h.retention_days == 30 for h in handlers)
    old = path/'main.log.old'
    old.touch()
    stamp = (datetime.now().timestamp()-31*86400)
    os.utime(old, (stamp, stamp))
    h = next(h for h in handlers if h.baseFilename.endswith('main.log'))
    h.doRollover()
    assert not old.exists()
    assert list(path.glob('main.log.*'))
    h.maxBytes = 1
    log_event(get_logger('main'), 'size rollover')
    log_event(get_logger('main'), 'second size rollover')
    assert len(list(path.glob('main.log.*'))) >= 2


def test_rate_limit_injected_clock():
    now = [datetime.now(timezone.utc)]
    manager = AlertManager(clock=lambda: now[0], console=Console(file=StringIO()))
    assert manager.send(alert())
    assert not manager.send(alert())
    assert manager.send(alert('other'))
    now[0] += timedelta(minutes=15)
    assert manager.send(alert())


def test_default_and_mocked_delivery(isolated_monitoring, monkeypatch):
    _, smtp, http = isolated_monitoring
    config = main.load_config()
    assert not config['monitoring']['email']['enabled']
    assert not config['monitoring']['webhook']['enabled']
    assert config['broker']['paper_trading'] is True
    assert config['risk']['max_exposure'] == .8
    AlertManager().send(alert())
    smtp.assert_not_called()
    http.assert_not_called()
    monkeypatch.setenv('ALERT_SMTP_PASSWORD', 'test-only')
    manager = AlertManager(email_config=dict(enabled=True, host='smtp.invalid', sender='a@example.invalid',
                          recipient='b@example.invalid', username='test'), webhook_url='https://example.invalid/hook')
    assert manager.send(alert())
    smtp.return_value.__enter__.return_value.login.assert_called_once_with('test', 'test-only')
    smtp.return_value.__enter__.return_value.send_message.assert_called_once()
    request = http.call_args.args[0]
    assert request.method == 'POST'
    assert json.loads(request.data)['event_type'] == 'regime_change'


def test_delivery_failure_contained(loop, isolated_monitoring):
    _, smtp, http = isolated_monitoring
    smtp.side_effect = RuntimeError('secret must not be logged')
    http.side_effect = RuntimeError('secret must not be logged')
    loop.alerts = AlertManager(email_config=dict(enabled=True, host='invalid', sender='x', recipient='y'),
                               webhook_url='https://example.invalid')
    loop.feed_dropped()
    loop.refresh()
    assert not loop.feed_ok
    assert 'secret must not be logged' not in (isolated_monitoring[0]/'alerts.log').read_text()


@pytest.mark.parametrize('event', ['regime_change', 'circuit_breaker', 'large_pnl', 'data_feed_down',
                                    'api_lost', 'hmm_retrained', 'flicker_exceeded'])
def test_loop_triggers(loop, event):
    if event == 'regime_change':
        from dataclasses import replace
        loop.last_regime = replace(loop.engine.update.return_value, label='old')
        loop.process_bar('SPY', bar(loop))
    elif event == 'circuit_breaker':
        loop.risk.update_equity(100000, loop.now())
        loop.portfolio.equity = 96000
        loop.portfolio.daily_pnl = -4000
        loop.check_breaker()
    elif event == 'large_pnl':
        loop.portfolio.daily_pnl = 3000
        loop.refresh()
    elif event == 'data_feed_down':
        loop.feed_dropped()
    elif event == 'api_lost':
        with pytest.raises(ConnectionError):
            loop.read(MagicMock(side_effect=ConnectionError()))
    elif event == 'hmm_retrained':
        loop.ensure_model()
    else:
        loop.engine.is_flickering.return_value = True
        loop.process_bar('SPY', bar(loop))
    assert event in [call.args[0].key for call in loop.alerts.send.call_args_list]
    matching = next(call.args[0] for call in loop.alerts.send.call_args_list if call.args[0].key == event)
    assert set(FIELDS) <= matching.context.keys()


@pytest.mark.parametrize('value,color', [(0, 'green'), (.0209, 'green'), (.021, 'yellow'), (.029, 'yellow'), (.03, 'red')])
def test_risk_colors(value, color):
    assert risk_color(value, .03) == color


def test_dashboard_captured_panels():
    stream = StringIO()
    console = Console(file=stream, width=160, height=40, force_terminal=True, no_color=False, color_system='standard')
    dashboard = Dashboard(console=console)
    dashboard.update(DashboardState(regime='BULL', confidence=.72, stability=14, flicker=1,
                     equity=105230, daily_pnl=340, allocation=.8, leverage=1.25, api_latency_ms=23,
                     hmm_age_days=2, daily_drawdown=.025, drawdown=.11,
                     positions=[dict(symbol='SPY', qty=1, current_price=520.30, entry_price=514,
                                     unrealized_pnl=6.3, stop=508, holding_period='3h')],
                     recent_events=['14:30 SPY Rebalance 60%→80% Low vol']))
    console.print(dashboard.build_layout())
    output = stream.getvalue()
    for label in ('REGIME', 'PORTFOLIO', 'POSITIONS', 'RECENT SIGNALS', 'RISK STATUS', 'SYSTEM',
                  'PAPER', '23ms', '2.0d ago', '14 bars', '1/20', '508', '3h', 'Rebalance'):
        assert label in output
    assert '\x1b[33m' in output and '\x1b[31m' in output
    assert dashboard.refresh_seconds == 5
    dashboard.state.paper = False
    with console.capture() as captured:
        console.print(dashboard.build_layout())
    assert 'LIVE' in captured.get()
