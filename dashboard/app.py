"""Standalone read-only monitoring service. Run uvicorn app:app from this folder."""
import html
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
MT = ZoneInfo('America/Denver')


def timestamp(value):
    if not value:
        return None
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (ValueError, TypeError):
        return None


def mt_time(value):
    value = timestamp(value)
    if value is None:
        return 'Unknown time'
    local = value.astimezone(MT)
    return local.strftime('%a %b %d, ') + local.strftime('%I:%M %p').lstrip('0') + ' MT'


def next_run(now):
    local = now.astimezone(MT)
    candidate = local.replace(hour=7, minute=35, second=0, microsecond=0)
    if candidate <= local:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def mood(label):
    label = str(label or 'unknown').lower().replace('_', ' ').replace('-', ' ')
    return {'strong bear': 'Strong downtrend', 'weak bull': 'Mild uptrend',
            'strong bull': 'Strong uptrend', 'weak bear': 'Mild downtrend',
            'bull': 'Uptrend', 'bear': 'Downtrend', 'neutral': 'Sideways',
            'sideways': 'Sideways', 'crash': 'Sharp selloff', 'euphoria': 'Strong rally'}.get(label, label.title())


def confidence_text(value):
    percent = float(value or 0) * 100
    label = 'Very sure' if percent >= 90 else 'Fairly sure' if percent >= 70 else 'Somewhat sure' if percent >= 55 else 'Unsure'
    return f'{label} ({percent:.0f}%)'


def reason(status, detail):
    detail = str(detail or '')
    lower = detail.lower()
    if 'stop' in lower and ('not below' in lower or 'above' in lower):
        return 'Skipped: stop-loss would be above the price'
    if 'trading halted' in lower:
        return 'Skipped: trading paused by safety limits'
    if status == 'modified':
        short = 'to stay within limits' if any(word in lower for word in ('cap', 'weight', 'exposure')) else detail[:160]
        return 'Approved, size reduced: ' + short
    if 'risk approved' in lower:
        return 'Approved'
    if status == 'rejected':
        return 'Skipped: ' + detail[:160]
    return 'Approved' if status == 'approved' else detail[:160] or str(status).title()


def build_view(status_row, decisions, regimes, events, now):
    """Convert database rows into a safe, plain-language public view."""
    now = timestamp(now)
    data = (status_row or {}).get('data', status_row or {})
    heartbeat = timestamp(data.get('heartbeat'))
    state = 'Running' if data.get('running') and heartbeat and timedelta(0) <= now-heartbeat <= timedelta(minutes=10) else 'Idle'
    ordered_events = sorted(events, key=lambda row: timestamp(row.get('ts')) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    error = next((row for row in ordered_events if row.get('kind') == 'error'), None)
    start = next((row for row in ordered_events if row.get('kind') == 'start'), None)
    if error and timestamp(error.get('ts')) and (not start or not timestamp(start.get('ts')) or timestamp(error['ts']) > timestamp(start['ts'])):
        state = 'Stopped with an error'
    badges = []
    if data.get('dry_run'):
        badges.append('Practice mode (no real orders)')
    if data.get('paper'):
        badges.append('Paper account')
    latest = {}
    for row in sorted(decisions, key=lambda row: (timestamp(row.get('ts')) or datetime.min.replace(tzinfo=timezone.utc), row.get('id', 0)), reverse=True):
        symbol = row.get('symbol', '')
        if symbol not in latest:
            latest[symbol] = dict(symbol=symbol, reason=reason(row.get('status'), row.get('detail')), time=mt_time(row.get('ts')))
    equity, pnl = float(data.get('equity') or 0), float(data.get('daily_pnl') or 0)
    breaker = str(data.get('trading_state') or 'unknown').lower()
    return dict(has_data=bool(status_row), state=state, next_run=mt_time(next_run(now)) if state != 'Running' else None,
                heartbeat=mt_time(heartbeat), badges=badges, mood=mood(data.get('regime')),
                confidence=confidence_text(data.get('confidence')),
                target_allocation=f"{sum(float(v) for v in (data.get('target_weights') or {}).values()):.1%}",
                equity=f'${equity:,.2f}', daily_pnl=f'{pnl:+,.2f}',
                daily_pnl_dollars=('-' if pnl < 0 else '+')+f'${abs(pnl):,.2f}', daily_pnl_percent=f'{pnl/(equity-pnl) if equity-pnl else 0:+.2%}',
                drawdown=f"{float(data.get('drawdown') or 0):.1%}", peak_limit=f"{float(data.get('peak_limit') or 0):.1%}",
                circuit_breaker={'normal': 'OK – trading allowed', 'reduced': 'Caution – trading at half size',
                                 'halted': 'Paused for the day', 'killed': 'Shut off – needs a manual reset'}.get(breaker, 'Status unknown'),
                decisions=list(latest.values()),
                regimes=[dict(mood=mood(r.get('regime')), confidence=confidence_text(r.get('confidence')), time=mt_time(r.get('ts')))
                         for r in sorted(regimes, key=lambda r: timestamp(r.get('ts')) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)[:20]],
                last_error=dict(message=str(error.get('message', ''))[:500], time=mt_time(error.get('ts'))) if error else None)


def load_data():
    """Use a read-only transaction for every request; never expose connection errors."""
    try:
        dsn = os.environ.get('DATABASE_URL')
        if not dsn:
            return None, [], [], []
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(dsn, connect_timeout=5, options='-c statement_timeout=5000 -c default_transaction_read_only=on', row_factory=dict_row) as connection:
            status = connection.execute('SELECT data FROM bot_status WHERE id=1').fetchone()
            decisions = connection.execute('SELECT DISTINCT ON (symbol) * FROM bot_decisions ORDER BY symbol, ts DESC, id DESC').fetchall()
            regimes = connection.execute('SELECT * FROM bot_regimes ORDER BY ts DESC, id DESC LIMIT 20').fetchall()
            # Include the latest start and error even when many other events follow them.
            events = connection.execute('SELECT DISTINCT ON (kind) * FROM bot_events ORDER BY kind, ts DESC, id DESC').fetchall()
        return status, decisions, regimes, events
    except Exception:
        return None, [], [], []


def current_view():
    return build_view(*load_data(), datetime.now(timezone.utc))


@app.get('/healthz')
def healthz():
    return {'ok': True}


@app.get('/api/status')
def status():
    return current_view()


def render(view):
    esc = lambda value: html.escape(str(value), quote=True)
    def card(title, content):
        return '<section><h2>' + esc(title) + '</h2>' + content + '</section>'
    content = card('Bot status', '<h1>' + esc(view['state']) + '</h1>' +
                   (('<p>Next scheduled run: ' + esc(view['next_run']) + '</p>') if view['next_run'] else '') +
                   ''.join('<p class="badge">' + esc(b) + '</p>' for b in view['badges']) +
                   '<p>Last heartbeat: ' + esc(view['heartbeat']) + '</p>')
    if not view['has_data']:
        content += card('Waiting for status', '<p>No data from the bot yet</p>')
    else:
        content += card('Market mood', '<h3>' + esc(view['mood']) + '</h3><p>' + esc(view['confidence']) + '</p>')
        content += card('Portfolio', '<p>Equity: ' + esc(view['equity']) + '</p><p>Today’s P&amp;L: ' + esc(view['daily_pnl_dollars']) + ' (' + esc(view['daily_pnl_percent']) + ')</p><p>Target allocation: ' + esc(view['target_allocation']) + '</p>')
        content += card('Safety limits', '<p>' + esc(view['circuit_breaker']) + '</p><p>Drawdown from peak: ' + esc(view['drawdown']) + ' / limit ' + esc(view['peak_limit']) + '</p>')
    for title, rows in [('Latest decisions', view['decisions']), ('Recent market moods', view['regimes'])]:
        content += card(title, ''.join('<p><strong>' + esc(r.get('symbol', r.get('mood', ''))) + '</strong> · ' + esc(r.get('reason', r.get('confidence', ''))) + '<br><small>' + esc(r['time']) + '</small></p>' for r in rows) or '<p>No updates yet</p>')
    if view['last_error']:
        content += card('Last error', '<p>' + esc(view['last_error']['message']) + '</p><small>' + esc(view['last_error']['time']) + '</small>')
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="30"><title>Trading bot status</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f3f6fa;color:#183044;margin:0;padding:24px}main{max-width:1000px;margin:auto;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}section{background:white;border:1px solid #dbe4ed;border-radius:14px;padding:24px;overflow-wrap:anywhere}h1,h2,h3{margin:0 0 12px}h2{font-size:18px;color:#486078}h1{font-size:30px}.badge{background:#e8f2fc;border-radius:8px;padding:6px 10px}small{color:#586b7d}footer{max-width:1000px;margin:24px auto}@media(max-width:640px){main{grid-template-columns:1fr}body{padding:14px}}
</style></head><body><main>''' + content + '</main><footer>View only. This page cannot place trades.</footer></body></html>'


@app.get('/', response_class=HTMLResponse)
def index():
    return render(current_view())
