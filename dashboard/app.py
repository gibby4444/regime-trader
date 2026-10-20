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


def build_view(status_row, decisions, regimes, events, now, intended_orders=None):
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
    today_orders, history_orders = [], []
    for row in sorted(intended_orders or [], key=lambda r: (timestamp(r.get('ts')) or datetime.min.replace(tzinfo=timezone.utc), r.get('id', 0)), reverse=True)[:300]:
        ts = timestamp(row.get('ts'))
        if ts is None or not now - timedelta(days=7) <= ts <= now:
            continue
        outcome = row.get('outcome', 'approved')
        original, final = float(row.get('original_qty') or 0), float(row.get('final_qty') or 0)
        shares = (f'{original:g} → {final:g} shares' if outcome == 'shrunk' else
                  f'{original:g} shares (not allowed)' if outcome == 'rejected' else f'{final:g} shares')
        item = dict(time=mt_time(ts), symbol=row.get('symbol', ''),
                    side=str(row.get('side', '')).title(), shares=shares,
                    dollar_value=f"${float(row.get('notional') or 0):,.2f}",
                    price=f"Limit ${float(row.get('price') or 0):,.2f}",
                    stop=f"Stop ${float(row['stop_price']):,.2f}" if row.get('stop_price') is not None else 'No stop',
                    mood=mood(row.get('regime')), outcome=outcome.title(),
                    reason=reason('modified' if outcome == 'shrunk' else outcome, row.get('reason')),
                    label='Dry run: not sent' if row.get('dry_run') or not row.get('sent') else 'Sent to paper account')
        (today_orders if ts.astimezone(MT).date() == now.astimezone(MT).date() else history_orders).append(item)
    equity, pnl = float(data.get('equity') or 0), float(data.get('daily_pnl') or 0)
    breaker = str(data.get('trading_state') or 'unknown').lower()
    return dict(would_trade_today=today_orders, order_history=history_orders, has_data=bool(status_row), state=state, next_run=mt_time(next_run(now)) if state != 'Running' else None,
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
            return None, [], [], [], []
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(dsn, connect_timeout=5, options='-c statement_timeout=5000 -c default_transaction_read_only=on', row_factory=dict_row) as connection:
            status = connection.execute('SELECT data FROM bot_status WHERE id=1').fetchone()
            decisions = connection.execute('SELECT DISTINCT ON (symbol) * FROM bot_decisions ORDER BY symbol, ts DESC, id DESC').fetchall()
            regimes = connection.execute('SELECT * FROM bot_regimes ORDER BY ts DESC, id DESC LIMIT 20').fetchall()
            # Include the latest start and error even when many other events follow them.
            events = connection.execute('SELECT DISTINCT ON (kind) * FROM bot_events ORDER BY kind, ts DESC, id DESC').fetchall()
            intended_orders = connection.execute("SELECT * FROM bot_intended_orders WHERE ts >= now() - interval '7 days' ORDER BY ts DESC, id DESC LIMIT 300").fetchall()
        return status, decisions, regimes, events, intended_orders
    except Exception:
        return None, [], [], [], []


def current_view():
    data = load_data()
    return build_view(*data[:4], datetime.now(timezone.utc), intended_orders=data[4] if len(data) > 4 else [])


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
    def orders_table(rows, empty):
        columns = [('time', 'Time (MT)'), ('symbol', 'Symbol'), ('side', 'Side'),
                   ('shares', 'Shares'), ('dollar_value', 'Value'), ('price', 'Price'),
                   ('stop', 'Stop'), ('mood', 'Market mood'), ('outcome', 'Outcome'),
                   ('reason', 'Reason'), ('label', 'Delivery')]
        if not rows:
            return '<p>' + esc(empty) + '</p>'
        return '<div class="orders-scroll"><table><thead><tr>' + ''.join('<th>' + esc(title) + '</th>' for _, title in columns) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(row[key]) + '</td>' for key, _ in columns) + '</tr>' for row in rows) + '</tbody></table></div>'
    content += card('Would-have-traded today', '<p class="badge"><strong>Dry run: not sent</strong></p>' + orders_table(view['would_trade_today'], 'No trades wanted yet today.'))
    content += card('Recent days', orders_table(view['order_history'], 'No trades wanted in recent days.'))
    if view['last_error']:
        content += card('Last error', '<p>' + esc(view['last_error']['message']) + '</p><small>' + esc(view['last_error']['time']) + '</small>')
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="30"><title>Trading bot status</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f3f6fa;color:#183044;margin:0;padding:24px}main{max-width:1000px;margin:auto;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}section{background:white;border:1px solid #dbe4ed;border-radius:14px;padding:24px;overflow-wrap:anywhere}h1,h2,h3{margin:0 0 12px}h2{font-size:18px;color:#486078}h1{font-size:30px}.badge{background:#e8f2fc;border-radius:8px;padding:6px 10px}small{color:#586b7d}.orders-scroll{overflow-x:auto}section:has(table){grid-column:1/-1;min-width:0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dbe4ed;min-width:100px}footer{max-width:1000px;margin:24px auto}@media(max-width:640px){main{grid-template-columns:1fr}body{padding:14px}}
</style></head><body><main>''' + content + '</main><footer>View only. This page cannot place trades.</footer></body></html>'


@app.get('/', response_class=HTMLResponse)
def index():
    return render(current_view())
