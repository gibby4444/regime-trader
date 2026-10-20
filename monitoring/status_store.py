"""Best-effort, bounded asynchronous status persistence for monitoring only."""
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class NullStatusStore:
    def update_status(self, data, force=False): pass
    def record_decision(self, symbol, status, detail, regime, confidence): pass
    def record_intended_order(self, **fields): pass
    def record_regime(self, regime, confidence): pass
    def record_event(self, kind, message): pass
    def close(self, timeout=2): pass


class StatusStore(NullStatusStore):
    def __init__(self, dsn=None, *, min_interval=30.0, clock=time.monotonic):
        self.enabled = False
        try:
            self._dsn = dsn if dsn is not None else os.environ.get('DATABASE_URL')
            if not self._dsn:
                return
            import psycopg
            self._driver = psycopg
            self._clock, self._interval = clock, min_interval
            self._last_status = None
            self._last_regime = None
            self._queue = queue.Queue(maxsize=500)
            self._lock = threading.Lock()
            self._stop = threading.Event()
            self._close_deadline = None
            self._thread = threading.Thread(target=self._worker, daemon=True, name='bot-status')
            self.enabled = True
            self._thread.start()
        except Exception:
            self.enabled = False
            log.debug('Status persistence unavailable')

    def _enqueue(self, sql, params):
        if not self.enabled:
            return False
        try:
            self._queue.put_nowait((sql, params))
            return True
        except Exception:
            return False

    def update_status(self, data, force=False):
        try:
            if not self.enabled:
                return
            with self._lock:
                now = self._clock()
                if not force and self._last_status is not None and now-self._last_status < self._interval:
                    return
                payload = dict(data, heartbeat=datetime.now(timezone.utc).isoformat())
                encoded = json.dumps(payload, default=str)
                if self._enqueue('INSERT INTO bot_status (id, updated_at, data) VALUES (1, now(), %s::jsonb) ON CONFLICT (id) DO UPDATE SET updated_at=EXCLUDED.updated_at, data=EXCLUDED.data', (encoded,)):
                    self._last_status = now
        except Exception:
            log.debug('Status update dropped')

    def record_decision(self, symbol, status, detail, regime, confidence):
        try:
            if not self.enabled:
                return
            self._enqueue('INSERT INTO bot_decisions (ts, symbol, status, detail, regime, confidence) VALUES (now(), %s, %s, %s, %s, %s)', (symbol, status, str(detail), regime, confidence))
        except Exception:
            log.debug('Decision update dropped')

    def record_intended_order(self, **fields):
        try:
            columns = ('symbol', 'side', 'original_qty', 'final_qty', 'price',
                       'order_type', 'stop_price', 'notional', 'regime', 'confidence',
                       'outcome', 'reason', 'dry_run', 'sent')
            self._enqueue('INSERT INTO bot_intended_orders (ts, ' + ', '.join(columns) +
                          ') VALUES (now(), ' + ', '.join(['%s'] * len(columns)) + ')',
                          tuple(fields.get(key) for key in columns))
        except Exception:
            log.debug('Intended order update dropped')

    def record_regime(self, regime, confidence):
        try:
            if not self.enabled:
                return
            with self._lock:
                if regime != self._last_regime and self._enqueue('INSERT INTO bot_regimes (ts, regime, confidence) VALUES (now(), %s, %s)', (regime, confidence)):
                    self._last_regime = regime
        except Exception:
            log.debug('Regime update dropped')

    def record_event(self, kind, message):
        try:
            if not self.enabled:
                return
            if kind in ('start', 'market_closed', 'error', 'stop'):
                self._enqueue('INSERT INTO bot_events (ts, kind, message) VALUES (now(), %s, %s)', (kind, str(message)))
        except Exception:
            log.debug('Event update dropped')

    def _worker(self):
        connection = None
        retry_at = 0
        try:
            while not self._stop.is_set() or not self._queue.empty():
                if self._close_deadline is not None and time.monotonic() >= self._close_deadline:
                    break
                try:
                    sql, params = self._queue.get(timeout=.05)
                except queue.Empty:
                    continue
                try:
                    if self._clock() < retry_at:
                        continue
                    if connection is None:
                        connection = self._driver.connect(self._dsn, connect_timeout=5, autocommit=True, options='-c statement_timeout=5000')
                        connection.execute('CREATE TABLE IF NOT EXISTS bot_status (id int primary key, updated_at timestamptz not null, data jsonb not null)')
                        connection.execute('CREATE TABLE IF NOT EXISTS bot_decisions (id bigserial primary key, ts timestamptz, symbol text, status text, detail text, regime text, confidence double precision)')
                        connection.execute('CREATE TABLE IF NOT EXISTS bot_regimes (id bigserial primary key, ts timestamptz, regime text, confidence double precision)')
                        connection.execute('CREATE TABLE IF NOT EXISTS bot_events (id bigserial primary key, ts timestamptz, kind text, message text)')
                        connection.execute('CREATE TABLE IF NOT EXISTS bot_intended_orders (id bigserial primary key, ts timestamptz, symbol text, side text, original_qty double precision, final_qty double precision, price double precision, order_type text, stop_price double precision, notional double precision, regime text, confidence double precision, outcome text, reason text, dry_run boolean, sent boolean)')
                        for table in ('bot_decisions', 'bot_regimes', 'bot_events', 'bot_intended_orders'):
                            connection.execute(f"DELETE FROM {table} WHERE ts < now() - interval '30 days'")
                    connection.execute(sql, params)
                except Exception:
                    log.warning('Status database unavailable; updates temporarily dropped')
                    if connection is not None:
                        try:
                            connection.close()
                        except Exception:
                            pass
                    connection = None
                    retry_at = self._clock()+60
                finally:
                    self._queue.task_done()
        except Exception:
            log.debug('Status worker stopped')
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def close(self, timeout=2):
        try:
            if self.enabled:
                self.enabled = False
                self._close_deadline = time.monotonic() + max(0, timeout)
                self._stop.set()
                self._thread.join(timeout=max(0, timeout))
        except Exception:
            log.debug('Status close failed')
