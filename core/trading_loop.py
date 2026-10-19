"""Live lifecycle with injected broker, streams, wall clock and sleeper.

Callbacks only enqueue events; all portfolio/order mutations run on the owner thread.
"""
from __future__ import annotations

import copy
import json
import logging
import signal
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from queue import Queue, Empty

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from core.risk_manager import PortfolioState, TradingState
from monitoring.dashboard import DashboardState
from monitoring.alerts import Alert, AlertSeverity

log = logging.getLogger(__name__)
WEEK = timedelta(days=7)


def retry_api(operation, sleep, attempts=3):
    """Three attempts for reads; writes must use broker idempotency/reconciliation."""
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt == attempts - 1:
                raise
            sleep(2 ** attempt)


class TradingLoop:
    def __init__(self, config, *, client, feed, engine, features, strategy, risk,
                 tracker, executor, dashboard, alerts, train, now=None,
                 sleep=time.sleep, dry_run=False, model_path='models/hmm.pkl',
                 snapshot_path='state_snapshot.json', output=print, warm_history=None):
        self.config = config
        self.warm_history = warm_history
        self.client, self.feed, self.engine = client, feed, engine
        self.features, self.strategy, self.risk = features, strategy, risk
        self.tracker, self.executor = tracker, executor
        self.dashboard, self.alerts, self.train = dashboard, alerts, train
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.sleep, self.dry_run, self.output = sleep, dry_run, output
        self.model_path, self.snapshot_path = Path(model_path), Path(snapshot_path)
        self.symbols = config['broker']['symbols']
        self.timeframe = config['broker'].get('live_timeframe', '5Min')
        self.portfolio = tracker.portfolio
        self.queue = Queue()
        self.running = self.online = False
        self.stopping = False
        self.feed_ok = True
        self.last_regime = None
        self.last_prediction = None
        self.trained_at = None
        self.bars = {}
        self.last_seen = {}
        self.exit_pending = set()
        self.target_weights = {}
        self.counts = dict(bars=0, approved=0, modified=0, rejected=0)
        self.started_resources = []
        self.executor.should_stop = lambda: self.stopping

    def read(self, operation):
        from broker.alpaca_client import AlpacaClient
        # The concrete adapter already retries reads/connect exactly three times.
        if isinstance(getattr(operation, '__self__', None), AlpacaClient):
            return operation()
        return retry_api(operation, self.sleep)

    def entries_allowed(self):
        return (not self.stopping and not Path(self.risk.config.lock_path).exists()
                and self.risk.state not in (TradingState.HALTED, TradingState.KILLED)
                and self.risk.can_trade())

    def ensure_model(self):
        if self.trained_at is None and self.model_path.exists():
            self.trained_at = datetime.fromtimestamp(self.model_path.stat().st_mtime, timezone.utc)
            metadata_path = self.model_path.with_suffix(self.model_path.suffix + '.json')
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get('training_date'):
                    self.trained_at = pd.Timestamp(metadata['training_date']).to_pydatetime()
            if self.now() - self.trained_at <= WEEK:
                self.engine = HMMEngine.load(self.model_path)
        if self.trained_at is None or self.now() - self.trained_at > WEEK:
            # Train callback supplies daily, completed history, independently of live bars.
            self.train(self.engine)
            self.last_regime = None
            self.last_prediction = None
            self.trained_at = self.now()
            self.engine.metadata['training_date'] = self.trained_at.isoformat()
            self.engine.save(self.model_path)
        self.strategy.update_regimes(self.engine.regime_info)

    def startup(self):
        self.read(self.client.connect)
        account = self.read(self.client.get_account)
        if account.get('trading_blocked') or account.get('account_blocked'):
            raise PermissionError('Alpaca account is blocked')
        if float(account['equity']) <= 0:
            raise ValueError('Account equity must be positive')
        clock = self.read(self.client.get_clock)
        if not clock['is_open']:
            self.output(f"Market closed. Next open: {clock.get('next_open', 'unknown')}")
            return False
        self.ensure_model()
        self.portfolio.equity = float(account['equity'])
        self.portfolio.cash = float(account['cash'])
        self.portfolio.buying_power = float(account['buying_power'])
        self.risk.update_equity(self.portfolio.equity, self.now())
        self.read(self.tracker.sync)
        self.restore_snapshot()
        self.check_breaker(execute=False)
        self.output(f'Entries enabled: {self.entries_allowed()}')
        self.started_resources.append(self.tracker.stop)
        self.tracker.start()
        if self.warm_history is not None:
            self.bars = self.read(self.warm_history)
        self.feed_started_at = self.now()
        self.feed.subscribe_bars(self.symbols, self.timeframe, self.enqueue_bar)
        self.started_resources.append(self.feed.stop_stream)
        self.feed.start_stream()
        self.started_resources.append(self.dashboard.stop)
        self.dashboard.start()
        self.refresh()
        self.output(f'System online: equity={self.portfolio.equity:.2f}, dry_run={self.dry_run}')
        log.info('System online')
        self.online = True
        return True

    def enqueue_bar(self, symbol, bar):
        self.queue.put(('bar', symbol, bar))

    def feed_dropped(self):
        self.feed_ok = False
        log.warning('Data feed dropped: signals paused; broker stops remain active')

    def feed_recovered(self):
        """Resume only after the stream owner confirms a healthy reconnection."""
        self.feed_ok = True
        log.info('Data feed recovered')

    def check_breaker(self, execute=True):
        self.risk.update_equity(self.portfolio.equity, self.now())
        if self.portfolio.daily_pnl:
            self.risk.day_start_equity = self.portfolio.equity - self.portfolio.daily_pnl
        if self.portfolio.weekly_pnl:
            self.risk.week_start_equity = self.portfolio.equity - self.portfolio.weekly_pnl
        self.risk.evaluate_state(self.portfolio.equity, self.portfolio.positions,
                                 self.last_regime.label if self.last_regime else None)
        # These exits deliberately bypass entry vetoes (Step 5).
        if execute and not self.stopping and self.risk.state in (TradingState.HALTED, TradingState.KILLED):
            for symbol in self.portfolio.positions:
                if not self.dry_run and symbol not in self.exit_pending:
                    self.executor.close_position(symbol)
                    self.exit_pending.add(symbol)
        self.exit_pending.intersection_update(self.portfolio.positions)

    def process_bar(self, symbol, bar):
        if self.stopping:
            return
        stamp = pd.Timestamp(bar.name)
        if symbol in self.bars and stamp <= self.bars[symbol].index[-1]:
            return
        previous = self.last_seen.get(symbol)
        if previous is not None and self.now()-previous > timedelta(minutes=10):
            self.feed_dropped()
        self.last_seen[symbol] = self.now()
        row = pd.DataFrame([bar.to_dict()], index=pd.DatetimeIndex([stamp]))
        old = self.bars.get(symbol)
        self.bars[symbol] = pd.concat([old, row]) if old is not None else row
        self.bars[symbol] = self.bars[symbol].tail(2000)
        self.counts['bars'] += 1
        self.tracker.update_prices({symbol: float(bar['close'])})
        allowed_before = self.entries_allowed()
        self.check_breaker()
        # Only predict once per completed universe bar; never use another symbol's future row.
        ready = all(s in self.bars and self.bars[s].index[-1] >= stamp for s in self.symbols)
        signals = []
        history = {s: b.loc[:stamp] for s, b in self.bars.items()}
        if ready and self.feed_ok and (self.last_prediction is None or stamp > self.last_prediction):
            self.last_prediction = stamp
            frame = self.features.build_features(history[self.symbols[0]])
            if not frame.empty and frame.index[-1] == stamp:
                # An HMM failure cannot partially advance the forward or stability filters.
                backup = copy.deepcopy(self.engine)
                try:
                    self.last_regime = self.engine.update(frame.iloc[-1], stamp)
                except Exception:
                    self.engine = backup
                    log.exception('HMM error: holding current regime')
                if self.last_regime is not None:
                    signals = self.strategy.generate_signals(self.symbols, history, self.last_regime,
                                                              self.engine.is_flickering())
            if self.feed_ok:
                for proposed in signals:
                    if not allowed_before or not self.entries_allowed():
                        log.info('rejected %s: trading halted', proposed.symbol)
                        self.counts['rejected'] += 1
                        continue
                    decision = self.risk.validate_signal(proposed, self.portfolio)
                    status = ('modified' if decision.modifications else 'approved') if decision.approved else 'rejected'
                    self.counts[status] += 1
                    self.target_weights.update(decision.adjusted_weights)
                    log.info('%s %s: %s', status, proposed.symbol,
                             decision.modifications or decision.rejection_reason or 'risk approved')
                    if decision.approved and self.entries_allowed() and not self.dry_run:
                        chosen = decision.modified_signal or proposed
                        self.executor.submit_order(chosen, risk_decision=decision)
        # Stop candidates come from the current regime even while signals are paused/halted.
        if self.last_regime is not None:
            stop_signals = signals or self.strategy.generate_signals(list(history), history, self.last_regime,
                                                                      self.engine.is_flickering())
            for sig in stop_signals:
                pos = self.tracker.positions.get(sig.symbol)
                if pos and sig.stop_loss is not None and (pos.stop is None or
                        (sig.stop_loss > pos.stop if pos.qty > 0 else sig.stop_loss < pos.stop)):
                    if not self.dry_run and not self.stopping:
                        self.executor.modify_stop(sig.symbol, sig.stop_loss)
                        pos.stop = sig.stop_loss
        self.check_breaker()
        self.refresh()
        if self.trained_at is not None and self.now() - self.trained_at > WEEK:
            self.ensure_model()

    def refresh(self):
        self.dashboard.update(DashboardState(
            regime=self.last_regime.label if self.last_regime else 'unknown',
            confidence=self.last_regime.probability if self.last_regime else 0,
            equity=self.portfolio.equity, daily_pnl=self.portfolio.daily_pnl,
            trading_state=self.risk.state.value,
            drawdown=self.risk.drawdown_from_peak(self.portfolio.equity),
            target_weights=dict(self.target_weights),
            positions=[dict(symbol=s, **p) for s, p in self.portfolio.positions.items()]))

    def save_snapshot(self):
        risk = {k: getattr(self.risk, k) for k in
                ('peak_equity', 'day_start_equity', 'week_start_equity', 'trades_today',
                 '_last_equity', '_acknowledged_peak', '_day', '_week')}
        payload = dict(version=1, model_trained_at=self.trained_at, risk=risk,
                       breaker=sorted(self.risk.breaker.active), lock_written=self.risk.breaker._lock_written,
                       state=self.risk.state.value,
                       counts=self.counts, positions={s: asdict(p) for s, p in self.tracker.positions.items()})
        if self.last_regime is not None:
            payload['regime'] = asdict(self.last_regime)
            payload['alpha'] = self.engine._log_alpha
            payload['stability'] = vars(self.engine.stability)
            payload['last_prediction'] = self.last_prediction
        def encode(value):
            if isinstance(value, (datetime, date, pd.Timestamp)):
                return value.isoformat()
            if isinstance(value, (np.ndarray, deque, tuple)):
                return list(value)
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.snapshot_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, default=encode, indent=2))
        temporary.replace(self.snapshot_path)

    def restore_snapshot(self):
        if not self.snapshot_path.exists():
            return
        data = json.loads(self.snapshot_path.read_text())
        for key, value in data['risk'].items():
            if key == '_day' and value:
                value = date.fromisoformat(value)
            if key == '_week' and value:
                value = tuple(value)
            setattr(self.risk, key, value)
        self.risk.breaker.active.update(data['breaker'])
        self.risk.breaker._lock_written = data.get('lock_written', False)
        self.risk.state = TradingState(data['state'])
        if self.risk._day == self.now().date() and self.risk.day_start_equity is not None:
            self.portfolio.daily_pnl = self.portfolio.equity-self.risk.day_start_equity
            self.portfolio.daily_trades = self.risk.trades_today
        if self.risk._week == self.now().isocalendar()[:2] and self.risk.week_start_equity is not None:
            self.portfolio.weekly_pnl = self.portfolio.equity-self.risk.week_start_equity
        # Broker owns quantities/prices. Recover only metadata, never resurrect positions.
        for symbol, saved in data.get('positions', {}).items():
            pos = self.tracker.positions.get(symbol)
            if pos:
                for key in ('regime_at_entry', 'trade_id', 'current_regime'):
                    setattr(pos, key, saved.get(key))
                if saved.get('entry_time'):
                    pos.entry_time = datetime.fromisoformat(saved['entry_time'])
                # Broker stop is authoritative; a snapshot cannot claim an unplaced stop.
        if data.get('regime') and self.trained_at is not None and data.get('model_trained_at') == self.trained_at.isoformat():
            state = data['regime']
            state['state_probabilities'] = np.asarray(state['state_probabilities'])
            state['timestamp'] = pd.Timestamp(state['timestamp']) if state['timestamp'] else None
            self.last_regime = RegimeState(**state)
            self.engine._log_alpha = np.asarray(data['alpha']) if data['alpha'] is not None else None
            for key, value in data['stability'].items():
                if key == '_raw_history':
                    value = deque(value, maxlen=self.engine.config.flicker_window + 1)
                setattr(self.engine.stability, key, value)
            self.engine._last_state = self.last_regime
            self.last_prediction = pd.Timestamp(data['last_prediction']) if data['last_prediction'] else None
        log.info('Recovered state snapshot')

    def request_stop(self, signum=None, frame=None):
        # Signal handlers only set flags; shutdown never calls any order interface.
        self.stopping = True
        self.running = False

    def shutdown(self):
        self.request_stop()
        for close in reversed(self.started_resources):
            try:
                close()
            except Exception:
                log.exception('Error closing stream/dashboard')
        self.started_resources.clear()
        self.save_snapshot()
        self.output(f'Session summary: {self.counts}; equity={self.portfolio.equity:.2f}')

    def run(self, *, install_signals=True):
        previous = {}
        try:
            if install_signals:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    previous[sig] = signal.signal(sig, self.request_stop)
            if not self.startup():
                return
            self.running = not self.stopping
            while self.running:
                try:
                    _, symbol, bar = self.queue.get_nowait()
                except Empty:
                    clock = self.read(self.client.get_clock)
                    if not clock['is_open']:
                        self.output(f"Market closed. Next open: {clock.get('next_open', 'unknown')}")
                        break
                    thread = getattr(self.feed, '_thread', None)
                    stale = any(self.now()-self.last_seen.get(s, self.feed_started_at) > timedelta(minutes=10) for s in self.symbols)
                    if (thread is not None and not thread.is_alive()) or stale:
                        self.feed_dropped()
                    self.sleep(1)
                    continue
                clock = self.read(self.client.get_clock)
                if not clock['is_open']:
                    self.output(f"Market closed. Next open: {clock.get('next_open', 'unknown')}")
                    break
                self.process_bar(symbol, bar)
        except InterruptedError:
            if not self.stopping:
                raise
        except Exception as exc:
            log.exception('Unhandled trading loop exception')
            try:
                self.save_snapshot()
            finally:
                self.alerts.send(Alert('loop_error', AlertSeverity.CRITICAL,
                                       'Trading loop stopped', str(exc), self.now()))
            raise
        finally:
            try:
                self.shutdown()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
