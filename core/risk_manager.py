"""Broker-independent risk vetoes. Monetary P&L inputs are in account currency."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class TradingState(str, Enum):
    NORMAL = 'normal'
    REDUCED = 'reduced'
    HALTED = 'halted'
    KILLED = 'killed'

@dataclass
class RiskConfig:
    max_risk_per_trade: float = .01
    max_exposure: float = .80
    max_leverage: float = 1.25
    max_single_position: float = .15
    max_sector_exposure: float = .30
    max_concurrent: int = 5
    max_daily_trades: int = 20
    daily_dd_reduce: float = .02
    daily_dd_halt: float = .03
    weekly_dd_reduce: float = .05
    weekly_dd_halt: float = .07
    max_dd_from_peak: float = .10
    lock_path: str = 'trading_halted.lock'
    min_position: float = 100
    gap_multiple: float = 3
    max_gap_risk: float = .02
    max_spread: float = .005
    duplicate_seconds: int = 60
    correlation_window: int = 60
    correlation_reduce: float = .7
    correlation_reject: float = .85
    min_confidence: float = .55
    flicker_threshold: float = 4

    @classmethod
    def from_dict(cls, cfg):
        return cls(**cfg)

@dataclass
class PortfolioState:
    equity: float
    cash: float
    buying_power: float
    positions: dict[str, dict] = field(default_factory=dict)
    daily_pnl: float = 0
    weekly_pnl: float = 0
    peak_equity: float = 0
    drawdown: float = 0
    circuit_breaker_status: str = 'normal'
    flicker_rate: float = 0
    timestamp: datetime | None = None
    daily_trades: int = 0
    returns: Any = None

@dataclass
class RiskDecision:
    approved: bool
    state: TradingState = TradingState.NORMAL
    adjusted_weights: dict[str, float] = field(default_factory=dict)
    size_multiplier: float = 1
    reasons: list[str] = field(default_factory=list)
    modified_signal: Any = None
    rejection_reason: str | None = None
    modifications: list[str] = field(default_factory=list)
    quantity: int = 0
    close_positions: list[str] = field(default_factory=list)

class CircuitBreaker:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.active: set[str] = set()
        self.history: list[dict] = []
        self._lock_written = False

    def reset_daily(self):
        self.active -= {'daily_reduce', 'daily_halt'}

    def reset_weekly(self):
        self.active -= {'weekly_reduce', 'weekly_halt'}

    def get_history(self):
        return list(self.history)

    def check(self, daily_dd=0., weekly_dd=0., peak_dd=0., equity=0., positions=(), regime=None):
        path = Path(self.config.lock_path)
        if self._lock_written and not path.exists():
            self.active.discard('peak_halt')
            self._lock_written = False
            # Manual deletion acknowledges the previous peak loss.
            peak_dd = 0
        for name, dd, limit in (
            ('daily_reduce', daily_dd, self.config.daily_dd_reduce),
            ('daily_halt', daily_dd, self.config.daily_dd_halt),
            ('weekly_reduce', weekly_dd, self.config.weekly_dd_reduce),
            ('weekly_halt', weekly_dd, self.config.weekly_dd_halt),
            ('peak_halt', peak_dd, self.config.max_dd_from_peak),
        ):
            if dd > limit + 1e-12 and name not in self.active:
                self.active.add(name)
                if name == 'peak_halt':
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('Trading halted: manually delete after review.\n')
                    self._lock_written = True
                event = dict(breaker_type=name, drawdown=dd, equity=equity,
                             positions_closed=list(positions) if 'halt' in name else [], regime=regime)
                self.history.append(event)
                logger.warning('circuit_breaker %s', event, extra={'breaker_event': event})
        if path.exists():
            self._lock_written = True
            return TradingState.KILLED
        if any('halt' in x for x in self.active):
            return TradingState.HALTED
        return TradingState.REDUCED if self.active else TradingState.NORMAL

    def update(self, pnl, **kwargs):
        """Accept an actual account snapshot, or explicit drawdown fractions.

        ``PortfolioState`` P&L is currency P&L relative to the period's starting
        equity, including open-position marks. Model labels are log context only.
        """
        if isinstance(pnl, PortfolioState):
            def loss(amount):
                anchor = pnl.equity - amount
                return max(0., -amount / anchor) if anchor > 0 else 0.
            peak = max(pnl.peak_equity, pnl.equity)
            return self.check(daily_dd=loss(pnl.daily_pnl), weekly_dd=loss(pnl.weekly_pnl),
                              peak_dd=max(pnl.drawdown, 1-pnl.equity/peak) if peak > 0 else 0.,
                              equity=pnl.equity, positions=pnl.positions, **kwargs)
        return self.check(**pnl, **kwargs)

class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.breaker = CircuitBreaker(self.config)
        self.peak_equity = self.day_start_equity = self.week_start_equity = None
        self.trades_today = 0
        self.state = TradingState.NORMAL
        self._day = self._week = None
        self._last_equity = None
        self._acknowledged_peak = 0.
        self._duplicates: dict[tuple, datetime] = {}

    def update_equity(self, equity, timestamp):
        day, week = timestamp.date(), timestamp.isocalendar()[:2]
        anchor = self._last_equity if self._last_equity is not None else equity
        if day != self._day:
            self.reset_daily()
            self.day_start_equity = anchor
            self._day = day
        if week != self._week:
            self.breaker.reset_weekly()
            self.week_start_equity = anchor
            self._week = week
        if self.breaker._lock_written and not Path(self.config.lock_path).exists():
            self._acknowledged_peak = max(self._acknowledged_peak, self.peak_equity or equity)
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity or equity, equity)
        self._last_equity = equity

    def daily_drawdown(self, equity):
        return max(0., 1-equity/(self.day_start_equity or equity))

    def weekly_drawdown(self, equity):
        return max(0., 1-equity/(self.week_start_equity or equity))

    def drawdown_from_peak(self, equity):
        return max(0., 1-equity/(self.peak_equity or equity))

    def evaluate_state(self, equity, positions=(), regime=None):
        self.state = self.breaker.check(self.daily_drawdown(equity), self.weekly_drawdown(equity),
                                        self.drawdown_from_peak(equity), equity, positions, regime)
        return self.state

    def position_size(self, equity, entry_price, stop_price, regime_max=1., overnight=False):
        if not stop_price or not all(math.isfinite(x) and x > 0 for x in (equity, entry_price, stop_price)) or stop_price >= entry_price:
            return 0
        distance = abs(entry_price-stop_price)
        qty = min(equity*self.config.max_risk_per_trade/distance,
                  equity*regime_max/entry_price, equity*self.config.max_single_position/entry_price)
        if overnight:
            qty = min(qty, equity*self.config.max_gap_risk/(self.config.gap_multiple*distance))
        qty = int(qty)
        return qty if qty*entry_price >= self.config.min_position else 0

    def leverage(self, signal, portfolio):
        m = signal.metadata
        if (m.get('vol_regime') != 'low' or m.get('uncertain') or signal.confidence < self.config.min_confidence
            or self.state != TradingState.NORMAL or portfolio.circuit_breaker_status != 'normal'
            or len(portfolio.positions) >= 3 or portfolio.flicker_rate > self.config.flicker_threshold
            or m.get('is_flickering')):
            return 1.
        return max(1., min(signal.leverage, self.config.max_leverage))

    def validate_signal(self, signal, portfolio: PortfolioState, quote=None):
        if not math.isfinite(portfolio.equity) or portfolio.equity <= 0:
            logger.info('order_rejected invalid_equity', extra={'risk_rejection': {'reason': 'invalid_equity'}})
            return RiskDecision(False, rejection_reason='invalid_equity')
        ts = portfolio.timestamp or signal.timestamp or datetime.now()
        self.update_equity(portfolio.equity, ts)
        # Explicit actual account P&L can be supplied without historical marks.
        if portfolio.daily_pnl:
            self.day_start_equity = portfolio.equity-portfolio.daily_pnl
        if portfolio.weekly_pnl:
            self.week_start_equity = portfolio.equity-portfolio.weekly_pnl
        if portfolio.peak_equity > self._acknowledged_peak:
            self.peak_equity = max(self.peak_equity, portfolio.peak_equity)
        self.evaluate_state(portfolio.equity, portfolio.positions, signal.regime_name)
        def reject(reason):
            logger.info('order_rejected %s', dict(symbol=signal.symbol, reason=reason),
                        extra={'risk_rejection': dict(symbol=signal.symbol, reason=reason)})
            return RiskDecision(False, self.state, rejection_reason=reason, reasons=[reason],
                                close_positions=list(portfolio.positions) if self.state in (TradingState.HALTED, TradingState.KILLED) else [])
        if not self.can_trade() or portfolio.daily_trades >= self.config.max_daily_trades:
            return reject('trading_halted_or_daily_cap')
        if signal.stop_loss is None or not math.isfinite(signal.stop_loss) or signal.stop_loss <= 0:
            return reject('missing_stop_loss')
        if signal.entry_price <= 0 or not math.isfinite(signal.entry_price):
            return reject('invalid_price')
        if signal.stop_loss >= signal.entry_price:
            return reject('stop_not_below_entry')
        quote = quote or signal.metadata.get('quote', {})
        if not quote.get('tradeable', True):
            return reject('not_tradeable')
        bid, ask = quote.get('bid', signal.entry_price), quote.get('ask', signal.entry_price)
        if not all(math.isfinite(x) for x in (bid, ask)) or bid <= 0 or ask < bid or (ask-bid)/((ask+bid)/2) >= self.config.max_spread:
            return reject('spread')
        key = (signal.symbol, signal.direction)
        if key in self._duplicates and (ts-self._duplicates[key]).total_seconds() < self.config.duplicate_seconds:
            return reject('duplicate')
        positions = portfolio.positions
        if signal.symbol not in positions and len(positions) >= self.config.max_concurrent:
            return reject('concurrent_cap')
        lev = self.leverage(signal, portfolio)
        mult = .5 if self.state == TradingState.REDUCED else 1.
        weight = signal.metadata.get('target_weight', signal.position_size_pct*signal.leverage)
        # Recompute the strategy weight if leverage was vetoed.
        weight *= lev/max(signal.leverage, 1.)
        qty = self.position_size(portfolio.equity, signal.entry_price, signal.stop_loss,
                                 weight, signal.metadata.get('overnight', True))
        qty = int(qty*mult)
        changes = []
        if lev != signal.leverage:
            changes.append('leverage_forced_1x')
        if mult < 1:
            changes.append('breaker_size_halved')
        if qty*signal.entry_price < portfolio.equity*weight:
            changes.append('position_risk_caps')
        corr = 0.
        if portfolio.returns is not None and signal.symbol in portfolio.returns:
            frame = portfolio.returns.tail(self.config.correlation_window)
            for sym in positions:
                if sym != signal.symbol and sym in frame and len(frame[[sym, signal.symbol]].dropna()) >= self.config.correlation_window:
                    value = frame[sym].corr(frame[signal.symbol])
                    if math.isfinite(value):
                        corr = max(corr, value)
        if corr > self.config.correlation_reject:
            return reject('correlation')
        if corr > self.config.correlation_reduce:
            qty //= 2
            changes.append('correlation_halved')
        sector = signal.metadata.get('sector', 'unknown')
        other = sum(abs(p['market_value']) for s,p in positions.items() if s != signal.symbol)
        sector_value = sum(abs(p['market_value']) for s,p in positions.items() if s != signal.symbol and p.get('sector', 'unknown') == sector)
        cap = min(portfolio.equity*self.config.max_exposure-other,
                  portfolio.equity*self.config.max_sector_exposure-sector_value,
                  portfolio.equity*self.config.max_leverage-other)
        capped = min(qty, max(0, int(cap/signal.entry_price)))
        if capped < qty:
            changes.append('portfolio_caps')
        qty = capped
        existing = positions.get(signal.symbol, {}).get('market_value', 0)
        if qty*signal.entry_price-existing > portfolio.buying_power+1e-8:
            return reject('buying_power')
        if qty*signal.entry_price < self.config.min_position:
            return reject('minimum_size_or_portfolio_cap')
        modified = replace(signal, leverage=lev, position_size_pct=qty*signal.entry_price/portfolio.equity,
                           metadata={**signal.metadata, 'quantity': qty, 'target_weight': qty*signal.entry_price/portfolio.equity})
        self._duplicates[key] = ts
        return RiskDecision(True, self.state, {signal.symbol: qty*signal.entry_price/portfolio.equity}, mult,
                            modified_signal=modified, modifications=changes, quantity=qty)

    def apply_limits(self, target, equity, open_positions):
        self.evaluate_state(equity)
        mult = .5 if self.state == TradingState.REDUCED else 1.
        weights = {s: min(w*mult, self.config.max_single_position) for s,w in list(target.weights.items())[:max(0,self.config.max_concurrent-open_positions)]}
        total = sum(weights.values())
        if total > self.config.max_exposure:
            weights = {s:w*self.config.max_exposure/total for s,w in weights.items()}
        return RiskDecision(self.can_trade(), self.state, weights if self.can_trade() else {}, mult)

    def can_trade(self):
        return (not Path(self.config.lock_path).exists() and self.state not in (TradingState.HALTED, TradingState.KILLED)
                and self.trades_today < self.config.max_daily_trades)

    def record_trade(self):
        self.trades_today += 1

    def reset_daily(self):
        self.trades_today = 0
        self.day_start_equity = None
        self.breaker.reset_daily()
