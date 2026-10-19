"""Risk management: the last gate before any order, with absolute veto.

The risk manager is deliberately independent of the HMM (defense in depth). It
reads only plain signal fields (symbol, entry, stop, requested weight, leverage)
and string metadata labels (``vol_regime``, ``uncertain``), never the model.

Layers, applied by :meth:`RiskManager.validate_signal`:

1. Circuit breakers on actual P&L (:class:`CircuitBreaker`): daily/weekly
   reduce (sizes x0.5) and halt (close all, no new risk) for the rest of the
   day/week; a peak-drawdown breach halts everything and writes a lock file that
   must be deleted manually.
2. Order validation: stop loss present, tradeable, bid-ask spread, buying power,
   duplicate orders, daily trade count, concurrent positions.
3. Position sizing: 1% risk per trade from the stop distance, overnight gap
   risk, regime cap, single-position cap, minimum position value.
4. Correlation with held positions (reduce or reject).
5. Portfolio caps: sector exposure and gross exposure = ``max_exposure`` x
   allowed leverage, where leverage is 1.0x unless the signal is a low-vol,
   confident signal with fewer than ``leverage_max_positions`` open positions,
   no active breaker, and low flicker.

Risk-reducing orders (target below the current position) are always approved.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from core.regime_strategies import Signal

logger = logging.getLogger("regime_trader.risk")

DEFAULT_SECTORS: dict[str, str] = {
    "SPY": "broad_market",
    "QQQ": "broad_market",
    "AAPL": "technology",
    "MSFT": "technology",
    "NVDA": "technology",
    "AMD": "technology",
    "GOOGL": "communication_services",
    "META": "communication_services",
    "AMZN": "consumer_discretionary",
    "TSLA": "consumer_discretionary",
}


class TradingState(str, Enum):
    """Risk-driven trading permission level (most severe active breaker)."""

    NORMAL = "normal"
    REDUCED = "reduced"
    HALTED = "halted"
    LOCKED = "locked"


class BreakerType(str, Enum):
    DAILY_REDUCE = "daily_reduce"
    DAILY_HALT = "daily_halt"
    WEEKLY_REDUCE = "weekly_reduce"
    WEEKLY_HALT = "weekly_halt"
    PEAK_HALT = "peak_halt"


@dataclass
class RiskConfig:
    """Risk parameters mirroring the ``risk`` section of settings.yaml."""

    max_risk_per_trade: float = 0.01
    max_exposure: float = 0.80
    max_leverage: float = 1.25
    default_leverage: float = 1.0
    max_single_position: float = 0.15
    max_sector_exposure: float = 0.30
    max_concurrent: int = 5
    max_daily_trades: int = 20
    daily_dd_reduce: float = 0.02
    daily_dd_halt: float = 0.03
    weekly_dd_reduce: float = 0.05
    weekly_dd_halt: float = 0.07
    max_dd_from_peak: float = 0.10
    reduce_size_mult: float = 0.50
    min_position_value: float = 100.0
    gap_stop_multiple: float = 3.0
    max_gap_loss: float = 0.02
    max_spread_pct: float = 0.005
    duplicate_window_seconds: float = 60.0
    correlation_window: int = 60
    correlation_reduce: float = 0.70
    correlation_reject: float = 0.85
    correlation_reduce_mult: float = 0.50
    leverage_max_positions: int = 3
    max_flicker_rate: int = 4
    regime_max_position: dict[str, float] = field(
        default_factory=lambda: {"low": 0.15, "mid": 0.15, "high": 0.10}
    )
    lock_file: str | None = "trading_halted.lock"
    sectors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SECTORS))

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> RiskConfig:
        """Build a config from the parsed ``risk`` settings section."""
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown risk config keys: {sorted(unknown)}")
        return cls(**cfg)

    def sector(self, symbol: str) -> str:
        return self.sectors.get(symbol.upper(), f"unclassified:{symbol.upper()}")


@dataclass(frozen=True)
class PositionInfo:
    """One open position as seen by the risk manager."""

    symbol: str
    qty: float
    market_price: float
    stop_loss: float | None = None

    @property
    def market_value(self) -> float:
        return self.qty * self.market_price


@dataclass
class PortfolioState:
    """Snapshot handed to the risk manager.

    ``daily_pnl`` / ``weekly_pnl`` are currency P&L since the start of the
    day/week (anchored at ``day_start_equity`` / ``week_start_equity``);
    ``drawdown`` is the positive fraction below ``peak_equity``. ``quotes`` maps
    symbol -> (bid, ask); ``returns`` holds recent per-symbol bar returns for the
    correlation check. Missing quotes/tradeable entries skip those checks.
    """

    equity: float
    cash: float
    buying_power: float
    positions: dict[str, PositionInfo] = field(default_factory=dict)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    day_start_equity: float | None = None
    week_start_equity: float | None = None
    peak_equity: float | None = None
    drawdown: float = 0.0
    circuit_breaker_status: TradingState = TradingState.NORMAL
    flicker_rate: int = 0
    timestamp: pd.Timestamp | None = None
    regime: str | None = None
    quotes: dict[str, tuple[float, float]] = field(default_factory=dict)
    tradeable: dict[str, bool] = field(default_factory=dict)
    returns: pd.DataFrame | None = None

    @property
    def daily_drawdown(self) -> float:
        base = self.day_start_equity or self.equity
        return max(0.0, -self.daily_pnl / base) if base > 0 else 0.0

    @property
    def weekly_drawdown(self) -> float:
        base = self.week_start_equity or self.equity
        return max(0.0, -self.weekly_pnl / base) if base > 0 else 0.0

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def open_positions(self) -> int:
        return sum(1 for p in self.positions.values() if p.qty != 0)


@dataclass(frozen=True)
class RiskDecision:
    """Outcome of :meth:`RiskManager.validate_signal`.

    ``modified_signal`` carries the approved size in ``metadata["target_weight"]``
    (fraction of equity) and ``metadata["target_value"]``, with ``leverage`` set to
    the allowed leverage.
    """

    approved: bool
    modified_signal: Signal | None
    rejection_reason: str | None = None
    modifications: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BreakerStatus:
    """Result of :meth:`CircuitBreaker.check`."""

    state: TradingState
    active: tuple[BreakerType, ...]
    size_multiplier: float
    close_all: bool

    @property
    def any_active(self) -> bool:
        return bool(self.active)


class CircuitBreaker:
    """Drawdown circuit breakers on actual P&L.

    Daily breakers persist until :meth:`reset_daily`, weekly ones until
    :meth:`reset_weekly`. The peak breaker writes ``lock_path`` (if set) and stays
    tripped until that file is deleted; with ``lock_path=None`` it stays tripped
    until :meth:`release_lock`.
    """

    def __init__(self, config: RiskConfig, lock_path: str | Path | None = None) -> None:
        self.config = config
        self.lock_path = Path(lock_path) if lock_path is not None else None
        self._active: set[BreakerType] = set()
        self._history: list[dict[str, Any]] = []
        self._locked = self.lock_path is not None and self.lock_path.exists()

    def update(
        self,
        pnl: PortfolioState,
        positions_closed: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate breakers on a P&L snapshot; return newly triggered events.

        Each event records breaker type, actual drawdown, equity, positions that
        must be closed, and the HMM regime at the time (``pnl.regime``).
        """
        c = self.config
        candidates = [
            (BreakerType.PEAK_HALT, pnl.drawdown, c.max_dd_from_peak),
            (BreakerType.DAILY_HALT, pnl.daily_drawdown, c.daily_dd_halt),
            (BreakerType.WEEKLY_HALT, pnl.weekly_drawdown, c.weekly_dd_halt),
            (BreakerType.DAILY_REDUCE, pnl.daily_drawdown, c.daily_dd_reduce),
            (BreakerType.WEEKLY_REDUCE, pnl.weekly_drawdown, c.weekly_dd_reduce),
        ]
        closing = positions_closed if positions_closed is not None else [
            s for s, p in pnl.positions.items() if p.qty != 0
        ]
        events = []
        for kind, dd, limit in candidates:
            if dd > limit + 1e-12 and kind not in self._active:
                self._active.add(kind)
                halt = kind in (BreakerType.PEAK_HALT, BreakerType.DAILY_HALT, BreakerType.WEEKLY_HALT)
                event = {
                    "timestamp": pnl.timestamp,
                    "breaker": kind.value,
                    "drawdown": float(dd),
                    "threshold": float(limit),
                    "equity": float(pnl.equity),
                    "positions_closed": list(closing) if halt else [],
                    "regime": pnl.regime,
                }
                if kind is BreakerType.PEAK_HALT:
                    self._lock(event)
                self._history.append(event)
                logger.warning("Circuit breaker triggered: %s", json.dumps(event, default=str))
                events.append(event)
        return events

    def _lock(self, event: dict[str, Any]) -> None:
        self._locked = True
        if self.lock_path is not None:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_path.write_text(json.dumps(
                {**event, "message": "Trading halted: drawdown from peak breached. Delete this file to resume."},
                indent=2, default=str,
            ))

    @property
    def locked(self) -> bool:
        if self.lock_path is not None:
            self._locked = self.lock_path.exists()
            if not self._locked:
                self._active.discard(BreakerType.PEAK_HALT)
        return self._locked

    def release_lock(self) -> None:
        """Clear the peak breaker programmatically (backtests; live requires deleting the lock file)."""
        self._locked = False
        self._active.discard(BreakerType.PEAK_HALT)
        if self.lock_path is not None and self.lock_path.exists():
            self.lock_path.unlink()

    def check(self) -> BreakerStatus:
        """Current most-severe state, size multiplier and whether positions must be closed."""
        locked = self.locked
        active = set(self._active)
        if locked:
            active.add(BreakerType.PEAK_HALT)
        if BreakerType.PEAK_HALT in active:
            state = TradingState.LOCKED
        elif active & {BreakerType.DAILY_HALT, BreakerType.WEEKLY_HALT}:
            state = TradingState.HALTED
        elif active:
            state = TradingState.REDUCED
        else:
            state = TradingState.NORMAL
        mult = {TradingState.NORMAL: 1.0, TradingState.REDUCED: self.config.reduce_size_mult}.get(state, 0.0)
        ordered = tuple(b for b in BreakerType if b in active)
        return BreakerStatus(state, ordered, mult, state in (TradingState.HALTED, TradingState.LOCKED))

    def reset_daily(self) -> None:
        self._active -= {BreakerType.DAILY_REDUCE, BreakerType.DAILY_HALT}

    def reset_weekly(self) -> None:
        self._active -= {BreakerType.WEEKLY_REDUCE, BreakerType.WEEKLY_HALT}

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)


class RiskManager:
    """Stateful risk gate: equity anchors, circuit breakers, order history."""

    def __init__(
        self,
        config: RiskConfig,
        lock_path: str | Path | None | bool = True,
    ) -> None:
        """``lock_path=True`` uses ``config.lock_file``; ``None``/``False`` keeps the lock in memory."""
        self.config = config
        path = config.lock_file if lock_path is True else (lock_path or None)
        self.circuit_breaker = CircuitBreaker(config, path)
        self.peak_equity: float | None = None
        self.day_start_equity: float | None = None
        self.week_start_equity: float | None = None
        self.trades_today: int = 0
        self.rejections: list[dict[str, Any]] = []
        self._orders: dict[tuple[str, str], float] = {}
        self._last_equity: float | None = None
        self._last_day: Any = None
        self._last_week: tuple[int, int] | None = None
        self._was_locked = self.circuit_breaker.locked

    @property
    def state(self) -> TradingState:
        return self.circuit_breaker.check().state

    # ------------------------------------------------------------ equity marks

    def update_equity(self, equity: float, timestamp: datetime) -> None:
        """Record an equity mark, rolling day/week anchors (and breaker resets) and the peak.

        Anchors are the last mark before the new day/week, so on daily bars the
        daily P&L is that bar's close-to-close change. If the lock file was
        deleted since the last mark, trading resumes with the peak reset to ``equity``.
        """
        ts = pd.Timestamp(timestamp)
        day, week = ts.date(), tuple(ts.isocalendar())[:2]
        if self._last_equity is None:
            self.day_start_equity = self.week_start_equity = self.peak_equity = equity
        else:
            if day != self._last_day:
                self.day_start_equity = self._last_equity
                self.trades_today = 0
                self.circuit_breaker.reset_daily()
            if week != self._last_week:
                self.week_start_equity = self._last_equity
                self.circuit_breaker.reset_weekly()
        locked = self.circuit_breaker.locked
        if self._was_locked and not locked:
            logger.warning("Lock released; resuming trading with peak reset to %.2f", equity)
            self.peak_equity = equity
        self._was_locked = locked
        self.peak_equity = max(self.peak_equity or equity, equity)
        self._last_equity, self._last_day, self._last_week = equity, day, week

    def release_lock(self, equity: float) -> None:
        """Programmatic resume after a peak halt (backtests): clear the lock, reset the peak."""
        self.circuit_breaker.release_lock()
        self._was_locked = False
        self.peak_equity = equity

    def daily_drawdown(self, equity: float) -> float:
        """Drawdown from the start-of-day equity as a positive fraction."""
        return _drawdown(self.day_start_equity, equity)

    def weekly_drawdown(self, equity: float) -> float:
        """Drawdown from the start-of-week equity as a positive fraction."""
        return _drawdown(self.week_start_equity, equity)

    def drawdown_from_peak(self, equity: float) -> float:
        """Drawdown from the all-time equity peak as a positive fraction."""
        return _drawdown(self.peak_equity, equity)

    def build_state(
        self,
        equity: float,
        cash: float,
        positions: dict[str, PositionInfo] | None = None,
        buying_power: float | None = None,
        **extra: Any,
    ) -> PortfolioState:
        """PortfolioState filled from the tracked anchors (call after :meth:`update_equity`)."""
        bp = buying_power if buying_power is not None else max(
            0.0, equity * self.config.max_leverage - sum(abs(p.market_value) for p in (positions or {}).values())
        )
        return PortfolioState(
            equity=equity,
            cash=cash,
            buying_power=bp,
            positions=dict(positions or {}),
            daily_pnl=equity - (self.day_start_equity or equity),
            weekly_pnl=equity - (self.week_start_equity or equity),
            day_start_equity=self.day_start_equity,
            week_start_equity=self.week_start_equity,
            peak_equity=self.peak_equity,
            drawdown=self.drawdown_from_peak(equity),
            circuit_breaker_status=self.state,
            **extra,
        )

    def check_circuit_breakers(self, state: PortfolioState) -> BreakerStatus:
        """Update breakers from ``state`` and return the resulting status."""
        self.circuit_breaker.update(state)
        status = self.circuit_breaker.check()
        state.circuit_breaker_status = status.state
        return status

    def evaluate_state(self, equity: float) -> TradingState:
        """Breaker state for ``equity`` given the tracked anchors (no positions/cash detail)."""
        return self.check_circuit_breakers(self.build_state(equity, equity)).state

    # ------------------------------------------------------------ sizing

    def position_size(self, equity: float, entry_price: float, stop_price: float) -> int:
        """Shares so that loss at ``stop_price`` <= ``max_risk_per_trade`` * equity."""
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0 or equity <= 0:
            return 0
        return int(np.floor(self.config.max_risk_per_trade * equity / risk_per_share))

    def gap_position_size(self, equity: float, entry_price: float, stop_price: float) -> int:
        """Shares so that a gap through the stop of ``gap_stop_multiple`` x stop distance loses <= ``max_gap_loss``."""
        dist = abs(entry_price - stop_price)
        if dist <= 0 or equity <= 0:
            return 0
        return int(np.floor(self.config.max_gap_loss * equity / (self.config.gap_stop_multiple * dist)))

    def allowed_leverage(self, signal: Signal, state: PortfolioState, status: BreakerStatus | None = None) -> tuple[float, list[str]]:
        """Default 1.0x; up to ``max_leverage`` only for confident low-vol signals in calm portfolio conditions."""
        c = self.config
        status = status or self.circuit_breaker.check()
        meta = signal.metadata or {}
        if meta.get("vol_regime") != "low":
            return c.default_leverage, []
        reasons = []
        if meta.get("uncertain"):
            reasons.append("regime uncertain")
        if status.any_active:
            reasons.append(f"circuit breaker active ({status.state.value})")
        if state.open_positions >= c.leverage_max_positions:
            reasons.append(f"{state.open_positions} positions open (>= {c.leverage_max_positions})")
        if state.flicker_rate > c.max_flicker_rate:
            reasons.append(f"flicker rate {state.flicker_rate} > {c.max_flicker_rate}")
        if reasons:
            return c.default_leverage, [f"leverage forced to {c.default_leverage}x: " + ", ".join(reasons)]
        return min(max(float(signal.leverage), c.default_leverage), c.max_leverage), []

    # ------------------------------------------------------------ validation

    def _reject(self, signal: Signal, reason: str, detail: str, state: PortfolioState) -> RiskDecision:
        record = {
            "event": "risk_rejection",
            "timestamp": state.timestamp if state.timestamp is not None else signal.timestamp,
            "symbol": signal.symbol,
            "reason": reason,
            "detail": detail,
            "regime": signal.regime_name,
            "equity": state.equity,
        }
        self.rejections.append(record)
        logger.warning("Risk rejection: %s", json.dumps(record, default=str))
        return RiskDecision(False, None, reason, [detail])

    def _now(self, state: PortfolioState, now: float | pd.Timestamp | None) -> float:
        if now is None:
            now = state.timestamp
        if now is None:
            return time.time()
        return pd.Timestamp(now).timestamp() if not isinstance(now, (int, float)) else float(now)

    def requested_weight(self, signal: Signal) -> float:
        meta = signal.metadata or {}
        if getattr(signal.direction, "value", signal.direction) == "FLAT":
            return 0.0
        if "target_weight" in meta:
            return float(meta["target_weight"])
        return float(signal.position_size_pct) * float(signal.leverage)

    def validate_signal(
        self, signal: Signal, portfolio_state: PortfolioState, now: float | pd.Timestamp | None = None
    ) -> RiskDecision:
        """Approve, resize, or reject one signal against ``portfolio_state``."""
        c, st = self.config, portfolio_state
        eq = st.equity
        sym = signal.symbol
        held = st.positions.get(sym)
        cur_val = held.market_value if held is not None else 0.0
        req_val = max(0.0, self.requested_weight(signal)) * eq
        mods: list[str] = []

        if eq <= 0:
            return self._reject(signal, "non_positive_equity", f"equity {eq:.2f}", st)
        if req_val <= cur_val + 1e-9:
            return self._approve(signal, req_val, eq, signal.leverage, ["risk-reducing order"] if req_val < cur_val else [])

        status = self.circuit_breaker.check()
        if status.state in (TradingState.HALTED, TradingState.LOCKED):
            return self._reject(signal, "circuit_breaker_halt",
                                f"{status.state.value}: {[b.value for b in status.active]}", st)
        stop = signal.stop_loss
        price = float(signal.entry_price)
        if stop is None or not np.isfinite(stop) or stop <= 0 or stop >= price:
            return self._reject(signal, "missing_stop_loss", f"stop {stop!r} invalid for long entry {price}", st)
        if st.tradeable.get(sym, True) is False:
            return self._reject(signal, "not_tradeable", f"{sym} not tradeable", st)
        if sym in st.quotes:
            bid, ask = st.quotes[sym]
            mid = (bid + ask) / 2.0
            spread = (ask - bid) / mid if mid > 0 else np.inf
            if spread >= c.max_spread_pct:
                return self._reject(signal, "spread_too_wide", f"spread {spread:.4%} >= {c.max_spread_pct:.2%}", st)
        t_now = self._now(st, now)
        last = self._orders.get((sym, "buy"))
        if last is not None and t_now - last < c.duplicate_window_seconds:
            return self._reject(signal, "duplicate_order", f"buy {sym} {t_now - last:.0f}s after previous", st)
        if self.trades_today >= c.max_daily_trades:
            return self._reject(signal, "max_daily_trades", f"{self.trades_today} trades today", st)
        if held is None or held.qty == 0:
            if st.open_positions >= c.max_concurrent:
                return self._reject(signal, "max_concurrent_positions", f"{st.open_positions} open", st)

        lev, lev_mods = self.allowed_leverage(signal, st, status)
        mods += lev_mods

        caps = {
            "1% risk-per-trade size": self.position_size(eq, price, stop) * price,
            "overnight gap-risk size": self.gap_position_size(eq, price, stop) * price,
            "regime max position": c.regime_max_position.get(str((signal.metadata or {}).get("vol_regime")),
                                                             c.max_single_position) * eq,
            "max single position": c.max_single_position * eq,
        }
        value = req_val
        for name, cap in caps.items():
            if cap < value:
                mods.append(f"capped by {name}: {value / eq:.2%} -> {cap / eq:.2%}")
                value = cap

        if status.state is TradingState.REDUCED:
            mods.append(f"circuit breaker reduce x{status.size_multiplier}")
            value *= status.size_multiplier

        corr = self._max_correlation(sym, st)
        if corr is not None:
            if corr > c.correlation_reject:
                return self._reject(signal, "correlation_too_high", f"rho={corr:.2f} > {c.correlation_reject}", st)
            if corr > c.correlation_reduce:
                mods.append(f"correlation rho={corr:.2f} > {c.correlation_reduce}: x{c.correlation_reduce_mult}")
                value *= c.correlation_reduce_mult

        sector = c.sector(sym)
        sector_other = sum(p.market_value for s, p in st.positions.items() if s != sym and c.sector(s) == sector)
        room = c.max_sector_exposure * eq - sector_other
        if value > room:
            mods.append(f"capped by sector {sector} limit: {value / eq:.2%} -> {max(room, 0) / eq:.2%}")
            value = max(room, 0.0)

        gross_other = st.gross_exposure - abs(cur_val)
        room = c.max_exposure * lev * eq - gross_other
        if value > room:
            mods.append(f"capped by gross exposure {c.max_exposure:.0%}x{lev}: {value / eq:.2%} -> {max(room, 0) / eq:.2%}")
            value = max(room, 0.0)

        if value - cur_val > st.buying_power:
            new_value = cur_val + max(st.buying_power, 0.0)
            mods.append(f"capped by buying power {st.buying_power:,.2f}")
            value = new_value

        if 0 < value < c.min_position_value:
            return self._reject(signal, "below_min_position", f"${value:.2f} < ${c.min_position_value:.0f}", st)
        if value <= 0 and cur_val <= 0:
            return self._reject(signal, "no_capacity", "; ".join(mods) or "zero size after limits", st)
        return self._approve(signal, value, eq, lev, mods)

    def _approve(self, signal: Signal, value: float, equity: float, leverage: float, mods: list[str]) -> RiskDecision:
        w = value / equity if equity > 0 else 0.0
        modified = replace(
            signal,
            leverage=float(leverage),
            position_size_pct=w / leverage if leverage else w,
            metadata={**(signal.metadata or {}), "target_weight": w, "target_value": value,
                      "risk_modifications": list(mods)},
        )
        return RiskDecision(True, modified, None, list(mods))

    def validate_signals(
        self, signals: list[Signal], portfolio_state: PortfolioState, now: float | pd.Timestamp | None = None
    ) -> list[RiskDecision]:
        """Validate a batch, reductions first, applying each approval to a working state
        so later signals see the capacity already used."""
        st = replace(portfolio_state, positions=dict(portfolio_state.positions))
        order = sorted(range(len(signals)), key=lambda i: self.requested_weight(signals[i]) * st.equity
                       - (st.positions[signals[i].symbol].market_value if signals[i].symbol in st.positions else 0.0))
        out: list[RiskDecision | None] = [None] * len(signals)
        for i in order:
            sig = signals[i]
            d = self.validate_signal(sig, st, now)
            out[i] = d
            if d.approved and d.modified_signal is not None:
                price = float(sig.entry_price)
                old = st.positions.get(sig.symbol)
                target = d.modified_signal.metadata["target_value"]
                delta = target - (old.market_value if old else 0.0)
                st.buying_power -= max(delta, 0.0)
                st.positions[sig.symbol] = PositionInfo(sig.symbol, target / price if price else 0.0, price, sig.stop_loss)
        return [d for d in out if d is not None]

    def _max_correlation(self, symbol: str, state: PortfolioState) -> float | None:
        r = state.returns
        held = [s for s, p in state.positions.items() if s != symbol and p.qty != 0]
        if r is None or symbol not in r or not held:
            return None
        window = r.tail(self.config.correlation_window)
        corrs = [window[symbol].corr(window[s]) for s in held if s in window]
        corrs = [x for x in corrs if np.isfinite(x)]
        return max(corrs) if corrs else None

    # ------------------------------------------------------------ order bookkeeping

    def record_order(self, symbol: str, side: str, when: float | pd.Timestamp | None = None) -> None:
        """Register a submitted order for duplicate detection and the daily trade count."""
        t = time.time() if when is None else (float(when) if isinstance(when, (int, float)) else pd.Timestamp(when).timestamp())
        self._orders[(symbol, side)] = t
        self.trades_today += 1

    def record_trade(self) -> None:
        """Increment today's trade counter."""
        self.trades_today += 1

    def can_trade(self) -> bool:
        """True if no halt is active and ``max_daily_trades`` isn't reached."""
        if self.circuit_breaker.check().state in (TradingState.HALTED, TradingState.LOCKED):
            return False
        return self.trades_today < self.config.max_daily_trades

    def reset_daily(self) -> None:
        """Reset per-day counters, anchors and daily breakers at the session open."""
        self.trades_today = 0
        self.circuit_breaker.reset_daily()
        if self._last_equity is not None:
            self.day_start_equity = self._last_equity


def _drawdown(anchor: float | None, equity: float) -> float:
    if anchor is None or anchor <= 0:
        return 0.0
    return max(0.0, 1.0 - equity / anchor)
