"""Volatility-regime allocation strategies (always long, never short).

The HMM classifies volatility environments, not direction. Each fitted regime is
ranked by ``RegimeInfo.expected_volatility`` and mapped to a strategy:

- lowest third  -> :class:`LowVolBullStrategy`        (fully invested, 1.25x)
- middle        -> :class:`MidVolCautiousStrategy`    (invested if trend intact, else reduced)
- top third     -> :class:`HighVolDefensiveStrategy`  (reduced, still partially invested)

Position sizing adjustments applied by :class:`StrategyOrchestrator`, multiplicatively:

- uncertainty (confirmed-regime probability < ``min_confidence`` or flickering):
  size x ``uncertainty_size_mult`` (0.50)
- pending regime transition (stability filter holding the previous regime):
  size x ``transition_size_mult`` (0.75)
- both: size x 0.375. Leverage is forced to 1.0x whenever either applies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState
from data.feature_engineering import atr, validate_bars

if TYPE_CHECKING:
    from core.hmm_engine import HMMEngine

logger = logging.getLogger("regime_trader.strategy")

BASE_LEVERAGE = 1.0


class Direction(str, Enum):
    """Signal direction. Strategies never emit shorts."""

    LONG = "LONG"
    FLAT = "FLAT"


class VolRegime(str, Enum):
    """Volatility environment derived from a regime's volatility rank."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"


@dataclass(frozen=True)
class Signal:
    """Per-symbol trade intent.

    ``position_size_pct`` is the strategy's portfolio allocation (0.60-0.95
    nominal, lower after uncertainty/transition scaling). The per-symbol
    portfolio weight is ``metadata["target_weight"]``.
    """

    symbol: str
    direction: Direction
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float | None
    position_size_pct: float
    leverage: float
    regime_id: int
    regime_name: str
    regime_probability: float
    timestamp: pd.Timestamp | None
    reasoning: str
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Strategy parameters.

    Mirrors the ``strategy`` section of settings.yaml; ``min_confidence`` and
    ``transition_size_mult`` are sourced from the ``hmm`` section.
    """

    low_vol_allocation: float = 0.95
    mid_vol_allocation_trend: float = 0.95
    mid_vol_allocation_no_trend: float = 0.60
    high_vol_allocation: float = 0.60
    low_vol_leverage: float = 1.25
    rebalance_threshold: float = 0.10
    uncertainty_size_mult: float = 0.50
    ema_window: int = 50
    atr_window: int = 14
    low_vol_atr_stop_mult: float = 3.0
    low_vol_ema_atr_mult: float = 0.5
    mid_vol_ema_atr_mult: float = 0.5
    high_vol_ema_atr_mult: float = 1.0
    low_vol_rank_cutoff: float = 0.33
    high_vol_rank_cutoff: float = 0.67
    min_confidence: float = 0.55
    transition_size_mult: float = 0.75

    @classmethod
    def from_dict(
        cls, cfg: dict[str, Any] | None, hmm_cfg: dict[str, Any] | None = None
    ) -> StrategyConfig:
        """Build from the ``strategy`` section, pulling shared keys from ``hmm``."""
        params = dict(cfg or {})
        names = {f.name for f in fields(cls)}
        unknown = set(params) - names
        if unknown:
            raise ValueError(f"Unknown strategy config keys: {sorted(unknown)}")
        for key in ("min_confidence", "transition_size_mult"):
            if hmm_cfg and key in hmm_cfg:
                params.setdefault(key, hmm_cfg[key])
        return cls(**params)


def vol_regime_for_rank(
    vol_rank: int, n_regimes: int, low_cutoff: float = 0.33, high_cutoff: float = 0.67
) -> VolRegime:
    """position = rank / (n - 1); <= low_cutoff -> LOW, >= high_cutoff -> HIGH, else MID."""
    position = vol_rank / (n_regimes - 1) if n_regimes > 1 else 0.0
    if position <= low_cutoff:
        return VolRegime.LOW
    if position >= high_cutoff:
        return VolRegime.HIGH
    return VolRegime.MID


@dataclass(frozen=True)
class Indicators:
    """Latest price, EMA and ATR used for entries and stops."""

    price: float
    ema: float
    atr: float


class BaseStrategy(ABC):
    """Common indicator/stop helpers; subclasses implement :meth:`generate_signal`."""

    name: ClassVar[str] = "BaseStrategy"
    vol_regime: ClassVar[VolRegime]

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    @abstractmethod
    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Signal | None:
        """Return a nominal LONG signal, or None if indicators are unavailable."""

    def indicators(self, bars: pd.DataFrame) -> Indicators | None:
        """Latest close, EMA(``ema_window``) and ATR(``atr_window``); None if not warmed up."""
        b = validate_bars(bars)
        if len(b) < max(self.config.ema_window, self.config.atr_window):
            return None
        close = b["close"].astype(float)
        ema = close.ewm(span=self.config.ema_window, adjust=False, min_periods=self.config.ema_window).mean()
        a = atr(b["high"].astype(float), b["low"].astype(float), close, self.config.atr_window)
        ind = Indicators(float(close.iloc[-1]), float(ema.iloc[-1]), float(a.iloc[-1]))
        if not all(np.isfinite([ind.price, ind.ema, ind.atr])) or ind.atr <= 0 or ind.price <= 0:
            return None
        return ind

    def build_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
        ind: Indicators,
        stop: float,
        size: float,
        leverage: float,
        reasoning: str,
    ) -> Signal:
        ts = bars.index[-1] if isinstance(bars.index, pd.DatetimeIndex) else regime_state.timestamp
        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability,
            entry_price=ind.price,
            stop_loss=stop,
            take_profit=None,
            position_size_pct=size,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=ts,
            reasoning=reasoning,
            strategy_name=self.name,
            metadata={"ema": ind.ema, "atr": ind.atr, "trend_intact": ind.price > ind.ema},
        )


class LowVolBullStrategy(BaseStrategy):
    """Calm market: fully invested with modest leverage.

    95% allocation, 1.25x, stop = max(price - 3 ATR, EMA50 - 0.5 ATR).
    """

    name = "LowVolBullStrategy"
    vol_regime = VolRegime.LOW

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Signal | None:
        ind = self.indicators(bars)
        if ind is None:
            return None
        c = self.config
        stop = max(ind.price - c.low_vol_atr_stop_mult * ind.atr, ind.ema - c.low_vol_ema_atr_mult * ind.atr)
        return self.build_signal(
            symbol, bars, regime_state, ind, stop, c.low_vol_allocation, c.low_vol_leverage,
            f"Low-vol regime: fully invested {c.low_vol_allocation:.0%} at {c.low_vol_leverage}x",
        )


class MidVolCautiousStrategy(BaseStrategy):
    """Moderate volatility: invested while trend holds, reduced otherwise.

    price > EMA50 -> 95%, else 60%; always 1.0x; stop = EMA50 - 0.5 ATR.
    """

    name = "MidVolCautiousStrategy"
    vol_regime = VolRegime.MID

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Signal | None:
        ind = self.indicators(bars)
        if ind is None:
            return None
        c = self.config
        trend = ind.price > ind.ema
        size = c.mid_vol_allocation_trend if trend else c.mid_vol_allocation_no_trend
        why = "trend intact (price > EMA)" if trend else "trend broken (price <= EMA)"
        return self.build_signal(
            symbol, bars, regime_state, ind, ind.ema - c.mid_vol_ema_atr_mult * ind.atr, size, BASE_LEVERAGE,
            f"Mid-vol regime, {why}: {size:.0%} at 1.0x",
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """Turbulent market: reduced but still partially invested.

    60% allocation, 1.0x, stop = EMA50 - 1.0 ATR.
    """

    name = "HighVolDefensiveStrategy"
    vol_regime = VolRegime.HIGH

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Signal | None:
        ind = self.indicators(bars)
        if ind is None:
            return None
        c = self.config
        return self.build_signal(
            symbol, bars, regime_state, ind, ind.ema - c.high_vol_ema_atr_mult * ind.atr,
            c.high_vol_allocation, BASE_LEVERAGE,
            f"High-vol regime: defensive {c.high_vol_allocation:.0%} at 1.0x",
        )


CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
MeanReversionStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy

# Readability / backward-compatibility map only. StrategyOrchestrator selects by
# volatility rank, never by these return-sorted labels.
LABEL_TO_STRATEGY: dict[str, type[BaseStrategy]] = {
    "CRASH": CrashDefensiveStrategy,
    "STRONG_BEAR": BearTrendStrategy,
    "BEAR": BearTrendStrategy,
    "WEAK_BEAR": MeanReversionStrategy,
    "NEUTRAL": MeanReversionStrategy,
    "WEAK_BULL": BullTrendStrategy,
    "BULL": BullTrendStrategy,
    "STRONG_BULL": BullTrendStrategy,
    "EUPHORIA": EuphoriaCautiousStrategy,
}

VOL_REGIME_TO_STRATEGY: dict[VolRegime, type[BaseStrategy]] = {
    VolRegime.LOW: LowVolBullStrategy,
    VolRegime.MID: MidVolCautiousStrategy,
    VolRegime.HIGH: HighVolDefensiveStrategy,
}


@dataclass(frozen=True)
class AllocationTarget:
    """Portfolio-level summary of one batch of signals.

    ``gross_exposure`` = mean over symbols of position_size_pct * leverage
    (equal-weight); ``weights`` holds each symbol's share of it.
    """

    vol_regime: VolRegime
    strategy_name: str
    gross_exposure: float
    size_multiplier: float
    uncertain: bool
    in_transition: bool
    weights: dict[str, float]


class StrategyOrchestrator:
    """Maps regimes to strategies by volatility rank and produces adjusted signals."""

    def __init__(self, config: StrategyConfig, regime_infos: list[RegimeInfo] | None = None) -> None:
        self.config = config
        self._strategies: dict[VolRegime, BaseStrategy] = {
            vr: cls(config) for vr, cls in VOL_REGIME_TO_STRATEGY.items()
        }
        if regime_infos is not None:
            self.update_regimes(regime_infos)

    @classmethod
    def from_engine(cls, config: StrategyConfig, engine: HMMEngine) -> StrategyOrchestrator:
        """Build from a fitted engine, checking its volatility ranking agrees."""
        orch = cls(config, engine.regime_info)
        if orch.volatility_ranking != engine.get_volatility_ranking():
            raise ValueError("Volatility ranking mismatch between engine and orchestrator")
        return orch

    def update_regimes(self, regime_infos: list[RegimeInfo]) -> None:
        """(Re)build regime -> vol rank -> strategy maps, e.g. after an HMM retrain."""
        if not regime_infos:
            raise ValueError("regime_infos must be non-empty")
        self.regime_infos = {ri.regime_id: ri for ri in regime_infos}
        ordered = sorted(regime_infos, key=lambda ri: (ri.expected_volatility, ri.regime_id))
        self.volatility_ranking: list[int] = [ri.regime_id for ri in ordered]
        n = len(ordered)
        self.vol_rank: dict[int, int] = {rid: rank for rank, rid in enumerate(self.volatility_ranking)}
        self.vol_regime: dict[int, VolRegime] = {
            rid: vol_regime_for_rank(rank, n, self.config.low_vol_rank_cutoff, self.config.high_vol_rank_cutoff)
            for rid, rank in self.vol_rank.items()
        }

    def get_strategy(self, regime_id: int) -> BaseStrategy:
        """Strategy instance for ``regime_id`` (by volatility rank)."""
        if regime_id not in self.vol_regime:
            raise ValueError(f"Unknown regime_id {regime_id}; call update_regimes after retraining")
        return self._strategies[self.vol_regime[regime_id]]

    def is_uncertain(self, regime_state: RegimeState, is_flickering: bool) -> bool:
        """Low confirmed-regime probability or flickering."""
        return is_flickering or regime_state.probability < self.config.min_confidence

    def size_multiplier(self, regime_state: RegimeState, is_flickering: bool) -> float:
        """Product of the uncertainty and pending-transition multipliers."""
        mult = 1.0
        if self.is_uncertain(regime_state, is_flickering):
            mult *= self.config.uncertainty_size_mult
        if regime_state.in_transition:
            mult *= self.config.transition_size_mult
        return mult

    def needs_rebalance(self, current_allocation: float, target_allocation: float) -> bool:
        """True if |target - current| gross exposure exceeds ``rebalance_threshold``."""
        return abs(target_allocation - current_allocation) > self.config.rebalance_threshold + 1e-9

    def target_allocation(self, signals: list[Signal]) -> AllocationTarget | None:
        """Summarise a signal batch; None if empty."""
        if not signals:
            return None
        m = signals[0].metadata
        return AllocationTarget(
            vol_regime=VolRegime(m["vol_regime"]),
            strategy_name=signals[0].strategy_name,
            gross_exposure=float(sum(s.metadata["target_weight"] for s in signals)),
            size_multiplier=m["size_multiplier"],
            uncertain=m["uncertain"],
            in_transition=m["in_transition"],
            weights={s.symbol: s.metadata["target_weight"] for s in signals},
        )

    def generate_signals(
        self,
        symbols: list[str],
        bars: dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool = False,
        current_allocation: float | None = None,
    ) -> list[Signal]:
        """Signals for ``symbols`` under the confirmed regime in ``regime_state``.

        Symbols without bars or warmed-up indicators are skipped. If
        ``current_allocation`` (current gross exposure as a fraction of equity)
        is given and the new target is within ``rebalance_threshold`` of it,
        returns an empty list to avoid churn.
        """
        strategy = self.get_strategy(regime_state.state_id)
        uncertain = self.is_uncertain(regime_state, is_flickering)
        mult = self.size_multiplier(regime_state, is_flickering)
        derisk = uncertain or regime_state.in_transition

        signals: list[Signal] = []
        for sym in symbols:
            sym_bars = bars.get(sym)
            if sym_bars is None or sym_bars.empty:
                logger.warning("No bars for %s; skipping", sym)
                continue
            sig = strategy.generate_signal(sym, sym_bars, regime_state)
            if sig is None:
                logger.debug("Indicators not ready for %s; skipping", sym)
                continue
            notes = []
            if uncertain:
                notes.append(f"uncertain (p={regime_state.probability:.2f}, flicker={is_flickering}) x{self.config.uncertainty_size_mult}")
            if regime_state.in_transition:
                notes.append(f"pending transition x{self.config.transition_size_mult}")
            signals.append(
                replace(
                    sig,
                    position_size_pct=sig.position_size_pct * mult,
                    leverage=BASE_LEVERAGE if derisk else sig.leverage,
                    reasoning=sig.reasoning + ("; " + "; ".join(notes) if notes else ""),
                    metadata={
                        **sig.metadata,
                        "vol_rank": self.vol_rank[regime_state.state_id],
                        "vol_regime": self.vol_regime[regime_state.state_id].value,
                        "nominal_size_pct": sig.position_size_pct,
                        "nominal_leverage": sig.leverage,
                        "size_multiplier": mult,
                        "uncertain": uncertain,
                        "in_transition": regime_state.in_transition,
                    },
                )
            )

        n = len(signals)
        for s in signals:
            s.metadata["target_weight"] = s.position_size_pct * s.leverage / n

        target = self.target_allocation(signals)
        if target is not None and current_allocation is not None and not self.needs_rebalance(
            current_allocation, target.gross_exposure
        ):
            logger.info(
                "No rebalance: target exposure %.3f within %.2f of current %.3f",
                target.gross_exposure, self.config.rebalance_threshold, current_allocation,
            )
            return []
        return signals
