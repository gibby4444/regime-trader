"""Technical indicators and HMM input features.

All functions are pure and strictly causal: the value at bar t depends only on
bars <= t. Recursive (Wilder) smoothers start from the first bar supplied, so
values are identical for any two inputs sharing the same start bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_20",
    "realized_vol_20",
    "vol_ratio_5_20",
    "volume_z",
    "volume_trend",
    "adx_14",
    "sma50_slope",
    "rsi_14",
    "dist_sma200",
    "roc_10",
    "roc_20",
    "natr_14",
)


@dataclass
class FeatureConfig:
    """Feature parameters mirroring the ``features`` section of settings.yaml."""

    return_windows: list[int] = field(default_factory=lambda: [1, 5, 20])
    vol_window: int = 20
    vol_ratio_short: int = 5
    volume_window: int = 50
    volume_sma_window: int = 10
    slope_lookback: int = 5
    adx_window: int = 14
    trend_sma_window: int = 50
    rsi_window: int = 14
    long_sma_window: int = 200
    roc_windows: list[int] = field(default_factory=lambda: [10, 20])
    atr_window: int = 14
    zscore_window: int = 252

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> FeatureConfig:
        """Build a config from the parsed ``features`` settings section."""
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown feature config keys: {sorted(unknown)}")
        return cls(**cfg)


def _clean(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def log_returns(close: pd.Series, window: int = 1) -> pd.Series:
    """``window``-bar log return: ln(close_t / close_{t-window})."""
    return np.log(close / close.shift(window))


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of 1-bar log returns (not annualised)."""
    return log_returns(close, 1).rolling(window, min_periods=window).std()


def volatility_ratio(close: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """Short-window over long-window realized volatility."""
    return _clean(realized_volatility(close, short) / realized_volatility(close, long))


def normalized_volume(volume: pd.Series, window: int = 50) -> pd.Series:
    """Volume z-score against its trailing ``window``-bar mean and std."""
    roll = volume.rolling(window, min_periods=window)
    return _clean((volume - roll.mean()) / roll.std())


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window, min_periods=window).mean()


def sma_slope(series: pd.Series, window: int, lookback: int = 5) -> pd.Series:
    """Average per-bar fractional change of the ``window`` SMA over ``lookback`` bars."""
    avg = sma(series, window)
    prev = avg.shift(lookback)
    return _clean((avg - prev) / (prev * lookback))


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's recursive moving average (EMA with alpha = 1/window)."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range; the first bar falls back to high - low."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    return wilder_smooth(true_range(high, low, close), window)


def normalized_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """ATR as a fraction of close."""
    return atr(high, low, close, window) / close


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index (Wilder), 0-100."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr_smooth = atr(high, low, close, window)
    plus_di = 100.0 * wilder_smooth(plus_dm, window) / tr_smooth
    minus_di = 100.0 * wilder_smooth(minus_dm, window) / tr_smooth
    dx = _clean(100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di))
    return wilder_smooth(dx, window)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder), 0-100."""
    delta = close.diff()
    avg_gain = wilder_smooth(delta.clip(lower=0.0), window)
    avg_loss = wilder_smooth(-delta.clip(upper=0.0), window)
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def distance_from_sma(close: pd.Series, window: int = 200) -> pd.Series:
    """(close - SMA) / close."""
    return (close - sma(close, window)) / close


def rate_of_change(close: pd.Series, window: int) -> pd.Series:
    """Simple ``window``-bar return: close_t / close_{t-window} - 1."""
    return close / close.shift(window) - 1.0


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Trailing z-score using the ``window`` bars ending at (and including) t."""
    roll = series.rolling(window, min_periods=window)
    return _clean((series - roll.mean()) / roll.std())


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Return ``bars`` with lowercase columns, checking required OHLCV and ordering."""
    out = bars.rename(columns=str.lower)
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    if not out.index.is_monotonic_increasing:
        raise ValueError("bars index must be sorted ascending")
    if out.index.has_duplicates:
        raise ValueError("bars index has duplicate timestamps")
    return out


def compute_raw_features(bars: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Compute unstandardized features (columns = ``FEATURE_COLUMNS``)."""
    cfg = config or FeatureConfig()
    b = validate_bars(bars)
    o, h, lo, c, v = (b[col].astype(float) for col in REQUIRED_COLUMNS)
    del o
    feats: dict[str, pd.Series] = {}
    for w in cfg.return_windows:
        feats[f"ret_{w}"] = log_returns(c, w)
    feats[f"realized_vol_{cfg.vol_window}"] = realized_volatility(c, cfg.vol_window)
    feats[f"vol_ratio_{cfg.vol_ratio_short}_{cfg.vol_window}"] = volatility_ratio(
        c, cfg.vol_ratio_short, cfg.vol_window
    )
    feats["volume_z"] = normalized_volume(v, cfg.volume_window)
    feats["volume_trend"] = sma_slope(v, cfg.volume_sma_window, cfg.slope_lookback)
    feats[f"adx_{cfg.adx_window}"] = adx(h, lo, c, cfg.adx_window)
    feats[f"sma{cfg.trend_sma_window}_slope"] = sma_slope(c, cfg.trend_sma_window, cfg.slope_lookback)
    feats[f"rsi_{cfg.rsi_window}"] = rsi(c, cfg.rsi_window)
    feats[f"dist_sma{cfg.long_sma_window}"] = distance_from_sma(c, cfg.long_sma_window)
    for w in cfg.roc_windows:
        feats[f"roc_{w}"] = rate_of_change(c, w)
    feats[f"natr_{cfg.atr_window}"] = normalized_atr(h, lo, c, cfg.atr_window)
    return pd.DataFrame(feats, index=b.index)


def standardize(raw: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Apply :func:`rolling_zscore` to every column."""
    return raw.apply(lambda col: rolling_zscore(col, window))


def build_features(
    bars: pd.DataFrame, config: FeatureConfig | None = None, dropna: bool = True
) -> pd.DataFrame:
    """Full pipeline: raw features -> rolling z-score -> drop warm-up rows.

    With default windows the first ~451 bars are warm-up (200-bar SMA plus a
    252-bar z-score window).
    """
    cfg = config or FeatureConfig()
    feats = standardize(compute_raw_features(bars, cfg), cfg.zscore_window)
    return feats.dropna() if dropna else feats


def warmup_bars(config: FeatureConfig | None = None) -> int:
    """Approximate number of leading bars consumed before features are valid."""
    cfg = config or FeatureConfig()
    longest = max(
        max(cfg.return_windows),
        cfg.vol_window,
        cfg.volume_window,
        cfg.trend_sma_window + cfg.slope_lookback,
        cfg.long_sma_window,
        max(cfg.roc_windows),
        2 * cfg.adx_window,
    )
    return longest + cfg.zscore_window - 1


class FeatureEngineer:
    """Convenience wrapper binding a :class:`FeatureConfig` to the pure functions."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def build_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Standardized, NaN-free feature matrix for one symbol / benchmark."""
        return build_features(bars, self.config)

    def returns(self, bars: pd.DataFrame) -> pd.Series:
        """Raw 1-bar log returns aligned with ``bars``."""
        return log_returns(validate_bars(bars)["close"].astype(float), 1)

    def build_universe_features(
        self, bars_by_symbol: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Apply :meth:`build_features` to each symbol."""
        return {sym: self.build_features(b) for sym, b in bars_by_symbol.items()}
