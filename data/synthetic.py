"""Synthetic regime-switching OHLCV bars for tests and offline backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_VOLS = np.array([0.006, 0.012, 0.028])
TRUE_DRIFTS = np.array([0.0006, 0.0001, -0.0012])
TRUE_TRANSMAT = np.array(
    [
        [0.985, 0.012, 0.003],
        [0.020, 0.970, 0.010],
        [0.010, 0.030, 0.960],
    ]
)


def make_synthetic_bars(
    n: int = 1300, seed: int = 7, start: str = "2019-01-01"
) -> tuple[pd.DataFrame, pd.Series]:
    """Regime-switching OHLCV bars plus the true volatility regime per bar."""
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=int)
    s = 0
    for t in range(n):
        states[t] = s
        s = int(rng.choice(3, p=TRUE_TRANSMAT[s]))
    vol = TRUE_VOLS[states]
    r = TRUE_DRIFTS[states] + vol * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(r))
    prev_close = np.r_[100.0, close[:-1]]
    open_ = prev_close * np.exp(0.2 * vol * rng.standard_normal(n))
    high = np.maximum(open_, close) * np.exp(0.5 * vol * np.abs(rng.standard_normal(n)))
    low = np.minimum(open_, close) * np.exp(-0.5 * vol * np.abs(rng.standard_normal(n)))
    volume = 1e6 * (1.0 + 30.0 * vol) * np.exp(0.25 * rng.standard_normal(n))
    index = pd.bdate_range(start, periods=n, name="timestamp")
    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )
    return bars, pd.Series(states, index=index, name="true_state")
