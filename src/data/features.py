"""
multi-feature builder for financial time series.

each transformation is strictly causal (uses only past values). returns an
(N, C) float array where C is the number of feature channels.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from .preprocessing import calculate_log_returns, clip_outliers


def realized_vol(returns: np.ndarray, window: int) -> np.ndarray:
    """log of rolling std of returns; causal (uses past `window` values incl. current)."""
    s = pd.Series(returns)
    sigma = s.rolling(window=window, min_periods=window).std().bfill().fillna(1e-8)
    return np.log(sigma.values + 1e-8)


def causal_zscore(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    roll = s.rolling(window=window, min_periods=window)
    mu = roll.mean()
    sigma = roll.std().replace(0, 1e-8)
    return ((s - mu) / sigma).fillna(0).values


def time_of_day_cyclic(timestamps_ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """sin/cos encoding of minute-of-day (UTC). assumes ms timestamps."""
    minute_of_day = (timestamps_ms // 60000) % 1440
    angle = 2 * np.pi * minute_of_day / 1440.0
    return np.sin(angle), np.cos(angle)


def momentum(returns: np.ndarray, horizon: int) -> np.ndarray:
    """cumulative log return over the past `horizon` bars (causal)."""
    s = pd.Series(returns)
    return s.rolling(window=horizon, min_periods=horizon).sum().fillna(0).values


def build_multifeature(
    prices: np.ndarray,
    timestamps_ms: np.ndarray,
    z_window: int = 30,
    vol_window: int = 30,
    mom_horizons: Tuple[int, ...] = (5, 20),
) -> np.ndarray:
    """
    returns (N, C) feature matrix. each channel is causally normalized.
    channels:
      0   z-scored log return                  (baseline signal)
      1   z-scored log realized vol
      2   sin(minute-of-day)
      3   cos(minute-of-day)
      4+  z-scored momentum at each horizon
    """
    rets = calculate_log_returns(prices)
    ret_z = clip_outliers(causal_zscore(rets, window=z_window))

    rv = realized_vol(rets, window=vol_window)
    rv_z = clip_outliers(causal_zscore(rv, window=z_window))

    tod_sin, tod_cos = time_of_day_cyclic(timestamps_ms)

    feats = [ret_z, rv_z, tod_sin, tod_cos]
    for h in mom_horizons:
        m = momentum(rets, horizon=h)
        feats.append(clip_outliers(causal_zscore(m, window=z_window)))

    out = np.stack(feats, axis=1).astype(np.float32)
    return out
