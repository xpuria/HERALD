"""
data preprocessing pipeline: log returns -> z-score -> clipping
"""

import numpy as np
import pandas as pd


def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """r_t = ln(P_t / P_{t-1})"""
    return np.diff(np.log(prices), prepend=prices[0])


def rolling_z_score(x: np.ndarray, window: int = 30) -> np.ndarray:
    """z_t = (x_t - mean) / std using rolling window"""
    s = pd.Series(x)
    roll = s.rolling(window=window, min_periods=window)
    mu = roll.mean()
    sigma = roll.std()
    sigma = sigma.replace(0, 1e-8)
    z = (s - mu) / sigma
    return z.fillna(0).values


def clip_outliers(x: np.ndarray, min_val: float = -5.0, max_val: float = 5.0) -> np.ndarray:
    """bound extreme values"""
    return np.clip(x, min_val, max_val)


def preprocess_pipeline(prices: np.ndarray, window: int = 64) -> np.ndarray:
    """full pipeline: log returns -> z-score -> clip"""
    returns = calculate_log_returns(prices)
    normalized = rolling_z_score(returns, window=window)
    clipped = clip_outliers(normalized)
    return clipped
