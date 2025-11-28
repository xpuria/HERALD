"""
zigzag trend labeling: identifies significant peaks and troughs
labels: 1 = uptrend (toward peak), 0 = downtrend (toward trough)
"""

import numpy as np
from typing import Tuple


def zigzag(prices: np.ndarray, deviation_pct: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    compute zigzag indicator and trend labels.
    deviation_pct: minimum % move to register reversal (0.01 = 1%)
    returns: (pivots, trends)
    """
    n = len(prices)
    if n < 2:
        return np.full(n, np.nan), np.zeros(n)

    pivots = np.full(n, np.nan)
    trends = np.zeros(n, dtype=int)
    
    last_pivot_idx = 0
    last_pivot_price = prices[0]
    current_trend = 0  # 0=undecided, 1=up, -1=down
    
    pivots[0] = last_pivot_price
    pivot_indices = [0]
    
    for i in range(1, n):
        price = prices[i]
        
        if current_trend == 0:
            if price > last_pivot_price * (1 + deviation_pct):
                current_trend = 1
            elif price < last_pivot_price * (1 - deviation_pct):
                current_trend = -1
        
        if current_trend == 1:
            if price > last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = i
            elif price < last_pivot_price * (1 - deviation_pct):
                pivots[last_pivot_idx] = last_pivot_price
                pivot_indices.append(last_pivot_idx)
                current_trend = -1
                last_pivot_price = price
                last_pivot_idx = i
        
        elif current_trend == -1:
            if price < last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = i
            elif price > last_pivot_price * (1 + deviation_pct):
                pivots[last_pivot_idx] = last_pivot_price
                pivot_indices.append(last_pivot_idx)
                current_trend = 1
                last_pivot_price = price
                last_pivot_idx = i
    
    pivots[last_pivot_idx] = last_pivot_price
    pivot_indices.append(last_pivot_idx)
    
    # fill trend labels between pivots
    for k in range(len(pivot_indices) - 1):
        start_idx = pivot_indices[k]
        end_idx = pivot_indices[k+1]
        
        if prices[end_idx] > prices[start_idx]:
            trends[start_idx:end_idx+1] = 1
        else:
            trends[start_idx:end_idx+1] = 0
    
    return pivots, trends
