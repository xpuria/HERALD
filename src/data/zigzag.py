"""
zigzag trend labeling.

two labelers are provided:

- `zigzag` (default, strictly causal): at every t, the label reflects ONLY the
  trend that has been confirmed from past prices. unconfirmed early steps and
  the unfinished tail are marked with `ignore_label`.
- `zigzag_lookahead`: the original implementation that fills labels between
  past AND future pivots. kept for reference / ablation only — it leaks the
  target into the input window and must not be used for training.

a third helper, `next_return_sign`, provides the standard finance label:
sign of the forward h-step log return. it is causal by construction (training
loss must ignore the final h positions).
"""

import numpy as np
from typing import Tuple


IGNORE_LABEL = -100


def zigzag(
    prices: np.ndarray,
    deviation_pct: float = 0.01,
    ignore_label: int = IGNORE_LABEL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    causal zigzag labels.

    walks the standard zigzag state machine forward in time. at each step t
    the emitted label is the currently established trend (1=up, 0=down) using
    only prices[0..t]. before the first confirmed reversal there is no trend,
    so those positions get `ignore_label`.

    returns (pivots, trends) — pivots is filled where they are confirmed.
    """
    n = len(prices)
    pivots = np.full(n, np.nan)
    trends = np.full(n, ignore_label, dtype=np.int64)
    if n < 2:
        return pivots, trends

    last_pivot_idx = 0
    last_pivot_price = float(prices[0])
    current_trend = 0  # 0 = undecided, 1 = up, -1 = down

    for i in range(1, n):
        price = float(prices[i])

        if current_trend == 0:
            if price > last_pivot_price * (1 + deviation_pct):
                current_trend = 1
            elif price < last_pivot_price * (1 - deviation_pct):
                current_trend = -1

        elif current_trend == 1:
            if price > last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = i
            elif price < last_pivot_price * (1 - deviation_pct):
                pivots[last_pivot_idx] = last_pivot_price
                current_trend = -1
                last_pivot_price = price
                last_pivot_idx = i

        else:  # current_trend == -1
            if price < last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = i
            elif price > last_pivot_price * (1 + deviation_pct):
                pivots[last_pivot_idx] = last_pivot_price
                current_trend = 1
                last_pivot_price = price
                last_pivot_idx = i

        if current_trend == 1:
            trends[i] = 1
        elif current_trend == -1:
            trends[i] = 0

    return pivots, trends


def zigzag_lookahead(
    prices: np.ndarray, deviation_pct: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """original lookahead-leaking labeler. for ablation only — do not train on this."""
    n = len(prices)
    if n < 2:
        return np.full(n, np.nan), np.zeros(n, dtype=np.int64)

    pivots = np.full(n, np.nan)
    trends = np.zeros(n, dtype=np.int64)

    last_pivot_idx = 0
    last_pivot_price = prices[0]
    current_trend = 0
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

    for k in range(len(pivot_indices) - 1):
        start_idx = pivot_indices[k]
        end_idx = pivot_indices[k + 1]
        if prices[end_idx] > prices[start_idx]:
            trends[start_idx:end_idx + 1] = 1
        else:
            trends[start_idx:end_idx + 1] = 0
    return pivots, trends


def next_return_sign(
    prices: np.ndarray, horizon: int = 1, ignore_label: int = IGNORE_LABEL
) -> np.ndarray:
    """label[t] = 1 if log(p[t+h]/p[t]) > 0 else 0. tail (length h) is ignored."""
    n = len(prices)
    labels = np.full(n, ignore_label, dtype=np.int64)
    if n <= horizon:
        return labels
    fwd = np.log(prices[horizon:]) - np.log(prices[:-horizon])
    labels[:-horizon] = (fwd > 0).astype(np.int64)
    return labels
