"""
temporal train/val/test splits for time-series data.

financial series must be split chronologically. an embargo gap of at least
`window` (rolling-stat lookback) + `seq_len` (model window) is placed between
splits so that no input window straddles a boundary.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Split:
    train: slice
    val: slice
    test: slice
    embargo: int

    def as_tuple(self) -> Tuple[slice, slice, slice]:
        return self.train, self.val, self.test


def temporal_split(
    n: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo: int = 0,
) -> Split:
    """
    chronological split with a gap of `embargo` samples between splits.

    n: length of the underlying series.
    embargo: samples to drop between train/val and val/test. set to
        rolling_window + model_seq_len (and any label horizon) to prevent
        any window from spanning a boundary.

    test fraction is the remainder. raises if the splits do not fit.
    """
    if not 0.0 < train_frac < 1.0 or not 0.0 < val_frac < 1.0:
        raise ValueError("fractions must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train + val fractions must leave room for test")
    if embargo < 0:
        raise ValueError("embargo must be non-negative")

    train_end = int(n * train_frac)
    val_start = train_end + embargo
    val_end = val_start + int(n * val_frac)
    test_start = val_end + embargo

    if test_start >= n:
        raise ValueError(
            f"embargo={embargo} too large for n={n} with train={train_frac}, val={val_frac}"
        )

    return Split(
        train=slice(0, train_end),
        val=slice(val_start, val_end),
        test=slice(test_start, n),
        embargo=embargo,
    )


def window_starts(split: slice, seq_len: int) -> np.ndarray:
    """valid window-start indices fully inside `split` for a model needing seq_len samples."""
    start, stop = split.start or 0, split.stop
    last = stop - seq_len
    if last < start:
        return np.empty(0, dtype=np.int64)
    return np.arange(start, last + 1, dtype=np.int64)
