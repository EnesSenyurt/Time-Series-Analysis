"""Piecewise Aggregate Approximation (PAA).

PAA reduces a 1-D series into segment means. With ``segment_size == 1`` it is the
identity; with larger values it down-samples by averaging consecutive samples.
"""
from __future__ import annotations

import numpy as np


def paa(series, segment_size: int) -> np.ndarray:
    """Return segment means of ``series`` using chunks of ``segment_size`` samples.

    Consecutive groups of ``segment_size`` samples are averaged; a trailing
    partial group (when the length is not divisible) becomes its own segment.
    """
    s = np.asarray(series, dtype=float).ravel()
    if segment_size <= 1:
        return s
    return np.array(
        [s[i : i + segment_size].mean() for i in range(0, len(s), segment_size)]
    )
