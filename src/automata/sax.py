"""Symbolic Aggregate approXimation (SAX).

A (z-normalised) numeric series is reduced with PAA and each value is mapped to a
symbol using Gaussian break-points, so that under a standard-normal assumption
the symbols are roughly equiprobable.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .paa import paa


def sax_breakpoints(alphabet_size: int) -> np.ndarray:
    """``alphabet_size - 1`` equiprobable Gaussian break-points."""
    quantiles = np.linspace(0.0, 1.0, alphabet_size + 1)[1:-1]
    return norm.ppf(quantiles)


def sax_alphabet(alphabet_size: int) -> list[str]:
    return [chr(ord("a") + i) for i in range(alphabet_size)]


def sax_transform(series, alphabet_size: int, paa_segment_size: int = 1) -> str:
    """Map a (z-normalised) series to a SAX word."""
    values = paa(series, paa_segment_size)
    breakpoints = sax_breakpoints(alphabet_size)
    symbols = sax_alphabet(alphabet_size)
    return "".join(symbols[int(np.searchsorted(breakpoints, v))] for v in values)


def build_sax_dictionary(words) -> set[str]:
    """The set of patterns observed in (training) data."""
    return set(words)
