"""Sliding-window pattern extraction over a SAX symbol string.

Each length-``w`` substring is a *pattern* (an automaton state); consecutive
patterns form the transitions used to learn the probabilistic automaton.
"""
from __future__ import annotations


def sliding_patterns(symbol_string: str, w: int) -> list[str]:
    """All length-``w`` contiguous substrings of ``symbol_string`` (step 1)."""
    if len(symbol_string) < w:
        return []
    return [symbol_string[i : i + w] for i in range(len(symbol_string) - w + 1)]


def transitions_from_patterns(patterns: list[str]) -> list[tuple[str, str]]:
    """Consecutive (pattern_t, pattern_t+1) pairs."""
    return list(zip(patterns[:-1], patterns[1:]))
