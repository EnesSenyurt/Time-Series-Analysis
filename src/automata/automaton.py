"""Probabilistic automaton over SAX patterns.

States are unique SAX patterns; transition probabilities are learned frequency-
based with optional Laplace smoothing:

    P(Si -> Sj) = (count(Si -> Sj) + alpha) / (sum_k count(Si -> Sk) + alpha * |V|)

The probability of a pattern sequence is the product of consecutive transition
probabilities. Unseen patterns are mapped to the nearest known pattern via
Levenshtein distance before their probability is evaluated.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .levenshtein import nearest_pattern
from .patterns import sliding_patterns, transitions_from_patterns
from .sax import sax_transform


class ProbabilisticAutomaton:
    """Frequency-based probabilistic automaton with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.vocab: set[str] = set()
        self.states: list[str] = []

    def fit(self, transitions) -> "ProbabilisticAutomaton":
        for src, dst in transitions:
            self.counts[src][dst] += 1
            self.vocab.update((src, dst))
        self.states = sorted(self.vocab)
        return self

    # -- probabilities ----------------------------------------------------- #
    def prob(self, src: str, dst: str) -> float:
        """Smoothed transition probability P(src -> dst)."""
        out = self.counts.get(src, {})
        total = sum(out.values())
        vocab_size = len(self.vocab) if self.vocab else 1
        denom = total + self.alpha * vocab_size
        if denom == 0:
            return 0.0
        return (out.get(dst, 0) + self.alpha) / denom

    def empirical_prob(self, src: str, dst: str) -> float:
        """Unsmoothed (count-based) transition probability."""
        out = self.counts.get(src, {})
        total = sum(out.values())
        return out.get(dst, 0) / total if total else 0.0

    def map_pattern(self, pattern: str) -> tuple[str, int, str]:
        """Return (mapped_pattern, distance, status) handling unseen patterns."""
        if pattern in self.vocab:
            return pattern, 0, "seen"
        nearest, distance = nearest_pattern(pattern, self.vocab)
        return nearest, distance, "unseen"

    def path_probability(self, patterns) -> float:
        """Product of consecutive (mapped) transition probabilities."""
        prob = 1.0
        for src, dst in zip(patterns[:-1], patterns[1:]):
            mapped_src, _, _ = self.map_pattern(src)
            mapped_dst, _, _ = self.map_pattern(dst)
            prob *= self.prob(mapped_src, mapped_dst)
        return prob

    def explain_step(self, prev: str, pattern: str) -> dict:
        """Single-step explanation record used by the explainability module."""
        mapped, distance, status = self.map_pattern(pattern)
        mapped_prev, _, _ = self.map_pattern(prev)
        return {
            "previous_state": prev,
            "pattern": pattern,
            "status": status,
            "mapped_to": mapped if status == "unseen" else pattern,
            "distance": distance,
            "transition_prob": self.prob(mapped_prev, mapped),
        }

    # -- structure metrics (for reporting / heatmaps) ---------------------- #
    @property
    def num_states(self) -> int:
        return len(self.states)

    def num_transitions(self) -> int:
        """Number of distinct observed (src -> dst) edges."""
        return sum(len(targets) for targets in self.counts.values())

    def transition_density(self) -> float:
        """Observed edges divided by the number of possible edges (states^2)."""
        n = self.num_states
        return self.num_transitions() / (n * n) if n else 0.0

    def transition_matrix(self, smoothed: bool = False):
        """Return (states, matrix) where matrix[i, j] = P(states[i] -> states[j])."""
        prob_fn = self.prob if smoothed else self.empirical_prob
        states = self.states
        matrix = np.zeros((len(states), len(states)))
        for i, src in enumerate(states):
            for j, dst in enumerate(states):
                matrix[i, j] = prob_fn(src, dst)
        return states, matrix


# --------------------------------------------------------------------------- #
# Symbolisation + automaton construction from a 1-D (PC1) series
# --------------------------------------------------------------------------- #
@dataclass
class SymbolizationParams:
    """Parameters needed to turn a 1-D series into SAX patterns (train-fitted)."""

    window_size: int
    alphabet_size: int
    paa_segment_size: int
    mu: float
    sd: float


def fit_symbolizer(series_1d, cfg, window_size: int, alphabet_size: int) -> SymbolizationParams:
    """Fit z-normalisation statistics on the (training) series."""
    s = np.asarray(series_1d, dtype=float).ravel()
    return SymbolizationParams(
        window_size=window_size,
        alphabet_size=alphabet_size,
        paa_segment_size=cfg.automaton.paa_segment_size,
        mu=float(s.mean()),
        sd=float(s.std()),
    )


def symbolize(series_1d, params: SymbolizationParams) -> str:
    """z-normalise (with train stats) then map to a SAX symbol string."""
    s = np.asarray(series_1d, dtype=float).ravel()
    z = (s - params.mu) / params.sd if params.sd > 1e-12 else s - params.mu
    return sax_transform(z, params.alphabet_size, params.paa_segment_size)


def pattern_sequence(series_1d, params: SymbolizationParams) -> list[str]:
    """SAX patterns (sliding window) for a 1-D series."""
    return sliding_patterns(symbolize(series_1d, params), params.window_size)


def build_automaton_from_segments(
    segments_1d, cfg, window_size: int, alphabet_size: int
) -> tuple[ProbabilisticAutomaton, SymbolizationParams]:
    """Fit a probabilistic automaton from one or more continuous 1-D segments.

    Transitions are gathered *within* each segment (never across segment/file
    boundaries) and pooled before fitting.
    """
    segments = [np.asarray(seg, dtype=float).ravel() for seg in segments_1d]
    concatenated = np.concatenate(segments) if segments else np.array([0.0])
    params = fit_symbolizer(concatenated, cfg, window_size, alphabet_size)

    transitions: list[tuple[str, str]] = []
    for segment in segments:
        transitions += transitions_from_patterns(pattern_sequence(segment, params))

    automaton = ProbabilisticAutomaton(alpha=cfg.automaton.smoothing_alpha).fit(transitions)
    return automaton, params


def anomaly_scores(automaton: ProbabilisticAutomaton, patterns, horizon: int) -> np.ndarray:
    """Per-transition anomaly score = -log(local path probability).

    Returns an array of length ``len(patterns) - 1`` (one score per transition).
    Higher score => lower-probability path => more anomalous.
    """
    scores = []
    for i in range(1, len(patterns)):
        low = max(0, i - horizon)
        local = automaton.path_probability(patterns[low : i + 1])
        scores.append(-np.log(local + 1e-12))
    return np.asarray(scores)
