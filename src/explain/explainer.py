"""Probabilistic explainability module.

For every decision the automaton makes, this module produces a deterministic,
reproducible record that justifies the decision through the probabilistic
transitions of the automaton: the current state, the observed pattern, whether
it was seen in training, the unseen-mapping mechanism, the realised transitions
and their probabilities, the total path probability, the confidence score, and
the final decision with its probabilistic rationale.

The mandatory output format is JSON (see :func:`to_json`); a tabular form is also
available via :func:`to_table`.
"""
from __future__ import annotations

import json

import pandas as pd

from src.automata.automaton import ProbabilisticAutomaton
from src.automata.levenshtein import levenshtein


def explain_decision(
    automaton: ProbabilisticAutomaton,
    prev_state: str,
    pattern: str,
    recent_patterns: list[str],
    cfg,
    *,
    threshold: float | None = None,
    time_step: int | None = None,
) -> dict:
    """Build the full explanation record for a single decision.

    ``recent_patterns`` is the local path (e.g. the last ``path_horizon + 1``
    patterns) whose probability drives the decision.
    """
    if threshold is None:
        threshold = cfg.automaton.explain_threshold

    mapped, distance, status = automaton.map_pattern(pattern)

    transitions = []
    for src, dst in zip(recent_patterns[:-1], recent_patterns[1:]):
        mapped_src, _, _ = automaton.map_pattern(src)
        mapped_dst, _, _ = automaton.map_pattern(dst)
        transitions.append(
            {"from": src, "to": dst, "prob": round(automaton.prob(mapped_src, mapped_dst), 6)}
        )

    path_probability = automaton.path_probability(recent_patterns)
    is_anomaly = path_probability < threshold

    return {
        "time_step": time_step,
        "state": prev_state,
        "pattern": pattern,
        "status": status,
        "mapped_to": mapped if status == "unseen" else None,
        "distance": distance if status == "unseen" else None,
        "transitions": transitions,
        "path_probability": path_probability,
        "confidence_score": path_probability,
        "decision": "anomaly" if is_anomaly else "normal",
        "rationale": (
            "Low probability path detected"
            if is_anomaly
            else "High probability path"
        ),
    }


def to_json(decision: dict) -> str:
    """Spec-compliant JSON for a decision record (Section X.F)."""
    payload = {
        "time_step": decision["time_step"],
        "state": decision["state"],
        "pattern": decision["pattern"],
        "status": decision["status"],
        "mapped_to": decision["mapped_to"],
        "probability": round(decision["path_probability"], 6),
        "decision": decision["decision"],
    }
    return json.dumps(payload, ensure_ascii=False)


def to_table(decisions: list[dict]) -> pd.DataFrame:
    """Tabular view of multiple decision records."""
    columns = [
        "time_step",
        "state",
        "pattern",
        "status",
        "mapped_to",
        "distance",
        "path_probability",
        "confidence_score",
        "decision",
    ]
    return pd.DataFrame([{c: d.get(c) for c in columns} for d in decisions])


# --------------------------------------------------------------------------- #
# Optional advanced analyses (bonus)
# --------------------------------------------------------------------------- #
def counterfactual(automaton: ProbabilisticAutomaton, prev_state: str, alt_patterns: list[str]) -> list[dict]:
    """How the transition probability changes under alternative incoming patterns."""
    mapped_prev, _, _ = automaton.map_pattern(prev_state)
    results = []
    for alt in alt_patterns:
        mapped, distance, status = automaton.map_pattern(alt)
        results.append(
            {
                "pattern": alt,
                "status": status,
                "mapped_to": mapped if status == "unseen" else alt,
                "distance": distance if status == "unseen" else 0,
                "transition_prob": round(automaton.prob(mapped_prev, mapped), 6),
            }
        )
    return results


def similarity_report(pattern: str, vocab, top_k: int = 3) -> list[dict]:
    """Closest known patterns (by edit distance) for an unseen pattern."""
    scored = sorted(((levenshtein(pattern, v), v) for v in vocab), key=lambda x: (x[0], x[1]))
    return [{"pattern": v, "distance": d} for d, v in scored[:top_k]]
