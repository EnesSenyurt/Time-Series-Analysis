"""Levenshtein (edit) distance and nearest-pattern matching for unseen patterns.

When a test pattern is absent from the training SAX dictionary it is mapped to
the closest known pattern by edit distance. Ties are broken lexicographically so
the mapping is fully deterministic and reproducible.
"""
from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (insertions/deletions/substitutions)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def nearest_pattern(pattern: str, vocab) -> tuple[str, int]:
    """Closest pattern in ``vocab`` by edit distance (lexicographic tie-break)."""
    if not vocab:
        raise ValueError("nearest_pattern: vocab bos olamaz")
    best = min(sorted(vocab), key=lambda candidate: (levenshtein(pattern, candidate), candidate))
    return best, levenshtein(pattern, best)
