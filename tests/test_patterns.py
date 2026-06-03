from src.automata.patterns import sliding_patterns, transitions_from_patterns


def test_sliding_window_words():
    assert sliding_patterns("abcde", 3) == ["abc", "bcd", "cde"]


def test_window_equal_to_length_returns_single_pattern():
    assert sliding_patterns("abc", 3) == ["abc"]


def test_window_larger_than_series_returns_empty():
    assert sliding_patterns("ab", 3) == []


def test_transitions_are_consecutive_pairs():
    patterns = ["abc", "bcd", "cde"]
    assert transitions_from_patterns(patterns) == [("abc", "bcd"), ("bcd", "cde")]


def test_transitions_empty_for_single_pattern():
    assert transitions_from_patterns(["abc"]) == []
