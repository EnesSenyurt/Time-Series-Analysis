from src.automata.levenshtein import levenshtein, nearest_pattern


def test_distance_identical_is_zero():
    assert levenshtein("aab", "aab") == 0


def test_distance_single_substitution():
    # spec example: "adc" vs "abc" -> 1
    assert levenshtein("adc", "abc") == 1


def test_distance_classic_kitten_sitting():
    assert levenshtein("kitten", "sitting") == 3


def test_distance_with_empty_string():
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3


def test_nearest_pattern_picks_min_distance():
    pattern, distance = nearest_pattern("adc", {"abc", "xyz", "aaa"})
    assert pattern == "abc"
    assert distance == 1


def test_nearest_pattern_deterministic_tie_break():
    # equal distance (1) -> lexicographically smallest ("ac" < "ad")
    pattern, distance = nearest_pattern("ab", {"ad", "ac"})
    assert pattern == "ac"
    assert distance == 1
