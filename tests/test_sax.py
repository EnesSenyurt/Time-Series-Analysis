import numpy as np

from src.automata.sax import (
    build_sax_dictionary,
    sax_breakpoints,
    sax_transform,
)


def test_breakpoints_alphabet_3_are_symmetric():
    bp = sax_breakpoints(3)
    assert len(bp) == 2
    assert bp[0] < 0 < bp[1]
    assert np.isclose(bp[0], -bp[1])


def test_sax_transform_increasing_series():
    word = sax_transform(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]), alphabet_size=3)
    assert word == "aabcc"


def test_sax_transform_word_length_matches_input():
    word = sax_transform(np.linspace(-3, 3, 6), alphabet_size=4, paa_segment_size=1)
    assert len(word) == 6
    assert set(word) <= set("abcd")


def test_dictionary_is_set_of_seen_words():
    assert build_sax_dictionary(["aab", "abc", "aab"]) == {"aab", "abc"}
