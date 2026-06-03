import numpy as np

from src.automata.paa import paa


def test_paa_identity_when_segment_size_1():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(paa(x, 1), x)


def test_paa_aggregates_by_mean():
    x = np.array([1.0, 3.0, 5.0, 7.0])
    assert np.allclose(paa(x, 2), [2.0, 6.0])


def test_paa_handles_non_divisible_length():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(paa(x, 2), [1.5, 3.5, 5.0])
