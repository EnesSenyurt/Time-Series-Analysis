import numpy as np

from src.automata.automaton import (
    ProbabilisticAutomaton,
    anomaly_scores,
    build_automaton_from_segments,
    pattern_sequence,
)


def test_transition_probabilities_frequency_based():
    a = ProbabilisticAutomaton(alpha=0.0).fit(
        [("s1", "s2"), ("s1", "s2"), ("s1", "s3")]
    )
    assert abs(a.prob("s1", "s2") - 2 / 3) < 1e-9
    assert abs(a.prob("s1", "s3") - 1 / 3) < 1e-9


def test_laplace_smoothing_nonzero_for_unseen_target():
    a = ProbabilisticAutomaton(alpha=1.0).fit([("s1", "s2")])
    assert a.prob("s1", "s2") > 0
    assert 0 < a.prob("s1", "s_unknown") < a.prob("s1", "s2")


def test_path_probability_is_product():
    a = ProbabilisticAutomaton(alpha=0.0).fit(
        [("aab", "abc"), ("aab", "abc"), ("abc", "bcc"), ("abc", "bcd")]
    )
    # P(aab->abc)=1.0 ; P(abc->bcc)=0.5
    assert abs(a.path_probability(["aab", "abc", "bcc"]) - 0.5) < 1e-9


def test_explain_step_unseen_uses_levenshtein_mapping():
    a = ProbabilisticAutomaton(alpha=1.0).fit([("aab", "abc"), ("abc", "bcc")])
    out = a.explain_step(prev="aab", pattern="adc")
    assert out["status"] == "unseen"
    assert out["mapped_to"] == "abc"
    assert out["distance"] == 1


def test_structure_metrics():
    a = ProbabilisticAutomaton(alpha=0.0).fit(
        [("s1", "s2"), ("s1", "s3"), ("s2", "s3")]
    )
    assert a.num_states == 3
    assert a.num_transitions() == 3  # distinct edges: s1->s2, s1->s3, s2->s3
    assert abs(a.transition_density() - 3 / 9) < 1e-9
    states, matrix = a.transition_matrix(smoothed=False)
    assert matrix.shape == (3, 3)


def test_build_and_score_flags_abnormal_segment(cfg):
    # Repetitive (sine) training -> stable, high-probability transitions.
    normal_train = np.sin(np.linspace(0, 20 * np.pi, 400))
    automaton, params = build_automaton_from_segments(
        [normal_train], cfg, window_size=4, alphabet_size=3
    )
    normal_test = pattern_sequence(np.sin(np.linspace(0, 10 * np.pi, 200)), params)
    rng = np.random.default_rng(0)
    abnormal_test = pattern_sequence(rng.normal(size=200) * 3.0, params)

    horizon = cfg.automaton.path_horizon
    normal_score = anomaly_scores(automaton, normal_test, horizon).mean()
    abnormal_score = anomaly_scores(automaton, abnormal_test, horizon).mean()
    assert abnormal_score > normal_score
