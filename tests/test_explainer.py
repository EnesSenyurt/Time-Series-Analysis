import json

from src.automata.automaton import ProbabilisticAutomaton
from src.explain.explainer import (
    counterfactual,
    explain_decision,
    similarity_report,
    to_json,
    to_table,
)


def _spec_example_automaton():
    # P(aab->abc)=72/100=0.72 ; P(abc->bcc)=15/100=0.15
    transitions = (
        [("aab", "abc")] * 72
        + [("aab", "zzz")] * 28
        + [("abc", "bcc")] * 15
        + [("abc", "yyy")] * 85
    )
    return ProbabilisticAutomaton(alpha=0.0).fit(transitions)


def test_explanation_matches_spec_example(cfg):
    automaton = _spec_example_automaton()
    decision = explain_decision(
        automaton,
        prev_state="aab",
        pattern="adc",
        recent_patterns=["aab", "abc", "bcc"],
        cfg=cfg,
        time_step=5,
    )
    assert decision["status"] == "unseen"
    assert decision["mapped_to"] == "abc"
    assert decision["distance"] == 1
    assert abs(decision["path_probability"] - (0.72 * 0.15)) < 1e-6  # 0.108
    assert decision["decision"] == "anomaly"
    assert decision["confidence_score"] == decision["path_probability"]


def test_output_is_valid_spec_json(cfg):
    automaton = _spec_example_automaton()
    decision = explain_decision(
        automaton, "aab", "adc", ["aab", "abc", "bcc"], cfg, time_step=5
    )
    payload = json.loads(to_json(decision))
    assert payload["time_step"] == 5
    assert payload["status"] == "unseen"
    assert payload["mapped_to"] == "abc"
    assert payload["decision"] == "anomaly"
    assert abs(payload["probability"] - 0.108) < 1e-6


def test_explanation_is_deterministic(cfg):
    automaton = _spec_example_automaton()
    args = ("aab", "adc", ["aab", "abc", "bcc"])
    d1 = explain_decision(automaton, *args, cfg=cfg)
    d2 = explain_decision(automaton, *args, cfg=cfg)
    assert d1 == d2


def test_high_probability_path_is_normal(cfg):
    automaton = _spec_example_automaton()
    decision = explain_decision(
        automaton, "aab", "abc", ["aab", "abc"], cfg
    )  # P=0.72 > 0.5 threshold
    assert decision["status"] == "seen"
    assert decision["decision"] == "normal"


def test_to_table_has_one_row_per_decision(cfg):
    automaton = _spec_example_automaton()
    decisions = [
        explain_decision(automaton, "aab", "abc", ["aab", "abc"], cfg),
        explain_decision(automaton, "aab", "adc", ["aab", "abc", "bcc"], cfg),
    ]
    table = to_table(decisions)
    assert len(table) == 2
    assert {"decision", "path_probability", "status"} <= set(table.columns)


def test_counterfactual_and_similarity(cfg):
    automaton = _spec_example_automaton()
    cf = counterfactual(automaton, "aab", ["abc", "zzz"])
    assert cf[0]["pattern"] == "abc" and abs(cf[0]["transition_prob"] - 0.72) < 1e-6
    sim = similarity_report("adc", automaton.vocab, top_k=2)
    assert sim[0]["pattern"] == "abc" and sim[0]["distance"] == 1
