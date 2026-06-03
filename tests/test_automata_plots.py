from src.automata.automaton import ProbabilisticAutomaton
from src.viz.automata_plots import plot_automaton, plot_transition_heatmap, to_mermaid


def _automaton():
    return ProbabilisticAutomaton(alpha=0.0).fit(
        [("aab", "abc"), ("aab", "abc"), ("abc", "bcc"), ("abc", "aab")]
    )


def test_to_mermaid_contains_transitions():
    mermaid = to_mermaid(_automaton())
    assert "stateDiagram" in mermaid
    assert "aab --> abc" in mermaid


def test_plots_write_files(tmp_path):
    automaton = _automaton()
    diagram = tmp_path / "automaton.png"
    heatmap = tmp_path / "heatmap.png"
    plot_automaton(automaton, str(diagram))
    plot_transition_heatmap(automaton, str(heatmap))
    assert diagram.exists() and diagram.stat().st_size > 0
    assert heatmap.exists() and heatmap.stat().st_size > 0
