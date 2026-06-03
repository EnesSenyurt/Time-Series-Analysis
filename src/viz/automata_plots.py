"""Visualisations for the probabilistic automaton (Ali's part of the viz layer).

Kept in a dedicated module (separate from the deep-learning / evaluation plots)
so the two owners do not edit the same file. Produces:

* a transition-probability heatmap,
* a state-transition diagram (directed graph),
* a Mermaid ``stateDiagram`` string that can be embedded directly in the report.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend (save to file, no GUI)

import matplotlib.pyplot as plt
import networkx as nx

from src.automata.automaton import ProbabilisticAutomaton


def _top_edges(automaton: ProbabilisticAutomaton, top_k: int) -> list[tuple[str, str, float]]:
    edges = [
        (src, dst, automaton.empirical_prob(src, dst))
        for src, targets in automaton.counts.items()
        for dst in targets
    ]
    edges.sort(key=lambda e: e[2], reverse=True)
    return edges[:top_k]


def to_mermaid(automaton: ProbabilisticAutomaton, top_k: int = 15) -> str:
    """Mermaid ``stateDiagram-v2`` of the strongest transitions (for the report)."""
    lines = ["stateDiagram-v2"]
    for src, dst, prob in _top_edges(automaton, top_k):
        lines.append(f"    {src} --> {dst}: {prob:.2f}")
    return "\n".join(lines)


def plot_transition_heatmap(automaton: ProbabilisticAutomaton, path: str, smoothed: bool = False):
    """Save a heatmap of the transition-probability matrix."""
    states, matrix = automaton.transition_matrix(smoothed=smoothed)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    fig.colorbar(image, ax=ax, label="P(Si -> Sj)")
    ax.set_xlabel("Sj (hedef state)")
    ax.set_ylabel("Si (kaynak state)")
    ax.set_title(f"Gecis Olasiligi Matrisi ({automaton.num_states} state)")
    if automaton.num_states <= 25:
        ax.set_xticks(range(len(states)), states, rotation=90, fontsize=6)
        ax.set_yticks(range(len(states)), states, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_automaton(automaton: ProbabilisticAutomaton, path: str, top_k: int = 30):
    """Save a directed state-transition diagram (edge width ~ probability)."""
    graph = nx.DiGraph()
    edges = _top_edges(automaton, top_k)
    for src, dst, prob in edges:
        graph.add_edge(src, dst, weight=prob)

    fig, ax = plt.subplots(figsize=(10, 8))
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Bos otomata", ha="center")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    pos = nx.spring_layout(graph, seed=42, k=0.8)
    widths = [3.0 * graph[u][v]["weight"] for u, v in graph.edges()]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#bcd", node_size=900)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(
        graph, pos, ax=ax, width=widths, edge_color="#555",
        arrowstyle="-|>", arrowsize=12, connectionstyle="arc3,rad=0.1",
    )
    edge_labels = {(u, v): f"{graph[u][v]['weight']:.2f}" for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=6)
    ax.set_title(f"Otomata State Diyagrami (ilk {len(edges)} gecis)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
