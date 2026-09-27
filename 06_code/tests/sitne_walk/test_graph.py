from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sitne_walk.data import load_training_triples
from sitne_walk.graph import build_packed_csr_graph
from sitne_walk.walks import AliasTypedWalker, make_generator


def _build(toy_train_tsv: Path, alpha: float = 0.0, beta: float = 0.0):
    triples = load_training_triples(toy_train_tsv)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",) * triples.num_relations,
        alpha=alpha,
        beta=beta,
        epsilon=1.0e-12,
        drop_self_loops=True,
    )
    return triples, graph


def test_graph_statistics_and_self_loop_policy(toy_train_tsv: Path) -> None:
    triples, graph = _build(toy_train_tsv)
    assert graph.num_edges == 10
    # A/B/C/D 各有两个唯一的非自环邻居。
    assert torch.equal(graph.protein_degree, torch.tensor([2.0, 2.0, 2.0, 2.0]))
    # r1 出现在 AB/BC；r2 出现在 AB/CD/AD。CC 自环不计 walk frequency。
    assert torch.equal(graph.relation_frequency, torch.tensor([2.0, 3.0]))


def test_alpha_beta_zero_is_uniform_per_outgoing_typed_edge(toy_train_tsv: Path) -> None:
    triples, graph = _build(toy_train_tsv, alpha=0.0, beta=0.0)
    a = triples.protein_to_id["A"]
    probabilities = graph.transition_probabilities(a)
    assert probabilities.numel() == 3
    assert torch.allclose(probabilities, torch.full((3,), 1 / 3))


def test_alias_table_reconstructs_transition_probabilities(toy_train_tsv: Path) -> None:
    _, graph = _build(toy_train_tsv, alpha=0.7, beta=1.3)
    for node in range(graph.num_proteins):
        start = int(graph.indptr[node])
        end = int(graph.indptr[node + 1])
        size = end - start
        if size == 0:
            continue
        reconstructed = np.zeros(size, dtype=np.float64)
        q = graph.alias_probability[start:end].numpy()
        aliases = graph.alias_local[start:end].numpy()
        for slot in range(size):
            reconstructed[slot] += q[slot] / size
            reconstructed[aliases[slot]] += (1.0 - q[slot]) / size
        expected = graph.transition_probabilities(node).numpy()
        assert np.allclose(reconstructed, expected, atol=1e-6)


def test_empirical_alias_frequency_matches_theory(toy_train_tsv: Path) -> None:
    triples, graph = _build(toy_train_tsv, alpha=0.5, beta=0.5)
    walker = AliasTypedWalker(graph)
    a = triples.protein_to_id["A"]
    current = torch.full((60_000,), a, dtype=torch.int64)
    next_nodes, relations, valid = walker.sample_next(current, make_generator("cpu", 7))
    assert valid.all()
    start = int(graph.indptr[a])
    end = int(graph.indptr[a + 1])
    expected = graph.transition_probabilities(a).numpy()
    observed = []
    for edge in range(start, end):
        observed.append(
            ((next_nodes == graph.destinations[edge]) & (relations == graph.relations[edge]))
            .float()
            .mean()
            .item()
        )
    assert np.allclose(observed, expected, atol=0.015)


def test_unordered_v1_rejects_directed_relation_mode(toy_train_tsv: Path) -> None:
    triples = load_training_triples(toy_train_tsv)
    with pytest.raises(ValueError, match="symmetric"):
        build_packed_csr_graph(
            triples,
            relation_modes=("directed", "symmetric"),
            alpha=0,
            beta=0,
            epsilon=1e-12,
        )


def test_all_self_loop_graph_fails_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "self_loops.tsv"
    path.write_text(
        "protein_i\tprotein_j\ttype_name\nA\tA\tr1\nB\tB\tr1\n",
        encoding="utf-8",
    )
    triples = load_training_triples(path)
    with pytest.raises(ValueError, match="没有可采样边"):
        build_packed_csr_graph(
            triples,
            relation_modes=("symmetric",),
            alpha=0,
            beta=0,
            epsilon=1e-12,
            drop_self_loops=True,
        )


def test_self_loop_frequency_matches_walk_policy(tmp_path: Path) -> None:
    path = tmp_path / "self_loops.tsv"
    path.write_text(
        "protein_i\tprotein_j\ttype_name\nA\tA\tr1\nB\tB\tr1\n",
        encoding="utf-8",
    )
    triples = load_training_triples(path)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",),
        alpha=0,
        beta=0.5,
        epsilon=1e-12,
        drop_self_loops=False,
    )
    assert graph.num_edges == 2
    assert torch.equal(graph.relation_frequency, torch.tensor([2.0]))
