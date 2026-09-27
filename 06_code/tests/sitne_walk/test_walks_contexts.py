from __future__ import annotations

from pathlib import Path

import torch

from sitne_walk.contexts import (
    build_topology_contexts,
    build_typed_relation_contexts,
    build_typed_transitions,
)
from sitne_walk.data import load_training_triples
from sitne_walk.graph import build_packed_csr_graph
from sitne_walk.walks import AliasTypedWalker, TypedWalkBatch, make_generator


def _graph(toy_train_tsv: Path):
    triples = load_training_triples(toy_train_tsv)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",) * triples.num_relations,
        alpha=0.5,
        beta=0.5,
        epsilon=1e-12,
    )
    return triples, graph


def test_walk_is_reproducible_and_uses_only_train_edges(toy_train_tsv: Path) -> None:
    triples, graph = _graph(toy_train_tsv)
    walker = AliasTypedWalker(graph)
    starts = torch.arange(triples.num_proteins)
    first = walker.generate(starts, 8, make_generator("cpu", 123))
    second = walker.generate(starts, 8, make_generator("cpu", 123))
    assert torch.equal(first.proteins, second.proteins)
    assert torch.equal(first.relations, second.relations)

    valid_edges = set()
    for node in range(graph.num_proteins):
        for edge in range(int(graph.indptr[node]), int(graph.indptr[node + 1])):
            valid_edges.add(
                (node, int(graph.relations[edge]), int(graph.destinations[edge]))
            )
    transitions = build_typed_transitions(first)
    for triple in zip(
        transitions.heads.tolist(),
        transitions.relations.tolist(),
        transitions.tails.tolist(),
    ):
        assert triple in valid_edges


def test_dead_end_is_padded(tmp_path: Path) -> None:
    path = tmp_path / "train.tsv"
    path.write_text(
        "protein_i\tprotein_j\ttype_name\nA\tB\tr1\nC\tC\tr1\n",
        encoding="utf-8",
    )
    triples = load_training_triples(path)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",),
        alpha=0,
        beta=0,
        epsilon=1e-12,
        drop_self_loops=True,
    )
    c = triples.protein_to_id["C"]
    walk = AliasTypedWalker(graph).generate(
        torch.tensor([c]), 4, make_generator("cpu", 3)
    )
    assert walk.proteins.tolist() == [[c, -1, -1, -1, -1]]
    assert walk.relations.tolist() == [[-1, -1, -1, -1]]
    assert not walk.valid_steps.any()


def test_contexts_match_hand_constructed_walk() -> None:
    walks = TypedWalkBatch(
        proteins=torch.tensor([[0, 1, 2]]),
        relations=torch.tensor([[3, 4]]),
        valid_steps=torch.tensor([[True, True]]),
    )
    topology = build_topology_contexts(walks, window_size=1)
    assert list(zip(topology.centers.tolist(), topology.contexts.tolist())) == [
        (0, 1),
        (1, 2),
        (1, 0),
        (2, 1),
    ]
    typed = build_typed_relation_contexts(walks, window_size=1)
    assert list(zip(typed.proteins.tolist(), typed.relations.tolist())) == [
        (0, 3),
        (1, 4),
        (1, 3),
        (2, 4),
    ]

