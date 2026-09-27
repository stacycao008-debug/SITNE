"""Dataset B Train-positive 无向图与确定性多步 random-walk contexts。"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from .data import PairTable


@dataclass(frozen=True)
class PositiveTopologyGraph:
    accessions: tuple[str, ...]
    neighbors: tuple[tuple[int, ...], ...]
    edge_count: int
    isolated_node_count: int

    @property
    def node_count(self) -> int:
        return len(self.accessions)


@dataclass(frozen=True, slots=True)
class WalkContext:
    protein_a: str
    protein_b: str


def build_train_positive_graph(pairs: PairTable) -> PositiveTopologyGraph:
    """仅使用显式 ``label=1``；label=0/未知 pair 永不进入图。"""

    accessions = tuple(sorted(pairs.accessions))
    index = {accession: node for node, accession in enumerate(accessions)}
    adjacency = [set() for _ in accessions]
    seen_edges: set[tuple[int, int]] = set()
    for row in pairs.records:
        if row.label != 1:
            continue
        left = index[row.protein_a]
        right = index[row.protein_b]
        edge = (min(left, right), max(left, right))
        if edge in seen_edges:
            raise ValueError(f"Train-positive graph 重复无向边: {row.undirected_pair_key}")
        seen_edges.add(edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    if not seen_edges:
        raise ValueError("Train-positive graph 没有边")
    neighbors = tuple(tuple(sorted(values)) for values in adjacency)
    isolated = sum(not values for values in neighbors)
    return PositiveTopologyGraph(accessions, neighbors, len(seen_edges), isolated)


def sample_walk_contexts(
    graph: PositiveTopologyGraph,
    walk_length: int,
    walks_per_node: int,
    context_window: int,
    seed: int,
) -> tuple[WalkContext, ...]:
    """在 CPU 上按固定 seed 采样多步 walk，并提取窗口内正 context。

    context 是图上共现信号而非负样本；模型只拉近这些正 context。每个 epoch
    使用独立且可复算的 ``seed``，provenance 记录 seed/参数/行数。
    """

    if walk_length < 2 or walks_per_node <= 0 or context_window <= 0:
        raise ValueError("walk/context 参数非法")
    rng = random.Random(seed)
    contexts: list[WalkContext] = []
    for start in range(graph.node_count):
        if not graph.neighbors[start]:
            continue
        for _ in range(walks_per_node):
            walk = [start]
            current = start
            for _step in range(1, walk_length):
                choices = graph.neighbors[current]
                if not choices:
                    break
                current = choices[rng.randrange(len(choices))]
                walk.append(current)
            for center in range(len(walk)):
                lower = max(0, center - context_window)
                upper = min(len(walk), center + context_window + 1)
                for context_position in range(lower, upper):
                    if context_position == center:
                        continue
                    left = walk[center]
                    right = walk[context_position]
                    if left == right:
                        continue
                    contexts.append(
                        WalkContext(graph.accessions[left], graph.accessions[right])
                    )
    if not contexts:
        raise ValueError("random walk 未产生任何 topology context")
    return tuple(contexts)
