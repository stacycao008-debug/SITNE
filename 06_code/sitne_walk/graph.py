"""Train-only packed CSR 图与分段 Alias Table。

随机游走的热点路径只使用 PyTorch 张量，因此把本对象迁移到 ``cuda`` 后，
下一跳采样不会回到 NumPy/Pandas 或产生逐节点 Python 循环。Alias Table 的构建
是一次性的 CPU 预处理；正式游走则在目标设备批量执行。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np
import torch

from .data import TrainingTriples

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PackedCSRGraph:
    """按源蛋白质压缩的 typed adjacency 与 Alias Table。"""

    indptr: torch.Tensor
    destinations: torch.Tensor
    relations: torch.Tensor
    edge_weights: torch.Tensor
    transition_weights: torch.Tensor
    alias_probability: torch.Tensor
    alias_local: torch.Tensor
    protein_degree: torch.Tensor
    relation_frequency: torch.Tensor
    symmetric_relation_mask: torch.Tensor
    relation_modes: tuple[str, ...]
    alpha: float
    beta: float
    epsilon: float

    @property
    def num_proteins(self) -> int:
        return int(self.indptr.numel() - 1)

    @property
    def num_relations(self) -> int:
        return int(self.relation_frequency.numel())

    @property
    def num_edges(self) -> int:
        return int(self.destinations.numel())

    @property
    def device(self) -> torch.device:
        return self.indptr.device

    @property
    def nodes_with_outgoing_edges(self) -> torch.Tensor:
        """返回至少有一个可采样出边的节点。"""

        degrees = self.indptr[1:] - self.indptr[:-1]
        return torch.nonzero(degrees > 0, as_tuple=False).flatten()

    def to(self, device: torch.device | str) -> "PackedCSRGraph":
        """把图及 Alias Table 一次性迁移到 CPU、CUDA 或 MPS。"""

        return PackedCSRGraph(
            indptr=self.indptr.to(device),
            destinations=self.destinations.to(device),
            relations=self.relations.to(device),
            edge_weights=self.edge_weights.to(device),
            transition_weights=self.transition_weights.to(device),
            alias_probability=self.alias_probability.to(device),
            alias_local=self.alias_local.to(device),
            protein_degree=self.protein_degree.to(device),
            relation_frequency=self.relation_frequency.to(device),
            symmetric_relation_mask=self.symmetric_relation_mask.to(device),
            relation_modes=self.relation_modes,
            alpha=self.alpha,
            beta=self.beta,
            epsilon=self.epsilon,
        )

    def validate(self) -> None:
        """检查 CSR 与 Alias Table 的结构不变量。"""

        if self.indptr.dtype != torch.int64:
            raise ValueError("indptr 必须是 int64")
        if self.indptr.ndim != 1 or self.indptr.numel() < 2:
            raise ValueError("indptr 必须是一维且至少含两个元素")
        if int(self.indptr[0].item()) != 0:
            raise ValueError("indptr[0] 必须为 0")
        if bool((self.indptr[1:] < self.indptr[:-1]).any()):
            raise ValueError("indptr 必须单调不减")
        if int(self.indptr[-1].item()) != self.num_edges:
            raise ValueError("indptr[-1] 与边数不一致")

        edge_tensors = (
            self.destinations,
            self.relations,
            self.edge_weights,
            self.transition_weights,
            self.alias_probability,
            self.alias_local,
        )
        if any(tensor.ndim != 1 or tensor.numel() != self.num_edges for tensor in edge_tensors):
            raise ValueError("所有 edge-level 张量必须是一维且长度等于边数")
        if self.num_edges == 0:
            raise ValueError("图没有可采样边")
        if bool((self.destinations < 0).any()) or bool(
            (self.destinations >= self.num_proteins).any()
        ):
            raise ValueError("destinations 越界")
        if bool((self.relations < 0).any()) or bool(
            (self.relations >= self.num_relations).any()
        ):
            raise ValueError("relations 越界")
        if not bool(torch.isfinite(self.transition_weights).all()) or bool(
            (self.transition_weights <= 0).any()
        ):
            raise ValueError("transition_weights 必须为有限正数")
        if bool((self.alias_probability < 0).any()) or bool(
            (self.alias_probability > 1).any()
        ):
            raise ValueError("alias_probability 必须位于 [0, 1]")
        if bool((self.alias_local < 0).any()):
            raise ValueError("alias_local 不能为负")
        row_sizes = self.indptr[1:] - self.indptr[:-1]
        edge_row_sizes = torch.repeat_interleave(row_sizes, row_sizes)
        if bool((self.alias_local.to(torch.int64) >= edge_row_sizes).any()):
            raise ValueError("alias_local 超出对应 CSR segment")

    def transition_probabilities(self, node_id: int) -> torch.Tensor:
        """返回某节点的理论下一跳概率，主要用于审计和单元测试。"""

        if not 0 <= node_id < self.num_proteins:
            raise IndexError(f"node_id 越界: {node_id}")
        start = int(self.indptr[node_id].item())
        end = int(self.indptr[node_id + 1].item())
        weights = self.transition_weights[start:end]
        if weights.numel() == 0:
            return weights
        return weights / weights.sum()


def _coalesce_expanded_edges(
    sources: np.ndarray,
    relations: np.ndarray,
    destinations: np.ndarray,
    weights: np.ndarray,
    num_proteins: int,
    num_relations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """合并方向扩展产生的重复边，取最大权重避免正反记录被重复计数。"""

    if len(sources) == 0:
        raise ValueError(
            "训练图没有可采样边；若训练集只有 self-loop，"
            "请检查数据或显式调整 drop_self_loops_from_walks"
        )

    encoded = (
        sources.astype(np.int64) * np.int64(num_relations * num_proteins)
        + relations.astype(np.int64) * np.int64(num_proteins)
        + destinations.astype(np.int64)
    )
    order = np.argsort(encoded, kind="stable")
    encoded = encoded[order]
    sources = sources[order]
    relations = relations[order]
    destinations = destinations[order]
    weights = weights[order]

    first = np.empty(len(encoded), dtype=np.bool_)
    first[0] = True
    first[1:] = encoded[1:] != encoded[:-1]
    group_ids = np.cumsum(first, dtype=np.int64) - 1
    unique_count = int(group_ids[-1]) + 1
    reduced_weights = np.full(unique_count, -np.inf, dtype=np.float64)
    np.maximum.at(reduced_weights, group_ids, weights.astype(np.float64))
    starts = np.flatnonzero(first)
    return (
        sources[starts],
        relations[starts],
        destinations[starts],
        reduced_weights.astype(np.float32),
    )


def _build_segmented_alias(
    transition_weights: np.ndarray,
    indptr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """为每个 CSR segment 构建 Vose Alias Table。

    ``alias_index`` 保存全局 edge index，这样 CUDA 采样只需一次 gather。
    """

    alias_probability = np.ones(len(transition_weights), dtype=np.float32)
    alias_local = np.zeros(len(transition_weights), dtype=np.int32)

    for node in range(len(indptr) - 1):
        start = int(indptr[node])
        end = int(indptr[node + 1])
        segment_size = end - start
        if segment_size <= 1:
            continue

        weights = transition_weights[start:end].astype(np.float64, copy=False)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"节点 {node} 的 transition weight 总和非法: {total}")
        scaled = weights / total * segment_size
        small = [index for index, value in enumerate(scaled) if value < 1.0]
        large = [index for index, value in enumerate(scaled) if value >= 1.0]

        while small and large:
            small_index = small.pop()
            large_index = large.pop()
            alias_probability[start + small_index] = np.float32(scaled[small_index])
            alias_local[start + small_index] = large_index
            scaled[large_index] = scaled[large_index] - (1.0 - scaled[small_index])
            if scaled[large_index] < 1.0:
                small.append(large_index)
            else:
                large.append(large_index)

        # 数值舍入可能留下若干桶；它们的接受概率应为 1。
        for index in small + large:
            alias_probability[start + index] = np.float32(1.0)
            alias_local[start + index] = index

    return alias_probability, alias_local


def build_packed_csr_graph(
    triples: TrainingTriples,
    relation_modes: Sequence[str],
    alpha: float,
    beta: float,
    epsilon: float,
    drop_self_loops: bool = True,
    correction_mode: str = "rank",
) -> PackedCSRGraph:
    """从训练三元组构建偏差校正图。

    ``d_j`` 使用原始、去重后的 train-only 三元组计算唯一非自环邻居数；
    ``f_k`` 使用实际进入 walk 的唯一 train pair-type 频数。只有显式标为
    ``symmetric`` 的关系才添加反向边，``directed`` 和 ``as_recorded`` 均保持
    输入方向。

    correction_mode:
        - "log": 原始公式 -alpha*log(d_j) -beta*log(f_k)
        - "rank": rank-normalized 公式 -alpha*rank(d_j)/N -beta*log(f_k)
          对 degree 使用分位数秩归一化，避免长尾分布导致过度校正
    """

    if len(relation_modes) != triples.num_relations:
        raise ValueError("relation_modes 长度必须等于训练关系数")
    if any(mode not in {"as_recorded", "directed", "symmetric"} for mode in relation_modes):
        raise ValueError("relation_modes 含非法取值")
    if alpha < 0 or beta < 0 or epsilon <= 0:
        raise ValueError("alpha/beta 必须非负且 epsilon 必须为正")
    if triples.pair_semantics != "unordered":
        raise ValueError("SITNE-Walk v1 只支持 unordered pair semantics")
    if any(mode != "symmetric" for mode in relation_modes):
        raise ValueError("unordered SITNE-Walk v1 的所有关系必须显式为 symmetric")

    source = triples.heads.numpy().astype(np.int64, copy=False)
    relation = triples.relations.numpy().astype(np.int64, copy=False)
    destination = triples.tails.numpy().astype(np.int64, copy=False)
    weights = triples.edge_weights.numpy().astype(np.float64, copy=False)

    # v1 的 degree 按唯一非自环邻居计数，而不是按 typed edge 数计数；这样同一
    # pair 拥有多个标签不会人为提高 hub 权重。
    non_self = source != destination
    pair_codes = source[non_self] * np.int64(triples.num_proteins) + destination[non_self]
    unique_pair_positions = np.unique(pair_codes, return_index=True)[1]
    degree_source = source[non_self][unique_pair_positions]
    degree_destination = destination[non_self][unique_pair_positions]
    protein_degree = np.zeros(triples.num_proteins, dtype=np.float64)
    np.add.at(protein_degree, degree_source, 1.0)
    np.add.at(protein_degree, degree_destination, 1.0)
    symmetric_mask = np.array(
        [mode == "symmetric" for mode in relation_modes], dtype=np.bool_
    )
    walk_mask = non_self if drop_self_loops else np.ones_like(non_self)
    # relation frequency 必须与实际 walk 边集合一致。默认排除 self-loop；若调用方
    # 显式允许 self-loop 游走，则它们也进入 f_k，避免只含自环的关系频数为零。
    relation_frequency = np.zeros(triples.num_relations, dtype=np.float64)
    np.add.at(relation_frequency, relation[walk_mask], 1.0)
    base_source = source[walk_mask]
    base_relation = relation[walk_mask]
    base_destination = destination[walk_mask]
    base_weights = weights[walk_mask]
    reverse_mask = symmetric_mask[base_relation] & (base_source != base_destination)
    expanded_source = np.concatenate([base_source, base_destination[reverse_mask]])
    expanded_relation = np.concatenate([base_relation, base_relation[reverse_mask]])
    expanded_destination = np.concatenate([base_destination, base_source[reverse_mask]])
    expanded_weights = np.concatenate([base_weights, base_weights[reverse_mask]])

    (
        expanded_source,
        expanded_relation,
        expanded_destination,
        expanded_weights,
    ) = _coalesce_expanded_edges(
        expanded_source,
        expanded_relation,
        expanded_destination,
        expanded_weights,
        num_proteins=triples.num_proteins,
        num_relations=triples.num_relations,
    )

    # CSR 要求所有相同 source 的边连续；次关键字使构建结果可复现。
    order = np.lexsort(
        (expanded_destination, expanded_relation, expanded_source)
    )
    expanded_source = expanded_source[order]
    expanded_relation = expanded_relation[order]
    expanded_destination = expanded_destination[order]
    expanded_weights = expanded_weights[order]

    counts = np.bincount(expanded_source, minlength=triples.num_proteins)
    indptr = np.empty(triples.num_proteins + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    destination_degree = protein_degree[expanded_destination]
    relation_count = relation_frequency[expanded_relation]

    # 根据 correction_mode 选择 degree correction 公式
    if correction_mode == "log":
        degree_correction = alpha * np.log(destination_degree + epsilon)
    elif correction_mode == "rank":
        from scipy.stats import rankdata
        degree_ranks = rankdata(protein_degree, method='average')
        normalized_ranks = degree_ranks / float(len(protein_degree))
        degree_correction = alpha * normalized_ranks[expanded_destination]
    else:
        raise ValueError(f"未知 correction_mode: {correction_mode}")

    log_weights = (
        np.log(expanded_weights.astype(np.float64))
        - degree_correction
        - beta * np.log(relation_count + epsilon)
    )
    # 每行减去最大 log-weight；只改变公共比例，不改变理论转移概率。
    row_max = np.full(triples.num_proteins, -np.inf, dtype=np.float64)
    np.maximum.at(row_max, expanded_source, log_weights)
    transition_weights = np.exp(log_weights - row_max[expanded_source])
    if not np.isfinite(transition_weights).all() or np.any(transition_weights <= 0):
        raise ValueError("校正后的 transition weights 出现非有限值或非正值")

    alias_probability, alias_local = _build_segmented_alias(
        transition_weights,
        indptr,
    )
    graph = PackedCSRGraph(
        indptr=torch.from_numpy(indptr),
        destinations=torch.from_numpy(expanded_destination.astype(np.int64, copy=False)),
        relations=torch.from_numpy(expanded_relation.astype(np.int64, copy=False)),
        edge_weights=torch.from_numpy(expanded_weights.astype(np.float32, copy=False)),
        transition_weights=torch.from_numpy(
            transition_weights.astype(np.float32, copy=False)
        ),
        alias_probability=torch.from_numpy(alias_probability),
        alias_local=torch.from_numpy(alias_local),
        protein_degree=torch.from_numpy(protein_degree.astype(np.float32)),
        relation_frequency=torch.from_numpy(relation_frequency.astype(np.float32)),
        symmetric_relation_mask=torch.from_numpy(symmetric_mask),
        relation_modes=tuple(relation_modes),
        alpha=float(alpha),
        beta=float(beta),
        epsilon=float(epsilon),
    )
    graph.validate()
    logger.info(
        "Packed CSR graph: proteins=%d relations=%d edges=%d alpha=%.4g beta=%.4g",
        graph.num_proteins,
        graph.num_relations,
        graph.num_edges,
        alpha,
        beta,
    )
    return graph
