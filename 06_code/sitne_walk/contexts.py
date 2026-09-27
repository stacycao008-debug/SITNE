"""从 typed walks 在线构造 topology 与 semantic context。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .walks import TypedWalkBatch, _rand


@dataclass(frozen=True)
class TopologyContexts:
    """忽略 relation token 后的有序 protein-protein context pairs。"""

    centers: torch.Tensor
    contexts: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.centers.numel())


@dataclass(frozen=True)
class TypedTransitions:
    """每个有效 walk step 对应的 ``(head, relation, tail)``。"""

    heads: torch.Tensor
    relations: torch.Tensor
    tails: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.heads.numel())


@dataclass(frozen=True)
class TypedRelationContexts:
    """semantic channel 的 ``(protein, relation-token)`` context。"""

    proteins: torch.Tensor
    relations: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.proteins.numel())


def build_topology_contexts(
    walks: TypedWalkBatch,
    window_size: int,
    max_pairs: int | None = None,
    generator: torch.Generator | None = None,
) -> TopologyContexts:
    """构造窗口内双向 protein context。

    关系 token 已由 ``TypedWalkBatch`` 单独存储，因此这里只在 protein 序列上滑动
    窗口。padding ``-1`` 会被严格过滤。
    """

    walks.validate()
    if window_size <= 0:
        raise ValueError("window_size 必须为正")
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs 必须为正或 None")

    proteins = walks.proteins
    sequence_length = proteins.shape[1]
    center_parts: list[torch.Tensor] = []
    context_parts: list[torch.Tensor] = []
    for offset in range(1, min(window_size, sequence_length - 1) + 1):
        left = proteins[:, :-offset]
        right = proteins[:, offset:]
        valid = (left >= 0) & (right >= 0)
        # 即使本 offset 没有有效 pair，也直接加入空张量；这样 CUDA 热路径不会
        # 为 ``valid.any()`` 强制同步到主机。
        center_parts.extend([left[valid], right[valid]])
        context_parts.extend([right[valid], left[valid]])

    if not center_parts:
        empty = torch.empty(0, dtype=torch.int64, device=proteins.device)
        return TopologyContexts(empty, empty)
    centers = torch.cat(center_parts)
    contexts = torch.cat(context_parts)

    if max_pairs is not None and centers.numel() > max_pairs:
        if generator is None:
            raise ValueError("需要 subsample context 时必须提供 generator")
        random_priority = _rand((centers.numel(),), centers.device, generator)
        selected = torch.topk(
            random_priority,
            k=max_pairs,
            largest=False,
            sorted=False,
        ).indices
        centers = centers[selected]
        contexts = contexts[selected]
    return TopologyContexts(centers=centers, contexts=contexts)


def build_typed_transitions(walks: TypedWalkBatch) -> TypedTransitions:
    """提取所有有效的一跳 typed transitions。"""

    walks.validate()
    valid = walks.valid_steps
    heads = walks.proteins[:, :-1][valid]
    relations = walks.relations[valid]
    tails = walks.proteins[:, 1:][valid]
    if bool((heads < 0).any()) or bool((relations < 0).any()) or bool((tails < 0).any()):
        raise RuntimeError("valid_steps 与 walk padding 不一致")
    return TypedTransitions(heads=heads, relations=relations, tails=tails)


def build_typed_relation_contexts(
    walks: TypedWalkBatch,
    window_size: int,
) -> TypedRelationContexts:
    """按交替序列定义构造 protein-to-relation context。

    relation ``r_l`` 位于 ``p_l`` 与 ``p_{l+1}`` 之间。窗口为 1 时只包含这两个
    端点；窗口增大时向两侧继续纳入蛋白质。该定义只把训练图中的 relation token
    当类别监督，不制造不存在的蛋白质对负例。
    """

    walks.validate()
    if window_size <= 0:
        raise ValueError("window_size 必须为正")
    relation_length = walks.relations.shape[1]
    protein_parts: list[torch.Tensor] = []
    relation_parts: list[torch.Tensor] = []

    # delta = protein_position - relation_left_endpoint_position
    for delta in range(-(window_size - 1), window_size + 1):
        relation_start = max(0, -delta)
        relation_end = min(relation_length, relation_length + 1 - delta)
        if relation_start >= relation_end:
            continue
        protein_start = relation_start + delta
        protein_end = relation_end + delta
        relation_slice = walks.relations[:, relation_start:relation_end]
        protein_slice = walks.proteins[:, protein_start:protein_end]
        step_valid = walks.valid_steps[:, relation_start:relation_end]
        valid = step_valid & (relation_slice >= 0) & (protein_slice >= 0)
        protein_parts.append(protein_slice[valid])
        relation_parts.append(relation_slice[valid])

    if not protein_parts:
        empty = torch.empty(0, dtype=torch.int64, device=walks.proteins.device)
        return TypedRelationContexts(proteins=empty, relations=empty)
    return TypedRelationContexts(
        proteins=torch.cat(protein_parts),
        relations=torch.cat(relation_parts),
    )
