"""泄漏边界明确的 filtered interaction-type ranking evaluator。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .data import IndexedTriples, PairRelationIndex, iter_batches
from .model import SITNEWalkModel


@dataclass(frozen=True)
class RankingResult:
    """包含覆盖率、逐查询 rank 和汇总指标的评测结果。"""

    metrics: Mapping[str, float]
    ranks: torch.Tensor
    relation_ids: torch.Tensor
    raw_input_rows: int
    input_rows: int
    supported_rows: int
    unsupported_protein_rows: int
    unsupported_relation_rows: int
    status: str

    @property
    def coverage(self) -> float:
        return self.supported_rows / self.input_rows if self.input_rows else 0.0

    def to_dict(self, include_ranks: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "status": self.status,
            "raw_input_rows": self.raw_input_rows,
            "input_rows": self.input_rows,
            "supported_rows": self.supported_rows,
            "unsupported_protein_rows": self.unsupported_protein_rows,
            "unsupported_relation_rows": self.unsupported_relation_rows,
            "coverage": self.coverage,
            "metrics": dict(self.metrics),
        }
        if include_ranks:
            result["ranks"] = self.ranks.tolist()
            result["relation_ids"] = self.relation_ids.tolist()
        return result


def _ranks_from_scores(
    filtered_scores: torch.Tensor,
    targets: torch.Tensor,
    tie_policy: str,
) -> torch.Tensor:
    """使用明确的 tie policy 计算 1-based ranks。"""

    target_scores = filtered_scores.gather(1, targets[:, None])
    # MPS 不支持 float64；rank 在设备侧用 float32 计算，迁移 CPU 后再统一转
    # float64 做最终指标归约。K 通常很小，float32 可精确表示这些半整数 rank。
    greater = (filtered_scores > target_scores).sum(dim=1).to(torch.float32)
    # 当前 target 自身总会等于 target score，因此从 tie 数中减去 1。
    equal_others = (
        (filtered_scores == target_scores).sum(dim=1).to(torch.float32) - 1.0
    )
    if tie_policy == "optimistic":
        return 1.0 + greater
    if tie_policy == "pessimistic":
        return 1.0 + greater + equal_others
    if tie_policy == "average":
        return 1.0 + greater + 0.5 * equal_others
    raise ValueError(f"未知 tie_policy: {tie_policy}")


@torch.no_grad()
def evaluate_filtered_type_ranking(
    model: SITNEWalkModel,
    triples: IndexedTriples,
    known_positive_index: PairRelationIndex,
    device: torch.device | str,
    batch_size: int = 1024,
    hits_at: Sequence[int] = (1, 3, 10),
    tie_policy: str = "average",
    require_full_coverage: bool = True,
) -> RankingResult:
    """评估查询 ``(protein_i, protein_j, ?)`` 的 filtered relation rank。

    只有 ``known_positive_index`` 用于屏蔽同一 pair 的其他真类型；它不会进入
    embedding、walk、训练 loss 或梯度。若 primary evaluation 要求完整覆盖，
    transductive 模型遇到 unseen protein/relation 时会 fail-fast。
    """

    if batch_size <= 0:
        raise ValueError("batch_size 必须为正")
    if tie_policy not in {"optimistic", "pessimistic", "average"}:
        raise ValueError(f"未知 tie_policy: {tie_policy}")
    if not hits_at or any(
        not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in hits_at
    ):
        raise ValueError("hits_at 必须包含正整数")
    if triples.input_rows == 0:
        raise ValueError("评测输入为空")
    if require_full_coverage and triples.supported_rows != triples.input_rows:
        raise ValueError(
            "transductive evaluator 无法覆盖全部查询: "
            f"supported={triples.supported_rows}/{triples.input_rows}, "
            f"unsupported_protein_rows={triples.unsupported_protein_rows}, "
            f"unsupported_relation_rows={triples.unsupported_relation_rows}"
        )
    if triples.supported_rows == 0:
        return RankingResult(
            metrics={},
            ranks=torch.empty(0, dtype=torch.float64),
            relation_ids=torch.empty(0, dtype=torch.int64),
            raw_input_rows=triples.raw_input_rows,
            input_rows=triples.input_rows,
            supported_rows=0,
            unsupported_protein_rows=triples.unsupported_protein_rows,
            unsupported_relation_rows=triples.unsupported_relation_rows,
            status="unsupported",
        )

    resolved_device = torch.device(device)
    filter_index = known_positive_index.to(resolved_device)
    was_training = model.training
    model.eval()
    rank_parts: list[torch.Tensor] = []
    relation_parts: list[torch.Tensor] = []

    try:
        for batch_indices in iter_batches(triples.num_triples, batch_size):
            heads = triples.heads[batch_indices].to(resolved_device)
            relations = triples.relations[batch_indices].to(resolved_device)
            tails = triples.tails[batch_indices].to(resolved_device)
            scores = model.score_all_relations(heads, tails).float()
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError(
                    "模型 relation scores 含 NaN/Inf，拒绝生成可能越界的 rank"
                )
            positive_mask, matched = filter_index.lookup(heads, tails)
            if not bool(matched.all()):
                raise ValueError("known-positive filter 缺少评测 pair")
            target_is_known = positive_mask.gather(1, relations[:, None]).squeeze(1)
            if not bool(target_is_known.all()):
                raise ValueError("known-positive filter 未包含当前 target relation")

            # 屏蔽其他已知真类型，但保留当前 target 本身。
            filter_other = positive_mask.clone()
            filter_other.scatter_(1, relations[:, None], False)
            filtered_scores = scores.masked_fill(filter_other, -torch.inf)
            ranks = _ranks_from_scores(filtered_scores, relations, tie_policy)
            rank_parts.append(ranks.cpu().to(torch.float64))
            relation_parts.append(relations.cpu())
    finally:
        model.train(was_training)
    ranks = torch.cat(rank_parts)
    relation_ids = torch.cat(relation_parts)
    reciprocal = 1.0 / ranks
    metrics: dict[str, float] = {"mrr": float(reciprocal.mean().item())}
    for k in sorted(set(hits_at)):
        metrics[f"hits@{k}"] = float((ranks <= k).to(torch.float64).mean().item())

    # 每种关系的 MRR 有助于识别高频关系掩盖稀有关系的情况。
    for relation_id in relation_ids.unique(sorted=True).tolist():
        relation_mask = relation_ids == relation_id
        metrics[f"relation/{relation_id}/mrr"] = float(
            (1.0 / ranks[relation_mask]).mean().item()
        )
        metrics[f"relation/{relation_id}/count"] = float(relation_mask.sum().item())

    status = "ok" if triples.supported_rows == triples.input_rows else "partial_coverage"
    return RankingResult(
        metrics=metrics,
        ranks=ranks,
        relation_ids=relation_ids,
        raw_input_rows=triples.raw_input_rows,
        input_rows=triples.input_rows,
        supported_rows=triples.supported_rows,
        unsupported_protein_rows=triples.unsupported_protein_rows,
        unsupported_relation_rows=triples.unsupported_relation_rows,
        status=status,
    )
