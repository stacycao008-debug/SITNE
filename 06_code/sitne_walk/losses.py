"""SITNE-Walk v1 的各损失项与数值稳定归约。"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F


def topology_sgns_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
) -> torch.Tensor:
    """标准 SGNS loss；negative 仅表示 noise token，不表示生物学负 PPI。"""

    if positive_logits.ndim != 1:
        raise ValueError("positive_logits 必须是一维")
    if negative_logits.ndim != 2 or negative_logits.shape[0] != positive_logits.numel():
        raise ValueError("negative_logits 必须是 [num_positive, num_negative]")
    if positive_logits.numel() == 0:
        raise ValueError("SGNS batch 不能为空")
    # 即使模型前向处于 AMP，这里也转 float32 进行 log-sigmoid 与归约。
    positive = positive_logits.float()
    negative = negative_logits.float()
    per_sample = -F.logsigmoid(positive) - F.logsigmoid(-negative).sum(dim=1)
    return per_sample.mean()


def typed_relation_cross_entropy(
    logits: torch.Tensor,
    target_relations: torch.Tensor,
) -> torch.Tensor:
    """semantic relation-token 全 softmax loss。"""

    if logits.ndim != 2:
        raise ValueError("typed logits 必须是二维")
    if target_relations.ndim != 1 or target_relations.numel() != logits.shape[0]:
        raise ValueError("target_relations 与 logits batch 不一致")
    if logits.shape[0] == 0:
        raise ValueError("typed relation batch 不能为空")
    return F.cross_entropy(logits.float(), target_relations)


def decorrelation_loss(
    topology_embeddings: torch.Tensor,
    semantic_embeddings: torch.Tensor,
) -> torch.Tensor:
    """实现草案中的 ``||(Z_topo^T Z_sem)/B||_F^2``。"""

    if topology_embeddings.shape != semantic_embeddings.shape:
        raise ValueError("topology 与 semantic embedding 形状必须一致")
    if topology_embeddings.ndim != 2 or topology_embeddings.shape[0] < 2:
        raise ValueError("decorrelation batch 必须是二维且至少含两个唯一蛋白质")
    # 仅调用 ``.float()`` 不足以阻止外层 CUDA autocast 把 matmul 再降为
    # FP16/BF16。这里显式关闭 autocast，确保协方差与平方归约全程 FP32。
    device_type = topology_embeddings.device.type
    autocast_disabled = (
        torch.autocast(device_type=device_type, enabled=False)
        if device_type in {"cpu", "cuda", "mps"}
        else nullcontext()
    )
    with autocast_disabled:
        topology = topology_embeddings.float()
        semantic = semantic_embeddings.float()
        topology = topology - topology.mean(dim=0, keepdim=True)
        semantic = semantic - semantic.mean(dim=0, keepdim=True)
        batch_size = topology.shape[0]
        cross_covariance = topology.transpose(0, 1) @ semantic / float(batch_size)
        return cross_covariance.square().sum()


def filtered_pairwise_ranking_loss(
    relation_scores: torch.Tensor,
    positive_mask: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """对每个 pair 的所有正类型和所有未标注候选执行 pairwise softplus。

    ``~positive_mask`` 只是训练用的 unlabeled contrastive candidates，不能解释为
    真实生物学负类型。损失先在 pair 内平均，再在 pair 间平均，避免多标签较多的
    pair 获得更大权重。
    """

    if relation_scores.ndim != 2:
        raise ValueError("relation_scores 必须是 [num_pairs, num_relations]")
    if positive_mask.shape != relation_scores.shape or positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask 必须是与 scores 同形的 bool 张量")
    positive_count = positive_mask.sum(dim=1)
    negative_mask = ~positive_mask
    negative_count = negative_mask.sum(dim=1)
    valid_pairs = (positive_count > 0) & (negative_count > 0)
    if not bool(valid_pairs.any()):
        raise ValueError("ranking batch 没有同时包含正类型和未标注候选的 pair")

    scores = relation_scores.float()
    # [B, positive_relation, negative_relation]
    differences = scores[:, None, :] - scores[:, :, None] + float(margin)
    comparisons = positive_mask[:, :, None] & negative_mask[:, None, :]
    values = F.softplus(differences) * comparisons
    per_pair_count = comparisons.sum(dim=(1, 2)).clamp_min(1)
    per_pair_loss = values.sum(dim=(1, 2)) / per_pair_count
    return per_pair_loss[valid_pairs].mean()


def nuisance_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """带 train-only inverse-frequency 权重的 nuisance 分类损失。"""

    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.numel():
        raise ValueError("nuisance logits/targets 形状不一致")
    weights = class_weights.float() if class_weights is not None else None
    return F.cross_entropy(logits.float(), targets, weight=weights)


def hierarchy_loss_disabled(
    relation_embeddings: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """v1 的显式 hierarchy stop condition。

    当前 coarse_type 不是已审核 ontology hierarchy，因此非零系数直接报错。
    """

    if coefficient != 0:
        raise ValueError("SITNE-Walk v1 不允许启用 hierarchy loss")
    return relation_embeddings.sum() * 0.0
