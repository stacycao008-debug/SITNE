"""Dataset B 全量 binary loss 与 Train-positive 辅助损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import LossConfig
from .model import PairOutput


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    binary: torch.Tensor
    topology: torch.Tensor
    typed: torch.Tensor
    binary_rows: int
    topology_rows: int
    typed_rows: int


def binary_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("binary logits/labels 必须是相同 shape 的一维 tensor")
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def train_positive_topology_loss(
    embedding_a: torch.Tensor, embedding_b: torch.Tensor, positive_mask: torch.Tensor
) -> torch.Tensor:
    """仅在显式 Train-positive 上保持两端表示接近，不构造未知负 PPI。"""

    if positive_mask.dtype != torch.bool or positive_mask.ndim != 1:
        raise ValueError("positive_mask 必须是一维 bool tensor")
    if not bool(positive_mask.any()):
        return (embedding_a.sum() + embedding_b.sum()) * 0.0
    similarity = F.cosine_similarity(
        embedding_a[positive_mask], embedding_b[positive_mask], dim=-1
    )
    return (1.0 - similarity).mean()


def masked_typed_loss(
    typed_logits: torch.Tensor | None,
    target_distribution: torch.Tensor,
    typed_mask: torch.Tensor,
) -> torch.Tensor:
    """对有安全 typed facts 的正例计算 multi-target soft CE。

    多标签 pair 的已观察类型被赋予均匀质量；mask 外行不贡献梯度。这里的
    softmax 是辅助关系分类似然，不把未知 pair 或 annotation availability 当作
    生物学负例。
    """

    if typed_logits is None:
        if bool(typed_mask.any()):
            raise ValueError("存在 typed target，但模型未构建 typed head")
        return target_distribution.sum() * 0.0
    if typed_logits.shape != target_distribution.shape:
        raise ValueError("typed logits 与 target distribution shape 不一致")
    if typed_mask.dtype != torch.bool or typed_mask.shape != typed_logits.shape[:1]:
        raise ValueError("typed_mask shape/dtype 错误")
    if not bool(typed_mask.any()):
        return typed_logits.sum() * 0.0
    selected_targets = target_distribution[typed_mask]
    row_sums = selected_targets.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums)):
        raise ValueError("masked typed target 每行必须归一化为 1")
    return -(
        selected_targets * F.log_softmax(typed_logits[typed_mask], dim=-1)
    ).sum(dim=-1).mean()


def composite_loss(
    output: PairOutput,
    labels: torch.Tensor,
    typed_targets: torch.Tensor,
    typed_mask: torch.Tensor,
    config: LossConfig,
    topology_output: PairOutput | None = None,
) -> LossBreakdown:
    positive_mask = labels == 1
    if bool((typed_mask & ~positive_mask).any()):
        raise ValueError("typed auxiliary 只能作用于显式 Train-positive")
    binary = binary_bce_loss(output.binary_logits, labels)
    if topology_output is None:
        topology = (output.embedding_a.sum() + output.embedding_b.sum()) * 0.0
        topology_rows = 0
    else:
        topology_rows = int(topology_output.embedding_a.shape[0])
        topology = train_positive_topology_loss(
            topology_output.embedding_a,
            topology_output.embedding_b,
            torch.ones(
                topology_rows,
                dtype=torch.bool,
                device=topology_output.embedding_a.device,
            ),
        )
    typed = masked_typed_loss(output.typed_logits, typed_targets, typed_mask)
    total = (
        config.binary_weight * binary
        + config.topology_weight * topology
        + config.typed_weight * typed
    )
    return LossBreakdown(
        total=total,
        binary=binary,
        topology=topology,
        typed=typed,
        binary_rows=int(labels.numel()),
        topology_rows=topology_rows,
        typed_rows=int(typed_mask.sum().item()),
    )
