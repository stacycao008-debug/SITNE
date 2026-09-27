"""SITNE-Walk 双通道 embedding、关系 decoder 与 nuisance adversary。"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn

from .gradient_reversal import gradient_reverse


class NuisanceHead(nn.Module):
    """两层 nuisance 分类器；GRL 在进入本 head 前应用。"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        if min(input_dim, hidden_dim, num_classes) <= 0:
            raise ValueError("nuisance head 的维度必须为正")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SITNEWalkModel(nn.Module):
    """SITNE-Walk v1。

    v1 与当前可审计数据保持一致，使用 unordered pair 和对称双通道 DistMult
    decoder。若未来获得逐事实方向标签，应建立独立的 directed 版本，而不是在
    本类中根据反向边是否存在来猜测方向。
    """

    def __init__(
        self,
        num_proteins: int,
        num_relations: int,
        embedding_dim: int,
        nuisance_cardinalities: Mapping[str, int] | None = None,
        nuisance_hidden_dim: int = 128,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()
        if min(num_proteins, num_relations, embedding_dim) <= 0:
            raise ValueError("protein/relation 数和 embedding_dim 必须为正")
        self.num_proteins = int(num_proteins)
        self.num_relations = int(num_relations)
        self.embedding_dim = int(embedding_dim)
        self.scale = math.sqrt(float(embedding_dim))

        # input/context 两套 topology embedding 对应标准 SGNS 参数化。
        self.topology_embedding = nn.Embedding(num_proteins, embedding_dim)
        self.topology_context_embedding = nn.Embedding(num_proteins, embedding_dim)
        self.semantic_embedding = nn.Embedding(num_proteins, embedding_dim)

        # semantic_relation 同时服务 L_typed 与最终 decoder，避免额外的、无法解释
        # 的 relation 表。topology relation 仅服务 topology decoder 分量。
        self.semantic_relation = nn.Embedding(num_relations, embedding_dim)
        self.topology_relation = nn.Embedding(num_relations, embedding_dim)
        self.decoder_dropout = nn.Dropout(decoder_dropout)

        cardinalities = dict(nuisance_cardinalities or {})
        self.nuisance_heads = nn.ModuleDict(
            {
                name: NuisanceHead(
                    input_dim=embedding_dim,
                    hidden_dim=nuisance_hidden_dim,
                    num_classes=num_classes,
                )
                for name, num_classes in sorted(cardinalities.items())
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用与 embedding 维度匹配的稳定初始化。"""

        bound = 1.0 / math.sqrt(self.embedding_dim)
        for embedding in (
            self.topology_embedding,
            self.topology_context_embedding,
            self.semantic_embedding,
            self.semantic_relation,
            self.topology_relation,
        ):
            nn.init.uniform_(embedding.weight, -bound, bound)
        for module in self.nuisance_heads.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def topology_sgns_logits(
        self,
        centers: torch.Tensor,
        positive_contexts: torch.Tensor,
        negative_contexts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 topology SGNS 的正样本与 noise-token logits。"""

        if centers.shape != positive_contexts.shape:
            raise ValueError("centers 与 positive_contexts 形状必须一致")
        if negative_contexts.ndim != 2 or negative_contexts.shape[0] != centers.numel():
            raise ValueError("negative_contexts 必须是 [num_pairs, num_negatives]")
        center_vectors = self.topology_embedding(centers)
        positive_vectors = self.topology_context_embedding(positive_contexts)
        negative_vectors = self.topology_context_embedding(negative_contexts)
        positive_logits = (center_vectors * positive_vectors).sum(dim=-1) / self.scale
        negative_logits = torch.einsum(
            "bd,bnd->bn",
            center_vectors,
            negative_vectors,
        ) / self.scale
        return positive_logits, negative_logits

    def typed_relation_logits(self, protein_ids: torch.Tensor) -> torch.Tensor:
        """预测 train-relation token 的全 softmax logits。"""

        semantic = self.semantic_embedding(protein_ids)
        return semantic @ self.semantic_relation.weight.transpose(0, 1) / self.scale

    def score_all_relations(
        self,
        head_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """为每个 unordered protein pair 计算所有关系类型得分。"""

        if head_ids.shape != tail_ids.shape:
            raise ValueError("head_ids 与 tail_ids 形状必须一致")
        topology_head = self.decoder_dropout(self.topology_embedding(head_ids))
        topology_tail = self.decoder_dropout(self.topology_embedding(tail_ids))
        semantic_head = self.decoder_dropout(self.semantic_embedding(head_ids))
        semantic_tail = self.decoder_dropout(self.semantic_embedding(tail_ids))
        topology_pair = topology_head * topology_tail
        semantic_pair = semantic_head * semantic_tail
        topology_scores = (
            topology_pair @ self.topology_relation.weight.transpose(0, 1)
        ) / self.scale
        semantic_scores = (
            semantic_pair @ self.semantic_relation.weight.transpose(0, 1)
        ) / self.scale
        return topology_scores + semantic_scores

    def score_triples(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """只返回给定关系的 typed decoder 得分。"""

        all_scores = self.score_all_relations(head_ids, tail_ids)
        return all_scores.gather(1, relation_ids.reshape(-1, 1)).squeeze(1)

    def nuisance_logits(
        self,
        protein_ids: torch.Tensor,
        head_name: str,
        gradient_reversal_coefficient: float,
    ) -> torch.Tensor:
        """从 semantic channel 预测 nuisance，encoder 梯度经 GRL 反转。"""

        if head_name not in self.nuisance_heads:
            raise KeyError(f"模型没有 nuisance head: {head_name}")
        semantic = self.semantic_embedding(protein_ids)
        reversed_semantic = gradient_reverse(
            semantic,
            coefficient=gradient_reversal_coefficient,
        )
        return self.nuisance_heads[head_name](reversed_semantic)

    def exported_embeddings(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 topology、semantic 和拼接表示的只读视图。"""

        topology = self.topology_embedding.weight
        semantic = self.semantic_embedding.weight
        return topology, semantic, torch.cat([topology, semantic], dim=-1)
