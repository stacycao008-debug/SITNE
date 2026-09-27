"""归纳式共享序列编码器和无向对称二分类/typed heads。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig
from .data import normalize_sequence


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_ID = {amino_acid: index + 1 for index, amino_acid in enumerate(AA_ALPHABET)}
PAD_ID = 0


@dataclass(frozen=True)
class ChunkPlan:
    sequence_index: int
    start: int
    end: int
    inverse_coverage: tuple[float, ...]


def plan_sequence_chunks(
    sequence_lengths: Sequence[int], chunk_size: int, stride: int
) -> list[ChunkPlan]:
    """生成无截断、无空洞的 chunk 计划；overlap 位点按覆盖数反向加权。"""

    if chunk_size <= 0 or stride <= 0 or stride > chunk_size:
        raise ValueError("chunk_size/stride 必须为正且 stride <= chunk_size")
    plans: list[ChunkPlan] = []
    for sequence_index, length in enumerate(sequence_lengths):
        if length <= 0:
            raise ValueError("序列长度必须大于 0")
        if length <= chunk_size:
            starts = [0]
        else:
            starts = list(range(0, length - chunk_size + 1, stride))
            final_start = length - chunk_size
            if starts[-1] != final_start:
                starts.append(final_start)
        coverage = [0] * length
        for start in starts:
            end = min(start + chunk_size, length)
            for position in range(start, end):
                coverage[position] += 1
        if any(value == 0 for value in coverage):
            raise AssertionError("chunk 计划出现未覆盖残基")
        for start in starts:
            end = min(start + chunk_size, length)
            plans.append(
                ChunkPlan(
                    sequence_index=sequence_index,
                    start=start,
                    end=end,
                    inverse_coverage=tuple(1.0 / coverage[pos] for pos in range(start, end)),
                )
            )
    return plans


class ChunkedSequenceEncoder(nn.Module):
    """轻量纯 PyTorch 编码器。

    所有残基均被处理。overlap 位点使用 ``1 / coverage`` 权重，随后以真实
    序列长度归一化，避免长序列或重叠区在 masked pooling 中获得额外权重。
    ``U`` 在 tokenization 前固定映射为 ``X``。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.chunk_size = config.chunk_size
        self.chunk_stride = config.chunk_stride
        self.chunk_batch_size = config.chunk_batch_size
        token_dim = config.token_embedding_dim
        self.embedding = nn.Embedding(len(AA_TO_ID) + 1, token_dim, padding_idx=PAD_ID)
        self.convolution = nn.Conv1d(
            token_dim,
            token_dim,
            kernel_size=config.convolution_kernel_size,
            padding=config.convolution_kernel_size // 2,
        )
        self.token_norm = nn.LayerNorm(token_dim)
        self.projection = nn.Sequential(
            nn.Linear(token_dim, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.embedding_dim),
        )

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def _token_ids(self, sequence: str) -> list[int]:
        return [AA_TO_ID[amino_acid] for amino_acid in normalize_sequence(sequence)]

    def forward(self, sequences: Sequence[str]) -> torch.Tensor:
        if not sequences:
            raise ValueError("sequence batch 不能为空")
        normalized = [normalize_sequence(sequence) for sequence in sequences]
        plans = plan_sequence_chunks(
            [len(sequence) for sequence in normalized], self.chunk_size, self.chunk_stride
        )
        pooled = torch.zeros(
            (len(normalized), self.embedding.embedding_dim),
            dtype=self.embedding.weight.dtype,
            device=self.device,
        )
        for start in range(0, len(plans), self.chunk_batch_size):
            group = plans[start : start + self.chunk_batch_size]
            max_length = max(plan.end - plan.start for plan in group)
            tokens = torch.full(
                (len(group), max_length),
                PAD_ID,
                dtype=torch.long,
                device=self.device,
            )
            weights = torch.zeros(
                (len(group), max_length),
                dtype=self.embedding.weight.dtype,
                device=self.device,
            )
            owners = torch.empty(len(group), dtype=torch.long, device=self.device)
            for row, plan in enumerate(group):
                chunk = normalized[plan.sequence_index][plan.start : plan.end]
                ids = torch.tensor(self._token_ids(chunk), dtype=torch.long, device=self.device)
                length = ids.numel()
                tokens[row, :length] = ids
                weights[row, :length] = torch.tensor(
                    plan.inverse_coverage,
                    dtype=self.embedding.weight.dtype,
                    device=self.device,
                )
                owners[row] = plan.sequence_index
            embedded = self.embedding(tokens)
            contextual = self.convolution(embedded.transpose(1, 2)).transpose(1, 2)
            contextual = self.token_norm(F.gelu(contextual) + embedded)
            chunk_sums = (contextual * weights.unsqueeze(-1)).sum(dim=1)
            pooled = pooled.index_add(0, owners, chunk_sums)
        lengths = torch.tensor(
            [len(sequence) for sequence in normalized],
            dtype=pooled.dtype,
            device=pooled.device,
        ).unsqueeze(1)
        return self.projection(pooled / lengths)


@dataclass(frozen=True)
class PairOutput:
    binary_logits: torch.Tensor
    typed_logits: torch.Tensor | None
    embedding_a: torch.Tensor
    embedding_b: torch.Tensor


class SITNEBXModel(nn.Module):
    """共享编码器 + 交换不变 pair 特征，保证无向 pair 结构对称。"""

    def __init__(self, config: ModelConfig, num_types: int = 0):
        super().__init__()
        self.config = config
        self.backend = config.backend
        if self.backend == "chunked_sequence":
            self.encoder: nn.Module = ChunkedSequenceEncoder(config)
        else:
            assert config.precomputed_dimension is not None
            self.encoder = nn.Sequential(
                nn.Linear(config.precomputed_dimension, config.embedding_dim),
                nn.GELU(),
                nn.LayerNorm(config.embedding_dim),
            )
        pair_dim = 3 * config.embedding_dim
        self.binary_head = nn.Sequential(
            nn.Linear(pair_dim, config.pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.pair_hidden_dim, 1),
        )
        self.typed_head = (
            nn.Sequential(
                nn.Linear(pair_dim, config.pair_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.pair_hidden_dim, num_types),
            )
            if num_types > 0
            else None
        )

    def encode(self, payload: Sequence[str] | torch.Tensor) -> torch.Tensor:
        if self.backend == "chunked_sequence":
            if isinstance(payload, torch.Tensor):
                raise TypeError("chunked_sequence 后端需要字符串序列")
            return self.encoder(payload)
        if not isinstance(payload, torch.Tensor):
            raise TypeError("precomputed_embedding 后端需要二维 tensor")
        if payload.ndim != 2:
            raise ValueError("precomputed embedding 必须是二维 tensor")
        return self.encoder(payload)

    @staticmethod
    def symmetric_pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError("pair 两端 embedding shape 不一致")
        return torch.cat((a + b, torch.abs(a - b), a * b), dim=-1)

    def forward(
        self,
        unique_payload: Sequence[str] | torch.Tensor,
        left_index: torch.Tensor,
        right_index: torch.Tensor,
    ) -> PairOutput:
        unique_embeddings = self.encode(unique_payload)
        embedding_a = unique_embeddings.index_select(0, left_index)
        embedding_b = unique_embeddings.index_select(0, right_index)
        pair_features = self.symmetric_pair_features(embedding_a, embedding_b)
        binary_logits = self.binary_head(pair_features).squeeze(-1)
        typed_logits = self.typed_head(pair_features) if self.typed_head else None
        return PairOutput(binary_logits, typed_logits, embedding_a, embedding_b)
