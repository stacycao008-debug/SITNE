"""CUDA 友好的批量 typed random walk 生成器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .graph import PackedCSRGraph


def make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    """创建与设备匹配的随机数生成器。

    CUDA 使用 device-local generator，避免每步把 CPU 随机数复制到 GPU。部分
    PyTorch/MPS 组合不支持 MPS generator，因此保留可检测的 CPU fallback。
    """

    resolved = torch.device(device)
    try:
        generator = torch.Generator(device=resolved)
    except RuntimeError:
        generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _rand(
    shape: Sequence[int],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """在 generator 支持的设备生成随机数，必要时显式迁移。"""

    generator_device = torch.device(getattr(generator, "device", "cpu"))
    if generator_device.type == device.type:
        return torch.rand(tuple(shape), device=device, generator=generator)
    return torch.rand(tuple(shape), device=generator_device, generator=generator).to(
        device
    )


@dataclass(frozen=True)
class TypedWalkBatch:
    """交替 typed walk 的紧凑表示。

    ``proteins`` 形状为 ``[batch, walk_length + 1]``，``relations`` 和
    ``valid_steps`` 为 ``[batch, walk_length]``。遇到出度为零的节点后，其余位置
    用 -1 填充且 ``valid_steps=False``。
    """

    proteins: torch.Tensor
    relations: torch.Tensor
    valid_steps: torch.Tensor

    def validate(self) -> None:
        if self.proteins.ndim != 2 or self.relations.ndim != 2:
            raise ValueError("walk 张量必须是二维")
        if self.valid_steps.shape != self.relations.shape:
            raise ValueError("valid_steps 与 relations 形状不一致")
        if self.proteins.shape[0] != self.relations.shape[0]:
            raise ValueError("protein/relation batch size 不一致")
        if self.proteins.shape[1] != self.relations.shape[1] + 1:
            raise ValueError("proteins 的序列长度必须比 relations 多 1")
        if self.proteins.device != self.relations.device:
            raise ValueError("walk 张量必须位于同一设备")


class AliasTypedWalker:
    """使用 packed segmented Alias Table 的批量随机游走器。"""

    def __init__(self, graph: PackedCSRGraph):
        graph.validate()
        self.graph = graph

    @torch.no_grad()
    def sample_next_from_uniforms(
        self,
        current_nodes: torch.Tensor,
        column_uniform: torch.Tensor,
        accept_uniform: torch.Tensor,
        *,
        validate_uniforms: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把给定 uniform tensors 转为下一跳。

        拆分 RNG 与 alias 变换后，可以把完全相同的 uniform 输入注入 CPU/CUDA，
        对 chosen edge 做逐元素 parity 测试。
        """

        if current_nodes.ndim != 1:
            raise ValueError("current_nodes 必须是一维")
        if current_nodes.device != self.graph.device:
            raise ValueError("current_nodes 与 graph 必须位于同一设备")
        if current_nodes.dtype != torch.int64:
            current_nodes = current_nodes.to(torch.int64)
        if column_uniform.shape != current_nodes.shape or accept_uniform.shape != current_nodes.shape:
            raise ValueError("uniform tensors 必须与 current_nodes 同形")
        if column_uniform.device != self.graph.device or accept_uniform.device != self.graph.device:
            raise ValueError("uniform tensors 与 graph 必须位于同一设备")
        if validate_uniforms:
            if not bool(
                ((column_uniform >= 0) & (column_uniform < 1)).all()
            ) or not bool(((accept_uniform >= 0) & (accept_uniform < 1)).all()):
                raise ValueError("uniform tensors 必须位于 [0, 1)")

        in_range = (current_nodes >= 0) & (current_nodes < self.graph.num_proteins)
        safe_nodes = current_nodes.clamp(min=0, max=self.graph.num_proteins - 1)
        starts = self.graph.indptr[safe_nodes]
        ends = self.graph.indptr[safe_nodes + 1]
        segment_sizes = ends - starts
        valid = in_range & (segment_sizes > 0)

        # inactive walker 使用 edge 0 作为安全占位，最终会被 valid mask 清除。
        safe_sizes = segment_sizes.clamp_min(1)
        local_column = torch.floor(column_uniform * safe_sizes).to(torch.int64)
        local_column = torch.minimum(local_column, safe_sizes - 1)
        candidate_edge = starts + local_column
        candidate_edge = torch.where(valid, candidate_edge, torch.zeros_like(candidate_edge))

        use_primary = accept_uniform < self.graph.alias_probability[candidate_edge]
        selected_local = torch.where(
            use_primary,
            local_column,
            self.graph.alias_local[candidate_edge].to(torch.int64),
        )
        selected_edge = starts + selected_local
        selected_edge = torch.where(valid, selected_edge, torch.zeros_like(selected_edge))
        next_nodes = self.graph.destinations[selected_edge]
        relations = self.graph.relations[selected_edge]
        invalid_value = torch.full_like(next_nodes, -1)
        return (
            torch.where(valid, next_nodes, invalid_value),
            torch.where(valid, relations, invalid_value),
            valid,
        )

    @torch.no_grad()
    def sample_next(
        self,
        current_nodes: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为一批当前节点采样 ``(next_protein, relation, valid)``。"""

        column_uniform = _rand(current_nodes.shape, self.graph.device, generator)
        accept_uniform = _rand(current_nodes.shape, self.graph.device, generator)
        return self.sample_next_from_uniforms(
            current_nodes,
            column_uniform,
            accept_uniform,
            validate_uniforms=False,
        )

    @torch.no_grad()
    def generate(
        self,
        start_nodes: torch.Tensor,
        walk_length: int,
        generator: torch.Generator,
    ) -> TypedWalkBatch:
        """从给定节点并行生成定长、可 padding 的 typed walks。"""

        if walk_length <= 0:
            raise ValueError("walk_length 必须为正")
        if start_nodes.ndim != 1 or start_nodes.numel() == 0:
            raise ValueError("start_nodes 必须是非空一维张量")
        start_nodes = start_nodes.to(device=self.graph.device, dtype=torch.int64)
        if bool((start_nodes < 0).any()) or bool(
            (start_nodes >= self.graph.num_proteins).any()
        ):
            raise ValueError("start_nodes 含越界 ID")

        batch_size = int(start_nodes.numel())
        proteins = torch.full(
            (batch_size, walk_length + 1),
            -1,
            dtype=torch.int64,
            device=self.graph.device,
        )
        relations = torch.full(
            (batch_size, walk_length),
            -1,
            dtype=torch.int64,
            device=self.graph.device,
        )
        valid_steps = torch.zeros(
            (batch_size, walk_length),
            dtype=torch.bool,
            device=self.graph.device,
        )
        proteins[:, 0] = start_nodes

        current = start_nodes
        active = torch.ones(batch_size, dtype=torch.bool, device=self.graph.device)
        for step in range(walk_length):
            # 已终止 walker 用节点 0 安全占位；active mask 保证其结果不会写入。
            safe_current = torch.where(active, current, torch.zeros_like(current))
            next_nodes, relation_ids, has_edge = self.sample_next(
                safe_current,
                generator,
            )
            step_valid = active & has_edge
            valid_steps[:, step] = step_valid
            proteins[:, step + 1] = torch.where(
                step_valid,
                next_nodes,
                torch.full_like(next_nodes, -1),
            )
            relations[:, step] = torch.where(
                step_valid,
                relation_ids,
                torch.full_like(relation_ids, -1),
            )
            current = torch.where(step_valid, next_nodes, torch.zeros_like(next_nodes))
            active = step_valid
            # 不根据 active.any() 提前 break，避免 CUDA 每一步发生 host sync；
            # 已终止 walker 后续仍固定消耗 RNG，复现行为也更稳定。

        batch = TypedWalkBatch(
            proteins=proteins,
            relations=relations,
            valid_steps=valid_steps,
        )
        batch.validate()
        return batch
