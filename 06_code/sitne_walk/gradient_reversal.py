"""Gradient Reversal Layer (GRL)。"""

from __future__ import annotations

import torch
from torch import nn


class _GradientReversalFunction(torch.autograd.Function):
    """前向恒等、反向乘以负系数的 autograd 原语。"""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        coefficient: float,
    ) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * gradient, None


def gradient_reverse(inputs: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    """函数式 GRL 接口。"""

    if coefficient < 0:
        raise ValueError("GRL coefficient 必须非负")
    return _GradientReversalFunction.apply(inputs, coefficient)


class GradientReversal(nn.Module):
    """可嵌入 ``nn.Sequential`` 风格代码的 GRL 模块。"""

    def __init__(self, coefficient: float = 1.0):
        super().__init__()
        if coefficient < 0:
            raise ValueError("GRL coefficient 必须非负")
        self.coefficient = float(coefficient)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return gradient_reverse(inputs, self.coefficient)

