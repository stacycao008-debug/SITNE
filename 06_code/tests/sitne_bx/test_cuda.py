from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from sitne_bx.config import BXConfig
from sitne_bx.losses import composite_loss
from sitne_bx.model import SITNEBXModel
from sitne_bx.runner import resolve_device


def _cuda_config(use_amp: bool) -> BXConfig:
    return BXConfig.from_mapping(
        {
            "data": {"dataset_bstar_root": "unused-bstar", "typed_level": "none"},
            "model": {
                "embedding_dim": 16,
                "token_embedding_dim": 8,
                "pair_hidden_dim": 12,
                "chunk_size": 16,
                "chunk_stride": 12,
                "chunk_batch_size": 4,
                "convolution_kernel_size": 3,
                "dropout": 0.0,
            },
            "loss": {
                "binary_weight": 1.0,
                "topology_weight": 0.1,
                "typed_weight": 0.0,
            },
            "training": {
                "device": "cuda",
                "allow_cpu_for_testing": False,
                "epochs": 1,
                "batch_size": 2,
                "use_amp": use_amp,
                "amp_dtype": "float16",
            },
            "evaluation": {"expected_test_rows": 2},
        }
    )


def test_explicit_cuda_request_never_falls_back(monkeypatch):
    config = _cuda_config(use_amp=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="禁止静默回退"):
        resolve_device(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real NVIDIA CUDA")
@pytest.mark.parametrize("use_amp", [False, True], ids=["fp32", "fp16_amp"])
def test_real_cuda_finite_gradient_memory_and_checkpoint_roundtrip(
    tmp_path, use_amp: bool
):
    config = _cuda_config(use_amp=use_amp)
    device = resolve_device(config)
    assert device.type == "cuda"
    torch.cuda.reset_peak_memory_stats(device)

    model = SITNEBXModel(config.model, num_types=0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    payload = ["ACDEFGHIKLMNPQRSTVWY" * 2, "MNPQRSTVWYACDEFGHIKL", "GGGGACDU"]
    left = torch.tensor([0, 1], dtype=torch.long, device=device)
    right = torch.tensor([1, 2], dtype=torch.long, device=device)
    labels = torch.tensor([1.0, 0.0], device=device)
    typed_targets = torch.zeros((2, 0), device=device)
    typed_mask = torch.zeros(2, dtype=torch.bool, device=device)
    context = (
        torch.autocast("cuda", dtype=torch.float16)
        if use_amp
        else nullcontext()
    )
    optimizer.zero_grad(set_to_none=True)
    with context:
        output = model(payload, left, right)
        topology_output = model(payload, right, left)
        losses = composite_loss(
            output,
            labels,
            typed_targets,
            typed_mask,
            config.loss,
            topology_output=topology_output,
        )
    assert losses.total.device.type == "cuda"
    assert torch.isfinite(losses.total)
    scaler.scale(losses.total).backward()
    scaler.unscale_(optimizer)
    finite_gradients = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert finite_gradients and all(bool(value) for value in finite_gradients)
    optimizer.step() if not use_amp else scaler.step(optimizer)
    if use_amp:
        scaler.update()
    torch.cuda.synchronize(device)
    assert torch.cuda.max_memory_allocated(device) > 0

    checkpoint_path = tmp_path / f"cuda_{'amp' if use_amp else 'fp32'}.pt"
    torch.save({"model_state_dict": model.state_dict(), "use_amp": use_amp}, checkpoint_path)
    restored_payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored = SITNEBXModel(config.model, num_types=0).to(device)
    restored.load_state_dict(restored_payload["model_state_dict"], strict=True)
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)
