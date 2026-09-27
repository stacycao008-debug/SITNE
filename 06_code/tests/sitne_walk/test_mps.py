from __future__ import annotations

import pytest
import torch

from sitne_walk.trainer import ProteinNoiseSampler
from sitne_walk.walks import make_generator


pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="需要 Apple MPS 环境",
)


def test_unigram_noise_sampler_uses_mps_compatible_dtype() -> None:
    device = torch.device("mps")
    sampler = ProteinNoiseSampler(
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        device=device,
        distribution="unigram",
        unigram_power=0.75,
    )
    positives = torch.tensor([0, 1, 2, 3], device=device)
    samples = sampler.sample(
        positives,
        num_negatives=3,
        generator=make_generator(device, 123),
    )
    assert samples.device.type == "mps"
    assert samples.dtype == torch.int64
    assert samples.shape == (4, 3)
    assert not bool((samples == positives[:, None]).any())
