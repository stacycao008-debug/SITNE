from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sitne_walk.config import SITNEConfig
from sitne_walk.data import load_training_triples
from sitne_walk.graph import build_packed_csr_graph
from sitne_walk.losses import decorrelation_loss
from sitne_walk.model import SITNEWalkModel
from sitne_walk.trainer import SITNETrainer, build_model
from sitne_walk.walks import AliasTypedWalker


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 NVIDIA CUDA 主机",
)


def _graph(toy_train_tsv: Path):
    triples = load_training_triples(toy_train_tsv)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",) * triples.num_relations,
        alpha=0.8,
        beta=0.6,
        epsilon=1e-12,
    )
    return triples, graph


def test_alias_transform_cpu_cuda_exact_parity(toy_train_tsv: Path) -> None:
    triples, graph_cpu = _graph(toy_train_tsv)
    graph_cuda = graph_cpu.to("cuda")
    current_cpu = torch.arange(triples.num_proteins, dtype=torch.int64)
    slot_cpu = torch.tensor([0.01, 0.25, 0.51, 0.99], dtype=torch.float32)
    coin_cpu = torch.tensor([0.13, 0.37, 0.73, 0.91], dtype=torch.float32)
    cpu_output = AliasTypedWalker(graph_cpu).sample_next_from_uniforms(
        current_cpu, slot_cpu, coin_cpu
    )
    cuda_output = AliasTypedWalker(graph_cuda).sample_next_from_uniforms(
        current_cpu.cuda(), slot_cpu.cuda(), coin_cpu.cuda()
    )
    for cpu_tensor, cuda_tensor in zip(cpu_output, cuda_output):
        assert torch.equal(cpu_tensor, cuda_tensor.cpu())


def test_model_cpu_cuda_forward_backward_parity() -> None:
    torch.manual_seed(17)
    cpu_model = SITNEWalkModel(8, 3, 6, decoder_dropout=0.0)
    cuda_model = SITNEWalkModel(8, 3, 6, decoder_dropout=0.0).cuda()
    cuda_model.load_state_dict(cpu_model.state_dict())
    heads = torch.tensor([0, 1, 2, 3])
    tails = torch.tensor([4, 5, 6, 7])
    cpu_scores = cpu_model.score_all_relations(heads, tails)
    cuda_scores = cuda_model.score_all_relations(heads.cuda(), tails.cuda())
    assert torch.allclose(cpu_scores, cuda_scores.cpu(), atol=1e-5, rtol=1e-4)
    cpu_scores.sum().backward()
    cuda_scores.sum().backward()
    assert torch.allclose(
        cpu_model.semantic_embedding.weight.grad,
        cuda_model.semantic_embedding.weight.grad.cpu(),
        atol=1e-5,
        rtol=1e-4,
    )


def test_decorrelation_stays_float32_inside_cuda_autocast() -> None:
    torch.manual_seed(19)
    topology = torch.randn(32, 16, device="cuda", dtype=torch.float32) * 1e-3
    semantic = torch.randn(32, 16, device="cuda", dtype=torch.float32) * 1e-3
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        actual = decorrelation_loss(topology, semantic)
    centered_topology = topology - topology.mean(dim=0, keepdim=True)
    centered_semantic = semantic - semantic.mean(dim=0, keepdim=True)
    expected = (
        centered_topology.transpose(0, 1) @ centered_semantic / topology.shape[0]
    ).square().sum()
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-6)


def test_cuda_amp_training_and_checkpoint_round_trip(
    toy_train_tsv: Path,
    tmp_path: Path,
) -> None:
    config = SITNEConfig.from_mapping(
        {
            "data": {
                "train_path": str(toy_train_tsv),
                "pair_semantics": "unordered",
                "default_relation_mode": "symmetric",
            },
            "walk": {
                "walk_length": 3,
                "walks_per_protein": 1,
                "walk_batch_size": 4,
                "context_window": 1,
                "negative_samples": 2,
                "max_topology_pairs_per_batch": 64,
            },
            "model": {
                "embedding_dim": 8,
                "nuisance_hidden_dim": 6,
                "nuisance_heads": ["degree"],
            },
            "training": {
                "device": "cuda",
                "epochs": 1,
                "rank_batch_size": 3,
                "early_stopping_patience": 1,
                "evaluation_batch_size": 4,
                "use_amp": True,
                "amp_dtype": "float16",
                "deterministic": False,
                "max_steps_per_epoch": 1,
                "log_interval": 1,
                "output_dir": str(tmp_path / "results"),
            },
        }
    )
    triples, graph_cpu = _graph(toy_train_tsv)
    model, labels, weights = build_model(config, triples, graph_cpu)
    trainer = SITNETrainer(
        config,
        triples,
        graph_cpu,
        model,
        labels,
        weights,
    )
    assert trainer.graph.device.type == "cuda"
    assert all(parameter.device.type == "cuda" for parameter in trainer.model.parameters())
    result = trainer.fit(checkpoint_dir=tmp_path / "checkpoints")
    assert result.best_checkpoint is not None
    assert torch.isfinite(torch.tensor(result.history[0]["loss"]))
    assert result.history[0]["cuda_peak_memory_bytes"] > 0
    assert trainer.load_checkpoint(result.best_checkpoint) == 1
