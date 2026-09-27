from __future__ import annotations

import torch

from sitne_bx.config import ModelConfig
from sitne_bx.data import PairRecord, PairTable, TypedAnnotations, typed_targets_for_batch
from sitne_bx.losses import composite_loss
from sitne_bx.model import SITNEBXModel, plan_sequence_chunks
from sitne_bx.topology import build_train_positive_graph, sample_walk_contexts
from sitne_bx.config import LossConfig


def test_chunk_plan_has_no_truncation_and_length_weighting():
    lengths = [3, 5, 11, 27]
    plans = plan_sequence_chunks(lengths, chunk_size=8, stride=5)
    for sequence_index, length in enumerate(lengths):
        coverage_weight = [0.0] * length
        for plan in plans:
            if plan.sequence_index != sequence_index:
                continue
            for offset, weight in enumerate(plan.inverse_coverage):
                coverage_weight[plan.start + offset] += weight
        assert all(abs(value - 1.0) < 1e-12 for value in coverage_weight)


def test_symmetric_model_and_masked_typed_backward():
    config = ModelConfig(
        embedding_dim=8,
        token_embedding_dim=6,
        pair_hidden_dim=10,
        chunk_size=5,
        chunk_stride=3,
        chunk_batch_size=4,
        convolution_kernel_size=3,
        dropout=0.0,
    )
    model = SITNEBXModel(config, num_types=2)
    model.eval()
    payload = ["ACDUFG", "MKL", "QRST"]
    left = torch.tensor([0, 1])
    right = torch.tensor([1, 2])
    forward = model(payload, left, right)
    reverse = model(payload, right, left)
    assert torch.allclose(forward.binary_logits, reverse.binary_logits, atol=1e-7)
    assert torch.allclose(forward.typed_logits, reverse.typed_logits, atol=1e-7)

    records = (
        PairRecord("p1", "A", "B", 1, "train", "A|B"),
        PairRecord("p2", "B", "C", 1, "train", "B|C"),
    )
    annotations = TypedAnnotations("fine", ("t1", "t2"), {"p1": (0, 1)}, 2)
    targets, mask = typed_targets_for_batch(records, annotations, torch.device("cpu"))
    topology = model(payload, left, right)
    losses = composite_loss(
        forward,
        torch.tensor([1.0, 0.0]),
        targets,
        mask,
        LossConfig(),
        topology_output=topology,
    )
    assert losses.binary_rows == 2
    assert losses.topology_rows == 2
    assert losses.typed_rows == 1
    losses.total.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_train_positive_graph_and_walk_are_deterministic():
    records = (
        PairRecord("p1", "A", "B", 1, "train", "A|B"),
        PairRecord("p2", "B", "C", 1, "train", "B|C"),
        PairRecord("n1", "A", "C", 0, "train", "A|C"),
    )
    graph = build_train_positive_graph(PairTable("train", records))
    assert graph.node_count == 3
    assert graph.edge_count == 2
    assert graph.isolated_node_count == 0
    first = sample_walk_contexts(graph, 5, 2, 2, seed=99)
    second = sample_walk_contexts(graph, 5, 2, 2, seed=99)
    assert first == second
    assert first


def test_precomputed_embedding_backend_is_symmetric_and_trainable():
    config = ModelConfig(
        backend="precomputed_embedding",
        precomputed_dimension=4,
        precomputed_train_path="train.npz",
        precomputed_validation_path="validation.npz",
        precomputed_test_path="test.npz",
        embedding_dim=6,
        token_embedding_dim=4,
        pair_hidden_dim=8,
        chunk_size=8,
        chunk_stride=6,
        chunk_batch_size=2,
        convolution_kernel_size=3,
        dropout=0.0,
    )
    model = SITNEBXModel(config)
    model.eval()
    payload = torch.randn(3, 4)
    left = torch.tensor([0, 1])
    right = torch.tensor([1, 2])
    forward = model(payload, left, right)
    reverse = model(payload, right, left)
    assert torch.allclose(forward.binary_logits, reverse.binary_logits, atol=1e-7)
    forward.binary_logits.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
