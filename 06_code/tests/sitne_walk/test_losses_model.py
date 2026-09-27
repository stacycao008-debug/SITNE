from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from sitne_walk.gradient_reversal import gradient_reverse
from sitne_walk.losses import (
    decorrelation_loss,
    filtered_pairwise_ranking_loss,
    hierarchy_loss_disabled,
    topology_sgns_loss,
    typed_relation_cross_entropy,
)
from sitne_walk.model import SITNEWalkModel


def test_topology_sgns_matches_manual_formula() -> None:
    positive = torch.tensor([0.2, -0.4])
    negative = torch.tensor([[0.1, -0.3], [0.5, -0.2]])
    expected = (
        -F.logsigmoid(positive) - F.logsigmoid(-negative).sum(dim=1)
    ).mean()
    assert torch.allclose(topology_sgns_loss(positive, negative), expected)


def test_typed_relation_cross_entropy_matches_pytorch() -> None:
    logits = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    targets = torch.tensor([1, 0])
    assert torch.allclose(
        typed_relation_cross_entropy(logits, targets),
        F.cross_entropy(logits, targets),
    )


def test_decorrelation_is_permutation_invariant_and_detects_correlation() -> None:
    topology = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    semantic = topology.clone()
    value = decorrelation_loss(topology, semantic)
    permutation = torch.tensor([2, 0, 1])
    assert value > 0
    assert torch.allclose(
        value,
        decorrelation_loss(topology[permutation], semantic[permutation]),
    )
    with pytest.raises(ValueError, match="至少含两个"):
        decorrelation_loss(topology[:1], semantic[:1])


def test_filtered_rank_loss_ignores_other_positive_types() -> None:
    scores = torch.tensor([[2.0, 1.5, 0.0]], requires_grad=True)
    positives = torch.tensor([[True, True, False]])
    base = filtered_pairwise_ranking_loss(scores, positives)
    improved = filtered_pairwise_ranking_loss(
        torch.tensor([[3.0, 2.5, 0.0]]), positives
    )
    assert improved < base
    base.backward()
    # 两个正类型都应被推高，未标注候选应被压低。
    assert scores.grad is not None
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 1] < 0
    assert scores.grad[0, 2] > 0


def test_gradient_reversal_only_reverses_encoder_gradient() -> None:
    inputs = torch.tensor([[1.0, -2.0]], requires_grad=True)
    weight = torch.tensor([[3.0], [4.0]], requires_grad=True)
    output = gradient_reverse(inputs, coefficient=0.5) @ weight
    output.sum().backward()
    assert torch.allclose(inputs.grad, torch.tensor([[-1.5, -2.0]]))
    # classifier 参数仍按正常方向更新，不经反转。
    assert torch.allclose(weight.grad, torch.tensor([[1.0], [-2.0]]))


def test_hierarchy_is_explicitly_disabled() -> None:
    relation = torch.randn(3, 4, requires_grad=True)
    zero = hierarchy_loss_disabled(relation, coefficient=0.0)
    assert zero.item() == 0.0
    with pytest.raises(ValueError, match="不允许"):
        hierarchy_loss_disabled(relation, coefficient=0.1)


def test_decoder_is_symmetric_for_unordered_pairs() -> None:
    model = SITNEWalkModel(
        num_proteins=5,
        num_relations=3,
        embedding_dim=4,
        decoder_dropout=0.0,
    )
    model.eval()
    forward = model.score_all_relations(torch.tensor([1]), torch.tensor([3]))
    reverse = model.score_all_relations(torch.tensor([3]), torch.tensor([1]))
    assert torch.allclose(forward, reverse)
    assert forward.shape == (1, 3)


def test_model_logits_have_expected_shapes() -> None:
    model = SITNEWalkModel(8, 3, 6, {"degree": 4}, nuisance_hidden_dim=5)
    centers = torch.tensor([0, 1, 2])
    positive = torch.tensor([1, 2, 3])
    negative = torch.tensor([[2, 3], [3, 4], [4, 5]])
    pos_logits, neg_logits = model.topology_sgns_logits(centers, positive, negative)
    assert pos_logits.shape == (3,)
    assert neg_logits.shape == (3, 2)
    assert model.typed_relation_logits(centers).shape == (3, 3)
    assert model.nuisance_logits(centers, "degree", 1.0).shape == (3, 4)

