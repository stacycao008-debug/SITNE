from __future__ import annotations

import pytest
import torch
from torch import nn

from sitne_walk.data import IndexedTriples, PairRelationIndex
from sitne_walk.evaluator import _ranks_from_scores, evaluate_filtered_type_ranking


class FixedScoreModel(nn.Module):
    def __init__(self, scores: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_scores", scores)

    def score_all_relations(
        self,
        head_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.fixed_scores[: head_ids.numel()].to(head_ids.device)


def _indexed() -> IndexedTriples:
    return IndexedTriples(
        heads=torch.tensor([0]),
        relations=torch.tensor([0]),
        tails=torch.tensor([1]),
        source_path="toy",
        raw_input_rows=1,
        input_rows=1,
        supported_rows=1,
        unsupported_protein_rows=0,
        unsupported_relation_rows=0,
        pair_semantics="unordered",
    )


def test_filtered_evaluator_removes_other_true_relation() -> None:
    # relation 1 得分最高，但它也是同 pair 的已知阳性，应在 target=0 时过滤。
    model = FixedScoreModel(torch.tensor([[0.5, 0.9, 0.4]]))
    index, _ = PairRelationIndex.build(
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        torch.tensor([1, 1]),
        num_proteins=3,
        num_relations=3,
        pair_semantics="unordered",
    )
    result = evaluate_filtered_type_ranking(
        model, _indexed(), index, "cpu", batch_size=1
    )
    assert result.metrics["mrr"] == pytest.approx(1.0)
    assert result.ranks.tolist() == [1.0]


def test_missing_target_in_filter_fails() -> None:
    model = FixedScoreModel(torch.tensor([[0.5, 0.9, 0.4]]))
    index, _ = PairRelationIndex.build(
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([1]),
        num_proteins=3,
        num_relations=3,
        pair_semantics="unordered",
    )
    with pytest.raises(ValueError, match="未包含当前 target"):
        evaluate_filtered_type_ranking(model, _indexed(), index, "cpu")
    assert model.training


def test_non_finite_scores_fail_instead_of_producing_invalid_rank() -> None:
    model = FixedScoreModel(torch.tensor([[float("nan"), 0.9, 0.4]]))
    index, _ = PairRelationIndex.build(
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([1]),
        num_proteins=3,
        num_relations=3,
        pair_semantics="unordered",
    )
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        evaluate_filtered_type_ranking(model, _indexed(), index, "cpu")
    assert model.training


def test_average_tie_policy() -> None:
    scores = torch.tensor([[1.0, 1.0, 0.5]])
    target = torch.tensor([0])
    assert _ranks_from_scores(scores, target, "optimistic").item() == 1.0
    assert _ranks_from_scores(scores, target, "pessimistic").item() == 2.0
    assert _ranks_from_scores(scores, target, "average").item() == 1.5


def test_zero_supported_queries_are_structured_unsupported() -> None:
    triples = IndexedTriples(
        heads=torch.empty(0, dtype=torch.int64),
        relations=torch.empty(0, dtype=torch.int64),
        tails=torch.empty(0, dtype=torch.int64),
        source_path="protein-disjoint",
        raw_input_rows=5,
        input_rows=5,
        supported_rows=0,
        unsupported_protein_rows=5,
        unsupported_relation_rows=0,
        pair_semantics="unordered",
    )
    index, _ = PairRelationIndex.build(
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([1]),
        num_proteins=2,
        num_relations=2,
    )
    result = evaluate_filtered_type_ranking(
        FixedScoreModel(torch.tensor([[0.0, 0.0]])),
        triples,
        index,
        "cpu",
        require_full_coverage=False,
    )
    assert result.status == "unsupported"
    assert result.metrics == {}
