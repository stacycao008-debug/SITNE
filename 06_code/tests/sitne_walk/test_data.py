from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sitne_walk.data import (
    IndexedTriples,
    PairRelationIndex,
    audit_split_pair_overlap,
    load_indexed_triples,
    load_training_triples,
    merge_indexed_triples,
    validate_split_manifest,
)


def test_unordered_loader_canonicalizes_and_deduplicates(toy_train_tsv: Path) -> None:
    triples = load_training_triples(toy_train_tsv, pair_semantics="unordered")
    assert triples.input_rows == 7
    assert triples.unique_rows == 6
    assert triples.num_proteins == 4
    assert triples.num_relations == 2
    # AB 同时有 r1/r2，但反向重复 r1 只保留一次。
    a = triples.protein_to_id["A"]
    b = triples.protein_to_id["B"]
    forward_mask, forward_hit = triples.pair_index.lookup(
        torch.tensor([a]), torch.tensor([b])
    )
    reverse_mask, reverse_hit = triples.pair_index.lookup(
        torch.tensor([b]), torch.tensor([a])
    )
    assert forward_hit.item() and reverse_hit.item()
    assert torch.equal(forward_mask, reverse_mask)
    assert forward_mask.sum().item() == 2


def test_pair_relation_index_unknown_pair_returns_empty(toy_train_tsv: Path) -> None:
    triples = load_training_triples(toy_train_tsv)
    index: PairRelationIndex = triples.pair_index
    a = triples.protein_to_id["A"]
    c = triples.protein_to_id["C"]
    mask, matched = index.lookup(torch.tensor([a]), torch.tensor([c]))
    assert not matched.item()
    assert not mask.any().item()


def test_pair_relation_index_rejects_out_of_range_ids() -> None:
    with pytest.raises(ValueError, match="relation ID 越界"):
        PairRelationIndex.build(
            torch.tensor([0]),
            torch.tensor([2]),
            torch.tensor([1]),
            num_proteins=2,
            num_relations=2,
        )


def test_merge_indexed_triples_rejects_mixed_pair_semantics() -> None:
    common = {
        "heads": torch.tensor([0]),
        "relations": torch.tensor([0]),
        "tails": torch.tensor([1]),
        "source_path": "toy",
        "raw_input_rows": 1,
        "input_rows": 1,
        "supported_rows": 1,
        "unsupported_protein_rows": 0,
        "unsupported_relation_rows": 0,
    }
    unordered = IndexedTriples(**common, pair_semantics="unordered")
    ordered = IndexedTriples(**common, pair_semantics="ordered")
    with pytest.raises(ValueError, match="pair_semantics 不一致"):
        merge_indexed_triples([unordered, ordered])


def test_indexed_split_reports_unseen_protein_and_relation(
    toy_train_tsv: Path,
    tmp_path: Path,
) -> None:
    train = load_training_triples(toy_train_tsv)
    evaluation = tmp_path / "evaluation.tsv"
    evaluation.write_text(
        "protein_i\tprotein_j\ttype_name\n"
        "A\tB\tr1\n"
        "A\tX\tr1\n"
        "A\tB\tr3\n",
        encoding="utf-8",
    )
    indexed = load_indexed_triples(
        evaluation,
        train.protein_to_id,
        train.relation_to_id,
    )
    assert indexed.input_rows == 3
    assert indexed.supported_rows == 1
    assert indexed.unsupported_protein_rows == 1
    assert indexed.unsupported_relation_rows == 1


def test_loader_preserves_opaque_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "opaque_ids.tsv"
    path.write_text(
        "protein_i\tprotein_j\ttype_name\n001\tNA\t0007\n",
        encoding="utf-8",
    )
    triples = load_training_triples(path)
    assert set(triples.id_to_protein) == {"001", "NA"}
    assert triples.id_to_relation == ("0007",)


def test_manifest_count_mismatch_fails(toy_train_tsv: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"num_train": 999}), encoding="utf-8")
    with pytest.raises(ValueError, match="行数冲突"):
        validate_split_manifest(manifest, toy_train_tsv)


def test_manifest_hash_mismatch_fails(toy_train_tsv: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"num_train": 7, "train_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 冲突"):
        validate_split_manifest(manifest, toy_train_tsv)


def test_pair_overlap_audit_fails(toy_train_tsv: Path, tmp_path: Path) -> None:
    validation = tmp_path / "validation.tsv"
    validation.write_text(
        "protein_i\tprotein_j\ttype_name\n B\tA \tr2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pair overlap"):
        audit_split_pair_overlap(toy_train_tsv, validation)
