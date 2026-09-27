#!/usr/bin/env python3
"""把 frequency prior / degree classifier 归一化到统一 per_query_scores.tsv schema。

- frequency prior：重新生成 4 个 raw relation-frequency scores + filtered/unfiltered rank。
- degree classifier：读 scores.npz 的 4 raw scores，补 unfiltered_rank + known_positive_mask，
  并做 replay gate（重算 filtered_rank vs 旧 per_query_ranks.tsv）。

用法（CPU）：
    PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/normalize_per_query_scores.py --project-root "$PWD"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_code"))
from sitne_walk.data import (  # noqa: E402
    build_known_positive_index,
    load_indexed_triples,
    load_training_triples,
)

RELATION_NAMES = {0: "enzymatic", 1: "general", 2: "physical", 3: "spatial"}


def _unfiltered_rank(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    ts = scores[np.arange(len(scores)), targets]
    greater = (scores > ts[:, None]).sum(1)
    equal = (scores == ts[:, None]).sum(1) - 1
    return 1.0 + greater + 0.5 * equal


def _filtered_rank(scores: np.ndarray, targets: np.ndarray, pm: np.ndarray) -> np.ndarray:
    masked = scores.copy()
    for i in range(len(scores)):
        for r in range(scores.shape[1]):
            if r != targets[i] and pm[i, r]:
                masked[i, r] = -np.inf
    return _unfiltered_rank(masked, targets)


def _build_df(fold, seed, triples, heads, tails, targets, scores, pm):
    lo = np.minimum(heads, tails)
    hi = np.maximum(heads, tails)
    filtered = _filtered_rank(scores, targets, pm)
    unfiltered = _unfiltered_rank(scores, targets)
    return pd.DataFrame({
        "fold": fold, "seed": seed,
        "protein_u": [triples.id_to_protein[int(x)] for x in lo],
        "protein_v": [triples.id_to_protein[int(x)] for x in hi],
        "target_relation": [RELATION_NAMES[int(x)] for x in targets],
        "target_relation_id": targets,
        "score_enzymatic": scores[:, 0],
        "score_general": scores[:, 1],
        "score_physical": scores[:, 2],
        "score_spatial": scores[:, 3],
        "known_positive_mask": [",".join(str(r) for r in range(triples.num_relations) if pm[i, r]) for i in range(len(heads))],
        "num_known_relations": pm.sum(1),
        "supported": True,
        "filtered_rank": filtered,
        "unfiltered_rank": unfiltered,
        "reciprocal_rank": 1.0 / filtered,
    })


def normalize_frequency_prior(root: Path) -> None:
    base = root / "rerun_submission_audit/03_controls/frequency_prior"
    split = root / "05_splits/typed_ranking_v2"
    for fold in range(5):
        triples = load_training_triples(str(split / f"fold_{fold}/train.tsv"), duplicate_policy="binary")
        test = load_indexed_triples(str(split / f"fold_{fold}/test.tsv"), triples.protein_to_id, triples.relation_to_id)
        known = load_indexed_triples(str(split / "all_known_positives.tsv"), triples.protein_to_id, triples.relation_to_id)
        known_index = build_known_positive_index(known, triples.num_proteins, triples.num_relations)

        freq = np.bincount(triples.relations.numpy(), minlength=triples.num_relations).astype(np.float64)
        freq = freq / freq.max()
        heads = test.heads.numpy()
        tails = test.tails.numpy()
        targets = test.relations.numpy()
        scores = np.tile(freq, (len(heads), 1)).astype(np.float32)

        pm = np.zeros((len(heads), triples.num_relations), dtype=bool)
        for i in range(len(heads)):
            mask, matched = known_index.lookup(torch.tensor([heads[i]]), torch.tensor([tails[i]]))
            if matched.item():
                pm[i] = mask[0].numpy()

        df = _build_df(fold, -1, triples, heads, tails, targets, scores, pm)
        df.to_csv(base / f"fold_{fold}/per_query_scores.tsv", sep="\t", index=False)
    print(f"frequency prior normalized -> {base}/fold_*/per_query_scores.tsv")


def normalize_degree_classifier(root: Path) -> None:
    base = root / "rerun_submission_audit/03_controls/degree_classifier"
    split = root / "05_splits/typed_ranking_v2"
    for fold in range(5):
        triples = load_training_triples(str(split / f"fold_{fold}/train.tsv"), duplicate_policy="binary")
        test = load_indexed_triples(str(split / f"fold_{fold}/test.tsv"), triples.protein_to_id, triples.relation_to_id)
        known = load_indexed_triples(str(split / "all_known_positives.tsv"), triples.protein_to_id, triples.relation_to_id)
        known_index = build_known_positive_index(known, triples.num_proteins, triples.num_relations)

        npz = np.load(base / f"fold_{fold}/per_query_scores.npz")
        scores = npz["scores"]
        heads = npz["head_id"]
        targets = npz["relation_id"]
        tails = npz["tail_id"]

        pm = np.zeros((len(heads), triples.num_relations), dtype=bool)
        for i in range(len(heads)):
            mask, matched = known_index.lookup(torch.tensor([heads[i]]), torch.tensor([tails[i]]))
            if matched.item():
                pm[i] = mask[0].numpy()

        df = _build_df(fold, 42, triples, heads, tails, targets, scores, pm)
        df.to_csv(base / f"fold_{fold}/per_query_scores.tsv", sep="\t", index=False)

        # replay gate：重算 filtered_rank vs 旧 per_query_ranks.tsv
        old = pd.read_csv(base / f"fold_{fold}/per_query_ranks.tsv", sep="\t")
        mismatch = ~np.isclose(old["filtered_rank"].to_numpy(), df["filtered_rank"].to_numpy())
        print(f"degree classifier fold_{fold}: replay mismatch={int(mismatch.sum())}/{len(df)}")
    print(f"degree classifier normalized -> {base}/fold_*/per_query_scores.tsv")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    normalize_frequency_prior(root)
    normalize_degree_classifier(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
