#!/usr/bin/env python3
"""Seeded random baseline：deterministic replay + per-query scores。

- 对每个 unordered pair，用 pair_key（min(u,v)*N + max(u,v)）作为固定随机种子，
  生成 4 个 raw relation scores（deterministic，replay 一致）。
- 按 EVALUATION_CONTRACT 语义计算 filtered_rank（known-positive masking）与
  unfiltered_rank（不 masking），均 tie_policy=average。
- 输出 per_query_scores.tsv（含 4 个 raw scores + filtered/unfiltered/reciprocal rank）。

用法（CPU）：
    PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/run_random_baseline.py --project-root "$PWD" \
        --output-dir rerun_submission_audit/03_controls/random_baseline
"""
from __future__ import annotations

import argparse
import hashlib
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
SEED_OFFSET = 42


def _raw_scores(heads, tails, num_proteins, num_relations):
    """deterministic random scores：由 pair_key 决定，replay 一致。"""
    h = heads.numpy()
    t = tails.numpy()
    lo = np.minimum(h, t)
    hi = np.maximum(h, t)
    keys = lo * np.int64(num_proteins) + hi
    scores = np.zeros((len(keys), num_relations), dtype=np.float32)
    for i, k in enumerate(keys):
        rng = np.random.RandomState(int(k) + SEED_OFFSET)
        scores[i] = rng.rand(num_relations).astype(np.float32)
    return scores


def _unfiltered_rank(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_score = scores[np.arange(len(scores)), target]
    greater = (scores > target_score[:, None]).sum(1)
    equal = (scores == target_score[:, None]).sum(1) - 1
    return 1.0 + greater + 0.5 * equal


def _filtered_rank(scores: np.ndarray, target: np.ndarray, positive_mask) -> np.ndarray:
    """按 contract §4：mask 掉其它已知真类型（保留 target），再算 rank。"""
    masked = scores.copy()
    for i in range(len(scores)):
        for r in range(scores.shape[1]):
            if r != target[i] and positive_mask[i, r]:
                masked[i, r] = -np.inf
    return _unfiltered_rank(masked, target)


def run_fold(root, split_dir, fold):
    train_path = split_dir / f"fold_{fold}" / "train.tsv"
    test_path = split_dir / f"fold_{fold}" / "test.tsv"
    known_path = split_dir / "all_known_positives.tsv"

    triples = load_training_triples(str(train_path), duplicate_policy="binary")
    num_proteins = triples.num_proteins
    num_relations = triples.num_relations
    test = load_indexed_triples(str(test_path), triples.protein_to_id, triples.relation_to_id)
    known = load_indexed_triples(str(known_path), triples.protein_to_id, triples.relation_to_id)
    known_index = build_known_positive_index(known, num_proteins, num_relations)

    heads = test.heads.numpy()
    tails = test.tails.numpy()
    target = test.relations.numpy()

    scores = _raw_scores(torch.from_numpy(heads), torch.from_numpy(tails), num_proteins, num_relations)

    # positive_mask：该 pair 的已知真类型
    positive_mask = np.zeros((len(heads), num_relations), dtype=bool)
    for i in range(len(heads)):
        mask, matched = known_index.lookup(
            torch.tensor([heads[i]]), torch.tensor([tails[i]])
        )
        if matched.item():
            positive_mask[i] = mask[0].numpy()

    filtered = _filtered_rank(scores, target, positive_mask)
    unfiltered = _unfiltered_rank(scores, target)
    reciprocal = 1.0 / filtered

    df = pd.DataFrame({
        "fold": fold, "seed": SEED_OFFSET,
        "protein_u": [triples.id_to_protein[int(x)] for x in np.minimum(heads, tails)],
        "protein_v": [triples.id_to_protein[int(x)] for x in np.maximum(heads, tails)],
        "target_relation": [RELATION_NAMES[int(x)] for x in target],
        "target_relation_id": target,
        "score_enzymatic": scores[:, 0],
        "score_general": scores[:, 1],
        "score_physical": scores[:, 2],
        "score_spatial": scores[:, 3],
        "known_positive_mask": [",".join(str(r) for r in range(num_relations) if positive_mask[i, r]) for i in range(len(heads))],
        "num_known_relations": positive_mask.sum(1),
        "supported": True,
        "filtered_rank": filtered,
        "unfiltered_rank": unfiltered,
        "reciprocal_rank": reciprocal,
    })
    mrr = float(reciprocal.mean())
    return df, mrr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    split_dir = root / "05_splits/typed_ranking_v2"

    # replay 验证：跑两遍，比较 per-query 结果的 digest 一致
    digests = []
    for run_i in range(2):
        all_dfs = []
        mrrs = []
        for fold in range(5):
            df, mrr = run_fold(root, split_dir, fold)
            all_dfs.append(df)
            mrrs.append(mrr)
        concat = pd.concat(all_dfs, ignore_index=True)
        digest = hashlib.sha256(concat.to_csv(index=False).encode()).hexdigest()
        digests.append(digest)
        if run_i == 0:
            concat.to_csv(out / "per_query_scores.tsv", sep="\t", index=False)
            summary = pd.DataFrame({"fold": range(5), "mrr": mrrs})
            summary.to_csv(out / "random_baseline_summary.tsv", sep="\t", index=False)
            print(f"mean MRR={np.mean(mrrs):.6f} (per-fold: {[round(m, 4) for m in mrrs]})")

    replay_ok = digests[0] == digests[1]
    print(f"replay check: {'PASS (deterministic)' if replay_ok else 'FAIL'}")
    print(f"per_query_scores.tsv -> {out / 'per_query_scores.tsv'}")
    print(f"random_baseline_summary.tsv -> {out / 'random_baseline_summary.tsv'}")
    return 0 if replay_ok else 1


if __name__ == "__main__":
    sys.exit(main())
