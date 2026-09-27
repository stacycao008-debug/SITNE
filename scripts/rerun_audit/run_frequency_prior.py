#!/usr/bin/env python3
"""Train-frequency prior：deterministic replay + per-query ranks（Tier-1 baseline）。

用途：为 sensitivity 的 delta-vs-prior 提供 per-query 输出。score = train relation
frequency（所有 pair 相同），经统一 filtered ranking evaluator 评估。

输出（每个 fold）：
  - frequency_prior_metrics.json
  - frequency_prior_ranks.tsv（head_id / relation_id / tail_id / filtered_rank）

用法（CPU，deterministic）：
    PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/run_frequency_prior.py --project-root "$PWD" \
        --output-dir rerun_submission_audit/03_controls/frequency_prior
"""
from __future__ import annotations

import argparse
import json
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
from sitne_walk.evaluator import evaluate_filtered_type_ranking  # noqa: E402


class ScoreModelProxy(torch.nn.Module):
    def __init__(self, score_fn):
        super().__init__()
        self._score_fn = score_fn

    def score_all_relations(self, heads, tails):
        return self._score_fn(heads, tails)


def make_frequency_scores(train_triples, num_relations, device):
    freq = torch.zeros(num_relations, dtype=torch.float32)
    for r in train_triples.relations.tolist():
        freq[r] += 1.0
    scores_template = freq / freq.max()

    def score(heads, tails):
        return scores_template.unsqueeze(0).expand(heads.shape[0], -1).to(device)

    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    split_dir = root / "05_splits/typed_ranking_v2"
    device = torch.device(args.device)

    all_metrics = []
    for fold in range(5):
        train_path = split_dir / f"fold_{fold}" / "train.tsv"
        test_path = split_dir / f"fold_{fold}" / "test.tsv"
        known_path = split_dir / "all_known_positives.tsv"

        train_triples = load_training_triples(str(train_path), duplicate_policy="binary")
        test_triples = load_indexed_triples(
            str(test_path), train_triples.protein_to_id, train_triples.relation_to_id
        )
        known = load_indexed_triples(
            str(known_path), train_triples.protein_to_id, train_triples.relation_to_id
        )
        known_index = build_known_positive_index(
            known, train_triples.num_proteins, train_triples.num_relations
        )

        model = ScoreModelProxy(
            make_frequency_scores(train_triples, train_triples.num_relations, device)
        )
        result = evaluate_filtered_type_ranking(
            model,
            test_triples,
            known_index,
            device,
            batch_size=1024,
            hits_at=(1, 3, 10),
            tie_policy="average",
            require_full_coverage=False,
        )

        fold_dir = out / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / "frequency_prior_metrics.json").open("w") as f:
            json.dump({"fold": fold, "metrics": result.metrics}, f, indent=2)

        ranks = result.ranks
        pd.DataFrame(
            {
                "head_id": test_triples.heads.numpy(),
                "relation_id": test_triples.relations.numpy(),
                "tail_id": test_triples.tails.numpy(),
                "filtered_rank": ranks.numpy(),
            }
        ).to_csv(fold_dir / "frequency_prior_ranks.tsv", sep="\t", index=False)

        mrr = result.metrics["mrr"]
        all_metrics.append({"fold": fold, "mrr": mrr})
        print(f"fold_{fold}: frequency prior MRR={mrr:.6f}")

    summary = out / "frequency_prior_summary.tsv"
    pd.DataFrame(all_metrics).to_csv(summary, sep="\t", index=False)
    mean = np.mean([m["mrr"] for m in all_metrics])
    print(f"mean MRR={mean:.6f} -> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
