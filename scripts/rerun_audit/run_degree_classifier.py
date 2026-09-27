#!/usr/bin/env python3
"""Degree-feature relation classifier（P0-3 的 C），输出 per-query scores/ranks。

与 relation-invariant degree tie baseline（A，所有 relation 同分，MRR≈0.408）彻底区分：
本 classifier 用 5 个 degree 特征经 MLP 输出 4 个 relation-specific logits。

5 特征（unordered pair）：
  f1=log1p(deg_h), f2=log1p(deg_t), f3=log1p(deg_h+deg_t),
  f4=log1p(max(deg_h,deg_t)), f5=log1p(min(deg_h,deg_t))
架构：2 层 MLP (5→64→64→4) + GELU，BCEWithLogitsLoss，AdamW。
输出（每 fold）：metrics.json + per_query_scores.tsv + per_query_ranks.tsv。

用法（CPU）：
    PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/run_degree_classifier.py --project-root "$PWD" \
        --output-dir rerun_submission_audit/03_controls/degree_classifier
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_code"))
from sitne_walk.data import (  # noqa: E402
    build_known_positive_index,
    load_indexed_triples,
    load_training_triples,
    iter_batches,
)
from sitne_walk.evaluator import evaluate_filtered_type_ranking  # noqa: E402


class DegreeClassifier(nn.Module):
    def __init__(self, num_relations, hidden_dim=64):
        super().__init__()
        self.num_relations = num_relations
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_relations),
        )

    def set_degree(self, degree):
        self.register_buffer("degree", degree)

    def _features(self, heads, tails):
        deg_h = self.degree[heads]
        deg_t = self.degree[tails]
        return torch.stack([
            torch.log1p(deg_h), torch.log1p(deg_t), torch.log1p(deg_h + deg_t),
            torch.log1p(torch.maximum(deg_h, deg_t)),
            torch.log1p(torch.minimum(deg_h, deg_t)),
        ], dim=-1)

    def score_all_relations(self, heads, tails):
        return self.net(self._features(heads, tails))


def compute_train_degree(triples):
    num_proteins = triples.num_proteins
    heads = triples.heads.numpy()
    tails = triples.tails.numpy()
    degree = np.zeros(num_proteins, dtype=np.float64)
    mask = heads != tails
    pair_codes = heads[mask] * np.int64(num_proteins) + tails[mask]
    unique = np.unique(pair_codes, return_index=True)[1]
    np.add.at(degree, heads[mask][unique], 1.0)
    np.add.at(degree, tails[mask][unique], 1.0)
    return degree


def run_fold(root, split_dir, fold, seed, device, out):
    train_path = split_dir / f"fold_{fold}" / "train.tsv"
    test_path = split_dir / f"fold_{fold}" / "test.tsv"
    val_path = split_dir / f"fold_{fold}" / "val.tsv"
    known_path = split_dir / "all_known_positives.tsv"

    triples = load_training_triples(str(train_path), duplicate_policy="binary")
    test_triples = load_indexed_triples(str(test_path), triples.protein_to_id, triples.relation_to_id)
    val_triples = load_indexed_triples(str(val_path), triples.protein_to_id, triples.relation_to_id)
    known = load_indexed_triples(str(known_path), triples.protein_to_id, triples.relation_to_id)
    known_index = build_known_positive_index(known, triples.num_proteins, triples.num_relations)

    num_proteins = triples.num_proteins
    num_relations = triples.num_relations
    degree = compute_train_degree(triples)
    heads = triples.heads.numpy()
    tails = triples.tails.numpy()
    relations = triples.relations.numpy()

    low = np.minimum(heads, tails)
    high = np.maximum(heads, tails)
    pair_key = low * np.int64(num_proteins) + high
    unique_keys, inverse = np.unique(pair_key, return_inverse=True)
    labels = np.zeros((len(unique_keys), num_relations), dtype=np.float32)
    labels[inverse, relations] = 1.0
    first_occ = np.unique(pair_key, return_index=True)[1]
    train_heads = torch.from_numpy(low[first_occ].astype(np.int64))
    train_tails = torch.from_numpy(high[first_occ].astype(np.int64))
    train_labels = torch.from_numpy(labels)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DegreeClassifier(num_relations).to(device)
    model.set_degree(torch.from_numpy(degree.astype(np.float32)).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_mrr = -1.0
    best_state = None
    no_improve = 0
    for epoch in range(50):
        model.train()
        perm = torch.randperm(len(train_heads))
        for batch in iter_batches(len(train_heads), 4096, perm):
            h = train_heads[batch].to(device)
            t = train_tails[batch].to(device)
            y = train_labels[batch].to(device)
            logits = model.score_all_relations(h, t)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_result = evaluate_filtered_type_ranking(
                model, val_triples, known_index, device,
                batch_size=1024, tie_policy="average", require_full_coverage=False,
            )
        val_mrr = val_result.metrics.get("mrr", 0.0)
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_result = evaluate_filtered_type_ranking(
            model, test_triples, known_index, device,
            batch_size=1024, tie_policy="average", require_full_coverage=False,
        )
        scores = model.score_all_relations(
            test_triples.heads.to(device), test_triples.tails.to(device)
        ).cpu().numpy()

    fold_dir = out / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    with (fold_dir / "metrics.json").open("w") as f:
        json.dump({"fold": fold, "seed": seed, "metrics": test_result.metrics}, f, indent=2)
    pd.DataFrame(
        {
            "head_id": test_triples.heads.numpy(),
            "relation_id": test_triples.relations.numpy(),
            "tail_id": test_triples.tails.numpy(),
            "filtered_rank": test_result.ranks.numpy(),
        }
    ).to_csv(fold_dir / "per_query_ranks.tsv", sep="\t", index=False)
    np.savez_compressed(
        fold_dir / "per_query_scores.npz", scores=scores,
        head_id=test_triples.heads.numpy(),
        relation_id=test_triples.relations.numpy(),
        tail_id=test_triples.tails.numpy(),
    )
    return test_result.metrics["mrr"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    split_dir = root / "05_splits/typed_ranking_v2"
    device = torch.device(args.device)

    results = []
    for fold in range(5):
        mrr = run_fold(root, split_dir, fold, args.seed, device, out)
        results.append({"fold": fold, "mrr": mrr})
        print(f"fold_{fold}: degree classifier MRR={mrr:.6f}")

    summary = out / "degree_classifier_summary.tsv"
    pd.DataFrame(results).to_csv(summary, sep="\t", index=False)
    mean = np.mean([r["mrr"] for r in results])
    print(f"mean MRR={mean:.6f} -> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
