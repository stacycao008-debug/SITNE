#!/usr/bin/env python3
"""shortcut_only 消融变体 — 仅用 degree/frequency 统计特征预测 relation。

设计（与 EXPERIMENT_PLAN.md Phase D #10 一致）：
- 不使用 walk 生成、不使用 embedding 学习；
- 仅使用 train-only 的 protein degree 与 relation frequency 统计特征；
- 通过一个小型 MLP 把 pair 的 degree 特征映射为 relation scores；
- 使用与 SITNE-Walk 完全相同的 filtered type-ranking evaluator 评估。

与 degree/binary baseline 的区别：这里 degree 特征经 MLP 非线性变换后
可为每个 relation 产生不同的分数（而非所有 relation 共享同一分数）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shortcut_only")

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "06_code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from sitne_walk.data import (
    load_training_triples,
    load_indexed_triples,
    build_known_positive_index,
    merge_indexed_triples,
    IndexedTriples,
    iter_batches,
)
from sitne_walk.evaluator import evaluate_filtered_type_ranking

SPLIT_DIR = ROOT / "05_splits/typed_ranking_v2"
OUTPUT_BASE = ROOT / "08_results/sitne_walk_paper_v2/r3_ablation/shortcut_only"


class ShortcutOnlyModel(nn.Module):
    """仅依赖 degree 统计特征的 relation scorer。

    特征（对每个 unordered pair）：
      [log1p(deg_h), log1p(deg_t), log1p(deg_h + deg_t),
       log1p(max(deg_h, deg_t)), log1p(min(deg_h, deg_t))]
    经两层 MLP 映射为 num_relations 个 relation logits。
    """

    def __init__(self, num_relations: int, hidden_dim: int = 64):
        super().__init__()
        self.num_relations = num_relations
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_relations),
        )

    def _pair_features(self, heads: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        deg_h = self.degree[heads]
        deg_t = self.degree[tails]
        f_h = torch.log1p(deg_h)
        f_t = torch.log1p(deg_t)
        f_sum = torch.log1p(deg_h + deg_t)
        f_max = torch.log1p(torch.maximum(deg_h, deg_t))
        f_min = torch.log1p(torch.minimum(deg_h, deg_t))
        return torch.stack([f_h, f_t, f_sum, f_max, f_min], dim=-1)

    def set_degree(self, degree: torch.Tensor):
        self.register_buffer("degree", degree)

    def score_all_relations(self, heads: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        return self.net(self._pair_features(heads, tails))


def compute_train_degree(triples) -> np.ndarray:
    """train-only 唯一非自环邻居 degree。"""
    num_proteins = triples.num_proteins
    heads = triples.heads.numpy()
    tails = triples.tails.numpy()
    degree = np.zeros(num_proteins, dtype=np.float64)
    mask = heads != tails
    pair_codes = heads[mask] * np.int64(num_proteins) + tails[mask]
    unique_positions = np.unique(pair_codes, return_index=True)[1]
    np.add.at(degree, heads[mask][unique_positions], 1.0)
    np.add.at(degree, tails[mask][unique_positions], 1.0)
    return degree


def run_fold(fold: int, seed: int = 42) -> dict:
    train_path = SPLIT_DIR / f"fold_{fold}" / "train.tsv"
    test_path = SPLIT_DIR / f"fold_{fold}" / "test.tsv"
    val_path = SPLIT_DIR / f"fold_{fold}" / "val.tsv"
    known_path = SPLIT_DIR / "all_known_positives.tsv"

    output_dir = OUTPUT_BASE / f"fold_{fold}" / f"seed_{seed}"
    test_metrics_path = output_dir / "test_metrics.json"
    if test_metrics_path.exists():
        with open(test_metrics_path) as f:
            return json.load(f)["metrics"]

    output_dir.mkdir(parents=True, exist_ok=True)

    triples = load_training_triples(str(train_path), duplicate_policy="binary")
    test_triples = load_indexed_triples(str(test_path), triples.protein_to_id, triples.relation_to_id)
    val_triples = load_indexed_triples(str(val_path), triples.protein_to_id, triples.relation_to_id)

    # known-positive filter
    known = load_indexed_triples(str(known_path), triples.protein_to_id, triples.relation_to_id)
    known_index = build_known_positive_index(known, triples.num_proteins, triples.num_relations)

    num_proteins = triples.num_proteins
    num_relations = triples.num_relations

    # 训练特征与标签：每个 unique pair 的所有 train relation 类型（多标签）
    degree = compute_train_degree(triples)
    heads = triples.heads.numpy()
    tails = triples.tails.numpy()
    relations = triples.relations.numpy()

    # 按 pair 聚合多标签
    low = np.minimum(heads, tails)
    high = np.maximum(heads, tails)
    pair_key = low * np.int64(num_proteins) + high
    unique_keys, inverse = np.unique(pair_key, return_inverse=True)
    labels = np.zeros((len(unique_keys), num_relations), dtype=np.float32)
    labels[inverse, relations] = 1.0

    # 每个 pair 的 (h, t) 端点（取第一个出现）
    first_occ = np.unique(pair_key, return_index=True)[1]
    train_heads = torch.from_numpy(low[first_occ].astype(np.int64))
    train_tails = torch.from_numpy(high[first_occ].astype(np.int64))
    train_labels = torch.from_numpy(labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ShortcutOnlyModel(num_relations).to(device)
    model.set_degree(torch.from_numpy(degree.astype(np.float32)).to(device))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_mrr = -1.0
    best_state = None
    epochs_no_improve = 0

    # 预计算 val 端点的 degree 特征在评估时由 model 内部完成
    for epoch in range(50):
        model.train()
        perm = torch.randperm(len(train_heads))
        total_loss = 0.0
        n_batches = 0
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
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # validation
        model.eval()
        with torch.no_grad():
            val_result = evaluate_filtered_type_ranking(
                model, val_triples, known_index, device,
                batch_size=1024, tie_policy="average", require_full_coverage=False,
            )
        val_mrr = val_result.metrics.get("mrr", 0.0)

        if epoch % 5 == 0 or epoch < 3:
            logger.info("fold_%d epoch %2d loss=%.4f val_mrr=%.4f", fold, epoch, avg_loss, val_mrr)

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5:
                logger.info("fold_%d early stop @ epoch %d", fold, epoch)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_result = evaluate_filtered_type_ranking(
            model, test_triples, known_index, device,
            batch_size=1024, tie_policy="average", require_full_coverage=False,
        )
    metrics = test_result.metrics

    with open(test_metrics_path, "w") as f:
        json.dump({"metrics": metrics, "fold": fold, "seed": seed}, f, indent=2)

    logger.info("fold_%d shortcut_only: MRR=%.4f", fold, metrics["mrr"])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    folds = list(range(5)) if args.all_folds else [args.fold if args.fold is not None else 0]

    results = []
    for fold in folds:
        m = run_fold(fold, args.seed)
        results.append({"fold": fold, **m})

    import pandas as pd
    df = pd.DataFrame(results)
    summary = OUTPUT_BASE / "shortcut_only_summary.tsv"
    df.to_csv(summary, sep="\t", index=False)
    logger.info("Summary: %s", summary)
    for _, row in df.iterrows():
        logger.info("fold_%d: MRR=%.4f", row["fold"], row["mrr"])


if __name__ == "__main__":
    main()
