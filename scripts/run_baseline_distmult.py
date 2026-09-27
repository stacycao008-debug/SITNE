#!/usr/bin/env python3
"""DistMult baseline for SITNE-Walk paper.

Uses the same fold/seed splits, filtered-type ranking evaluator, and
all-known-positive masking as SITNE-Walk. This ensures a fair head-to-head
comparison that only differs in the modelling approach.

DistMult scoring: f(h,r,t) = ⟨e_h, r_r, e_t⟩ = sum(e_h * r_r * e_t)

Usage:
    # Single fold (default: fold_0)
    python scripts/run_baseline_distmult.py

    # All 5 folds
    python scripts/run_baseline_distmult.py --all-folds

    # Specific fold
    python scripts/run_baseline_distmult.py --fold 3
"""
import argparse
import json, logging, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("distmult_baseline")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "06_code"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sitne_walk.data import (
    TrainingTriples,
    IndexedTriples,
    PairRelationIndex,
    build_known_positive_index,
    load_training_triples,
    load_indexed_triples,
    merge_indexed_triples,
    iter_batches,
)
from sitne_walk.evaluator import evaluate_filtered_type_ranking
from sitne_walk.provenance import guard_historical_output


# ---------------------------------------------------------------------------
# DistMult Model
# ---------------------------------------------------------------------------

class DistMultModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.embedding_dim = embedding_dim
        self.scale = embedding_dim ** 0.5
        self._reset_parameters()

    def _reset_parameters(self):
        bound = 1.0 / self.embedding_dim ** 0.5
        nn.init.uniform_(self.entity_embedding.weight, -bound, bound)
        nn.init.uniform_(self.relation_embedding.weight, -bound, bound)

    def score_all_relations(self, head_ids: torch.Tensor, tail_ids: torch.Tensor) -> torch.Tensor:
        """Score all relation types for each pair: sum(h * r_k * t) for all k."""
        h = self.entity_embedding(head_ids)       # [B, D]
        t = self.entity_embedding(tail_ids)       # [B, D]
        pair = h * t                               # [B, D]
        scores = pair @ self.relation_embedding.weight.transpose(0, 1) / self.scale  # [B, R]
        return scores

    def score_triples(self, head_ids, relation_ids, tail_ids):
        all_scores = self.score_all_relations(head_ids, tail_ids)
        return all_scores.gather(1, relation_ids.reshape(-1, 1)).squeeze(1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_distmult(
    model: DistMultModel,
    triples: TrainingTriples,
    val_triples: IndexedTriples,
    test_triples: IndexedTriples,
    known_positive: PairRelationIndex,
    device: torch.device,
    *,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-5,
    negative_samples: int = 16,
    early_stopping_patience: int = 5,
    output_dir: Path,
    seed: int = 42,
    fold: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    heads = triples.heads
    relations = triples.relations
    tails = triples.tails
    num_entities = triples.num_proteins
    num_relations_val = triples.num_relations

    # Negative sampling distribution (unigram^0.75)
    head_counts = torch.bincount(heads, minlength=num_entities).float()
    tail_counts = torch.bincount(tails, minlength=num_entities).float()
    total_counts = head_counts + tail_counts
    total_counts.clamp_min_(1)
    neg_weights = total_counts ** 0.75
    neg_dist = neg_weights / neg_weights.sum()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mrr = -1.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        perm = torch.randperm(len(heads))
        for batch_idx in iter_batches(len(heads), batch_size, perm):
            idx = batch_idx
            h = heads[idx].to(device)
            r = relations[idx].to(device)
            t = tails[idx].to(device)
            batch_size_actual = len(h)

            # Positive scores
            pos_scores = model.score_triples(h, r, t)

            # Negative samples: corrupt tail
            neg_tails = torch.multinomial(neg_dist, batch_size_actual * negative_samples, replacement=True).to(device)
            neg_tails_flat = neg_tails  # [batch * neg]
            h_expanded = h.unsqueeze(1).expand(-1, negative_samples).reshape(-1)      # [batch * neg]
            r_expanded = r.unsqueeze(1).expand(-1, negative_samples).reshape(-1)      # [batch * neg]
            neg_scores = model.score_triples(h_expanded, r_expanded, neg_tails_flat)  # [batch * neg]
            neg_scores = neg_scores.view(batch_size_actual, negative_samples)

            # Margin ranking loss
            pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, negative_samples)
            margin_loss = torch.clamp(1.0 + neg_scores - pos_scores_expanded, min=0).mean()

            # L2 regularization on embeddings
            reg = (model.entity_embedding.weight.norm(p=2) + model.relation_embedding.weight.norm(p=2)) * 0.0001

            loss = margin_loss + reg
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        # Validation
        if val_triples.supported_rows > 0:
            val_result = evaluate_filtered_type_ranking(
                model, val_triples, known_positive, device,
                batch_size=128, tie_policy="average", require_full_coverage=False,
            )
            val_mrr = val_result.metrics.get("mrr", 0.0)
        else:
            val_mrr = 0.0

        history.append({"epoch": epoch, "loss": avg_loss, "val_mrr": val_mrr})

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        log.info("epoch %3d  loss=%.4f  val_mrr=%.4f  best=%.4f@%d",
                 epoch, avg_loss, val_mrr, best_val_mrr, best_epoch)

        if epochs_no_improve >= early_stopping_patience:
            log.info("Early stopping at epoch %d", epoch)
            break

    # Restore best and test
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_result = evaluate_filtered_type_ranking(
        model, test_triples, known_positive, device,
        batch_size=128, tie_policy="average", require_full_coverage=False,
    )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(test_result.metrics)
    coverage = test_result.supported_rows / test_result.input_rows if test_result.input_rows else 0.0

    result = {
        "model": "DistMult",
        "embedding_dim": model.embedding_dim,
        "fold": fold,
        "seed": seed,
        "epochs_trained": best_epoch + 1 if best_epoch >= 0 else 0,
        "best_val_mrr": best_val_mrr,
        "test_mrr": metrics.get("mrr", 0.0),
        "test_hits1": metrics.get("hits@1", 0.0),
        "test_hits3": metrics.get("hits@3", 0.0),
        "test_hits10": metrics.get("hits@10", 0.0),
        "coverage": coverage,
        "status": test_result.status,
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "input_rows": test_result.input_rows,
                    "supported_rows": test_result.supported_rows}, f, indent=2)

    # Save ranks
    ranks_df = pd.DataFrame({
        "head_id": test_triples.heads.numpy().astype(int),
        "tail_id": test_triples.tails.numpy().astype(int),
        "relation_id": test_result.relation_ids.numpy().astype(int),
        "filtered_rank": test_result.ranks.numpy().astype(float),
        "fold": fold,
        "seed": seed,
    })
    ranks_df["pair_key"] = ranks_df.apply(
        lambda r: f"{min(r.head_id, r.tail_id)}_{max(r.head_id, r.tail_id)}", axis=1)
    ranks_df["fold"] = fold
    ranks_df.to_csv(output_dir / "test_ranks.tsv", sep="\t", index=False)

    with open(output_dir / "provenance.json", "w") as f:
        json.dump({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": "DistMult",
            "embedding_dim": model.embedding_dim,
            "seed": seed,
            "fold": fold,
        }, f, indent=2)

    log.info("DistMult fold %d seed %d: test MRR=%.4f Hits@1=%.3f",
             fold, seed, result["test_mrr"], result["test_hits1"])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_fold(fold: int, device: torch.device, params: dict) -> dict:
    """对单个 fold 执行 DistMult 训练与评估。"""
    # Config
    EMBEDDING_DIM = params["embedding_dim"]
    EPOCHS = params["epochs"]
    BATCH_SIZE = params["batch_size"]
    LR = params["learning_rate"]
    SEED = params["seed"]

    train_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
    test_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/test.tsv"
    val_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/val.tsv"
    known_pos_path = ROOT / "05_splits/typed_ranking_v2/all_known_positives.tsv"

    if not train_path.exists() or not test_path.exists():
        log.warning("fold_%d 数据缺失，跳过", fold)
        return {"fold": fold, "status": "data_missing"}

    log.info("=== Fold %d: 加载数据 ===", fold)
    triples = load_training_triples(
        str(train_path), weight_column=None, duplicate_policy="binary", pair_semantics="unordered")

    log.info("Fold %d: 加载测试数据...", fold)
    test_triples = load_indexed_triples(
        str(test_path), triples.protein_to_id, triples.relation_to_id, pair_semantics="unordered")

    # 加载 fold 专用 known_positives（包含 train+val+test 所有已知正例）
    val_triples_raw = load_indexed_triples(
        str(val_path), triples.protein_to_id, triples.relation_to_id, pair_semantics="unordered")

    if known_pos_path.exists():
        known_pos = load_indexed_triples(
            str(known_pos_path), triples.protein_to_id, triples.relation_to_id, pair_semantics="unordered")
    else:
        known_pos = merge_indexed_triples([
            IndexedTriples(
                heads=triples.heads, relations=triples.relations, tails=triples.tails,
                source_path=str(train_path), raw_input_rows=triples.input_rows,
                input_rows=triples.input_rows, supported_rows=triples.input_rows,
                unsupported_protein_rows=0, unsupported_relation_rows=0,
                pair_semantics=triples.pair_semantics),
            val_triples_raw,
            test_triples,
        ])

    known_positive = build_known_positive_index(
        known_pos, triples.num_proteins, triples.num_relations)

    # 验证集：从 train 中随机采 10% 用于 early stopping
    n_val = max(100, int(triples.num_triples * 0.1))
    perm = torch.randperm(triples.num_triples, generator=torch.Generator().manual_seed(SEED))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    val_triples = IndexedTriples(
        heads=triples.heads[val_idx], relations=triples.relations[val_idx],
        tails=triples.tails[val_idx],
        source_path=str(train_path), raw_input_rows=n_val, input_rows=n_val,
        supported_rows=n_val, unsupported_protein_rows=0, unsupported_relation_rows=0,
        pair_semantics=triples.pair_semantics,
    )
    train_subset = TrainingTriples(
        heads=triples.heads[train_idx], relations=triples.relations[train_idx],
        tails=triples.tails[train_idx], edge_weights=triples.edge_weights[train_idx],
        pair_ids=triples.pair_ids[train_idx], pair_index=triples.pair_index,
        protein_to_id=triples.protein_to_id, relation_to_id=triples.relation_to_id,
        id_to_protein=triples.id_to_protein, id_to_relation=triples.id_to_relation,
        source_path=triples.source_path, source_sha256=triples.source_sha256,
        input_rows=triples.input_rows, unique_rows=triples.unique_rows,
        pair_semantics=triples.pair_semantics,
    )

    log.info("Fold %d: Proteins=%d Relations=%d Train=%d Val=%d Test=%d",
             fold, triples.num_proteins, triples.num_relations,
             train_subset.num_triples, n_val, test_triples.num_triples)

    # 检查是否已完成
    output_root = params.get("output_root", "08_results/sitne_walk_paper_v2/r2_main_performance/baselines")
    output_dir = ROOT / output_root / "distmult" / f"fold_{fold}" / f"seed_{SEED}"
    if (output_dir / "test_metrics.json").exists():
        log.info("Fold %d: 已完成 (test_metrics.json 存在)，跳过", fold)
        with open(output_dir / "test_metrics.json") as f:
            cached = json.load(f)
        cached_mrr = cached.get("metrics", {}).get("mrr", 0.0)
        log.info("Fold %d: cached MRR=%.4f", fold, cached_mrr)
        return {"fold": fold, "status": "cached", "test_mrr": cached_mrr}

    # 构建模型
    model = DistMultModel(triples.num_proteins, triples.num_relations, EMBEDDING_DIM).to(device)
    log.info("Fold %d: DistMult params: %d", fold, sum(p.numel() for p in model.parameters()))

    # 训练
    result = train_distmult(
        model, train_subset, val_triples, test_triples, known_positive, device,
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,
        weight_decay=1e-5, negative_samples=16,
        early_stopping_patience=5, output_dir=output_dir, seed=SEED, fold=fold,
    )

    log.info("Fold %d complete: MRR=%.4f Hits@1=%.4f Hits@3=%.4f",
             fold, result["test_mrr"], result["test_hits1"], result["test_hits3"])
    return result


def main():
    parser = argparse.ArgumentParser(description="DistMult baseline")
    parser.add_argument("--all-folds", action="store_true", help="对所有 5 个 fold 运行")
    parser.add_argument("--fold", type=int, default=None, help="指定单个 fold (0-4)")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="rerun_submission_audit/06_comparators",
                        help="baselines 输出根目录")
    args = parser.parse_args()
    guard_historical_output(ROOT / args.output_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    params = {
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "output_root": args.output_root,
    }

    # 确定要运行的 folds
    if args.fold is not None:
        folds = [args.fold]
    elif args.all_folds:
        folds = [0, 1, 2, 3, 4]
    else:
        folds = [0]  # 默认仅 fold_0

    log.info("将运行 folds: %s", folds)

    all_results = []
    for fold in folds:
        result = run_single_fold(fold, device, params)
        all_results.append(result)

    # 汇总
    successful = [r for r in all_results if r.get("test_mrr") is not None]
    if successful:
        mrrs = [r["test_mrr"] for r in successful]
        log.info("=" * 50)
        log.info("DistMult Summary: %d folds, Mean MRR=%.4f ± %.4f",
                 len(successful), np.mean(mrrs), np.std(mrrs))
        log.info("=" * 50)

    # 保存汇总文件
    if len(folds) > 1:
        summary_dir = ROOT / args.output_root / "distmult"
        summary_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_results)
        df.to_csv(summary_dir / "distmult_5fold_summary.tsv", sep="\t", index=False)
        log.info("5-fold 汇总已保存: %s", summary_dir / "distmult_5fold_summary.tsv")


if __name__ == "__main__":
    main()
