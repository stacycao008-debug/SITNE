#!/usr/bin/env python3
"""metapath2vec-style baseline for SITNE-Walk paper.

策略:
  1. 在训练图上用 AliasTypedWalker 生成 typed random walks (记录 protein + relation)
  2. 从 walks 中提取 (center, context, relation) triplets (skip-gram window)
  3. 用负采样 skip-gram 训练 protein embedding
  4. 将 protein embedding 冻结，再用 DistMult-style decoder 学习 relation embedding
  5. 用 filtered ranking evaluator 评估，5折汇总

Usage:
    python scripts/run_baseline_metapath2vec.py              # fold_0 only
    python scripts/run_baseline_metapath2vec.py --all-folds   # all 5 folds
    python scripts/run_baseline_metapath2vec.py --fold 3     # specific fold
"""

import argparse
import json, logging, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("metapath2vec")

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
from sitne_walk.graph import build_packed_csr_graph, PackedCSRGraph
from sitne_walk.walks import AliasTypedWalker, make_generator
from sitne_walk.evaluator import evaluate_filtered_type_ranking
from sitne_walk.provenance import guard_historical_output


# ---------------------------------------------------------------------------
# Skip-Gram Model
# ---------------------------------------------------------------------------

class SkipGramModel(nn.Module):
    """负采样 skip-gram: 给定 center embedding 预测 context embedding。"""

    def __init__(self, num_entities: int, embedding_dim: int):
        super().__init__()
        self.center_embedding = nn.Embedding(num_entities, embedding_dim)
        self.context_embedding = nn.Embedding(num_entities, embedding_dim)
        self.embedding_dim = embedding_dim
        self._reset_parameters()

    def _reset_parameters(self):
        bound = 1.0 / self.embedding_dim ** 0.5
        nn.init.uniform_(self.center_embedding.weight, -bound, bound)
        nn.init.uniform_(self.context_embedding.weight, -bound, bound)

    def forward_logits(self, centers, contexts):
        """Dot product between center and context embeddings."""
        return (self.center_embedding(centers) * self.context_embedding(contexts)).sum(dim=1)


# ---------------------------------------------------------------------------
# Walk generation
# ---------------------------------------------------------------------------

def generate_walks(
    graph: PackedCSRGraph,
    num_proteins: int,
    walks_per_node: int = 10,
    walk_length: int = 5,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对每个蛋白生成 walks，返回 (proteins, relations, valid_steps)。

    Returns tensors on CPU.
    """
    gen = make_generator("cpu", seed)
    walker = AliasTypedWalker(graph)

    all_proteins = []
    all_relations = []
    all_valid = []

    chunk_size = 200  # proteins per chunk
    for start in range(0, num_proteins, chunk_size):
        end = min(start + chunk_size, num_proteins)
        chunk_proteins = list(range(start, end))
        starts_tensor = torch.tensor(chunk_proteins, dtype=torch.long)

        for _ in range(walks_per_node):
            batch = walker.generate(starts_tensor, walk_length, gen)
            all_proteins.append(batch.proteins.cpu())
            all_relations.append(batch.relations.cpu())
            all_valid.append(batch.valid_steps.cpu())

        if (start + chunk_size) % 500 == 0:
            log.info("  walk generation: %d/%d proteins", end, num_proteins)

    proteins = torch.cat(all_proteins, dim=0)       # [N*walks, walk_len+1]
    relations = torch.cat(all_relations, dim=0)     # [N*walks, walk_len]
    valid = torch.cat(all_valid, dim=0)
    return proteins, relations, valid


def build_skipgram_pairs(
    proteins: torch.Tensor,
    relations: torch.Tensor,
    valid: torch.Tensor,
    window_size: int = 3,
    max_pairs: int = 5_000_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 walks 生成 (center, context) pairs。

    Sliding window: for each walk step, pair center protein with context
    proteins within ±window_size steps.
    """
    centers_list = []
    contexts_list = []

    num_walks, walk_len_plus_1 = proteins.shape
    walk_len = walk_len_plus_1 - 1

    for w in range(num_walks):
        for i in range(walk_len + 1):
            center = proteins[w, i].item()
            if center < 0:
                continue
            for j in range(max(0, i - window_size), min(walk_len + 1, i + window_size + 1)):
                if i == j:
                    continue
                context = proteins[w, j].item()
                if context < 0:
                    continue
                centers_list.append(center)
                contexts_list.append(context)

        if len(centers_list) >= max_pairs:
            break

    centers = torch.tensor(centers_list[:max_pairs], dtype=torch.long)
    contexts = torch.tensor(contexts_list[:max_pairs], dtype=torch.long)
    return centers, contexts


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_skipgram(
    centers: torch.Tensor,
    contexts: torch.Tensor,
    num_entities: int,
    device: torch.device,
    *,
    embedding_dim: int = 128,
    epochs: int = 5,
    batch_size: int = 4096,
    negative_samples: int = 5,
    learning_rate: float = 0.01,
) -> SkipGramModel:
    """Train skip-gram with negative sampling on walk pairs."""
    model = SkipGramModel(num_entities, embedding_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    num_pairs = len(centers)
    neg_dist = torch.ones(num_entities, device=device) / num_entities  # uniform negative sampling

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        perm = torch.randperm(num_pairs)

        for start in range(0, num_pairs, batch_size):
            idx = perm[start:start + batch_size]
            c = centers[idx].to(device)
            ctx = contexts[idx].to(device)
            bs = len(c)

            # Positive scores
            pos = model.forward_logits(c, ctx)
            pos_loss = -torch.nn.functional.logsigmoid(pos).mean()

            # Negative samples
            neg_ctxt = torch.randint(0, num_entities, (bs * negative_samples,), device=device)
            c_exp = c.unsqueeze(1).expand(-1, negative_samples).reshape(-1)
            neg = model.forward_logits(c_exp, neg_ctxt)
            neg_loss = -torch.nn.functional.logsigmoid(-neg).mean()

            loss = pos_loss + neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        log.info("  skip-gram epoch %d/%d loss=%.4f", epoch + 1, epochs, avg_loss)

    return model


class Metapath2VecScorer(nn.Module):
    """Uses pre-trained skip-gram center embeddings + learnable relation
    embeddings for DistMult-style relation scoring."""

    def __init__(self, skipgram: SkipGramModel, num_relations: int):
        super().__init__()
        # Freeze pre-trained protein embeddings
        self.entity_embedding = skipgram.center_embedding
        for p in self.entity_embedding.parameters():
            p.requires_grad = False
        self.relation_embedding = nn.Embedding(num_relations, skipgram.embedding_dim)
        self.embedding_dim = skipgram.embedding_dim
        self.scale = self.embedding_dim ** 0.5
        nn.init.uniform_(self.relation_embedding.weight,
                         -1.0 / self.embedding_dim ** 0.5,
                         1.0 / self.embedding_dim ** 0.5)

    def score_all_relations(self, head_ids, tail_ids):
        h = self.entity_embedding(head_ids)
        t = self.entity_embedding(tail_ids)
        pair = h * t
        return pair @ self.relation_embedding.weight.T / self.scale

    def score_triples(self, head_ids, relation_ids, tail_ids):
        all_scores = self.score_all_relations(head_ids, tail_ids)
        return all_scores.gather(1, relation_ids.reshape(-1, 1)).squeeze(1)


def train_relation_scorer(
    scorer: Metapath2VecScorer,
    triples: TrainingTriples,
    val_triples: IndexedTriples,
    test_triples: IndexedTriples,
    known_positive: PairRelationIndex,
    device: torch.device,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.01,
    negative_samples: int = 16,
    early_stopping_patience: int = 5,
    output_dir: Path,
    seed: int = 42,
    fold: int = 0,
) -> dict:
    """Fine-tune relation embeddings on top of frozen protein embeddings."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    heads = triples.heads
    relations = triples.relations
    tails = triples.tails
    num_entities = triples.num_proteins

    head_counts = torch.bincount(heads, minlength=num_entities).float()
    tail_counts = torch.bincount(tails, minlength=num_entities).float()
    total_counts = head_counts + tail_counts
    total_counts.clamp_min_(1)
    neg_dist = (total_counts ** 0.75)
    neg_dist = neg_dist / neg_dist.sum()

    optimizer = optim.AdamW(
        [p for p in scorer.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mrr = -1.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        scorer.train()
        total_loss = 0.0
        num_batches = 0

        perm = torch.randperm(len(heads))
        for batch_idx in iter_batches(len(heads), batch_size, perm):
            idx = batch_idx
            h = heads[idx].to(device)
            r = relations[idx].to(device)
            t = tails[idx].to(device)
            bs = len(h)

            pos_scores = scorer.score_triples(h, r, t)

            neg_tails = torch.multinomial(neg_dist, bs * negative_samples, replacement=True).to(device)
            h_expanded = h.unsqueeze(1).expand(-1, negative_samples).reshape(-1)
            r_expanded = r.unsqueeze(1).expand(-1, negative_samples).reshape(-1)
            neg_scores = scorer.score_triples(h_expanded, r_expanded, neg_tails)
            neg_scores = neg_scores.view(bs, negative_samples)

            pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, negative_samples)
            margin_loss = torch.clamp(1.0 + neg_scores - pos_scores_expanded, min=0).mean()

            reg = scorer.relation_embedding.weight.norm(p=2) * 0.0001
            loss = margin_loss + reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if val_triples.supported_rows > 0:
            val_result = evaluate_filtered_type_ranking(
                scorer, val_triples, known_positive, device,
                batch_size=128, tie_policy="average", require_full_coverage=False,
            )
            val_mrr = val_result.metrics.get("mrr", 0.0)
        else:
            val_mrr = 0.0

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in scorer.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        log.info("  scorer epoch %3d  loss=%.4f  val_mrr=%.4f  best=%.4f@%d",
                 epoch, avg_loss, val_mrr, best_val_mrr, best_epoch)

        if epochs_no_improve >= early_stopping_patience:
            log.info("  Early stopping at epoch %d", epoch)
            break

    if best_state is not None:
        scorer.load_state_dict(best_state)

    scorer.eval()
    test_result = evaluate_filtered_type_ranking(
        scorer, test_triples, known_positive, device,
        batch_size=128, tie_policy="average", require_full_coverage=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(test_result.metrics)

    result = {
        "model": "metapath2vec",
        "fold": fold,
        "seed": seed,
        "epochs_trained": best_epoch + 1 if best_epoch >= 0 else 0,
        "best_val_mrr": best_val_mrr,
        "test_mrr": metrics.get("mrr", 0.0),
        "test_hits1": metrics.get("hits@1", 0.0),
        "test_hits3": metrics.get("hits@3", 0.0),
        "test_hits10": metrics.get("hits@10", 0.0),
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "input_rows": test_result.input_rows,
                    "supported_rows": test_result.supported_rows}, f, indent=2)

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
    ranks_df.to_csv(output_dir / "test_ranks.tsv", sep="\t", index=False)

    log.info("metapath2vec fold %d seed %d: test MRR=%.4f Hits@1=%.3f",
             fold, seed, result["test_mrr"], result["test_hits1"])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_fold(fold: int, device: torch.device, params: dict) -> dict:
    EMB_DIM = params["embedding_dim"]
    WALKS_PER = params["walks_per_node"]
    WALK_LEN = params["walk_length"]
    SEED = params["seed"]

    train_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
    test_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/test.tsv"
    val_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/val.tsv"
    known_pos_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/known_positives.tsv"

    if not train_path.exists() or not test_path.exists():
        log.warning("fold_%d 数据缺失，跳过", fold)
        return {"fold": fold, "status": "data_missing"}

    # Cache check
    output_root = params.get("output_root", "08_results/sitne_walk_paper_v2/r2_main_performance/baselines")
    output_dir = ROOT / output_root / "metapath2vec" / f"fold_{fold}" / f"seed_{SEED}"
    if (output_dir / "test_metrics.json").exists():
        log.info("Fold %d: 已完成 (test_metrics.json 存在)，跳过", fold)
        with open(output_dir / "test_metrics.json") as f:
            cached = json.load(f)
        cached_mrr = cached.get("metrics", {}).get("mrr", 0.0)
        log.info("Fold %d: cached MRR=%.4f", fold, cached_mrr)
        return {"fold": fold, "status": "cached", "test_mrr": cached_mrr}

    # --- Load data ---
    log.info("=== Fold %d: 加载数据 ===", fold)
    triples = load_training_triples(
        str(train_path), weight_column=None, duplicate_policy="binary", pair_semantics="unordered")

    test_triples = load_indexed_triples(
        str(test_path), triples.protein_to_id, triples.relation_to_id, pair_semantics="unordered")
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
            val_triples_raw, test_triples,
        ])
    known_positive = build_known_positive_index(
        known_pos, triples.num_proteins, triples.num_relations)

    # Validation split (10%)
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

    # --- Phase 1: Build graph & generate walks ---
    log.info("Fold %d: 构建图 (α=0, β=0, no correction)...", fold)
    t0 = time.time()
    from sitne_walk.data import load_relation_modes
    rmodes = load_relation_modes(None, triples.id_to_relation, default_mode="symmetric")
    graph = build_packed_csr_graph(
        train_subset, relation_modes=rmodes, alpha=0.0, beta=0.0,
        epsilon=1e-12, drop_self_loops=True)

    log.info("Fold %d: 生成 walks (walks_per_node=%d, walk_length=%d)...",
             fold, WALKS_PER, WALK_LEN)
    proteins_w, relations_w, valid_w = generate_walks(
        graph, triples.num_proteins,
        walks_per_node=WALKS_PER, walk_length=WALK_LEN, seed=SEED + fold)

    n_walks = proteins_w.shape[0]
    n_steps = int(valid_w.sum())
    log.info("Fold %d: %d walks, %d valid steps (%.1fs)",
             fold, n_walks, n_steps, time.time() - t0)

    # --- Phase 2: Extract skip-gram pairs & train ---
    log.info("Fold %d: 提取 skip-gram pairs...", fold)
    centers, contexts = build_skipgram_pairs(
        proteins_w, relations_w, valid_w, window_size=3, max_pairs=5_000_000)
    log.info("Fold %d: %d pairs (%.1fs)", fold, len(centers), time.time() - t0)

    log.info("Fold %d: 训练 skip-gram...", fold)
    sg_model = train_skipgram(
        centers, contexts, triples.num_proteins, device,
        embedding_dim=EMB_DIM, epochs=5, batch_size=4096,
        negative_samples=5, learning_rate=0.01)

    # --- Phase 3: Train relation scorer ---
    log.info("Fold %d: 训练 relation scorer...", fold)
    scorer = Metapath2VecScorer(sg_model, triples.num_relations).to(device)
    log.info("Fold %d: scorer params: %d (frozen entity + trainable relation)",
             fold, sum(p.numel() for p in scorer.parameters()))

    result = train_relation_scorer(
        scorer, train_subset, val_triples, test_triples, known_positive, device,
        epochs=30, batch_size=128, learning_rate=0.01, negative_samples=16,
        early_stopping_patience=5, output_dir=output_dir, seed=SEED, fold=fold,
    )

    elapsed = time.time() - t0
    log.info("Fold %d complete: MRR=%.4f Hits@1=%.4f Hits@3=%.4f (%.1fs total)",
             fold, result["test_mrr"], result["test_hits1"], result["test_hits3"], elapsed)
    return result


def main():
    parser = argparse.ArgumentParser(description="metapath2vec baseline")
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--walks-per-node", type=int, default=10)
    parser.add_argument("--walk-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="rerun_submission_audit/06_comparators",
                        help="baselines 输出根目录")
    args = parser.parse_args()
    guard_historical_output(ROOT / args.output_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    params = {
        "embedding_dim": args.embedding_dim,
        "walks_per_node": args.walks_per_node,
        "walk_length": args.walk_length,
        "seed": args.seed,
        "output_root": args.output_root,
    }

    if args.fold is not None:
        folds = [args.fold]
    elif args.all_folds:
        folds = [0, 1, 2, 3, 4]
    else:
        folds = [0]

    log.info("将运行 folds: %s", folds)

    all_results = []
    for fold in folds:
        result = run_single_fold(fold, device, params)
        all_results.append(result)

    successful = [r for r in all_results if r.get("test_mrr") is not None]
    if successful:
        mrrs = [r["test_mrr"] for r in successful]
        log.info("=" * 50)
        log.info("metapath2vec Summary: %d folds, Mean MRR=%.4f ± %.4f",
                 len(successful), np.mean(mrrs), np.std(mrrs))
        log.info("=" * 50)

    if len(folds) > 1:
        summary_dir = ROOT / args.output_root / "metapath2vec"
        summary_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_results)
        df.to_csv(summary_dir / "metapath2vec_5fold_summary.tsv", sep="\t", index=False)
        log.info("5-fold 汇总已保存: %s", summary_dir / "metapath2vec_5fold_summary.tsv")


if __name__ == "__main__":
    main()
