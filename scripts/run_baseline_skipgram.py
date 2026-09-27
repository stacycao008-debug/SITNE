#!/usr/bin/env python3
"""Typed Skip-Gram 基线 (v2 splits) — 支持 Optuna 逐 fold 调参

使用与 SITNE-Walk 相同的 typed walk 生成器（uncorrected α=β=0），
但仅使用 single-channel skip-gram（无 semantic channel、无 decorrelation、
无 adversary、无 ranking loss）。

训练采用两阶段（对齐 metapath2vec 基线范式）：
  Phase 1: 负采样 skip-gram 训练 protein embedding（center/context 双套）。
  Phase 2: 冻结 protein embedding，用 DistMult-style decoder 在训练三元组上
           做 margin ranking loss 训练 relation embedding，filtered MRR 早停。

两种模式：
  1. 固定参数（默认，SITNE-Walk 对齐预算）：
     walk_length=81, walks_per_protein=16, context_window=1,
     embedding_dim=512, epochs=50, negative_samples=16, lr=0.01
  2. Optuna 调参（--optuna）：每个 fold 独立搜索超参（用该 fold 的 val 集
     作目标），记录每个 fold 的最佳参数，再用最佳参数 + 多 seed 评估。

Usage:
    python scripts/run_baseline_skipgram.py --all-folds           # 固定参数 5 折
    python scripts/run_baseline_skipgram.py --optuna --fold 0     # fold 0 调参
    python scripts/run_baseline_skipgram.py --optuna --all-folds  # 逐 fold 调参
"""
import argparse
import json, logging, os, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("skipgram_baseline")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "06_code"))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sitne_walk.data import (
    load_training_triples, load_indexed_triples, load_relation_modes,
    build_known_positive_index, merge_indexed_triples, IndexedTriples,
    iter_batches,
)
from sitne_walk.graph import build_packed_csr_graph
from sitne_walk.walks import AliasTypedWalker, make_generator
from sitne_walk.evaluator import evaluate_filtered_type_ranking


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SkipGramModel(nn.Module):
    """负采样 skip-gram: center/context 双套 embedding（标准 SGNS）。"""

    def __init__(self, num_proteins: int, embedding_dim: int = 512):
        super().__init__()
        self.center_embedding = nn.Embedding(num_proteins, embedding_dim)
        self.context_embedding = nn.Embedding(num_proteins, embedding_dim)
        self.embedding_dim = embedding_dim
        self._reset_parameters()

    def _reset_parameters(self):
        bound = 1.0 / self.embedding_dim ** 0.5
        nn.init.uniform_(self.center_embedding.weight, -bound, bound)
        nn.init.uniform_(self.context_embedding.weight, -bound, bound)

    def forward_logits(self, centers, contexts):
        return (self.center_embedding(centers) * self.context_embedding(contexts)).sum(dim=1)


class TypedSkipGramScorer(nn.Module):
    """冻结 skip-gram center embedding + 可训练 relation embedding 的 DistMult 打分器。"""

    def __init__(self, skipgram: SkipGramModel, num_relations: int):
        super().__init__()
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


# ---------------------------------------------------------------------------
# Walk generation & pair extraction
# ---------------------------------------------------------------------------

def generate_walk_corpus(graph, walker, generator, walks_per_protein, walk_length, device,
                         max_walks=300000):
    """生成 typed walk corpus (batch)，带 max_walks 上限防极端参数爆炸。"""
    all_walks = []
    batch_size = 256
    num_proteins = graph.num_proteins

    for start in range(0, num_proteins, batch_size):
        end = min(start + batch_size, num_proteins)
        batch_nodes = torch.arange(start, end, dtype=torch.int64, device=device)

        for _ in range(walks_per_protein):
            walk_batch = walker.generate(batch_nodes, walk_length=walk_length, generator=generator)
            proteins = walk_batch.proteins.cpu()
            relations = walk_batch.relations.cpu()
            valid = walk_batch.valid_steps.cpu()

            for i in range(len(batch_nodes)):
                if valid[i].sum() == 0:
                    continue
                steps = int(valid[i].sum().item())
                prot_seq = proteins[i, :steps + 1]
                rel_seq = relations[i, :steps]
                all_walks.append((prot_seq, rel_seq))
                if len(all_walks) >= max_walks:
                    return all_walks

    return all_walks


def extract_pairs(walks, context_window):
    """从 walks 提取 (center, context) pairs。"""
    all_centers = []
    all_contexts = []
    for prot_seq, _ in walks:
        L = len(prot_seq)
        for i, center in enumerate(prot_seq):
            if center < 0:
                continue
            center = int(center)
            for j in range(max(0, i - context_window), min(L, i + context_window + 1)):
                if i == j:
                    continue
                ctx = int(prot_seq[j])
                if ctx < 0:
                    continue
                all_centers.append(center)
                all_contexts.append(ctx)
    return (torch.tensor(all_centers, dtype=torch.int64),
            torch.tensor(all_contexts, dtype=torch.int64))


# ---------------------------------------------------------------------------
# Phase 1: skip-gram training
# ---------------------------------------------------------------------------

def train_skipgram(model, walks, num_proteins, device, epochs=50, negative_samples=16,
                   context_window=1, lr=0.01, weight_decay=1e-5, batch_size=4096,
                   early_stopping_patience=5):
    """使用负采样 skip-gram 训练 protein embedding（center/context 双套）。

    用无监督 SGNS loss 做早停。
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Unigram noise distribution (based on walk frequency)
    protein_counts = torch.zeros(num_proteins, dtype=torch.float32)
    for prot_seq, _ in walks:
        for p in prot_seq:
            if p >= 0:
                protein_counts[int(p)] += 1

    protein_counts.clamp_min_(1)
    noise_weights = protein_counts ** 0.75
    noise_dist = noise_weights / noise_weights.sum()

    all_centers, all_contexts = extract_pairs(walks, context_window)
    total_pairs = len(all_centers)
    if total_pairs == 0:
        raise RuntimeError("skip-gram pairs 为空，检查 walk 生成")

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        indices = torch.randperm(total_pairs)

        for start in range(0, total_pairs, batch_size):
            batch_indices = indices[start:start + batch_size]

            centers = all_centers[batch_indices].to(device)
            contexts = all_contexts[batch_indices].to(device)
            n = len(centers)

            pos_scores = model.forward_logits(centers, contexts)

            neg_contexts = torch.multinomial(noise_dist, n * negative_samples, replacement=True).to(device)
            c_exp = centers.unsqueeze(1).expand(-1, negative_samples).reshape(-1)
            neg_scores = model.forward_logits(c_exp, neg_contexts).view(n, negative_samples)

            pos_loss = -torch.nn.functional.logsigmoid(pos_scores).mean()
            neg_loss = -torch.nn.functional.logsigmoid(-neg_scores).mean()
            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_loss


# ---------------------------------------------------------------------------
# Phase 2: relation scorer training
# ---------------------------------------------------------------------------

def train_relation_scorer(scorer, triples, val_triples, known_index, device,
                          epochs=50, batch_size=128, lr=0.01, negative_samples=16,
                          early_stopping_patience=5, margin=1.0, seed=42):
    """冻结 protein embedding，用 margin ranking loss 训练 relation embedding。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    heads = triples.heads
    relations = triples.relations
    tails = triples.tails
    num_entities = triples.num_proteins

    tail_counts = torch.bincount(tails, minlength=num_entities).float()
    tail_counts.clamp_min_(1)
    neg_dist = tail_counts ** 0.75
    neg_dist = neg_dist / neg_dist.sum()

    optimizer = optim.AdamW(
        [p for p in scorer.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-5)
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
        for idx in iter_batches(len(heads), batch_size, perm):
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
            margin_loss = torch.clamp(margin + neg_scores - pos_scores_expanded, min=0).mean()

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
                scorer, val_triples, known_index, device,
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

        if epochs_no_improve >= early_stopping_patience:
            break

    if best_state is not None:
        scorer.load_state_dict(best_state)
    return best_val_mrr


# ---------------------------------------------------------------------------
# Fold data loading
# ---------------------------------------------------------------------------

def load_fold_data(fold):
    """加载单个 fold 的数据、graph、walker，返回可复用的对象。"""
    train_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
    test_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/test.tsv"
    val_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/val.tsv"
    known_pos_path = ROOT / "05_splits/typed_ranking_v2/all_known_positives.tsv"

    triples = load_training_triples(train_path, duplicate_policy="binary")
    test_triples = load_indexed_triples(test_path, triples.protein_to_id, triples.relation_to_id)
    val_triples = load_indexed_triples(val_path, triples.protein_to_id, triples.relation_to_id)

    if known_pos_path.exists():
        known = load_indexed_triples(known_pos_path, triples.protein_to_id, triples.relation_to_id)
    else:
        train_idx = IndexedTriples(
            heads=triples.heads, relations=triples.relations, tails=triples.tails,
            source_path=str(train_path), raw_input_rows=triples.input_rows,
            input_rows=triples.input_rows, supported_rows=triples.input_rows,
            unsupported_protein_rows=0, unsupported_relation_rows=0,
            pair_semantics=triples.pair_semantics,
        )
        known = merge_indexed_triples([train_idx, val_triples, test_triples])

    known_index = build_known_positive_index(known, triples.num_proteins, triples.num_relations)

    rmodes = load_relation_modes(None, triples.id_to_relation, default_mode="symmetric")
    graph = build_packed_csr_graph(triples, relation_modes=rmodes, alpha=0.0, beta=0.0,
                                   epsilon=1e-8, drop_self_loops=True)

    return {
        "triples": triples,
        "test_triples": test_triples,
        "val_triples": val_triples,
        "known_index": known_index,
        "graph": graph,
    }


def train_and_evaluate(fold, params, seed=42, device=None, walk_cache=None):
    """用给定参数训练两阶段模型，返回 (val_mrr, test_metrics)。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_fold_data(fold)
    triples = data["triples"]
    test_triples = data["test_triples"]
    val_triples = data["val_triples"]
    known_index = data["known_index"]
    graph = data["graph"]

    graph = graph.to(device)
    walker = AliasTypedWalker(graph)
    generator = make_generator(device, seed)

    walk_length = params.get("walk_length", 81)
    walks_per_protein = params.get("walks_per_protein", 16)

    # Walk 缓存（同一 walk 配置复用，避免重复生成）
    walk_key = (fold, walk_length, walks_per_protein)
    if walk_cache is not None and walk_key in walk_cache:
        walks = walk_cache[walk_key]
    else:
        torch.manual_seed(seed)
        walks = generate_walk_corpus(graph, walker, generator,
                                     walks_per_protein=walks_per_protein,
                                     walk_length=walk_length, device=device,
                                     max_walks=params.get("max_walks", 300000))
        if walk_cache is not None:
            walk_cache[walk_key] = walks

    # Phase 1
    embedding_dim = params.get("embedding_dim", 512)
    sg_model = SkipGramModel(triples.num_proteins, embedding_dim=embedding_dim).to(device)
    train_skipgram(
        sg_model, walks, triples.num_proteins, device,
        epochs=params.get("epochs", 50),
        negative_samples=params.get("negative_samples", 16),
        context_window=params.get("context_window", 1),
        lr=params.get("lr", 0.01),
        weight_decay=params.get("weight_decay", 1e-5),
        batch_size=params.get("batch_size", 4096),
        early_stopping_patience=params.get("early_stopping_patience", 5),
    )

    # Phase 2
    scorer = TypedSkipGramScorer(sg_model, triples.num_relations).to(device)
    val_mrr = train_relation_scorer(
        scorer, triples, val_triples, known_index, device,
        epochs=params.get("phase2_epochs", 50),
        batch_size=params.get("phase2_batch_size", 128),
        lr=params.get("phase2_lr", 0.01),
        negative_samples=params.get("phase2_negative_samples", 16),
        early_stopping_patience=params.get("phase2_early_stopping_patience", 5),
        margin=params.get("margin", 1.0),
        seed=seed,
    )

    # Test evaluation
    scorer.eval()
    with torch.no_grad():
        test_result = evaluate_filtered_type_ranking(
            scorer, test_triples, known_index, device,
            batch_size=128, tie_policy="average", require_full_coverage=False,
        )

    return val_mrr, test_result.metrics


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

OPTUNA_SPACE = {
    "walk_length": ("int", 5, 40),
    "walks_per_protein": ("int", 2, 16),
    "context_window": ("int", 1, 3),
    "embedding_dim": ("categorical", [64, 128, 256, 512]),
    "negative_samples": ("int", 5, 20),
    "lr": ("log_float", 1e-4, 5e-2),
    "epochs": ("int", 5, 30),
    "phase2_lr": ("log_float", 1e-4, 5e-2),
    "phase2_negative_samples": ("int", 5, 20),
    "margin": ("float", 0.5, 3.0),
}


def suggest_params(trial, space):
    params = {}
    for name, (kind, *bounds) in space.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, bounds[0], bounds[1])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, bounds[0])
        elif kind == "float":
            params[name] = trial.suggest_float(name, bounds[0], bounds[1])
        elif kind == "log_float":
            params[name] = trial.suggest_float(name, bounds[0], bounds[1], log=True)
    return params


def run_optuna_fold(fold, n_trials=50, seed=42, timeout=None):
    """对单个 fold 做 Optuna 搜索，返回 (best_params, best_val_mrr)。"""
    import optuna

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    walk_cache = {}  # (fold, walk_length, walks_per_protein) -> walks

    def objective(trial):
        params = suggest_params(trial, OPTUNA_SPACE)
        t0 = time.time()
        log.info("[fold%d trial%d] walk=%d walks_per=%d ctx=%d dim=%d neg=%d lr=%.5f epochs=%d p2lr=%.5f margin=%.2f",
                 fold, trial.number, params["walk_length"], params["walks_per_protein"],
                 params["context_window"], params["embedding_dim"], params["negative_samples"],
                 params["lr"], params["epochs"], params["phase2_lr"], params["margin"])
        val_mrr, _ = train_and_evaluate(fold, params, seed=seed, device=device,
                                        walk_cache=walk_cache)
        log.info("[fold%d trial%d] val_mrr=%.4f (%.1fs)", fold, trial.number, val_mrr, time.time() - t0)
        if not np.isfinite(val_mrr):
            raise optuna.TrialPruned(f"NaN val_mrr")
        return val_mrr

    out = ROOT / "08_results/sitne_walk_paper_v2/r2_main_performance/baselines/typed_skipgram/optuna"
    out.mkdir(parents=True, exist_ok=True)
    db = f"sqlite:///{out}/optuna_fold{fold}.db"

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(study_name=f"typed_skipgram_fold{fold}", storage=db,
                                sampler=sampler, direction="maximize", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    log.info("Fold %d Optuna best: MRR=%.4f params=%s", fold, study.best_value, study.best_params)

    (out / f"optuna_results_fold{fold}.json").write_text(json.dumps(dict(
        fold=fold, best_value=study.best_value, best_params=study.best_params,
        n_trials=len(study.trials)), indent=2))
    study.trials_dataframe().to_csv(out / f"optuna_trials_fold{fold}.tsv", sep="\t", index=False)

    return study.best_params, study.best_value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Typed Skip-Gram baseline")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optuna", action="store_true", help="对每个 fold 做 Optuna 调参")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="最佳参数下的多 seed 列表")
    args = parser.parse_args()

    if args.all_folds:
        folds = list(range(5))
    else:
        folds = [args.fold]

    summary_dir = ROOT / "08_results/sitne_walk_paper_v2/r2_main_performance/baselines/typed_skipgram"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if args.optuna:
        all_results = []
        all_best_params = {}
        for fold in folds:
            log.info("=" * 60)
            log.info("Fold %d: Optuna 调参开始 (%d trials)", fold, args.n_trials)
            best_params, best_val = run_optuna_fold(fold, n_trials=args.n_trials, seed=args.seed)
            all_best_params[fold] = best_params
            log.info("Fold %d: 最佳参数 val_mrr=%.4f, params=%s", fold, best_val, best_params)

            # 用最佳参数 + 多 seed 评估
            fold_metrics = {}
            for s in args.seeds:
                _, test_metrics = train_and_evaluate(fold, best_params, seed=s)
                fold_metrics[s] = test_metrics
                log.info("Fold %d seed %d: Test MRR=%.4f", fold, s, test_metrics["mrr"])

            # 保存
            out_dir = summary_dir / f"fold_{fold}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "best_params.json", "w") as f:
                json.dump(best_params, f, indent=2)
            with open(out_dir / "test_metrics_multi_seed.json", "w") as f:
                json.dump({str(s): m for s, m in fold_metrics.items()}, f, indent=2)

            # 用 seed=42 作为主结果（与其它 baseline 一致），同时记录多 seed 均值
            primary = fold_metrics[args.seeds[0]]
            all_results.append({
                "fold": fold,
                "mrr": primary["mrr"],
                "hits@1": primary.get("hits@1", 0),
                "hits@3": primary.get("hits@3", 0),
                "hits@10": primary.get("hits@10", 0),
                "best_val_mrr": best_val,
                "n_seeds": len(args.seeds),
                "mean_mrr_multi_seed": float(np.mean([m["mrr"] for m in fold_metrics.values()])),
                "best_params": json.dumps(best_params),
            })

        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(summary_dir / "typed_skipgram_summary.tsv", sep="\t", index=False)
        log.info("=== Typed Skip-Gram Optuna Summary ===")
        for _, row in df.iterrows():
            log.info("Fold %d: MRR=%.4f (multi-seed mean=%.4f) params=%s",
                     row["fold"], row["mrr"], row["mean_mrr_multi_seed"], row["best_params"])

    else:
        # 固定参数模式（默认 SITNE-Walk 对齐预算）
        default_params = dict(
            walk_length=81, walks_per_protein=16, context_window=1,
            embedding_dim=512, epochs=50, negative_samples=16, lr=0.01,
            phase2_epochs=50, phase2_lr=0.01, phase2_negative_samples=16, margin=1.0,
        )
        results = []
        for fold in folds:
            out_dir = summary_dir / f"fold_{fold}" / f"seed_{args.seed}"
            test_metrics_path = out_dir / "test_metrics.json"
            if test_metrics_path.exists():
                with open(test_metrics_path) as f:
                    metrics = json.load(f)["metrics"]
                log.info("Fold %d: SKIPPED (already done)", fold)
            else:
                _, metrics = train_and_evaluate(fold, default_params, seed=args.seed)
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(test_metrics_path, "w") as f:
                    json.dump({"metrics": metrics, "fold": fold, "seed": args.seed}, f, indent=2)
                log.info("Fold %d: Test MRR=%.4f", fold, metrics["mrr"])
            results.append({"fold": fold, **metrics})

        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(summary_dir / "typed_skipgram_summary.tsv", sep="\t", index=False)
        log.info("=== Typed Skip-Gram Summary ===")
        for _, row in df.iterrows():
            log.info("Fold %d: MRR=%.4f Hits@1=%.4f", row["fold"], row["mrr"], row.get("hits@1", 0))


if __name__ == "__main__":
    main()
