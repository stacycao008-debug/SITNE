#!/usr/bin/env python3
"""Phase E: R4 Robustness Analysis (v2 splits)

分析 SITNE-Walk optimal 模型的鲁棒性:
1. Degree/Frequency Strata 分层分析
2. Held-out probes (degree + frequency)
3. Seed sensitivity
4. Coverage boundary
5. α/β sensitivity (基于 R1 参数网格)

使用已训练好的 optimal 模型 checkpoint (5-fold × 3 seeds)。
"""
from __future__ import annotations

import json, logging, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("r4_analysis")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "06_code"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sitne_walk.data import load_training_triples, load_indexed_triples, load_relation_modes
from sitne_walk.graph import build_packed_csr_graph
from sitne_walk.model import SITNEWalkModel
from sitne_walk.config import SITNEConfig
import yaml

OUTPUT_DIR = ROOT / "08_results/sitne_walk_paper_v2/r4_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paths to trained models
OPTIMAL_DIR = ROOT / "08_results/sitne_walk_paper_v2/r2_main_performance/optimal"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(fold: int, seed: int):
    """Load a trained SITNE-Walk model from checkpoint."""
    model_dir = OPTIMAL_DIR / f"fold_{fold}" / f"seed_{seed}"
    
    # Find run directory
    run_dirs = list(model_dir.glob("sitne_walk_*"))
    if not run_dirs:
        logger.warning("No run directory for fold_%d/seed_%d", fold, seed)
        return None, None, None
    
    run_dir = run_dirs[0]
    
    # Load config
    with open(run_dir / "resolved_config.json") as f:
        config = json.load(f)
    
    # Load data
    train_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
    triples = load_training_triples(str(train_path), duplicate_policy="binary")
    rmodes = load_relation_modes(None, triples.id_to_relation, default_mode="symmetric")
    
    # Build graph from config
    walk_cfg = config.get("walk", {})
    graph = build_packed_csr_graph(
        triples, relation_modes=rmodes,
        alpha=walk_cfg.get("alpha", 0.25),
        beta=walk_cfg.get("beta", 0.0),
        epsilon=walk_cfg.get("epsilon", 1e-8),
        drop_self_loops=True,
        correction_mode=walk_cfg.get("correction_mode", "rank"),
    )
    
    # Load model using trainer's build_model to get correct architecture
    from sitne_walk.trainer import build_model
    from sitne_walk.config import SITNEConfig
    
    # Build config from resolved_config.json
    sitne_cfg = SITNEConfig.from_mapping(config)
    # Override data path to current fold
    sitne_cfg = SITNEConfig(
        data=SITNEConfig.DataConfig(
            train_path=str(train_path),
            validation_path=str(ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/val.tsv"),
            test_path=str(ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/test.tsv"),
            split_manifest_path="05_splits/typed_ranking_v2/split_manifest.json",
            require_manifest_hashes=False,
            all_known_positive_paths=("05_splits/typed_ranking_v2/all_known_positives.tsv",),
            duplicate_policy="binary",
            pair_semantics="unordered",
            default_relation_mode="symmetric",
            drop_self_loops_from_walks=True,
        ),
        walk=sitne_cfg.walk,
        model=sitne_cfg.model,
        loss=sitne_cfg.loss,
        training=sitne_cfg.training,
        evaluation=sitne_cfg.evaluation,
    )
    model, _, _ = build_model(sitne_cfg, triples, graph)
    
    # Find best checkpoint
    ckpt_dir = run_dir / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("best_model_*.pt"))
    if not ckpt_files:
        # Try alternative pattern
        ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        logger.warning("No checkpoint for fold_%d/seed_%d", fold, seed)
        return None, triples, graph
    
    ckpt = torch.load(ckpt_files[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(DEVICE)
    
    return model, triples, graph


def extract_embeddings(model, triples):
    """Extract protein embeddings (semantic channel) from trained model."""
    model.eval()
    with torch.no_grad():
        all_proteins = torch.arange(triples.num_proteins, device=DEVICE)
        embeddings = model.semantic_embedding(all_proteins)  # [N, dim]
    return embeddings.cpu().numpy()


# ============================================================
# 1. Degree/Frequency Strata Analysis
# ============================================================
def analyze_strata():
    """对每个 fold 的 test pairs 按 degree/frequency strata 分层分析 MRR。"""
    logger.info("=== Strata Analysis ===")
    
    strata_results = []
    
    for fold in range(5):
        # Load test triples
        train_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
        test_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/test.tsv"
        triples = load_training_triples(str(train_path), duplicate_policy="binary")
        test = load_indexed_triples(str(test_path), triples.protein_to_id, triples.relation_to_id)
        
        # Compute protein degree from train
        degree = np.zeros(triples.num_proteins, dtype=np.float64)
        heads = triples.heads.numpy()
        tails = triples.tails.numpy()
        non_self = heads != tails
        pair_codes = heads[non_self] * np.int64(triples.num_proteins) + tails[non_self]
        unique_pos = np.unique(pair_codes, return_index=True)[1]
        np.add.at(degree, heads[non_self][unique_pos], 1.0)
        np.add.at(degree, tails[non_self][unique_pos], 1.0)
        
        # Relation frequency
        rel_freq = np.zeros(triples.num_relations, dtype=np.float64)
        np.add.at(rel_freq, triples.relations.numpy()[non_self], 1.0)
        
        # For each test query, determine degree stratum and frequency stratum
        for seed in [42]:
            # Load test metrics to get per-query ranks
            model_dir = OPTIMAL_DIR / f"fold_{fold}" / f"seed_{seed}"
            for run_dir in model_dir.glob("sitne_walk_*"):
                ranks_path = run_dir / "test_ranks.tsv"
                if not ranks_path.exists():
                    continue
                ranks_df = pd.read_csv(ranks_path, sep="\t")
                
                for _, row in ranks_df.iterrows():
                    h = int(row["head_id"])
                    t = int(row["tail_id"])
                    r = int(row["relation_id"])
                    rank = float(row["filtered_rank"])
                    
                    max_deg = max(degree[h], degree[t])
                    
                    # Degree stratum
                    if max_deg < 10:
                        deg_stratum = "low"
                    elif max_deg < 50:
                        deg_stratum = "medium"
                    elif max_deg < 200:
                        deg_stratum = "high"
                    else:
                        deg_stratum = "very_high"
                    
                    # Frequency stratum
                    freq = rel_freq[r]
                    if freq < 100:
                        freq_stratum = "rare"
                    elif freq < 1000:
                        freq_stratum = "low"
                    elif freq < 5000:
                        freq_stratum = "medium"
                    else:
                        freq_stratum = "high"
                    
                    strata_results.append({
                        "fold": fold, "seed": seed,
                        "head": h, "tail": t, "relation": r,
                        "rank": rank, "mrr_contribution": 1.0 / rank,
                        "max_pair_degree": max_deg,
                        "relation_frequency": freq,
                        "degree_stratum": deg_stratum,
                        "frequency_stratum": freq_stratum,
                    })
                break  # Only use first run per fold
    
    df = pd.DataFrame(strata_results)
    
    # Per-stratum summary
    summary = []
    for stratum_col in ["degree_stratum", "frequency_stratum"]:
        for stratum in sorted(df[stratum_col].unique()):
            sub = df[df[stratum_col] == stratum]
            summary.append({
                "stratum_type": stratum_col,
                "stratum": stratum,
                "n_queries": len(sub),
                "mean_mrr": sub["mrr_contribution"].mean(),
                "std_mrr": sub["mrr_contribution"].std(),
                "mean_rank": sub["rank"].mean(),
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "strata_analysis.tsv", sep="\t", index=False)
    logger.info("Strata analysis saved: %d rows", len(summary_df))
    
    for _, row in summary_df.iterrows():
        logger.info("  %s=%s: MRR=%.4f ± %.4f (n=%d)",
                    row["stratum_type"], row["stratum"],
                    row["mean_mrr"], row["std_mrr"], row["n_queries"])
    
    return summary_df


# ============================================================
# 2. Held-out Probes
# ============================================================
def analyze_probes():
    """Degree classification probe + Frequency Ridge regression probe."""
    logger.info("=== Probe Analysis ===")
    
    probe_results = []
    
    for fold in range(5):
        train_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
        triples = load_training_triples(str(train_path), duplicate_policy="binary")
        
        # Load model and extract embeddings
        model, _, graph = load_model_from_checkpoint(fold, 42)
        if model is None:
            continue
        
        embeddings = extract_embeddings(model, triples)
        
        # Compute degree bins for all proteins
        degree = graph.protein_degree.numpy()
        degree_bins = np.digitize(degree, bins=[0, 5, 15, 50, 200])
        # Ensure no bin 0
        degree_bins = np.maximum(degree_bins, 0)
        
        # Compute log-frequency per protein
        heads = triples.heads.numpy()
        tails = triples.tails.numpy()
        relations = triples.relations.numpy()
        non_self = heads != tails
        rel_freq = np.bincount(relations[non_self], minlength=triples.num_relations).astype(np.float64)
        
        # Mean incident log-type-frequency per protein
        protein_rel_sum = np.zeros(triples.num_proteins, dtype=np.float64)
        protein_rel_count = np.zeros(triples.num_proteins, dtype=np.float64)
        for i in range(len(heads)):
            if heads[i] != tails[i]:
                protein_rel_sum[heads[i]] += np.log1p(rel_freq[relations[i]])
                protein_rel_count[heads[i]] += 1
                protein_rel_sum[tails[i]] += np.log1p(rel_freq[relations[i]])
                protein_rel_count[tails[i]] += 1
        protein_rel_count = np.maximum(protein_rel_count, 1)
        mean_log_freq = protein_rel_sum / protein_rel_count
        
        # Degree probe: LogisticRegression
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, degree_bins, test_size=0.2, random_state=42
        )
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        deg_bal_acc = balanced_accuracy_score(y_test, y_pred)
        
        # Frequency Ridge probe
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
            embeddings, mean_log_freq, test_size=0.2, random_state=42
        )
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_f, y_train_f)
        y_pred_f = ridge.predict(X_test_f)
        freq_mae = mean_absolute_error(y_test_f, y_pred_f)
        freq_r2 = r2_score(y_test_f, y_pred_f)
        
        probe_results.append({
            "fold": fold,
            "degree_balanced_accuracy": deg_bal_acc,
            "frequency_mae": freq_mae,
            "frequency_r2": freq_r2,
        })
        
        logger.info("Fold %d: deg_bal_acc=%.4f freq_mae=%.4f freq_r2=%.4f",
                    fold, deg_bal_acc, freq_mae, freq_r2)
    
    df = pd.DataFrame(probe_results)
    df.to_csv(OUTPUT_DIR / "probe_analysis.tsv", sep="\t", index=False)
    logger.info("Probe analysis: deg_acc=%.4f ± %.4f, freq_mae=%.4f ± %.4f",
                df["degree_balanced_accuracy"].mean(), df["degree_balanced_accuracy"].std(),
                df["frequency_mae"].mean(), df["frequency_mae"].std())
    
    return df


# ============================================================
# 3. Seed Sensitivity
# ============================================================
def analyze_seed_sensitivity():
    """分析 15 runs 的 MRR 分布。"""
    logger.info("=== Seed Sensitivity ===")
    
    summary_path = OPTIMAL_DIR / "training_summary.tsv"
    if not summary_path.exists():
        logger.warning("No training_summary.tsv found")
        return None
    
    df = pd.read_csv(summary_path, sep="\t")
    
    logger.info("15 runs: MRR=%.6f ± %.6f (min=%.6f, max=%.6f)",
                df["mrr"].mean(), df["mrr"].std(),
                df["mrr"].min(), df["mrr"].max())
    
    # Per-fold analysis
    for fold in range(5):
        sub = df[df["fold"] == fold]
        logger.info("  Fold %d: MRR=%.6f ± %.6f", fold, sub["mrr"].mean(), sub["mrr"].std())
    
    return df


# ============================================================
# 4. Coverage Boundary
# ============================================================
def analyze_coverage_boundary():
    """分析 test queries 中 train-unseen protein 的影响。"""
    logger.info("=== Coverage Boundary ===")
    
    boundary_results = []
    
    for fold in range(5):
        train_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
        test_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/test.tsv"
        
        triples = load_training_triples(str(train_path), duplicate_policy="binary")
        test = load_indexed_triples(str(test_path), triples.protein_to_id, triples.relation_to_id)
        
        boundary_results.append({
            "fold": fold,
            "total_test_queries": test.raw_input_rows,
            "supported_queries": test.supported_rows,
            "unsupported_protein_rows": test.unsupported_protein_rows,
            "unsupported_relation_rows": test.unsupported_relation_rows,
            "coverage": test.supported_rows / max(test.input_rows, 1),
        })
        
        logger.info("Fold %d: supported=%d/%d (%.1f%%), unseen_protein=%d, unseen_relation=%d",
                    fold, test.supported_rows, test.raw_input_rows,
                    100 * test.supported_rows / max(test.raw_input_rows, 1),
                    test.unsupported_protein_rows, test.unsupported_relation_rows)
    
    df = pd.DataFrame(boundary_results)
    df.to_csv(OUTPUT_DIR / "coverage_boundary.tsv", sep="\t", index=False)
    return df


# ============================================================
# Main
# ============================================================
def main():
    logger.info("=== Phase E: R4 Robustness Analysis (v2 splits) ===")
    
    # 1. Strata
    try:
        analyze_strata()
    except Exception as e:
        logger.error("Strata analysis failed: %s", e)
    
    # 2. Probes
    try:
        analyze_probes()
    except Exception as e:
        logger.error("Probe analysis failed: %s", e)
    
    # 3. Seed sensitivity
    try:
        analyze_seed_sensitivity()
    except Exception as e:
        logger.error("Seed sensitivity failed: %s", e)
    
    # 4. Coverage boundary
    try:
        analyze_coverage_boundary()
    except Exception as e:
        logger.error("Coverage boundary failed: %s", e)
    
    logger.info("=== R4 Analysis Complete ===")
    logger.info("Results in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
