#!/usr/bin/env python3
"""R4 Probes Fix — 直接从 checkpoint 加载模型进行 probe 分析。

由于 resolved_config.json 格式与 SITNEConfig.from_mapping 不完全兼容，
这里直接从 checkpoint 读取 config，然后用 trainer.build_model 构建模型。
"""
from __future__ import annotations

import json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("r4_probes")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "06_code"))

from sitne_walk.data import load_training_triples, load_relation_modes
from sitne_walk.graph import build_packed_csr_graph
from sitne_walk.model import SITNEWalkModel

OUTPUT_DIR = ROOT / "08_results/sitne_walk_paper_v2/r4_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_DIR = ROOT / "08_results/sitne_walk_paper_v2/r3_ablation"
VARIANTS = {
    "full":        ABLATION_DIR / "full",
    "uncorrected": ABLATION_DIR / "uncorrected",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_graph(base_dir: Path, fold: int, seed: int = 42):
    """Load model and graph for a specific fold/seed."""
    model_dir = base_dir / f"fold_{fold}" / f"seed_{seed}"
    run_dirs = list(model_dir.glob("sitne_walk_*"))
    if not run_dirs:
        logger.warning("No run dir for fold_%d/seed_%d", fold, seed)
        return None, None, None
    run_dir = run_dirs[0]

    # Load checkpoint
    ckpt_dir = run_dir / "checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        logger.warning("No checkpoint for fold_%d", fold)
        return None, None, None
    ckpt = torch.load(ckpt_files[-1], map_location="cpu", weights_only=False)
    
    # Load data
    train_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
    triples = load_training_triples(str(train_path), duplicate_policy="binary")
    rmodes = load_relation_modes(None, triples.id_to_relation, default_mode="symmetric")
    
    # Build graph from checkpoint config
    ckpt_cfg = ckpt.get("config", {})
    walk_cfg = ckpt_cfg.get("walk", {})
    graph = build_packed_csr_graph(
        triples, relation_modes=rmodes,
        alpha=walk_cfg.get("alpha", 0.25),
        beta=walk_cfg.get("beta", 0.0),
        epsilon=walk_cfg.get("epsilon", 1e-12),
        drop_self_loops=True,
        correction_mode=walk_cfg.get("correction_mode", "rank"),
    )
    
    # Build model from checkpoint config
    model_cfg = ckpt_cfg.get("model", {})
    # Compute nuisance cardinalities from graph
    degree_labels = torch.zeros_like(graph.protein_degree, dtype=torch.int64)
    positive = graph.protein_degree > 0
    degree_labels[positive] = 1 + torch.floor(torch.log2(graph.protein_degree[positive].float())).to(torch.int64)
    num_degree_classes = int(degree_labels.max().item()) + 1
    
    model = SITNEWalkModel(
        num_proteins=triples.num_proteins,
        num_relations=triples.num_relations,
        embedding_dim=model_cfg.get("embedding_dim", 512),
        nuisance_cardinalities={"degree": num_degree_classes},
        nuisance_hidden_dim=model_cfg.get("nuisance_hidden_dim", 32),
        decoder_dropout=model_cfg.get("decoder_dropout", 0.1),
    )
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(DEVICE)
    
    return model, triples, graph


def extract_semantic_embeddings(model, num_proteins):
    """Extract semantic embeddings for all proteins."""
    model.eval()
    with torch.no_grad():
        all_ids = torch.arange(num_proteins, device=DEVICE)
        embeddings = model.semantic_embedding(all_ids)
    return embeddings.cpu().numpy()


def run_probes():
    logger.info("=== R4 Held-out Probes (full vs uncorrected) ===")
    
    results = []
    
    for variant, base_dir in VARIANTS.items():
        for fold in range(5):
            logger.info("[%s] Fold %d: loading model...", variant, fold)
            model, triples, graph = load_model_and_graph(base_dir, fold, 42)
            if model is None:
                logger.warning("[%s] Skip fold %d (no model)", variant, fold)
                continue
            
            embeddings = extract_semantic_embeddings(model, triples.num_proteins)
            degree = graph.protein_degree.numpy()
            
            # Degree probe: predict degree bin
            # Create bins: 0, [1,5), [5,15), [15,50), [50,∞)
            degree_bins = np.zeros(len(degree), dtype=np.int32)
            degree_bins[(degree >= 1) & (degree < 5)] = 1
            degree_bins[(degree >= 5) & (degree < 15)] = 2
            degree_bins[(degree >= 15) & (degree < 50)] = 3
            degree_bins[degree >= 50] = 4
            
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, degree_bins, test_size=0.2, random_state=42 + fold
            )
            
            clf = LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            deg_bal_acc = balanced_accuracy_score(y_test, y_pred)
            deg_macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            
            # Frequency probe: predict mean incident log-type-frequency
            heads = triples.heads.numpy()
            tails = triples.tails.numpy()
            relations = triples.relations.numpy()
            non_self = heads != tails
            
            rel_freq = np.bincount(relations[non_self], minlength=triples.num_relations).astype(np.float64)
            
            protein_log_freq_sum = np.zeros(triples.num_proteins, dtype=np.float64)
            protein_edge_count = np.zeros(triples.num_proteins, dtype=np.float64)
            for i in range(len(heads)):
                if heads[i] != tails[i]:
                    lf = np.log1p(rel_freq[relations[i]])
                    protein_log_freq_sum[heads[i]] += lf
                    protein_edge_count[heads[i]] += 1
                    protein_log_freq_sum[tails[i]] += lf
                    protein_edge_count[tails[i]] += 1
            
            protein_edge_count = np.maximum(protein_edge_count, 1)
            mean_log_freq = protein_log_freq_sum / protein_edge_count
            
            X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
                embeddings, mean_log_freq, test_size=0.2, random_state=42 + fold
            )
            
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_f, y_train_f)
            y_pred_f = ridge.predict(X_test_f)
            freq_mae = mean_absolute_error(y_test_f, y_pred_f)
            freq_r2 = r2_score(y_test_f, y_pred_f)
            
            results.append({
                "variant": variant,
                "fold": fold,
                "degree_balanced_accuracy": float(deg_bal_acc),
                "degree_macro_f1": float(deg_macro_f1),
                "frequency_mae": float(freq_mae),
                "frequency_r2": float(freq_r2),
            })
            
            logger.info("  [%s] Fold %d: deg_bal_acc=%.4f freq_mae=%.4f freq_r2=%.4f",
                        variant, fold, deg_bal_acc, freq_mae, freq_r2)
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "probe_comparison.tsv", sep="\t", index=False)
    
    for metric in ["degree_balanced_accuracy", "degree_macro_f1", "frequency_mae", "frequency_r2"]:
        fu = df[df["variant"] == "full"][metric]
        un = df[df["variant"] == "uncorrected"][metric]
        logger.info("%s: full=%.4f±%.4f  uncorrected=%.4f±%.4f  Δ(full-uncorrected)=%.4f",
                    metric, fu.mean(), fu.std(), un.mean(), un.std(),
                    fu.mean() - un.mean())
    
    return df


if __name__ == "__main__":
    run_probes()
