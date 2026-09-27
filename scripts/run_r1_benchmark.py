#!/usr/bin/env python3
"""R1 Walk Correction Benchmark — 参数网格扫描 (v2 splits)

对 α,β ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0} 的 36 个组合，在 fold_0 train graph 上执行
transition calibration benchmark。测量 theoretical vs empirical probability 的 MAE/R²/TV
以及 empirical-degree 和 empirical-frequency 的 Spearman correlation。

输出: transition_summary.tsv + parameter_grid_results.tsv
"""
from __future__ import annotations

import json, logging, os, sys, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("r1_benchmark")

ROOT = Path(__file__).resolve().parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, str(ROOT / "06_code"))

from sitne_walk.config import SITNEConfig
from sitne_walk.data import load_training_triples, load_relation_modes
from sitne_walk.graph import build_packed_csr_graph
from sitne_walk.walks import AliasTypedWalker, make_generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Configuration
# ============================================================
SPLIT_DIR = ROOT / "05_splits/typed_ranking_v2"
OUTPUT_DIR = ROOT / "08_results/sitne_walk_paper_v2/r1_walk_correction"
TRAIN_PATH = SPLIT_DIR / "fold_0" / "train.tsv"

# 参数网格: α, β
ALPHA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
BETA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

# Benchmark 参数
MC_SAMPLES = 5000  # 每节点蒙特卡洛采样数
MAX_NODES = 200    # 最大节点数（只测试有出边的节点）
WALK_LENGTH = 2    # 采样游走长度（只需 1 步来测下一跳概率）
SEED = 42

# ============================================================
# Load data
# ============================================================
log.info("Loading training triples from %s", TRAIN_PATH)
triples = load_training_triples(
    str(TRAIN_PATH),
    duplicate_policy="binary",
    pair_semantics="unordered",
)
log.info("Train: %d proteins, %d relations, %d edges", triples.num_proteins, triples.num_relations, triples.num_triples)

rmodes = load_relation_modes(None, triples.id_to_relation, default_mode="symmetric")


def build_graph(alpha: float, beta: float):
    """构建指定 alpha/beta 的图"""
    return build_packed_csr_graph(
        triples,
        relation_modes=rmodes,
        alpha=alpha,
        beta=beta,
        epsilon=1e-8,
        drop_self_loops=True,
    )


def run_transition_benchmark(
    graph,
    variant_name: str,
    alpha: float,
    beta: float,
    nodes: np.ndarray,
    mc_samples: int,
    walk_length: int,
    seed: int,
) -> list[dict]:
    """对给定图执行 transition calibration benchmark"""
    graph = graph.to(DEVICE)
    walker = AliasTypedWalker(graph)
    generator = make_generator(DEVICE, seed)

    rows = []

    for i, src in enumerate(nodes):
        src = int(src)
        start = int(graph.indptr[src].item())
        end = int(graph.indptr[src + 1].item())
        degree = end - start
        if degree == 0:
            continue

        # 理论概率（transition_weights 已归一化）
        theory_weights = graph.transition_weights[start:end].cpu().numpy().astype(np.float64)
        theory_sum = theory_weights.sum()
        if theory_sum <= 0:
            continue
        theory_probs = theory_weights / theory_sum

        destinations = graph.destinations[start:end].cpu().numpy()
        relations = graph.relations[start:end].cpu().numpy()

        # 蒙特卡洛采样
        counts = np.zeros(degree, dtype=np.float64)
        # 批量采样
        start_tensor = torch.full((mc_samples,), src, dtype=torch.int64, device=DEVICE)
        batch = walker.generate(start_tensor, walk_length=walk_length, generator=generator)
        next_nodes = batch.proteins[:, 1].cpu().numpy()  # 第一步后的节点

        for j in range(mc_samples):
            dst = next_nodes[j]
            if dst >= 0:  # valid step
                # 找到对应的 edge index
                for e in range(degree):
                    if destinations[e] == dst:
                        counts[e] += 1
                        break

        empirical_probs = counts / max(counts.sum(), 1)

        for e in range(degree):
            dst = int(destinations[e])
            dst_degree = float(graph.protein_degree[dst].item())
            rel_freq = float(graph.relation_frequency[int(relations[e])].item())
            rows.append({
                "variant": variant_name,
                "alpha": alpha,
                "beta": beta,
                "node_id": src,
                "destination": dst,
                "relation_id": int(relations[e]),
                "theoretical_probability": float(theory_probs[e]),
                "empirical_probability": float(empirical_probs[e]),
                "destination_degree": dst_degree,
                "relation_frequency": rel_freq,
            })

        if (i + 1) % 50 == 0:
            log.info("  [%s] node %d/%d, %d rows so far", variant_name, i + 1, len(nodes), len(rows))

    return rows


def summarize_benchmark(rows: list[dict]) -> dict:
    """从采样行计算汇总指标"""
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return {}

    theory = df["theoretical_probability"].to_numpy()
    empirical = df["empirical_probability"].to_numpy()
    dst_degree = df["destination_degree"].to_numpy()
    rel_freq = df["relation_frequency"].to_numpy()

    # Total Variation (TV)
    tv = 0.5 * np.abs(theory - empirical).sum() / len(theory)

    # MAE
    mae = np.abs(theory - empirical).mean()

    # R²
    ss_res = np.sum((theory - empirical) ** 2)
    ss_tot = np.sum((empirical - empirical.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-16)

    # Spearman
    from scipy.stats import spearmanr
    deg_corr, _ = spearmanr(empirical, dst_degree)
    freq_corr, _ = spearmanr(rel_freq, empirical)

    return {
        "num_rows": len(df),
        "mean_total_variation": float(tv),
        "probability_mae": float(mae),
        "probability_r2": float(r2),
        "empirical_degree_spearman": float(deg_corr),
        "empirical_frequency_spearman": float(freq_corr),
    }


# ============================================================
# Main
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 选择有出边的节点
    log.info("Identifying nodes with outgoing edges...")
    base_graph = build_graph(0.0, 0.0)  # uncorrected
    valid_nodes = base_graph.nodes_with_outgoing_edges.cpu().numpy()
    np.random.seed(SEED)
    n_nodes = min(MAX_NODES, len(valid_nodes))
    selected_nodes = np.random.choice(valid_nodes, size=n_nodes, replace=False)
    log.info("Selected %d nodes out of %d with outgoing edges", n_nodes, len(valid_nodes))

    # 参数网格扫描
    all_summaries = []
    all_rows = []

    param_grid = list(itertools.product(ALPHA_VALUES, BETA_VALUES))
    log.info("Starting parameter grid scan: %d combinations (α∈%s, β∈%s)",
             len(param_grid), ALPHA_VALUES, BETA_VALUES)

    for idx, (alpha, beta) in enumerate(param_grid):
        variant = f"alpha_{alpha}_beta_{beta}"
        log.info("[%d/%d] α=%.2f, β=%.2f", idx + 1, len(param_grid), alpha, beta)

        graph = build_graph(alpha, beta)
        rows = run_transition_benchmark(
            graph, variant, alpha, beta,
            selected_nodes, MC_SAMPLES, WALK_LENGTH, SEED
        )
        all_rows.extend(rows)

        summary = summarize_benchmark(rows)
        summary["variant"] = variant
        summary["alpha"] = alpha
        summary["beta"] = beta
        all_summaries.append(summary)

        log.info("  MAE=%.4f, R²=%.4f, TV=%.4f, Spearman(deg)=%.4f, Spearman(freq)=%.4f",
                 summary["probability_mae"], summary["probability_r2"],
                 summary["mean_total_variation"],
                 summary["empirical_degree_spearman"],
                 summary["empirical_frequency_spearman"])

    # 保存结果
    summary_df = pd.DataFrame(all_summaries)
    summary_path = OUTPUT_DIR / "parameter_grid_results.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    log.info("Saved parameter grid results → %s", summary_path)

    # 保存详细采样数据
    detail_df = pd.DataFrame(all_rows)
    detail_path = OUTPUT_DIR / "transition_samples.tsv"
    detail_df.to_csv(detail_path, sep="\t", index=False)
    log.info("Saved %d transition sample rows → %s", len(detail_df), detail_path)

    # 打印最优结果
    log.info("\n=== Best Results ===")
    best_mae = summary_df.loc[summary_df["probability_mae"].idxmin()]
    log.info("Best MAE: α=%.2f β=%.2f MAE=%.6f", best_mae["alpha"], best_mae["beta"], best_mae["probability_mae"])
    best_r2 = summary_df.loc[summary_df["probability_r2"].idxmax()]
    log.info("Best R²:  α=%.2f β=%.2f R²=%.6f", best_r2["alpha"], best_r2["beta"], best_r2["probability_r2"])

    # 与 uncorrected (α=β=0) 比较
    uncorrected = summary_df[(summary_df["alpha"] == 0.0) & (summary_df["beta"] == 0.0)]
    if len(uncorrected) > 0:
        uc = uncorrected.iloc[0]
        log.info("\nUncorrected (α=β=0): MAE=%.6f R²=%.4f Spearman(deg)=%.4f Spearman(freq)=%.4f",
                 uc["probability_mae"], uc["probability_r2"],
                 uc["empirical_degree_spearman"], uc["empirical_frequency_spearman"])

    # 寻找全面优于 uncorrected 的组合
    if len(uncorrected) > 0:
        uc = uncorrected.iloc[0]
        better = summary_df[
            (summary_df["probability_mae"] < uc["probability_mae"]) &
            (summary_df["probability_r2"] > uc["probability_r2"]) &
            (summary_df["empirical_degree_spearman"].abs() < abs(uc["empirical_degree_spearman"])) &
            (summary_df["empirical_frequency_spearman"].abs() < abs(uc["empirical_frequency_spearman"]))
        ]
        log.info("\nCombinations strictly better than uncorrected on ALL metrics: %d", len(better))
        if len(better) > 0:
            log.info("Best overall: α=%.2f β=%.2f", better.iloc[0]["alpha"], better.iloc[0]["beta"])

    log.info("\nDone. Results in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
