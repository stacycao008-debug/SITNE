#!/usr/bin/env python3
"""Phase G (续): 生成 Figure 4 (参数敏感性), Figure 5 (表示探针), Figure 6 (shortcut 鲁棒性) + Table 4."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase_g_456")

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "08_results/sitne_walk_paper_v2"
FIG = R / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150})

# ============================================================
# Figure 4: Parameter sensitivity (α/β grid from R1)
# ============================================================
def generate_figure4():
    logger.info("=== Figure 4: Parameter sensitivity ===")
    grid = pd.read_csv(R / "r1_walk_correction/parameter_grid_results.tsv", sep="\t")

    alphas = sorted(grid["alpha"].unique())
    betas = sorted(grid["beta"].unique())

    mae_matrix = grid.pivot_table(index="beta", columns="alpha", values="probability_mae")
    r2_matrix = grid.pivot_table(index="beta", columns="alpha", values="probability_r2")
    deg_matrix = grid.pivot_table(index="beta", columns="alpha", values="empirical_degree_spearman")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel A: MAE heatmap
    ax = axes[0]
    sns.heatmap(mae_matrix, ax=ax, cmap="viridis_r", annot=True, fmt=".4f", cbar_kws={"label": "MAE"})
    ax.set_title("A. Calibration MAE")
    ax.set_xlabel("α (degree correction)"); ax.set_ylabel("β (frequency correction)")

    # Panel B: R² heatmap
    ax = axes[1]
    sns.heatmap(r2_matrix, ax=ax, cmap="viridis", annot=True, fmt=".4f", cbar_kws={"label": "R²"})
    ax.set_title("B. Calibration R²")
    ax.set_xlabel("α (degree correction)"); ax.set_ylabel("β (frequency correction)")

    # Panel C: degree Spearman heatmap
    ax = axes[2]
    sns.heatmap(deg_matrix, ax=ax, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                cbar_kws={"label": "Spearman(degree)"})
    ax.set_title("C. Degree exposure correlation")
    ax.set_xlabel("α (degree correction)"); ax.set_ylabel("β (frequency correction)")

    plt.tight_layout()
    fig.savefig(FIG / "figure_4_parameter_sensitivity.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "figure_4_parameter_sensitivity.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Figure 4 saved")

    # Table 4: parameter sensitivity (selected rows)
    key_configs = [
        (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
        (0.0, 0.25), (0.0, 0.5), (0.25, 0.25), (0.5, 0.5), (1.0, 1.0),
    ]
    rows = []
    for a, b in key_configs:
        row = grid[(grid["alpha"] == a) & (grid["beta"] == b)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        rows.append({
            "alpha": a, "beta": b,
            "MAE": r["probability_mae"],
            "R2": r["probability_r2"],
            "TV": r["mean_total_variation"],
            "Spearman_degree": r["empirical_degree_spearman"],
            "Spearman_frequency": r["empirical_frequency_spearman"],
        })
    table4 = pd.DataFrame(rows)
    table4.to_csv(FIG / "table_4_parameter_sensitivity.tsv", sep="\t", index=False)
    logger.info("Table 4 saved (%d configs)", len(table4))
    return table4


# ============================================================
# Figure 5: Representation probes
# ============================================================
def generate_figure5():
    logger.info("=== Figure 5: Representation probes (full vs uncorrected) ===")
    pc = pd.read_csv(R / "r4_robustness/probe_comparison.tsv", sep="\t")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    folds = sorted(pc["fold"].unique())
    variants = ["full", "uncorrected"]
    colors = {"full": "steelblue", "uncorrected": "coral"}
    x = np.arange(len(folds))
    width = 0.38

    def _grouped_bars(ax, metric, title, ylabel, chance=None, ylim=None):
        for i, variant in enumerate(variants):
            vals = pc[pc["variant"] == variant].sort_values("fold")[metric].tolist()
            ax.bar(x + (i - 0.5) * width, vals, width,
                   color=colors[variant], label=variant)
        ax.set_xticks(x)
        ax.set_xticklabels([f"fold {f}" for f in folds])
        ax.set_xlabel("Fold"); ax.set_ylabel(ylabel)
        ax.set_title(title)
        if chance is not None:
            ax.axhline(y=chance, color="red", linestyle="--", alpha=0.6,
                       label="chance (5 bins)")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)

    # Panel A: degree probe balanced accuracy (full vs uncorrected per fold)
    _grouped_bars(axes[0], "degree_balanced_accuracy", "A. Degree-bin probe",
                  "Balanced accuracy", chance=0.2, ylim=(0, 0.6))

    # Panel B: frequency Ridge MAE
    _grouped_bars(axes[1], "frequency_mae", "B. Frequency Ridge probe (MAE)",
                  "MAE (log-frequency)")

    # Panel C: frequency Ridge R²
    _grouped_bars(axes[2], "frequency_r2", "C. Frequency Ridge probe (R²)",
                  "R²", ylim=(0, 1))

    plt.tight_layout()
    fig.savefig(FIG / "figure_5_representation_probes.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "figure_5_representation_probes.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Figure 5 saved (full vs uncorrected)")


# ============================================================
# Figure 6: Shortcut robustness (strata + coverage)
# ============================================================
def generate_figure6():
    logger.info("=== Figure 6: Shortcut robustness ===")
    st = pd.read_csv(R / "r4_robustness/strata_analysis.tsv", sep="\t")
    cb = pd.read_csv(R / "r4_robustness/coverage_boundary.tsv", sep="\t")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel A: degree strata
    ax = axes[0]
    deg = st[st["stratum_type"] == "degree_stratum"].copy()
    order = ["low", "medium", "high", "very_high"]
    deg["stratum"] = pd.Categorical(deg["stratum"], categories=order, ordered=True)
    deg = deg.sort_values("stratum")
    ax.bar(deg["stratum"], deg["mean_mrr"], color="steelblue")
    ax.set_xlabel("Degree stratum (train-only)"); ax.set_ylabel("Pair MRR")
    ax.set_title("A. Degree strata")
    ax.set_ylim(0.9, 1.0)
    for i, (m, n) in enumerate(zip(deg["mean_mrr"], deg["n_queries"])):
        ax.text(i, m + 0.002, f"n={int(n)}", ha="center", fontsize=8)

    # Panel B: frequency strata
    ax = axes[1]
    freq = st[st["stratum_type"] == "frequency_stratum"].copy()
    f_order = ["low", "high"]
    freq["stratum"] = pd.Categorical(freq["stratum"], categories=f_order, ordered=True)
    freq = freq.sort_values("stratum")
    ax.bar(freq["stratum"], freq["mean_mrr"], color="coral")
    ax.set_xlabel("Frequency stratum (train-only)"); ax.set_ylabel("Pair MRR")
    ax.set_title("B. Frequency strata")
    ax.set_ylim(0, 1.0)
    for i, (m, n) in enumerate(zip(freq["mean_mrr"], freq["n_queries"])):
        ax.text(i, m + 0.02, f"n={int(n)}", ha="center", fontsize=8)

    # Panel C: coverage boundary
    ax = axes[2]
    folds = cb["fold"].tolist()
    coverage = cb["coverage"].tolist()
    unsupported = cb["unsupported_protein_rows"].tolist()
    ax.bar(folds, coverage, color="darkgreen", label="coverage")
    ax.set_xlabel("Fold"); ax.set_ylabel("Coverage")
    ax.set_title("C. Transductive coverage boundary")
    ax.set_ylim(0.95, 1.0)
    ax2 = ax.twinx()
    ax2.plot(folds, unsupported, "ro-", label="unsupported proteins")
    ax2.set_ylabel("Unsupported protein rows", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG / "figure_6_shortcut_robustness.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "figure_6_shortcut_robustness.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Figure 6 saved")


def main():
    generate_figure4()
    generate_figure5()
    generate_figure6()
    logger.info("\n=== Figures 4-6 + Table 4 完成 ===")


if __name__ == "__main__":
    main()
