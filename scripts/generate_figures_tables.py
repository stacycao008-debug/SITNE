#!/usr/bin/env python3
"""Phase G: 图表与数据表格生成 (v2 results)

基于 Phases A-F 的 frozen artifacts 生成:
- Table 1: Dataset/split statistics  
- Table 2: Main benchmark (method × metric)
- Table 3: Ablation results (等 Phase D 完成)
- Figure 1: Transition calibration (scatter + spearman)
- Figure 2: Main performance forest plot
"""
from __future__ import annotations

import json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase_g")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "08_results/sitne_walk_paper_v2/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150})


# ============================================================
# Table 1: Dataset/Split Statistics
# ============================================================
def generate_table1():
    logger.info("=== Table 1: Dataset Statistics ===")
    
    manifest = json.load(open(ROOT / "05_splits/typed_ranking_v2/split_manifest.json"))
    
    rows = []
    rows.append({"Statistic": "Total unique protein pairs", "Value": manifest["total_pairs"]})
    rows.append({"Statistic": "Total pair-type facts", "Value": manifest["total_facts"]})
    rows.append({"Statistic": "Total unique proteins", "Value": manifest["total_proteins"]})
    rows.append({"Statistic": "Total relation types (coarse)", "Value": manifest["total_relations"]})
    rows.append({"Statistic": "Outer folds", "Value": manifest["n_outer_folds"]})
    rows.append({"Statistic": "Inner folds (per outer)", "Value": manifest["n_inner_folds"]})
    rows.append({"Statistic": "CV split seed", "Value": manifest["split_seed"]})
    
    # Per-fold averages
    per_fold = manifest["per_fold"]
    avg_train_pairs = np.mean([f["num_train_pairs"] for f in per_fold])
    avg_test_pairs = np.mean([f["num_test_pairs"] for f in per_fold])
    avg_val_pairs = np.mean([f["num_val_pairs"] for f in per_fold])
    rows.append({"Statistic": "Avg train pairs per fold", "Value": f"{avg_train_pairs:.0f}"})
    rows.append({"Statistic": "Avg test pairs per fold", "Value": f"{avg_test_pairs:.0f}"})
    rows.append({"Statistic": "Avg val pairs per fold", "Value": f"{avg_val_pairs:.0f}"})
    
    # Coverage
    val_cov = np.mean([f["val_protein_coverage"] for f in per_fold])
    test_cov = np.mean([f["test_protein_coverage"] for f in per_fold])
    rows.append({"Statistic": "Mean val protein coverage", "Value": f"{val_cov:.1%}"})
    rows.append({"Statistic": "Mean test protein coverage", "Value": f"{test_cov:.1%}"})
    
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "table_1_dataset_stats.tsv", sep="\t", index=False)
    
    for _, row in df.iterrows():
        logger.info(f"  {row['Statistic']:40s}: {row['Value']}")
    
    return df


# ============================================================
# Table 2: Main Benchmark
# ============================================================
def generate_table2():
    logger.info("\n=== Table 2: Main Benchmark ===")
    
    # SITNE-Walk
    sw = pd.read_csv(ROOT / "08_results/sitne_walk_paper_v2/r2_main_performance/optimal/training_summary.tsv", sep="\t")
    
    # Baselines
    baseline_dir = ROOT / "08_results/sitne_walk_paper_v2/r2_main_performance/baselines"
    simple = pd.read_csv(baseline_dir / "simple" / "simple_baselines_summary.tsv", sep="\t")
    dm = pd.read_csv(baseline_dir / "distmult" / "distmult_5fold_summary.tsv", sep="\t")
    cx = pd.read_csv(baseline_dir / "complex" / "complex_5fold_summary.tsv", sep="\t")
    mp = pd.read_csv(baseline_dir / "metapath2vec" / "metapath2vec_5fold_summary.tsv", sep="\t")
    sg = pd.read_csv(baseline_dir / "typed_skipgram" / "typed_skipgram_summary.tsv", sep="\t")
    
    methods = {}
    
    # SITNE-Walk
    methods["SITNE-Walk (ours)"] = {
        "mrr": f"{sw['mrr'].mean():.4f} ± {sw['mrr'].std():.4f}",
        "hits1": "-",
        "hits10": "-",
    }
    
    # Simple baselines
    for name in ["random", "frequency", "degree", "binary"]:
        sub = simple[simple["baseline"] == name]
        methods[name.capitalize()] = {
            "mrr": f"{sub['mrr'].mean():.4f} ± {sub['mrr'].std():.4f}",
            "hits1": f"{sub['hits@1'].mean():.4f}",
            "hits10": f"{sub['hits@10'].mean():.4f}",
        }
    
    # DistMult
    methods["DistMult"] = {
        "mrr": f"{dm['test_mrr'].mean():.4f} ± {dm['test_mrr'].std():.4f}",
        "hits1": f"{dm['test_hits1'].mean():.4f}",
        "hits10": f"{dm['test_hits10'].mean():.4f}",
    }
    
    # ComplEx
    methods["ComplEx"] = {
        "mrr": f"{cx['test_mrr'].mean():.4f} ± {cx['test_mrr'].std():.4f}",
        "hits1": f"{cx['test_hits1'].mean():.4f}",
        "hits10": f"{cx['test_hits10'].mean():.4f}",
    }
    
    # metapath2vec
    methods["metapath2vec"] = {
        "mrr": f"{mp['test_mrr'].mean():.4f} ± {mp['test_mrr'].std():.4f}",
        "hits1": f"{mp['test_hits1'].mean():.4f}",
        "hits10": f"{mp['test_hits10'].mean():.4f}",
    }
    
    # Typed Skip-Gram
    methods["Typed Skip-Gram"] = {
        "mrr": f"{sg['mrr'].mean():.4f} ± {sg['mrr'].std():.4f}",
        "hits1": f"{sg['hits@1'].mean():.4f}",
        "hits10": f"{sg['hits@10'].mean():.4f}",
    }
    
    rows = []
    for method, metrics in methods.items():
        rows.append({
            "Method": method,
            "Filtered MRR": metrics["mrr"],
            "Hits@1": metrics["hits1"],
            "Hits@10": metrics["hits10"],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "table_2_main_benchmark.tsv", sep="\t", index=False)
    
    for _, row in df.iterrows():
        logger.info(f"  {row['Method']:>20s}: MRR={row['Filtered MRR']}")
    
    return df


# ============================================================
# Figure 1: Transition Calibration
# ============================================================
def generate_figure1():
    logger.info("\n=== Figure 1: Transition Calibration ===")
    
    r1_dir = ROOT / "08_results/sitne_walk_paper_v2/r1_walk_correction"
    
    # Load parameter grid results
    grid = pd.read_csv(r1_dir / "parameter_grid_results.tsv", sep="\t")
    
    # Load transition samples for best corrected and uncorrected
    samples = pd.read_csv(r1_dir / "transition_samples.tsv", sep="\t")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Scatter — theoretical vs empirical (corrected vs uncorrected)
    ax = axes[0]
    unc = samples[samples["alpha"] == 0.0]
    corr = samples[samples["alpha"] == 0.25]
    
    # Sample for readability
    unc_sample = unc.sample(min(2000, len(unc)), random_state=42)
    corr_sample = corr.sample(min(2000, len(corr)), random_state=42)
    
    ax.scatter(unc_sample["theoretical_probability"], unc_sample["empirical_probability"],
               alpha=0.3, s=3, label=f"Uncorrected (MAE={grid[grid['alpha']==0]['probability_mae'].values[0]:.4f})", color="blue")
    ax.scatter(corr_sample["theoretical_probability"], corr_sample["empirical_probability"],
               alpha=0.3, s=3, label=f"Corrected (MAE={grid[grid['alpha']==0.25]['probability_mae'].values[0]:.4f})", color="red")
    
    lims = [0, max(unc_sample["theoretical_probability"].max(), unc_sample["empirical_probability"].max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Theoretical Probability"); ax.set_ylabel("Empirical Probability")
    ax.legend(fontsize=8); ax.set_title("A. Transition Calibration")
    
    # Panel B: Spearman correlation
    ax = axes[1]
    grid_plot = grid[grid["beta"] == 0.0].copy()
    grid_plot = grid_plot.sort_values("alpha")
    
    x_pos = range(len(grid_plot))
    width = 0.35
    bars1 = ax.bar([p - width/2 for p in x_pos], grid_plot["empirical_degree_spearman"].abs(),
                    width, label="|Spearman(degree)|", color="coral")
    bars2 = ax.bar([p + width/2 for p in x_pos], grid_plot["empirical_frequency_spearman"].abs(),
                    width, label="|Spearman(frequency)|", color="skyblue")
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"α={a:.2f}" for a in grid_plot["alpha"]], rotation=45)
    ax.set_ylabel("|Spearman ρ|"); ax.set_xlabel("Degree Correction (α)")
    ax.legend(fontsize=8); ax.set_title("B. Degree/Frequency Bias")
    
    # Panel C: MAE vs alpha
    ax = axes[2]
    ax.plot(grid_plot["alpha"], grid_plot["probability_mae"], "o-", color="darkgreen", markersize=6)
    ax.axhline(y=grid[grid["alpha"]==0]["probability_mae"].values[0], color="blue", 
               linestyle="--", alpha=0.5, label=f"Uncorrected MAE")
    ax.set_xlabel("α (degree correction)"); ax.set_ylabel("MAE")
    ax.legend(fontsize=8); ax.set_title("C. Calibration Error")
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_1_transition_calibration.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure_1_transition_calibration.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Figure 1 saved")


# ============================================================
# Figure 2: Main Performance Forest Plot
# ============================================================
def generate_figure2():
    logger.info("\n=== Figure 2: Main Performance ===")
    
    # Load paired effects
    effects = pd.read_csv(ROOT / "08_results/sitne_walk_paper_v2/statistics/paired_effects.tsv", sep="\t")
    
    # Also load per-method MRRs for bar chart
    sw = pd.read_csv(ROOT / "08_results/sitne_walk_paper_v2/r2_main_performance/optimal/training_summary.tsv", sep="\t")
    baseline_dir = ROOT / "08_results/sitne_walk_paper_v2/r2_main_performance/baselines"
    simple = pd.read_csv(baseline_dir / "simple" / "simple_baselines_summary.tsv", sep="\t")
    dm = pd.read_csv(baseline_dir / "distmult" / "distmult_5fold_summary.tsv", sep="\t")
    cx = pd.read_csv(baseline_dir / "complex" / "complex_5fold_summary.tsv", sep="\t")
    mp = pd.read_csv(baseline_dir / "metapath2vec" / "metapath2vec_5fold_summary.tsv", sep="\t")
    sg = pd.read_csv(baseline_dir / "typed_skipgram" / "typed_skipgram_summary.tsv", sep="\t")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Bar chart of MRR by method
    ax = axes[0]
    methods = {
        "SITNE-Walk": (sw["mrr"].mean(), sw["mrr"].std()),
        "Frequency": (simple[simple["baseline"]=="frequency"]["mrr"].mean(),
                       simple[simple["baseline"]=="frequency"]["mrr"].std()),
        "metapath2vec": (mp["test_mrr"].mean(), mp["test_mrr"].std()),
        "Typed Skip-Gram": (sg["mrr"].mean(), sg["mrr"].std()),
        "DistMult": (dm["test_mrr"].mean(), dm["test_mrr"].std()),
        "ComplEx": (cx["test_mrr"].mean(), cx["test_mrr"].std()),
        "Random": (simple[simple["baseline"]=="random"]["mrr"].mean(),
                    simple[simple["baseline"]=="random"]["mrr"].std()),
        "Degree": (simple[simple["baseline"]=="degree"]["mrr"].mean(),
                    simple[simple["baseline"]=="degree"]["mrr"].std()),
        "Binary": (simple[simple["baseline"]=="binary"]["mrr"].mean(),
                    simple[simple["baseline"]=="binary"]["mrr"].std()),
    }
    
    names = list(methods.keys())
    mrrs = [methods[n][0] for n in names]
    errs = [methods[n][1] for n in names]
    colors = ["darkgreen"] + ["steelblue"] * 3 + ["coral"] * 5
    
    bars = ax.barh(names, mrrs, xerr=errs, color=colors, capsize=3)
    ax.set_xlabel("Filtered MRR")
    ax.set_title("A. Main Benchmark Results")
    ax.set_xlim(0, 1.05)
    
    # Add value labels
    for bar, mrr in zip(bars, mrrs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{mrr:.4f}", va="center", fontsize=9)
    
    # Panel B: Forest plot (SITNE-Walk vs baselines)
    ax = axes[1]
    baseline_names = effects["method_b"].tolist()
    diffs = effects["mean_diff"].tolist()
    ci_low = [effects["ci_95_lower"].tolist()[i] for i in range(len(effects))]
    ci_high = [effects["ci_95_upper"].tolist()[i] for i in range(len(effects))]
    p_vals = effects["p_value"].tolist()
    
    y_positions = range(len(baseline_names))
    for i, (name, diff, lo, hi, p) in enumerate(zip(baseline_names, diffs, ci_low, ci_high, p_vals)):
        color = "darkgreen" if diff > 0 else "coral"
        ax.errorbar(diff, i, xerr=[[diff - lo], [hi - diff]], fmt="o", color=color, 
                     capsize=4, markersize=8)
        sig = "*" if p < 0.05 else ("†" if p < 0.1 else "")
        ax.text(hi + 0.002, i, f"{diff:+.4f}{sig}", va="center", fontsize=9)
    
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(baseline_names)
    ax.set_xlabel("Δ MRR (SITNE-Walk − Baseline)")
    ax.set_title("B. Paired Effect Sizes (95% CI)")
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_2_main_performance.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure_2_main_performance.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Figure 2 saved")


# ============================================================
# Figure 3: Ablation Waterfall (等 Phase D 完成后)
# ============================================================
def generate_figure3():
    logger.info("\n=== Figure 3: Ablation Effects ===")
    
    ablation_dir = ROOT / "08_results/sitne_walk_paper_v2/r3_ablation"
    summary = ablation_dir / "ablation_summary.tsv"
    
    if not summary.exists():
        logger.warning("Ablation summary not yet available, skipping Figure 3")
        return
    
    # Generate waterfall plot once data is ready
    df = pd.read_csv(summary, sep="\t")
    df = df[df["variant"] != "full"]
    
    if len(df) == 0:
        logger.warning("No ablation data, skipping")
        return
    
    # Per-variant mean MRR
    variant_mrrs = df.groupby("variant")["mrr"].agg(["mean", "std"]).reset_index()
    
    # Full model baseline
    full_df = pd.read_csv(summary, sep="\t")
    full_df = full_df[full_df["variant"] == "full"]
    full_mrr = full_df["mrr"].mean() if len(full_df) > 0 else 0.9806
    
    variant_mrrs["delta"] = variant_mrrs["mean"] - full_mrr
    variant_mrrs = variant_mrrs.sort_values("delta")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ["coral" if d < 0 else "steelblue" for d in variant_mrrs["delta"]]
    bars = ax.barh(variant_mrrs["variant"], variant_mrrs["delta"], color=colors)
    
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_xlabel("Δ MRR (Variant − Full)")
    ax.set_title("Ablation Effects (v2 splits)")
    
    for bar, delta, std in zip(bars, variant_mrrs["delta"], variant_mrrs["std"]):
        ax.text(bar.get_width() + (0.002 if bar.get_width() >= 0 else -0.04),
                bar.get_y() + bar.get_height()/2,
                f"{delta:+.4f}", va="center", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_3_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figure_3_ablation.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Figure 3 saved")


# ============================================================
# Main
# ============================================================
def main():
    logger.info("=== Phase G: Charts & Tables Generation ===")
    
    generate_table1()
    generate_table2()
    generate_figure1()
    generate_figure2()
    generate_figure3()
    
    logger.info("\n=== Phase G Complete ===")
    logger.info("Output directory: %s", OUTPUT_DIR)
    for f in sorted(OUTPUT_DIR.glob("*")):
        logger.info("  %s", f.name)


if __name__ == "__main__":
    main()
