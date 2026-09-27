#!/usr/bin/env python3
"""Phase D 收尾: 生成 Table 3 (消融结果) + Figure 3 (消融效应)。

合并 run_ablation_v2.py 的 9 个变体结果 + run_shortcut_only.py 的 shortcut_only 结果，
生成统一的 ablation_summary.tsv、Table 3 和 Figure 3。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("table3_figure3")

ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = ROOT / "08_results/sitne_walk_paper_v2/r3_ablation"
FIGURE_DIR = ROOT / "08_results/sitne_walk_paper_v2/figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def collect_all_results() -> pd.DataFrame:
    """扫描 ablation 目录下所有 test_metrics.json，收集每个 (variant, fold, seed) 的 MRR。"""
    rows = []
    for variant_dir in sorted(ABLATION_DIR.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        for fold_dir in sorted(variant_dir.glob("fold_*")):
            try:
                fold = int(fold_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            for seed_dir in sorted(fold_dir.glob("seed_*")):
                try:
                    seed = int(seed_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                for run_dir in seed_dir.glob("sitne_walk_*"):
                    tf = run_dir / "test_metrics.json"
                    if tf.exists():
                        metrics = json.load(open(tf))["metrics"]
                        rows.append({
                            "variant": variant,
                            "fold": fold,
                            "seed": seed,
                            "mrr": metrics["mrr"],
                            "hits1": metrics.get("hits@1", np.nan),
                            "hits3": metrics.get("hits@3", np.nan),
                            "hits10": metrics.get("hits@10", np.nan),
                        })
                # shortcut_only 直接写在 seed_dir 下（无 sitne_walk_* 子目录）
                tf = seed_dir / "test_metrics.json"
                if tf.exists():
                    metrics = json.load(open(tf))["metrics"]
                    rows.append({
                        "variant": variant,
                        "fold": fold,
                        "seed": seed,
                        "mrr": metrics["mrr"],
                        "hits1": metrics.get("hits@1", np.nan),
                        "hits3": metrics.get("hits@3", np.nan),
                        "hits10": metrics.get("hits@10", np.nan),
                    })
    return pd.DataFrame(rows)


def main():
    logger.info("=== 收集消融结果 ===")
    df = collect_all_results()
    if len(df) == 0:
        logger.error("未找到任何消融结果")
        return

    # 去重（同一个 variant/fold/seed 可能被重复记录）
    df = df.drop_duplicates(subset=["variant", "fold", "seed"], keep="first")

    # 合并退化变体: 由于 optimal 配置 beta=0，alpha0 / binary_walk / uncorrected
    # 三者退化为同一配置 (alpha=0, beta=0)，统一标注为 "uncorrected"。
    degenerate = {"alpha0": "uncorrected", "binary_walk": "uncorrected"}
    df["variant"] = df["variant"].replace(degenerate)
    df = df.drop_duplicates(subset=["variant", "fold", "seed"], keep="first")

    summary_path = ABLATION_DIR / "ablation_summary.tsv"
    df.to_csv(summary_path, sep="\t", index=False)
    logger.info("已保存 %d 行 -> %s", len(df), summary_path)

    # 每个变体的均值/标准差
    grouped = df.groupby("variant").agg(
        mean_mrr=("mrr", "mean"),
        std_mrr=("mrr", "std"),
        n=("mrr", "count"),
        mean_hits1=("hits1", "mean"),
        mean_hits10=("hits10", "mean"),
    ).reset_index()

    logger.info("\n=== 消融结果汇总 ===")
    for _, row in grouped.iterrows():
        logger.info("  %-18s: MRR=%.4f ± %.4f (n=%d)", row["variant"], row["mean_mrr"], row["std_mrr"], row["n"])

    # === Table 3 ===
    full_mrr = grouped.loc[grouped["variant"] == "full", "mean_mrr"]
    full_val = full_mrr.values[0] if len(full_mrr) > 0 else 0.9806

    grouped["delta_mrr"] = grouped["mean_mrr"] - full_val
    # 排序: full 第一，然后按 delta 降序
    full_row = grouped[grouped["variant"] == "full"]
    rest = grouped[grouped["variant"] != "full"].sort_values("delta_mrr", ascending=False)
    table3 = pd.concat([full_row, rest], ignore_index=True)

    table3_out = table3[["variant", "mean_mrr", "std_mrr", "delta_mrr", "mean_hits1", "mean_hits10", "n"]].copy()
    table3_out.columns = ["Variant", "MRR", "Std", "ΔMRR", "Hits@1", "Hits@10", "n"]
    table3_path = FIGURE_DIR / "table_3_ablation.tsv"
    table3_out.to_csv(table3_path, sep="\t", index=False)
    logger.info("\nTable 3 已保存 -> %s", table3_path)

    # === Figure 3: 消融效应瀑布图 ===
    rest_sorted = rest.sort_values("delta_mrr")
    labels = rest_sorted["variant"].tolist()
    deltas = rest_sorted["delta_mrr"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#d62728" if d < -0.01 else ("#2ca02c" if d > 0.001 else "#7f7f7f") for d in deltas]
    bars = ax.barh(labels, deltas, color=colors, edgecolor="black", linewidth=0.5)

    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_xlabel("Δ MRR (Variant − Full model)", fontsize=12)
    ax.set_title("Ablation Effects on Filtered MRR", fontsize=13)
    ax.axvline(x=0, color="black", linewidth=1)

    for bar, delta in zip(bars, deltas):
        offset = 0.0005 if delta >= 0 else -0.005
        ax.text(delta + offset, bar.get_y() + bar.get_height() / 2,
                f"{delta:+.4f}", va="center", ha="left" if delta >= 0 else "right", fontsize=9)

    # 处理 no_rank 的大负值（可能超出坐标范围）
    ax.set_xlim(min(deltas) - 0.05, max(deltas) + 0.01)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "figure_3_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure_3_ablation.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Figure 3 已保存 -> %s", FIGURE_DIR / "figure_3_ablation.png")

    # === Figure 3 附加: 排除 no_rank 的放大视图（否则其他变体被压缩） ===
    rest_norank = rest[rest["variant"] != "no_rank"].sort_values("delta_mrr")
    if len(rest_norank) > 0:
        labels2 = rest_norank["variant"].tolist()
        deltas2 = rest_norank["delta_mrr"].tolist()
        fig2, ax2 = plt.subplots(figsize=(10, 4.5))
        colors2 = ["#d62728" if d < -0.001 else "#2ca02c" for d in deltas2]
        ax2.barh(labels2, deltas2, color=colors2, edgecolor="black", linewidth=0.5)
        ax2.axvline(x=0, color="black", linewidth=1)
        ax2.set_xlabel("Δ MRR (Variant − Full model)", fontsize=12)
        ax2.set_title("Ablation Effects (excluding no_rank)", fontsize=13)
        for i, delta in enumerate(deltas2):
            offset = 0.0003 if delta >= 0 else -0.0003
            ax2.text(delta + offset, i, f"{delta:+.4f}", va="center",
                     ha="left" if delta >= 0 else "right", fontsize=9)
        plt.tight_layout()
        fig2.savefig(FIGURE_DIR / "figure_3_ablation_zoomed.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Figure 3 (zoomed) 已保存")

    logger.info("\n=== Phase D 完成 ===")


if __name__ == "__main__":
    main()
