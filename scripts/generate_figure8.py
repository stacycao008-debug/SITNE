#!/usr/bin/env python3
"""T11 (续): 生成 Figure 8 (案例研究候选对可视化) 骨架 + Table 6 骨架.

输入:
  - 08_results/sitne_walk_paper_v2/case_studies/case_candidates.tsv  (已冻结)

输出:
  - 08_results/sitne_walk_paper_v2/figures/figure_8_case_studies.{png,pdf}
  - 08_results/sitne_walk_paper_v2/case_studies/table_6_case_studies.tsv

说明:
  - 本图/表为「骨架」：展示候选对在各 relation 下的排序分布与跨 fold/seed 稳定性，
    证据列 (known_interaction/known_type/pmid/consistency/note) 由领域专家后续填充。
  - 只主张 known-pair prioritization，不主张新 PPI discovery。
"""
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
logger = logging.getLogger("figure8")

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "08_results/sitne_walk_paper_v2/case_studies"
FIG = ROOT / "08_results/sitne_walk_paper_v2/figures"
FIG.mkdir(parents=True, exist_ok=True)

CANDIDATES = CASE / "case_candidates.tsv"
RELATION_NAMES = {0: "enzymatic", 1: "general", 2: "physical", 3: "spatial"}
RELATION_COLORS = {"enzymatic": "#440154", "general": "#21918c",
                   "physical": "#fde725", "spatial": "#5ec962"}

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 150})


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["relation_name"] = df["relation_id"].map(RELATION_NAMES)
    return df


def generate_figure8(df: pd.DataFrame) -> None:
    """4 面板: 每 relation 候选数 / mean_rank 分布 / 稳定性(std) / top pair 示例表。"""
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          hspace=0.45, wspace=0.3)
    rel_order = ["enzymatic", "general", "physical", "spatial"]
    colors = [RELATION_COLORS[r] for r in rel_order]

    # Panel A: 每 relation 候选对数量（柱状）
    ax_a = fig.add_subplot(gs[0, 0])
    counts = [int((df["relation_name"] == r).sum()) for r in rel_order]
    ax_a.bar(rel_order, counts, color=colors, edgecolor="grey", linewidth=0.6)
    ax_a.set_ylabel("n candidate pairs")
    ax_a.set_title("A. Candidate count per relation")
    for i, c in enumerate(counts):
        ax_a.text(i, c, str(c), ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: mean_rank 分布（按 relation 的箱线图）
    ax_b = fig.add_subplot(gs[0, 1])
    data_by_rel = [df[df["relation_name"] == r]["mean_rank"].dropna().values
                   for r in rel_order]
    bp = ax_b.boxplot(data_by_rel, tick_labels=rel_order, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_b.set_ylabel("mean filtered rank")
    ax_b.set_title("B. Mean rank distribution (enzymatic/general have discriminative rank)")
    ax_b.set_ylim(0.5, 5.5)

    # Panel C: std_rank 稳定性（跨 fold/seed）
    ax_c = fig.add_subplot(gs[1, 0])
    std_by_rel = [df[df["relation_name"] == r]["std_rank"].dropna().values
                  for r in rel_order]
    bp2 = ax_c.boxplot(std_by_rel, tick_labels=rel_order, patch_artist=True, showfliers=False)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_c.set_ylabel("rank std (cross fold/seed)")
    ax_c.set_title("C. Cross-fold/seed stability (std of rank)")

    # Panel D: 文字说明占位（证据待专家填充）
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_axis_off()
    txt = (
        "D. Evidence pending expert review\n\n"
        "Candidate pairs frozen at:\n"
        f"  {CANDIDATES}\n\n"
        "Expert to fill (Table 6):\n"
        "  - known_interaction  (BioGRID / IntAct / STRING)\n"
        "  - known_type\n"
        "  - pmid\n"
        "  - consistency\n"
        "  - note\n\n"
        "Claim: known-pair prioritization\n"
        "  (NOT novel PPI discovery)\n\n"
        "[TODO: expert evidence columns]"
    )
    ax_d.text(0.02, 0.98, txt, ha="left", va="top", fontsize=9, family="monospace",
              transform=ax_d.transAxes, linespacing=1.5)

    plt.suptitle("Case study candidates (SITNE-Walk v2, 4-relation system)",
                 fontsize=14, y=0.99)
    fig.savefig(FIG / "figure_8_case_studies.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "figure_8_case_studies.pdf", bbox_inches="tight")
    plt.close()
    logger.info("Figure 8 saved")


def generate_table6(df: pd.DataFrame) -> None:
    """以 case_candidates 为基础，追加 5 个专家待填证据列，输出 Table 6 骨架。"""
    table = df.copy()
    # 保持列顺序：核心列 + 证据占位列
    core_cols = ["pair_key", "head_name", "tail_name", "relation_name",
                 "mean_rank", "min_rank", "std_rank", "n_seeds", "n_folds",
                 "total_appearances"]
    core_cols = [c for c in core_cols if c in table.columns]
    table = table[core_cols].copy()

    # 专家待填列（空）
    table["known_interaction"] = ""   # 是否已知交互 (yes/no/unknown)
    table["known_type"] = ""          # 已知交互类型 (PPI/genetic/physical...)
    table["pmid"] = ""                # 支撑文献 PMID
    table["consistency"] = ""         # 与 gold-standard 是否一致
    table["note"] = ""                # 备注

    out = CASE / "table_6_case_studies.tsv"
    table.to_csv(out, sep="\t", index=False)
    logger.info("Table 6 skeleton saved (%d rows)", len(table))


def main() -> None:
    if not CANDIDATES.exists():
        logger.error("候选清单不存在: %s", CANDIDATES)
        sys.exit(1)
    df = load_candidates(CANDIDATES)
    logger.info("加载候选对: %d 行", len(df))
    generate_figure8(df)
    generate_table6(df)
    logger.info("T11 Figure 8 + Table 6 骨架完成。")


if __name__ == "__main__":
    main()
