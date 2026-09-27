#!/usr/bin/env python3
"""T10 (续): 生成 Figure 7 (生物学解释) + Table 5 (biological enrichment).

输入:
  - 08_results/sitne_walk_paper_v2/biology/enrichment/go_enrichment.tsv
  - 08_results/sitne_walk_paper_v2/biology/enrichment/reactome_enrichment.tsv
  - 08_results/sitne_walk_paper_v2/biology/enrichment/relation_similarity.tsv
  - 02_data_canonical/external_annotations/go-basic.obo  (GO term -> name)

输出:
  - 08_results/sitne_walk_paper_v2/figures/figure_7_biological_interpretation.{png,pdf}
  - 08_results/sitne_walk_paper_v2/figures/table_5_biological_enrichment.tsv
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
logger = logging.getLogger("figure7")

ROOT = Path(__file__).resolve().parents[1]
ENRICH = ROOT / "08_results/sitne_walk_paper_v2/biology/enrichment"
FIG = ROOT / "08_results/sitne_walk_paper_v2/figures"
FIG.mkdir(parents=True, exist_ok=True)
OBO = ROOT / "02_data_canonical/external_annotations/go-basic.obo"
REACTOME_ANN = ROOT / "02_data_canonical/external_annotations/reactome_annotation.tsv"

RELATION_NAMES = {0: "enzymatic", 1: "general", 2: "physical", 3: "spatial"}
FDR_CUTOFF = 0.05
TOP_N = 15  # 每个 relation 展示的 top 显著 term 数

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 150})


def load_go_names(obo_path: Path) -> dict[str, str]:
    """从 go-basic.obo 提取 GO:xxxxxxx -> name 映射。"""
    names: dict[str, str] = {}
    if not obo_path.exists():
        logger.warning("GO obo 不存在: %s，GO term 名称将使用 ID 原样", obo_path)
        return names
    cur_id = None
    with open(obo_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur_id = None
            elif line.startswith("id: "):
                cur_id = line[4:].strip()
            elif line.startswith("name: ") and cur_id:
                names[cur_id] = line[6:].strip()
                cur_id = None
    return names


def load_enrichment(path: Path, name_map: dict[str, str]) -> pd.DataFrame:
    """加载富集结果，仅保留 per_relation 来源，映射 relation 名与 term 名。"""
    df = pd.read_csv(path, sep="\t", dtype=str)
    if df.empty:
        return df
    df = df[df["source"] == "per_relation"].copy()
    df["relation_id"] = df["relation_id"].astype(float).astype(int)
    df["relation"] = df["relation_id"].map(RELATION_NAMES)
    df["FDR"] = df["FDR"].astype(float)
    df["overlap"] = df["overlap"].astype(int)
    df["foreground_total"] = df["foreground_total"].astype(int)
    df["background_total"] = df["background_total"].astype(int)
    df["term_name"] = df["term"].map(lambda t: name_map.get(t, t))
    return df


def top_terms_per_relation(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """每个 relation 取 FDR 最小的 top_n 个显著 term（仅 FDR<cutoff）。"""
    sig = df[df["FDR"] < FDR_CUTOFF]
    if sig.empty:
        return sig
    # 每个 relation 取 FDR 最小的 top_n 个（保留 relation 列）
    kept = []
    for rel, g in sig.groupby("relation", sort=False):
        kept.append(g.head(top_n))
    return pd.concat(kept, ignore_index=True) if kept else sig.iloc[0:0]


def generate_figure7(go_df: pd.DataFrame, react_df: pd.DataFrame,
                     sim: pd.DataFrame) -> None:
    rel_names = [RELATION_NAMES[i] for i in range(len(sim))]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.25)

    def _bar_panel(ax, df, title, cmap, rel_names, n_show=8):
        """水平条形图: 每个 relation 取 n_show 个 top term。"""
        if df.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(title)
            ax.set_axis_off()
            return
        blocks = []
        for rel in rel_names:
            sub = df[df["relation"] == rel].sort_values("FDR").head(n_show)
            if sub.empty:
                blocks.append((rel, []))
                continue
            blocks.append((rel, list(sub.itertuples(index=False))))
        # 布局: y 从下往上，每个 block 自底向上堆叠 term，块间留白
        y_ticks, y_labels, y_bars = [], [], []
        y = 0
        block_centers = []
        for rel, rows in blocks:
            block_start_y = y
            for r in rows:
                # term 截断以适配宽度
                name = (getattr(r, "term_name", "") or "")[:55]
                y_ticks.append(y)
                y_labels.append(name)
                y_bars.append((y, -np.log10(max(r.FDR, 1e-300)), rel))
                y += 1
            if rows:
                block_centers.append((rel, (block_start_y + y - 1) / 2))
            y += 1  # 块间留白
        if not y_bars:
            ax.text(0.5, 0.5, "no significant terms", ha="center", va="center")
            ax.set_title(title)
            ax.set_axis_off()
            return
        ys = [b[0] for b in y_bars]
        vs = [b[1] for b in y_bars]
        rels = [b[2] for b in y_bars]
        rel_color = {r: i for i, r in enumerate(rel_names)}
        colors = [plt.get_cmap(cmap)(rel_color[r]) for r in rels]
        ax.barh(ys, vs, color=colors, edgecolor="grey", linewidth=0.4)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_xlabel("-log10(FDR)")
        ax.set_title(title)
        # 在右侧标 relation 块中心
        for rel, cy in block_centers:
            ax.text(-max(vs) * 0.02, cy, rel, ha="right", va="center",
                    fontsize=9, fontweight="bold")

    go_top = top_terms_per_relation(go_df, TOP_N)
    react_top = top_terms_per_relation(react_df, TOP_N)
    ax_a = fig.add_subplot(gs[0, :])
    _bar_panel(ax_a, go_top, "A. GO enrichment (top 8 terms per relation, FDR<0.05)",
               cmap="viridis", rel_names=rel_names, n_show=8)
    ax_b = fig.add_subplot(gs[1, 0])
    _bar_panel(ax_b, react_top, "B. Reactome enrichment (top 8 per relation)",
               cmap="magma", rel_names=rel_names, n_show=8)

    # Panel C: relation embedding 相似度热图
    ax_c = fig.add_subplot(gs[1, 1])
    sns.heatmap(sim, ax=ax_c, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                xticklabels=rel_names, yticklabels=rel_names,
                cbar_kws={"label": "cosine similarity"})
    ax_c.set_title("C. Relation embedding similarity (4×4)")

    plt.suptitle("Biological interpretation of semantic relation representations",
                 fontsize=14, y=0.99)
    fig.savefig(FIG / "figure_7_biological_interpretation.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "figure_7_biological_interpretation.pdf",
                bbox_inches="tight")
    plt.close()
    logger.info("Figure 7 saved")


def generate_table5(go_df: pd.DataFrame, react_df: pd.DataFrame) -> None:
    """汇总 FDR<0.05 的显著项为 Table 5。"""
    rows = []
    for label, df in [("GO", go_df), ("Reactome", react_df)]:
        if df.empty:
            continue
        sig = df[df["FDR"] < FDR_CUTOFF].copy()
        for rel in RELATION_NAMES.values():
            sub = sig[sig["relation"] == rel].sort_values("FDR")
            if sub.empty:
                rows.append({
                    "annotation": label, "relation": rel,
                    "n_significant": 0, "top_term": "", "top_term_name": "",
                    "top_FDR": "", "top_overlap": "",
                })
                continue
            top = sub.iloc[0]
            rows.append({
                "annotation": label, "relation": rel,
                "n_significant": len(sub),
                "top_term": top["term"], "top_term_name": top["term_name"],
                "top_FDR": "%.3g" % top["FDR"], "top_overlap": top["overlap"],
            })
    table = pd.DataFrame(rows)
    table.to_csv(FIG / "table_5_biological_enrichment.tsv", sep="\t", index=False)
    logger.info("Table 5 saved (%d rows)", len(table))
    logger.info("\n%s", table.to_string(index=False))


def load_reactome_names(ann_path: Path) -> dict[str, str]:
    """从 reactome_annotation.tsv 提取 pathway_id -> pathway_name。"""
    if not ann_path.exists():
        return {}
    df = pd.read_csv(ann_path, sep="\t", dtype=str)
    return dict(zip(df["pathway_id"], df["pathway_name"]))


def main() -> None:
    go_names = load_go_names(OBO)
    logger.info("加载 GO term 名称: %d 个", len(go_names))
    reactome_names = load_reactome_names(REACTOME_ANN)
    logger.info("加载 Reactome pathway 名称: %d 个", len(reactome_names))

    go_df = load_enrichment(ENRICH / "go_enrichment.tsv", go_names)
    react_df = load_enrichment(ENRICH / "reactome_enrichment.tsv", reactome_names)
    sim = pd.read_csv(ENRICH / "relation_similarity.tsv", sep="\t", index_col=0)

    generate_figure7(go_df, react_df, sim)
    generate_table5(go_df, react_df)
    logger.info("T10 Figure 7 + Table 5 完成。")


if __name__ == "__main__":
    main()
