#!/usr/bin/env python3
"""T11 Case Studies — 从 R2 optimal 结果中提取高置信度 protein pairs。

策略:
  1. 加载所有 15 个 optimal/test_ranks.tsv (5 folds × 3 seeds)
  2. 按 pair_key 聚合，计算 mean rank, rank std, 跨 seed 一致性
  3. 筛选 mean_rank ≤ 5 的候选对
  4. 输出 case_candidates.tsv + case_studies_report.md

Usage:
    python scripts/run_case_studies.py
    python scripts/run_case_studies.py --max-rank 10 --min-seeds 2
"""

import argparse
import json, logging, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("case_studies")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "06_code"))


# ======================================================================
# 1. Load all test_ranks.tsv files
# ======================================================================

def load_all_rankings(optimal_dir: Path) -> pd.DataFrame:
    """加载所有 optimal 模型的 test_ranks.tsv 并合并。"""
    rank_files = sorted(optimal_dir.glob("fold_*/seed_*/*/test_ranks.tsv"))
    if not rank_files:
        log.error("未找到任何 test_ranks.tsv 文件 (路径: %s)", optimal_dir)
        return pd.DataFrame()

    log.info("找到 %d 个 test_ranks.tsv 文件", len(rank_files))
    frames = []
    for f in rank_files:
        try:
            df = pd.read_csv(f, sep="\t")
            # Parse fold/seed from path: fold_X/seed_Y/.../test_ranks.tsv
            parts = f.parts
            fold = seed = -1
            for i, p in enumerate(parts):
                if p.startswith("fold_") and p[5:].isdigit():
                    fold = int(p[5:])
                if p.startswith("seed_") and p[5:].isdigit():
                    seed = int(p[5:])
            # Ensure fold/seed columns exist
            if "fold" not in df.columns:
                df["fold"] = fold
            if "seed" not in df.columns:
                df["seed"] = seed
            frames.append(df)
        except Exception as e:
            log.warning("无法加载 %s: %s", f, e)

    if not frames:
        log.error("无法加载任何 test_ranks.tsv")
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    log.info("总共 %d 条排名记录, folds=%s, seeds=%s",
             len(all_df),
             sorted(all_df["fold"].unique()),
             sorted(all_df["seed"].unique()))
    return all_df


# ======================================================================
# 2. Load protein & relation ID mappings
# ======================================================================

def load_mappings(fold: int = 0) -> tuple[dict[int, str], dict[int, str], dict[int, int]]:
    """从训练数据加载 ID→name 映射，并计算每个蛋白的训练集 degree。"""
    from sitne_walk.data import load_training_triples

    train_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
    if not train_path.exists():
        log.warning("训练文件缺失: %s", train_path)
        return {}, {}, {}

    triples = load_training_triples(
        str(train_path), weight_column=None, duplicate_policy="binary", pair_semantics="unordered")
    id_to_protein = {i: name for i, name in enumerate(triples.id_to_protein)}
    id_to_relation = {i: name for i, name in enumerate(triples.id_to_relation)}

    # 计算每个蛋白在训练集中的连接度（unordered pair 语义下，head/tail 均计入）
    degree = defaultdict(int)
    for head, tail in zip(triples.heads, triples.tails):
        degree[int(head)] += 1
        degree[int(tail)] += 1
    return id_to_protein, id_to_relation, dict(degree)


# ======================================================================
# 3. Aggregate by pair_key
# ======================================================================

def aggregate_by_pair(rankings: pd.DataFrame) -> pd.DataFrame:
    """按 (pair_key, relation_id) 聚合，计算跨 fold/seed 统计量。

    保留 relation 维度：同一对蛋白在不同 relation 下（不同交互类型）视为独立候选，
    不强制合并，避免 physical 关系因 mode 聚合被放大。
    """
    if "pair_key" not in rankings.columns:
        rankings["pair_key"] = rankings.apply(
            lambda r: f"{min(r.head_id, r.tail_id)}_{max(r.head_id, r.tail_id)}", axis=1)

    grouped = rankings.groupby(["pair_key", "relation_id"]).agg(
        head_id=("head_id", "first"),
        tail_id=("tail_id", "first"),
        mean_rank=("filtered_rank", "mean"),
        median_rank=("filtered_rank", "median"),
        min_rank=("filtered_rank", "min"),
        max_rank=("filtered_rank", "max"),
        std_rank=("filtered_rank", "std"),
        n_seeds=("seed", "nunique"),
        n_folds=("fold", "nunique"),
        total_appearances=("filtered_rank", "count"),
    ).reset_index()

    # Handle NaN std (single appearance)
    grouped["std_rank"] = grouped["std_rank"].fillna(0.0)
    grouped["cv_rank"] = grouped["std_rank"] / grouped["mean_rank"].replace(0, np.nan)
    grouped["cv_rank"] = grouped["cv_rank"].fillna(0.0)

    log.info("Aggregated: %d unique pairs from %d records",
             len(grouped), len(rankings))
    return grouped


# 分层阈值：不同 relation 的模型排序能力差异大（如 physical 类 MRR≈0.994，
# 几乎全部 rank=1），因此对高 MRR 的 relation 收紧阈值，对低 MRR 的 relation 放宽。
# relation_id 语义（typed_ranking_v2 的 4 类 coarse relation）:
#   0=enzymatic, 1=general, 2=physical, 3=spatial
RELATION_THRESHOLDS = {
    # 低 MRR 关系（排序能力较弱）：保留 mean_rank≤5
    0: {"max_rank": 5.0, "max_std": None},   # enzymatic
    1: {"max_rank": 5.0, "max_std": None},   # general
    # 高 MRR 关系（排序能力极强）：mean_rank==1 且 std_rank==0（跨 fold/seed 完全一致）
    2: {"max_rank": 1.0, "max_std": 0.0},    # physical
    3: {"max_rank": 1.0, "max_std": 0.0},    # spatial
}


def filter_candidates(pairs: pd.DataFrame, max_rank: float = 5.0,
                      min_seeds: int = 2) -> pd.DataFrame:
    """筛选候选案例对，按 relation 分层收紧阈值。

    - 全局门槛: n_seeds >= min_seeds
    - enzymatic/general: mean_rank <= 5.0（模型排序较弱，放宽）
    - physical/spatial:   mean_rank == 1.0 且 std_rank == 0.0（模型排序极强，收紧）
    """
    cands = pairs[pairs["n_seeds"] >= min_seeds].copy()

    masks = []
    for rel_id, th in RELATION_THRESHOLDS.items():
        m = cands["relation_id"] == rel_id
        m &= cands["mean_rank"] <= th["max_rank"]
        if th["max_std"] is not None:
            m &= cands["std_rank"] <= th["max_std"]
        masks.append(m)

    # 未知 relation_id（不在 0-3 内）: 回退到全局 mean_rank≤max_rank
    fallback = ~cands["relation_id"].isin(list(RELATION_THRESHOLDS.keys()))
    fallback &= cands["mean_rank"] <= max_rank

    candidates = cands[np.logical_or.reduce(masks) | fallback].copy()
    candidates.sort_values(["relation_id", "mean_rank", "std_rank"], inplace=True)

    for rel_id, th in RELATION_THRESHOLDS.items():
        n = int((candidates["relation_id"] == rel_id).sum())
        log.info("  relation %d: mean_rank≤%.1f%s → %d 候选对",
                 rel_id, th["max_rank"],
                 f" & std_rank≤{th['max_std']}" if th["max_std"] is not None else "",
                 n)
    log.info("筛选 n_seeds≥%d + 分层阈值 → 共 %d 候选对", min_seeds, len(candidates))
    return candidates


# 每类 top-k 截断上限：physical/spatial 的 rank 无区分度（几乎全为 rank=1），
# 需截断到可人工查证的规模；enzymatic/general 数量本身已很少，取全部。
RELATION_TOP_K = {
    0: None,   # enzymatic  — 取全部（327）
    1: None,   # general    — 取全部（653）
    2: 50,     # physical   — 取前 50
    3: 50,     # spatial    — 取前 50
}


def truncate_topk(candidates: pd.DataFrame,
                  degree: dict[int, int]) -> pd.DataFrame:
    """对每类 relation 做 top-k 截断。

    rank 无区分度的关系（physical/spatial）用辅助指标排序：
      1) total_appearances 降序（跨 fold/seed 被测试到的总次数，越高越一致）
      2) 平均蛋白 degree 降序（hub 蛋白的预测更有生物学意义）
    enzymatic/general 保留全部，仅按 mean_rank 升序。
    """
    kept = []
    for rel_id, k in RELATION_TOP_K.items():
        sub = candidates[candidates["relation_id"] == rel_id].copy()
        if sub.empty:
            continue
        if k is None:
            sub = sub.sort_values(["mean_rank", "std_rank", "total_appearances"],
                                  ascending=[True, True, False])
        else:
            sub["avg_degree"] = (
                sub["head_id"].map(degree).fillna(0)
                + sub["tail_id"].map(degree).fillna(0)
            ) / 2.0
            sub = sub.sort_values(
                ["total_appearances", "avg_degree", "head_id"],
                ascending=[False, False, True])
            sub = sub.head(k)
        kept.append(sub)

    if not kept:
        return candidates.iloc[0:0].copy()

    result = pd.concat(kept, ignore_index=True)
    result.sort_values(["relation_id", "mean_rank", "std_rank"], inplace=True)
    for rel_id, k in RELATION_TOP_K.items():
        n = int((result["relation_id"] == rel_id).sum())
        log.info("  relation %d: top-k 截断后 → %d 候选对 (k=%s)", rel_id, n, k)
    log.info("top-k 截断后 → 共 %d 候选对", len(result))
    return result


# ======================================================================
# 4. Add protein names
# ======================================================================

def annotate_pairs(candidates: pd.DataFrame,
                   id_to_protein: dict[int, str],
                   id_to_relation: dict[int, str]) -> pd.DataFrame:
    """为候选对添加可读的蛋白/关系名称。"""
    df = candidates.copy()
    df["head_name"] = df["head_id"].map(id_to_protein)
    df["tail_name"] = df["tail_id"].map(id_to_protein)
    df["relation_name"] = [id_to_relation.get(int(x), str(x)) for x in df["relation_id"]]
    return df


# ======================================================================
# 5. Generate markdown report
# ======================================================================

def generate_report(
    candidates: pd.DataFrame,
    id_to_relation: dict[int, str],
    output_md: Path,
) -> str:
    """生成包含候选案例表和生物学解释占位的 markdown 报告。"""
    lines = []
    lines.append("# SITNE-Walk 案例研究报告")
    lines.append("")
    lines.append("> 本文档由 `scripts/run_case_studies.py` 自动生成。")
    lines.append("> 候选对筛选自 R2 optimal 模型中跨 5 folds × 3 seeds 一致的高排位 protein pairs。")
    lines.append("")
    lines.append("## 筛选条件")
    lines.append("")
    lines.append(f"- 总候选对数: {len(candidates)}")
    lines.append(f"- 主要指标: 跨 seed ≥ 2，按 relation 分层阈值（enzymatic/general: mean rank ≤ 5；physical/spatial: mean rank == 1 且 std == 0）")
    lines.append(f"- top-k 截断: enzymatic/general 取全部；physical/spatial 各取前 50（按 cross-fold 一致性与蛋白 degree 排序）")
    lines.append(f"- 数据来源: `08_results/sitne_walk_paper_v2/r2_main_performance/optimal/fold_*/seed_*/*/test_ranks.tsv`")
    lines.append("")

    # Summary statistics
    lines.append("## 摘要")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 候选对数 | {len(candidates)} |")
    if not candidates.empty:
        lines.append(f"| Mean rank 范围 | {candidates['mean_rank'].min():.2f} – {candidates['mean_rank'].max():.2f} |")
        lines.append(f"| 涉及关系类型数 | {candidates['relation_id'].nunique()} |")
        lines.append(f"| 涉及蛋白数 | {len(set(candidates['head_id'])|set(candidates['tail_id']))} |")
    lines.append("")

    # Candidate table
    lines.append("## 候选案例对列表")
    lines.append("")
    cols = ["pair_key", "head_name", "tail_name", "relation_name",
            "mean_rank", "min_rank", "std_rank", "n_seeds", "n_folds"]
    available = [c for c in cols if c in candidates.columns]

    header = "| " + " | ".join(available) + " |"
    sep = "|" + "|".join([" --- " for _ in available]) + "|"
    lines.append(header)
    lines.append(sep)

    for _, row in candidates.head(50).iterrows():
        vals = []
        for c in available:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("> 以上展示了前 50 对。完整列表见 `case_candidates.tsv`。")

    # Per-relation sections
    lines.append("")
    lines.append("## 按关系类型分组的候选案例")
    lines.append("")

    if not candidates.empty:
        for rel_id in sorted(candidates["relation_id"].unique()):
            rel_name = id_to_relation.get(int(rel_id), f"rel_{int(rel_id)}")
            lines.append(f"### {rel_name} (id={int(rel_id)})")
            lines.append("")

            sub = candidates[candidates["relation_id"] == rel_id]
            sub_sorted = sub.sort_values("mean_rank").head(5)
            for _, row in sub_sorted.iterrows():
                head_n = row.get("head_name", f"protein_{int(row['head_id'])}")
                tail_n = row.get("tail_name", f"protein_{int(row['tail_id'])}")
                lines.append(f"- **{head_n}** ↔ **{tail_n}**: "
                            f"mean rank = {row['mean_rank']:.2f} "
                            f"(min={row['min_rank']:.1f}, "
                            f"n_seeds={int(row['n_seeds'])}/{int(row['n_folds'])})")
                lines.append(f"  - 🖊 *[TODO: 添加生物学已知交互类型]*")
                lines.append(f"  - 🖊 *[TODO: 添加文献引用 PMID]*")
                lines.append(f"  - 🖊 *[TODO: 添加注释：该预测是否与已知 gold-standard 一致]*")
            lines.append("")

    # Discussion template
    lines.append("## 讨论")
    lines.append("")
    lines.append("🖊 *[TODO: 总结跨关系类型的高置信度模式]*")
    lines.append("")
    lines.append("🖊 *[TODO: 指出预测与已知生物学知识一致或不一致的案例]*")
    lines.append("")
    lines.append("🖊 *[TODO: 讨论 SITNE-Walk embedding 在案例上的可解释性]*")
    lines.append("")
    lines.append("## 参考文献")
    lines.append("")
    lines.append("🖊 *[TODO: 参考文献列表]*")
    lines.append("")

    report = "\n".join(lines)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w") as f:
        f.write(report)
    log.info("报告已保存: %s", output_md)
    return report


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="SITNE-Walk case studies")
    parser.add_argument("--max-rank", type=float, default=5.0,
                        help="候选对最大 mean rank (默认 5.0)")
    parser.add_argument("--min-seeds", type=int, default=2,
                        help="最少跨 seed 数 (默认 2)")
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "08_results/sitne_walk_paper_v2/case_studies"),
                        help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    optimal_dir = ROOT / "08_results/sitne_walk_paper_v2/r2_main_performance/optimal"

    log.info("=" * 60)
    log.info("T11 Case Studies: SITNE-Walk 最优模型案例分析")
    log.info("=" * 60)

    # Step 1: Load all rankings
    log.info("Step 1: 加载所有 test_ranks.tsv...")
    rankings = load_all_rankings(optimal_dir)
    if rankings.empty:
        log.error("无数据可分析，退出")
        sys.exit(1)

    # Step 2: Load ID mappings
    log.info("Step 2: 加载蛋白/关系 ID 映射...")
    id_to_protein, id_to_relation, degree = load_mappings(fold=0)
    log.info("  Proteins: %d, Relations: %d", len(id_to_protein), len(id_to_relation))

    # Step 3: Aggregate
    log.info("Step 3: 按 pair_key 聚合...")
    aggregated = aggregate_by_pair(rankings)

    # Step 4: Filter candidates
    log.info("Step 4: 筛选候选对...")
    candidates = filter_candidates(aggregated, args.max_rank, args.min_seeds)

    # Step 4b: Truncate to top-k per relation
    log.info("Step 4b: 按 relation top-k 截断...")
    candidates = truncate_topk(candidates, degree)

    # Step 5: Annotate
    log.info("Step 5: 添加蛋白名称...")
    candidates = annotate_pairs(candidates, id_to_protein, id_to_relation)

    # Step 6: Save TSV
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "case_candidates.tsv"
    candidates.to_csv(tsv_path, sep="\t", index=False)
    log.info("候选表已保存: %s (%d 行)", tsv_path, len(candidates))

    # Step 7: Generate markdown report
    log.info("Step 6: 生成 markdown 报告...")
    md_path = output_dir / "case_studies_report.md"
    generate_report(candidates, id_to_relation, md_path)

    # Summary
    log.info("=" * 60)
    log.info("T11 Case Studies 完成")
    log.info("  候选对总数: %d", len(candidates))
    if not candidates.empty:
        log.info("  Top 5 pairs:")
        for _, row in candidates.head(5).iterrows():
            log.info("    %s ↔ %s  rank=%.2f ± %.2f (n_seeds=%d)",
                     row["head_name"], row["tail_name"],
                     row["mean_rank"], row["std_rank"], int(row["n_seeds"]))
    log.info("  输出文件:")
    log.info("    %s", tsv_path)
    log.info("    %s", md_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
