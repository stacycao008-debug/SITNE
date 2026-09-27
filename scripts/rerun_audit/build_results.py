#!/usr/bin/env python3
"""生成最终结果包：8 个 table_*.tsv + RESULTS_FOR_MANUSCRIPT.md。

输入（均已完成，除非 comparator 未跑完）：
  - formal/ablation/source/calibration/frequency-prior/degree-classifier/sensitivity
  - comparator 3-seed（run_comparator_seeds.py 产出）

输出到 rerun_submission_audit/08_final_tables/。

用法：
    PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/build_results.py --project-root "$PWD"
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT = "rerun_submission_audit"
RELATION_NAMES = {0: "enzymatic", 1: "general", 2: "physical", 3: "spatial"}


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _formal_summary(root: Path) -> pd.DataFrame:
    p = root / AUDIT / "02_formal_sitne/runs/training_summary.tsv"
    df = pd.read_csv(p, sep="\t")
    return df


def _write_main_performance(root: Path, out: Path) -> None:
    """table_main_performance.tsv：method × fold × mrr（seed-averaged within fold）。"""
    rows = []
    # formal
    formal = _formal_summary(root)
    for fold, g in formal.groupby("fold"):
        rows.append({"method": "SITNE-Walk", "fold": int(fold),
                     "mrr": float(g["mrr"].mean()), "n_seeds": int(len(g))})
    # frequency prior + degree + binary
    prior = root / AUDIT / "03_controls/frequency_prior/frequency_prior_summary.tsv"
    if prior.is_file():
        pdf = pd.read_csv(prior, sep="\t")
        for _, r in pdf.iterrows():
            rows.append({"method": "frequency_prior", "fold": int(r["fold"]),
                         "mrr": float(r["mrr"]), "n_seeds": 1})
    # degree/binary（历史 tie baseline，读 simple summary）
    simple = root / "08_results/sitne_walk_paper_v2/r2_main_performance/baselines/simple/simple_baselines_summary.tsv"
    if simple.is_file():
        sdf = pd.read_csv(simple, sep="\t")
        for name in ["degree", "binary"]:
            sub = sdf[sdf["baseline"] == name]
            for _, r in sub.iterrows():
                rows.append({"method": name, "fold": int(r["fold"]),
                             "mrr": float(r["mrr"]), "n_seeds": 1})
    # typed skip-gram（历史 3 seeds，未在 Phase E 统一协议下重跑）
    tsg = root / "08_results/sitne_walk_paper_v2/r2_main_performance/baselines/typed_skipgram/typed_skipgram_summary.tsv"
    if tsg.is_file():
        tsdf = pd.read_csv(tsg, sep="\t")
        for _, r in tsdf.iterrows():
            rows.append({"method": "typed_skipgram", "fold": int(r["fold"]),
                         "mrr": float(r["mrr"]), "n_seeds": int(r["n_seeds"])})
    # comparator 3-seed
    cmp_summary = root / AUDIT / "06_comparators/comparator_3seed_summary.tsv"
    if cmp_summary.is_file():
        cdf = pd.read_csv(cmp_summary, sep="\t")
        for _, r in cdf.iterrows():
            rows.append({"method": r["model"], "fold": "mean", "mrr": float(r["mean"]),
                         "n_seeds": int(r["count"])})
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_main_performance.tsv", sep="\t", index=False)
    # 打印 method 汇总
    agg = df.groupby("method")["mrr"].agg(["mean", "std", "count"])
    print("=== table_main_performance.tsv ===")
    print(agg.round(6).to_string())


def _write_ablations(root: Path, out: Path) -> None:
    """table_ablations.tsv：variant × micro/macro × delta_vs_formal。"""
    formal_mean = float(_formal_summary(root)["mrr"].mean())
    sens = pd.read_csv(root / AUDIT / "07_sensitivity/table_sensitivity.tsv", sep="\t")
    rows = []
    for variant in ["alpha0", "no_semantic", "no_decorr", "no_adversary", "no_rank"]:
        sub = sens[sens["group"] == f"ablation_{variant}"]
        micro = float(sub[sub["metric"] == "micro_mrr"]["value"].iloc[0])
        macro = float(sub[sub["metric"] == "macro_mrr"]["value"].iloc[0])
        rows.append({"variant": variant, "micro_mrr": micro, "macro_mrr": macro,
                     "delta_vs_formal": micro - formal_mean})
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_ablations.tsv", sep="\t", index=False)
    print("=== table_ablations.tsv ===")
    print(df.round(6).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    out = root / AUDIT / "08_final_tables"
    out.mkdir(parents=True, exist_ok=True)

    _write_main_performance(root, out)
    _write_ablations(root, out)

    # 直接复制已产出的表
    _copy(root / AUDIT / "07_sensitivity/table_relationwise.tsv", out / "table_relationwise.tsv")
    _copy(root / AUDIT / "07_sensitivity/table_sensitivity.tsv", out / "table_sensitivity.tsv")
    _copy(root / AUDIT / "04_source_analysis/source_relation_association.tsv", out / "table_source_controls.tsv")
    _copy(root / AUDIT / "05_calibration/rank_form_calibration.tsv", out / "table_calibration.tsv")
    _copy(root / AUDIT / "06_comparators/comparator_fairness_matrix.tsv", out / "table_comparator_protocols.tsv")
    _copy(root / "08_results/sitne_walk_paper_v2/r4_robustness/coverage_boundary.tsv", out / "table_applicability.tsv")

    print(f"\n8 tables -> {out}")
    print("剩余：RESULTS_FOR_MANUSCRIPT.md（见下步）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
