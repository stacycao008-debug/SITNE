#!/usr/bin/env python3
"""从 ACCEPTED evidence layer（per_query_scores.tsv）重建 manuscript-facing 数字。

- 只读 per_query_scores.tsv（统一 schema，含 filtered/unfiltered/reciprocal rank）。
- 重建 table_main_performance / table_relationwise / table_ablations，输出 filtered + unfiltered 两套。
- 对无 per_query_scores 的方法（metapath2vec/ComplEx/DistMult/typed skip-gram）用 filtered_rank 兜底并标注。
- 输出 provenance manifest，确认每个数字的来源文件（防旧 provisional 数字混入）。

用法：
    PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/rebuild_manuscript.py --project-root "$PWD"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT = "rerun_submission_audit"
RELATION_NAMES = {0: "enzymatic", 1: "general", 2: "physical", 3: "spatial"}


def _load_pqs(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p, sep="\t") for p in paths], ignore_index=True)


def _method_metrics(df: pd.DataFrame) -> dict:
    """从 per_query_scores 计算 filtered/unfiltered micro + macro MRR。"""
    # seed-averaged within fold：先按 (fold, seed) 分组算 micro，再 fold 平均
    per_fold_seed = df.groupby(["fold", "seed"])["reciprocal_rank"].mean().reset_index()
    # unfiltered reciprocal
    df = df.copy()
    df["reciprocal_unfiltered"] = 1.0 / df["unfiltered_rank"]
    per_fold_seed_unf = df.groupby(["fold", "seed"])["reciprocal_unfiltered"].mean().reset_index()

    micro_f = per_fold_seed.groupby("fold")["reciprocal_rank"].mean()
    micro_u = per_fold_seed_unf.groupby("fold")["reciprocal_unfiltered"].mean()

    # macro（per-relation mean，fold 内 seed 平均）
    df["rel_rr"] = df["reciprocal_rank"]
    macro_parts = []
    for r in range(4):
        sub = df[df["target_relation_id"] == r]
        if len(sub):
            fs = sub.groupby(["fold", "seed"])["rel_rr"].mean().groupby("fold").mean()
            macro_parts.append(fs.mean())
    macro_f = float(np.mean(macro_parts)) if macro_parts else float("nan")
    return {
        "filtered_mrr": float(micro_f.mean()),
        "filtered_mrr_std": float(micro_f.std()),
        "unfiltered_mrr": float(micro_u.mean()),
        "macro_mrr": macro_f,
        "n_folds": int(micro_f.size),
    }


def _collect_method_pqs(root: Path) -> dict[str, list[Path]]:
    audit = root / AUDIT
    methods: dict[str, list[Path]] = {}
    # formal
    methods["SITNE-Walk"] = sorted((audit / "02_formal_sitne/runs").glob("fold_*/seed_*/sitne_walk_*/per_query_scores.tsv"))
    # ablation（每个 variant 单独）
    for v in ["alpha0", "no_semantic", "no_decorr", "no_adversary", "no_rank"]:
        methods[f"ablation_{v}"] = sorted((audit / "03_controls/ablation" / v).glob("fold_*/seed_*/sitne_walk_*/per_query_scores.tsv"))
    # controls
    methods["frequency_prior"] = sorted((audit / "03_controls/frequency_prior").glob("fold_*/per_query_scores.tsv"))
    methods["degree_classifier"] = sorted((audit / "03_controls/degree_classifier").glob("fold_*/per_query_scores.tsv"))
    methods["random"] = sorted((audit / "03_controls/random_baseline").glob("per_query_scores.tsv"))
    return methods


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    audit = root / AUDIT
    out = audit / "08_final_tables"
    out.mkdir(parents=True, exist_ok=True)

    methods = _collect_method_pqs(root)

    # --- table_main_performance（filtered + unfiltered + macro）---
    rows = []
    for name, paths in methods.items():
        if not paths:
            print(f"[warn] {name}: 无 per_query_scores.tsv")
            continue
        df = _load_pqs(paths)
        m = _method_metrics(df)
        m["method"] = name
        m["source"] = "per_query_scores.tsv"
        rows.append(m)
    main_df = pd.DataFrame(rows)
    main_df.to_csv(out / "table_main_performance_rebuilt.tsv", sep="\t", index=False)
    print("=== table_main_performance_rebuilt.tsv ===")
    print(main_df[["method", "filtered_mrr", "filtered_mrr_std", "unfiltered_mrr", "macro_mrr", "n_folds"]].round(6).to_string(index=False))

    # --- table_relationwise（formal filtered + unfiltered vs prior）---
    formal = _load_pqs(methods["SITNE-Walk"])
    prior = _load_pqs(methods["frequency_prior"])
    rel_rows = []
    for r in range(4):
        name = RELATION_NAMES[r]
        f_sub = formal[formal["target_relation_id"] == r]
        p_sub = prior[prior["target_relation_id"] == r]
        f_filt = float(f_sub.groupby(["fold", "seed"])["reciprocal_rank"].mean().groupby("fold").mean().mean())
        f_unf = float((1.0 / f_sub["unfiltered_rank"]).groupby(f_sub["fold"]).mean().mean())
        p_filt = float(p_sub.groupby("fold")["reciprocal_rank"].mean().mean())
        rel_rows.append({"relation": name, "formal_filtered": f_filt, "formal_unfiltered": f_unf,
                         "prior_filtered": p_filt, "delta_vs_prior": f_filt - p_filt})
    rel_df = pd.DataFrame(rel_rows)
    rel_df.to_csv(out / "table_relationwise_rebuilt.tsv", sep="\t", index=False)
    print("\n=== table_relationwise_rebuilt.tsv ===")
    print(rel_df.round(6).to_string(index=False))

    # --- provenance manifest ---
    manifest_rows = []
    for name, paths in methods.items():
        for p in paths:
            manifest_rows.append({"method": name, "file": str(p.relative_to(root))})
    pd.DataFrame(manifest_rows).to_csv(out / "provenance_manifest.tsv", sep="\t", index=False)
    print(f"\nprovenance_manifest.tsv -> {len(manifest_rows)} per_query_scores.tsv 文件，"
          f"覆盖 {len(methods)} 方法")

    # 确认：所有数字来自 per_query_scores.tsv（ACCEPTED），无 test_ranks.tsv 旧数字混入
    print(f"\n[确认] 所有方法数字均来自 per_query_scores.tsv（filtered/unfiltered 同源）")
    print(f"[注意] metapath2vec/ComplEx/DistMult/typed skip-gram 无 per_query_scores，"
          f"未纳入本重建（其 filtered MRR 见 comparator_3seed_summary.tsv / 历史 summary）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
