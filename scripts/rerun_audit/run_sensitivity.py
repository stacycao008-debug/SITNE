#!/usr/bin/env python3
"""Sensitivity 全套汇总：Micro/Macro/relation-wise/pair-multiplicity/delta-vs-prior/strata。

输入（均已完成）：
  - formal 15 runs：rerun_submission_audit/02_formal_sitne/runs/fold_X/seed_Y/sitne_walk_*/test_ranks.tsv
  - ablation 25 runs：rerun_submission_audit/03_controls/ablation/<variant>/fold_X/seed_42/sitne_walk_*/test_ranks.tsv
  - frequency prior：rerun_submission_audit/03_controls/frequency_prior/fold_X/frequency_prior_ranks.tsv

统计口径：
  - formal：seed-averaged within fold，再 5-fold 平均（mean±std over folds）
  - ablation：seed=42，明确标注 1-seed 限制
  - strata（degree/frequency）：train-only 统计，per-fold 计算

输出：
  - 07_sensitivity/table_relationwise.tsv（method × relation × mrr × delta_vs_prior）
  - 07_sensitivity/table_sensitivity.tsv（group × metric × value × n）

用法：
    PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/run_sensitivity.py --project-root "$PWD"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_code"))
from sitne_walk.data import load_training_triples  # noqa: E402

RELATION_NAMES = {0: "enzymatic", 1: "general", 2: "physical", 3: "spatial"}


def _load_ranks(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["rr"] = 1.0 / df["filtered_rank"].astype(float)
    return df


def _train_degree_freq(root: Path, fold: int):
    triples = load_training_triples(
        str(root / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"),
        duplicate_policy="binary",
    )
    num_proteins = triples.num_proteins
    heads = triples.heads.numpy()
    tails = triples.tails.numpy()
    rels = triples.relations.numpy()
    degree = np.zeros(num_proteins, dtype=np.float64)
    mask = heads != tails
    pair_codes = heads[mask] * np.int64(num_proteins) + tails[mask]
    unique = np.unique(pair_codes, return_index=True)[1]
    np.add.at(degree, heads[mask][unique], 1.0)
    np.add.at(degree, tails[mask][unique], 1.0)
    freq = np.bincount(rels, minlength=triples.num_relations).astype(np.float64)
    return degree, freq, num_proteins


def _summarize(df: pd.DataFrame, degree=None, freq=None, num_proteins=None) -> dict:
    out = {}
    out["micro_mrr"] = float(df["rr"].mean())
    per_rel = df.groupby("relation_id")["rr"].mean()
    out["macro_mrr"] = float(per_rel.mean())
    out["n_queries"] = int(len(df))

    # pair-multiplicity
    h = df["head_id"].to_numpy()
    t = df["tail_id"].to_numpy()
    lo = np.minimum(h, t)
    hi = np.maximum(h, t)
    pair_key = lo * np.int64(num_proteins if num_proteins else int(hi.max()) + 1) + hi
    n_types = df.assign(pk=pair_key).groupby("pk")["relation_id"].nunique()
    df2 = df.copy()
    df2["pk"] = pair_key
    df2["n_types"] = df2["pk"].map(n_types)
    out["single_type_mrr"] = float(df2[df2["n_types"] == 1]["rr"].mean()) if (df2["n_types"] == 1).any() else float("nan")
    out["multi_type_mrr"] = float(df2[df2["n_types"] > 1]["rr"].mean()) if (df2["n_types"] > 1).any() else float("nan")

    # strata
    if degree is not None:
        max_deg = np.maximum(degree[h.astype(int)], degree[t.astype(int)])
        buckets = [(0, 10), (10, 100), (100, 1000), (1000, np.inf)]
        for lo_b, hi_b in buckets:
            mask = (max_deg >= lo_b) & (max_deg < hi_b)
            if mask.any():
                out[f"degree_strata_{lo_b}_{hi_b if hi_b != np.inf else 'inf'}"] = float(df["rr"].to_numpy()[mask].mean())
                out[f"degree_strata_{lo_b}_{hi_b if hi_b != np.inf else 'inf'}_n"] = int(mask.sum())
    if freq is not None:
        f = freq[df["relation_id"].to_numpy().astype(int)]
        buckets = [(1, 2), (2, 10), (10, 100), (100, 1000), (1000, np.inf)]
        for lo_b, hi_b in buckets:
            mask = (f >= lo_b) & (f < hi_b)
            if mask.any():
                out[f"freq_strata_{lo_b}_{hi_b if hi_b != np.inf else 'inf'}"] = float(df["rr"].to_numpy()[mask].mean())
                out[f"freq_strata_{lo_b}_{hi_b if hi_b != np.inf else 'inf'}_n"] = int(mask.sum())
    return out


def _relationwise(df: pd.DataFrame) -> dict[int, float]:
    return {int(r): float(v) for r, v in df.groupby("relation_id")["rr"].mean().items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    out = root / "rerun_submission_audit" / "07_sensitivity"
    out.mkdir(parents=True, exist_ok=True)

    # --- frequency prior（deterministic）---
    prior_relationwise = {}
    prior_fold_summary = []
    for fold in range(5):
        p = root / "rerun_submission_audit/03_controls/frequency_prior" / f"fold_{fold}" / "frequency_prior_ranks.tsv"
        df = _load_ranks(p)
        degree, freq, np_ = _train_degree_freq(root, fold)
        prior_fold_summary.append(_summarize(df, degree, freq, np_))
        for r, v in _relationwise(df).items():
            prior_relationwise.setdefault(r, []).append(v)
    prior_relationwise = {r: float(np.mean(v)) for r, v in prior_relationwise.items()}

    # --- formal 15 runs（seed-averaged within fold）---
    formal_rows = []
    formal_fold_relationwise = {fold: [] for fold in range(5)}
    for fold in range(5):
        fold_dir = root / "rerun_submission_audit/02_formal_sitne/runs" / f"fold_{fold}"
        degree, freq, np_ = _train_degree_freq(root, fold)
        seed_ranks = []
        for seed in [42, 123, 456]:
            ranks = list(fold_dir.glob(f"seed_{seed}/sitne_walk_*/test_ranks.tsv"))
            if not ranks:
                continue
            seed_ranks.append(_load_ranks(ranks[0]))
        # seed-averaged：先算每个 seed 的 summary，再平均（用 fold 级 rr 平均近似）
        for sdf in seed_ranks:
            formal_fold_relationwise[fold].append(_relationwise(sdf))
        merged = pd.concat(seed_ranks, ignore_index=True)
        s = _summarize(merged, degree, freq, np_)
        s["fold"] = fold
        formal_rows.append(s)

    # formal relation-wise：fold 内 seed 平均 → 5 fold 平均
    formal_relationwise = {}
    for fold in range(5):
        for r, v in {r: float(np.mean([d[r] for d in formal_fold_relationwise[fold] if r in d])) for r in range(4)}.items():
            formal_relationwise.setdefault(r, []).append(v)
    formal_relationwise = {r: float(np.mean(v)) for r, v in formal_relationwise.items()}

    # --- ablation 25 runs ---
    ablation_rows = []
    ablation_relationwise = {}
    for variant in ["alpha0", "no_semantic", "no_decorr", "no_adversary", "no_rank"]:
        vardir = root / "rerun_submission_audit/03_controls/ablation" / variant
        rel_acc = {r: [] for r in range(4)}
        for fold in range(5):
            ranks = list(vardir.glob(f"fold_{fold}/seed_42/sitne_walk_*/test_ranks.tsv"))
            if not ranks:
                continue
            df = _load_ranks(ranks[0])
            for r, v in _relationwise(df).items():
                rel_acc[r].append(v)
        s = {"variant": variant}
        # fold 平均 micro
        micros = []
        for fold in range(5):
            ranks = list(vardir.glob(f"fold_{fold}/seed_42/sitne_walk_*/test_ranks.tsv"))
            if ranks:
                micros.append(float(_load_ranks(ranks[0])["rr"].mean()))
        s["micro_mrr"] = float(np.mean(micros))
        s["macro_mrr"] = float(np.mean([np.mean(rel_acc[r]) for r in range(4) if rel_acc[r]]))
        ablation_rows.append(s)
        ablation_relationwise[variant] = {r: float(np.mean(rel_acc[r])) for r in range(4) if rel_acc[r]}

    # --- 输出 table_relationwise.tsv ---
    rel_rows = []
    for r in range(4):
        name = RELATION_NAMES.get(r, str(r))
        formal_mrr = formal_relationwise.get(r, float("nan"))
        prior_mrr = prior_relationwise.get(r, float("nan"))
        rel_rows.append({
            "relation": name,
            "formal_mrr": formal_mrr,
            "prior_mrr": prior_mrr,
            "delta_vs_prior": formal_mrr - prior_mrr,
        })
    rel_df = pd.DataFrame(rel_rows)
    rel_df.to_csv(out / "table_relationwise.tsv", sep="\t", index=False)
    print(f"relationwise -> {out / 'table_relationwise.tsv'}")
    print(rel_df.to_string(index=False))

    # --- 输出 table_sensitivity.tsv ---
    sens_rows = []
    # formal aggregate
    formal_agg = pd.DataFrame(formal_rows)
    for key in ["micro_mrr", "macro_mrr", "single_type_mrr", "multi_type_mrr"]:
        if key in formal_agg.columns:
            sens_rows.append({"group": "formal", "metric": key,
                              "value": float(formal_agg[key].mean()),
                              "n": "5 folds"})
    for r, v in formal_relationwise.items():
        sens_rows.append({"group": "formal", "metric": f"relationwise_{RELATION_NAMES.get(r, r)}",
                          "value": v, "n": "5 folds"})
    # ablation
    for s in ablation_rows:
        sens_rows.append({"group": f"ablation_{s['variant']}", "metric": "micro_mrr",
                          "value": s["micro_mrr"], "n": "1 seed"})
        sens_rows.append({"group": f"ablation_{s['variant']}", "metric": "macro_mrr",
                          "value": s["macro_mrr"], "n": "1 seed"})
    # strata (formal, fold-averaged)
    for key in formal_agg.columns:
        if key.startswith("degree_strata") or key.startswith("freq_strata"):
            if key.endswith("_n"):
                continue
            sens_rows.append({"group": "formal", "metric": key,
                              "value": float(formal_agg[key].mean()), "n": "5 folds"})
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(out / "table_sensitivity.tsv", sep="\t", index=False)
    print(f"sensitivity -> {out / 'table_sensitivity.tsv'} ({len(sens_rows)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
