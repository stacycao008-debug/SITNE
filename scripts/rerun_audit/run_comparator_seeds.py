#!/usr/bin/env python3
"""Comparator 补 3 seeds 编排：metapath2vec / ComplEx / DistMult。

对每个 baseline 补 seed 123/456（seed 42 已在历史 v2 路径），输出到
rerun_submission_audit/06_comparators/。串行调用现有 baseline 脚本（--all-folds）。
跑完后把 3-seed（历史 seed_42 + 新 seed_123/456）汇总为 mean±std。

用法（GPU 0）：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/run_comparator_seeds.py --project-root "$PWD" \
        --output-root rerun_submission_audit/06_comparators

--dry-run 仅打印命令不执行。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASELINES = {
    "metapath2vec": "scripts/run_baseline_metapath2vec.py",
    "complex": "scripts/run_baseline_complex.py",
    "distmult": "scripts/run_baseline_distmult.py",
}
SEEDS = [123, 456]
HISTORICAL_ROOT = "08_results/sitne_walk_paper_v2/r2_main_performance/baselines"


def _run(root: Path, script: str, seed: int, output_root: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, str(root / script),
            "--all-folds", "--seed", str(seed),
            "--output-root", output_root,
        ],
        env={**__import__("os").environ, "PYTHONPATH": str(root / "06_code"),
             "CUDA_VISIBLE_DEVICES": "0"},
        capture_output=True, text=True, check=False,
    )


def _read_mrr(metrics_path: Path) -> float | None:
    if not metrics_path.is_file():
        return None
    try:
        return float(json.loads(metrics_path.read_text())["metrics"]["mrr"])
    except (KeyError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output_root = args.output_root
    historical_root = HISTORICAL_ROOT

    # 1. 补 seed 123/456
    total = len(BASELINES) * len(SEEDS)
    current = 0
    failures = []
    for name, script in BASELINES.items():
        for seed in SEEDS:
            current += 1
            if args.dry_run:
                print(f"[{current}/{total}] dry-run {name} seed={seed} "
                      f"--output-root {output_root}")
                continue
            print(f"[{current}/{total}] {name} seed={seed} ...", flush=True)
            result = _run(root, script, seed, output_root)
            ok = result.returncode == 0
            if not ok:
                failures.append((name, seed, result.stderr[-300:]))
                print(f"  FAILED: {result.stderr[-200:]}")
            else:
                print(f"  done (exit=0)")

    # 2. 汇总 3-seed（历史 seed_42 + 新 seed_123/456）
    rows = []
    for name in BASELINES:
        for fold in range(5):
            mrrs = {}
            for seed in [42, 123, 456]:
                if seed == 42:
                    p = root / historical_root / name / f"fold_{fold}" / f"seed_{seed}" / "test_metrics.json"
                else:
                    p = root / output_root / name / f"fold_{fold}" / f"seed_{seed}" / "test_metrics.json"
                m = _read_mrr(p)
                if m is not None:
                    mrrs[seed] = m
            for seed, mrr in mrrs.items():
                rows.append({"model": name, "fold": fold, "seed": seed, "mrr": mrr})

    if not args.dry_run and rows:
        df = pd.DataFrame(rows)
        # seed-averaged within fold → fold-level summary
        fold_sum = df.groupby(["model", "fold"])["mrr"].mean().reset_index()
        agg = df.groupby("model")["mrr"].agg(["mean", "std", "count"]).reset_index()
        summary_path = root / output_root / "comparator_3seed_summary.tsv"
        agg.to_csv(summary_path, sep="\t", index=False)
        print(f"3-seed summary -> {summary_path}")
        for _, r in agg.iterrows():
            print(f"  {r['model']}: mean={r['mean']:.6f} std={r['std']:.6f} n={int(r['count'])}")

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for name, seed, err in failures:
            print(f"  - {name} seed={seed}: {err[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
