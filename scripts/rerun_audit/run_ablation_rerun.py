#!/usr/bin/env python3
"""关键 ablation 重跑：基于 formal_v2（bfloat16 + fail_fast）。

变体（对应 P0-3/P1 的 5 个关键 ablation）：
  alpha0        walk.alpha=0.0
  no_semantic   loss.lambda_typed=0.0
  no_decorr     loss.lambda_decorr=0.0
  no_adversary  loss.lambda_adversary=0.0
  no_rank       loss.lambda_rank=0.0

与历史 run_ablation_v2.py 的差异：
  - amp_dtype=bfloat16（历史 float16 会导致梯度溢出）
  - non_finite_policy=fail_fast
  - 输出到 rerun_submission_audit/03_controls/ablation/

用法（GPU 0）：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/run_ablation_rerun.py --project-root "$PWD" \
        --output-base rerun_submission_audit/03_controls/ablation
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

TEMPLATE = Path(__file__).resolve().parents[2] / "rerun_submission_audit" / "02_formal_sitne" / "formal_v2.yaml"

VARIANTS = {
    "alpha0": {"walk": {"alpha": 0.0}},
    "no_semantic": {"loss": {"lambda_typed": 0.0}},
    "no_decorr": {"loss": {"lambda_decorr": 0.0}},
    "no_adversary": {"loss": {"lambda_adversary": 0.0}},
    "no_rank": {"loss": {"lambda_rank": 0.0}},
}


def _fill(raw: dict, variant: dict, fold: int, seed: int, out_dir: Path) -> dict:
    import copy
    cfg = copy.deepcopy(raw)
    for section, overrides in variant.items():
        cfg[section].update(overrides)
    cfg["data"]["train_path"] = f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
    cfg["data"]["validation_path"] = f"05_splits/typed_ranking_v2/fold_{fold}/val.tsv"
    cfg["data"]["test_path"] = f"05_splits/typed_ranking_v2/fold_{fold}/test.tsv"
    cfg["training"]["seed"] = seed
    cfg["training"]["output_dir"] = str(out_dir)
    return cfg


def _run(project_root: Path, config_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(project_root / "06_code/run_sitne_walk.py"),
         "train", "--config", str(config_path), "--project-root", str(project_root)],
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(project_root / "06_code"),
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        capture_output=True, text=True, check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    raw = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    seeds = [int(x) for x in args.seeds.split(",")]
    output_base = root / args.output_base
    output_base.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(VARIANTS) * 5 * len(seeds)
    current = 0
    for vname, variant in VARIANTS.items():
        for fold in range(5):
            for seed in seeds:
                current += 1
                out_dir = output_base / vname / f"fold_{fold}" / f"seed_{seed}"
                if any(out_dir.rglob("test_metrics.json")):
                    print(f"[{current}/{total}] {vname} f{fold} s{seed} SKIPPED")
                    continue
                if args.dry_run:
                    print(f"[{current}/{total}] dry-run {vname} f{fold} s{seed} -> {out_dir}")
                    continue
                cfg = _fill(raw, variant, fold, seed, out_dir)
                config_path = root / f"06_code/configs/_auto_ab_{vname}_f{fold}_s{seed}.yaml"
                config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                t0 = time.time()
                try:
                    res = _run(root, config_path)
                finally:
                    config_path.unlink(missing_ok=True)
                elapsed = time.time() - t0
                tm = sorted(out_dir.rglob("test_metrics.json"))
                if tm and res.returncode == 0:
                    mrr = json.loads(tm[0].read_text())["metrics"]["mrr"]
                    results.append((vname, fold, seed, mrr))
                    print(f"[{current}/{total}] {vname} f{fold} s{seed} MRR={mrr:.6f} ({elapsed:.0f}s)")
                else:
                    results.append((vname, fold, seed, float("nan")))
                    print(f"[{current}/{total}] {vname} f{fold} s{seed} FAILED")

    if not args.dry_run:
        with (output_base / "ablation_summary.tsv").open("w", encoding="utf-8") as f:
            f.write("variant\tfold\tseed\tmrr\n")
            for vname, fold, seed, mrr in sorted(results):
                f.write(f"{vname}\t{fold}\t{seed}\t{mrr:.6f}\n")
        print(f"summary -> {output_base / 'ablation_summary.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
