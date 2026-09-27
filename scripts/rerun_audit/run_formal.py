#!/usr/bin/env python3
"""formal_v2 正式重跑入口：5 outer folds × 3 seeds，non_finite_policy=fail_fast。

用途：GPU 空闲后由人工触发。每个 run 由 06_code/run_sitne_walk.py 独立执行，
产出 run 目录（resolved_config.json / provenance.json / training_history.json /
epoch_summary.tsv / nonfinite_events.tsv / step_log.tsv.gz / test_metrics.json /
test_ranks.tsv / selected_checkpoint.pt）。

用法（GPU 0）：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/run_formal.py --project-root "$PWD" \
        --output-base rerun_submission_audit/02_formal_sitne/runs

--dry-run 仅打印每个 run 的命令与配置路径，不执行。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[2] / "rerun_submission_audit" / "02_formal_sitne" / "formal_v2.yaml"
FOLDS = [0, 1, 2, 3, 4]
SEEDS = [42, 123, 456]


def _fill(template: str, fold: int, seed: int, out_dir: Path) -> str:
    return (
        template
        .replace("FOLD_TRAIN_PLACEHOLDER", f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv")
        .replace("FOLD_VAL_PLACEHOLDER", f"05_splits/typed_ranking_v2/fold_{fold}/val.tsv")
        .replace("FOLD_TEST_PLACEHOLDER", f"05_splits/typed_ranking_v2/fold_{fold}/test.tsv")
        .replace("SEED_PLACEHOLDER", str(seed))
        .replace("OUTPUT_DIR_PLACEHOLDER", str(out_dir))
    )


def _run(project_root: Path, config_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(project_root / "06_code/run_sitne_walk.py"),
            "train",
            "--config",
            str(config_path),
            "--project-root",
            str(project_root),
        ],
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(project_root / "06_code"),
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        capture_output=True,
        text=True,
        check=False,
    )


def _run_passed(run_dir: Path) -> tuple[bool, str]:
    """正式结果准入 gate（EVALUATION_CONTRACT §14）。

    只有 RUN_SUCCESS（completed 无 failure）且 non_finite 全为 0 才 PASS；
    否则返回 EXCLUDED_FAILED_RUN，不得进入正式 aggregate。
    """
    completed = (run_dir / "completed.json").exists()
    failure = (run_dir / "failure.json").exists()
    if failure or not completed:
        return False, "EXCLUDED_FAILED_RUN"
    nfe = run_dir / "nonfinite_events.tsv"
    if nfe.exists():
        lines = nfe.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > 1:  # 有数据行 → non_finite > 0
            return False, "EXCLUDED_FAILED_RUN(non_finite>0)"
    return True, "PASSED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--folds", default=None, help="逗号分隔，如 0,1,2；默认全部 0-4")
    parser.add_argument("--seeds", default=None, help="逗号分隔，如 42,123,456；默认全部")
    args = parser.parse_args()

    folds = (
        [int(x) for x in args.folds.split(",")] if args.folds else FOLDS
    )
    seeds = (
        [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS
    )

    root = Path(args.project_root).resolve()
    template = TEMPLATE.read_text(encoding="utf-8")
    output_base = root / args.output_base
    output_base.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(folds) * len(seeds)
    current = 0
    for fold in folds:
        for seed in seeds:
            current += 1
            out_dir = output_base / f"fold_{fold}" / f"seed_{seed}"
            config_path = root / f"06_code/configs/_auto_formal_v2_f{fold}_s{seed}.yaml"
            config_text = _fill(template, fold, seed, out_dir)

            if args.dry_run:
                print(f"[{current}/{total}] dry-run fold={fold} seed={seed} -> {out_dir}")
                continue

            existing = sorted(out_dir.glob("sitne_walk_*"))
            if existing:
                run_dir = existing[-1]
                passed, status = _run_passed(run_dir)
                tm = sorted(run_dir.glob("test_metrics.json"))
                mrr = json.loads(tm[0].read_text())["metrics"]["mrr"] if tm else float("nan")
                results.append((fold, seed, mrr, 0.0, status))
                print(f"[{current}/{total}] fold={fold} seed={seed} SKIPPED "
                      f"(MRR={mrr:.6f}, {status})")
                continue

            config_path.write_text(config_text, encoding="utf-8")
            t0 = time.time()
            try:
                result = _run(root, config_path)
            finally:
                config_path.unlink(missing_ok=True)
            elapsed = time.time() - t0

            run_dirs = sorted(out_dir.glob("sitne_walk_*"))
            if run_dirs:
                run_dir = run_dirs[-1]
                passed, status = _run_passed(run_dir)
                tm = sorted(run_dir.glob("test_metrics.json"))
                if tm and passed:
                    mrr = json.loads(tm[0].read_text())["metrics"]["mrr"]
                    results.append((fold, seed, mrr, elapsed, status))
                    print(f"[{current}/{total}] fold={fold} seed={seed} MRR={mrr:.6f} "
                          f"({elapsed:.0f}s) [{status}]")
                else:
                    results.append((fold, seed, float("nan"), elapsed, status))
                    print(f"[{current}/{total}] fold={fold} seed={seed} FAILED [{status}]")
            else:
                results.append((fold, seed, float("nan"), elapsed, "EXCLUDED_FAILED_RUN"))
                print(f"[{current}/{total}] fold={fold} seed={seed} FAILED "
                      f"[EXCLUDED_FAILED_RUN: no run dir]")

    if not args.dry_run:
        summary_path = output_base / "training_summary.tsv"
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write("fold\tseed\tmrr\telapsed_seconds\tstatus\n")
            for fold, seed, mrr, t, status in sorted(results):
                handle.write(f"{fold}\t{seed}\t{mrr:.6f}\t{t:.0f}\t{status}\n")
        excluded = [r for r in results if r[4] != "PASSED"]
        if excluded:
            print(f"WARNING: {len(excluded)} run(s) EXCLUDED from formal aggregate: "
                  f"{[(r[0], r[1], r[4]) for r in excluded]}")
            print("拒绝生成 final 结果：存在 non_finite>0 或失败的 run，请修复后重跑。")
        else:
            print(f"summary -> {summary_path} (all PASSED)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
