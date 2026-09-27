#!/usr/bin/env python3
"""Phase C: SITNE-Walk Optimal 5-fold x 3 seeds 训练 (v2 splits)

基于 BlackBox typed v4 canonical 数据重建的 typed_ranking_v2 splits，
使用 rank-normalized correction (alpha=0.25, beta=0.0)。
"""
from __future__ import annotations

import json, logging, os, sys, time, subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_all_v2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "06_code"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEMPLATE = ROOT / "06_code/configs/sitne_walk.optimal.yaml"
with open(TEMPLATE) as f:
    template = f.read()

FOLDS = [0, 1, 2, 3, 4]
SEEDS = [42, 123, 456]
TOTAL = len(FOLDS) * len(SEEDS)

OUTPUT_BASE = ROOT / "08_results/sitne_walk_paper_v2/r2_main_performance/optimal"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

results = []
current = 0

logger.info("=== Phase C: SITNE-Walk Optimal 5-fold CV (v2 splits) ===")
logger.info("GPU: 0 | Folds: %s | Seeds: %s | Total: %d", FOLDS, SEEDS, TOTAL)

for fold in FOLDS:
    for seed in SEEDS:
        current += 1
        out_dir = OUTPUT_BASE / f"fold_{fold}" / f"seed_{seed}"

        # Skip if already completed
        completed_file = out_dir / "completed.json"
        if completed_file.exists():
            # Check if test_metrics also exists
            for p in out_dir.rglob("test_metrics.json"):
                logger.info("[%d/%d] Fold %d Seed %d: SKIPPED (already completed)", current, TOTAL, fold, seed)
                with open(p) as f:
                    metrics = json.load(f)["metrics"]
                results.append((fold, seed, metrics["mrr"], 0))
                break
            continue

        # Generate config with v2 paths
        config_yaml = template \
            .replace("FOLD_TRAIN_PLACEHOLDER", f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv") \
            .replace("FOLD_VAL_PLACEHOLDER",   f"05_splits/typed_ranking_v2/fold_{fold}/val.tsv") \
            .replace("FOLD_TEST_PLACEHOLDER",  f"05_splits/typed_ranking_v2/fold_{fold}/test.tsv") \
            .replace("SEED_PLACEHOLDER", str(seed)) \
            .replace("OUTPUT_DIR_PLACEHOLDER", str(out_dir))

        config_path = ROOT / f"06_code/configs/_auto_v2_fold{fold}_seed{seed}.yaml"
        config_path.write_text(config_yaml)

        logger.info("[%d/%d] Fold %d Seed %d: training...", current, TOTAL, fold, seed)
        t0 = time.time()

        result = subprocess.run(
            [sys.executable, str(ROOT / "06_code/run_sitne_walk.py"),
             "train", "--config", str(config_path),
             "--project-root", str(ROOT)],
            timeout=7200, text=True,  # 2 hour timeout per run
            env={**os.environ, "PYTHONPATH": str(ROOT / "06_code")},
            capture_output=True,
        )

        elapsed = time.time() - t0

        # Find test_metrics.json
        test_metric_path = None
        for p in Path(out_dir).rglob("test_metrics.json"):
            test_metric_path = p
            break

        if test_metric_path and test_metric_path.exists():
            with open(test_metric_path) as f:
                metrics = json.load(f)["metrics"]
            mrr = metrics["mrr"]
            results.append((fold, seed, mrr, elapsed))
            logger.info("  Fold %d Seed %d: MRR=%.4f (%.0fs)", fold, seed, mrr, elapsed)
        else:
            logger.error("  Fold %d Seed %d: FAILED (no test_metrics.json)", fold, seed)
            logger.error("  stderr: %s", result.stderr[-500:] if result.stderr else "none")
            results.append((fold, seed, 0.0, elapsed))

        # Clean up temp config
        config_path.unlink(missing_ok=True)

# Summary
logger.info("\n=== Phase C Training Complete ===")
summary_path = OUTPUT_BASE / "training_summary.tsv"
with open(summary_path, "w") as f:
    f.write("fold\tseed\tmrr\telapsed_seconds\n")
    for fold, seed, mrr, t in sorted(results):
        f.write(f"{fold}\t{seed}\t{mrr:.6f}\t{t:.0f}\n")
        logger.info("Fold %d Seed %3d: MRR=%.4f  time=%.0fs", fold, seed, mrr, t)

mrrs = [r[2] for r in results if r[2] > 0]
if mrrs:
    mean_mrr = sum(mrrs) / len(mrrs)
    std_mrr = (sum((x - mean_mrr) ** 2 for x in mrrs) / len(mrrs)) ** 0.5
    logger.info("Mean MRR: %.6f ± %.6f (n=%d)", mean_mrr, std_mrr, len(mrrs))

logger.info("Summary saved to: %s", summary_path)
