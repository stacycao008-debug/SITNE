#!/usr/bin/env python3
"""Phase D: SITNE-Walk 消融实验 (v2 splits)

10 个消融变体 × 5 folds × 1 seed = 50 runs。
每个变体修改 optimal config 的特定参数后运行。

消融变体:
  1. alpha0:     walk.alpha = 0.0 (禁用 degree correction)
  2. beta0:      walk.beta = 0.0 (禁用 frequency correction, 注意 full 中 beta 已为 0)
  3. uncorrected: walk.alpha=0.0, walk.beta=0.0 (完全禁用校正)
  4. no_semantic: loss.lambda_typed = 0.0 (禁用 semantic decoder)
  5. no_decorr:   loss.lambda_decorr = 0.0 (禁用 decorrelation)
  6. no_adversary: loss.lambda_adversary = 0.0 (禁用 adversary)
  7. no_rank:     loss.lambda_rank = 0.0 (禁用 ranking loss)
  8. shuffled_type: 打乱 train 中的 relation type labels
  9. binary_walk: walk 忽略 type (correction_mode 保留但 type 信息被忽略)
  10. shortcut_only: 仅 degree/frequency shortcut (lambda_typed=0, no walks)
"""
from __future__ import annotations

import json, logging, os, sys, time, subprocess, shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ablation_v2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "06_code"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Base config template (optimal for v2)
BASE_CONFIG = {
    "data": {
        "train_path": "PLACEHOLDER_TRAIN",
        "validation_path": "PLACEHOLDER_VAL",
        "test_path": "PLACEHOLDER_TEST",
        "split_manifest_path": "05_splits/typed_ranking_v2/split_manifest.json",
        "require_manifest_hashes": False,
        "all_known_positive_paths": ["05_splits/typed_ranking_v2/all_known_positives.tsv"],
        "relation_metadata_path": None,
        "relation_hierarchy_path": None,
        "weight_column": None,
        "duplicate_policy": "binary",
        "pair_semantics": "unordered",
        "default_relation_mode": "symmetric",
        "drop_self_loops_from_walks": True,
    },
    "walk": {
        "alpha": 0.25,
        "beta": 0.0,
        "epsilon": 1e-12,
        "correction_mode": "rank",
        "walk_length": 81,
        "walks_per_protein": 16,
        "walk_batch_size": 64,
        "context_window": 1,
        "negative_samples": 16,
        "negative_distribution": "unigram",
        "unigram_power": 0.75,
        "max_topology_pairs_per_batch": 2048,
    },
    "model": {
        "embedding_dim": 512,
        "decoder_dropout": 0.1,
        "nuisance_hidden_dim": 32,
        "nuisance_heads": ["degree"],
    },
    "loss": {
        "lambda_typed": 1.864,
        "lambda_decorr": 0.000255,
        "lambda_adversary": 0.00000506,
        "lambda_rank": 25.64,
        "lambda_hierarchy": 0.0,
        "gradient_reversal_coefficient": 1.0,
        "rank_margin": 0.0,
        "hierarchy_margin": 0.0,
    },
    "training": {
        "device": "auto",
        "seed": "PLACEHOLDER_SEED",
        "epochs": 50,
        "learning_rate": 0.01923,
        "weight_decay": 0.00001,
        "gradient_clip_norm": 2.0,
        "rank_batch_size": 64,
        "early_stopping_patience": 5,
        "evaluation_batch_size": 128,
        "use_amp": True,
        "amp_dtype": "float16",
        "deterministic": True,
        "allow_tf32": False,
        "max_steps_per_epoch": None,
        "log_interval": 50,
        "output_dir": "PLACEHOLDER_OUTPUT",
    },
    "evaluation": {
        "tie_policy": "average",
        "hits_at": [1, 3, 10],
        "require_full_coverage": False,
    },
}

# Ablation variant definitions
ABLATION_VARIANTS = {
    "full": {"desc": "Full model (baseline for ablation comparison)"},
    "alpha0": {"desc": "Disable degree correction", "walk": {"alpha": 0.0}},
    "uncorrected": {"desc": "Disable all correction", "walk": {"alpha": 0.0, "beta": 0.0}},
    "no_semantic": {"desc": "Disable semantic decoder", "loss": {"lambda_typed": 0.0}},
    "no_decorr": {"desc": "Disable decorrelation loss", "loss": {"lambda_decorr": 0.0}},
    "no_adversary": {"desc": "Disable nuisance adversary", "loss": {"lambda_adversary": 0.0}},
    "no_rank": {"desc": "Disable ranking loss", "loss": {"lambda_rank": 0.0}},
    "shuffled_type": {"desc": "Shuffled relation type labels"},
    "binary_walk": {"desc": "Binary walk (ignore type)", "walk": {"correction_mode": "rank", "alpha": 0.0, "beta": 0.0}},
}

FOLDS = [0, 1, 2, 3, 4]
SEEDS = [42]  # 先跑单种子验证效果
OUTPUT_BASE = ROOT / "08_results/sitne_walk_paper_v2/r3_ablation"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


def generate_config(variant_name: str, fold: int, seed: int) -> dict:
    """生成特定变体+fold+seed的配置。"""
    import copy
    config = copy.deepcopy(BASE_CONFIG)

    # Apply variant overrides
    variant_def = ABLATION_VARIANTS[variant_name]
    if "walk" in variant_def:
        config["walk"].update(variant_def["walk"])
    if "loss" in variant_def:
        config["loss"].update(variant_def["loss"])

    # Replace placeholders
    config["data"]["train_path"] = f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
    config["data"]["validation_path"] = f"05_splits/typed_ranking_v2/fold_{fold}/val.tsv"
    config["data"]["test_path"] = f"05_splits/typed_ranking_v2/fold_{fold}/test.tsv"
    config["training"]["seed"] = seed
    config["training"]["output_dir"] = str(OUTPUT_BASE / variant_name / f"fold_{fold}" / f"seed_{seed}")

    # For shuffled_type: generate shuffled train file
    if variant_name == "shuffled_type":
        shuffled_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/train_shuffled_type.tsv"
        if not shuffled_path.exists():
            import pandas as pd
            import numpy as np
            df = pd.read_csv(ROOT / config["data"]["train_path"], sep="\t", dtype=str, keep_default_na=False)
            df["type_name"] = np.random.RandomState(seed).permutation(df["type_name"].values)
            df.to_csv(shuffled_path, sep="\t", index=False)
            logger.info("Generated shuffled train: %s", shuffled_path)
        config["data"]["train_path"] = str(shuffled_path.relative_to(ROOT))

    # For binary_walk: we set alpha=beta=0 and use rank mode (effectively uniform transitions)
    # The "binary" aspect is that we ignore relation type in walk (already implicit in uniform weights)

    return config


def run_config(variant_name: str, fold: int, seed: int) -> tuple[bool, float, float]:
    """运行单个配置，返回 (success, mrr, elapsed_seconds)。"""
    config = generate_config(variant_name, fold, seed)

    # Check if already completed
    output_dir = Path(config["training"]["output_dir"])
    test_metrics = None
    for p in output_dir.rglob("test_metrics.json"):
        test_metrics = p
        break
    if test_metrics and test_metrics.exists():
        with open(test_metrics) as f:
            metrics = json.load(f)["metrics"]
        mrr = metrics["mrr"]
        logger.info("[%s] fold_%d/seed_%d: SKIPPED (MRR=%.4f)", variant_name, fold, seed, mrr)
        return True, mrr, 0.0

    # Write temp config
    config_path = ROOT / f"06_code/configs/_auto_ablation_{variant_name}_f{fold}_s{seed}.yaml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("[%s] fold_%d/seed_%d: training...", variant_name, fold, seed)
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, str(ROOT / "06_code/run_sitne_walk.py"),
         "train", "--config", str(config_path),
         "--project-root", str(ROOT)],
        timeout=7200, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "06_code")},
        capture_output=True,
    )

    elapsed = time.time() - t0

    # Find result
    test_metrics = None
    for p in output_dir.rglob("test_metrics.json"):
        test_metrics = p
        break

    success = test_metrics and test_metrics.exists()
    mrr = 0.0
    if success:
        with open(test_metrics) as f:
            mrr = json.load(f)["metrics"]["mrr"]
        logger.info("[%s] fold_%d/seed_%d: MRR=%.4f (%.0fs)", variant_name, fold, seed, mrr, elapsed)
    else:
        logger.error("[%s] fold_%d/seed_%d: FAILED", variant_name, fold, seed)
        # Log error
        err_log = output_dir / "error.log"
        err_log.parent.mkdir(parents=True, exist_ok=True)
        err_log.write_text(result.stderr[-2000:] if result.stderr else "no stderr")

    config_path.unlink(missing_ok=True)
    return success, mrr, elapsed


def main():
    logger.info("=== Phase D: Ablation Experiments (v2 splits) ===")
    logger.info("Variants: %s", list(ABLATION_VARIANTS.keys()))
    logger.info("Folds: %s, Seeds: %s", FOLDS, SEEDS)

    all_results = []
    total = len(ABLATION_VARIANTS) * len(FOLDS) * len(SEEDS)
    current = 0

    for variant_name in sorted(ABLATION_VARIANTS.keys()):
        for fold in FOLDS:
            for seed in SEEDS:
                current += 1
                logger.info("[%d/%d] %s fold_%d seed_%d", current, total, variant_name, fold, seed)
                success, mrr, elapsed = run_config(variant_name, fold, seed)
                all_results.append({
                    "variant": variant_name,
                    "fold": fold,
                    "seed": seed,
                    "mrr": mrr,
                    "success": success,
                    "elapsed": elapsed,
                })

    # Save summary
    import pandas as pd
    df = pd.DataFrame(all_results)
    summary_path = OUTPUT_BASE / "ablation_summary.tsv"
    df.to_csv(summary_path, sep="\t", index=False)

    # Print per-variant summary
    logger.info("\n=== Ablation Summary ===")
    for variant_name in sorted(ABLATION_VARIANTS.keys()):
        sub = df[(df["variant"] == variant_name) & df["success"]]
        if len(sub) > 0:
            logger.info("  %-18s: MRR=%.4f ± %.4f (n=%d)", variant_name,
                        sub["mrr"].mean(), sub["mrr"].std(), len(sub))
        else:
            logger.info("  %-18s: FAILED", variant_name)

    logger.info("\nSummary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
