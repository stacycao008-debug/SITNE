"""SITNE-Walk 超参数优化 (Optuna, 论文 §4.5b / §4.3).

多进程并行 + GPU 轮询:
  - 每个 worker 进程启动时获取唯一 GPU ID (0–7 轮询)
  - 使用 nvidia-smi 选择当前利用率最低的 GPU
  - 设置 CUDA_VISIBLE_DEVICES 后加载 PyTorch，避免争抢
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import optuna
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU 轮询
# ---------------------------------------------------------------------------

GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

def _gpu_util(gpu_id: int) -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
             f"--id={gpu_id}"], capture_output=True, text=True, timeout=5,
        )
        return float(out.stdout.strip())
    except Exception:
        return 100.0

def _pick_best_gpu() -> int:
    """轮询 8 张 GPU，选利用率最低的。"""
    best, best_u = 0, 100.0
    for g in GPU_IDS:
        u = _gpu_util(g)
        if u < best_u:
            best_u, best = u, g
    return best

# ---------------------------------------------------------------------------
# 搜索空间
# ---------------------------------------------------------------------------

SEARCH_SPACE: dict[str, tuple[str, Any]] = {
    "embedding_dim":    ("categorical", [64, 128, 256, 512, 1024]),
    "walk_length":      ("int", (10, 150)),
    "walks_per_protein":("int", (2, 40)),
    "context_window":   ("int", (1, 15)),
    "alpha":            ("float", (0.0, 2.0)),
    "beta":             ("float", (0.0, 2.0)),
    "negative_samples": ("int", (5, 50)),
    "lambda_typed":     ("log_float", (1e-4, 1e1)),
    "lambda_decorr":    ("log_float", (1e-6, 1e-1)),
    "lambda_adversary": ("log_float", (1e-6, 1e-1)),
    "lambda_rank":      ("log_float", (1e-4, 1e2)),
    "learning_rate":    ("log_float", (1e-5, 5e-2)),
}

SECTION = {
    "embedding_dim": "model",
    "alpha": "walk", "beta": "walk", "walk_length": "walk",
    "walks_per_protein": "walk", "context_window": "walk",
    "negative_samples": "walk",
    "lambda_typed": "loss", "lambda_decorr": "loss",
    "lambda_adversary": "loss", "lambda_rank": "loss",
    "learning_rate": "training",
}

FIXED = dict(epochs=15, early_stopping_patience=3, max_steps_per_epoch=50,
             deterministic=True, allow_tf32=False, gradient_clip_norm=2.0,
             use_amp=True, amp_dtype="float16", weight_decay=1e-5,
             rank_batch_size=64, evaluation_batch_size=128, log_interval=200)

def _suggest(trial: optuna.Trial, base: dict) -> dict:
    for n, (k, b) in SEARCH_SPACE.items():
        if k == "categorical":
            v = trial.suggest_categorical(n, b)
        elif k == "int":
            v = trial.suggest_int(n, b[0], b[1])
        elif k == "float":
            v = trial.suggest_float(n, b[0], b[1])
        elif k == "log_float":
            v = trial.suggest_float(n, b[0], b[1], log=True)
        else:
            raise ValueError(f"Unknown kind: {k}")
        base.setdefault(SECTION[n], {})[n] = v
    for k, v in FIXED.items():
        base.setdefault("training", {})[k] = v
    return base

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

class _Objective:
    """Picklable objective。每个 worker 在 __call__ 里轮询 GPU 并独立加载数据。"""

    def __init__(self, config_path: str, project_root: str):
        self.config_path = config_path
        self.project_root = project_root

    def __call__(self, trial: optuna.Trial) -> float:
        # --- 1. GPU 固定 GPU 0 ---
        gpu = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        time.sleep(1.0)

        import torch
        assert torch.cuda.is_available(), f"GPU {gpu} not available"

        # --- 2. 延迟导入 ---
        import yaml
        from sitne_walk.cli import _combine_filter_index, _load_all_known_positive_parts
        from sitne_walk.config import SITNEConfig
        from sitne_walk.data import load_indexed_triples, load_relation_modes, load_training_triples
        from sitne_walk.graph import build_packed_csr_graph
        from sitne_walk.trainer import SITNETrainer, build_model

        root = Path(self.project_root)
        with open(self.config_path) as f:
            base = yaml.safe_load(f)

        raw = _suggest(trial, dict(base))
        cfg = SITNEConfig.from_mapping(raw)

        logger.info("[gpu=%d trial=%d] α=%.2f β=%.2f dim=%d walk=%d lr=%.4f",
                    gpu, trial.number, cfg.walk.alpha, cfg.walk.beta,
                    cfg.model.embedding_dim, cfg.walk.walk_length, cfg.training.learning_rate)

        # --- 3. 加载数据 + 训练 ---
        triples = load_training_triples(
            str(root / cfg.data.train_path), weight_column=cfg.data.weight_column,
            duplicate_policy=cfg.data.duplicate_policy, pair_semantics=cfg.data.pair_semantics,
        )
        validation = load_indexed_triples(
            str(root / cfg.data.validation_path), triples.protein_to_id,
            triples.relation_to_id, pair_semantics=triples.pair_semantics,
        )
        parts = _load_all_known_positive_parts(cfg, root, triples)
        vf = _combine_filter_index(triples, parts) if parts else _combine_filter_index(triples, [validation])
        rmodes = load_relation_modes(None, triples.id_to_relation, default_mode=cfg.data.default_relation_mode)
        graph = build_packed_csr_graph(triples, relation_modes=rmodes,
                                       alpha=cfg.walk.alpha, beta=cfg.walk.beta,
                                       epsilon=cfg.walk.epsilon,
                                       drop_self_loops=cfg.data.drop_self_loops_from_walks,
                                       correction_mode=cfg.walk.correction_mode)
        model, nl, nw = build_model(cfg, triples, graph)
        trainer = SITNETrainer(cfg, triples, graph, model, nl, nw)
        result = trainer.fit(validation=validation, validation_filter=vf)

        mrr = float(result.best_selection_metric)
        if not math.isfinite(mrr):
            raise optuna.TrialPruned(f"NaN MRR={mrr}")
        trial.set_user_attr("best_epoch", result.best_epoch)
        trial.set_user_attr("gpu", gpu)
        logger.info("[gpu=%d trial=%d] MRR=%.4f epoch=%d", gpu, trial.number, mrr, result.best_epoch)
        return mrr

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_optuna_study(config_path, project_root, n_trials=50, n_jobs=1,
                     study_name="sitne_walk_hpo", storage=None, seed=42, timeout=None):
    root = Path(project_root).resolve()
    config_path = str(Path(config_path).resolve())
    # 输出统一落到 rerun 审计的 prospective provenance 目录，不再写 V1 残留路径。
    out = root / "rerun_submission_audit" / "09_provenance" / "optuna" / study_name
    out.mkdir(parents=True, exist_ok=True)

    if storage is None:
        storage = f"sqlite:///{out}/optuna.db"

    code = str(root / "06_code")
    if code not in sys.path:
        sys.path.insert(0, code)

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(study_name=study_name, storage=storage,
                                sampler=sampler, direction="maximize", load_if_exists=True)
    obj = _Objective(config_path=config_path, project_root=str(root))

    logger.info("Optuna HPO: %d trials × %d workers | %s", n_trials, n_jobs, study_name)
    logger.info("GPU strategy: per-worker nvidia-smi poll → CUDA_VISIBLE_DEVICES")
    logger.info("Storage: %s", storage)

    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout, show_progress_bar=True)

    logger.info("=== DONE ===")
    logger.info("Best #%d MRR=%.4f", study.best_trial.number, study.best_value)
    logger.info("Params:\n%s", json.dumps(study.best_params, indent=2))

    (out / "optuna_results.json").write_text(json.dumps(dict(
        study_name=study_name, n_trials=n_trials, n_jobs=n_jobs,
        best_trial=study.best_trial.number, best_value=study.best_value,
        best_params=study.best_params), indent=2))
    study.trials_dataframe().to_csv(out / "optuna_trials.tsv", sep="\t", index=False)

    # Prospective provenance：完整保存 search space / selection rule / best trial，
    # 使未来任何新 search 可追溯（区别于 formal_v1 已丢失的历史搜索记录）。
    (out / "search_space.yaml").write_text(
        yaml.safe_dump(
            {"search_space": SEARCH_SPACE, "section": SECTION, "fixed": FIXED},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (out / "selection_rule.md").write_text(
        "## Selection rule\n\n"
        "- Objective: maximize filtered validation MRR (transductive typed ranking).\n"
        "- Sampler: TPESampler(multivariate=True, seed={}).\n"
        "- Direction: maximize.\n"
        "- Pruning: trials with non-finite MRR are pruned (`TrialPruned`).\n"
        "- Storage: {}.\n"
        "- Selection is inner-validation-only; outer test never enters HPO.\n".format(
            seed, storage
        ),
        encoding="utf-8",
    )
    (out / "best_trial.json").write_text(
        json.dumps(
            {
                "study_name": study_name,
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
            },
            indent=2,
        )
    )

    try:
        imp = optuna.importance.get_param_importances(study)
        (out / "param_importances.json").write_text(json.dumps(imp, indent=2))
    except Exception:
        pass
    try:
        import matplotlib; matplotlib.use("Agg")
        optuna.visualization.matplotlib.plot_optimization_history(study).figure.savefig(out / "optimization_history.png", dpi=150)
        optuna.visualization.matplotlib.plot_param_importances(study).figure.savefig(out / "param_importances.png", dpi=150)
        optuna.visualization.matplotlib.plot_parallel_coordinate(study).figure.savefig(out / "parallel_coordinate.png", dpi=150)
    except Exception:
        pass

    return study
