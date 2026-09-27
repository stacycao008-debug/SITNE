#!/usr/bin/env python3
"""从 checkpoint 导出统一 per_query_scores.tsv，并做 replay consistency gate。

对每个 formal/ablation run：
  1. 从 selected checkpoint 加载模型（复用 run_r4_probes_fix 的加载逻辑）；
  2. 用 evaluate_filtered_type_ranking 重算 filtered_rank；
  3. 用 score_all_relations 导出 4 raw candidate scores；
  4. 手动算 unfiltered_rank（不 masking，tie=average）；
  5. 与旧 test_ranks.tsv 的 filtered_rank 逐 query 比较（replay gate）。

输出（每个 run 目录内）：
  - per_query_scores.tsv（统一 schema，见 EXPECTED_OUTPUT_SCHEMA §5）
  - replay_gate.json（old vs new filtered_rank 一致性统计）

汇总：
  - replay_gate_summary.tsv（每个 run 的匹配率）

用法（CPU）：
    PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/export_per_query_scores.py --project-root "$PWD"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_code"))
from sitne_walk.data import (  # noqa: E402
    build_known_positive_index,
    load_indexed_triples,
    load_relation_modes,
    load_training_triples,
)
from sitne_walk.evaluator import evaluate_filtered_type_ranking  # noqa: E402
from sitne_walk.graph import build_packed_csr_graph  # noqa: E402
from sitne_walk.model import SITNEWalkModel  # noqa: E402

RELATION_NAMES = {0: "enzymatic", 1: "general", 2: "physical", 3: "spatial"}
DEVICE = torch.device("cpu")


def load_model(run_dir: Path, root: Path):
    ckpt_files = sorted((run_dir / "checkpoints").glob("checkpoint_epoch*.pt"))
    if not ckpt_files:
        return None, None
    ckpt = torch.load(ckpt_files[-1], map_location="cpu", weights_only=False)
    fold = int(run_dir.parents[1].name.split("_")[1])
    train_path = root / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
    triples = load_training_triples(str(train_path), duplicate_policy="binary")
    rmodes = load_relation_modes(None, triples.id_to_relation, default_mode="symmetric")
    ckpt_cfg = ckpt.get("config", {})
    walk_cfg = ckpt_cfg.get("walk", {})
    graph = build_packed_csr_graph(
        triples, relation_modes=rmodes,
        alpha=walk_cfg.get("alpha", 0.25), beta=walk_cfg.get("beta", 0.0),
        epsilon=walk_cfg.get("epsilon", 1e-12), drop_self_loops=True,
        correction_mode=walk_cfg.get("correction_mode", "rank"),
    )
    model_cfg = ckpt_cfg.get("model", {})
    degree_labels = torch.zeros_like(graph.protein_degree, dtype=torch.int64)
    positive = graph.protein_degree > 0
    degree_labels[positive] = 1 + torch.floor(
        torch.log2(graph.protein_degree[positive].float())
    ).to(torch.int64)
    num_degree_classes = int(degree_labels.max().item()) + 1
    model = SITNEWalkModel(
        num_proteins=triples.num_proteins,
        num_relations=triples.num_relations,
        embedding_dim=model_cfg.get("embedding_dim", 512),
        nuisance_cardinalities={"degree": num_degree_classes},
        nuisance_hidden_dim=model_cfg.get("nuisance_hidden_dim", 32),
        decoder_dropout=model_cfg.get("decoder_dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    return model, triples


def _unfiltered_rank(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_scores = scores.gather(1, targets[:, None])
    greater = (scores > target_scores).sum(dim=1).to(torch.float32)
    equal_others = (scores == target_scores).sum(dim=1).to(torch.float32) - 1.0
    return 1.0 + greater + 0.5 * equal_others


def export_run(run_dir: Path, root: Path) -> dict | None:
    fold = int(run_dir.parents[1].name.split("_")[1])
    seed = int(run_dir.parents[0].name.split("_")[1])
    if not (run_dir / "test_ranks.tsv").exists():
        return {"fold": fold, "seed": seed, "status": "NO_TEST_RANKS"}
    model, triples = load_model(run_dir, root)
    if model is None:
        return {"fold": fold, "seed": seed, "status": "NO_CHECKPOINT"}

    test_path = root / f"05_splits/typed_ranking_v2/fold_{fold}/test.tsv"
    known_path = root / "05_splits/typed_ranking_v2/all_known_positives.tsv"
    test = load_indexed_triples(str(test_path), triples.protein_to_id, triples.relation_to_id)
    known = load_indexed_triples(str(known_path), triples.protein_to_id, triples.relation_to_id)
    known_index = build_known_positive_index(known, triples.num_proteins, triples.num_relations)

    # 重算 filtered_rank + raw scores
    result = evaluate_filtered_type_ranking(
        model, test, known_index, DEVICE,
        batch_size=1024, hits_at=(1, 3, 10), tie_policy="average",
        require_full_coverage=False,
    )
    with torch.no_grad():
        raw_scores = model.score_all_relations(test.heads.to(DEVICE), test.tails.to(DEVICE)).float().cpu().numpy()
    targets = test.relations.numpy()
    unfiltered = _unfiltered_rank(
        torch.from_numpy(raw_scores), torch.from_numpy(targets)
    ).numpy()
    new_filtered = result.ranks.numpy()

    # 旧 test_ranks.tsv
    old = pd.read_csv(run_dir / "test_ranks.tsv", sep="\t")
    old_filtered = old["filtered_rank"].to_numpy()

    # replay gate
    if len(old_filtered) != len(new_filtered):
        match_rate = 0.0
        n_mismatch = abs(len(old_filtered) - len(new_filtered))
    else:
        mismatch = ~np.isclose(old_filtered, new_filtered, rtol=0, atol=0)
        n_mismatch = int(mismatch.sum())
        match_rate = 1.0 - n_mismatch / len(new_filtered)

    # 保存 per_query_scores.tsv
    heads = test.heads.numpy()
    tails = test.tails.numpy()
    lo = np.minimum(heads, tails)
    hi = np.maximum(heads, tails)
    pm = np.zeros((len(heads), triples.num_relations), dtype=bool)
    for i in range(len(heads)):
        mask, matched = known_index.lookup(
            torch.tensor([heads[i]]), torch.tensor([tails[i]])
        )
        if matched.item():
            pm[i] = mask[0].numpy()
    df = pd.DataFrame({
        "fold": fold, "seed": seed,
        "protein_u": [triples.id_to_protein[int(x)] for x in lo],
        "protein_v": [triples.id_to_protein[int(x)] for x in hi],
        "target_relation": [RELATION_NAMES[int(x)] for x in targets],
        "target_relation_id": targets,
        "score_enzymatic": raw_scores[:, 0],
        "score_general": raw_scores[:, 1],
        "score_physical": raw_scores[:, 2],
        "score_spatial": raw_scores[:, 3],
        "known_positive_mask": [",".join(str(r) for r in range(triples.num_relations) if pm[i, r]) for i in range(len(heads))],
        "num_known_relations": pm.sum(1),
        "supported": True,
        "filtered_rank": new_filtered,
        "unfiltered_rank": unfiltered,
        "reciprocal_rank": 1.0 / new_filtered,
    })
    df.to_csv(run_dir / "per_query_scores.tsv", sep="\t", index=False)
    gate = {"fold": fold, "seed": seed, "status": "PASS" if n_mismatch == 0 else "FAIL",
            "n_old": int(len(old_filtered)), "n_new": int(len(new_filtered)),
            "n_mismatch": n_mismatch, "match_rate": round(match_rate, 6)}
    (run_dir / "replay_gate.json").write_text(json.dumps(gate, indent=2))
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    # 收集所有 formal + ablation run 目录
    run_dirs = []
    run_dirs += sorted((root / "rerun_submission_audit/02_formal_sitne/runs").glob("fold_*/seed_*/sitne_walk_*"))
    for variant in ["alpha0", "no_semantic", "no_decorr", "no_adversary", "no_rank"]:
        run_dirs += sorted((root / "rerun_submission_audit/03_controls/ablation" / variant).glob("fold_*/seed_*/sitne_walk_*"))
    print(f"总 run 数: {len(run_dirs)}")

    if args.dry_run:
        for rd in run_dirs[:5]:
            print(f"  dry-run: {rd.relative_to(root)}")
        print("  (其余略)")
        return 0

    results = []
    for i, rd in enumerate(run_dirs):
        gate = export_run(rd, root)
        if gate:
            results.append(gate)
            print(f"[{i+1}/{len(run_dirs)}] {rd.relative_to(root)} -> {gate['status']} "
                  f"(mismatch={gate.get('n_mismatch', '?')})")

    summary = pd.DataFrame(results)
    summary.to_csv(root / "rerun_submission_audit/07_sensitivity/replay_gate_summary.tsv",
                   sep="\t", index=False)
    n_pass = int((summary["status"] == "PASS").sum()) if len(summary) else 0
    n_fail = int((summary["status"] == "FAIL").sum()) if len(summary) else 0
    n_nock = int((summary["status"] == "NO_CHECKPOINT").sum()) if len(summary) else 0
    print(f"\nreplay gate summary -> 07_sensitivity/replay_gate_summary.tsv")
    print(f"PASS={n_pass} FAIL={n_fail} NO_CHECKPOINT={n_nock}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
