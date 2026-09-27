#!/usr/bin/env python3
"""rank-form walk calibration：验证正式配置（correction_mode=rank, alpha=0.25, beta=0）
的 empirical transition distribution 是否匹配理论 transition distribution。

背景（P0-4）：formal_v1 用 rank-form，但旧 Figure 4 的 parameter_grid 是 log-form，
未直接验证 rank-form alpha=0.25。本脚本用正式图构建器 build_packed_csr_graph 在 CPU
上做 Monte Carlo 诊断，输出 MAE / R² / TV / Spearman(degree) / Spearman(freq)。

用法（CPU 即可）：
    PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/run_calibration.py --project-root "$PWD" \
        --output-dir rerun_submission_audit/05_calibration
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_code"))
from sitne_walk.data import load_training_triples, load_relation_modes  # noqa: E402
from sitne_walk.graph import build_packed_csr_graph  # noqa: E402
from sitne_walk.walks import AliasTypedWalker, make_generator  # noqa: E402

MC_SAMPLES = 5000
MAX_NODES = 200
WALK_LENGTH = 2
SEED = 42


def run_node_benchmark(walker, generator, device, src, graph):
    start = int(graph.indptr[src].item())
    end = int(graph.indptr[src + 1].item())
    degree = end - start
    if degree == 0:
        return []
    theory_weights = graph.transition_weights[start:end].cpu().numpy().astype(np.float64)
    theory_sum = theory_weights.sum()
    if theory_sum <= 0:
        return []
    theory_probs = theory_weights / theory_sum
    destinations = graph.destinations[start:end].cpu().numpy()
    relations = graph.relations[start:end].cpu().numpy()

    counts = np.zeros(degree, dtype=np.float64)
    start_tensor = torch.full((MC_SAMPLES,), src, dtype=torch.int64, device=device)
    batch = walker.generate(start_tensor, walk_length=WALK_LENGTH, generator=generator)
    next_nodes = batch.proteins[:, 1].cpu().numpy()
    dest_to_edge = {int(destinations[e]): e for e in range(degree)}
    for dst in next_nodes:
        if dst >= 0:
            e = dest_to_edge.get(int(dst))
            if e is not None:
                counts[e] += 1
    empirical_probs = counts / max(counts.sum(), 1)

    rows = []
    for e in range(degree):
        dst = int(destinations[e])
        rows.append({
            "node_id": int(src),
            "destination": dst,
            "theoretical_probability": float(theory_probs[e]),
            "empirical_probability": float(empirical_probs[e]),
            "destination_degree": float(graph.protein_degree[dst].item()),
            "relation_frequency": float(graph.relation_frequency[int(relations[e])].item()),
        })
    return rows


def summarize(rows):
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    t = df["theoretical_probability"].to_numpy()
    e = df["empirical_probability"].to_numpy()
    d = df["destination_degree"].to_numpy()
    f = df["relation_frequency"].to_numpy()
    mae = np.abs(t - e).mean()
    tv = 0.5 * np.abs(t - e).sum() / len(t)
    ss_res = np.sum((t - e) ** 2)
    ss_tot = np.sum((e - e.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-16)
    deg_corr, _ = spearmanr(e, d)
    freq_corr, _ = spearmanr(e, f)
    return {
        "num_rows": int(len(df)),
        "probability_mae": float(mae),
        "mean_total_variation": float(tv),
        "probability_r2": float(r2),
        "empirical_degree_spearman": float(deg_corr),
        "empirical_frequency_spearman": float(freq_corr),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_path = root / f"05_splits/typed_ranking_v2/fold_{args.fold}/train.tsv"
    triples = load_training_triples(str(train_path), duplicate_policy="binary",
                                    pair_semantics="unordered")
    rmodes = load_relation_modes(None, triples.id_to_relation, default_mode="symmetric")

    # 正式点 + 小型 rank alpha grid。
    configs = [
        ("rank_a0.0", "rank", 0.0, 0.0),
        ("rank_a0.25_formal", "rank", 0.25, 0.0),
        ("rank_a0.5", "rank", 0.5, 0.0),
        ("rank_a1.0", "rank", 1.0, 0.0),
    ]

    base_graph = build_packed_csr_graph(
        triples, relation_modes=rmodes, alpha=0.0, beta=0.0,
        epsilon=1e-12, drop_self_loops=True, correction_mode="rank",
    )
    valid_nodes = base_graph.nodes_with_outgoing_edges.cpu().numpy()
    rng = np.random.default_rng(SEED)
    n_nodes = min(MAX_NODES, len(valid_nodes))
    selected = rng.choice(valid_nodes, size=n_nodes, replace=False)

    summaries = []
    for name, mode, alpha, beta in configs:
        graph = build_packed_csr_graph(
            triples, relation_modes=rmodes, alpha=alpha, beta=beta,
            epsilon=1e-12, drop_self_loops=True, correction_mode=mode,
        ).to(device)
        walker = AliasTypedWalker(graph)
        generator = make_generator(device, SEED)
        rows = []
        for i, src in enumerate(selected):
            rows.extend(run_node_benchmark(walker, generator, device, int(src), graph))
            if (i + 1) % 50 == 0:
                print(f"  [{name}] node {i + 1}/{n_nodes}")
        s = summarize(rows)
        s["variant"] = name
        s["correction_mode"] = mode
        s["alpha"] = alpha
        s["beta"] = beta
        summaries.append(s)
        print(f"  {name}: MAE={s['probability_mae']:.6f} R²={s['probability_r2']:.4f} "
              f"deg_spearman={s['empirical_degree_spearman']:.4f} "
              f"freq_spearman={s['empirical_frequency_spearman']:.4f}")

    summary_df = pd.DataFrame(summaries)
    summary_path = out / "rank_form_calibration.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"calibration -> {summary_path}")

    formal = next(s for s in summaries if s["variant"] == "rank_a0.25_formal")
    interp = [
        "# rank-form calibration interpretation",
        "",
        "## 正式点（correction_mode=rank, alpha=0.25, beta=0）",
        "",
        f"- MAE = {formal['probability_mae']:.6f}",
        f"- R²  = {formal['probability_r2']:.4f}",
        f"- TV  = {formal['mean_total_variation']:.6f}",
        f"- Spearman(prob, destination degree) = {formal['empirical_degree_spearman']:.4f}",
        f"- Spearman(prob, relation frequency) = {formal['empirical_frequency_spearman']:.4f}",
        "",
        "## 判定",
        "",
        "- MAE 应接近 0（empirical ≈ theoretical），证明 rank-form sampler 数值忠实地实现",
        "  了指定 transition distribution；若 MAE 明显偏大，说明 alias/walker 与理论分布不一致。",
        "- 与旧 log-form grid 的 degree Spearman（α=0 时 ≈ -0.148）对比，rank-form 的",
        "  degree Spearman 反映 rank correction 是否如理论改变 degree 暴露方向。",
        "",
        "## 备注",
        "",
        "- 本脚本用正式 build_packed_csr_graph（rank 模式），与 formal run 的图构建一致；",
        "- 旧 Figure 4 的 log-form grid 保留为 sensitivity，不替代本 rank-form 验证。",
        "",
    ]
    (out / "interpretation.md").write_text("\n".join(interp), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
