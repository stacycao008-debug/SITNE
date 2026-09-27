#!/usr/bin/env python3
"""R1 Walk Correction Benchmark v2 — 测试不同的 degree correction 公式。

除了原始公式 log(degree) 外，测试:
1. sqrt(degree / median_degree)
2. log(1 + degree / median_degree)  
3. rank_normalized: rank(degree) / N
"""
from __future__ import annotations

import json, logging, os, sys, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("r1_benchmark_v2")

ROOT = Path(__file__).resolve().parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(ROOT / "06_code"))

from sitne_walk.data import load_training_triples, load_relation_modes
from sitne_walk.graph import build_packed_csr_graph, PackedCSRGraph, _build_segmented_alias
from sitne_walk.walks import AliasTypedWalker, make_generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_PATH = ROOT / "05_splits/typed_ranking_v2/fold_0/train.tsv"
MC_SAMPLES = 5000
MAX_NODES = 200
WALK_LENGTH = 2
SEED = 42

log.info("Loading training triples...")
triples = load_training_triples(str(TRAIN_PATH), duplicate_policy="binary", pair_semantics="unordered")
rmodes = load_relation_modes(None, triples.id_to_relation, default_mode="symmetric")


def build_custom_graph(alpha: float, beta: float, correction_mode: str = "log") -> PackedCSRGraph:
    """构建自定义 correction 公式的图。
    
    correction_mode:
    - "log": 原始公式 -alpha*log(degree) -beta*log(freq)
    - "sqrt": -alpha*sqrt(degree/median) -beta*log(freq)  
    - "rank": -alpha*rank(degree)/N -beta*log(freq)
    """
    import numpy as np
    import torch
    
    source = triples.heads.numpy().astype(np.int64, copy=False)
    relation = triples.relations.numpy().astype(np.int64, copy=False)
    destination = triples.tails.numpy().astype(np.int64, copy=False)
    weights = triples.edge_weights.numpy().astype(np.float64, copy=False)
    
    # degree computation (same as original)
    non_self = source != destination
    pair_codes = source[non_self] * np.int64(triples.num_proteins) + destination[non_self]
    unique_pair_positions = np.unique(pair_codes, return_index=True)[1]
    degree_source = source[non_self][unique_pair_positions]
    degree_destination = destination[non_self][unique_pair_positions]
    protein_degree = np.zeros(triples.num_proteins, dtype=np.float64)
    np.add.at(protein_degree, degree_source, 1.0)
    np.add.at(protein_degree, degree_destination, 1.0)
    
    symmetric_mask = np.array([mode == "symmetric" for mode in rmodes], dtype=np.bool_)
    walk_mask = non_self
    relation_frequency = np.zeros(triples.num_relations, dtype=np.float64)
    np.add.at(relation_frequency, relation[walk_mask], 1.0)
    
    base_source = source[walk_mask]
    base_relation = relation[walk_mask]
    base_destination = destination[walk_mask]
    base_weights = weights[walk_mask]
    reverse_mask = symmetric_mask[base_relation] & (base_source != base_destination)
    expanded_source = np.concatenate([base_source, base_destination[reverse_mask]])
    expanded_relation = np.concatenate([base_relation, base_relation[reverse_mask]])
    expanded_destination = np.concatenate([base_destination, base_source[reverse_mask]])
    expanded_weights = np.concatenate([base_weights, base_weights[reverse_mask]])
    
    # coalesce (simplified - same as original)
    encoded = (expanded_source.astype(np.int64) * np.int64(triples.num_relations * triples.num_proteins)
               + expanded_relation.astype(np.int64) * np.int64(triples.num_proteins)
               + expanded_destination.astype(np.int64))
    order = np.argsort(encoded, kind="stable")
    encoded = encoded[order]
    expanded_source = expanded_source[order]
    expanded_relation = expanded_relation[order]
    expanded_destination = expanded_destination[order]
    expanded_weights = expanded_weights[order]
    first = np.empty(len(encoded), dtype=np.bool_)
    first[0] = True
    first[1:] = encoded[1:] != encoded[:-1]
    group_ids = np.cumsum(first, dtype=np.int64) - 1
    unique_count = int(group_ids[-1]) + 1
    reduced_weights = np.full(unique_count, -np.inf, dtype=np.float64)
    np.maximum.at(reduced_weights, group_ids, expanded_weights.astype(np.float64))
    starts = np.flatnonzero(first)
    expanded_source = expanded_source[starts]
    expanded_relation = expanded_relation[starts]
    expanded_destination = expanded_destination[starts]
    expanded_weights = reduced_weights.astype(np.float32)
    
    # CSR ordering
    order = np.lexsort((expanded_destination, expanded_relation, expanded_source))
    expanded_source = expanded_source[order]
    expanded_relation = expanded_relation[order]
    expanded_destination = expanded_destination[order]
    expanded_weights = expanded_weights[order]
    
    counts = np.bincount(expanded_source, minlength=triples.num_proteins)
    indptr = np.empty(triples.num_proteins + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    
    destination_degree = protein_degree[expanded_destination]
    relation_count = relation_frequency[expanded_relation]
    epsilon = 1e-8
    
    # Apply different correction formulas
    if correction_mode == "log":
        log_weights = (np.log(expanded_weights.astype(np.float64))
                       - alpha * np.log(destination_degree + epsilon)
                       - beta * np.log(relation_count + epsilon))
    elif correction_mode == "sqrt":
        median_deg = np.median(protein_degree[protein_degree > 0])
        log_weights = (np.log(expanded_weights.astype(np.float64))
                       - alpha * np.sqrt(destination_degree / max(median_deg, 1))
                       - beta * np.log(relation_count + epsilon))
    elif correction_mode == "rank":
        ranks = rankdata(protein_degree, method='average')
        normalized_ranks = ranks / len(protein_degree)
        dst_ranks = normalized_ranks[expanded_destination]
        log_weights = (np.log(expanded_weights.astype(np.float64))
                       - alpha * dst_ranks
                       - beta * np.log(relation_count + epsilon))
    else:
        raise ValueError(f"Unknown correction_mode: {correction_mode}")
    
    row_max = np.full(triples.num_proteins, -np.inf, dtype=np.float64)
    np.maximum.at(row_max, expanded_source, log_weights)
    transition_weights = np.exp(log_weights - row_max[expanded_source])
    
    if not np.isfinite(transition_weights).all() or np.any(transition_weights <= 0):
        raise ValueError("校正后的 transition weights 出现非有限值或非正值")
    
    alias_probability, alias_local = _build_segmented_alias(transition_weights, indptr)
    
    graph = PackedCSRGraph(
        indptr=torch.from_numpy(indptr),
        destinations=torch.from_numpy(expanded_destination.astype(np.int64, copy=False)),
        relations=torch.from_numpy(expanded_relation.astype(np.int64, copy=False)),
        edge_weights=torch.from_numpy(expanded_weights.astype(np.float32, copy=False)),
        transition_weights=torch.from_numpy(transition_weights.astype(np.float32, copy=False)),
        alias_probability=torch.from_numpy(alias_probability),
        alias_local=torch.from_numpy(alias_local),
        protein_degree=torch.from_numpy(protein_degree.astype(np.float32)),
        relation_frequency=torch.from_numpy(relation_frequency.astype(np.float32)),
        symmetric_relation_mask=torch.from_numpy(symmetric_mask),
        relation_modes=tuple(rmodes),
        alpha=float(alpha),
        beta=float(beta),
        epsilon=float(epsilon),
    )
    graph.validate()
    return graph


def run_benchmark(graph, variant_name, alpha, beta, nodes):
    graph = graph.to(DEVICE)
    walker = AliasTypedWalker(graph)
    generator = make_generator(DEVICE, SEED)
    rows = []
    
    for i, src in enumerate(nodes):
        src = int(src)
        start = int(graph.indptr[src].item())
        end = int(graph.indptr[src + 1].item())
        degree = end - start
        if degree == 0: continue
        
        theory_weights = graph.transition_weights[start:end].cpu().numpy().astype(np.float64)
        theory_sum = theory_weights.sum()
        if theory_sum <= 0: continue
        theory_probs = theory_weights / theory_sum
        destinations = graph.destinations[start:end].cpu().numpy()
        relations = graph.relations[start:end].cpu().numpy()
        
        counts = np.zeros(degree, dtype=np.float64)
        start_tensor = torch.full((MC_SAMPLES,), src, dtype=torch.int64, device=DEVICE)
        batch = walker.generate(start_tensor, walk_length=WALK_LENGTH, generator=generator)
        next_nodes = batch.proteins[:, 1].cpu().numpy()
        
        for j in range(MC_SAMPLES):
            dst = next_nodes[j]
            if dst >= 0:
                for e in range(degree):
                    if destinations[e] == dst:
                        counts[e] += 1
                        break
        
        empirical_probs = counts / max(counts.sum(), 1)
        
        for e in range(degree):
            dst = int(destinations[e])
            rows.append({
                "variant": variant_name,
                "alpha": alpha, "beta": beta,
                "node_id": src, "destination": dst,
                "theoretical_probability": float(theory_probs[e]),
                "empirical_probability": float(empirical_probs[e]),
                "destination_degree": float(graph.protein_degree[dst].item()),
                "relation_frequency": float(graph.relation_frequency[int(relations[e])].item()),
            })
        
        if (i + 1) % 100 == 0:
            log.info("  [%s] node %d/%d", variant_name, i + 1, len(nodes))
    
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    if len(df) == 0: return {}
    t, e = df["theoretical_probability"].to_numpy(), df["empirical_probability"].to_numpy()
    d, f = df["destination_degree"].to_numpy(), df["relation_frequency"].to_numpy()
    mae = np.abs(t - e).mean()
    tv = 0.5 * np.abs(t - e).sum() / len(t)
    ss_res = np.sum((t - e) ** 2)
    ss_tot = np.sum((e - e.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-16)
    deg_corr, _ = spearmanr(e, d)
    freq_corr, _ = spearmanr(e, f)
    return {"num_rows": len(df), "probability_mae": float(mae), "mean_total_variation": float(tv),
            "probability_r2": float(r2), "empirical_degree_spearman": float(deg_corr),
            "empirical_frequency_spearman": float(freq_corr)}


def main():
    OUTPUT_DIR = ROOT / "08_results/sitne_walk_paper_v2/r1_walk_correction"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get nodes with outgoing edges
    log.info("Identifying valid nodes...")
    base_graph = build_custom_graph(0.0, 0.0, "log")
    valid_nodes = base_graph.nodes_with_outgoing_edges.cpu().numpy()
    np.random.seed(SEED)
    n_nodes = min(MAX_NODES, len(valid_nodes))
    selected_nodes = np.random.choice(valid_nodes, size=n_nodes, replace=False)
    log.info("Selected %d nodes", n_nodes)
    
    # Test different correction formulas
    configs = [
        # (name, mode, alpha, beta)
        ("uncorrected", "log", 0.0, 0.0),
        ("log_a0.25", "log", 0.25, 0.0),
        ("log_a0.50", "log", 0.50, 0.0),
        ("log_a0.75", "log", 0.75, 0.0),
        ("sqrt_a0.5", "sqrt", 0.5, 0.0),
        ("sqrt_a1.0", "sqrt", 1.0, 0.0),
        ("sqrt_a2.0", "sqrt", 2.0, 0.0),
        ("sqrt_a3.0", "sqrt", 3.0, 0.0),
        ("rank_a0.5", "rank", 0.5, 0.0),
        ("rank_a1.0", "rank", 1.0, 0.0),
        ("rank_a2.0", "rank", 2.0, 0.0),
    ]
    
    all_summaries = []
    for name, mode, alpha, beta in configs:
        log.info("Testing %s (mode=%s, a=%.2f, b=%.2f)", name, mode, alpha, beta)
        graph = build_custom_graph(alpha, beta, mode)
        rows = run_benchmark(graph, name, alpha, beta, selected_nodes)
        s = summarize(rows)
        s["variant"] = name
        s["correction_mode"] = mode
        s["alpha"] = alpha
        s["beta"] = beta
        all_summaries.append(s)
        log.info("  MAE=%.6f R²=%.4f deg_spearman=%.4f freq_spearman=%.4f",
                 s["probability_mae"], s["probability_r2"],
                 s["empirical_degree_spearman"], s["empirical_frequency_spearman"])
    
    summary_df = pd.DataFrame(all_summaries)
    print("\n=== Formula Comparison ===")
    print(summary_df[["variant","correction_mode","alpha","beta","probability_mae","probability_r2",
                       "empirical_degree_spearman","empirical_frequency_spearman"]].to_string())
    
    summary_df.to_csv(OUTPUT_DIR / "correction_formula_comparison.tsv", sep="\t", index=False)
    log.info("Saved → %s", OUTPUT_DIR / "correction_formula_comparison.tsv")


if __name__ == "__main__":
    main()
