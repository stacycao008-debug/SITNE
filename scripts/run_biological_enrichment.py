#!/usr/bin/env python3
"""T10 GO/Reactome enrichment — 基于 relation embedding 的生物学富集分析。

策略:
  1. 加载 R2 optimal 模型的 semantic_relation 权重 (4 个 relation × d)
  2. 对 relation embedding 做 K-means / hierarchical 聚类
  3. 对每个 cluster 做超几何检验 (GO term / Reactome pathway)
  4. 输出 enrichment_results.tsv

⚠ TODO(DATA_DEPENDENCY): 本分析需要外部 GO annotation 文件 (protein-to-GO mapping)。
   本项目目前没有 GO/Reactome 映射数据。脚本中设计了完整的富集分析框架，
   并在数据缺失时优雅降级，仅输出 relation embedding 聚类结果。

所需外部数据文件 (需手动提供):
  - GO annotation: UniProt accession → GO term 列表
    推荐来源: https://www.uniprot.org/id-mapping 或 BioMart
    期望格式: TSV, 列: uniprot_id, go_term, go_namespace (BP/MF/CC)
  - GO term info: GO term → name, namespace
    推荐来源: http://geneontology.org/docs/download-ontology/
    期望格式: TSV, 列: go_term, go_name, go_namespace
  - Reactome pathway: UniProt accession → pathway ID → pathway name
    推荐来源: https://reactome.org/download-data
    期望格式: TSV, 列: uniprot_id, pathway_id, pathway_name

Usage:
    python scripts/run_biological_enrichment.py
    python scripts/run_biological_enrichment.py --go-annotation /path/to/protein_go.tsv
"""

import argparse
import json, logging, os, sys, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enrichment")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "06_code"))

OUT_DIR = ROOT / "08_results/sitne_walk_paper_v2/biology/enrichment"

# ======================================================================
# 1. Load relation embeddings from R2 optimal checkpoints
# ======================================================================

def find_optimal_checkpoints() -> list[Path]:
    """查找所有 optimal checkpoint 路径。

    实际文件结构: optimal/fold_X/seed_Y/run_dir/checkpoints/checkpoint_epochNNNN.pt
    对每个 run_dir 只取最后一个 epoch（最高编号）的 checkpoint。
    """
    ckpt_dir = ROOT / "08_results/sitne_walk_paper_v2/r2_main_performance/optimal"

    # Collect checkpoints per run dir
    run_checkpoints: dict[str, list[Path]] = defaultdict(list)
    for ckpt in sorted(ckpt_dir.glob("fold_*/seed_*/*/checkpoints/checkpoint_epoch*.pt")):
        run_dir = str(ckpt.parent.parent)  # run_dir = .../fold_X/seed_Y/run_name
        run_checkpoints[run_dir].append(ckpt)

    best_checkpoints = []
    for run_dir, ckpts in run_checkpoints.items():
        # Take the highest epoch (last sorted)
        best_checkpoints.append(ckpts[-1])

    if not best_checkpoints:
        log.warning("未找到 optimal checkpoint (pattern: fold_*/seed_*/*/checkpoints/checkpoint_epoch*.pt)")
    else:
        log.info("找到 %d 个 run_dir, 取每个最高 epoch checkpoint", len(best_checkpoints))
    return sorted(best_checkpoints)


def load_relation_embeddings(checkpoints: list[Path]) -> np.ndarray:
    """从 checkpoints 加载 relation embeddings。

    注意: 不同 fold 的 relation 数量可能不同（通常为 4 个 coarse relations）。
    我们取第一个可用 checkpoint 的 embedding（所有 fold 共享的 4 个 core relations）。

    返回: [num_relations, embedding_dim]
    """
    all_embeddings = []
    all_shapes = set()
    for ckpt_path in checkpoints:
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            target_keys = ["semantic_relation.weight", "relation_embedding.weight", "rel_emb.weight"]
            found_in_this_ckpt = False
            for source in [state, state.get("model_state", {}), state.get("model_state_dict", {})]:
                if not isinstance(source, dict):
                    continue
                for key in target_keys:
                    if key in source:
                        arr = source[key].numpy()
                        all_embeddings.append(arr)
                        all_shapes.add(arr.shape)
                        found_in_this_ckpt = True
                        break
                if found_in_this_ckpt:
                    break
        except Exception as e:
            log.warning("无法加载 %s: %s", ckpt_path, e)

    if not all_embeddings:
        log.error("从 %d 个 checkpoint 中未找到任何 relation embedding", len(checkpoints))
        return np.array([])

    # 跨不同 fold 的关系数可能不一致，取最常见的形状
    from collections import Counter
    shape_counts = Counter(all_shapes)
    most_common_shape = shape_counts.most_common(1)[0][0]
    filtered = [e for e in all_embeddings if e.shape == most_common_shape]

    if len(filtered) < len(all_embeddings):
        log.warning("不同 checkpoint 的 relation 数量不同: shapes=%s, 使用 %d/%d 个形状为 %s 的 embeddings",
                     sorted(str(s) for s in all_shapes), len(filtered), len(all_embeddings), most_common_shape)

    # Average across checkpoints with the same shape
    stacked = np.stack(filtered, axis=0)
    averaged = stacked.mean(axis=0)
    log.info("从 %d/%d 个 checkpoint 加载并平均 relation embedding: shape=%s",
             len(filtered), len(checkpoints), averaged.shape)
    return averaged


def cluster_relations(embeddings: np.ndarray, method: str = "ward", max_k: int = 5) -> pd.DataFrame:
    """对 relation embeddings 做层次聚类。

    Returns:
        DataFrame: relation_id, cluster_id, cluster_label
    """
    num_relations = embeddings.shape[0]
    if num_relations < 3:
        # Too few to cluster
        return pd.DataFrame({
            "relation_id": list(range(num_relations)),
            "cluster_id": [0] * num_relations,
            "cluster_label": ["cluster_0"] * num_relations,
        })

    dist = pdist(embeddings, metric="cosine")
    Z = linkage(dist, method=method)
    clusters = fcluster(Z, t=min(max_k, num_relations), criterion="maxclust")
    clusters = clusters - 1  # 0-indexed

    df = pd.DataFrame({
        "relation_id": list(range(num_relations)),
        "cluster_id": clusters,
    })
    df["cluster_label"] = df["cluster_id"].apply(lambda x: f"cluster_{x}")
    log.info("Relation clusters: %s", dict(df.groupby("cluster_id").size()))
    return df


# ======================================================================
# 2. Load GO / Reactome annotation
# ======================================================================

def load_go_annotation(go_path: Path) -> dict[str, set[str]]:
    """加载 protein→GO term 映射。

    Args:
        go_path: TSV 文件, 列: uniprot_id, go_term, go_namespace (optional)

    Returns:
        dict: accession → set of GO terms
    """
    if not go_path or not Path(go_path).exists():
        log.warning("GO annotation 文件不存在: %s — 将跳过富集分析", go_path)
        return {}

    df = pd.read_csv(go_path, sep="\t", comment="#", dtype=str)
    required_cols = [c for c in ["uniprot_id", "protein", "accession"] if c in df.columns]
    go_col = [c for c in ["go_term", "go_id", "GO_term"] if c in df.columns]

    if not required_cols or not go_col:
        log.error("GO 文件列名不匹配: 需要 [uniprot_id|protein|accession] + [go_term|go_id|GO_term]")
        log.error("实际列名: %s", list(df.columns))
        return {}

    id_col = required_cols[0]
    go_col = go_col[0]
    mapping = defaultdict(set)
    for _, row in df.iterrows():
        mapping[row[id_col]].add(row[go_col])

    log.info("GO annotation: %d proteins → %d unique GO terms",
             len(mapping), len(set(t for terms in mapping.values() for t in terms)))
    return dict(mapping)


def load_reactome_annotation(reactome_path: Path) -> dict[str, set[str]]:
    """加载 protein→Reactome pathway 映射。

    Args:
        reactome_path: TSV 文件

    Returns:
        dict: accession → set of pathway IDs
    """
    if not reactome_path or not Path(reactome_path).exists():
        log.warning("Reactome annotation 文件不存在: %s — 将跳过", reactome_path)
        return {}

    df = pd.read_csv(reactome_path, sep="\t", comment="#", dtype=str)
    id_col = [c for c in ["uniprot_id", "protein", "accession"] if c in df.columns]
    pw_col = [c for c in ["pathway_id", "reactome_id"] if c in df.columns]

    if not id_col or not pw_col:
        log.error("Reactome 文件列名不匹配")
        return {}

    mapping = defaultdict(set)
    for _, row in df.iterrows():
        mapping[row[id_col[0]]].add(row[pw_col[0]])
    return dict(mapping)


# ======================================================================
# 3. Build protein-to-relation association
# ======================================================================

def build_protein_relation_association(fold: int = 0) -> dict[str, dict[int, list[str]]]:
    """对于每个 relation type，找出关联的 protein pairs。

    从 fold_0 训练数据中提取: 哪些蛋白参与到哪些关系类型的三元组。

    Returns:
        dict: relation_id → {"heads": [protein_ids], "tails": [protein_ids]}
    """
    from sitne_walk.data import load_training_triples

    train_path = ROOT / f"05_splits/typed_ranking_v2/fold_{fold}/train.tsv"
    if not train_path.exists():
        log.warning("训练文件缺失: %s", train_path)
        return {}

    triples = load_training_triples(
        str(train_path), weight_column=None, duplicate_policy="binary", pair_semantics="unordered")

    # Map relation → set of proteins
    assoc = defaultdict(lambda: {"heads": set(), "tails": set()})
    for rel in range(triples.num_relations):
        mask = (triples.relations == rel).numpy()
        assoc[rel]["heads"] = {triples.id_to_protein[int(h)] for h in triples.heads[mask].numpy()}
        assoc[rel]["tails"] = {triples.id_to_protein[int(t)] for t in triples.tails[mask].numpy()}

    log.info("Protein-relation association: %d relations loaded from fold %d",
             len(assoc), fold)
    return dict(assoc)


# ======================================================================
# 4. Hypergeometric enrichment test
# ======================================================================

def hypergeometric_test(
    foreground: set[str],
    background: set[str],
    annotation: dict[str, set[str]],
    min_overlap: int = 2,
) -> pd.DataFrame:
    """对 foreground 蛋白做富集分析。

    Args:
        foreground: 目标蛋白集合 (如某 relation type 关联的蛋白)
        background: 背景蛋白集合 (所有已知蛋白)
        annotation: protein → set of terms (GO or pathway)

    Returns:
        DataFrame: term, overlap, p_value, FDR
    """
    if not annotation:
        return pd.DataFrame()

    foreground_annotated = {p for p in foreground if p in annotation}
    background_annotated = {p for p in background if p in annotation}

    N = len(background_annotated)  # total annotated proteins
    n = len(foreground_annotated)  # foreground annotated proteins

    if N == 0 or n == 0:
        return pd.DataFrame()

    # Build term → foreground count & background count
    all_terms_in_fg = set()
    for t_set in annotation.values():
        all_terms_in_fg |= t_set
    # Restrict to terms that appear in foreground
    term_fg_counts = defaultdict(int)
    for prot in foreground_annotated:
        for term in annotation.get(prot, set()):
            term_fg_counts[term] += 1

    term_bg_counts = defaultdict(int)
    for prot in background_annotated:
        for term in annotation.get(prot, set()):
            term_bg_counts[term] += 1

    results = []
    for term, k in term_fg_counts.items():
        if k < min_overlap:
            continue
        K = term_bg_counts.get(term, 0)
        if K == 0:
            continue
        # Hypergeometric: P(X >= k)
        p_value = stats.hypergeom.sf(k - 1, N, K, n)
        results.append({
            "term": term,
            "overlap": k,
            "term_frequency_in_bg": K,
            "foreground_total": n,
            "background_total": N,
            "p_value": p_value,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["FDR"] = _benjamini_hochberg(df["p_value"].values)
    df.sort_values("FDR", inplace=True)
    return df


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    ranks = np.arange(1, n + 1)
    adjusted = sorted_p * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.zeros(n)
    result[order] = adjusted
    return result


# ======================================================================
# 5. Main enrichment pipeline
# ======================================================================

def run_enrichment(
    relation_assoc: dict,
    relation_clusters: pd.DataFrame,
    go_mapping: dict[str, set[str]],
    reactome_mapping: dict[str, set[str]],
    output_dir: Path,
):
    """对每个 relation 和 cluster 做富集分析。"""
    # Background: all proteins in the union of all associations
    background = set()
    for assoc in relation_assoc.values():
        background |= assoc["heads"]
        background |= assoc["tails"]

    all_go_results = []
    all_reactome_results = []

    # Enrich by individual relation
    for rel_id, assoc in relation_assoc.items():
        fg_heads = assoc["heads"]
        fg_tails = assoc["tails"]
        fg_all = fg_heads | fg_tails

        if go_mapping:
            go_df = hypergeometric_test(fg_all, background, go_mapping)
            if not go_df.empty:
                go_df["relation_id"] = rel_id
                go_df["source"] = "per_relation"
                all_go_results.append(go_df)

        if reactome_mapping:
            react_df = hypergeometric_test(fg_all, background, reactome_mapping)
            if not react_df.empty:
                react_df["relation_id"] = rel_id
                react_df["source"] = "per_relation"
                all_reactome_results.append(react_df)

    # Enrich by cluster
    if not relation_clusters.empty:
        for cluster_id, group in relation_clusters.groupby("cluster_id"):
            rel_ids = group["relation_id"].tolist()
            fg_cluster = set()
            for rid in rel_ids:
                if rid in relation_assoc:
                    fg_cluster |= relation_assoc[rid]["heads"]
                    fg_cluster |= relation_assoc[rid]["tails"]

            if go_mapping:
                go_df = hypergeometric_test(fg_cluster, background, go_mapping)
                if not go_df.empty:
                    go_df["cluster_id"] = int(cluster_id)
                    go_df["source"] = "per_cluster"
                    all_go_results.append(go_df)

            if reactome_mapping:
                react_df = hypergeometric_test(fg_cluster, background, reactome_mapping)
                if not react_df.empty:
                    react_df["cluster_id"] = int(cluster_id)
                    react_df["source"] = "per_cluster"
                    all_reactome_results.append(react_df)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_go_results:
        go_combined = pd.concat(all_go_results, ignore_index=True)
        go_combined.sort_values("FDR", inplace=True)
        go_combined.to_csv(output_dir / "go_enrichment.tsv", sep="\t", index=False)
        log.info("GO enrichment: %d significant terms saved", len(go_combined))

    if all_reactome_results:
        react_combined = pd.concat(all_reactome_results, ignore_index=True)
        react_combined.sort_values("FDR", inplace=True)
        react_combined.to_csv(output_dir / "reactome_enrichment.tsv", sep="\t", index=False)
        log.info("Reactome enrichment: %d significant pathways saved", len(react_combined))

    if not all_go_results and not all_reactome_results:
        log.warning("未产生任何富集结果 (annotation 数据缺失或无效)")


# ======================================================================
# 6. Without GO data: output relation similarity matrix
# ======================================================================

def output_relation_similarity(embeddings: np.ndarray, clusters: pd.DataFrame, output_dir: Path):
    """输出 relation embedding 相似度矩阵和聚类结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cosine similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    sim_df = pd.DataFrame(sim_matrix)
    sim_df.index = [f"rel_{i}" for i in range(sim_matrix.shape[0])]
    sim_df.columns = [f"rel_{i}" for i in range(sim_matrix.shape[1])]
    sim_df.to_csv(output_dir / "relation_similarity.tsv", sep="\t")
    log.info("关系相似度矩阵: %s", output_dir / "relation_similarity.tsv")

    if not clusters.empty:
        clusters.to_csv(output_dir / "relation_clusters.tsv", sep="\t", index=False)
        log.info("关系聚类结果: %s", output_dir / "relation_clusters.tsv")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="GO/Reactome enrichment analysis")
    parser.add_argument("--go-annotation", type=str, default="",
                        help="Protein-to-GO term TSV 文件路径")
    parser.add_argument("--reactome-annotation", type=str, default="",
                        help="Protein-to-Reactome pathway TSV 文件路径")
    args = parser.parse_args()

    import torch  # needed for checkpoint loading
    log.info("=" * 60)

    # Step 1: Load relation embeddings
    checkpoints = find_optimal_checkpoints()
    if not checkpoints:
        log.error("未找到 optimal 模型文件，退出")
        sys.exit(1)

    log.info("找到 %d 个 optimal checkpoints", len(checkpoints))
    embeddings = load_relation_embeddings(checkpoints)

    if embeddings.size == 0:
        log.error("无法加载 relation embeddings")
        sys.exit(1)

    # Step 2: Cluster relations
    log.info("对 relation embeddings 聚类...")
    clusters = cluster_relations(embeddings, method="ward", max_k=5)
    output_relation_similarity(embeddings, clusters, OUT_DIR)

    # Step 3: Build protein-relation association
    log.info("构建 protein-relation 关联...")
    relation_assoc = build_protein_relation_association(fold=0)

    # Step 4: Load annotations & run enrichment
    go_mapping = {}
    reactome_mapping = {}

    if args.go_annotation:
        go_mapping = load_go_annotation(Path(args.go_annotation))

    if args.reactome_annotation:
        reactome_mapping = load_reactome_annotation(Path(args.reactome_annotation))

    if not go_mapping and not reactome_mapping:
        log.warning("=" * 60)
        log.warning("TODO(DATA_DEPENDENCY): 未提供 GO/Reactome annotation 文件。")
        log.warning("富集分析已跳过。请提供以下格式的 TSV 文件:")
        log.warning("  --go-annotation  : uniprot_id<TAB>go_term<TAB>go_namespace")
        log.warning("  --reactome-annotation: uniprot_id<TAB>pathway_id<TAB>pathway_name")
        log.warning("")
        log.warning("已输出的文件:")
        log.warning("  %s/relation_similarity.tsv — 4×4 余弦相似度", OUT_DIR)
        log.warning("  %s/relation_clusters.tsv  — 层次聚类归属", OUT_DIR)
        log.warning("=" * 60)
    else:
        run_enrichment(relation_assoc, clusters, go_mapping, reactome_mapping, OUT_DIR)

    log.info("T10 完成。")


if __name__ == "__main__":
    main()
