"""基于 BlackBox typed v4 canonical 数据构建 pair-grouped 5-fold CV splits。

核心设计:
1. 合并 train+val+test 全部 positive pair-type facts (coarse type)
2. 按 canonical unordered protein pair 分组
3. GroupKFold(n_splits=5) 以 protein pair 为 group，确保同一 pair 的所有 MI types 
   原子进入同一 fold
4. 对每个 outer-training fold，内部再做 3-fold GroupKFold 生成 inner folds
5. 生成 all_known_positives.tsv (全量 pair-MI facts，用于 filtered evaluation filter)
6. 输出 manifest + audit 文件
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = ["protein_a", "protein_b", "coarse_type"]
OUTPUT_COLUMNS = ["protein_i", "protein_j", "type_name"]
REQUIRED_CANONICAL_COLUMNS = [
    "protocol_row_id", "paper_record_id", "protein_a", "protein_b",
    "split_role", "relation_level", "relation_id", "relation_name",
    "coarse_type", "supporting_event_count", "example_event_id", "seen_in_train",
]


@dataclass
class FoldStats:
    fold_id: int
    num_train_pairs: int
    num_train_facts: int
    num_train_proteins: int
    num_train_relations: int
    num_val_pairs: int
    num_val_facts: int
    num_val_proteins: int
    num_val_relations: int
    num_test_pairs: int
    num_test_facts: int
    num_test_proteins: int
    num_test_relations: int
    val_protein_coverage: float  # val proteins that appear in train
    test_protein_coverage: float  # test proteins that appear in train
    val_relation_coverage: float
    test_relation_coverage: float


@dataclass
class SplitManifest:
    split_uid: str
    canonical_source: str
    n_outer_folds: int
    n_inner_folds: int
    split_seed: int
    total_pairs: int
    total_facts: int
    total_proteins: int
    total_relations: int
    per_fold_stats: list[FoldStats] = field(default_factory=list)
    all_known_positives_hash: str = ""
    coverage_audit: dict = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_coarse_facts(canonical_dir: Path) -> pd.DataFrame:
    """从 canonical 数据加载 coarse type 级别的 pair-type facts。

    合并 train + validation + test 三个 coarse edge 文件。
    只保留 positive facts (seen_in_train=true 或 split_role 为非 train 的).
    实际上所有 coarse 文件里的行都是 positive。
    """
    train_path = canonical_dir / "edges" / "train_pair_type_coarse.tsv"
    val_path = canonical_dir / "targets" / "validation_pair_type_coarse.tsv"
    test_path = canonical_dir / "targets" / "test_pair_type_coarse.tsv"

    frames = []
    for path, split_role in [(train_path, "train"), (val_path, "validation"), (test_path, "test")]:
        if not path.is_file():
            raise FileNotFoundError(f"Canonical 文件不存在: {path}")
        df = pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
            dtype={col: str for col in CANONICAL_COLUMNS},
            keep_default_na=False,
        )
        df["_source_split"] = split_role
        frames.append(df)
        logger.info("加载 %s: %d 行 (来源=%s)", path.name, len(df), split_role)

    all_facts = pd.concat(frames, ignore_index=True)
    logger.info("合并后共 %d 条 coarse pair-type facts", len(all_facts))
    return all_facts


def _build_pair_groups(facts: pd.DataFrame) -> pd.DataFrame:
    """构建 canonical unordered pair key 并去重 pair-type。

    同一个 pair + type 可能出现在多个 split 中（如 train 和 val 共享了部分事实），
    这里保留第一次出现。
    """
    # 构建 canonical unordered pair key
    low = np.minimum(facts["protein_a"].to_numpy(), facts["protein_b"].to_numpy())
    high = np.maximum(facts["protein_a"].to_numpy(), facts["protein_b"].to_numpy())
    facts = facts.copy()
    facts["protein_i"] = low
    facts["protein_j"] = high
    facts["type_name"] = facts["coarse_type"].astype(str).str.strip()
    facts["pair_key"] = facts["protein_i"] + "|||" + facts["protein_j"]

    # 去重: 同一个 (protein_i, protein_j, type_name) 只保留一次
    before = len(facts)
    facts = facts.drop_duplicates(
        subset=["protein_i", "protein_j", "type_name"], keep="first"
    )
    after = len(facts)
    if before != after:
        logger.info("去重后: %d -> %d 条 unique pair-type facts (去除 %d 条重复)", before, after, before - after)

    # 按 pair_key 分组，得到每个 pair 的所有 types
    pair_groups = facts.groupby("pair_key", sort=False)
    pair_info = pair_groups.agg(
        num_types=("type_name", "nunique"),
        types_list=("type_name", lambda x: "|".join(sorted(set(x)))),
    ).reset_index()

    logger.info("共 %d 个 unique canonical pairs, %d 个 unique pair-type facts",
                len(pair_info), len(facts))
    return facts, pair_info


def _generate_outer_folds(
    facts: pd.DataFrame,
    pair_info: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 20240101,
) -> tuple[list[list[str]], pd.DataFrame]:
    """用 GroupKFold 生成 outer folds。

    group = canonical protein pair key，确保同一 pair 的所有 types 进入同一 fold。
    返回每个 fold 的 pair_key 列表和标注了 fold_id 的 pair_info。
    """
    gkf = GroupKFold(n_splits=n_splits)
    # pair_info 按 pair_key 排序确保可复现
    pair_info = pair_info.sort_values("pair_key").reset_index(drop=True)
    pair_keys = pair_info["pair_key"].to_numpy()
    # group labels: 每个 pair 自己就是 group
    group_labels = np.arange(len(pair_keys))

    fold_assignments = np.full(len(pair_keys), -1, dtype=np.int32)
    fold_pairs: list[list[str]] = []

    for fold_idx, (_, test_indices) in enumerate(gkf.split(pair_keys, groups=group_labels)):
        fold_assignments[test_indices] = fold_idx
        fold_pairs.append(pair_keys[test_indices].tolist())

    pair_info["outer_fold"] = fold_assignments

    for fold_idx in range(n_splits):
        n = len(fold_pairs[fold_idx])
        logger.info("Outer fold %d: %d pairs", fold_idx, n)

    # 验证 pair overlap=0
    for i in range(n_splits):
        set_i = set(fold_pairs[i])
        for j in range(i + 1, n_splits):
            overlap = len(set_i & set(fold_pairs[j]))
            if overlap:
                raise ValueError(f"Outer fold {i} 和 {j} 有 {overlap} 个重叠 pair!")

    return fold_pairs, pair_info


def _generate_inner_folds(
    train_pairs: list[str],
    pair_info: pd.DataFrame,
    n_inner: int = 3,
    random_state: int = 20240101,
) -> list[tuple[list[str], list[str]]]:
    """对 outer-training pairs 做内部 GroupKFold。

    返回 [(inner_train_pairs, inner_val_pairs), ...]
    """
    train_pair_info = pair_info[pair_info["pair_key"].isin(train_pairs)].copy()
    train_pair_info = train_pair_info.sort_values("pair_key").reset_index(drop=True)
    pair_keys = train_pair_info["pair_key"].to_numpy()

    if len(pair_keys) < n_inner:
        raise ValueError(f"Inner fold 要求至少 {n_inner} 个 training pair，实际 {len(pair_keys)}")

    gkf = GroupKFold(n_splits=n_inner)
    group_labels = np.arange(len(pair_keys))

    inner_folds = []
    for _, (train_idx, val_idx) in enumerate(gkf.split(pair_keys, groups=group_labels)):
        inner_train = pair_keys[train_idx].tolist()
        inner_val = pair_keys[val_idx].tolist()
        inner_folds.append((inner_train, inner_val))
        logger.info("  Inner fold: train=%d pairs, val=%d pairs", len(inner_train), len(inner_val))

    return inner_folds


def _write_fold_files(
    facts: pd.DataFrame,
    pair_info: pd.DataFrame,
    fold_idx: int,
    test_pairs: list[str],
    output_dir: Path,
    inner_folds_data: list[tuple[list[str], list[str]]] | None = None,
) -> FoldStats:
    """为一个 outer fold 写 train/val/test TSV 文件。

    - test_pairs: 当前 fold 的 outer-test pairs
    - train_pairs: 剩余所有 pairs
    - val: 从 train pairs 中按 paper_record_id 分层采样 10%，或使用 inner fold
    """
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    test_mask = pair_info["pair_key"].isin(test_pairs)
    train_mask = ~test_mask

    train_pair_keys = pair_info.loc[train_mask, "pair_key"]
    test_pair_keys = pair_info.loc[test_mask, "pair_key"]

    # 写 test: 全部 test pairs 的 facts
    test_facts = facts[facts["pair_key"].isin(test_pair_keys)].copy()
    test_output = test_facts[["protein_i", "protein_j", "type_name"]].drop_duplicates()
    test_output.to_csv(fold_dir / "test.tsv", sep="\t", index=False)

    # 写 train: 全部 train pairs 的 facts
    train_facts = facts[facts["pair_key"].isin(train_pair_keys)].copy()
    train_output = train_facts[["protein_i", "protein_j", "type_name"]].drop_duplicates()
    train_output.to_csv(fold_dir / "train.tsv", sep="\t", index=False)

    # val: 使用 paper_record_id 分层采样 10% 的 train pairs
    # 从 train_facts 中提取 paper_record_id 信息
    train_with_record = facts[
        facts["pair_key"].isin(train_pair_keys) & facts["paper_record_id"].notna()
    ]
    # 按 paper_record_id 分组采样
    if "_source_split" in facts.columns:
        train_with_record = train_with_record[train_with_record["_source_split"] == "train"]

    # 按 pair 分组，随机选 10% 的 pairs 作为 val
    unique_train_pairs = train_pair_keys.unique()
    np.random.seed(20240101 + fold_idx)
    n_val_pairs = max(1, int(len(unique_train_pairs) * 0.10))
    val_pair_keys = np.random.choice(unique_train_pairs, size=n_val_pairs, replace=False)
    val_pair_set = set(val_pair_keys)

    val_facts = facts[facts["pair_key"].isin(val_pair_set)].copy()
    val_output = val_facts[["protein_i", "protein_j", "type_name"]].drop_duplicates()
    val_output.to_csv(fold_dir / "val.tsv", sep="\t", index=False)

    # 从 train 中移除 val pairs
    train_output_final = train_output[~train_output.apply(
        lambda row: (min(row["protein_i"], row["protein_j"]) + "|||" +
                     max(row["protein_i"], row["protein_j"])) in val_pair_set,
        axis=1
    )]
    train_output_final.to_csv(fold_dir / "train.tsv", sep="\t", index=False)

    # Inner folds
    if inner_folds_data:
        inner_dir = fold_dir / "inner_folds"
        inner_dir.mkdir(parents=True, exist_ok=True)
        for inner_idx, (inner_train_keys, inner_val_keys) in enumerate(inner_folds_data):
            inner_fold_dir = inner_dir / f"fold_{inner_idx}"
            inner_fold_dir.mkdir(parents=True, exist_ok=True)
            inner_train = facts[facts["pair_key"].isin(inner_train_keys)]
            inner_val = facts[facts["pair_key"].isin(inner_val_keys)]
            inner_train[["protein_i", "protein_j", "type_name"]].drop_duplicates().to_csv(
                inner_fold_dir / "train.tsv", sep="\t", index=False
            )
            inner_val[["protein_i", "protein_j", "type_name"]].drop_duplicates().to_csv(
                inner_fold_dir / "val.tsv", sep="\t", index=False
            )

    # 统计
    train_proteins = set(train_output_final["protein_i"]) | set(train_output_final["protein_j"])
    val_proteins = set(val_output["protein_i"]) | set(val_output["protein_j"])
    test_proteins = set(test_output["protein_i"]) | set(test_output["protein_j"])
    train_relations = set(train_output_final["type_name"])
    val_relations = set(val_output["type_name"])
    test_relations = set(test_output["type_name"])

    val_prot_cov = len(val_proteins & train_proteins) / max(len(val_proteins), 1)
    test_prot_cov = len(test_proteins & train_proteins) / max(len(test_proteins), 1)
    val_rel_cov = len(val_relations & train_relations) / max(len(val_relations), 1)
    test_rel_cov = len(test_relations & train_relations) / max(len(test_relations), 1)

    stats = FoldStats(
        fold_id=fold_idx,
        num_train_pairs=len(set(zip(train_output_final["protein_i"], train_output_final["protein_j"]))),
        num_train_facts=len(train_output_final),
        num_train_proteins=len(train_proteins),
        num_train_relations=len(train_relations),
        num_val_pairs=len(set(zip(val_output["protein_i"], val_output["protein_j"]))),
        num_val_facts=len(val_output),
        num_val_proteins=len(val_proteins),
        num_val_relations=len(val_relations),
        num_test_pairs=len(set(zip(test_output["protein_i"], test_output["protein_j"]))),
        num_test_facts=len(test_output),
        num_test_proteins=len(test_proteins),
        num_test_relations=len(test_relations),
        val_protein_coverage=round(val_prot_cov, 4),
        test_protein_coverage=round(test_prot_cov, 4),
        val_relation_coverage=round(val_rel_cov, 4),
        test_relation_coverage=round(test_rel_cov, 4),
    )

    logger.info(
        "Fold %d: train=%d pairs/%d facts, val=%d/%d, test=%d/%d | "
        "prot_cov: val=%.1f%% test=%.1f%% | rel_cov: val=%.1f%% test=%.1f%%",
        fold_idx,
        stats.num_train_pairs, stats.num_train_facts,
        stats.num_val_pairs, stats.num_val_facts,
        stats.num_test_pairs, stats.num_test_facts,
        stats.val_protein_coverage * 100, stats.test_protein_coverage * 100,
        stats.val_relation_coverage * 100, stats.test_relation_coverage * 100,
    )

    return stats


def _generate_all_known_positives(facts: pd.DataFrame, output_dir: Path) -> str:
    """生成 all_known_positives.tsv: 全量 unique pair-type facts。"""
    path = output_dir / "all_known_positives.tsv"
    ap = facts[["protein_i", "protein_j", "type_name"]].drop_duplicates()
    ap.to_csv(path, sep="\t", index=False)
    sha = _sha256_file(path)
    logger.info("all_known_positives.tsv: %d unique facts, SHA256=%s", len(ap), sha[:16])
    return sha


def _audit_coverage(
    facts: pd.DataFrame,
    pair_info: pd.DataFrame,
    output_dir: Path,
    fold_pairs: list[list[str]],
    per_fold_stats: list[FoldStats],
) -> dict:
    """生成 coverage audit 报告。"""
    audit = {
        "total_unique_pairs": int(pair_info["pair_key"].nunique()),
        "total_unique_facts": int(len(facts[["protein_i", "protein_j", "type_name"]].drop_duplicates())),
        "total_proteins": int(len(set(facts["protein_i"]) | set(facts["protein_j"]))),
        "total_relations": int(facts["type_name"].nunique()),
        "relation_list": sorted(facts["type_name"].unique().tolist()),
        "per_fold": [],
    }

    # 检查 pair overlap
    for i in range(len(fold_pairs)):
        set_i = set(fold_pairs[i])
        for j in range(i + 1, len(fold_pairs)):
            overlap = len(set_i & set(fold_pairs[j]))
            if overlap:
                audit["pair_overlap_error"] = f"Fold {i} and {j} overlap: {overlap} pairs"
                logger.error(audit["pair_overlap_error"])

    # 检查 protein coverage
    all_coverage_ok = True
    for stats in per_fold_stats:
        fold_audit = {
            "fold": stats.fold_id,
            "val_protein_coverage": stats.val_protein_coverage,
            "test_protein_coverage": stats.test_protein_coverage,
            "val_relation_coverage": stats.val_relation_coverage,
            "test_relation_coverage": stats.test_relation_coverage,
        }
        if stats.val_protein_coverage < 1.0:
            fold_audit["val_protein_coverage_warning"] = f"Below 100%: {stats.val_protein_coverage}"
            all_coverage_ok = False
        if stats.test_protein_coverage < 1.0:
            fold_audit["test_protein_coverage_warning"] = f"Below 100%: {stats.test_protein_coverage}"
            all_coverage_ok = False
        if stats.val_relation_coverage < 1.0:
            fold_audit["val_relation_coverage_warning"] = f"Below 100%: {stats.val_relation_coverage}"
            all_coverage_ok = False
        if stats.test_relation_coverage < 1.0:
            fold_audit["test_relation_coverage_warning"] = f"Below 100%: {stats.test_relation_coverage}"
            all_coverage_ok = False
        audit["per_fold"].append(fold_audit)

    audit["all_coverage_100"] = all_coverage_ok
    if not all_coverage_ok:
        logger.warning("部分 fold 的 protein/relation coverage 未达到 100%!")

    audit_path = output_dir / "split_audit.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    return audit


def build_pair_grouped_splits(
    canonical_data_dir: str | Path,
    output_dir: str | Path,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    split_seed: int = 20240101,
) -> SplitManifest:
    """从 BlackBox typed v4 canonical 数据构建 pair-grouped CV splits。

    Args:
        canonical_data_dir: canonical 数据目录 (dataset_Bstar/v2/)
        output_dir: 输出目录 (05_splits/typed_ranking_v2/)
        n_outer_folds: outer CV fold 数
        n_inner_folds: 每个 outer-training fold 的 inner fold 数
        split_seed: 随机种子

    Returns:
        SplitManifest 包含全部统计和审计信息
    """
    canonical_dir = Path(canonical_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 coarse facts
    logger.info("=== Step 1: 加载 canonical coarse facts ===")
    facts = _load_coarse_facts(canonical_dir)

    # 2. 构建 pair groups
    logger.info("=== Step 2: 构建 pair groups ===")
    facts, pair_info = _build_pair_groups(facts)

    # 3. 生成 outer folds
    logger.info("=== Step 3: 生成 outer folds (GroupKFold n=%d, seed=%d) ===", n_outer_folds, split_seed)
    np.random.seed(split_seed)
    fold_pairs, pair_info = _generate_outer_folds(
        facts, pair_info, n_splits=n_outer_folds, random_state=split_seed
    )

    # 4. 为每个 fold 写文件
    logger.info("=== Step 4: 写 fold 文件 ===")
    per_fold_stats: list[FoldStats] = []
    for fold_idx in range(n_outer_folds):
        test_pairs = fold_pairs[fold_idx]
        train_pairs = [
            p for i, pairs in enumerate(fold_pairs) if i != fold_idx
            for p in pairs
        ]

        # 生成 inner folds
        inner_folds_data = _generate_inner_folds(
            train_pairs, pair_info, n_inner=n_inner_folds, random_state=split_seed
        )

        stats = _write_fold_files(
            facts, pair_info, fold_idx, test_pairs, output_dir, inner_folds_data
        )
        per_fold_stats.append(stats)

    # 5. 生成 all_known_positives.tsv
    logger.info("=== Step 5: 生成 all_known_positives.tsv ===")
    ap_hash = _generate_all_known_positives(facts, output_dir)

    # 6. Audit
    logger.info("=== Step 6: Coverage audit ===")
    audit = _audit_coverage(facts, pair_info, output_dir, fold_pairs, per_fold_stats)

    # 7. 生成 manifest
    logger.info("=== Step 7: 生成 manifest ===")
    manifest = SplitManifest(
        split_uid=f"typed_ranking_v2_seed{split_seed}",
        canonical_source=str(canonical_dir.resolve()),
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        split_seed=split_seed,
        total_pairs=int(pair_info["pair_key"].nunique()),
        total_facts=int(len(facts[["protein_i", "protein_j", "type_name"]].drop_duplicates())),
        total_proteins=int(len(set(facts["protein_i"]) | set(facts["protein_j"]))),
        total_relations=int(facts["type_name"].nunique()),
        per_fold_stats=per_fold_stats,
        all_known_positives_hash=ap_hash,
        coverage_audit=audit,
    )

    manifest_dict = {
        "split_uid": manifest.split_uid,
        "canonical_source": manifest.canonical_source,
        "n_outer_folds": manifest.n_outer_folds,
        "n_inner_folds": manifest.n_inner_folds,
        "split_seed": manifest.split_seed,
        "total_pairs": manifest.total_pairs,
        "total_facts": manifest.total_facts,
        "total_proteins": manifest.total_proteins,
        "total_relations": manifest.total_relations,
        "all_known_positives_sha256": manifest.all_known_positives_hash,
        "coverage_audit": manifest.coverage_audit,
        "per_fold": [
            {
                "fold": s.fold_id,
                "num_train_pairs": s.num_train_pairs,
                "num_train_facts": s.num_train_facts,
                "num_train_proteins": s.num_train_proteins,
                "num_train_relations": s.num_train_relations,
                "num_val_pairs": s.num_val_pairs,
                "num_val_facts": s.num_val_facts,
                "num_val_proteins": s.num_val_proteins,
                "num_val_relations": s.num_val_relations,
                "num_test_pairs": s.num_test_pairs,
                "num_test_facts": s.num_test_facts,
                "num_test_proteins": s.num_test_proteins,
                "num_test_relations": s.num_test_relations,
                "val_protein_coverage": s.val_protein_coverage,
                "test_protein_coverage": s.test_protein_coverage,
                "val_relation_coverage": s.val_relation_coverage,
                "test_relation_coverage": s.test_relation_coverage,
            }
            for s in per_fold_stats
        ],
    }
    manifest_path = output_dir / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2, ensure_ascii=False)

    logger.info("=== Split Builder 完成 ===")
    logger.info("输出目录: %s", output_dir)
    logger.info("总 pairs: %d, 总 facts: %d, 总 proteins: %d, 总 relations: %d",
                manifest.total_pairs, manifest.total_facts,
                manifest.total_proteins, manifest.total_relations)
    if audit.get("all_coverage_100"):
        logger.info("所有 fold 的 protein 和 relation coverage 均为 100%")
    else:
        logger.warning("存在 coverage 不达标的 fold，请检查 audit 报告")

    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import sys
    canonical = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "02_data_canonical" / "dataset_Bstar" / "v2")
    output = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[2] / "05_splits" / "typed_ranking_v2")
    build_pair_grouped_splits(canonical, output)
