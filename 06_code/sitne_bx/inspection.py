"""Dataset B/B* 全量只读身份、hash、覆盖和隔离合同审计。"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .config import BXConfig
from .data import load_pair_table, load_sequences_tsv, load_typed_annotations
from .guarded_io import BSTAR_TRAIN_ALLOWLIST, B_SPLITS, GuardedDatasetIO, Phase


CANONICAL_COUNTS = {
    "train": (163192, 81596, 81596),
    "validation": (59260, 29630, 29630),
    "test": (52048, 26024, 26024),
}


def _json_from_guard(guard: GuardedDatasetIO, dataset: str, path: str) -> dict[str, Any]:
    value = json.loads(guard.read_text(dataset, path))
    if not isinstance(value, dict):
        raise ValueError(f"{dataset}:{path} 根节点不是 object")
    return value


def inspect_datasets(config: BXConfig, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    b_root = root / config.data.dataset_b_root
    x_root = root / config.data.dataset_bstar_root if config.data.dataset_bstar_root else None
    guard = GuardedDatasetIO(b_root, x_root, Phase.INSPECT)

    dataset_manifest = _json_from_guard(guard, "b", "manifest/dataset.json")
    if dataset_manifest.get("dataset_uid") != config.data.dataset_uid:
        raise ValueError("Dataset B UID 与配置不一致")
    validation = _json_from_guard(guard, "b", "reports/validation.json")
    if validation.get("status") != "PASS":
        raise ValueError("Dataset B 官方 validation status 不是 PASS")

    tables = {
        split: load_pair_table(
            guard, "b", relative, split, expected_uid=config.data.dataset_uid
        )
        for split, relative in B_SPLITS.items()
    }
    with guard.open_text("b", "manifest/split_manifest.tsv") as handle:
        split_manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
    split_manifest_by_role: dict[str, dict[str, str]] = {}
    for row in split_manifest_rows:
        role = str(row.get("split_role", ""))
        if role in split_manifest_by_role:
            raise ValueError(f"split_manifest 重复 split_role: {role}")
        split_manifest_by_role[role] = row
    if set(split_manifest_by_role) != set(B_SPLITS):
        raise ValueError("split_manifest 必须且只能包含 train/validation/test")
    split_stats: dict[str, Any] = {}
    for split, table in tables.items():
        positive = sum(row.label for row in table.records)
        negative = len(table.records) - positive
        manifest_stats = dataset_manifest.get("stats", {}).get(split, {})
        if (len(table.records), positive, negative) != (
            int(manifest_stats.get("rows", -1)),
            int(manifest_stats.get("positive", -1)),
            int(manifest_stats.get("negative", -1)),
        ):
            raise ValueError(f"Dataset B {split} 实测计数与 dataset.json 不一致")
        if config.data.dataset_uid == "BERNETT_V3_FULL_PAPER_SPLIT_V1":
            if (len(table.records), positive, negative) != CANONICAL_COUNTS[split]:
                raise ValueError(f"Dataset B canonical {split} 冻结计数不一致")
        split_stats[split] = {
            "rows": len(table.records),
            "positive": positive,
            "negative": negative,
            "proteins": len(table.accessions),
        }
        if config.data.verify_hashes:
            opened_split = next(
                item
                for item in guard.opened_paths
                if item["dataset"] == "b"
                and item["relative_path"] == B_SPLITS[split]
            )
            if opened_split["sha256"] != manifest_stats.get("output_sha256"):
                raise ValueError(f"Dataset B {split} SHA-256 与 dataset.json 不一致")
            if opened_split["bytes"] != manifest_stats.get("output_bytes"):
                raise ValueError(f"Dataset B {split} bytes 与 dataset.json 不一致")
            split_row = split_manifest_by_role[split]
            expected_from_split_manifest = {
                "rows": int(split_row["rows"]),
                "positive": int(split_row["positive"]),
                "negative": int(split_row["negative"]),
                "sha256": split_row["sha256"],
                "bytes": int(split_row["bytes"]),
            }
            actual_for_split_manifest = {
                "rows": len(table.records),
                "positive": positive,
                "negative": negative,
                "sha256": opened_split["sha256"],
                "bytes": opened_split["bytes"],
            }
            if actual_for_split_manifest != expected_from_split_manifest:
                raise ValueError(
                    f"Dataset B {split} 与 split_manifest.tsv 不一致"
                )

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        protein_overlap = tables[left].accessions & tables[right].accessions
        pair_overlap = {
            row.undirected_pair_key for row in tables[left].records
        } & {row.undirected_pair_key for row in tables[right].records}
        if protein_overlap or pair_overlap:
            raise ValueError(
                f"split 隔离失败 {left}/{right}: proteins={len(protein_overlap)}, "
                f"pairs={len(pair_overlap)}"
            )

    all_accessions = frozenset().union(*(table.accessions for table in tables.values()))
    sequences = load_sequences_tsv(
        guard, "b", "sequences/proteins.tsv", all_accessions
    )
    with guard.open_binary("b", "sequences/proteins.fasta"):
        pass
    if len(sequences) != len(all_accessions):
        raise AssertionError("序列覆盖内部不一致")
    if config.data.verify_hashes and config.data.dataset_uid == "BERNETT_V3_FULL_PAPER_SPLIT_V1":
        with guard.open_text("b", "manifest/asset_manifest.tsv") as handle:
            asset_rows = list(csv.DictReader(handle, delimiter="\t"))
        expected_assets: dict[str, dict[str, str]] = {}
        for row in asset_rows:
            for relative in (*B_SPLITS.values(), "sequences/proteins.tsv", "sequences/proteins.fasta"):
                if str(row.get("path", "")).endswith("/" + relative):
                    expected_assets[relative] = row
        required_assets = set(B_SPLITS.values()) | {
            "sequences/proteins.tsv",
            "sequences/proteins.fasta",
        }
        if set(expected_assets) != required_assets:
            raise ValueError("Dataset B asset_manifest 缺少 canonical split/sequence 资产")
        opened_by_relative = {
            str(item["relative_path"]): item
            for item in guard.opened_paths
            if item["dataset"] == "b"
        }
        for relative, expected in expected_assets.items():
            actual = opened_by_relative[relative]
            if actual["sha256"] != expected["sha256"] or actual["bytes"] != int(expected["bytes"]):
                raise ValueError(f"Dataset B asset_manifest hash/bytes 不匹配: {relative}")

    xstar_report: dict[str, Any] | None = None
    if x_root is not None:
        contract = _json_from_guard(guard, "bstar", "loader_contract.json")
        manifest = _json_from_guard(guard, "bstar", "views/train_input_manifest.json")
        build = _json_from_guard(guard, "bstar", "build_manifest.json")
        if manifest.get("dataset_uid") != config.data.dataset_uid:
            raise ValueError("B* parent dataset UID 不一致")
        if manifest.get("xstar_uid") != config.data.xstar_uid:
            raise ValueError("B* UID 不一致")
        allowed_contract = set(contract.get("allowed_training_inputs", []))
        allowed_manifest = {
            str(item.get("path")) for item in manifest.get("allowed_training_inputs", [])
        }
        if allowed_contract != BSTAR_TRAIN_ALLOWLIST or allowed_manifest != BSTAR_TRAIN_ALLOWLIST:
            raise ValueError("B* 七项 allow-list 与运行时冻结合同不一致")
        for item in manifest["allowed_training_inputs"]:
            relative = str(item["path"])
            with guard.open_binary("bstar", relative):
                pass
            opened = next(
                record
                for record in guard.opened_paths
                if record["dataset"] == "bstar" and record["relative_path"] == relative
            )
            if opened["sha256"] != item["sha256"] or opened["bytes"] != item["bytes"]:
                raise ValueError(f"B* allow-list 资产 hash/bytes 不匹配: {relative}")
        x_train = load_pair_table(
            guard, "bstar", "views/X/train.tsv", "train", expected_uid=None
        )
        b_train_identity = [
            (row.protocol_row_id, row.protein_a, row.protein_b, row.label)
            for row in tables["train"].records
        ]
        x_train_identity = [
            (row.protocol_row_id, row.protein_a, row.protein_b, row.label)
            for row in x_train.records
        ]
        if x_train_identity != b_train_identity:
            raise ValueError("B* X/train projection 与 Dataset B Train 不一致")
        train_sequences = load_sequences_tsv(
            guard, "bstar", "nodes/train_protein.tsv", x_train.accessions
        )
        fine = load_typed_annotations(guard, "fine", x_train)
        coarse = load_typed_annotations(guard, "coarse", x_train)
        if len(x_train.records) != int(manifest["train_pair_rows"]):
            raise ValueError("B* train pair count 不匹配")
        if fine.fact_count != int(
            next(
                item["rows"]
                for item in manifest["allowed_training_inputs"]
                if item["path"] == "edges/train_pair_type_fine.tsv"
            )
        ):
            raise ValueError("B* fine fact count 不匹配")
        if coarse.fact_count != int(
            next(
                item["rows"]
                for item in manifest["allowed_training_inputs"]
                if item["path"] == "edges/train_pair_type_coarse.tsv"
            )
        ):
            raise ValueError("B* coarse fact count 不匹配")
        if not bool(manifest.get("default_deny")):
            raise ValueError("B* manifest 未声明 default_deny")
        xstar_report = {
            "xstar_uid": config.data.xstar_uid,
            "build_status": build.get("build_status"),
            "runtime_loader_enforcement": "PASS_BY_SITNE_BX_GUARDED_IO",
            "train_rows": len(x_train.records),
            "train_proteins": len(train_sequences),
            "fine_facts": fine.fact_count,
            "fine_types": len(fine.type_vocabulary),
            "fine_covered_positive_pairs": len(fine.by_protocol_row_id),
            "coarse_facts": coarse.fact_count,
            "coarse_types": len(coarse.type_vocabulary),
            "coarse_covered_positive_pairs": len(coarse.by_protocol_row_id),
            "training_allowlist": sorted(BSTAR_TRAIN_ALLOWLIST),
            "targets_used_as_model_input": False,
        }

    return {
        "schema_version": "sitne-bx-inspection-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "dataset_b": {
            "dataset_uid": config.data.dataset_uid,
            "official_exactness_status": dataset_manifest.get("exactness_status"),
            "official_validation_status": validation.get("status"),
            "license_status": dataset_manifest.get("license_status"),
            "redistribution_status": dataset_manifest.get("redistribution_status"),
            "splits": split_stats,
            "all_required_sequences": len(sequences),
            "protein_overlap_between_splits": 0,
            "pair_overlap_between_splits": 0,
            "split_and_sequence_hashes_verified": config.data.verify_hashes,
        },
        "dataset_bstar": xstar_report,
        "opened_paths": guard.opened_paths,
        "scientific_evidence": False,
        "evidence_boundary": "本报告验证数据与运行时隔离，不是模型性能证据。",
    }
