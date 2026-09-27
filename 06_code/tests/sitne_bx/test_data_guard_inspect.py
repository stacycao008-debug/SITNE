from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sitne_bx.config import BXConfig, load_config
from sitne_bx.data import load_precomputed_npz, load_train_data, normalize_sequence
from sitne_bx.guarded_io import GuardedDatasetIO, Phase
from sitne_bx.inspection import inspect_datasets
from sitne_bx.runner import resolve_roots


def test_full_synthetic_inspection_and_train_allowlist(synthetic_project):
    root, config_path = synthetic_project
    config = load_config(config_path)
    report = inspect_datasets(config, root)
    assert report["status"] == "PASS"
    assert report["dataset_b"]["protein_overlap_between_splits"] == 0
    assert report["dataset_bstar"]["fine_facts"] == 3
    assert report["dataset_bstar"]["targets_used_as_model_input"] is False

    b_root, x_root = resolve_roots(config, root)
    guard = GuardedDatasetIO(b_root, x_root, Phase.TRAIN)
    pairs, sequences, typed = load_train_data(config, guard)
    assert len(pairs.records) == 6
    assert len(sequences) == 4
    assert typed.fact_count == 3
    opened = {item["relative_path"] for item in guard.opened_paths}
    assert opened == {
        "views/X/train.tsv",
        "nodes/train_protein.tsv",
        "edges/train_pair_type_fine.tsv",
    }
    with pytest.raises(PermissionError):
        guard.open_text("bstar", "targets/should_never_open.tsv")
    with pytest.raises(PermissionError):
        guard.open_text("b", "splits/test.tsv")
    with pytest.raises(PermissionError):
        guard.open_text("b", "sequences/proteins.tsv")

    select_guard = GuardedDatasetIO(b_root, x_root, Phase.SELECT)
    with pytest.raises(PermissionError):
        select_guard.open_text("bstar", "targets/should_never_open.tsv")
    test_without_capability = GuardedDatasetIO(b_root, x_root, Phase.TEST)
    with pytest.raises(PermissionError):
        test_without_capability.open_text("bstar", "targets/should_never_open.tsv")
    inspect_guard = GuardedDatasetIO(b_root, x_root, Phase.INSPECT)
    with pytest.raises(PermissionError):
        inspect_guard.open_text("bstar", "targets/should_never_open.tsv")
    with pytest.raises(PermissionError):
        guard.open_text("bstar", "build_manifest.json")


def test_u_to_x_and_unknown_normalization():
    assert normalize_sequence("acu bzjo*") == "ACXXXXXX"
    with pytest.raises(ValueError):
        normalize_sequence("ACD?")


def test_precomputed_npz_is_phase_exact_and_paths_are_distinct(tmp_path: Path):
    path = tmp_path / "train.npz"
    np.savez(
        path,
        accessions=np.asarray(["A", "EXTRA"]),
        embeddings=np.ones((2, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="当前阶段之外"):
        load_precomputed_npz(path, {"A"})
    with pytest.raises(ValueError, match="路径必须互不相同"):
        BXConfig.from_mapping(
            {
                "model": {
                    "backend": "precomputed_embedding",
                    "precomputed_dimension": 4,
                    "precomputed_train_path": "same.npz",
                    "precomputed_validation_path": "same.npz",
                    "precomputed_test_path": "test.npz",
                }
            }
        )


def test_inspect_rejects_tampered_split_manifest(synthetic_project):
    root, config_path = synthetic_project
    manifest_path = root / "02_data_canonical/dataset_B/v1/manifest/split_manifest.tsv"
    original = manifest_path.read_text(encoding="utf-8")
    original_hash = original.splitlines()[1].split("\t")[4]
    manifest_path.write_text(
        original.replace(original_hash, "f" * 64, 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="split_manifest"):
        inspect_datasets(load_config(config_path), root)
