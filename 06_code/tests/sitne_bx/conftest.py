from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from sitne_bx.guarded_io import BSTAR_TRAIN_ALLOWLIST


PAIR_HEADER = [
    "protocol_row_id",
    "dataset_uid",
    "dataset_letter_alias",
    "paper_record_id",
    "author_source_file",
    "author_source_line",
    "protein_a_raw",
    "protein_b_raw",
    "label",
    "partition",
    "split_role",
    "source_version",
    "source_file_sha256",
    "author_row_sha256",
    "undirected_pair_key",
    "selection_rule",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _pair_rows(uid: str, split: str, pairs: list[tuple[str, str, int]]) -> list[list[object]]:
    rows = []
    partition = {"train": "Intra1", "validation": "Intra0", "test": "Intra2"}[split]
    for index, (left, right, label) in enumerate(pairs, 1):
        row_id = f"{uid}:{split}:{index:06d}"
        rows.append(
            [
                row_id,
                uid,
                "B",
                f"paper:{split}:{index}",
                "source.tsv",
                index + 1,
                left,
                right,
                label,
                partition,
                split,
                "synthetic",
                "0" * 64,
                "1" * 64,
                "|".join(sorted((left, right))),
                "synthetic test only",
            ]
        )
    return rows


@pytest.fixture
def synthetic_project(tmp_path: Path) -> tuple[Path, Path]:
    uid = "SYNTHETIC_B_V1"
    xuid = "SYNTHETIC_BSTAR_V2"
    b = tmp_path / "02_data_canonical/dataset_B/v1"
    x = tmp_path / "02_data_canonical/dataset_Bstar/v2"
    train_pairs = [
        ("A", "B", 1),
        ("B", "C", 1),
        ("C", "D", 1),
        ("A", "C", 0),
        ("A", "D", 0),
        ("B", "D", 0),
    ]
    validation_pairs = [("E", "F", 1), ("E", "G", 0), ("F", "G", 1), ("E", "H", 0)]
    test_pairs = [("I", "J", 1), ("I", "K", 0), ("J", "L", 1), ("K", "L", 0)]
    split_pairs = {"train": train_pairs, "validation": validation_pairs, "test": test_pairs}
    stats = {}
    for split, pairs in split_pairs.items():
        path = b / f"splits/{split}.tsv"
        _write_tsv(path, PAIR_HEADER, _pair_rows(uid, split, pairs))
        stats[split] = {
            "rows": len(pairs),
            "positive": sum(label for _, _, label in pairs),
            "negative": sum(1 - label for _, _, label in pairs),
            "output_sha256": _sha(path),
            "output_bytes": path.stat().st_size,
        }
    sequences = {
        accession: ("ACDU" + "G" * index)
        for index, accession in enumerate("ABCDEFGHIJKL", 1)
    }
    _write_tsv(
        b / "sequences/proteins.tsv",
        ["accession", "sequence", "length", "sequence_sha256", "sequence_source"],
        [
            [key, value, len(value), hashlib.sha256(value.replace("U", "X").encode()).hexdigest(), "synthetic"]
            for key, value in sequences.items()
        ],
    )
    (b / "sequences/proteins.fasta").write_text(
        "".join(f">{key}\n{value}\n" for key, value in sequences.items()), encoding="utf-8"
    )
    (b / "manifest").mkdir(parents=True)
    (b / "manifest/dataset.json").write_text(
        json.dumps(
            {
                "dataset_uid": uid,
                "exactness_status": "SYNTHETIC_TEST",
                "license_status": "TEST_ONLY",
                "redistribution_status": "TEST_ONLY",
                "stats": stats,
            }
        ),
        encoding="utf-8",
    )
    _write_tsv(
        b / "manifest/split_manifest.tsv",
        ["split_role", "rows", "positive", "negative", "sha256", "bytes"],
        [
            [split, value["rows"], value["positive"], value["negative"], value["output_sha256"], value["output_bytes"]]
            for split, value in stats.items()
        ],
    )
    _write_tsv(b / "manifest/asset_manifest.tsv", ["path", "sha256"], [])
    (b / "reports").mkdir(parents=True)
    (b / "reports/validation.json").write_text(
        json.dumps({"dataset_uid": uid, "status": "PASS"}), encoding="utf-8"
    )

    # B* train projection exactly preserves B Train identity/order.
    x_rows = _pair_rows(uid, "train", train_pairs)
    x_header = [
        "protocol_row_id",
        "paper_record_id",
        "protein_a_raw",
        "protein_b_raw",
        "label",
        "partition",
        "split_role",
        "undirected_pair_key",
    ]
    header_index = {name: PAIR_HEADER.index(name) for name in x_header}
    _write_tsv(
        x / "views/X/train.tsv",
        x_header,
        [[row[header_index[name]] for name in x_header] for row in x_rows],
    )
    train_sequences = {key: sequences[key] for key in "ABCD"}
    _write_tsv(
        x / "nodes/train_protein.tsv",
        ["protein_id", "sequence", "sequence_length", "sequence_sha256", "sequence_source"],
        [
            [key, value, len(value), hashlib.sha256(value.replace("U", "X").encode()).hexdigest(), "synthetic"]
            for key, value in train_sequences.items()
        ],
    )
    (x / "nodes/train_proteins.fasta").write_text(
        "".join(f">{key}\n{value}\n" for key, value in train_sequences.items()), encoding="utf-8"
    )
    _write_tsv(
        x / "nodes/train_type_registry.tsv",
        ["mi_id", "fine_type", "coarse_type", "ontology_name", "mapping_status", "mapping_rule", "train_embedding_policy"],
        [["MI:1", "binding", "physical", "binding", "ACCEPTED", "synthetic", "TRAINABLE"], ["MI:2", "reaction", "enzymatic", "reaction", "ACCEPTED", "synthetic", "TRAINABLE"]],
    )
    typed_header = [
        "protocol_row_id", "paper_record_id", "protein_a", "protein_b", "split_role",
        "relation_level", "relation_id", "relation_name", "coarse_type",
        "supporting_event_count", "example_event_id", "seen_in_train",
    ]
    positive_rows = x_rows[:3]
    fine_rows = []
    coarse_rows = []
    for index, row in enumerate(positive_rows):
        relation = "MI:1" if index < 2 else "MI:2"
        fine_rows.append([row[0], row[3], row[6], row[7], "train", "fine", relation, relation, "physical", 1, f"event:{index}", "true"])
        coarse_rows.append([row[0], row[3], row[6], row[7], "train", "coarse", "physical", "physical", "physical", 1, f"event:{index}", "true"])
    _write_tsv(x / "edges/train_pair_type_fine.tsv", typed_header, fine_rows)
    _write_tsv(x / "edges/train_pair_type_coarse.tsv", typed_header, coarse_rows)
    _write_tsv(
        x / "views/X_type/model_input_event_membership.tsv",
        ["event_id", "protocol_row_id", "split_role", "visibility"],
        [["event:0", positive_rows[0][0], "train", "TRAIN_MODEL_INPUT"]],
    )
    (x / "targets").mkdir(parents=True)
    (x / "targets/should_never_open.tsv").write_text("secret\n", encoding="utf-8")
    manifest_entries = []
    for relative in sorted(BSTAR_TRAIN_ALLOWLIST):
        path = x / relative
        rows = max(path.read_text(encoding="utf-8").count("\n") - 1, 0)
        manifest_entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
                "rows": rows,
            }
        )
    (x / "loader_contract.json").write_text(
        json.dumps(
            {
                "allowed_training_inputs": sorted(BSTAR_TRAIN_ALLOWLIST),
                "forbidden_during_training": ["targets/**", "views/X/validation.tsv", "views/X/test.tsv"],
                "input_policy": "DEFAULT_DENY_EXPLICIT_ALLOWLIST",
            }
        ), encoding="utf-8"
    )
    (x / "views/train_input_manifest.json").write_text(
        json.dumps(
            {
                "dataset_uid": uid,
                "xstar_uid": xuid,
                "default_deny": True,
                "source_train_split_sha256": stats["train"]["output_sha256"],
                "train_pair_rows": len(train_pairs),
                "allowed_training_inputs": manifest_entries,
            }
        ), encoding="utf-8"
    )
    (x / "build_manifest.json").write_text(
        json.dumps({"dataset_uid": uid, "xstar_uid": xuid, "build_status": "SYNTHETIC_PASS"}),
        encoding="utf-8",
    )

    config = {
        "data": {
            "dataset_b_root": "02_data_canonical/dataset_B/v1",
            "dataset_bstar_root": "02_data_canonical/dataset_Bstar/v2",
            "dataset_uid": uid,
            "xstar_uid": xuid,
            "typed_level": "fine",
            "verify_hashes": True,
        },
        "model": {
            "backend": "chunked_sequence",
            "embedding_dim": 8,
            "token_embedding_dim": 6,
            "pair_hidden_dim": 10,
            "chunk_size": 5,
            "chunk_stride": 3,
            "chunk_batch_size": 4,
            "convolution_kernel_size": 3,
            "dropout": 0.0,
        },
        "loss": {"binary_weight": 1.0, "topology_weight": 0.1, "typed_weight": 0.2},
        "topology": {"walk_length": 4, "walks_per_node": 1, "context_window": 1, "context_batch_size": 2},
        "training": {
            "device": "cpu",
            "allow_cpu_for_testing": True,
            "seed": 7,
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "use_amp": False,
            "amp_dtype": "float16",
            "deterministic": True,
            "allow_tf32": False,
            "max_batches_per_epoch": None,
            "output_dir": "results",
        },
        "evaluation": {
            "batch_size": 2,
            "selection_metric": "auprc",
            "threshold_metric": "mcc",
            "expected_test_rows": 4,
        },
    }
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return tmp_path, config_path
