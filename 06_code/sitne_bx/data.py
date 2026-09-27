"""Dataset B/B* loader、序列解析器与 phase-specific feature provider。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from .config import BXConfig
from .guarded_io import B_SPLITS, GuardedDatasetIO, Phase


PAIR_COLUMNS = {
    "protocol_row_id",
    "protein_a_raw",
    "protein_b_raw",
    "label",
    "split_role",
    "undirected_pair_key",
}
TYPED_COLUMNS = {
    "protocol_row_id",
    "protein_a",
    "protein_b",
    "split_role",
    "relation_level",
    "relation_id",
    "seen_in_train",
}
SEQUENCE_TRANSLATION = str.maketrans({char: "X" for char in "UBZJO*"})
VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWYX")


@dataclass(frozen=True, slots=True)
class PairRecord:
    protocol_row_id: str
    protein_a: str
    protein_b: str
    label: int
    split_role: str
    undirected_pair_key: str


@dataclass(frozen=True)
class PairTable:
    split_role: str
    records: tuple[PairRecord, ...]

    @property
    def labels(self) -> np.ndarray:
        return np.fromiter((row.label for row in self.records), dtype=np.int64)

    @property
    def accessions(self) -> frozenset[str]:
        return frozenset(
            protein
            for row in self.records
            for protein in (row.protein_a, row.protein_b)
        )


@dataclass(frozen=True)
class TypedAnnotations:
    level: str
    type_vocabulary: tuple[str, ...]
    by_protocol_row_id: Mapping[str, tuple[int, ...]]
    fact_count: int

    @classmethod
    def empty(cls) -> "TypedAnnotations":
        return cls("none", (), {}, 0)


def normalize_sequence(raw: str) -> str:
    sequence = "".join(raw.split()).upper().translate(SEQUENCE_TRANSLATION)
    if not sequence:
        raise ValueError("蛋白质序列不能为空")
    unexpected = sorted(set(sequence) - VALID_AA)
    if unexpected:
        raise ValueError(f"序列包含不支持字符: {unexpected}")
    return sequence


def _reader(handle: Iterable[str], required: set[str], name: str) -> csv.DictReader:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = set(reader.fieldnames or ())
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"{name} 缺少必需列: {missing}")
    return reader


def load_pair_table(
    guard: GuardedDatasetIO,
    dataset: str,
    relative_path: str,
    expected_split: str,
    expected_uid: str | None = None,
) -> PairTable:
    records: list[PairRecord] = []
    seen_ids: set[str] = set()
    seen_pairs: set[str] = set()
    with guard.open_text(dataset, relative_path) as handle:
        reader = _reader(handle, PAIR_COLUMNS, relative_path)
        has_uid = "dataset_uid" in set(reader.fieldnames or ())
        for line_number, row in enumerate(reader, 2):
            split = row["split_role"].strip().lower()
            if split != expected_split:
                raise ValueError(
                    f"{relative_path}:{line_number} split_role={split!r}，"
                    f"预期 {expected_split!r}"
                )
            if has_uid and expected_uid and row["dataset_uid"] != expected_uid:
                raise ValueError(
                    f"{relative_path}:{line_number} dataset_uid 不匹配"
                )
            try:
                label = int(row["label"])
            except ValueError as exc:
                raise ValueError(f"{relative_path}:{line_number} label 非整数") from exc
            if label not in (0, 1):
                raise ValueError(f"{relative_path}:{line_number} label 必须为 0/1")
            row_id = row["protocol_row_id"].strip()
            protein_a = row["protein_a_raw"].strip()
            protein_b = row["protein_b_raw"].strip()
            pair_key = row["undirected_pair_key"].strip()
            expected_key = "|".join(sorted((protein_a, protein_b)))
            if not row_id or not protein_a or not protein_b:
                raise ValueError(f"{relative_path}:{line_number} 存在空身份字段")
            if pair_key != expected_key:
                raise ValueError(f"{relative_path}:{line_number} undirected_pair_key 错误")
            if row_id in seen_ids:
                raise ValueError(f"{relative_path} 重复 protocol_row_id: {row_id}")
            if pair_key in seen_pairs:
                raise ValueError(f"{relative_path} 重复无向 pair: {pair_key}")
            seen_ids.add(row_id)
            seen_pairs.add(pair_key)
            records.append(
                PairRecord(row_id, protein_a, protein_b, label, split, pair_key)
            )
    if not records:
        raise ValueError(f"{relative_path} 没有数据行")
    return PairTable(expected_split, tuple(records))


def load_sequences_tsv(
    guard: GuardedDatasetIO,
    dataset: str,
    relative_path: str,
    required_accessions: set[str] | frozenset[str] | None = None,
) -> dict[str, str]:
    required = set(required_accessions) if required_accessions is not None else None
    sequences: dict[str, str] = {}
    with guard.open_text(dataset, relative_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        accession_column = "accession" if "accession" in fields else "protein_id"
        missing = {accession_column, "sequence"} - fields
        if missing:
            raise ValueError(f"{relative_path} 缺少序列列: {sorted(missing)}")
        for line_number, row in enumerate(reader, 2):
            accession = row[accession_column].strip()
            if required is not None and accession not in required:
                continue
            if accession in sequences:
                raise ValueError(f"{relative_path} 重复 accession: {accession}")
            sequence = normalize_sequence(row["sequence"])
            if row.get("sequence_length") and int(row["sequence_length"]) != len(sequence):
                raise ValueError(f"{relative_path}:{line_number} sequence_length 不匹配")
            if row.get("length") and int(row["length"]) != len(sequence):
                raise ValueError(f"{relative_path}:{line_number} length 不匹配")
            sequences[accession] = sequence
    if required is not None:
        missing_accessions = sorted(required - sequences.keys())
        if missing_accessions:
            preview = missing_accessions[:10]
            raise ValueError(
                f"{relative_path} 缺少 {len(missing_accessions)} 个所需序列: {preview}"
            )
    return sequences


def load_typed_annotations(
    guard: GuardedDatasetIO,
    level: str,
    train_pairs: PairTable,
) -> TypedAnnotations:
    if level == "none":
        return TypedAnnotations.empty()
    relative_path = f"edges/train_pair_type_{level}.tsv"
    train_by_id = {row.protocol_row_id: row for row in train_pairs.records}
    raw: dict[str, set[str]] = {}
    fact_count = 0
    with guard.open_text("bstar", relative_path) as handle:
        reader = _reader(handle, TYPED_COLUMNS, relative_path)
        for line_number, row in enumerate(reader, 2):
            if row["split_role"].strip().lower() != "train":
                raise ValueError(f"{relative_path}:{line_number} 暴露非 train typed fact")
            if row["relation_level"].strip().lower() != level:
                raise ValueError(f"{relative_path}:{line_number} relation_level 不匹配")
            if row["seen_in_train"].strip().lower() != "true":
                raise ValueError(f"{relative_path}:{line_number} seen_in_train 必须为 true")
            row_id = row["protocol_row_id"].strip()
            pair = train_by_id.get(row_id)
            if pair is None:
                raise ValueError(f"typed fact 引用了非 Train row: {row_id}")
            if pair.label != 1:
                raise ValueError(f"typed fact 引用了 label=0 row: {row_id}")
            if {pair.protein_a, pair.protein_b} != {
                row["protein_a"].strip(),
                row["protein_b"].strip(),
            }:
                raise ValueError(f"{relative_path}:{line_number} pair endpoint 不一致")
            relation_id = row["relation_id"].strip()
            if not relation_id:
                raise ValueError(f"{relative_path}:{line_number} relation_id 为空")
            raw.setdefault(row_id, set()).add(relation_id)
            fact_count += 1
    vocabulary = tuple(sorted({relation for values in raw.values() for relation in values}))
    index = {relation: idx for idx, relation in enumerate(vocabulary)}
    encoded = {
        row_id: tuple(sorted(index[relation] for relation in relations))
        for row_id, relations in raw.items()
    }
    if not vocabulary:
        raise ValueError(f"{relative_path} 没有 typed fact")
    return TypedAnnotations(level, vocabulary, encoded, fact_count)


def load_train_data(
    config: BXConfig, guard: GuardedDatasetIO
) -> tuple[PairTable, dict[str, str], TypedAnnotations]:
    if guard.phase != Phase.TRAIN:
        raise ValueError("load_train_data 只能在 train phase 调用")
    if not config.data.dataset_bstar_root:
        raise ValueError("Train 必须使用 B* 七项 allow-list；dataset_bstar_root 不能为空")
    pairs = load_pair_table(
        guard,
        "bstar",
        "views/X/train.tsv",
        "train",
        expected_uid=None,
    )
    sequences = load_sequences_tsv(
        guard, "bstar", "nodes/train_protein.tsv", pairs.accessions
    )
    typed = load_typed_annotations(guard, config.data.typed_level, pairs)
    return pairs, sequences, typed


def load_evaluation_data(
    config: BXConfig, guard: GuardedDatasetIO, split_role: str
) -> tuple[PairTable, dict[str, str]]:
    if split_role not in {"validation", "test"}:
        raise ValueError("评测 split 只能是 validation 或 test")
    expected_phase = Phase.SELECT if split_role == "validation" else Phase.TEST
    if guard.phase != expected_phase:
        raise ValueError(
            f"加载 {split_role} 需要 phase={expected_phase.value}，"
            f"当前为 {guard.phase.value}"
        )
    pairs = load_pair_table(
        guard,
        "b",
        B_SPLITS[split_role],
        split_role,
        config.data.dataset_uid,
    )
    sequences = load_sequences_tsv(
        guard, "b", "sequences/proteins.tsv", pairs.accessions
    )
    return pairs, sequences


class FeatureProvider:
    """按 accession 提供原始序列或冻结的预计算表示。"""

    def __init__(self, features: Mapping[str, str] | Mapping[str, np.ndarray], backend: str):
        self.features = dict(features)
        self.backend = backend

    def pair_payload(
        self, records: Sequence[PairRecord], device: torch.device
    ) -> tuple[list[str] | torch.Tensor, torch.Tensor, torch.Tensor]:
        unique: list[str] = []
        index: dict[str, int] = {}
        left: list[int] = []
        right: list[int] = []
        for row in records:
            for accession, target in ((row.protein_a, left), (row.protein_b, right)):
                if accession not in self.features:
                    raise KeyError(f"缺少 feature: {accession}")
                if accession not in index:
                    index[accession] = len(unique)
                    unique.append(accession)
                target.append(index[accession])
        if self.backend == "chunked_sequence":
            payload: list[str] | torch.Tensor = [str(self.features[key]) for key in unique]
        elif self.backend == "precomputed_embedding":
            payload = torch.as_tensor(
                np.stack([np.asarray(self.features[key], dtype=np.float32) for key in unique]),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise ValueError(f"未知 backend: {self.backend}")
        return (
            payload,
            torch.tensor(left, dtype=torch.long, device=device),
            torch.tensor(right, dtype=torch.long, device=device),
        )


def load_precomputed_npz(path: str | Path, required: set[str]) -> dict[str, np.ndarray]:
    """读取不执行任意代码的 NPZ；要求 accessions/embeddings 一一对应。"""

    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != {"accessions", "embeddings"}:
            raise ValueError("预计算 NPZ 必须且只能包含 accessions、embeddings")
        accessions = archive["accessions"]
        embeddings = archive["embeddings"]
    if accessions.ndim != 1 or embeddings.ndim != 2 or len(accessions) != len(embeddings):
        raise ValueError("预计算 embedding 维度不合法")
    result: dict[str, np.ndarray] = {}
    for accession, vector in zip(accessions.tolist(), embeddings, strict=True):
        key = str(accession)
        if key in result:
            raise ValueError(f"预计算 NPZ 重复 accession: {key}")
        if not np.isfinite(vector).all():
            raise ValueError(f"预计算 embedding 含非有限值: {key}")
        result[key] = np.asarray(vector, dtype=np.float32)
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f"预计算 NPZ 缺少 {len(missing)} 个所需 accession: {missing[:10]}")
    unexpected = sorted(result.keys() - required)
    if unexpected:
        raise ValueError(
            f"预计算 NPZ 包含 {len(unexpected)} 个当前阶段之外的 accession: "
            f"{unexpected[:10]}"
        )
    return {key: result[key] for key in required}


def resolve_feature_provider(
    config: BXConfig,
    split_role: str,
    sequences: Mapping[str, str],
    project_root: str | Path,
) -> FeatureProvider:
    if config.model.backend == "chunked_sequence":
        return FeatureProvider(sequences, "chunked_sequence")
    phase_name = "train" if split_role == "train" else split_role
    configured = getattr(config.model, f"precomputed_{phase_name}_path")
    if configured is None:
        raise ValueError(f"缺少 {phase_name} precomputed path")
    path = Path(configured)
    if not path.is_absolute():
        path = Path(project_root) / path
    features = load_precomputed_npz(path, set(sequences))
    expected_dim = config.model.precomputed_dimension
    actual_dims = {vector.shape[0] for vector in features.values()}
    if actual_dims != {expected_dim}:
        raise ValueError(
            f"预计算维度不匹配: expected={expected_dim}, actual={sorted(actual_dims)}"
        )
    return FeatureProvider(features, "precomputed_embedding")


def typed_targets_for_batch(
    records: Sequence[PairRecord],
    annotations: TypedAnnotations,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回均匀 multi-target 分布和 mask；无 typed fact 的行完全不进 typed loss。"""

    targets = torch.zeros(
        (len(records), len(annotations.type_vocabulary)),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros(len(records), dtype=torch.bool, device=device)
    for row_index, row in enumerate(records):
        types = annotations.by_protocol_row_id.get(row.protocol_row_id, ())
        if types:
            mask[row_index] = True
            weight = 1.0 / len(types)
            targets[row_index, list(types)] = weight
    return targets, mask


def iter_batches(
    records: Sequence[PairRecord],
    batch_size: int,
    order: Sequence[int] | None = None,
) -> Iterator[tuple[PairRecord, ...]]:
    indices = list(range(len(records))) if order is None else list(order)
    for start in range(0, len(indices), batch_size):
        yield tuple(records[index] for index in indices[start : start + batch_size])
