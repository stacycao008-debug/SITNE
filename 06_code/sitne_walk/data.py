"""SITNE-Walk 的数据读取、索引与 filtered-positive 数据结构。

科研边界：

* 蛋白质和关系词表只由训练集建立；
* 验证/测试中的未知蛋白质或未知关系不会被偷偷加入 embedding 表；
* 训练阶段的 relation-corruption 只过滤训练阳性；
* 评测阶段可使用单独传入的 all-known-positive 索引做“仅过滤”，该索引不参与
  walk、degree、frequency 或梯度计算。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

REQUIRED_TRIPLE_COLUMNS = ("protein_i", "protein_j", "type_name")
VALID_RELATION_MODES = {"as_recorded", "directed", "symmetric"}


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 SHA-256，避免把大型 TSV 一次读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_triple_frame(
    path: str | Path,
    weight_column: str | None = None,
) -> pd.DataFrame:
    """读取并严格检查三元组 TSV。"""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"三元组文件不存在: {input_path}")

    # opaque 生物实体 ID 必须保留原字面值：默认 dtype 推断会把 ``001`` 变成
    # ``1``，默认 NA 解析还会把合法字符串 ``NA`` 当缺失。
    frame = pd.read_csv(
        input_path,
        sep="\t",
        low_memory=False,
        dtype={column: str for column in REQUIRED_TRIPLE_COLUMNS},
        keep_default_na=False,
    )
    required = set(REQUIRED_TRIPLE_COLUMNS)
    if weight_column:
        required.add(weight_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{input_path} 缺少必要列: {missing}")
    if frame.empty:
        raise ValueError(f"三元组文件为空: {input_path}")

    selected = list(REQUIRED_TRIPLE_COLUMNS)
    if weight_column:
        selected.append(weight_column)
    frame = frame[selected].copy()

    if frame[list(REQUIRED_TRIPLE_COLUMNS)].isna().any(axis=None):
        bad_rows = int(frame[list(REQUIRED_TRIPLE_COLUMNS)].isna().any(axis=1).sum())
        raise ValueError(f"{input_path} 有 {bad_rows} 行关键字段为空")

    for column in REQUIRED_TRIPLE_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
        empty_count = int((frame[column] == "").sum())
        if empty_count:
            raise ValueError(f"{input_path} 的 {column} 有 {empty_count} 个空字符串")

    if weight_column:
        frame[weight_column] = pd.to_numeric(frame[weight_column], errors="raise")
        weights = frame[weight_column].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(f"{input_path} 的 {weight_column} 必须是有限正数")
    return frame


def _deduplicate_training_frame(
    frame: pd.DataFrame,
    duplicate_policy: str,
    weight_column: str | None,
) -> pd.DataFrame:
    """按明确策略处理完全相同的 (head, relation, tail) 记录。"""

    key_columns = list(REQUIRED_TRIPLE_COLUMNS)
    duplicate_mask = frame.duplicated(key_columns, keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows and duplicate_policy == "error":
        raise ValueError(
            f"训练数据含 {duplicate_rows} 行重复三元组，duplicate_policy=error"
        )

    if duplicate_policy == "binary":
        result = frame.drop_duplicates(key_columns, keep="first").copy()
        result["_edge_weight"] = np.float32(1.0)
        return result

    if duplicate_policy == "sum":
        weighted = frame.copy()
        if weight_column:
            weighted["_edge_weight"] = weighted[weight_column].astype(np.float64)
        else:
            weighted["_edge_weight"] = np.float64(1.0)
        return (
            weighted.groupby(key_columns, sort=False, as_index=False)["_edge_weight"]
            .sum()
            .reset_index(drop=True)
        )

    # duplicate_policy=error 且没有重复时，边权仍按输入或二值方式生成。
    result = frame.copy()
    if weight_column:
        result["_edge_weight"] = result[weight_column].astype(np.float64)
    else:
        result["_edge_weight"] = np.float32(1.0)
    return result


@dataclass(frozen=True)
class PairRelationIndex:
    """以排序 pair key 保存多标签阳性关系集合。

    对当前 44 个关系类型，dense bool mask 比 Python ``dict[tuple, set]`` 更适合
    批量 GPU 查询，也更容易检查训练/评测过滤边界。
    """

    pair_keys: torch.Tensor
    relation_mask: torch.Tensor
    num_proteins: int
    num_relations: int
    pair_semantics: str

    @classmethod
    def build(
        cls,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        num_proteins: int,
        num_relations: int,
        pair_semantics: str = "unordered",
    ) -> tuple["PairRelationIndex", torch.Tensor]:
        if num_proteins <= 0 or num_relations <= 0:
            raise ValueError("num_proteins 和 num_relations 必须为正")
        if not (heads.ndim == relations.ndim == tails.ndim == 1):
            raise ValueError("heads、relations、tails 必须是一维张量")
        if not (len(heads) == len(relations) == len(tails)):
            raise ValueError("heads、relations、tails 长度必须一致")
        if len(heads) == 0:
            raise ValueError("不能从空三元组构建 PairRelationIndex")
        if pair_semantics not in {"unordered", "ordered"}:
            raise ValueError("pair_semantics 必须是 unordered 或 ordered")

        heads_np = heads.detach().cpu().numpy().astype(np.int64, copy=False)
        tails_np = tails.detach().cpu().numpy().astype(np.int64, copy=False)
        relations_np = relations.detach().cpu().numpy().astype(np.int64, copy=False)
        if (
            np.any(heads_np < 0)
            or np.any(heads_np >= num_proteins)
            or np.any(tails_np < 0)
            or np.any(tails_np >= num_proteins)
        ):
            raise ValueError("PairRelationIndex 的 protein ID 越界")
        if np.any(relations_np < 0) or np.any(relations_np >= num_relations):
            raise ValueError("PairRelationIndex 的 relation ID 越界")
        if pair_semantics == "unordered":
            low = np.minimum(heads_np, tails_np)
            high = np.maximum(heads_np, tails_np)
            keys = low * np.int64(num_proteins) + high
        else:
            keys = heads_np * np.int64(num_proteins) + tails_np
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        mask = np.zeros((len(unique_keys), num_relations), dtype=np.bool_)
        mask[inverse, relations_np] = True

        index = cls(
            pair_keys=torch.from_numpy(unique_keys),
            relation_mask=torch.from_numpy(mask),
            num_proteins=num_proteins,
            num_relations=num_relations,
            pair_semantics=pair_semantics,
        )
        return index, torch.from_numpy(inverse.astype(np.int64, copy=False))

    def to(self, device: torch.device | str) -> "PairRelationIndex":
        """把索引迁移到目标设备；原对象保持不变。"""

        return PairRelationIndex(
            pair_keys=self.pair_keys.to(device),
            relation_mask=self.relation_mask.to(device),
            num_proteins=self.num_proteins,
            num_relations=self.num_relations,
            pair_semantics=self.pair_semantics,
        )

    def lookup(
        self,
        heads: torch.Tensor,
        tails: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回每个 pair 的关系 mask 和是否命中的布尔标记。

        未命中的 pair 返回全 False mask。输入张量必须与索引位于同一设备，避免
        在热点路径发生隐式 CPU/GPU 拷贝。
        """

        if heads.shape != tails.shape:
            raise ValueError("heads 与 tails 形状必须一致")
        if heads.device != self.pair_keys.device or tails.device != self.pair_keys.device:
            raise ValueError("查询张量与 PairRelationIndex 必须位于同一设备")

        flat_heads = heads.reshape(-1).to(torch.int64)
        flat_tails = tails.reshape(-1).to(torch.int64)
        if bool(
            (
                (flat_heads < 0)
                | (flat_heads >= self.num_proteins)
                | (flat_tails < 0)
                | (flat_tails >= self.num_proteins)
            ).any()
        ):
            raise ValueError("PairRelationIndex 查询中的 protein ID 越界")
        if self.pair_semantics == "unordered":
            low = torch.minimum(flat_heads, flat_tails)
            high = torch.maximum(flat_heads, flat_tails)
            keys = low * self.num_proteins + high
        else:
            keys = flat_heads * self.num_proteins + flat_tails
        positions = torch.searchsorted(self.pair_keys, keys)
        in_range = positions < self.pair_keys.numel()
        safe_positions = positions.clamp(max=max(self.pair_keys.numel() - 1, 0))
        matched = in_range & (self.pair_keys[safe_positions] == keys)

        masks = torch.zeros(
            (keys.numel(), self.num_relations),
            dtype=torch.bool,
            device=keys.device,
        )
        if bool(matched.any()):
            masks[matched] = self.relation_mask[safe_positions[matched]]
        output_shape = (*heads.shape, self.num_relations)
        return masks.reshape(output_shape), matched.reshape(heads.shape)


@dataclass(frozen=True)
class TrainingTriples:
    """训练集建立的整数化三元组及词表。"""

    heads: torch.Tensor
    relations: torch.Tensor
    tails: torch.Tensor
    edge_weights: torch.Tensor
    pair_ids: torch.Tensor
    pair_index: PairRelationIndex
    protein_to_id: Mapping[str, int]
    relation_to_id: Mapping[str, int]
    id_to_protein: tuple[str, ...]
    id_to_relation: tuple[str, ...]
    source_path: str
    source_sha256: str
    input_rows: int
    unique_rows: int
    pair_semantics: str

    @property
    def num_proteins(self) -> int:
        return len(self.id_to_protein)

    @property
    def num_relations(self) -> int:
        return len(self.id_to_relation)

    @property
    def num_triples(self) -> int:
        return int(self.heads.numel())


@dataclass(frozen=True)
class IndexedTriples:
    """按训练词表映射后的验证、测试或过滤三元组。"""

    heads: torch.Tensor
    relations: torch.Tensor
    tails: torch.Tensor
    source_path: str
    raw_input_rows: int
    input_rows: int
    supported_rows: int
    unsupported_protein_rows: int
    unsupported_relation_rows: int
    pair_semantics: str

    @property
    def num_triples(self) -> int:
        return int(self.heads.numel())


def load_training_triples(
    path: str | Path,
    duplicate_policy: str = "binary",
    weight_column: str | None = None,
    pair_semantics: str = "unordered",
) -> TrainingTriples:
    """从训练 TSV 建立唯一词表和训练阳性索引。"""

    if pair_semantics not in {"unordered", "ordered"}:
        raise ValueError("pair_semantics 必须是 unordered 或 ordered")
    raw = _read_triple_frame(path, weight_column=weight_column)
    input_rows = len(raw)
    if pair_semantics == "unordered":
        # 项目当前 split 以无序 canonical pair 分组。先统一端点再去重，确保同一
        # pair 的多标签不会因为输入方向不同而进入两个 filtered mask。
        low = np.minimum(raw["protein_i"].to_numpy(), raw["protein_j"].to_numpy())
        high = np.maximum(raw["protein_i"].to_numpy(), raw["protein_j"].to_numpy())
        raw["protein_i"] = low
        raw["protein_j"] = high
    frame = _deduplicate_training_frame(raw, duplicate_policy, weight_column)

    proteins = tuple(
        sorted(set(frame["protein_i"].tolist()) | set(frame["protein_j"].tolist()))
    )
    relations = tuple(sorted(frame["type_name"].unique().tolist()))
    protein_to_id = {value: index for index, value in enumerate(proteins)}
    relation_to_id = {value: index for index, value in enumerate(relations)}

    heads = torch.from_numpy(
        frame["protein_i"].map(protein_to_id).to_numpy(dtype=np.int64)
    )
    tails = torch.from_numpy(
        frame["protein_j"].map(protein_to_id).to_numpy(dtype=np.int64)
    )
    relation_ids = torch.from_numpy(
        frame["type_name"].map(relation_to_id).to_numpy(dtype=np.int64)
    )
    edge_weights = torch.from_numpy(
        frame["_edge_weight"].to_numpy(dtype=np.float32)
    )
    pair_index, pair_ids = PairRelationIndex.build(
        heads,
        relation_ids,
        tails,
        num_proteins=len(proteins),
        num_relations=len(relations),
        pair_semantics=pair_semantics,
    )

    logger.info(
        "训练集索引完成: input=%d unique=%d proteins=%d relations=%d pairs=%d",
        input_rows,
        len(frame),
        len(proteins),
        len(relations),
        pair_index.pair_keys.numel(),
    )
    return TrainingTriples(
        heads=heads,
        relations=relation_ids,
        tails=tails,
        edge_weights=edge_weights,
        pair_ids=pair_ids,
        pair_index=pair_index,
        protein_to_id=protein_to_id,
        relation_to_id=relation_to_id,
        id_to_protein=proteins,
        id_to_relation=relations,
        source_path=str(Path(path).resolve()),
        source_sha256=sha256_file(path),
        input_rows=input_rows,
        unique_rows=len(frame),
        pair_semantics=pair_semantics,
    )


def load_indexed_triples(
    path: str | Path,
    protein_to_id: Mapping[str, int],
    relation_to_id: Mapping[str, int],
    pair_semantics: str = "unordered",
) -> IndexedTriples:
    """使用训练词表映射其他 split，并显式统计 transductive 不支持项。"""

    if pair_semantics not in {"unordered", "ordered"}:
        raise ValueError("pair_semantics 必须是 unordered 或 ordered")
    frame = _read_triple_frame(path)
    raw_input_rows = len(frame)
    if pair_semantics == "unordered":
        low = np.minimum(frame["protein_i"].to_numpy(), frame["protein_j"].to_numpy())
        high = np.maximum(frame["protein_i"].to_numpy(), frame["protein_j"].to_numpy())
        frame["protein_i"] = low
        frame["protein_j"] = high
    frame = frame.drop_duplicates(list(REQUIRED_TRIPLE_COLUMNS), keep="first")
    input_rows = len(frame)
    mapped_heads = frame["protein_i"].map(protein_to_id)
    mapped_tails = frame["protein_j"].map(protein_to_id)
    mapped_relations = frame["type_name"].map(relation_to_id)

    unsupported_protein = mapped_heads.isna() | mapped_tails.isna()
    unsupported_relation = mapped_relations.isna()
    supported = ~(unsupported_protein | unsupported_relation)

    return IndexedTriples(
        heads=torch.from_numpy(mapped_heads[supported].to_numpy(dtype=np.int64)),
        relations=torch.from_numpy(
            mapped_relations[supported].to_numpy(dtype=np.int64)
        ),
        tails=torch.from_numpy(mapped_tails[supported].to_numpy(dtype=np.int64)),
        source_path=str(Path(path).resolve()),
        raw_input_rows=raw_input_rows,
        input_rows=input_rows,
        supported_rows=int(supported.sum()),
        unsupported_protein_rows=int(unsupported_protein.sum()),
        unsupported_relation_rows=int(unsupported_relation.sum()),
        pair_semantics=pair_semantics,
    )


def merge_indexed_triples(parts: Sequence[IndexedTriples]) -> IndexedTriples:
    """合并多个只用于过滤的三元组集合。"""

    if not parts:
        raise ValueError("parts 不能为空")
    pair_semantics = {part.pair_semantics for part in parts}
    if len(pair_semantics) != 1:
        raise ValueError("不能合并 pair_semantics 不一致的 IndexedTriples")
    return IndexedTriples(
        heads=torch.cat([part.heads for part in parts]),
        relations=torch.cat([part.relations for part in parts]),
        tails=torch.cat([part.tails for part in parts]),
        source_path=";".join(part.source_path for part in parts),
        raw_input_rows=sum(part.raw_input_rows for part in parts),
        input_rows=sum(part.input_rows for part in parts),
        supported_rows=sum(part.supported_rows for part in parts),
        unsupported_protein_rows=sum(part.unsupported_protein_rows for part in parts),
        unsupported_relation_rows=sum(part.unsupported_relation_rows for part in parts),
        pair_semantics=parts[0].pair_semantics,
    )


def build_known_positive_index(
    triples: IndexedTriples | TrainingTriples,
    num_proteins: int,
    num_relations: int,
) -> PairRelationIndex:
    """从明确提供的阳性集合建立 filtered-ranking 索引。"""

    index, _ = PairRelationIndex.build(
        triples.heads,
        triples.relations,
        triples.tails,
        num_proteins=num_proteins,
        num_relations=num_relations,
        pair_semantics=triples.pair_semantics,
    )
    return index


def validate_split_manifest(
    manifest_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path | None = None,
    test_path: str | Path | None = None,
    require_hashes: bool = False,
) -> dict[str, object]:
    """核对 split manifest 的行数与可选 SHA-256。

    只要 manifest 声称的计数或 hash 与实体文件冲突就立即报错。支持顶层
    ``train_sha256`` 等字段，或 ``files.train.sha256`` 形式；不根据目录名称猜测
    split 类型。
    """

    manifest_file = Path(manifest_path)
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("split manifest 根节点必须是 JSON object")

    paths = {
        "train": Path(train_path),
        "val": Path(validation_path) if validation_path else None,
        "test": Path(test_path) if test_path else None,
    }
    count_fields = {"train": "num_train", "val": "num_val", "test": "num_test"}
    hash_fields = {
        "train": "train_sha256",
        "val": "validation_sha256",
        "test": "test_sha256",
    }
    report: dict[str, object] = {"manifest_path": str(manifest_file.resolve())}
    files_section = manifest.get("files", {})
    for split_name, path in paths.items():
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"split 文件不存在: {path}")
        # wc -l 等价逻辑：首行为 header，因此 pandas 行数就是事实条目数。
        actual_count = len(pd.read_csv(path, sep="\t", usecols=["protein_i"]))
        claimed_count = manifest.get(count_fields[split_name])
        if claimed_count is not None and int(claimed_count) != actual_count:
            raise ValueError(
                f"manifest {split_name} 行数冲突: claimed={claimed_count}, "
                f"actual={actual_count}, file={path}"
            )
        nested = files_section.get(split_name, {}) if isinstance(files_section, dict) else {}
        claimed_hash = manifest.get(hash_fields[split_name])
        if claimed_hash is None and isinstance(nested, dict):
            claimed_hash = nested.get("sha256")
        if require_hashes and not claimed_hash:
            raise ValueError(f"manifest 缺少 {split_name} SHA-256")
        actual_hash = sha256_file(path)
        if claimed_hash and str(claimed_hash).lower() != actual_hash:
            raise ValueError(
                f"manifest {split_name} SHA-256 冲突: claimed={claimed_hash}, "
                f"actual={actual_hash}"
            )
        report[split_name] = {
            "path": str(path.resolve()),
            "rows": actual_count,
            "sha256": actual_hash,
        }
    return report


def audit_split_pair_overlap(
    train_path: str | Path,
    validation_path: str | Path | None = None,
    test_path: str | Path | None = None,
) -> dict[str, int]:
    """按 canonical protein pair 检查 split 交集。

    使用实际 TSV，不依赖 manifest 中的行政标签。任何交集都会抛出异常。
    """

    def pair_set(path: str | Path) -> set[tuple[str, str]]:
        frame = pd.read_csv(
            path,
            sep="\t",
            usecols=["protein_i", "protein_j"],
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
        if frame[["protein_i", "protein_j"]].isna().any(axis=None):
            raise ValueError(f"split pair 审计发现空 endpoint: {path}")
        for column in ("protein_i", "protein_j"):
            frame[column] = frame[column].astype(str).str.strip()
            if bool((frame[column] == "").any()):
                raise ValueError(f"split pair 审计发现空 endpoint: {path}")
        low = np.minimum(frame["protein_i"].to_numpy(), frame["protein_j"].to_numpy())
        high = np.maximum(frame["protein_i"].to_numpy(), frame["protein_j"].to_numpy())
        return set(zip(low.tolist(), high.tolist()))

    sets: dict[str, set[tuple[str, str]]] = {"train": pair_set(train_path)}
    if validation_path:
        sets["validation"] = pair_set(validation_path)
    if test_path:
        sets["test"] = pair_set(test_path)
    report: dict[str, int] = {
        f"num_pairs_{name}": len(values) for name, values in sets.items()
    }
    names = list(sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = len(sets[left_name] & sets[right_name])
            report[f"pair_overlap_{left_name}_{right_name}"] = overlap
            if overlap:
                raise ValueError(
                    f"split canonical pair overlap: {left_name}/{right_name}={overlap}"
                )
    return report


def load_relation_modes(
    path: str | Path | None,
    id_to_relation: Sequence[str],
    default_mode: str = "as_recorded",
) -> tuple[str, ...]:
    """读取关系方向模式 TSV。

    文件必须含 ``type_name`` 与 ``mode`` 两列。未列出的训练关系使用
    ``default_mode``；代码不会从正反边出现情况推断方向性。
    """

    if default_mode not in VALID_RELATION_MODES:
        raise ValueError(f"未知 default relation mode: {default_mode}")
    modes = {relation: default_mode for relation in id_to_relation}
    if path is None:
        return tuple(modes[relation] for relation in id_to_relation)

    metadata_path = Path(path)
    frame = pd.read_csv(
        metadata_path,
        sep="\t",
        low_memory=False,
        dtype={"type_name": str, "mode": str},
        keep_default_na=False,
    )
    required = {"type_name", "mode"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{metadata_path} 缺少列: {missing}")
    if frame["type_name"].duplicated().any():
        raise ValueError(f"{metadata_path} 含重复 type_name")

    for row in frame.itertuples(index=False):
        relation_name = str(getattr(row, "type_name")).strip()
        mode = str(getattr(row, "mode")).strip().lower()
        if mode not in VALID_RELATION_MODES:
            raise ValueError(
                f"关系 {relation_name!r} 的 mode={mode!r} 非法；"
                f"允许值为 {sorted(VALID_RELATION_MODES)}"
            )
        if relation_name in modes:
            modes[relation_name] = mode
        else:
            logger.warning("方向元数据中的关系不在训练词表中，已忽略: %s", relation_name)
    return tuple(modes[relation] for relation in id_to_relation)


def load_hierarchy_edges(
    path: str | Path,
    relation_to_id: Mapping[str, int],
) -> torch.Tensor:
    """读取 ``child_type`` -> ``parent_type`` 层级边。

    开启 hierarchy loss 时未知关系会导致报错，避免把不完整 ontology 静默用于
    正式训练。
    """

    hierarchy_path = Path(path)
    frame = pd.read_csv(
        hierarchy_path,
        sep="\t",
        low_memory=False,
        dtype={"child_type": str, "parent_type": str},
        keep_default_na=False,
    )
    required = {"child_type", "parent_type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{hierarchy_path} 缺少列: {missing}")
    if frame.empty:
        raise ValueError(f"层级文件为空: {hierarchy_path}")

    unknown = sorted(
        (set(frame["child_type"]) | set(frame["parent_type"]))
        - set(relation_to_id)
    )
    if unknown:
        raise ValueError(f"层级文件含训练关系词表之外的类型: {unknown[:20]}")
    children = frame["child_type"].map(relation_to_id).to_numpy(dtype=np.int64)
    parents = frame["parent_type"].map(relation_to_id).to_numpy(dtype=np.int64)
    return torch.from_numpy(np.stack([children, parents], axis=0))


def quantile_bin_labels(values: torch.Tensor, num_bins: int) -> torch.Tensor:
    """用训练集分位点把连续 nuisance 量转换为稳定的类别标签。"""

    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values 必须是非空一维张量")
    if num_bins < 2:
        raise ValueError("num_bins 至少为 2")
    float_values = values.to(torch.float64)
    quantiles = torch.linspace(0, 1, num_bins + 1, dtype=torch.float64)
    boundaries = torch.quantile(float_values.cpu(), quantiles)[1:-1]
    boundaries = torch.unique(boundaries, sorted=True)
    return torch.bucketize(float_values.cpu(), boundaries).to(torch.int64)


def iter_batches(
    size: int,
    batch_size: int,
    order: torch.Tensor | None = None,
) -> Iterable[torch.Tensor]:
    """生成一维 batch 索引；由调用方决定是否打乱。"""

    if size < 0 or batch_size <= 0:
        raise ValueError("size 必须非负且 batch_size 必须为正")
    indices = torch.arange(size, dtype=torch.int64) if order is None else order
    if indices.numel() != size:
        raise ValueError("order 长度与 size 不一致")
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]
