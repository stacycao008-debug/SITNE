"""SITNE-Walk-BX 的严格配置定义。

配置解析拒绝未知字段。正式配置必须显式使用 CUDA；CPU 只可通过
``allow_cpu_for_testing`` 在合成测试中启用，不能被误认为服务器验收结果。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _reject_unknown(section: str, raw: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"配置段 {section!r} 包含未知字段: {unknown}")


def _positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须为正整数")


@dataclass(frozen=True)
class DataConfig:
    dataset_b_root: str = "02_data_canonical/dataset_B/v1"
    dataset_bstar_root: str | None = "02_data_canonical/dataset_Bstar/v2"
    dataset_uid: str = "BERNETT_V3_FULL_PAPER_SPLIT_V1"
    xstar_uid: str = "BERNETT_V3_FULL_PAPER_SPLIT_XSTAR_V2"
    typed_level: str = "fine"
    verify_hashes: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DataConfig":
        _reject_unknown("data", raw, set(cls.__dataclass_fields__))
        return cls(**raw)

    def validate(self) -> None:
        if not self.dataset_b_root or not self.dataset_uid:
            raise ValueError("data.dataset_b_root 和 dataset_uid 不能为空")
        if self.typed_level not in {"none", "fine", "coarse"}:
            raise ValueError("data.typed_level 必须是 none、fine 或 coarse")
        if self.verify_hashes is not True:
            raise ValueError("data.verify_hashes 是冻结安全项，必须为 true")
        if not self.dataset_bstar_root:
            raise ValueError(
                "所有模式都必须配置 dataset_bstar_root：Train pair/sequence 固定从 "
                "B* 七项 allow-list 的 X/train 与 nodes/train_protein 读取，禁止训练期"
                "打开含 Validation/Test 蛋白的 B 全量序列表"
            )


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "chunked_sequence"
    embedding_dim: int = 128
    token_embedding_dim: int = 64
    pair_hidden_dim: int = 256
    chunk_size: int = 1024
    chunk_stride: int = 768
    chunk_batch_size: int = 16
    convolution_kernel_size: int = 5
    dropout: float = 0.1
    precomputed_dimension: int | None = None
    precomputed_train_path: str | None = None
    precomputed_validation_path: str | None = None
    precomputed_test_path: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ModelConfig":
        _reject_unknown("model", raw, set(cls.__dataclass_fields__))
        return cls(**raw)

    def validate(self) -> None:
        if self.backend not in {"chunked_sequence", "precomputed_embedding"}:
            raise ValueError(
                "model.backend 必须是 chunked_sequence 或 precomputed_embedding"
            )
        for name in (
            "embedding_dim",
            "token_embedding_dim",
            "pair_hidden_dim",
            "chunk_size",
            "chunk_stride",
            "chunk_batch_size",
            "convolution_kernel_size",
        ):
            _positive_int(f"model.{name}", getattr(self, name))
        if self.chunk_stride > self.chunk_size:
            raise ValueError("model.chunk_stride 不得大于 chunk_size，否则会遗漏残基")
        if self.convolution_kernel_size % 2 == 0:
            raise ValueError("model.convolution_kernel_size 必须是奇数")
        if not 0 <= self.dropout < 1:
            raise ValueError("model.dropout 必须位于 [0, 1)")
        if self.backend == "precomputed_embedding":
            _positive_int("model.precomputed_dimension", self.precomputed_dimension)
            for phase in ("train", "validation", "test"):
                if not getattr(self, f"precomputed_{phase}_path"):
                    raise ValueError(
                        "precomputed_embedding 后端必须提供 train/validation/test 三个"
                        "独立 NPZ，避免跨阶段打开 held-out 表示"
                    )
            paths = {
                self.precomputed_train_path,
                self.precomputed_validation_path,
                self.precomputed_test_path,
            }
            if len(paths) != 3:
                raise ValueError(
                    "precomputed train/validation/test 路径必须互不相同"
                )


@dataclass(frozen=True)
class LossConfig:
    binary_weight: float = 1.0
    topology_weight: float = 0.1
    typed_weight: float = 0.2

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LossConfig":
        _reject_unknown("loss", raw, set(cls.__dataclass_fields__))
        return cls(**raw)

    def validate(self) -> None:
        for name in ("binary_weight", "topology_weight", "typed_weight"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"loss.{name} 必须非负")
        if self.binary_weight <= 0:
            raise ValueError("loss.binary_weight 必须大于 0")


@dataclass(frozen=True)
class TopologyConfig:
    """仅基于 Dataset B Train-positive 图的多步游走/context 参数。"""

    walk_length: int = 8
    walks_per_node: int = 1
    context_window: int = 2
    context_batch_size: int = 32

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TopologyConfig":
        _reject_unknown("topology", raw, set(cls.__dataclass_fields__))
        return cls(**raw)

    def validate(self) -> None:
        for name in (
            "walk_length",
            "walks_per_node",
            "context_window",
            "context_batch_size",
        ):
            _positive_int(f"topology.{name}", getattr(self, name))
        if self.walk_length < 2:
            raise ValueError("topology.walk_length 至少为 2")


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda"
    allow_cpu_for_testing: bool = False
    seed: int = 42
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 5.0
    use_amp: bool = True
    amp_dtype: str = "float16"
    deterministic: bool = True
    allow_tf32: bool = False
    max_batches_per_epoch: int | None = None
    output_dir: str = "08_results/sitne_bx"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TrainingConfig":
        _reject_unknown("training", raw, set(cls.__dataclass_fields__))
        return cls(**raw)

    def validate(self) -> None:
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("training.device 只能是 cuda 或 cpu；不支持自动回退/MPS")
        if self.device == "cpu" and not self.allow_cpu_for_testing:
            raise ValueError(
                "正式配置必须为 CUDA；CPU 仅可设置 allow_cpu_for_testing=true 用于合成测试"
            )
        if self.device == "cpu" and self.use_amp:
            raise ValueError("CPU 合成测试必须关闭 AMP")
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("training.amp_dtype 必须是 float16 或 bfloat16")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("training.seed 必须为非负整数")
        for name in ("epochs", "batch_size"):
            _positive_int(f"training.{name}", getattr(self, name))
        if self.max_batches_per_epoch is not None:
            _positive_int(
                "training.max_batches_per_epoch", self.max_batches_per_epoch
            )
        for name in ("learning_rate", "gradient_clip_norm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"training.{name} 必须大于 0")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay 必须非负")


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 128
    selection_metric: str = "auprc"
    threshold_metric: str = "mcc"
    expected_test_rows: int = 52048

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvaluationConfig":
        _reject_unknown("evaluation", raw, set(cls.__dataclass_fields__))
        return cls(**raw)

    def validate(self) -> None:
        _positive_int("evaluation.batch_size", self.batch_size)
        _positive_int("evaluation.expected_test_rows", self.expected_test_rows)
        if self.selection_metric != "auprc":
            raise ValueError("当前冻结协议要求 selection_metric=auprc")
        if self.threshold_metric != "mcc":
            raise ValueError("当前冻结协议要求 threshold_metric=mcc")


@dataclass(frozen=True)
class BXConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BXConfig":
        _reject_unknown(
            "root", raw, {"data", "model", "loss", "topology", "training", "evaluation"}
        )
        config = cls(
            data=DataConfig.from_mapping(raw.get("data", {})),
            model=ModelConfig.from_mapping(raw.get("model", {})),
            loss=LossConfig.from_mapping(raw.get("loss", {})),
            topology=TopologyConfig.from_mapping(raw.get("topology", {})),
            training=TrainingConfig.from_mapping(raw.get("training", {})),
            evaluation=EvaluationConfig.from_mapping(raw.get("evaluation", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.loss.validate()
        self.topology.validate()
        self.training.validate()
        self.evaluation.validate()
        if self.data.typed_level == "none" and self.loss.typed_weight != 0:
            raise ValueError("typed_level=none 时 loss.typed_weight 必须为 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> BXConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("配置文件根节点必须是 mapping")
    return BXConfig.from_mapping(raw)
