"""SITNE-Walk 的强类型配置定义。

本模块只负责读取、校验配置，不执行任何训练或文件写入。所有默认值都偏向
当前 v1 已冻结且可审计的行为，例如：

* unordered v1 明确把全部关系作为 symmetric，不从记录方向推断生物学方向；
* hierarchy loss 默认关闭；
* 显式请求 CUDA 但 CUDA 不可用时立即报错，而不是静默退回 CPU。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _is_strict_int(value: object) -> bool:
    """排除 bool 和 float；避免配置通过后才在 ``range`` 等位置失败。"""

    return isinstance(value, int) and not isinstance(value, bool)


def _reject_unknown_keys(
    section: str,
    raw: Mapping[str, Any],
    allowed: set[str],
) -> None:
    """拒绝拼写错误或尚未支持的配置项，避免实验悄悄使用错误默认值。"""

    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"配置段 {section!r} 包含未知字段: {unknown}")


@dataclass(frozen=True)
class DataConfig:
    """输入数据及语义元数据配置。"""

    train_path: str
    validation_path: str | None = None
    test_path: str | None = None
    all_known_positive_paths: tuple[str, ...] = ()
    relation_metadata_path: str | None = None
    relation_hierarchy_path: str | None = None
    split_manifest_path: str | None = None
    require_manifest_hashes: bool = False
    weight_column: str | None = None
    duplicate_policy: str = "binary"
    pair_semantics: str = "unordered"
    default_relation_mode: str = "symmetric"
    drop_self_loops_from_walks: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DataConfig":
        allowed = {field_.name for field_ in cls.__dataclass_fields__.values()}
        _reject_unknown_keys("data", raw, allowed)
        values = dict(raw)
        positive_paths = values.get("all_known_positive_paths", ())
        if isinstance(positive_paths, str):
            raise ValueError("data.all_known_positive_paths 必须是路径列表")
        values["all_known_positive_paths"] = tuple(positive_paths)
        return cls(**values)

    def validate(self) -> None:
        if not self.train_path:
            raise ValueError("data.train_path 不能为空")
        if self.require_manifest_hashes and not self.split_manifest_path:
            raise ValueError(
                "data.require_manifest_hashes=true 时必须提供 split_manifest_path"
            )
        if self.relation_hierarchy_path is not None:
            raise ValueError(
                "SITNE-Walk v1 未启用 hierarchy；relation_hierarchy_path 必须为 null"
            )
        if self.duplicate_policy not in {"binary", "sum", "error"}:
            raise ValueError(
                "data.duplicate_policy 必须是 binary、sum 或 error"
            )
        if self.pair_semantics not in {"unordered", "ordered"}:
            raise ValueError("data.pair_semantics 必须是 unordered 或 ordered")
        if self.pair_semantics != "unordered":
            raise ValueError(
                "SITNE-Walk v1 的 decoder 与 filtered query 固定为 unordered；"
                "有方向版本需要独立、经审核的数学合同"
            )
        if self.default_relation_mode != "symmetric":
            raise ValueError(
                "unordered v1 要求 data.default_relation_mode=symmetric"
            )
        if self.default_relation_mode not in {
            "as_recorded",
            "directed",
            "symmetric",
        }:
            raise ValueError(
                "data.default_relation_mode 必须是 as_recorded、directed 或 symmetric"
            )


@dataclass(frozen=True)
class WalkConfig:
    """偏差校正随机游走与 context 采样配置。"""

    alpha: float = 0.5
    beta: float = 0.5
    epsilon: float = 1.0e-8
    correction_mode: str = "rank"
    walk_length: int = 40
    walks_per_protein: int = 10
    walk_batch_size: int = 256
    context_window: int = 5
    negative_samples: int = 5
    negative_distribution: str = "unigram"
    unigram_power: float = 0.75
    max_topology_pairs_per_batch: int = 32_768

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WalkConfig":
        allowed = {field_.name for field_ in cls.__dataclass_fields__.values()}
        _reject_unknown_keys("walk", raw, allowed)
        return cls(**raw)

    def validate(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("walk.alpha 和 walk.beta 必须非负")
        if self.epsilon <= 0:
            raise ValueError("walk.epsilon 必须大于 0")
        if self.correction_mode not in {"log", "rank"}:
            raise ValueError("walk.correction_mode 必须是 log 或 rank")
        for name in (
            "walk_length",
            "walks_per_protein",
            "walk_batch_size",
            "context_window",
            "negative_samples",
            "max_topology_pairs_per_batch",
        ):
            value = getattr(self, name)
            if not _is_strict_int(value) or value <= 0:
                raise ValueError(f"walk.{name} 必须为正整数")
        if self.negative_distribution not in {"uniform", "unigram"}:
            raise ValueError(
                "walk.negative_distribution 必须是 uniform 或 unigram"
            )
        if self.unigram_power <= 0:
            raise ValueError("walk.unigram_power 必须大于 0")


@dataclass(frozen=True)
class ModelConfig:
    """双通道 embedding、decoder 和 nuisance adversary 配置。"""

    embedding_dim: int = 128
    decoder_dropout: float = 0.1
    nuisance_hidden_dim: int = 128
    nuisance_heads: tuple[str, ...] = ("degree",)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ModelConfig":
        allowed = {field_.name for field_ in cls.__dataclass_fields__.values()}
        _reject_unknown_keys("model", raw, allowed)
        values = dict(raw)
        nuisance_heads = values.get("nuisance_heads", ("degree",))
        if isinstance(nuisance_heads, str):
            raise ValueError("model.nuisance_heads 必须是名称列表")
        values["nuisance_heads"] = tuple(nuisance_heads)
        return cls(**values)

    def validate(self) -> None:
        if not _is_strict_int(self.embedding_dim) or self.embedding_dim <= 0:
            raise ValueError("model.embedding_dim 必须为正整数")
        if not 0 <= self.decoder_dropout < 1:
            raise ValueError("model.decoder_dropout 必须位于 [0, 1)")
        if (
            not _is_strict_int(self.nuisance_hidden_dim)
            or self.nuisance_hidden_dim <= 0
        ):
            raise ValueError("model.nuisance_hidden_dim 必须为正整数")
        # 当前清洗数据不含 source 和 detection_method 字段。这里采用 fail-fast，
        # 防止代码用推断标签冒充真实 nuisance metadata。
        unsupported = sorted(set(self.nuisance_heads) - {"degree"})
        if unsupported:
            raise ValueError(
                "当前实现仅能从 train graph 构造 degree nuisance；"
                f"以下 head 缺少经审核的标签适配器: {unsupported}"
            )


@dataclass(frozen=True)
class LossConfig:
    """SITNE-Walk 各损失项的权重。"""

    lambda_typed: float = 1.0
    lambda_decorr: float = 0.01
    lambda_adversary: float = 0.1
    lambda_rank: float = 1.0
    lambda_hierarchy: float = 0.0
    gradient_reversal_coefficient: float = 1.0
    rank_margin: float = 0.0
    hierarchy_margin: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LossConfig":
        allowed = {field_.name for field_ in cls.__dataclass_fields__.values()}
        _reject_unknown_keys("loss", raw, allowed)
        return cls(**raw)

    def validate(self) -> None:
        for name in (
            "lambda_typed",
            "lambda_decorr",
            "lambda_adversary",
            "lambda_rank",
            "lambda_hierarchy",
            "gradient_reversal_coefficient",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"loss.{name} 必须非负")
        if self.hierarchy_margin != 0:
            raise ValueError("SITNE-Walk v1 未启用 hierarchy；hierarchy_margin 必须为 0")


@dataclass(frozen=True)
class TrainingConfig:
    """优化、设备、早停和输出配置。"""

    device: str = "auto"
    seed: int = 42
    epochs: int = 20
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 5.0
    rank_batch_size: int = 1024
    early_stopping_patience: int = 5
    evaluation_batch_size: int = 1024
    use_amp: bool = True
    amp_dtype: str = "float16"
    non_finite_policy: str = "skip_and_log"
    deterministic: bool = True
    allow_tf32: bool = False
    max_steps_per_epoch: int | None = None
    log_interval: int = 20
    output_dir: str = "08_results/sitne_walk"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TrainingConfig":
        allowed = {field_.name for field_ in cls.__dataclass_fields__.values()}
        _reject_unknown_keys("training", raw, allowed)
        return cls(**raw)

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("training.device 必须是 auto、cpu、cuda 或 mps")
        if not _is_strict_int(self.seed) or self.seed < 0:
            raise ValueError("training.seed 必须为非负整数")
        for name in (
            "epochs",
            "rank_batch_size",
            "early_stopping_patience",
            "evaluation_batch_size",
            "log_interval",
        ):
            value = getattr(self, name)
            if not _is_strict_int(value) or value <= 0:
                raise ValueError(f"training.{name} 必须为正整数")
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate 必须大于 0")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay 必须非负")
        if self.gradient_clip_norm <= 0:
            raise ValueError("training.gradient_clip_norm 必须大于 0")
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("training.amp_dtype 必须是 float16 或 bfloat16")
        if self.non_finite_policy not in {"skip_and_log", "fail_fast"}:
            raise ValueError(
                "training.non_finite_policy 必须是 skip_and_log 或 fail_fast"
            )
        if self.max_steps_per_epoch is not None and (
            not _is_strict_int(self.max_steps_per_epoch)
            or self.max_steps_per_epoch <= 0
        ):
            raise ValueError("training.max_steps_per_epoch 必须为正整数或 null")


@dataclass(frozen=True)
class EvaluationConfig:
    """Filtered type-ranking 的确定性规则。"""

    tie_policy: str = "average"
    hits_at: tuple[int, ...] = (1, 3, 10)
    require_full_coverage: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvaluationConfig":
        allowed = {field_.name for field_ in cls.__dataclass_fields__.values()}
        _reject_unknown_keys("evaluation", raw, allowed)
        values = dict(raw)
        hits_at = values.get("hits_at", (1, 3, 10))
        if isinstance(hits_at, (str, bytes)):
            raise ValueError("evaluation.hits_at 必须是正整数列表")
        values["hits_at"] = tuple(hits_at)
        return cls(**values)

    def validate(self) -> None:
        if self.tie_policy not in {"optimistic", "pessimistic", "average"}:
            raise ValueError(
                "evaluation.tie_policy 必须是 optimistic、pessimistic 或 average"
            )
        if not self.hits_at or any(
            not _is_strict_int(k) or k <= 0 for k in self.hits_at
        ):
            raise ValueError("evaluation.hits_at 必须包含正整数")


@dataclass(frozen=True)
class SITNEConfig:
    """完整的 SITNE-Walk 配置。"""

    data: DataConfig
    walk: WalkConfig = field(default_factory=WalkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SITNEConfig":
        _reject_unknown_keys(
            "root",
            raw,
            {"data", "walk", "model", "loss", "training", "evaluation"},
        )
        if "data" not in raw:
            raise ValueError("配置必须包含 data 段")
        config = cls(
            data=DataConfig.from_mapping(raw["data"]),
            walk=WalkConfig.from_mapping(raw.get("walk", {})),
            model=ModelConfig.from_mapping(raw.get("model", {})),
            loss=LossConfig.from_mapping(raw.get("loss", {})),
            training=TrainingConfig.from_mapping(raw.get("training", {})),
            evaluation=EvaluationConfig.from_mapping(raw.get("evaluation", {})),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SITNEConfig":
        """从 YAML 文件读取配置，并检查根节点类型。"""

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError(f"配置文件根节点必须是 mapping: {config_path}")
        return cls.from_mapping(raw)

    def validate(self) -> None:
        self.data.validate()
        self.walk.validate()
        self.model.validate()
        self.loss.validate()
        self.training.validate()
        self.evaluation.validate()
        if self.loss.lambda_hierarchy > 0:
            raise ValueError(
                "SITNE-Walk v1 未冻结 hierarchy 数学与可靠 ontology；"
                "loss.lambda_hierarchy 必须为 0"
            )

    def to_dict(self) -> dict[str, Any]:
        """返回适合写入 provenance JSON 的纯 Python 字典。"""

        return asdict(self)
