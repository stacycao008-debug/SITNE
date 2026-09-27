"""SITNE-Walk 的流式 CPU/CUDA/MPS 训练器。

设计重点：

* 全量 walk corpus 不落盘、不一次性驻留 GPU；
* start scheduling、walk 和 SGNS noise 使用独立 RNG 流；
* validation 只用于早停，test 不进入本模块的模型选择；
* checkpoint 保存模型、优化器、AMP scaler 与 RNG 状态。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from .config import SITNEConfig
from .contexts import build_topology_contexts, build_typed_relation_contexts
from .data import IndexedTriples, PairRelationIndex, TrainingTriples, iter_batches
from .evaluator import RankingResult, evaluate_filtered_type_ranking
from .graph import PackedCSRGraph
from .losses import (
    decorrelation_loss,
    filtered_pairwise_ranking_loss,
    hierarchy_loss_disabled,
    nuisance_cross_entropy,
    topology_sgns_loss,
    typed_relation_cross_entropy,
)
from .model import SITNEWalkModel
from .walks import AliasTypedWalker, _rand, make_generator

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 1


def resolve_device(preferred: str) -> torch.device:
    """解析设备；显式请求不可用设备时绝不静默 fallback。"""

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "显式请求 CUDA，但当前 PyTorch/CUDA 不可用；"
                "请安装 CUDA 版 PyTorch 并在 NVIDIA 主机运行"
            )
        return torch.device("cuda")
    if preferred == "mps":
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not available:
            raise RuntimeError("显式请求 MPS，但当前环境不可用")
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred != "auto":
        raise ValueError(f"未知 device: {preferred}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_reproducibility(config: SITNEConfig, device: torch.device) -> None:
    """设置种子和确定性开关。"""

    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = config.training.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.training.allow_tf32
        torch.backends.cudnn.benchmark = not config.training.deterministic
        if config.training.deterministic:
            # cuBLAS 要求该变量在 CUDA context 初始化前设置；CLI 会在构造模型前
            # 调用本函数。若外部程序已初始化 CUDA，provenance 应记录该事实。
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(config.training.deterministic)


def degree_log2_labels(degrees: torch.Tensor) -> torch.Tensor:
    """把 train-only degree 转为 ``0`` 或 ``1+floor(log2 d)``。"""

    if degrees.ndim != 1 or bool((degrees < 0).any()):
        raise ValueError("degrees 必须是非负一维张量")
    labels = torch.zeros_like(degrees, dtype=torch.int64)
    positive = degrees > 0
    labels[positive] = 1 + torch.floor(torch.log2(degrees[positive].float())).to(
        torch.int64
    )
    return labels


def inverse_frequency_class_weights(
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """仅从 train 标签计算归一化 inverse-frequency class weights。"""

    counts = torch.bincount(labels.cpu(), minlength=num_classes).to(torch.float64)
    weights = torch.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / counts[nonzero]
    if bool(nonzero.any()):
        weights[nonzero] *= nonzero.sum() / weights[nonzero].sum()
    return weights.to(torch.float32)


class ProteinNoiseSampler:
    """SGNS protein noise-token sampler。"""

    def __init__(
        self,
        degrees: torch.Tensor,
        device: torch.device,
        distribution: str,
        unigram_power: float,
    ):
        if degrees.ndim != 1 or degrees.numel() < 2:
            raise ValueError("noise sampler 至少需要两个蛋白质")
        if distribution == "uniform":
            probabilities = torch.ones_like(degrees, dtype=torch.float64)
        elif distribution == "unigram":
            probabilities = degrees.to(torch.float64).clamp_min(0).pow(unigram_power)
            # 只出现自环但未进入 walk 的节点可能为 0；保留零概率是预期行为。
            if float(probabilities.sum().item()) <= 0:
                raise ValueError("degree-based noise distribution 没有正概率节点")
        else:
            raise ValueError(f"未知 noise distribution: {distribution}")
        self.probabilities_cpu = probabilities / probabilities.sum()
        self.num_proteins = int(degrees.numel())
        self.device = device
        self.distribution = distribution

    def _draw(
        self,
        shape: tuple[int, ...],
        generator: torch.Generator,
    ) -> torch.Tensor:
        count = math.prod(shape)
        generator_device = torch.device(getattr(generator, "device", "cpu"))
        if self.distribution == "uniform":
            uniform = _rand((count,), self.device, generator)
            return torch.floor(uniform * self.num_proteins).to(torch.int64).reshape(shape)
        # MPS 不支持 float64；CUDA 侧 multinomial 也无需双精度。CPU 中保留
        # float64 统计，迁移到加速器时显式转为 float32。
        target_dtype = (
            torch.float64 if generator_device.type == "cpu" else torch.float32
        )
        probabilities = self.probabilities_cpu.to(
            device=generator_device,
            dtype=target_dtype,
        )
        samples = torch.multinomial(
            probabilities,
            num_samples=count,
            replacement=True,
            generator=generator,
        )
        return samples.to(self.device).reshape(shape)

    def sample(
        self,
        positive_contexts: torch.Tensor,
        num_negatives: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """采样 noise tokens，并拒绝与当前正 context 完全相同的 ID。"""

        if positive_contexts.ndim != 1 or num_negatives <= 0:
            raise ValueError("positive_contexts 必须一维且 num_negatives 必须为正")
        shape = (positive_contexts.numel(), num_negatives)
        samples = self._draw(shape, generator)
        forbidden = positive_contexts[:, None]
        # 固定迭代次数使 RNG 消耗不依赖命中数量。
        for _ in range(4):
            replacement = self._draw(shape, generator)
            samples = torch.where(samples == forbidden, replacement, samples)
        remaining = samples == forbidden
        fallback = (forbidden + 1) % self.num_proteins
        return torch.where(remaining, fallback, samples)


def build_model(
    config: SITNEConfig,
    triples: TrainingTriples,
    graph: PackedCSRGraph,
) -> tuple[SITNEWalkModel, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """根据 train-only nuisance 标签建立可复现模型及 class weights。

    模型在 ``SITNETrainer`` 构造前建立，因此不能依赖 trainer 稍后设置全局种子。
    这里用独立 CPU RNG 上下文按配置种子初始化参数，同时不污染调用方 RNG。
    """

    nuisance_labels: dict[str, torch.Tensor] = {}
    nuisance_weights: dict[str, torch.Tensor] = {}
    cardinalities: dict[str, int] = {}
    if "degree" in config.model.nuisance_heads:
        labels = degree_log2_labels(graph.protein_degree)
        num_classes = int(labels.max().item()) + 1
        nuisance_labels["degree"] = labels
        nuisance_weights["degree"] = inverse_frequency_class_weights(
            labels,
            num_classes,
        )
        cardinalities["degree"] = num_classes

    with torch.random.fork_rng(devices=[]):
        # 只设置 CPU default generator；``torch.manual_seed`` 还会改动 CUDA/MPS
        # 全局 RNG，与本函数“不污染调用方 RNG”的合同冲突。
        torch.random.default_generator.manual_seed(config.training.seed)
        model = SITNEWalkModel(
            num_proteins=triples.num_proteins,
            num_relations=triples.num_relations,
            embedding_dim=config.model.embedding_dim,
            nuisance_cardinalities=cardinalities,
            nuisance_hidden_dim=config.model.nuisance_hidden_dim,
            decoder_dropout=config.model.decoder_dropout,
        )
    return model, nuisance_labels, nuisance_weights


@dataclass(frozen=True)
class FitResult:
    """训练历史与最佳 checkpoint。"""

    history: tuple[Mapping[str, float], ...]
    best_epoch: int
    best_selection_metric: float
    best_checkpoint: str | None


class SITNETrainer:
    """流式训练总控。"""

    def __init__(
        self,
        config: SITNEConfig,
        triples: TrainingTriples,
        graph: PackedCSRGraph,
        model: SITNEWalkModel,
        nuisance_labels: Mapping[str, torch.Tensor],
        nuisance_class_weights: Mapping[str, torch.Tensor],
    ):
        self.config = config
        self.triples = triples
        self.device = resolve_device(config.training.device)
        configure_reproducibility(config, self.device)
        if (
            config.training.use_amp
            and config.training.amp_dtype == "bfloat16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "当前 CUDA GPU 不支持 bfloat16 AMP；请改用 float16 或关闭 AMP"
            )
        self.graph = graph.to(self.device)
        self.walker = AliasTypedWalker(self.graph)
        self.model = model.to(self.device)
        self.nuisance_labels = {
            name: values.to(self.device) for name, values in nuisance_labels.items()
        }
        self.nuisance_class_weights = {
            name: values.to(self.device)
            for name, values in nuisance_class_weights.items()
        }
        self.noise_sampler = ProteinNoiseSampler(
            graph.protein_degree,
            device=self.device,
            distribution=config.walk.negative_distribution,
            unigram_power=config.walk.unigram_power,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        amp_enabled = config.training.use_amp and self.device.type == "cuda"
        self.amp_enabled = amp_enabled
        self.amp_dtype = (
            torch.float16 if config.training.amp_dtype == "float16" else torch.bfloat16
        )
        scaler_enabled = amp_enabled and self.amp_dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):  # PyTorch 2.0 compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        base_seed = config.training.seed
        self.schedule_generator = torch.Generator(device="cpu").manual_seed(base_seed + 11)
        self.walk_generator = make_generator(self.device, base_seed + 23)
        self.negative_generator = make_generator(self.device, base_seed + 37)
        self.rank_generator = torch.Generator(device="cpu").manual_seed(base_seed + 53)
        self._resume_epoch: int | None = None
        self._resume_history: list[Mapping[str, float]] = []
        self._resume_best_metric = -math.inf
        self._resume_checkpoint: str | None = None

        # 数值可靠性诊断状态（不进入 checkpoint 的模型/RNG 状态）。
        self.non_finite_policy = config.training.non_finite_policy
        self.non_finite_loss_steps = 0
        self.non_finite_gradient_steps = 0
        self.skipped_updates = 0
        self.nonfinite_events: list[dict[str, Any]] = []
        # step_log 仅在 debug/smoke（skip_and_log）模式下累积，避免正式训练 I/O 与内存膨胀。
        self.step_log: list[dict[str, Any]] = []

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
            enabled=True,
        )

    def _rank_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """均匀采样 train canonical pairs，而不是按标签行采样。"""

        pair_count = int(self.triples.pair_index.pair_keys.numel())
        batch_size = min(self.config.training.rank_batch_size, pair_count)
        pair_ids = torch.randint(
            0,
            pair_count,
            (batch_size,),
            generator=self.rank_generator,
            dtype=torch.int64,
        )
        keys = self.triples.pair_index.pair_keys[pair_ids]
        heads = torch.div(keys, self.triples.num_proteins, rounding_mode="floor")
        tails = keys % self.triples.num_proteins
        masks = self.triples.pair_index.relation_mask[pair_ids]
        return heads.to(self.device), tails.to(self.device), masks.to(self.device)

    def _record_nonfinite(
        self,
        kind: str,
        epoch: int,
        step: int,
        detail: Mapping[str, Any],
    ) -> None:
        """统一记录 non-finite 事件，并按 policy 决定跳过或 fail-fast。

        ``kind`` 为 ``"loss"`` 或 ``"gradient"``。``fail_fast``（formal）模式下
        立即抛 ``RuntimeError`` 终止 run；``skip_and_log``（debug/smoke）模式下
        仅计数并记录事件，随后由调用方跳过本次 update。
        """

        if kind == "loss":
            self.non_finite_loss_steps += 1
        elif kind == "gradient":
            self.non_finite_gradient_steps += 1
        else:
            raise ValueError(f"未知 non-finite kind: {kind}")
        self.skipped_updates += 1
        event: dict[str, Any] = {"epoch": epoch, "step": step, "kind": kind, **detail}
        self.nonfinite_events.append(event)
        if self.non_finite_policy == "fail_fast":
            raise RuntimeError(
                f"non-finite {kind} at epoch={epoch} step={step}: "
                f"non_finite_loss_steps={self.non_finite_loss_steps}, "
                f"non_finite_gradient_steps={self.non_finite_gradient_steps}, "
                f"skipped_updates={self.skipped_updates}; "
                f"policy=fail_fast 立即终止 run（detail={detail}）"
            )
        logger.warning(
            "non-finite %s at epoch=%d step=%d — policy=skip_and_log 跳过本次 update",
            kind,
            epoch,
            step,
        )

    def _single_step(
        self,
        start_nodes: torch.Tensor,
        epoch: int = 0,
        step: int = 0,
    ) -> dict[str, float]:
        walks = self.walker.generate(
            start_nodes=start_nodes,
            walk_length=self.config.walk.walk_length,
            generator=self.walk_generator,
        )
        topology_contexts = build_topology_contexts(
            walks,
            window_size=self.config.walk.context_window,
            max_pairs=self.config.walk.max_topology_pairs_per_batch,
            generator=self.negative_generator,
        )
        typed_contexts = build_typed_relation_contexts(
            walks,
            window_size=self.config.walk.context_window,
        )
        if topology_contexts.size == 0 or typed_contexts.size == 0:
            raise RuntimeError("walk batch 没有产生可训练 context")
        negative_contexts = self.noise_sampler.sample(
            topology_contexts.contexts,
            num_negatives=self.config.walk.negative_samples,
            generator=self.negative_generator,
        )
        rank_heads, rank_tails, rank_positive_mask = self._rank_batch()

        unique_proteins = torch.unique(
            torch.cat(
                [
                    topology_contexts.centers,
                    topology_contexts.contexts,
                    typed_contexts.proteins,
                    rank_heads,
                    rank_tails,
                ]
            )
        )
        if unique_proteins.numel() < 2:
            raise RuntimeError("decorrelation batch 少于两个唯一蛋白质")

        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            positive_logits, negative_logits = self.model.topology_sgns_logits(
                topology_contexts.centers,
                topology_contexts.contexts,
                negative_contexts,
            )
            loss_topology = topology_sgns_loss(positive_logits, negative_logits)
            typed_logits = self.model.typed_relation_logits(typed_contexts.proteins)
            loss_typed = typed_relation_cross_entropy(
                typed_logits,
                typed_contexts.relations,
            )
            topology_embeddings = self.model.topology_embedding(unique_proteins)
            semantic_embeddings = self.model.semantic_embedding(unique_proteins)
            loss_decorr = decorrelation_loss(
                topology_embeddings,
                semantic_embeddings,
            )
            rank_scores = self.model.score_all_relations(rank_heads, rank_tails)
            loss_rank = filtered_pairwise_ranking_loss(
                rank_scores,
                rank_positive_mask,
                margin=self.config.loss.rank_margin,
            )

            adversary_terms: list[torch.Tensor] = []
            for head_name in self.model.nuisance_heads:
                nuisance_logits = self.model.nuisance_logits(
                    unique_proteins,
                    head_name=head_name,
                    gradient_reversal_coefficient=(
                        self.config.loss.gradient_reversal_coefficient
                    ),
                )
                adversary_terms.append(
                    nuisance_cross_entropy(
                        nuisance_logits,
                        self.nuisance_labels[head_name][unique_proteins],
                        self.nuisance_class_weights[head_name],
                    )
                )
            loss_adversary = (
                torch.stack(adversary_terms).mean()
                if adversary_terms
                else loss_topology.new_zeros(())
            )
            loss_hierarchy = hierarchy_loss_disabled(
                self.model.semantic_relation.weight,
                coefficient=self.config.loss.lambda_hierarchy,
            )
            total_loss = (
                loss_topology
                + self.config.loss.lambda_typed * loss_typed
                + self.config.loss.lambda_decorr * loss_decorr
                + self.config.loss.lambda_adversary * loss_adversary
                + self.config.loss.lambda_rank * loss_rank
                + loss_hierarchy
            )

        _nan_skip = {
            "loss": float("nan"),
            "loss_topology": float("nan"),
            "loss_typed": float("nan"),
            "loss_decorr": float("nan"),
            "loss_adversary": float("nan"),
            "loss_rank": float("nan"),
            "gradient_norm": float("nan"),
            "topology_pairs": float("nan"),
            "typed_contexts": float("nan"),
            "valid_walk_steps": float("nan"),
        }
        if not bool(torch.isfinite(total_loss)):
            self.optimizer.zero_grad(set_to_none=True)
            self._record_nonfinite(
                "loss",
                epoch,
                step,
                {"total_loss": float("nan")},
            )
            return _nan_skip
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        try:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.gradient_clip_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.step(self.optimizer)  # dummy step to reset scaler state
            self.scaler.update()
            self._record_nonfinite(
                "gradient",
                epoch,
                step,
                {"gradient_norm": float("nan")},
            )
            return _nan_skip
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": float(total_loss.detach().item()),
            "loss_topology": float(loss_topology.detach().item()),
            "loss_typed": float(loss_typed.detach().item()),
            "loss_decorr": float(loss_decorr.detach().item()),
            "loss_adversary": float(loss_adversary.detach().item()),
            "loss_rank": float(loss_rank.detach().item()),
            "gradient_norm": float(gradient_norm.detach().item()),
            "topology_pairs": float(topology_contexts.size),
            "typed_contexts": float(typed_contexts.size),
            "valid_walk_steps": float(walks.valid_steps.sum().item()),
        }

    def _checkpoint_payload(
        self,
        epoch: int,
        best_metric: float,
        history: list[Mapping[str, float]],
    ) -> dict[str, Any]:
        return {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": epoch,
            "best_selection_metric": best_metric,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "schedule_rng_state": self.schedule_generator.get_state(),
            "walk_rng_state": self.walk_generator.get_state(),
            "negative_rng_state": self.negative_generator.get_state(),
            "rank_rng_state": self.rank_generator.get_state(),
            # Dropout 等算子使用全局 device RNG，不能只保存上面的自建 Generator。
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
            "torch_mps_rng_state": (
                torch.mps.get_rng_state()
                if self.device.type == "mps"
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "get_rng_state")
                else None
            ),
            "checkpoint_device_type": self.device.type,
            "config": self.config.to_dict(),
            "protein_vocab": self.triples.id_to_protein,
            "relation_vocab": self.triples.id_to_relation,
            "train_sha256": self.triples.source_sha256,
            "history": history,
        }

    def load_checkpoint(self, path: str | Path) -> int:
        """恢复模型、优化器、scaler 和全部 RNG；返回下一 epoch。"""

        # 先全部加载到 CPU：CPU Generator.set_state 要求 CPU ByteTensor。随后
        # load_state_dict 会把模型参数复制到现有目标设备，优化器 state 则显式迁移。
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch 2.0 尚无 weights_only 参数
            checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "checkpoint 格式版本不兼容；请使用当前代码重新训练或显式迁移"
            )
        if tuple(checkpoint["protein_vocab"]) != self.triples.id_to_protein:
            raise ValueError("checkpoint protein vocab 与当前训练集不一致")
        if tuple(checkpoint["relation_vocab"]) != self.triples.id_to_relation:
            raise ValueError("checkpoint relation vocab 与当前训练集不一致")
        if checkpoint["train_sha256"] != self.triples.source_sha256:
            raise ValueError("checkpoint train SHA-256 与当前输入不一致")
        if checkpoint.get("config") != self.config.to_dict():
            raise ValueError(
                "checkpoint config 与当前配置不一致；拒绝混用模型、优化器和图参数"
            )
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.schedule_generator.set_state(checkpoint["schedule_rng_state"])
        self.walk_generator.set_state(checkpoint["walk_rng_state"])
        self.negative_generator.set_state(checkpoint["negative_rng_state"])
        self.rank_generator.set_state(checkpoint["rank_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"])

        checkpoint_device = checkpoint.get("checkpoint_device_type")
        if self.device.type == "cuda" and checkpoint["torch_cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state_all"])
        elif (
            self.device.type == "mps"
            and checkpoint["torch_mps_rng_state"] is not None
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "set_rng_state")
        ):
            torch.mps.set_rng_state(checkpoint["torch_mps_rng_state"])
        if checkpoint_device != self.device.type:
            logger.warning(
                "checkpoint 由 %s 设备创建，当前为 %s；模型可评测，但不承诺跨后端续训逐位复现",
                checkpoint_device,
                self.device.type,
            )

        next_epoch = int(checkpoint["epoch"]) + 1
        self._resume_epoch = next_epoch
        self._resume_history = list(checkpoint.get("history", []))
        self._resume_best_metric = float(checkpoint["best_selection_metric"])
        self._resume_checkpoint = str(Path(path).resolve())
        return next_epoch

    def fit(
        self,
        validation: IndexedTriples | None = None,
        validation_filter: PairRelationIndex | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> FitResult:
        """执行训练；如有 validation，则以 filtered MRR 早停。"""

        if (validation is None) != (validation_filter is None):
            raise ValueError("validation 与 validation_filter 必须同时提供或同时为空")
        output_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        eligible = self.graph.nodes_with_outgoing_edges.cpu()
        if eligible.numel() == 0:
            raise ValueError("训练图没有可作为 walk 起点的节点")
        base_schedule = eligible.repeat_interleave(self.config.walk.walks_per_protein)
        start_epoch = self._resume_epoch or 0
        history: list[Mapping[str, float]] = list(self._resume_history)
        best_metric = self._resume_best_metric
        best_epoch = start_epoch - 1 if self._resume_epoch is not None else -1
        best_checkpoint: str | None = self._resume_checkpoint
        epochs_without_improvement = 0

        if start_epoch >= self.config.training.epochs:
            return FitResult(
                history=tuple(history),
                best_epoch=best_epoch,
                best_selection_metric=best_metric,
                best_checkpoint=best_checkpoint,
            )

        for epoch in range(start_epoch, self.config.training.epochs):
            self.model.train()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            permutation = torch.randperm(
                base_schedule.numel(),
                generator=self.schedule_generator,
            )
            schedule = base_schedule[permutation]
            step_metrics: list[Mapping[str, float]] = []
            for step, batch_indices in enumerate(
                iter_batches(
                    schedule.numel(),
                    self.config.walk.walk_batch_size,
                    order=torch.arange(schedule.numel()),
                )
            ):
                if (
                    self.config.training.max_steps_per_epoch is not None
                    and step >= self.config.training.max_steps_per_epoch
                ):
                    break
                starts = schedule[batch_indices].to(self.device)
                metrics = self._single_step(starts, epoch=epoch, step=step)
                step_metrics.append(metrics)
                if self.non_finite_policy == "skip_and_log":
                    self.step_log.append(
                        {"epoch": epoch, "step": step, **metrics}
                    )
                if step % self.config.training.log_interval == 0:
                    logger.info(
                        "epoch=%d step=%d loss=%.6f topo=%.6f typed=%.6f rank=%.6f",
                        epoch,
                        step,
                        metrics["loss"],
                        metrics["loss_topology"],
                        metrics["loss_typed"],
                        metrics["loss_rank"],
                    )
            if not step_metrics:
                raise RuntimeError("本 epoch 没有执行任何训练 step")

            epoch_metrics: dict[str, float] = {
                key: float(np.mean([item[key] for item in step_metrics]))
                for key in step_metrics[0]
            }
            epoch_metrics["epoch"] = float(epoch)
            if self.device.type == "cuda":
                epoch_metrics["cuda_peak_memory_bytes"] = float(
                    torch.cuda.max_memory_allocated(self.device)
                )

            if validation is not None and validation_filter is not None:
                ranking: RankingResult = evaluate_filtered_type_ranking(
                    self.model,
                    validation,
                    validation_filter,
                    device=self.device,
                    batch_size=self.config.training.evaluation_batch_size,
                    hits_at=self.config.evaluation.hits_at,
                    tie_policy=self.config.evaluation.tie_policy,
                    require_full_coverage=self.config.evaluation.require_full_coverage,
                )
                if ranking.status == "unsupported" or "mrr" not in ranking.metrics:
                    raise RuntimeError("validation 对 transductive 模型完全 unsupported")
                selection_metric = float(ranking.metrics["mrr"])
                epoch_metrics["validation_mrr"] = selection_metric
                epoch_metrics["validation_coverage"] = ranking.coverage
            else:
                # 无验证集仅适用于单元/SMOKE；用负训练 loss 作为技术性早停量。
                selection_metric = -epoch_metrics["loss"]

            history.append(epoch_metrics)
            improved = selection_metric > best_metric
            if improved:
                best_metric = selection_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                if output_dir is not None:
                    checkpoint_path = output_dir / f"checkpoint_epoch{epoch:04d}.pt"
                    if checkpoint_path.exists():
                        raise FileExistsError(f"拒绝覆盖 checkpoint: {checkpoint_path}")
                    torch.save(
                        self._checkpoint_payload(epoch, best_metric, history),
                        checkpoint_path,
                    )
                    best_checkpoint = str(checkpoint_path.resolve())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.training.early_stopping_patience:
                    logger.info("early stopping at epoch=%d", epoch)
                    break

        return FitResult(
            history=tuple(history),
            best_epoch=best_epoch,
            best_selection_metric=best_metric,
            best_checkpoint=best_checkpoint,
        )
