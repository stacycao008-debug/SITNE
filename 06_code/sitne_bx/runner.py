"""CUDA/no-fallback 训练器和 inspect/train/select/test/verify-run 状态机。"""

from __future__ import annotations

from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .artifacts import (
    create_run_directory,
    make_run_id,
    object_sha256,
    read_json,
    require_file_hash,
    seal_payload,
    torch_save_new,
    verify_sealed_payload,
    write_json_new,
    write_tsv_new,
)
from .config import BXConfig
from .data import (
    PairRecord,
    PairTable,
    TypedAnnotations,
    iter_batches,
    load_evaluation_data,
    load_pair_table,
    load_train_data,
    resolve_feature_provider,
    typed_targets_for_batch,
)
from .guarded_io import (
    BSTAR_TRAIN_ALLOWLIST,
    GuardedDatasetIO,
    Phase,
    _issue_post_selection_capability,
    sha256_file,
)
from .inspection import CANONICAL_COUNTS
from .losses import composite_loss
from .metrics import binary_metrics, select_mcc_threshold
from .model import SITNEBXModel
from .provenance import collect_implementation_hashes, collect_provenance
from .topology import (
    WalkContext,
    build_train_positive_graph,
    sample_walk_contexts,
)
from .losses import train_positive_topology_loss


CHECKPOINT_FORMAT_VERSION = 1


def resolve_roots(config: BXConfig, project_root: str | Path) -> tuple[Path, Path | None]:
    root = Path(project_root).resolve()
    b_root = Path(config.data.dataset_b_root)
    if not b_root.is_absolute():
        b_root = root / b_root
    x_root: Path | None = None
    if config.data.dataset_bstar_root:
        x_root = Path(config.data.dataset_bstar_root)
        if not x_root.is_absolute():
            x_root = root / x_root
    return b_root.resolve(), x_root.resolve() if x_root else None


def resolve_device(config: BXConfig) -> torch.device:
    requested = config.training.device
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "正式配置显式请求 CUDA，但当前 PyTorch/CUDA 不可用；"
                "SITNE-Walk-BX 禁止静默回退 CPU/MPS"
            )
        device = torch.device("cuda")
        if config.training.use_amp and config.training.amp_dtype == "bfloat16":
            supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            if not supported:
                raise RuntimeError("当前 NVIDIA GPU/PyTorch 不支持 BF16 AMP")
        return device
    if requested == "cpu" and config.training.allow_cpu_for_testing:
        return torch.device("cpu")
    raise RuntimeError("非 CUDA 运行只允许显式的 CPU 合成测试配置")


def configure_reproducibility(config: BXConfig, device: torch.device) -> None:
    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = config.training.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.training.allow_tf32
        torch.backends.cudnn.benchmark = not config.training.deterministic
    torch.use_deterministic_algorithms(config.training.deterministic)


def _autocast(config: BXConfig, device: torch.device):
    if device.type != "cuda" or not config.training.use_amp:
        return nullcontext()
    dtype = (
        torch.float16
        if config.training.amp_dtype == "float16"
        else torch.bfloat16
    )
    return torch.autocast(device_type="cuda", dtype=dtype)


def _json_from_guard(guard: GuardedDatasetIO, dataset: str, relative: str) -> dict[str, Any]:
    value = json.loads(guard.read_text(dataset, relative))
    if not isinstance(value, dict):
        raise ValueError(f"{dataset}:{relative} JSON 根节点不是 object")
    return value


def _validate_train_contract(config: BXConfig, guard: GuardedDatasetIO) -> dict[str, Any]:
    b_manifest = _json_from_guard(guard, "b", "manifest/dataset.json")
    validation = _json_from_guard(guard, "b", "reports/validation.json")
    if b_manifest.get("dataset_uid") != config.data.dataset_uid:
        raise ValueError("Dataset B UID 不匹配")
    if validation.get("status") != "PASS":
        raise ValueError("Dataset B 官方 validation status 不是 PASS")
    result: dict[str, Any] = {
        "dataset_uid": config.data.dataset_uid,
        "train_split_sha256": b_manifest.get("stats", {}).get("train", {}).get("output_sha256"),
        "train_split_bytes": b_manifest.get("stats", {}).get("train", {}).get("output_bytes"),
    }
    # Dataset B 是 binary label 的最终来源。实际打开并 hash 冻结 Train split；
    # B* X/train 随后独立按 train_input_manifest 校验。
    with guard.open_binary("b", "splits/train.tsv"):
        pass
    if config.data.dataset_bstar_root:
        contract = _json_from_guard(guard, "bstar", "loader_contract.json")
        manifest = _json_from_guard(guard, "bstar", "views/train_input_manifest.json")
        if set(contract.get("allowed_training_inputs", [])) != BSTAR_TRAIN_ALLOWLIST:
            raise ValueError("B* loader_contract 七项 allow-list 不匹配")
        manifest_paths = {
            str(item.get("path")) for item in manifest.get("allowed_training_inputs", [])
        }
        if manifest_paths != BSTAR_TRAIN_ALLOWLIST or not manifest.get("default_deny"):
            raise ValueError("B* train_input_manifest 未通过 default-deny allow-list 检查")
        if manifest.get("dataset_uid") != config.data.dataset_uid:
            raise ValueError("B* parent dataset UID 不匹配")
        if manifest.get("xstar_uid") != config.data.xstar_uid:
            raise ValueError("B* UID 不匹配")
        if manifest.get("source_train_split_sha256") != result["train_split_sha256"]:
            raise ValueError("B* source Train hash 与 Dataset B 不一致")
        result.update(
            {
                "xstar_uid": config.data.xstar_uid,
                "training_input_policy": "DEFAULT_DENY_EXPLICIT_ALLOWLIST",
                "training_allowlist": sorted(BSTAR_TRAIN_ALLOWLIST),
                "bstar_expected_assets": {
                    str(item["path"]): {
                        "sha256": str(item["sha256"]),
                        "bytes": int(item["bytes"]),
                    }
                    for item in manifest["allowed_training_inputs"]
                },
            }
        )
    return result


def _verify_train_opened_hashes(
    config: BXConfig, guard: GuardedDatasetIO, contract: Mapping[str, Any]
) -> None:
    if not config.data.verify_hashes:
        return
    opened = {
        (str(item["dataset"]), str(item["relative_path"])): item
        for item in guard.opened_paths
    }
    b_train = opened.get(("b", "splits/train.tsv"))
    if b_train is None:
        raise ValueError("Dataset B Train split 未在训练身份检查中打开")
    if (
        b_train["sha256"] != contract["train_split_sha256"]
        or b_train["bytes"] != contract["train_split_bytes"]
    ):
        raise ValueError("Dataset B Train split 实际 hash/bytes 与 dataset.json 不一致")
    if config.data.dataset_bstar_root:
        expected_assets = contract.get("bstar_expected_assets", {})
        for (dataset, relative), actual in opened.items():
            if dataset != "bstar" or relative not in expected_assets:
                continue
            expected = expected_assets[relative]
            if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
                raise ValueError(f"B* Train input hash/bytes 不匹配: {relative}")
        required_model_inputs = {
            "views/X/train.tsv",
            "nodes/train_protein.tsv",
        }
        if config.data.typed_level != "none":
            required_model_inputs.add(
                f"edges/train_pair_type_{config.data.typed_level}.tsv"
            )
        if not required_model_inputs.issubset(
            {relative for dataset, relative in opened if dataset == "bstar"}
        ):
            raise ValueError("B* Train 必需 allow-list 资产未全部打开")


def _verify_evaluation_opened_hashes(
    config: BXConfig,
    guard: GuardedDatasetIO,
    split_role: str,
) -> None:
    if not config.data.verify_hashes:
        return
    manifest = _json_from_guard(guard, "b", "manifest/dataset.json")
    expected = manifest.get("stats", {}).get(split_role, {})
    relative = f"splits/{split_role}.tsv"
    actual = next(
        item
        for item in guard.opened_paths
        if item["dataset"] == "b" and item["relative_path"] == relative
    )
    if actual["sha256"] != expected.get("output_sha256") or actual["bytes"] != expected.get("output_bytes"):
        raise ValueError(f"Dataset B {split_role} split hash/bytes 与 dataset.json 不一致")


def _assert_train_counts(config: BXConfig, pairs: PairTable) -> None:
    positive = sum(row.label for row in pairs.records)
    negative = len(pairs.records) - positive
    if config.data.dataset_uid == "BERNETT_V3_FULL_PAPER_SPLIT_V1":
        if (len(pairs.records), positive, negative) != CANONICAL_COUNTS["train"]:
            raise ValueError("Dataset B Train 冻结计数不一致")


def _external_feature_record(config: BXConfig, phase: str, project_root: Path) -> list[dict[str, object]]:
    if config.model.backend != "precomputed_embedding":
        return []
    name = "train" if phase == "train" else phase
    configured = getattr(config.model, f"precomputed_{name}_path")
    assert configured is not None
    path = Path(configured)
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return [
        {
            "dataset": "precomputed_embedding",
            "relative_path": configured,
            "absolute_path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    ]


def _build_model(config: BXConfig, type_vocabulary: Sequence[str], device: torch.device) -> SITNEBXModel:
    # 初始化由配置 seed 唯一决定，且发生在 checkpoint/optimizer 构建前。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.training.seed)
        model = SITNEBXModel(config.model, len(type_vocabulary))
    return model.to(device)


def _walk_context_batches(
    contexts: Sequence[WalkContext], batch_size: int, epoch: int
) -> Iterable[tuple[PairRecord, ...]]:
    for start in range(0, len(contexts), batch_size):
        yield tuple(
            PairRecord(
                protocol_row_id=f"walk-context:epoch{epoch}:{index}",
                protein_a=context.protein_a,
                protein_b=context.protein_b,
                label=1,
                split_role="train",
                undirected_pair_key="|".join(
                    sorted((context.protein_a, context.protein_b))
                ),
            )
            for index, context in enumerate(
                contexts[start : start + batch_size], start=start
            )
        )


def _make_scaler(config: BXConfig, device: torch.device):
    enabled = device.type == "cuda" and config.training.use_amp
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old supported torch fallback
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _checkpoint_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - torch < 2.0 fallback
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint 根节点非法: {path}")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"checkpoint format_version 不支持: {path}")
    return payload


def _load_model_from_checkpoint(
    config: BXConfig, checkpoint_path: Path, device: torch.device
) -> tuple[SITNEBXModel, dict[str, Any]]:
    payload = _checkpoint_load(checkpoint_path, device)
    if payload.get("config_sha256") != object_sha256(config.to_dict()):
        raise ValueError("checkpoint config hash 与当前配置不一致")
    vocabulary = tuple(str(value) for value in payload.get("type_vocabulary", []))
    model = _build_model(config, vocabulary, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def train(
    config: BXConfig,
    project_root: str | Path,
    run_id: str | None = None,
) -> Path:
    root = Path(project_root).resolve()
    b_root, x_root = resolve_roots(config, root)
    guard = GuardedDatasetIO(b_root, x_root, Phase.TRAIN)
    contract = _validate_train_contract(config, guard)
    pairs, sequences, annotations = load_train_data(config, guard)
    _verify_train_opened_hashes(config, guard, contract)
    _assert_train_counts(config, pairs)
    topology_graph = build_train_positive_graph(pairs)
    if config.data.dataset_uid == "BERNETT_V3_FULL_PAPER_SPLIT_V1":
        if topology_graph.node_count != 4286 or topology_graph.edge_count != 81596:
            raise ValueError(
                "Dataset B Train-positive topology graph 冻结计数不一致: "
                f"nodes={topology_graph.node_count}, edges={topology_graph.edge_count}"
            )
    if config.data.typed_level != annotations.level:
        raise AssertionError("typed level 内部不一致")
    if config.data.typed_level != "none" and not annotations.by_protocol_row_id:
        raise ValueError("启用 typed auxiliary 但没有安全 Train typed facts")

    device = resolve_device(config)
    configure_reproducibility(config, device)
    provider = resolve_feature_provider(config, "train", sequences, root)
    model = _build_model(config, annotations.type_vocabulary, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = _make_scaler(config, device)

    output_base = Path(config.training.output_dir)
    if not output_base.is_absolute():
        output_base = root / output_base
    actual_run_id = run_id or make_run_id(config.training.seed)
    run_dir = create_run_directory(output_base, actual_run_id)
    config_dict = config.to_dict()
    config_sha = object_sha256(config_dict)
    write_json_new(run_dir / "config_snapshot.json", config_dict)

    history: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(config.training.seed + 1009)
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        walk_seed = config.training.seed + 1_000_003 * epoch
        walk_contexts = sample_walk_contexts(
            topology_graph,
            walk_length=config.topology.walk_length,
            walks_per_node=config.topology.walks_per_node,
            context_window=config.topology.context_window,
            seed=walk_seed,
        )
        if config.training.max_batches_per_epoch is not None:
            walk_contexts = walk_contexts[
                : config.training.max_batches_per_epoch
                * config.topology.context_batch_size
            ]
        topology_batches = iter(
            _walk_context_batches(
                walk_contexts, config.topology.context_batch_size, epoch
            )
        )
        order = torch.randperm(len(pairs.records), generator=order_generator).tolist()
        aggregate = {
            "binary_sum": 0.0,
            "binary_rows": 0,
            "topology_sum": 0.0,
            "topology_rows": 0,
            "typed_sum": 0.0,
            "typed_rows": 0,
            "total_sum": 0.0,
            "batches": 0,
        }
        for batch_index, records in enumerate(
            iter_batches(pairs.records, config.training.batch_size, order), start=1
        ):
            if (
                config.training.max_batches_per_epoch is not None
                and batch_index > config.training.max_batches_per_epoch
            ):
                break
            payload, left, right = provider.pair_payload(records, device)
            labels = torch.tensor(
                [row.label for row in records], dtype=torch.float32, device=device
            )
            typed_targets, typed_mask = typed_targets_for_batch(
                records, annotations, device
            )
            try:
                context_records = next(topology_batches)
            except StopIteration:
                context_records = ()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                output = model(payload, left, right)
                topology_output = None
                if context_records:
                    context_payload, context_left, context_right = provider.pair_payload(
                        context_records, device
                    )
                    topology_output = model(
                        context_payload, context_left, context_right
                    )
                losses = composite_loss(
                    output,
                    labels,
                    typed_targets,
                    typed_mask,
                    config.loss,
                    topology_output=topology_output,
                )
            if not bool(torch.isfinite(losses.total)):
                raise FloatingPointError(
                    f"epoch={epoch} batch={batch_index} 出现非有限 loss"
                )
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(
                    f"epoch={epoch} batch={batch_index} 出现非有限 gradient"
                )
            scaler.step(optimizer)
            scaler.update()
            aggregate["binary_sum"] += float(losses.binary.detach()) * losses.binary_rows
            aggregate["binary_rows"] += losses.binary_rows
            aggregate["topology_sum"] += float(losses.topology.detach()) * losses.topology_rows
            aggregate["topology_rows"] += losses.topology_rows
            aggregate["typed_sum"] += float(losses.typed.detach()) * losses.typed_rows
            aggregate["typed_rows"] += losses.typed_rows
            aggregate["total_sum"] += float(losses.total.detach())
            aggregate["batches"] += 1
        # 一般默认参数下 topology context batches 少于 binary batches；这里仍
        # 完整处理剩余 contexts，避免用户调整 walk 参数后静默丢失图信号。
        for context_records in topology_batches:
            optimizer.zero_grad(set_to_none=True)
            context_payload, context_left, context_right = provider.pair_payload(
                context_records, device
            )
            with _autocast(config, device):
                topology_output = model(
                    context_payload, context_left, context_right
                )
                topology_loss = train_positive_topology_loss(
                    topology_output.embedding_a,
                    topology_output.embedding_b,
                    torch.ones(
                        len(context_records), dtype=torch.bool, device=device
                    ),
                )
                total_loss = config.loss.topology_weight * topology_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError(
                    f"epoch={epoch} topology-only batch 出现非有限 loss"
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(
                    f"epoch={epoch} topology-only batch 出现非有限 gradient"
                )
            scaler.step(optimizer)
            scaler.update()
            context_rows = len(context_records)
            aggregate["topology_sum"] += float(topology_loss.detach()) * context_rows
            aggregate["topology_rows"] += context_rows
            aggregate["total_sum"] += float(total_loss.detach())
            aggregate["batches"] += 1
        if aggregate["batches"] == 0:
            raise RuntimeError("训练 epoch 没有执行任何 batch")
        epoch_record = {
            "epoch": epoch,
            "batches": aggregate["batches"],
            "binary_rows": aggregate["binary_rows"],
            "positive_topology_rows": aggregate["topology_rows"],
            "walk_context_rows_generated": len(walk_contexts),
            "walk_seed": walk_seed,
            "typed_masked_rows": aggregate["typed_rows"],
            "mean_total_loss": aggregate["total_sum"] / aggregate["batches"],
            "mean_binary_loss": aggregate["binary_sum"] / aggregate["binary_rows"],
            "mean_topology_loss": aggregate["topology_sum"]
            / max(aggregate["topology_rows"], 1),
            "mean_typed_loss": aggregate["typed_sum"]
            / max(aggregate["typed_rows"], 1),
        }
        if not all(
            math.isfinite(value)
            for key, value in epoch_record.items()
            if key.startswith("mean_")
        ):
            raise FloatingPointError(f"epoch={epoch} 汇总 loss 非有限")
        history.append(epoch_record)
        checkpoint_path = run_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        checkpoint_payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": epoch,
            "config_sha256": config_sha,
            "dataset_uid": config.data.dataset_uid,
            "xstar_uid": config.data.xstar_uid if x_root else None,
            "type_level": annotations.level,
            "type_vocabulary": list(annotations.type_vocabulary),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch_record": epoch_record,
            "resolved_device": device.type,
            "topology_graph": {
                "nodes": topology_graph.node_count,
                "positive_edges": topology_graph.edge_count,
                "isolated_nodes": topology_graph.isolated_node_count,
                "walk_length": config.topology.walk_length,
                "walks_per_node": config.topology.walks_per_node,
                "context_window": config.topology.context_window,
                "walk_context_rows": len(walk_contexts),
                "walk_seed": walk_seed,
                "negative_edges_used": 0,
            },
        }
        torch_save_new(checkpoint_path, checkpoint_payload)
        checkpoint_records.append(
            {
                "epoch": epoch,
                "path": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            }
        )

    opened = guard.opened_paths + _external_feature_record(config, "train", root)
    provenance = collect_provenance(
        root, "train", config_dict, opened, resolved_device=device.type
    )
    provenance["contract"] = contract
    provenance["train_counts"] = {
        "rows": len(pairs.records),
        "positive": int(sum(row.label for row in pairs.records)),
        "negative": int(sum(1 - row.label for row in pairs.records)),
        "proteins": len(pairs.accessions),
        "typed_facts": annotations.fact_count,
        "typed_covered_positive_pairs": len(annotations.by_protocol_row_id),
    }
    provenance["train_positive_topology_graph"] = {
        "nodes": topology_graph.node_count,
        "positive_edges": topology_graph.edge_count,
        "isolated_nodes": topology_graph.isolated_node_count,
        "negative_edges_used": 0,
        "walk_length": config.topology.walk_length,
        "walks_per_node": config.topology.walks_per_node,
        "context_window": config.topology.context_window,
        "context_batch_size": config.topology.context_batch_size,
        "epochs": [
            {
                "epoch": item["epoch"],
                "walk_seed": item["walk_seed"],
                "walk_context_rows": item["walk_context_rows_generated"],
            }
            for item in history
        ],
    }
    write_json_new(run_dir / "provenance_train.json", provenance)
    write_json_new(
        run_dir / "train_history.json",
        {
            "schema_version": "sitne-bx-train-history-v1",
            "epochs": history,
            "scientific_evidence": False,
        },
    )
    train_complete = seal_payload(
        {
            "schema_version": "sitne-bx-train-complete-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha,
            "provenance_train_sha256": sha256_file(run_dir / "provenance_train.json"),
            "train_history_sha256": sha256_file(run_dir / "train_history.json"),
            "checkpoints": checkpoint_records,
            "selection_status": "NOT_STARTED",
            "test_accessed": False,
            "resolved_device": device.type,
            "scientific_evidence": False,
        }
    )
    write_json_new(run_dir / "train_complete.json", train_complete)
    return run_dir


def _verify_recorded_inputs_current(
    config: BXConfig,
    project_root: str | Path,
    records: Sequence[Mapping[str, object]],
    evidence_name: str,
) -> None:
    """逐项重算阶段 provenance 中每个输入的 bytes 与 SHA-256。"""

    root = Path(project_root).resolve()
    b_root, x_root = resolve_roots(config, root)
    for item in records:
        dataset = str(item.get("dataset"))
        relative = str(item.get("relative_path"))
        if dataset == "b":
            current_path = (b_root / relative).resolve()
            try:
                current_path.relative_to(b_root)
            except ValueError as exc:
                raise ValueError(f"{evidence_name} B input 路径逃逸") from exc
        elif dataset == "bstar":
            if x_root is None:
                raise ValueError(f"{evidence_name} 声明 B* input，但未配置 B* root")
            current_path = (x_root / relative).resolve()
            try:
                current_path.relative_to(x_root)
            except ValueError as exc:
                raise ValueError(f"{evidence_name} B* input 路径逃逸") from exc
        elif dataset == "precomputed_embedding":
            current_path = Path(str(item.get("absolute_path"))).resolve()
        else:
            raise ValueError(f"{evidence_name} 包含未知 input dataset: {dataset}")
        if not current_path.is_file():
            raise ValueError(f"{evidence_name} input 已缺失: {current_path}")
        if (
            current_path.stat().st_size != int(item.get("bytes", -1))
            or sha256_file(current_path) != item.get("sha256")
        ):
            raise ValueError(
                f"{evidence_name} input 当前 bytes/SHA-256 不匹配: "
                f"{dataset}:{relative}"
            )


def _load_run_contract(
    config: BXConfig, run_dir: str | Path, project_root: str | Path
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    directory = Path(run_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    snapshot = read_json(directory / "config_snapshot.json")
    if snapshot != config.to_dict():
        raise ValueError("当前配置与 run/config_snapshot.json 不一致")
    complete = read_json(directory / "train_complete.json")
    verify_sealed_payload(complete)
    if complete.get("config_sha256") != object_sha256(snapshot):
        raise ValueError("train_complete config hash 不一致")
    require_file_hash(
        directory / "provenance_train.json", str(complete["provenance_train_sha256"])
    )
    train_provenance = read_json(directory / "provenance_train.json")
    if train_provenance.get("config_sha256") != object_sha256(snapshot):
        raise ValueError("Train provenance config hash 不一致")
    if train_provenance.get("implementation_hashes") != collect_implementation_hashes(
        project_root
    ):
        raise ValueError("当前实现文件 hash 与 Train provenance 不一致")
    _verify_recorded_inputs_current(
        config,
        project_root,
        train_provenance.get("opened_input_paths", []),
        "Train provenance",
    )
    require_file_hash(
        directory / "train_history.json", str(complete["train_history_sha256"])
    )
    for item in complete.get("checkpoints", []):
        path = (directory / str(item["path"])).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ValueError("checkpoint path 逃逸 run directory") from exc
        require_file_hash(path, str(item["sha256"]))
    if not complete.get("checkpoints"):
        raise ValueError("train_complete 没有 checkpoint")
    return directory, snapshot, complete


@torch.no_grad()
def _predict(
    config: BXConfig,
    model: SITNEBXModel,
    records: Sequence[PairRecord],
    provider,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    for batch in iter_batches(records, config.evaluation.batch_size):
        payload, left, right = provider.pair_payload(batch, device)
        with _autocast(config, device):
            logits = model(payload, left, right).binary_logits
        batch_probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        if not np.isfinite(batch_probabilities).all():
            raise FloatingPointError("prediction 包含非有限概率")
        probabilities.append(batch_probabilities)
    result = np.concatenate(probabilities)
    if result.shape != (len(records),):
        raise AssertionError("prediction 数量内部不一致")
    return result


def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1 - 1e-12)
    y = labels.astype(np.float64)
    return float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean())


def _prediction_rows(
    pairs: PairTable, probabilities: np.ndarray, threshold: float
) -> Iterable[dict[str, object]]:
    for row, probability in zip(pairs.records, probabilities.tolist(), strict=True):
        yield {
            "protocol_row_id": row.protocol_row_id,
            "split_role": row.split_role,
            "protein_a": row.protein_a,
            "protein_b": row.protein_b,
            "undirected_pair_key": row.undirected_pair_key,
            "label": row.label,
            "probability": format(probability, ".17g"),
            "threshold": format(threshold, ".17g"),
            "predicted_label": int(probability >= threshold),
        }


PREDICTION_FIELDS = (
    "protocol_row_id",
    "split_role",
    "protein_a",
    "protein_b",
    "undirected_pair_key",
    "label",
    "probability",
    "threshold",
    "predicted_label",
)


def select(config: BXConfig, project_root: str | Path, run_dir: str | Path) -> Path:
    root = Path(project_root).resolve()
    directory, _, complete = _load_run_contract(config, run_dir, root)
    if (directory / "selection_manifest.json").exists():
        raise FileExistsError("selection_manifest.json 已存在；禁止覆盖模型选择证据")
    device = resolve_device(config)
    configure_reproducibility(config, device)
    b_root, x_root = resolve_roots(config, root)
    guard = GuardedDatasetIO(b_root, x_root, Phase.SELECT)
    pairs, sequences = load_evaluation_data(config, guard, "validation")
    _verify_evaluation_opened_hashes(config, guard, "validation")
    provider = resolve_feature_provider(config, "validation", sequences, root)
    labels = pairs.labels
    candidates: list[dict[str, Any]] = []
    cached_probabilities: dict[str, np.ndarray] = {}
    for item in complete["checkpoints"]:
        checkpoint_path = directory / str(item["path"])
        model, checkpoint = _load_model_from_checkpoint(config, checkpoint_path, device)
        probabilities = _predict(config, model, pairs.records, provider, device)
        metrics = binary_metrics(labels, probabilities, threshold=0.5)
        record = {
            "epoch": int(checkpoint["epoch"]),
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_sha256": item["sha256"],
            "validation_auprc": metrics["auprc"],
            "validation_auroc": metrics["auroc"],
            "validation_log_loss": _binary_log_loss(labels, probabilities),
        }
        candidates.append(record)
        cached_probabilities[checkpoint_path.name] = probabilities
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected = max(
        candidates,
        key=lambda item: (
            item["validation_auprc"],
            -item["validation_log_loss"],
            -item["epoch"],
        ),
    )
    probabilities = cached_probabilities[selected["checkpoint_path"]]
    threshold, selected_mcc = select_mcc_threshold(labels, probabilities)
    metrics = binary_metrics(labels, probabilities, threshold)
    metrics.update(
        {
            "schema_version": "sitne-bx-validation-metrics-v1",
            "split_role": "validation",
            "checkpoint": selected["checkpoint_path"],
            "checkpoint_selection_rule": (
                "max validation AUPRC; ties: min validation log loss; ties: earliest epoch"
            ),
            "threshold_selection_rule": (
                "max validation MCC; ties: closest to 0.5; ties: highest threshold"
            ),
            "selected_mcc_crosscheck": selected_mcc,
            "scientific_evidence": False,
        }
    )
    write_tsv_new(
        directory / "validation_predictions.tsv",
        PREDICTION_FIELDS,
        _prediction_rows(pairs, probabilities, threshold),
    )
    write_json_new(directory / "validation_metrics.json", metrics)
    opened = guard.opened_paths + _external_feature_record(config, "validation", root)
    validation_split = next(
        item for item in opened if item["dataset"] == "b" and item["relative_path"] == "splits/validation.tsv"
    )
    selection_payload = seal_payload(
        {
            "schema_version": "sitne-bx-selection-manifest-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": object_sha256(config.to_dict()),
            "train_complete_sha256": sha256_file(directory / "train_complete.json"),
            "selected_checkpoint": selected["checkpoint_path"],
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
            "selected_epoch": selected["epoch"],
            "selection_metric": "validation_auprc",
            "selection_candidates": candidates,
            "threshold": threshold,
            "threshold_metric": "validation_mcc",
            "validation_metrics_sha256": sha256_file(directory / "validation_metrics.json"),
            "validation_predictions_sha256": sha256_file(directory / "validation_predictions.tsv"),
            "validation_split_sha256": validation_split["sha256"],
            "training_state_frozen": True,
            "test_accessed": False,
            "opened_validation_paths": opened,
            "scientific_evidence": False,
        }
    )
    # 最后写入 sealed manifest；其存在是 test 解锁的唯一状态门。
    write_json_new(directory / "selection_manifest.json", selection_payload)
    return directory / "selection_manifest.json"


def _load_selection_gate(
    config: BXConfig,
    directory: Path,
    complete: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    selection_path = directory / "selection_manifest.json"
    if not selection_path.is_file():
        raise PermissionError("test 被锁定：缺少 sealed selection_manifest.json")
    selection = read_json(selection_path)
    verify_sealed_payload(selection)
    if selection.get("training_state_frozen") is not True:
        raise PermissionError("test 被锁定：training_state_frozen 不是 true")
    if selection.get("config_sha256") != object_sha256(config.to_dict()):
        raise ValueError("selection manifest config hash 不匹配")
    if selection.get("train_complete_sha256") != sha256_file(directory / "train_complete.json"):
        raise ValueError("selection manifest 的 train_complete hash 不匹配")
    checkpoint = directory / str(selection.get("selected_checkpoint"))
    require_file_hash(checkpoint, str(selection.get("selected_checkpoint_sha256")))
    require_file_hash(
        directory / "validation_metrics.json", str(selection["validation_metrics_sha256"])
    )
    require_file_hash(
        directory / "validation_predictions.tsv",
        str(selection["validation_predictions_sha256"]),
    )
    checkpoint_hashes = {str(item["sha256"]) for item in complete["checkpoints"]}
    if selection.get("selected_checkpoint_sha256") not in checkpoint_hashes:
        raise ValueError("selection checkpoint 不属于冻结训练 checkpoint 集合")
    _verify_recorded_inputs_current(
        config,
        project_root,
        selection.get("opened_validation_paths", []),
        "Selection manifest",
    )
    return selection


def create_post_selection_guard(
    config: BXConfig,
    project_root: str | Path,
    run_dir: str | Path,
    phase: Phase | str,
) -> tuple[GuardedDatasetIO, dict[str, Any]]:
    """验证完整 run/selection 链后，创建可解锁 B* held-out 的 guard。

    binary pipeline 本身不会读取 held-out typed targets；该接口供冻结选择后的
    coverage/open-set 分析使用。SELECT 阶段明确不接受 capability。
    """

    requested_phase = Phase(phase)
    if requested_phase not in {Phase.TEST, Phase.VERIFY_RUN}:
        raise ValueError("post-selection guard 只能用于 test 或 verify-run")
    root = Path(project_root).resolve()
    directory, _, complete = _load_run_contract(config, run_dir, root)
    selection = _load_selection_gate(config, directory, complete, root)
    selection_sha = sha256_file(directory / "selection_manifest.json")
    capability = _issue_post_selection_capability(directory, selection_sha)
    b_root, x_root = resolve_roots(config, root)
    return (
        GuardedDatasetIO(
            b_root,
            x_root,
            requested_phase,
            post_selection_capability=capability,
        ),
        selection,
    )


def test(config: BXConfig, project_root: str | Path, run_dir: str | Path) -> Path:
    root = Path(project_root).resolve()
    directory, _, complete = _load_run_contract(config, run_dir, root)
    selection = _load_selection_gate(config, directory, complete, root)
    for artifact in ("test_predictions.tsv", "test_metrics.json", "test_provenance.json"):
        if (directory / artifact).exists():
            raise FileExistsError(f"{artifact} 已存在；禁止覆盖 Test 证据")
    device = resolve_device(config)
    configure_reproducibility(config, device)
    # selection gate 完整验证后才构造 TEST guard 并打开 Test split。
    capability = _issue_post_selection_capability(
        directory, sha256_file(directory / "selection_manifest.json")
    )
    b_root, x_root = resolve_roots(config, root)
    guard = GuardedDatasetIO(
        b_root, x_root, Phase.TEST, post_selection_capability=capability
    )
    pairs, sequences = load_evaluation_data(config, guard, "test")
    _verify_evaluation_opened_hashes(config, guard, "test")
    if len(pairs.records) != config.evaluation.expected_test_rows:
        raise ValueError(
            f"Test 行数 {len(pairs.records)} != expected_test_rows "
            f"{config.evaluation.expected_test_rows}"
        )
    provider = resolve_feature_provider(config, "test", sequences, root)
    checkpoint_path = directory / str(selection["selected_checkpoint"])
    model, checkpoint = _load_model_from_checkpoint(config, checkpoint_path, device)
    probabilities = _predict(config, model, pairs.records, provider, device)
    threshold = float(selection["threshold"])
    metrics = binary_metrics(pairs.labels, probabilities, threshold)
    metrics.update(
        {
            "schema_version": "sitne-bx-test-metrics-v1",
            "split_role": "test",
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": selection["selected_checkpoint_sha256"],
            "selected_epoch": int(checkpoint["epoch"]),
            "threshold_source": "sealed_validation_selection_manifest",
            "selection_manifest_sha256": sha256_file(directory / "selection_manifest.json"),
            "scientific_evidence": False,
        }
    )
    write_tsv_new(
        directory / "test_predictions.tsv",
        PREDICTION_FIELDS,
        _prediction_rows(pairs, probabilities, threshold),
    )
    metrics["test_predictions_sha256"] = sha256_file(directory / "test_predictions.tsv")
    opened = guard.opened_paths + _external_feature_record(config, "test", root)
    test_split = next(
        item for item in opened if item["dataset"] == "b" and item["relative_path"] == "splits/test.tsv"
    )
    metrics["test_split_sha256"] = test_split["sha256"]
    write_json_new(directory / "test_metrics.json", metrics)
    provenance = collect_provenance(
        root, "test", config.to_dict(), opened, resolved_device=device.type
    )
    provenance.update(
        {
            "selection_manifest_sha256": sha256_file(directory / "selection_manifest.json"),
            "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
            "test_metrics_sha256": sha256_file(directory / "test_metrics.json"),
            "test_predictions_sha256": sha256_file(directory / "test_predictions.tsv"),
            "post_selection_capability": guard.capability_record,
        }
    )
    write_json_new(directory / "test_provenance.json", provenance)
    return directory / "test_metrics.json"


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != PREDICTION_FIELDS:
            raise ValueError("prediction TSV 表头不符合冻结 schema")
        return list(reader)


def verify_run(
    config: BXConfig, project_root: str | Path, run_dir: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    directory, _, complete = _load_run_contract(config, run_dir, root)
    selection = _load_selection_gate(config, directory, complete, root)
    metrics = read_json(directory / "test_metrics.json")
    provenance = read_json(directory / "test_provenance.json")
    prediction_path = directory / "test_predictions.tsv"
    require_file_hash(prediction_path, str(metrics["test_predictions_sha256"]))
    require_file_hash(prediction_path, str(provenance["test_predictions_sha256"]))
    require_file_hash(
        directory / "test_metrics.json", str(provenance["test_metrics_sha256"])
    )
    if provenance.get("selection_manifest_sha256") != sha256_file(
        directory / "selection_manifest.json"
    ):
        raise ValueError("test provenance 的 selection manifest hash 不匹配")
    if provenance.get("resolved_device") != config.training.device:
        raise ValueError("test provenance resolved_device 与正式配置不一致")
    environment = provenance.get("environment", {})
    if config.training.device == "cuda" and not environment.get("cuda_available"):
        raise ValueError("正式 CUDA run provenance 未证明 CUDA available")
    if config.training.device == "cuda" and int(environment.get("cuda_device_count", 0)) <= 0:
        raise ValueError("正式 CUDA run provenance 没有 NVIDIA device")
    current_hashes = collect_implementation_hashes(root)
    if provenance.get("implementation_hashes") != current_hashes:
        raise ValueError("当前实现文件 hash 与 Test provenance 不一致")
    _verify_recorded_inputs_current(
        config,
        root,
        provenance.get("opened_input_paths", []),
        "Test provenance",
    )

    predictions = _read_predictions(prediction_path)
    if len(predictions) != config.evaluation.expected_test_rows:
        raise ValueError("Test prediction 行数不等于 expected_test_rows")
    ids = [row["protocol_row_id"] for row in predictions]
    if len(set(ids)) != len(ids):
        raise ValueError("Test prediction protocol_row_id 不唯一")
    probabilities = np.asarray([float(row["probability"]) for row in predictions])
    labels = np.asarray([int(row["label"]) for row in predictions])
    thresholds = {float(row["threshold"]) for row in predictions}
    if len(thresholds) != 1 or next(iter(thresholds)) != float(selection["threshold"]):
        raise ValueError("Test prediction threshold 与 selection manifest 不一致")
    if not np.isfinite(probabilities).all() or bool(
        (probabilities < 0).any() or (probabilities > 1).any()
    ):
        raise ValueError("Test prediction probability 非有限或越界")
    for row, probability in zip(predictions, probabilities.tolist(), strict=True):
        if int(row["predicted_label"]) != int(probability >= float(selection["threshold"])):
            raise ValueError("Test predicted_label 与 probability/threshold 不一致")

    capability = _issue_post_selection_capability(
        directory, sha256_file(directory / "selection_manifest.json")
    )
    b_root, x_root = resolve_roots(config, root)
    guard = GuardedDatasetIO(
        b_root,
        x_root,
        Phase.VERIFY_RUN,
        post_selection_capability=capability,
    )
    canonical = load_pair_table(
        guard,
        "b",
        "splits/test.tsv",
        "test",
        expected_uid=config.data.dataset_uid,
    )
    _verify_evaluation_opened_hashes(config, guard, "test")
    canonical_identity = [
        (row.protocol_row_id, row.protein_a, row.protein_b, row.label, row.undirected_pair_key)
        for row in canonical.records
    ]
    prediction_identity = [
        (
            row["protocol_row_id"],
            row["protein_a"],
            row["protein_b"],
            int(row["label"]),
            row["undirected_pair_key"],
        )
        for row in predictions
    ]
    if prediction_identity != canonical_identity:
        raise ValueError("Test predictions 与 canonical Test 身份/顺序不一致")
    split_record = next(
        item for item in guard.opened_paths if item["relative_path"] == "splits/test.tsv"
    )
    if split_record["sha256"] != metrics.get("test_split_sha256"):
        raise ValueError("Test split hash 与 metrics 不一致")

    recomputed = binary_metrics(labels, probabilities, float(selection["threshold"]))
    for key in (
        "rows",
        "threshold",
        "auroc",
        "auprc",
        "mcc",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
    ):
        observed = metrics.get(key)
        expected = recomputed[key]
        if isinstance(expected, float):
            if not math.isclose(float(observed), expected, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Test metric {key} 与 predictions 重算不一致")
        elif observed != expected:
            raise ValueError(f"Test metric {key} 与 predictions 重算不一致")
    if metrics.get("confusion_matrix") != recomputed["confusion_matrix"]:
        raise ValueError("Test confusion matrix 重算不一致")
    return {
        "schema_version": "sitne-bx-verify-run-v1",
        "status": "PASS",
        "run_dir": str(directory),
        "test_rows": len(predictions),
        "unique_protocol_row_ids": len(set(ids)),
        "resolved_device": provenance["resolved_device"],
        "cuda_provenance_valid": config.training.device == "cuda",
        "selection_manifest_seal_valid": True,
        "checkpoint_hash_valid": True,
        "implementation_hashes_valid": True,
        "test_split_hash_valid": True,
        "prediction_metrics_recomputed": True,
        "scientific_evidence": False,
        "evidence_boundary": (
            "verify-run 证明工程 artifact 自洽，不单独构成论文性能结论。"
        ),
    }
