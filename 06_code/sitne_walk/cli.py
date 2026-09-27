"""SITNE-Walk 命令行入口。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Sequence

import pandas as pd
import torch

from .config import SITNEConfig
from .data import (
    IndexedTriples,
    PairRelationIndex,
    TrainingTriples,
    audit_split_pair_overlap,
    iter_batches,
    load_indexed_triples,
    load_relation_modes,
    load_training_triples,
    sha256_file,
    validate_split_manifest,
)
from .evaluator import RankingResult, evaluate_filtered_type_ranking
from .graph import PackedCSRGraph, build_packed_csr_graph
from .provenance import (
    collect_runtime_provenance,
    create_new_run_directory,
    make_run_id,
    write_json_new,
)
from .trainer import SITNETrainer, build_model

logger = logging.getLogger(__name__)


def _resolve(project_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_core(
    config: SITNEConfig,
    project_root: Path,
) -> tuple[TrainingTriples, PackedCSRGraph, dict[str, object]]:
    """执行训练前数据、manifest、overlap、方向与图构建检查。"""

    train_path = _resolve(project_root, config.data.train_path)
    validation_path = _resolve(project_root, config.data.validation_path)
    test_path = _resolve(project_root, config.data.test_path)
    assert train_path is not None

    split_audit: dict[str, object] = {
        "pair_overlap": audit_split_pair_overlap(
            train_path,
            validation_path,
            test_path,
        )
    }
    if config.data.split_manifest_path:
        manifest_path = _resolve(project_root, config.data.split_manifest_path)
        assert manifest_path is not None
        split_audit["manifest"] = validate_split_manifest(
            manifest_path,
            train_path,
            validation_path,
            test_path,
            require_hashes=config.data.require_manifest_hashes,
        )

    triples = load_training_triples(
        train_path,
        duplicate_policy=config.data.duplicate_policy,
        weight_column=config.data.weight_column,
        pair_semantics=config.data.pair_semantics,
    )
    relation_metadata_path = _resolve(
        project_root,
        config.data.relation_metadata_path,
    )
    relation_modes = load_relation_modes(
        relation_metadata_path,
        triples.id_to_relation,
        default_mode=config.data.default_relation_mode,
    )
    graph = build_packed_csr_graph(
        triples,
        relation_modes=relation_modes,
        alpha=config.walk.alpha,
        beta=config.walk.beta,
        epsilon=config.walk.epsilon,
        drop_self_loops=config.data.drop_self_loops_from_walks,
        correction_mode=config.walk.correction_mode,
    )
    return triples, graph, split_audit


def _load_optional_split(
    path: Path | None,
    triples: TrainingTriples,
) -> IndexedTriples | None:
    if path is None:
        return None
    return load_indexed_triples(
        path,
        triples.protein_to_id,
        triples.relation_to_id,
        pair_semantics=triples.pair_semantics,
    )


def _load_all_known_positive_parts(
    config: SITNEConfig,
    project_root: Path,
    triples: TrainingTriples,
) -> list[IndexedTriples]:
    """读取只用于 filtered evaluation 的全局阳性集合。"""

    parts: list[IndexedTriples] = []
    for raw_path in config.data.all_known_positive_paths:
        path = _resolve(project_root, raw_path)
        assert path is not None
        parts.append(
            load_indexed_triples(
                path,
                triples.protein_to_id,
                triples.relation_to_id,
                pair_semantics=triples.pair_semantics,
            )
        )
    return parts


def _assert_filter_contains_targets(
    split_name: str,
    split: IndexedTriples,
    filter_index: PairRelationIndex,
    batch_size: int = 65_536,
) -> dict[str, int]:
    """训练前核对 filtered-positive 集合覆盖全部可支持 target。"""

    missing_pairs = 0
    missing_targets = 0
    for batch_indices in iter_batches(split.num_triples, batch_size):
        heads = split.heads[batch_indices]
        relations = split.relations[batch_indices]
        tails = split.tails[batch_indices]
        masks, matched = filter_index.lookup(heads, tails)
        target_known = masks.gather(1, relations[:, None]).squeeze(1)
        missing_pairs += int((~matched).sum().item())
        missing_targets += int((matched & ~target_known).sum().item())
    if missing_pairs or missing_targets:
        raise ValueError(
            f"all-known-positive filter 未覆盖 {split_name} targets: "
            f"missing_pairs={missing_pairs}, missing_targets={missing_targets}"
        )
    return {
        "checked_supported_targets": split.num_triples,
        "missing_pairs": 0,
        "missing_targets": 0,
    }


def _combine_filter_index(
    triples: TrainingTriples,
    extras: Sequence[IndexedTriples],
) -> PairRelationIndex:
    heads = [triples.heads, *(extra.heads for extra in extras)]
    relations = [triples.relations, *(extra.relations for extra in extras)]
    tails = [triples.tails, *(extra.tails for extra in extras)]
    index, _ = PairRelationIndex.build(
        torch.cat(heads),
        torch.cat(relations),
        torch.cat(tails),
        num_proteins=triples.num_proteins,
        num_relations=triples.num_relations,
        pair_semantics=triples.pair_semantics,
    )
    return index


def _input_record(path: Path, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _write_vocabularies(run_dir: Path, triples: TrainingTriples) -> None:
    protein_path = run_dir / "protein_vocab.tsv"
    relation_path = run_dir / "relation_vocab.tsv"
    pd.DataFrame(
        {"protein_id": range(triples.num_proteins), "protein": triples.id_to_protein}
    ).to_csv(protein_path, sep="\t", index=False, mode="x")
    pd.DataFrame(
        {"relation_id": range(triples.num_relations), "type_name": triples.id_to_relation}
    ).to_csv(relation_path, sep="\t", index=False, mode="x")


def _write_numerical_diagnostics(
    run_dir: Path,
    trainer: "SITNETrainer",
    history: Sequence[dict[str, float]],
) -> None:
    """落盘数值可靠性诊断：epoch 汇总、non-finite 事件与 debug step 日志。

    只在文件尚不存在时创建（exclusive），绝不覆盖已有证据。fail_fast 下正常完成
    会产生仅含表头的空 ``nonfinite_events.tsv``；发生 non-finite 失败时该文件
    会记录触发事件，便于定位 NaN 来源。
    """

    if history:
        pd.DataFrame(history).to_csv(
            run_dir / "epoch_summary.tsv", sep="\t", index=False, mode="x"
        )
    if trainer.nonfinite_events:
        pd.DataFrame(trainer.nonfinite_events).to_csv(
            run_dir / "nonfinite_events.tsv", sep="\t", index=False, mode="x"
        )
    else:
        pd.DataFrame(columns=["epoch", "step", "kind"]).to_csv(
            run_dir / "nonfinite_events.tsv", sep="\t", index=False, mode="x"
        )
    if trainer.non_finite_policy == "skip_and_log" and trainer.step_log:
        pd.DataFrame(trainer.step_log).to_csv(
            run_dir / "step_log.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
            mode="x",
        )


def _save_ranks(
    path: Path,
    triples: IndexedTriples,
    ranking: RankingResult,
) -> None:
    if ranking.ranks.numel() != triples.num_triples:
        raise ValueError("rank 数与 supported triples 数不一致")
    pd.DataFrame(
        {
            "head_id": triples.heads.numpy(),
            "relation_id": triples.relations.numpy(),
            "tail_id": triples.tails.numpy(),
            "filtered_rank": ranking.ranks.numpy(),
        }
    ).to_csv(path, sep="\t", index=False, mode="x")


def command_inspect(config: SITNEConfig, project_root: Path) -> int:
    triples, graph, split_audit = _load_core(config, project_root)
    validation = _load_optional_split(
        _resolve(project_root, config.data.validation_path),
        triples,
    )
    test = _load_optional_split(_resolve(project_root, config.data.test_path), triples)
    all_known_parts = _load_all_known_positive_parts(config, project_root, triples)
    filter_audit: dict[str, object] = {
        "mode": "explicit_all_known" if all_known_parts else "split_union_fallback",
        "sources": [
            {
                "source": part.source_path,
                "raw_input_rows": part.raw_input_rows,
                "input_rows": part.input_rows,
                "supported_rows": part.supported_rows,
            }
            for part in all_known_parts
        ],
    }
    if all_known_parts:
        explicit_filter = _combine_filter_index(triples, all_known_parts)
        for split_name, split in (("validation", validation), ("test", test)):
            if split is not None:
                filter_audit[split_name] = _assert_filter_contains_targets(
                    split_name,
                    split,
                    explicit_filter,
                )
    report: dict[str, object] = {
        "train": {
            "source": triples.source_path,
            "sha256": triples.source_sha256,
            "input_rows": triples.input_rows,
            "unique_rows": triples.unique_rows,
            "num_proteins": triples.num_proteins,
            "num_relations": triples.num_relations,
            "num_pairs": int(triples.pair_index.pair_keys.numel()),
        },
        "graph": {
            "num_edges": graph.num_edges,
            "nodes_with_outgoing": int(graph.nodes_with_outgoing_edges.numel()),
            "alpha": graph.alpha,
            "beta": graph.beta,
            "drop_self_loops": config.data.drop_self_loops_from_walks,
        },
        "split_audit": split_audit,
        "known_positive_filter_audit": filter_audit,
    }
    for name, split in (("validation", validation), ("test", test)):
        if split is not None:
            report[name] = {
                "raw_input_rows": split.raw_input_rows,
                "input_rows": split.input_rows,
                "supported_rows": split.supported_rows,
                "unsupported_protein_rows": split.unsupported_protein_rows,
                "unsupported_relation_rows": split.unsupported_relation_rows,
                "coverage": split.supported_rows / split.input_rows,
            }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_train(config: SITNEConfig, project_root: Path) -> int:
    triples, graph, split_audit = _load_core(config, project_root)
    validation_path = _resolve(project_root, config.data.validation_path)
    test_path = _resolve(project_root, config.data.test_path)
    validation = _load_optional_split(validation_path, triples)
    test = _load_optional_split(test_path, triples)

    if config.evaluation.require_full_coverage:
        for split_name, split in (("validation", validation), ("test", test)):
            if split is not None and split.supported_rows != split.input_rows:
                raise ValueError(
                    f"{split_name} 含 transductive 模型不支持的查询；"
                    f"supported={split.supported_rows}/{split.input_rows}, "
                    f"unsupported_protein_rows={split.unsupported_protein_rows}, "
                    f"unsupported_relation_rows={split.unsupported_relation_rows}"
                )

    all_known_positive_parts = _load_all_known_positive_parts(
        config,
        project_root,
        triples,
    )

    validation_filter = None
    explicit_filter: PairRelationIndex | None = None
    if all_known_positive_parts:
        explicit_filter = _combine_filter_index(triples, all_known_positive_parts)
        for split_name, split in (("validation", validation), ("test", test)):
            if split is not None:
                _assert_filter_contains_targets(split_name, split, explicit_filter)
    if validation is not None and explicit_filter is not None:
        # all-known positives 只影响 filtered candidate mask，不进入图、loss 或梯度。
        validation_filter = explicit_filter
    elif validation is not None:
        validation_filter = _combine_filter_index(
            triples,
            [validation],
        )
    model, nuisance_labels, nuisance_weights = build_model(config, triples, graph)
    trainer = SITNETrainer(
        config,
        triples,
        graph,
        model,
        nuisance_labels,
        nuisance_weights,
    )

    run_id = make_run_id("sitne_walk", config.training.seed)
    output_base = _resolve(project_root, config.training.output_dir)
    assert output_base is not None
    run_dir = create_new_run_directory(output_base, run_id)
    try:
        # 从 run_dir 创建后的第一项写操作起闭合状态机：任何异常都必须留下
        # failure.json，避免出现无法判断成败的半成品目录。
        _write_vocabularies(run_dir, triples)

        input_files: dict[str, dict[str, object]] = {
            "train": _input_record(
                Path(triples.source_path),
                rows=triples.input_rows,
            )
        }
        if validation_path and validation is not None:
            input_files["validation"] = _input_record(
                validation_path,
                rows=validation.raw_input_rows,
            )
        if test_path and test is not None:
            input_files["test"] = _input_record(
                test_path,
                rows=test.raw_input_rows,
            )
        if config.data.split_manifest_path:
            manifest_path = _resolve(project_root, config.data.split_manifest_path)
            assert manifest_path is not None
            input_files["split_manifest"] = _input_record(manifest_path)
        if config.data.relation_metadata_path:
            relation_metadata_path = _resolve(
                project_root,
                config.data.relation_metadata_path,
            )
            assert relation_metadata_path is not None
            input_files["relation_metadata"] = _input_record(
                relation_metadata_path
            )
        for index, raw_path in enumerate(config.data.all_known_positive_paths):
            positive_path = _resolve(project_root, raw_path)
            assert positive_path is not None
            input_files[f"all_known_positive_{index}"] = _input_record(
                positive_path,
                rows=all_known_positive_parts[index].raw_input_rows,
            )
        write_json_new(run_dir / "resolved_config.json", config.to_dict())
        provenance = collect_runtime_provenance(
            project_root,
            config.to_dict(),
            input_files,
        )
        provenance["split_audit"] = split_audit
        provenance["selected_device"] = str(trainer.device)
        write_json_new(run_dir / "provenance.json", provenance)

        fit_result = trainer.fit(
            validation=validation,
            validation_filter=validation_filter,
            checkpoint_dir=run_dir / "checkpoints",
        )
        write_json_new(
            run_dir / "training_history.json",
            {
                "best_epoch": fit_result.best_epoch,
                "best_selection_metric": fit_result.best_selection_metric,
                "best_checkpoint": fit_result.best_checkpoint,
                "history": list(fit_result.history),
            },
        )
        _write_numerical_diagnostics(run_dir, trainer, list(fit_result.history))

        if fit_result.best_checkpoint:
            trainer.load_checkpoint(fit_result.best_checkpoint)

        if test is not None:
            if explicit_filter is not None:
                test_filter = explicit_filter
            else:
                # 未显式给出全局 positive manifest 时，至少合并当前已提供的 train、
                # validation、test；该行为会写入 provenance/config，绝不静默发生。
                filter_parts: list[IndexedTriples] = []
                if validation is not None:
                    filter_parts.append(validation)
                filter_parts.append(test)
                test_filter = _combine_filter_index(triples, filter_parts)
            test_result = evaluate_filtered_type_ranking(
                trainer.model,
                test,
                test_filter,
                device=trainer.device,
                batch_size=config.training.evaluation_batch_size,
                hits_at=config.evaluation.hits_at,
                tie_policy=config.evaluation.tie_policy,
                require_full_coverage=config.evaluation.require_full_coverage,
            )
            write_json_new(run_dir / "test_metrics.json", test_result.to_dict())
            if test_result.status != "unsupported":
                _save_ranks(run_dir / "test_ranks.tsv", test, test_result)

        write_json_new(
            run_dir / "completed.json",
            {
                "status": "completed",
                "best_epoch": fit_result.best_epoch,
                "best_selection_metric": fit_result.best_selection_metric,
            },
        )
    except Exception as error:
        # run_dir 已经创建后发生的错误必须留下机器可读证据，便于区分完成运行和
        # 部分运行；错误仍会向上传播，绝不把失败伪装成成功。
        # fail_fast 触发的 non-finite 也在此处保留诊断快照，便于定位 NaN 来源。
        _write_numerical_diagnostics(run_dir, trainer, [])
        write_json_new(
            run_dir / "failure.json",
            {
                "status": "failed",
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise

    logger.info("run completed: %s", run_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SITNE-Walk: CUDA-ready shortcut-corrected typed walks",
    )
    parser.add_argument(
        "command",
        choices=("inspect", "train"),
        help="inspect 仅审计输入；train 执行训练",
    )
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument(
        "--project-root",
        default=None,
        help="项目根目录；默认由当前脚本位置推断",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[2]
    )
    config = SITNEConfig.from_yaml(args.config)
    if args.command == "inspect":
        return command_inspect(config, project_root)
    return command_train(config, project_root)


if __name__ == "__main__":
    sys.exit(main())
