from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

from sitne_walk.config import SITNEConfig
from sitne_walk.cli import command_train
from sitne_walk.data import load_training_triples
from sitne_walk.graph import build_packed_csr_graph
from sitne_walk.trainer import FitResult, SITNETrainer, build_model, resolve_device


def _config(train_path: Path, output_dir: Path) -> SITNEConfig:
    return SITNEConfig.from_mapping(
        {
            "data": {
                "train_path": str(train_path),
                "duplicate_policy": "binary",
                "pair_semantics": "unordered",
                "default_relation_mode": "symmetric",
            },
            "walk": {
                "alpha": 0.5,
                "beta": 0.5,
                "walk_length": 3,
                "walks_per_protein": 1,
                "walk_batch_size": 4,
                "context_window": 1,
                "negative_samples": 2,
                "max_topology_pairs_per_batch": 64,
            },
            "model": {
                "embedding_dim": 8,
                "decoder_dropout": 0.0,
                "nuisance_hidden_dim": 6,
                "nuisance_heads": ["degree"],
            },
            "loss": {
                "lambda_typed": 1.0,
                "lambda_decorr": 0.01,
                "lambda_adversary": 0.1,
                "lambda_rank": 1.0,
                "lambda_hierarchy": 0.0,
            },
            "training": {
                "device": "cpu",
                "seed": 9,
                "epochs": 1,
                "rank_batch_size": 3,
                "early_stopping_patience": 1,
                "evaluation_batch_size": 4,
                "use_amp": False,
                "deterministic": True,
                "max_steps_per_epoch": 1,
                "log_interval": 1,
                "output_dir": str(output_dir),
            },
        }
    )


def test_config_rejects_hierarchy_and_unknown_nuisance(toy_train_tsv: Path) -> None:
    with pytest.raises(ValueError, match="lambda_hierarchy"):
        SITNEConfig.from_mapping(
            {
                "data": {"train_path": str(toy_train_tsv)},
                "loss": {"lambda_hierarchy": 0.1},
            }
        )
    with pytest.raises(ValueError, match="relation_hierarchy_path 必须为 null"):
        SITNEConfig.from_mapping(
            {
                "data": {
                    "train_path": str(toy_train_tsv),
                    "relation_hierarchy_path": "hierarchy.tsv",
                }
            }
        )
    with pytest.raises(ValueError, match="hierarchy_margin 必须为 0"):
        SITNEConfig.from_mapping(
            {
                "data": {"train_path": str(toy_train_tsv)},
                "loss": {"hierarchy_margin": 0.5},
            }
        )
    with pytest.raises(ValueError, match="缺少经审核"):
        SITNEConfig.from_mapping(
            {
                "data": {"train_path": str(toy_train_tsv)},
                "model": {"nuisance_heads": ["source"]},
            }
        )
    with pytest.raises(ValueError, match="必须提供 split_manifest_path"):
        SITNEConfig.from_mapping(
            {
                "data": {
                    "train_path": str(toy_train_tsv),
                    "require_manifest_hashes": True,
                }
            }
        )


def test_config_rejects_fractional_integer_fields(toy_train_tsv: Path) -> None:
    with pytest.raises(ValueError, match="training.epochs 必须为正整数"):
        SITNEConfig.from_mapping(
            {
                "data": {"train_path": str(toy_train_tsv)},
                "training": {"epochs": 1.5},
            }
        )
    with pytest.raises(ValueError, match="hits_at 必须包含正整数"):
        SITNEConfig.from_mapping(
            {
                "data": {"train_path": str(toy_train_tsv)},
                "evaluation": {"hits_at": [1.9]},
            }
        )


def test_explicit_cuda_does_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="显式请求 CUDA"):
        resolve_device("cuda")


def test_model_initialization_uses_config_seed(
    toy_train_tsv: Path,
    tmp_path: Path,
) -> None:
    config = _config(toy_train_tsv, tmp_path / "results")
    triples = load_training_triples(toy_train_tsv)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",) * triples.num_relations,
        alpha=config.walk.alpha,
        beta=config.walk.beta,
        epsilon=config.walk.epsilon,
    )
    torch.manual_seed(1)
    first, _, _ = build_model(config, triples, graph)
    torch.manual_seed(999)
    second, _, _ = build_model(config, triples, graph)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="需要 Apple MPS 环境",
)
def test_model_build_does_not_change_mps_global_rng(
    toy_train_tsv: Path,
    tmp_path: Path,
) -> None:
    config = _config(toy_train_tsv, tmp_path / "results")
    triples = load_training_triples(toy_train_tsv)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",) * triples.num_relations,
        alpha=config.walk.alpha,
        beta=config.walk.beta,
        epsilon=config.walk.epsilon,
    )
    before = torch.mps.get_rng_state().clone()
    build_model(config, triples, graph)
    after = torch.mps.get_rng_state()
    assert torch.equal(before, after)


def test_one_step_cpu_training_and_checkpoint(
    toy_train_tsv: Path,
    tmp_path: Path,
) -> None:
    config = _config(toy_train_tsv, tmp_path / "results")
    triples = load_training_triples(toy_train_tsv)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",) * triples.num_relations,
        alpha=config.walk.alpha,
        beta=config.walk.beta,
        epsilon=config.walk.epsilon,
    )
    model, labels, weights = build_model(config, triples, graph)
    trainer = SITNETrainer(config, triples, graph, model, labels, weights)
    result = trainer.fit(checkpoint_dir=tmp_path / "checkpoints")
    assert result.best_epoch == 0
    assert result.best_checkpoint is not None
    assert Path(result.best_checkpoint).is_file()
    assert len(result.history) == 1
    assert torch.isfinite(torch.tensor(result.history[0]["loss"]))
    next_epoch = trainer.load_checkpoint(result.best_checkpoint)
    assert next_epoch == 1


def test_checkpoint_replays_next_step_and_rejects_config_drift(
    toy_train_tsv: Path,
    tmp_path: Path,
) -> None:
    base = _config(toy_train_tsv, tmp_path / "results")
    config = replace(
        base,
        model=replace(base.model, decoder_dropout=0.35),
    )
    triples = load_training_triples(toy_train_tsv)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",) * triples.num_relations,
        alpha=config.walk.alpha,
        beta=config.walk.beta,
        epsilon=config.walk.epsilon,
    )
    model, labels, weights = build_model(config, triples, graph)
    trainer = SITNETrainer(config, triples, graph, model, labels, weights)
    result = trainer.fit(checkpoint_dir=tmp_path / "checkpoints")
    assert result.best_checkpoint is not None
    starts = graph.nodes_with_outgoing_edges[:4]

    trainer.load_checkpoint(result.best_checkpoint)
    first_metrics = trainer._single_step(starts)
    first_state = {
        name: tensor.detach().clone()
        for name, tensor in trainer.model.state_dict().items()
    }
    trainer.load_checkpoint(result.best_checkpoint)
    second_metrics = trainer._single_step(starts)
    second_state = trainer.model.state_dict()
    assert first_metrics == second_metrics
    assert all(
        torch.equal(first_state[name], second_state[name]) for name in first_state
    )

    changed_config = replace(
        config,
        training=replace(config.training, learning_rate=0.002),
    )
    changed_model, changed_labels, changed_weights = build_model(
        changed_config,
        triples,
        graph,
    )
    changed_trainer = SITNETrainer(
        changed_config,
        triples,
        graph,
        changed_model,
        changed_labels,
        changed_weights,
    )
    with pytest.raises(ValueError, match="config 与当前配置不一致"):
        changed_trainer.load_checkpoint(result.best_checkpoint)


def test_non_finite_formal_mode_fails_before_optimizer_step(
    toy_train_tsv: Path,
    tmp_path: Path,
) -> None:
    config = _config(toy_train_tsv, tmp_path / "results")
    config = replace(
        config,
        training=replace(config.training, non_finite_policy="fail_fast"),
    )
    triples = load_training_triples(toy_train_tsv)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",) * triples.num_relations,
        alpha=config.walk.alpha,
        beta=config.walk.beta,
        epsilon=config.walk.epsilon,
    )
    model, labels, weights = build_model(config, triples, graph)
    trainer = SITNETrainer(config, triples, graph, model, labels, weights)
    handle = trainer.model.semantic_embedding.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            trainer._single_step(graph.nodes_with_outgoing_edges[:4])
    finally:
        handle.remove()
    assert trainer.non_finite_gradient_steps == 1
    assert trainer.skipped_updates == 1
    assert len(trainer.nonfinite_events) == 1


def test_non_finite_debug_mode_skips_and_logs(
    toy_train_tsv: Path,
    tmp_path: Path,
) -> None:
    # 默认 non_finite_policy="skip_and_log"：跳过 update 但记录事件，不抛异常。
    config = _config(toy_train_tsv, tmp_path / "results")
    triples = load_training_triples(toy_train_tsv)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",) * triples.num_relations,
        alpha=config.walk.alpha,
        beta=config.walk.beta,
        epsilon=config.walk.epsilon,
    )
    model, labels, weights = build_model(config, triples, graph)
    trainer = SITNETrainer(config, triples, graph, model, labels, weights)
    handle = trainer.model.semantic_embedding.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    try:
        metrics = trainer._single_step(graph.nodes_with_outgoing_edges[:4])
    finally:
        handle.remove()
    assert math.isnan(metrics["loss"])
    assert trainer.non_finite_gradient_steps == 1
    assert trainer.skipped_updates == 1
    assert len(trainer.nonfinite_events) == 1


def test_training_history_finite_and_no_nonfinite_counts(
    toy_train_tsv: Path,
    tmp_path: Path,
) -> None:
    config = _config(toy_train_tsv, tmp_path / "results")
    config = replace(
        config,
        training=replace(
            config.training,
            epochs=2,
            max_steps_per_epoch=3,
            non_finite_policy="skip_and_log",
        ),
    )
    triples = load_training_triples(toy_train_tsv)
    graph = build_packed_csr_graph(
        triples,
        relation_modes=("symmetric",) * triples.num_relations,
        alpha=config.walk.alpha,
        beta=config.walk.beta,
        epsilon=config.walk.epsilon,
    )
    model, labels, weights = build_model(config, triples, graph)
    trainer = SITNETrainer(config, triples, graph, model, labels, weights)
    result = trainer.fit(checkpoint_dir=tmp_path / "checkpoints")
    assert len(result.history) >= 1
    for record in result.history:
        assert math.isfinite(record["loss"])
    assert trainer.non_finite_loss_steps == 0
    assert trainer.non_finite_gradient_steps == 0
    assert trainer.skipped_updates == 0
    assert trainer.nonfinite_events == []


def test_cli_writes_completed_status(toy_train_tsv: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    config = _config(toy_train_tsv, output_dir)
    assert command_train(config, tmp_path) == 0
    run_directories = list(output_dir.iterdir())
    assert len(run_directories) == 1
    assert (run_directories[0] / "completed.json").is_file()
    assert not (run_directories[0] / "failure.json").exists()
    provenance = json.loads((run_directories[0] / "provenance.json").read_text())
    assert provenance["selected_device"] == "cpu"
    implementation_files = provenance["implementation_files"]
    assert any(path.endswith("sitne_walk/trainer.py") for path in implementation_files)
    assert all(
        len(record["sha256"]) == 64 for record in implementation_files.values()
    )


def test_cli_writes_failure_status(
    toy_train_tsv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "results"
    config = _config(toy_train_tsv, output_dir)

    def fail_write(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("intentional test failure")

    monkeypatch.setattr("sitne_walk.cli._write_vocabularies", fail_write)
    with pytest.raises(RuntimeError, match="intentional test failure"):
        command_train(config, tmp_path)
    run_directories = list(output_dir.iterdir())
    assert len(run_directories) == 1
    assert (run_directories[0] / "failure.json").is_file()
    assert not (run_directories[0] / "completed.json").exists()


def test_validation_filter_includes_all_known_positive_types(
    toy_train_tsv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path = tmp_path / "validation.tsv"
    validation_path.write_text(
        "protein_i\tprotein_j\ttype_name\nA\tC\tr1\n",
        encoding="utf-8",
    )
    all_known_path = tmp_path / "all_known.tsv"
    all_known_path.write_text(
        "protein_i\tprotein_j\ttype_name\nA\tC\tr1\nA\tC\tr2\n",
        encoding="utf-8",
    )
    config = _config(toy_train_tsv, tmp_path / "results")
    config = replace(
        config,
        data=replace(
            config.data,
            validation_path=str(validation_path),
            all_known_positive_paths=(str(all_known_path),),
        ),
    )
    observed: dict[str, int] = {}

    def inspect_fit(
        trainer: SITNETrainer,
        validation=None,
        validation_filter=None,
        checkpoint_dir=None,
    ) -> FitResult:
        del checkpoint_dir
        assert validation is not None and validation_filter is not None
        head = trainer.triples.protein_to_id["A"]
        tail = trainer.triples.protein_to_id["C"]
        mask, matched = validation_filter.lookup(
            torch.tensor([head]), torch.tensor([tail])
        )
        assert matched.item()
        observed["positive_types"] = int(mask.sum().item())
        return FitResult(
            history=({"loss": 0.0},),
            best_epoch=0,
            best_selection_metric=0.5,
            best_checkpoint=None,
        )

    monkeypatch.setattr(SITNETrainer, "fit", inspect_fit)
    assert command_train(config, tmp_path) == 0
    assert observed["positive_types"] == 2


def test_all_known_filter_missing_validation_target_fails_preflight(
    toy_train_tsv: Path,
    tmp_path: Path,
) -> None:
    validation_path = tmp_path / "validation.tsv"
    validation_path.write_text(
        "protein_i\tprotein_j\ttype_name\nA\tC\tr1\n",
        encoding="utf-8",
    )
    incomplete_known_path = tmp_path / "incomplete_known.tsv"
    incomplete_known_path.write_text(
        "protein_i\tprotein_j\ttype_name\nA\tC\tr2\n",
        encoding="utf-8",
    )
    config = _config(toy_train_tsv, tmp_path / "results")
    config = replace(
        config,
        data=replace(
            config.data,
            validation_path=str(validation_path),
            all_known_positive_paths=(str(incomplete_known_path),),
        ),
    )
    with pytest.raises(ValueError, match="missing_targets=1"):
        command_train(config, tmp_path)
    assert not (tmp_path / "results").exists()
