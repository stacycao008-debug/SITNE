from __future__ import annotations

import json

import pytest

from sitne_bx.config import load_config
from sitne_bx.guarded_io import Phase
from sitne_bx.runner import (
    create_post_selection_guard,
    select,
    test as run_test,
    train,
    verify_run,
)


def test_cpu_synthetic_end_to_end_state_machine(synthetic_project):
    root, config_path = synthetic_project
    config = load_config(config_path)
    run_dir = train(config, root, run_id="synthetic_state_machine")
    assert (run_dir / "train_complete.json").is_file()
    provenance = json.loads((run_dir / "provenance_train.json").read_text())
    graph = provenance["train_positive_topology_graph"]
    assert graph["nodes"] == 4
    assert graph["positive_edges"] == 3
    assert graph["negative_edges_used"] == 0
    opened = {item["relative_path"] for item in provenance["opened_input_paths"]}
    assert "splits/validation.tsv" not in opened
    assert "splits/test.tsv" not in opened
    assert not any(path.startswith("targets/") for path in opened)

    with pytest.raises(PermissionError):
        run_test(config, root, run_dir)

    selection_path = select(config, root, run_dir)
    assert selection_path.is_file()
    post_selection_guard, selection = create_post_selection_guard(
        config, root, run_dir, Phase.TEST
    )
    assert selection["training_state_frozen"] is True
    with post_selection_guard.open_text(
        "bstar", "targets/should_never_open.tsv"
    ) as handle:
        assert handle.read().strip() == "secret"
    assert post_selection_guard.capability_record is not None
    test_metrics = run_test(config, root, run_dir)
    assert test_metrics.is_file()
    report = verify_run(config, root, run_dir)
    assert report["status"] == "PASS"
    assert report["test_rows"] == 4
    assert report["prediction_metrics_recomputed"] is True
    test_split_path = root / "02_data_canonical/dataset_B/v1/splits/test.tsv"
    test_split_path.write_text(
        test_split_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Test provenance input"):
        verify_run(config, root, run_dir)


def test_cuda_config_never_falls_back():
    # CUDA 主机上该分支会成功，因此这里只覆盖配置合同；本机 CUDA 状态不作为断言。
    from sitne_bx.config import BXConfig

    with pytest.raises(ValueError):
        BXConfig.from_mapping(
            {
                "data": {"dataset_bstar_root": "unused-bstar", "typed_level": "none"},
                "loss": {"typed_weight": 0.0},
                "training": {"device": "cpu", "allow_cpu_for_testing": False, "use_amp": False},
            }
        )
