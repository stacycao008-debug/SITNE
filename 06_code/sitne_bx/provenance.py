"""运行环境、实现文件与输入资产 provenance。"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import torch
import numpy as np
import yaml

from .artifacts import object_sha256
from .guarded_io import sha256_file


def collect_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        result["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
        result["cudnn_version"] = torch.backends.cudnn.version()
    return result


def collect_implementation_hashes(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    package = Path(__file__).resolve().parent
    candidates = sorted(package.glob("*.py"))
    launcher = package.parent / "run_sitne_bx.py"
    if launcher.is_file():
        candidates.append(launcher)
    result: dict[str, str] = {}
    for path in sorted(candidates):
        resolved = path.resolve()
        try:
            key = resolved.relative_to(root).as_posix()
        except ValueError:
            key = str(resolved)
        result[key] = sha256_file(resolved)
    return result


def collect_provenance(
    project_root: str | Path,
    phase: str,
    config: Mapping[str, Any],
    opened_paths: list[Mapping[str, object]],
    resolved_device: str,
) -> dict[str, Any]:
    return {
        "schema_version": "sitne-bx-provenance-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "working_directory": os.getcwd(),
        "project_root": str(Path(project_root).resolve()),
        "resolved_device": resolved_device,
        "config_sha256": object_sha256(config),
        "config": dict(config),
        "environment": collect_environment(),
        "implementation_hashes": collect_implementation_hashes(project_root),
        "opened_input_paths": [dict(item) for item in opened_paths],
        "scientific_evidence": False,
        "evidence_boundary": (
            "单次工程运行及其指标不是论文结论；正式科学主张仍需预注册协议、"
            "多随机种子、基线/消融、统计检验和独立复核。"
        ),
    }
