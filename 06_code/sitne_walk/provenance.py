"""SITNE-Walk 运行环境、输入 hash 与不可覆盖输出辅助函数。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping
import uuid

import torch


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """计算实现文件 hash，确保未提交代码也能被精确追踪。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment() -> dict[str, Any]:
    """收集复现实验所需的软件和加速器信息。"""

    environment: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "mps_built": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        ),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    if torch.cuda.is_available():
        devices = []
        for device_id in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(device_id)
            devices.append(
                {
                    "id": device_id,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": properties.total_memory,
                }
            )
        environment["cuda_devices"] = devices
        environment["cudnn_version"] = torch.backends.cudnn.version()
    return environment


def current_git_commit(project_root: str | Path) -> str | None:
    """只读获取 Git commit；不在 Git 仓库时返回 None。"""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def git_worktree_dirty(project_root: str | Path) -> bool | None:
    """只读判断 worktree 是否有 tracked/untracked 改动。"""

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def collect_implementation_hashes(project_root: str | Path) -> dict[str, dict[str, str]]:
    """记录本次实际执行的 launcher 与包内 Python 源码 SHA-256。"""

    root = Path(project_root).resolve()
    package_dir = Path(__file__).resolve().parent
    candidates = sorted(package_dir.glob("*.py"))
    launcher = package_dir.parent / "run_sitne_walk.py"
    if launcher.is_file():
        candidates.append(launcher)

    records: dict[str, dict[str, str]] = {}
    for path in sorted(candidates):
        resolved = path.resolve()
        try:
            key = str(resolved.relative_to(root))
        except ValueError:
            key = str(resolved)
        records[key] = {
            "path": str(resolved),
            "sha256": _sha256_file(resolved),
        }
    return records


def make_run_id(prefix: str, seed: int) -> str:
    """生成不依赖模糊 ``latest/final`` 命名的唯一 run ID。"""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_seed{seed}_{suffix}"


def create_new_run_directory(base_dir: str | Path, run_id: str) -> Path:
    """创建全新目录，若名称已存在则拒绝覆盖。"""

    output = Path(base_dir) / run_id
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_json_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    """以 exclusive-create 方式写 JSON，绝不覆盖已有证据文件。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def collect_runtime_provenance(
    project_root: str | Path,
    config: Mapping[str, Any],
    input_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """组装一次运行的最小 provenance 记录。"""

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": os.getcwd(),
        "project_root": str(Path(project_root).resolve()),
        "git_commit": current_git_commit(project_root),
        "git_worktree_dirty": git_worktree_dirty(project_root),
        "environment": collect_environment(),
        "implementation_files": collect_implementation_hashes(project_root),
        "config": dict(config),
        "inputs": {name: dict(value) for name, value in input_files.items()},
    }


def guard_historical_output(path: str | Path) -> None:
    """防止任何重跑脚本写入历史正式结果目录（READ_ONLY）。

    历史目录 `08_results/sitne_walk_paper_v2/` 只读保留；所有新产物必须写到
    `rerun_submission_audit/` 下，避免误覆盖历史结果。
    """

    historical_root = (
        Path(__file__).resolve().parents[2] / "08_results" / "sitne_walk_paper_v2"
    ).resolve()
    resolved = Path(path).resolve()
    if str(resolved) == str(historical_root) or str(resolved).startswith(
        str(historical_root) + os.sep
    ):
        raise RuntimeError(
            f"拒绝写入历史结果目录 {historical_root}: {resolved}。"
            f"请改用 rerun_submission_audit/ 下的新路径。"
        )
