"""不可覆盖 artifact、hash 与 sealed manifest 辅助函数。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import uuid

import torch

from .guarded_io import sha256_file


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def seal_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "seal_sha256" in payload:
        raise ValueError("待 seal payload 不得预先包含 seal_sha256")
    result = dict(payload)
    result["seal_sha256"] = object_sha256(result)
    return result


def verify_sealed_payload(payload: Mapping[str, Any]) -> None:
    expected = payload.get("seal_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("manifest 缺少合法 seal_sha256")
    unsealed = dict(payload)
    del unsealed["seal_sha256"]
    actual = object_sha256(unsealed)
    if actual != expected:
        raise ValueError(f"manifest seal 不匹配: expected={expected}, actual={actual}")


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点不是 object: {path}")
    return value


def write_json_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_tsv_new(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def torch_save_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        torch.save(dict(payload), handle)


def make_run_id(seed: int) -> str:
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"sitne_bx_{timestamp}_seed{seed}_{uuid.uuid4().hex[:8]}"


def create_run_directory(base: str | Path, run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id 必须是单个安全路径段")
    output = Path(base) / run_id
    output.mkdir(parents=True, exist_ok=False)
    return output


def require_file_hash(path: str | Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"文件 hash 不匹配: {path}: expected={expected_sha256}, actual={actual}")
