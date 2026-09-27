"""Dataset B/B* 的 phase-aware 默认拒绝文件访问层。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path, PurePosixPath
from typing import BinaryIO, TextIO


class Phase(str, Enum):
    INSPECT = "inspect"
    TRAIN = "train"
    SELECT = "select"
    TEST = "test"
    VERIFY_RUN = "verify-run"


B_SPLITS = {
    "train": "splits/train.tsv",
    "validation": "splits/validation.tsv",
    "test": "splits/test.tsv",
}
B_CONTROL = {
    "manifest/dataset.json",
    "manifest/split_manifest.tsv",
    "manifest/asset_manifest.tsv",
    "reports/validation.json",
}
B_SEQUENCES = {"sequences/proteins.tsv", "sequences/proteins.fasta"}

BSTAR_TRAIN_ALLOWLIST = {
    "views/X/train.tsv",
    "views/X_type/model_input_event_membership.tsv",
    "edges/train_pair_type_fine.tsv",
    "edges/train_pair_type_coarse.tsv",
    "nodes/train_protein.tsv",
    "nodes/train_proteins.fasta",
    "nodes/train_type_registry.tsv",
}
BSTAR_CONTROL = {
    "DATA_CARD.md",
    "loader_contract.json",
    "build_manifest.json",
    "views/train_input_manifest.json",
    "reports/train_input_isolation_audit.json",
    "reports/leakage_audit.json",
}
BSTAR_TRAIN_CONTROL = {
    "loader_contract.json",
    "views/train_input_manifest.json",
}
BSTAR_HELDOUT_VIEWS = {
    "views/X/validation.tsv",
    "views/X/test.tsv",
    "views/X/pairs.tsv",
}

_CAPABILITY_ISSUER = object()


class PostSelectionCapability:
    """由 runner 在完整验证 sealed selection 后签发的进程内 capability。

    这是一道防止误用/阶段越权的工程门，不是针对恶意 Python 进程的安全沙箱。
    构造器要求模块私有 issuer；调用方应使用 runner 的验证入口获取 guard。
    """

    __slots__ = ("run_dir", "selection_manifest_sha256")

    def __init__(self, run_dir: str, selection_manifest_sha256: str, issuer: object):
        if issuer is not _CAPABILITY_ISSUER:
            raise PermissionError("PostSelectionCapability 只能由已验证 runner 签发")
        if len(selection_manifest_sha256) != 64:
            raise ValueError("selection manifest SHA-256 非法")
        self.run_dir = run_dir
        self.selection_manifest_sha256 = selection_manifest_sha256


def _issue_post_selection_capability(
    run_dir: str | Path, selection_manifest_sha256: str
) -> PostSelectionCapability:
    return PostSelectionCapability(
        str(Path(run_dir).resolve()), selection_manifest_sha256, _CAPABILITY_ISSUER
    )


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class OpenedPath:
    dataset: str
    relative_path: str
    absolute_path: str
    bytes: int
    sha256: str


class GuardedDatasetIO:
    """只允许当前 phase 明确列出的只读路径，并记录实际打开的资产。"""

    def __init__(
        self,
        dataset_b_root: str | Path,
        dataset_bstar_root: str | Path | None,
        phase: Phase | str,
        post_selection_capability: PostSelectionCapability | None = None,
    ) -> None:
        self.dataset_b_root = Path(dataset_b_root).resolve()
        self.dataset_bstar_root = (
            Path(dataset_bstar_root).resolve() if dataset_bstar_root else None
        )
        self.phase = Phase(phase)
        if post_selection_capability is not None and self.phase not in {
            Phase.TEST,
            Phase.VERIFY_RUN,
        }:
            raise PermissionError(
                "post-selection capability 只能绑定 TEST/VERIFY_RUN guard"
            )
        self.post_selection_capability = post_selection_capability
        self._opened: dict[tuple[str, str], OpenedPath] = {}

    @staticmethod
    def _normalize(relative_path: str) -> str:
        pure = PurePosixPath(relative_path.replace("\\", "/"))
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise PermissionError(f"拒绝非规范相对路径: {relative_path!r}")
        return pure.as_posix()

    def _allowed_b(self, rel: str) -> bool:
        if self.phase == Phase.INSPECT:
            return rel in B_CONTROL | B_SEQUENCES | set(B_SPLITS.values())
        if self.phase == Phase.TRAIN:
            # 所有训练模式的 pair/sequence 都从 B* 七项 allow-list 读取。
            # B 侧只开放身份/许可控制面和冻结 Train split（当前 runner 不需
            # 打开后者），全量序列表在 TRAIN 无条件拒绝。
            return rel in B_CONTROL | {B_SPLITS["train"]}
        if self.phase == Phase.SELECT:
            return rel in B_CONTROL | B_SEQUENCES | {B_SPLITS["validation"]}
        if self.phase == Phase.TEST:
            return rel in B_CONTROL | B_SEQUENCES | {B_SPLITS["test"]}
        if self.phase == Phase.VERIFY_RUN:
            return rel in B_CONTROL | {B_SPLITS["test"]}
        return False

    def _allowed_bstar(self, rel: str) -> bool:
        if self.phase == Phase.INSPECT:
            # inspect 只核验训练投影及控制面；held-out targets/views 仍保持锁定。
            # 完整镜像的逐文件完整性由服务器 bundle manifest 独立负责。
            return rel in BSTAR_TRAIN_ALLOWLIST | BSTAR_CONTROL
        if self.phase == Phase.TRAIN:
            # Train 数据面严格限制为七项 allow-list；额外只开放验证该
            # allow-list 所必需的两份控制面合同，不开放报告或 held-out 资产。
            return rel in BSTAR_TRAIN_ALLOWLIST | BSTAR_TRAIN_CONTROL
        if self.phase in {Phase.TEST, Phase.VERIFY_RUN}:
            if self.post_selection_capability is not None:
                return (
                    rel in BSTAR_CONTROL
                    or rel in BSTAR_HELDOUT_VIEWS
                    or rel.startswith("targets/")
                )
            return rel in BSTAR_CONTROL
        # SELECT 没有 post-selection capability；B* held-out targets/views 保持锁定。
        return rel in BSTAR_CONTROL

    def _resolve(self, dataset: str, relative_path: str) -> tuple[str, Path]:
        rel = self._normalize(relative_path)
        if dataset == "b":
            root = self.dataset_b_root
            allowed = self._allowed_b(rel)
        elif dataset == "bstar":
            if self.dataset_bstar_root is None:
                raise FileNotFoundError("未配置 Dataset B* 根目录")
            root = self.dataset_bstar_root
            allowed = self._allowed_bstar(rel)
        else:
            raise ValueError(f"未知 dataset: {dataset}")
        if not allowed:
            raise PermissionError(
                f"phase={self.phase.value} 默认拒绝 {dataset}:{rel}；"
                "held-out/非 allow-list 资产不得被打开"
            )
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"路径逃逸数据根目录: {relative_path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return rel, path

    def _record(self, dataset: str, rel: str, path: Path) -> None:
        key = (dataset, rel)
        if key not in self._opened:
            self._opened[key] = OpenedPath(
                dataset=dataset,
                relative_path=rel,
                absolute_path=str(path),
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )

    def open_text(
        self, dataset: str, relative_path: str, *, newline: str | None = ""
    ) -> TextIO:
        rel, path = self._resolve(dataset, relative_path)
        self._record(dataset, rel, path)
        return path.open("r", encoding="utf-8", newline=newline)

    def open_binary(self, dataset: str, relative_path: str) -> BinaryIO:
        rel, path = self._resolve(dataset, relative_path)
        self._record(dataset, rel, path)
        return path.open("rb")

    def read_text(self, dataset: str, relative_path: str) -> str:
        with self.open_text(dataset, relative_path, newline=None) as handle:
            return handle.read()

    @property
    def opened_paths(self) -> list[dict[str, object]]:
        return [
            {
                "dataset": item.dataset,
                "relative_path": item.relative_path,
                "absolute_path": item.absolute_path,
                "bytes": item.bytes,
                "sha256": item.sha256,
            }
            for item in sorted(
                self._opened.values(), key=lambda value: (value.dataset, value.relative_path)
            )
        ]

    @property
    def capability_record(self) -> dict[str, str] | None:
        capability = self.post_selection_capability
        if capability is None:
            return None
        return {
            "scope": "BSTAR_POST_SELECTION_TARGETS_AND_HELDOUT_VIEWS",
            "run_dir": capability.run_dir,
            "selection_manifest_sha256": capability.selection_manifest_sha256,
        }
