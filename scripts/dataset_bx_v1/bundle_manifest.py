#!/usr/bin/env python3
"""Build, verify, and package a stable SITNE-Walk-BX server bundle.

The ZIP writer only archives regular files selected by the policy below.  It
does not copy filesystem extended attributes, and it gives every archive entry
a fixed timestamp so that unchanged inputs yield stable ZIP content.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
from typing import Iterable, Sequence
import zipfile


DEFAULT_MANIFEST = (
    "10_reproducibility/dataset_bx_v1/BUNDLE_SHA256SUMS.txt"
)
CHUNK_SIZE = 4 * 1024 * 1024

# These are the complete, versioned exclusion rules for bundle v1.  A file is
# excluded when any directory component matches one of these names.
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".ipynb_checkpoints",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
    }
)
EXCLUDED_FILE_NAMES = frozenset({".coverage", ".DS_Store"})


@dataclass(frozen=True)
class FileRecord:
    relative_path: str
    absolute_path: Path
    size: int
    mode: int


@dataclass(frozen=True)
class ScanResult:
    included: tuple[FileRecord, ...]
    excluded_count: int
    excluded_bytes: int
    excluded_reasons: dict[str, int]


class BundleError(RuntimeError):
    """Expected, user-actionable bundle validation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_relative_path(raw: str, label: str) -> str:
    if not raw or "\n" in raw or "\r" in raw or "\0" in raw:
        raise BundleError(f"{label} is empty or contains a forbidden character")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts or raw.startswith("./"):
        raise BundleError(f"{label} must be a normalized relative POSIX path: {raw!r}")
    normalized = candidate.as_posix()
    if normalized != raw:
        raise BundleError(f"{label} is not normalized: {raw!r}")
    return normalized


def exclusion_reason(relative_path: str, manifest_relative: str) -> str | None:
    path = PurePosixPath(relative_path)
    if relative_path == manifest_relative:
        return "manifest_self"
    if path.parts and path.parts[0] == "08_results":
        return "dynamic_results"
    for component in path.parts[:-1]:
        if component in EXCLUDED_DIRECTORY_NAMES:
            return f"directory:{component}"
        if component.startswith(".venv."):
            return "directory:.venv.variant"
    name = path.name
    if name in EXCLUDED_FILE_NAMES:
        return f"file:{name}"
    if name.startswith("._"):
        return "macos_appledouble"
    if name.endswith((".pyc", ".pyo")):
        return "python_bytecode"
    return None


def scan_tree(root: Path, manifest_relative: str = DEFAULT_MANIFEST) -> ScanResult:
    root = root.resolve()
    if not root.is_dir():
        raise BundleError(f"bundle root is not a directory: {root}")
    manifest_relative = _validate_relative_path(
        manifest_relative, "manifest relative path"
    )

    included: list[FileRecord] = []
    excluded_reasons: Counter[str] = Counter()
    excluded_count = 0
    excluded_bytes = 0

    # Path.rglob does not descend through a directory symlink.  Symlinks outside
    # an excluded tree are explicitly omitted so an archive can never point
    # outside the project root.
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        reason = exclusion_reason(relative, manifest_relative)
        try:
            metadata = path.lstat()
        except OSError as error:
            raise BundleError(f"cannot stat {relative}: {error}") from error

        if stat.S_ISDIR(metadata.st_mode):
            continue
        if reason is None and stat.S_ISLNK(metadata.st_mode):
            reason = "symlink"
        if reason is None and not stat.S_ISREG(metadata.st_mode):
            reason = "non_regular_file"

        if reason is not None:
            excluded_count += 1
            excluded_bytes += int(metadata.st_size)
            excluded_reasons[reason] += 1
            continue

        _validate_relative_path(relative, "file path")
        included.append(
            FileRecord(
                relative_path=relative,
                absolute_path=path,
                size=int(metadata.st_size),
                mode=stat.S_IMODE(metadata.st_mode),
            )
        )

    included.sort(key=lambda record: record.relative_path.encode("utf-8"))
    return ScanResult(
        included=tuple(included),
        excluded_count=excluded_count,
        excluded_bytes=excluded_bytes,
        excluded_reasons=dict(sorted(excluded_reasons.items())),
    )


def hash_records(records: Iterable[FileRecord]) -> dict[str, str]:
    return {
        record.relative_path: sha256_file(record.absolute_path)
        for record in records
    }


def render_manifest(hashes: dict[str, str]) -> bytes:
    lines = [
        "# SITNE-Walk-BX CUDA server bundle SHA-256 manifest v1",
        "# Format: <sha256><two spaces><relative-posix-path>",
    ]
    for relative_path in sorted(hashes, key=lambda value: value.encode("utf-8")):
        lines.append(f"{hashes[relative_path]}  {relative_path}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_manifest(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BundleError(f"cannot read manifest {path}: {error}") from error

    expected: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        if "  " not in line:
            raise BundleError(f"malformed manifest line {line_number}")
        digest, relative_path = line.split("  ", 1)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise BundleError(f"invalid SHA-256 at manifest line {line_number}")
        relative_path = _validate_relative_path(
            relative_path, f"manifest line {line_number} path"
        )
        if relative_path in expected:
            raise BundleError(f"duplicate manifest path: {relative_path}")
        expected[relative_path] = digest
    if not expected:
        raise BundleError("manifest contains no file records")
    return expected


def summary_payload(
    scan: ScanResult, manifest_relative: str = DEFAULT_MANIFEST
) -> dict[str, object]:
    return {
        "policy_version": 1,
        "included_files": len(scan.included),
        "included_bytes": sum(record.size for record in scan.included),
        "excluded_files": scan.excluded_count,
        "excluded_bytes": scan.excluded_bytes,
        "excluded_reasons": scan.excluded_reasons,
        "manifest_relative_path": manifest_relative,
    }


def print_summary(
    scan: ScanResult,
    *,
    dry_run: bool,
    manifest_relative: str = DEFAULT_MANIFEST,
) -> None:
    payload = summary_payload(scan, manifest_relative)
    payload["dry_run"] = dry_run
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def generate_manifest(root: Path, manifest_relative: str) -> None:
    manifest_relative = _validate_relative_path(
        manifest_relative, "manifest relative path"
    )
    destination = root.resolve() / manifest_relative
    if destination.exists() or destination.is_symlink():
        raise BundleError(f"refusing to overwrite existing manifest: {destination}")
    if not destination.parent.is_dir():
        raise BundleError(f"manifest parent directory does not exist: {destination.parent}")

    scan = scan_tree(root, manifest_relative)
    content = render_manifest(hash_records(scan.included))
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError as error:
        raise BundleError(f"refusing to overwrite existing manifest: {destination}") from error
    print(f"WROTE_MANIFEST={destination}")
    print_summary(scan, dry_run=False, manifest_relative=manifest_relative)


def verify_manifest(root: Path, manifest_relative: str) -> None:
    root = root.resolve()
    manifest_relative = _validate_relative_path(
        manifest_relative, "manifest relative path"
    )
    manifest_path = root / manifest_relative
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise BundleError(f"regular manifest file is missing: {manifest_path}")
    expected = parse_manifest(manifest_path)
    scan = scan_tree(root, manifest_relative)
    actual_paths = {record.relative_path for record in scan.included}
    expected_paths = set(expected)
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing or unexpected:
        details = {
            "missing": missing[:20],
            "missing_count": len(missing),
            "unexpected": unexpected[:20],
            "unexpected_count": len(unexpected),
        }
        raise BundleError(f"manifest file-set mismatch: {json.dumps(details)}")

    mismatches: list[str] = []
    for record in scan.included:
        actual_digest = sha256_file(record.absolute_path)
        if actual_digest != expected[record.relative_path]:
            mismatches.append(record.relative_path)
    if mismatches:
        raise BundleError(
            f"SHA-256 mismatch for {len(mismatches)} file(s): {mismatches[:20]}"
        )
    print(f"BUNDLE_MANIFEST_VERIFIED files={len(expected)} root={root}")


def _zip_info(archive_path: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(archive_path, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | (mode & 0o777)) << 16
    return info


def package_bundle(
    root: Path,
    output: Path,
    manifest_relative: str,
    archive_prefix: str | None,
) -> None:
    root = root.resolve()
    output = output.expanduser().resolve()
    manifest_relative = _validate_relative_path(
        manifest_relative, "manifest relative path"
    )
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise BundleError("ZIP output must be outside the bundle root")
    if output.suffix.lower() != ".zip":
        raise BundleError("ZIP output must end in .zip")
    if not output.parent.is_dir():
        raise BundleError(f"ZIP output parent does not exist: {output.parent}")
    checksum_path = output.with_name(output.name + ".sha256")
    for destination in (output, checksum_path):
        if destination.exists() or destination.is_symlink():
            raise BundleError(f"refusing to overwrite existing output: {destination}")

    prefix = archive_prefix or output.stem
    prefix = _validate_relative_path(prefix, "archive prefix")
    if len(PurePosixPath(prefix).parts) != 1:
        raise BundleError("archive prefix must be one directory name")

    scan = scan_tree(root, manifest_relative)
    hashes = hash_records(scan.included)
    manifest_content = render_manifest(hashes)

    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()
    linked_output = False
    checksum_created = False
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for record in scan.included:
                archive_name = f"{prefix}/{record.relative_path}"
                digest = hashlib.sha256()
                info = _zip_info(archive_name, record.mode)
                with record.absolute_path.open("rb") as source, archive.open(
                    info, mode="w", force_zip64=True
                ) as destination:
                    for block in iter(lambda: source.read(CHUNK_SIZE), b""):
                        digest.update(block)
                        destination.write(block)
                if digest.hexdigest() != hashes[record.relative_path]:
                    raise BundleError(
                        "source changed while packaging: " + record.relative_path
                    )

            manifest_name = f"{prefix}/{manifest_relative}"
            archive.writestr(_zip_info(manifest_name, 0o644), manifest_content)

        zip_digest = sha256_file(temporary_path)
        # A hard link provides create-if-absent semantics and therefore cannot
        # overwrite a file created concurrently after the checks above.
        try:
            os.link(temporary_path, output)
        except FileExistsError as error:
            raise BundleError(f"refusing to overwrite existing output: {output}") from error
        linked_output = True
        try:
            with checksum_path.open("x", encoding="utf-8", newline="\n") as handle:
                checksum_created = True
                handle.write(f"{zip_digest}  {output.name}\n")
        except FileExistsError as error:
            raise BundleError(
                f"refusing to overwrite existing output: {checksum_path}"
            ) from error
    except Exception:
        if checksum_created:
            checksum_path.unlink(missing_ok=True)
        if linked_output:
            output.unlink(missing_ok=True)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"WROTE_ZIP={output}")
    print(f"WROTE_ZIP_SHA256={checksum_path}")
    print(f"ZIP_SHA256={zip_digest}")
    print_summary(scan, dry_run=False, manifest_relative=manifest_relative)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="show include/exclude statistics")
    scan_parser.add_argument("--root", type=Path, required=True)
    scan_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)

    generate_parser = subparsers.add_parser(
        "generate", help="write a new manifest and refuse overwrite"
    )
    generate_parser.add_argument("--root", type=Path, required=True)
    generate_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)

    verify_parser = subparsers.add_parser("verify", help="verify all manifest records")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)

    package_parser = subparsers.add_parser(
        "package", help="create a deterministic ZIP plus external ZIP checksum"
    )
    package_parser.add_argument("--root", type=Path, required=True)
    package_parser.add_argument("--output", type=Path, required=True)
    package_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    package_parser.add_argument("--archive-prefix")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scan":
            print_summary(
                scan_tree(args.root, args.manifest),
                dry_run=True,
                manifest_relative=args.manifest,
            )
        elif args.command == "generate":
            generate_manifest(args.root, args.manifest)
        elif args.command == "verify":
            verify_manifest(args.root, args.manifest)
        elif args.command == "package":
            package_bundle(
                args.root,
                args.output,
                args.manifest,
                args.archive_prefix,
            )
        else:  # pragma: no cover - argparse constrains this value.
            raise AssertionError(args.command)
    except BundleError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
