#!/usr/bin/env python3
"""生成 Handoff.zip —— 代码侧增量交付包（CODE-FINAL）。

打包三类增量，排除可再生的 .pt checkpoint 与 __pycache__：
  A. rerun_submission_audit/           审计结果包
  B. scripts/rerun_audit/              新增重跑脚本
  C. 06_code/ 下本轮修改的代码文件     数值可靠性修复

用法：
    python3 scripts/rerun_audit/make_handoff_zip.py [--project-root PATH] [--output PATH]

产出：
    Handoff.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# 本轮修改的代码文件（数值可靠性修复 + provenance 调整）
MODIFIED_CODE = [
    "06_code/sitne_walk/trainer.py",
    "06_code/sitne_walk/config.py",
    "06_code/sitne_walk/cli.py",
    "06_code/sitne_walk/optuna_search.py",
    "06_code/sitne_walk/provenance.py",
    "06_code/tests/sitne_walk/test_config_trainer.py",
]

# 顶层包含项（目录会被递归打包）
TOP_DIRS = [
    "rerun_submission_audit",
    "scripts/rerun_audit",
]

# zip 根的说明文件
MANIFEST = "HANDOFF_ZIP_MANIFEST.md"


def should_exclude(path: Path) -> bool:
    """排除可再生的 checkpoint 与编译缓存。"""
    if path.suffix == ".pt":
        return True
    if "__pycache__" in path.parts:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = (args.output or root / "Handoff.zip").resolve()

    files: list[Path] = []
    for rel in [MANIFEST, *MODIFIED_CODE]:
        p = root / rel
        if p.is_file():
            files.append(p)
        else:
            print(f"[skip] missing: {rel}")
    for rel in TOP_DIRS:
        d = root / rel
        if d.is_dir():
            files.extend(
                f
                for f in sorted(d.rglob("*"))
                if f.is_file() and not should_exclude(f)
            )
        else:
            print(f"[skip] missing dir: {rel}")

    count = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = f.relative_to(root).as_posix()
            zf.write(f, arcname)
            count += 1

    size_mb = output.stat().st_size / 1e6
    print(f"wrote {output} ({size_mb:.1f} MB, {count} files)")


if __name__ == "__main__":
    main()
