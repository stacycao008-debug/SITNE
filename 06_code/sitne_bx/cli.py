"""SITNE-Walk-BX phase CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifacts import write_json_new
from .config import load_config
from .inspection import inspect_datasets
from .runner import select, test, train, verify_run


def _default_project_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True, help="SITNE-BX YAML 配置")
    parser.add_argument(
        "--project-root", default=_default_project_root(), help="服务器包根目录"
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_sitne_bx.py",
        description="Dataset B/B* 归纳式 CUDA pipeline（禁止设备 fallback）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()

    inspect_parser = subparsers.add_parser(
        "inspect", parents=[common], help="核验 UID/hash/count/sequence/B* 合同"
    )
    inspect_parser.add_argument(
        "--output", default=None, help="可选：以 exclusive-create 写 inspection JSON"
    )

    train_parser = subparsers.add_parser(
        "train", parents=[common], help="仅打开 Train 与 B* 七项 allow-list"
    )
    train_parser.add_argument("--run-id", default=None, help="可选安全 run ID")

    for command, help_text in (
        ("select", "仅打开 Validation，选择 checkpoint 和 MCC 阈值"),
        ("test", "验证 sealed selection 后才打开 Test"),
        ("verify-run", "只读核验 Test predictions/metrics/provenance"),
    ):
        command_parser = subparsers.add_parser(command, parents=[common], help=help_text)
        command_parser.add_argument("--run-dir", required=True, help="唯一 run 输出目录")
    return parser


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    root = Path(args.project_root).resolve()
    if args.command == "inspect":
        report = inspect_datasets(config, root)
        if args.output:
            output = Path(args.output)
            if not output.is_absolute():
                output = root / output
            write_json_new(output, report)
        _print(report)
        return 0
    if args.command == "train":
        run_dir = train(config, root, run_id=args.run_id)
        _print({"status": "PASS", "phase": "train", "run_dir": str(run_dir)})
        return 0
    if args.command == "select":
        manifest = select(config, root, args.run_dir)
        _print(
            {"status": "PASS", "phase": "select", "selection_manifest": str(manifest)}
        )
        return 0
    if args.command == "test":
        metrics = test(config, root, args.run_dir)
        _print({"status": "PASS", "phase": "test", "test_metrics": str(metrics)})
        return 0
    if args.command == "verify-run":
        _print(verify_run(config, root, args.run_dir))
        return 0
    raise AssertionError(f"未处理 command: {args.command}")
