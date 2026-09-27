#!/usr/bin/env python3
"""formal_v1 configuration reconciliation：从 15 个 archived runs 反查并冻结。

流程（A3 缺口修复）：
  1. 读 15 个 archived resolved_config.json；
  2. 规范化（排除 fold/seed/output 特定字段）；
  3. 逐字段比较一致性，输出 formal_config_reconciliation.tsv；
  4. 冻结 formal_v1_archived.yaml（历史真值：float16、无 non_finite_policy）。

用法：
    PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/run_reconcile_config.py --project-root "$PWD"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ARCHIVED_GLOB = "08_results/sitne_walk_paper_v2/r2_main_performance/optimal/fold_*/seed_*/sitne_walk_*/resolved_config.json"
# 这些字段是 fold/seed/output 特定，规范化时替换为 placeholder，不参与一致性比较。
NORMALIZE_RULES = [
    (("data", "train_path"), "FOLD_TRAIN_PLACEHOLDER"),
    (("data", "validation_path"), "FOLD_VAL_PLACEHOLDER"),
    (("data", "test_path"), "FOLD_TEST_PLACEHOLDER"),
    (("training", "seed"), "SEED_PLACEHOLDER"),
    (("training", "output_dir"), "OUTPUT_DIR_PLACEHOLDER"),
]


def _flatten(cfg: dict, prefix: tuple = ()) -> dict[tuple, object]:
    out = {}
    for k, v in cfg.items():
        path = prefix + (k,)
        if isinstance(v, dict):
            out.update(_flatten(v, path))
        else:
            out[path] = v
    return out


def _normalize(path: tuple, value: object) -> tuple[tuple, object]:
    for rule_path, placeholder in NORMALIZE_RULES:
        if path == rule_path:
            return path, placeholder
    return path, value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    files = sorted(root.glob(ARCHIVED_GLOB))
    if len(files) != 15:
        print(f"[warn] 期望 15 个 archived config，实际 {len(files)} 个")

    # 收集每个 run 的规范化字段
    per_run: list[tuple[str, dict]] = []
    field_values: dict[tuple, set] = {}
    for f in files:
        cfg = json.loads(f.read_text())
        flat = _flatten(cfg)
        norm = {}
        for path, value in flat.items():
            npath, nval = _normalize(path, value)
            norm[".".join(npath)] = nval
            field_values.setdefault(npath, set()).add(repr(nval))
        per_run.append((str(f), norm))

    # 输出 reconciliation：每个字段在 15 runs 中的值集合 + 是否一致
    out_dir = root / "rerun_submission_audit" / "02_formal_sitne"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(field_values, key=".".join):
        vals = field_values[path]
        consistent = len(vals) == 1
        rows.append({
            "field": ".".join(path),
            "consistent": "YES" if consistent else "NO",
            "n_distinct_values": len(vals),
            "values": " | ".join(sorted(vals)),
        })
    with (out_dir / "formal_config_reconciliation.tsv").open("w", encoding="utf-8") as f:
        f.write("field\tconsistent\tn_distinct_values\tvalues\n")
        for r in rows:
            f.write(f"{r['field']}\t{r['consistent']}\t{r['n_distinct_values']}\t{r['values']}\n")

    inconsistent = [r for r in rows if r["consistent"] == "NO"]
    print(f"reconciliation -> {out_dir / 'formal_config_reconciliation.tsv'}")
    print(f"总字段数={len(rows)}，不一致字段数={len(inconsistent)}")
    for r in inconsistent:
        print(f"  DIFF: {r['field']} = {r['values']}")

    # 冻结 formal_v1_archived.yaml（从第一个 run 提取，恢复 placeholder）
    raw = json.loads(files[0].read_text())
    raw["data"]["train_path"] = "FOLD_TRAIN_PLACEHOLDER"
    raw["data"]["validation_path"] = "FOLD_VAL_PLACEHOLDER"
    raw["data"]["test_path"] = "FOLD_TEST_PLACEHOLDER"
    raw["training"]["seed"] = "SEED_PLACEHOLDER"
    raw["training"]["output_dir"] = "OUTPUT_DIR_PLACEHOLDER"
    header = (
        "# formal_v1_archived.yaml — 从 15 个 archived runs 的 resolved_config 反查冻结\n"
        "# 历史真值：amp_dtype=float16、无 non_finite_policy（fail_fast 字段不存在）。\n"
        "# 与当前 sitne_walk.optimal.yaml 的差异见 formal_config_reconciliation.tsv 之外的专门比对。\n"
    )
    (out_dir / "formal_v1_archived.yaml").write_text(
        header + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    print(f"frozen -> {out_dir / 'formal_v1_archived.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
