#!/usr/bin/env python3
"""Source confounding 静态分析（无泄漏，纯 CPU）。

只做三件无泄漏的事：
  1. relation × source contingency 表（train/validation/test 三 split 分开 + 合并）；
  2. source–relation association（conditional proportions + 每 relation 的来源集中度）；
  3. 输出 interpretation 提示（哪些 relation 被单一 source 独占，即存在混杂）。

明确不做：source-only predictive baseline（等于用标签生成过程的信息预测标签，
是数据泄漏，标为 OPTIONAL / DESIGN REVIEW REQUIRED）。

用法：
    PYTHONPATH="$PWD/06_code:$PYTHONPATH" python3 \
        scripts/rerun_audit/run_source_analysis.py --project-root "$PWD" \
        --output-dir rerun_submission_audit/04_source_analysis
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

EVIDENCE_FILES = {
    "train": "02_data_canonical/dataset_Bstar/v2/views/X_type/model_input_event_membership.tsv",
    "validation": "02_data_canonical/dataset_Bstar/v2/targets/validation_type_evidence_events.tsv",
    "test": "02_data_canonical/dataset_Bstar/v2/targets/test_type_evidence_events.tsv",
}


def _load(path: Path) -> list[tuple[str, str, str]]:
    """返回 (split_role, coarse_type, source_release_id) 列表。"""
    rows = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            src = row.get("source_release_id", "UNKNOWN") or "UNKNOWN"
            ct = row.get("coarse_type", "UNKNOWN") or "UNKNOWN"
            role = row.get("split_role", "UNKNOWN") or "UNKNOWN"
            rows.append((role, ct, src))
    return rows


def _contingency(rows: list[tuple[str, str, str]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = collections.Counter()
    for _role, ct, src in rows:
        counts[(ct, src)] += 1
    return counts


def _write_contingency(
    path: Path,
    counts: dict[tuple[str, str], int],
    split_role: str,
) -> None:
    relations = sorted({ct for ct, _ in counts})
    sources = sorted({src for _, src in counts})
    with path.open("w", encoding="utf-8") as handle:
        handle.write("split_role\trelation\tsource\tevents\n")
        for ct in relations:
            for src in sources:
                handle.write(f"{split_role}\t{ct}\t{src}\t{counts.get((ct, src), 0)}\n")


def _association(counts: dict[tuple[str, str], int]) -> list[dict]:
    """每 relation 计算：总事件数、各 source 占比、最大 source 份额（集中度）。"""
    relations = sorted({ct for ct, _ in counts})
    out = []
    for ct in relations:
        total = sum(v for (r, _), v in counts.items() if r == ct)
        by_src = {src: v for (r, src), v in counts.items() if r == ct}
        max_src = max(by_src, key=by_src.get) if by_src else "NONE"
        max_share = (by_src[max_src] / total) if total else 0.0
        out.append({
            "relation": ct,
            "total_events": total,
            "max_source": max_src,
            "max_source_share": round(max_share, 4),
            "source_counts": dict(sorted(by_src.items())),
            "exclusive": len(by_src) == 1,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple[str, str, str]] = []
    contingency_path = out / "source_relation_contingency.tsv"
    with contingency_path.open("w", encoding="utf-8") as handle:
        handle.write("split_role\trelation\tsource\tevents\n")
        for split_role, rel_path in EVIDENCE_FILES.items():
            path = root / rel_path
            if not path.is_file():
                print(f"[warn] missing {path}")
                continue
            rows = _load(path)
            all_rows.extend(rows)
            counts = _contingency(rows)
            relations = sorted({ct for ct, _ in counts})
            sources = sorted({src for _, src in counts})
            for ct in relations:
                for src in sources:
                    handle.write(f"{split_role}\t{ct}\t{src}\t{counts.get((ct, src), 0)}\n")

    merged = _contingency(all_rows)
    assoc = _association(merged)
    assoc_path = out / "source_relation_association.tsv"
    with assoc_path.open("w", encoding="utf-8") as handle:
        handle.write("relation\ttotal_events\tmax_source\tmax_source_share\texclusive\n")
        for a in assoc:
            handle.write(
                f"{a['relation']}\t{a['total_events']}\t{a['max_source']}\t"
                f"{a['max_source_share']}\t{str(a['exclusive']).lower()}\n"
            )

    # interpretation
    interp_lines = [
        "# Source confounding interpretation",
        "",
        "## relation × source 集中度（合并三 split）",
        "",
        "| relation | total events | max source | max share | exclusive |",
        "|---|---|---|---|---|",
    ]
    for a in assoc:
        interp_lines.append(
            f"| {a['relation']} | {a['total_events']} | {a['max_source']} | "
            f"{a['max_source_share']:.4f} | {str(a['exclusive']).lower()} |"
        )
    interp_lines += [
        "",
        "## 判定",
        "",
    ]
    exclusive = [a for a in assoc if a["exclusive"]]
    if exclusive:
        names = ", ".join(a["relation"] for a in exclusive)
        interp_lines.append(
            f"- 以下 relation 被单一 source 独占（存在 source/relation 混杂）：{names}。"
        )
        interp_lines.append(
            "  模型学到的可能是 source/curation-pattern signal 而非纯 relation-specific signal。"
        )
    else:
        interp_lines.append("- 无 relation 被单一 source 独占。")
    interp_lines += [
        "",
        "## 待办（Phase E 有 per-query 结果后）",
        "",
        "- source-stratified evaluation：在 IntAct-supporting subset 内报告各 relation 性能；",
        "- 判断 SOURCE-CONFOUNDING LOW / MATERIAL / UNRESOLVED（需结合 stratified 性能差异）。",
        "- 注意：source-only predictive baseline 标 OPTIONAL / DESIGN REVIEW REQUIRED（数据泄漏风险）。",
        "",
    ]
    (out / "interpretation.md").write_text("\n".join(interp_lines), encoding="utf-8")

    print(f"contingency -> {contingency_path}")
    print(f"association  -> {assoc_path}")
    for a in assoc:
        print(f"  {a['relation']}: total={a['total_events']} "
              f"max_source={a['max_source']} share={a['max_source_share']} "
              f"exclusive={a['exclusive']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
