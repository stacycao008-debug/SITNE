#!/usr/bin/env python3
"""拓扑感知实验任务编排主控脚本。

读取 manifests/pending_tasks.json → 构建 DAG → 拓扑排序 → 
检查 output_check_path 跳过已完成 → 逐任务执行 subprocess →
写入 logs/{task_id}.log → 更新任务状态。

支持 --dry-run（打印执行计划不运行）、--resume（跳过已完成）、
--timeout（每任务超时，默认3600s）。
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TOPOLOGY] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "_topology.log", mode="a"),
    ],
)
log = logging.getLogger("topology")


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """加载任务清单并验证结构。"""
    with open(manifest_path) as f:
        manifest = json.load(f)
    if "tasks" not in manifest:
        raise ValueError("manifest 缺少 'tasks' 字段")
    log.info("已加载 manifest: %d 个任务", len(manifest["tasks"]))
    return manifest


def topological_sort(tasks: dict[str, Any]) -> list[str]:
    """按依赖关系进行拓扑排序，返回有序的任务 ID 列表。"""
    in_degree: dict[str, int] = {}
    adj: dict[str, list[str]] = {}
    
    for tid in tasks:
        in_degree[tid] = 0
        adj[tid] = []
    
    for tid, task in tasks.items():
        for dep in task.get("depends_on", []):
            if dep in tasks:
                adj.setdefault(dep, []).append(tid)
                in_degree[tid] = in_degree.get(tid, 0) + 1
            else:
                log.warning("任务 %s 依赖未知任务 %s，已忽略", tid, dep)
    
    queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
    result = []
    
    while queue:
        # 按 priority 排序出队（数字小的优先）
        queue_items = sorted(queue, key=lambda x: tasks[x].get("priority", 99))
        tid = queue_items[0]
        queue.remove(tid)
        result.append(tid)
        for neighbor in adj.get(tid, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(result) != len(tasks):
        remaining = set(tasks) - set(result)
        log.error("检测到循环依赖！未排序的任务: %s", remaining)
        # 将剩余任务追加到末尾
        remaining_sorted = sorted(remaining, key=lambda x: tasks[x].get("priority", 99))
        result.extend(remaining_sorted)
    
    return result


def is_task_complete(task: dict[str, Any]) -> bool:
    """检查任务的 output_check_path 是否存在。"""
    check_path = task.get("output_check_path", "")
    if not check_path:
        return False
    full_path = ROOT / check_path
    return full_path.exists()


def execute_task(task_id: str, task: dict[str, Any], timeout_s: int, dry_run: bool = False) -> dict[str, Any]:
    """执行单个任务，返回执行结果。"""
    result = {
        "task_id": task_id,
        "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": 0.0,
        "output": "",
        "error": "",
    }
    
    if is_task_complete(task):
        log.info("[%s] 已完成 (output_check_path 存在)，跳过", task_id)
        result["status"] = "cached"
        return result
    
    if task.get("status") == "blocked":
        log.info("[%s] BLOCKED: %s", task_id, task.get("blocked_reason", "未知原因"))
        result["status"] = "blocked"
        return result
    
    if dry_run:
        log.info("[%s] DRY-RUN: 将执行 %s %s", task_id, task["script"], " ".join(task.get("args", [])))
        result["status"] = "dry_run"
        return result
    
    script = ROOT / task["script"]
    if not script.exists():
        log.error("[%s] 脚本不存在: %s", task_id, script)
        result["status"] = "script_missing"
        return result
    
    log_path = LOGS_DIR / f"{task_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [sys.executable, str(script)] + task.get("args", [])
    log.info("[%s] 开始执行: %s", task_id, " ".join(cmd))
    
    t0 = time.time()
    try:
        with open(log_path, "w") as log_file:
            proc = subprocess.run(
                cmd,
                timeout=timeout_s,
                cwd=str(ROOT),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        elapsed = time.time() - t0
        result["elapsed_s"] = elapsed
        
        if proc.returncode == 0:
            log.info("[%s] 成功完成 (%.0fs)", task_id, elapsed)
            result["status"] = "success"
        else:
            log.error("[%s] 失败, returncode=%d (%.0fs)", task_id, proc.returncode, elapsed)
            result["status"] = "failed"
            result["error"] = f"returncode={proc.returncode}"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log.error("[%s] 超时 (>%ds)", task_id, timeout_s)
        result["status"] = "timeout"
        result["elapsed_s"] = elapsed
        result["error"] = f"timeout after {timeout_s}s"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="拓扑感知实验任务编排器")
    parser.add_argument("--manifest", type=str, default=str(ROOT / "manifests/pending_tasks.json"),
                        help="任务清单路径")
    parser.add_argument("--dry-run", action="store_true", help="仅打印执行计划，不实际运行")
    parser.add_argument("--timeout", type=int, default=3600, help="每个任务的超时秒数 (默认3600)")
    parser.add_argument("--resume", action="store_true", help="跳过已完成任务 (默认行为)")
    parser.add_argument("--tasks", type=str, default="", help="仅执行指定任务 (逗号分隔的 task_id)")
    parser.add_argument("--level", type=int, default=0, help="最大执行层级 (默认: 执行所有层级)")
    args = parser.parse_args()
    
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest(Path(args.manifest))
    tasks = manifest["tasks"]
    
    # 过滤层级
    filtered_tasks = {tid: t for tid, t in tasks.items() if t.get("level", 99) <= args.level}
    if len(filtered_tasks) < len(tasks):
        log.info("层级过滤: 从 %d 缩减到 %d 个任务 (level <= %d)", 
                 len(tasks), len(filtered_tasks), args.level)
        tasks = filtered_tasks
    
    # 过滤指定任务
    if args.tasks:
        specified = set(args.tasks.split(","))
        filtered_tasks = {tid: t for tid, t in tasks.items() if tid in specified}
        log.info("任务过滤: 仅执行 %s", ", ".join(filtered_tasks))
        tasks = filtered_tasks
    
    # 拓扑排序
    ordered = topological_sort(tasks)
    
    log.info("=" * 70)
    log.info("拓扑排序后的执行顺序 (%d 个任务):", len(ordered))
    for i, tid in enumerate(ordered):
        t = tasks[tid]
        cached = " [CACHED]" if is_task_complete(t) else ""
        blocked = f" [BLOCKED: {t.get('blocked_reason', '')[:40]}...]" if t.get("status") == "blocked" else ""
        log.info("  %2d. %-30s level=%d priority=%d est=%dmin%s%s",
                 i + 1, tid, t.get("level", 0), t.get("priority", 0),
                 t.get("estimated_minutes", 0), cached, blocked)
    log.info("=" * 70)
    
    # 汇总预估
    pending_tasks = [tid for tid in ordered if not is_task_complete(tasks[tid]) and tasks[tid].get("status") != "blocked"]
    total_est = sum(tasks[tid].get("estimated_minutes", 0) for tid in pending_tasks)
    log.info("待执行: %d 个任务, 预计总耗时: %d 分钟 (约 %.1f 小时)", 
             len(pending_tasks), total_est, total_est / 60.0)
    
    if args.dry_run:
        log.info("DRY-RUN 模式: 仅打印计划，不实际执行")
        return
    
    # 执行
    results: list[dict[str, Any]] = []
    total_start = time.time()
    
    for tid in ordered:
        task = tasks[tid]
        
        if is_task_complete(task):
            log.info("[%s] 跳过 (已完成)", tid)
            results.append({"task_id": tid, "status": "cached", "elapsed_s": 0.0})
            continue
        
        if task.get("status") == "blocked":
            log.info("[%s] 跳过 (blocked)", tid)
            results.append({"task_id": tid, "status": "blocked", "elapsed_s": 0.0})
            continue
        
        # 检查依赖是否都已完成
        deps_ok = True
        for dep in task.get("depends_on", []):
            dep_complete = is_task_complete(tasks.get(dep, {}))
            if not dep_complete:
                log.warning("[%s] 依赖 %s 未完成，跳过", tid, dep)
                deps_ok = False
        if not deps_ok:
            results.append({"task_id": tid, "status": "deps_unmet", "elapsed_s": 0.0})
            continue
        
        result = execute_task(tid, task, args.timeout, dry_run=False)
        results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # 汇总报告
    log.info("=" * 70)
    log.info("执行完成 (%.0fs)", total_elapsed)
    status_counts: dict[str, int] = {}
    for r in results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    for s, c in sorted(status_counts.items()):
        log.info("  %-15s: %d", s, c)
    
    # 保存结果
    summary_path = LOGS_DIR / "topology_results.json"
    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_s": total_elapsed,
        "status_counts": status_counts,
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("执行结果已保存: %s", summary_path)


if __name__ == "__main__":
    main()
