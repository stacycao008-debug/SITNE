#!/usr/bin/env python3
"""进度监控脚本。

读取 manifests/pending_tasks.json + logs/ 目录 →
输出表格：TaskID | Status | Deps | Elapsed | ETA | Output。
支持 --watch（每60s刷新）、--json（输出JSON供外部消费）。
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
MANIFEST_PATH = ROOT / "manifests/pending_tasks.json"


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def task_status(task: dict) -> str:
    """判断任务当前状态。"""
    tid = task["id"]
    check_path = task.get("output_check_path", "")
    
    # 检查是否已完成
    if check_path and (ROOT / check_path).exists():
        return "completed"
    
    # 检查是否被 blocked
    if task.get("status") == "blocked":
        return "blocked"
    
    # 检查是否有日志文件（正在运行）
    log_path = LOGS_DIR / f"{tid}.log"
    if log_path.exists():
        mtime = log_path.stat().st_mtime
        age_s = time.time() - mtime
        if age_s < 300:  # 5分钟内更新过 → 运行中
            return "running"
        else:
            return "stale"  # 日志存在但很久没更新
    
    return "pending"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def show_table(manifest: dict):
    """打印表格化状态。"""
    tasks = manifest["tasks"]
    
    header = f"{'TaskID':<25} {'Status':<12} {'Level':<6} {'Priority':<9} {'Est':<8} {'Deps':<20} {'Output'}"
    print(header)
    print("-" * len(header))
    
    for tid in sorted(tasks, key=lambda x: (tasks[x].get("level", 99), tasks[x].get("priority", 99))):
        t = tasks[tid]
        status = task_status(t)
        deps = ", ".join(t.get("depends_on", [])) if t.get("depends_on") else "-"
        est = format_duration(t.get("estimated_minutes", 0) * 60)
        output = t.get("output_check_path", "-")
        
        # Status 上色（简单字符标记）
        status_icon = {"completed": "✅", "running": "🔄", "pending": "⏳", "blocked": "🚫", "stale": "⚠️"}.get(status, "❓")
        
        print(f"{tid:<25} {status_icon} {status:<9} {t.get('level',0):<6} {t.get('priority',0):<9} {est:<8} {deps:<20} {output}")
    
    # 汇总统计
    statuses = {}
    for tid in tasks:
        s = task_status(tasks[tid])
        statuses[s] = statuses.get(s, 0) + 1
    
    print()
    print("=" * 70)
    print("汇总:", " | ".join(f"{s}: {c}" for s, c in sorted(statuses.items())))
    
    pending = statuses.get("pending", 0)
    running = statuses.get("running", 0)
    if running > 0:
        print(f"当前 {running} 个任务运行中, {pending} 个等待中")
    elif pending > 0:
        print(f"{pending} 个任务等待执行")
    else:
        not_done = statuses.get("pending", 0) + statuses.get("running", 0) + statuses.get("stale", 0)
        if not_done == 0:
            non_blocked = sum(1 for s in statuses if s != "blocked")
            blocked_count = statuses.get("blocked", 0)
            if blocked_count > 0:
                print(f"Level 0-2 任务已全部完成 ({non_blocked} 个)。{blocked_count} 个 Level 3 任务处于 blocked 状态。")
            else:
                print("所有任务已完成！")


def show_json(manifest: dict):
    """输出 JSON 格式供外部消费。"""
    tasks = manifest["tasks"]
    result = {}
    for tid in sorted(tasks):
        t = tasks[tid]
        result[tid] = {
            "name": t["name"],
            "status": task_status(t),
            "level": t.get("level", 0),
            "priority": t.get("priority", 0),
            "estimated_minutes": t.get("estimated_minutes", 0),
            "depends_on": t.get("depends_on", []),
            "output_check_path": t.get("output_check_path", ""),
            "blocked_reason": t.get("blocked_reason", ""),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="实验进度监控")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                        help="每隔 N 秒刷新一次 (默认: 不刷新，输出一次)")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH),
                        help="任务清单路径")
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    
    if args.watch > 0:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print(f"=== SITNE-Walk 实验进度监控 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
                print(f"刷新间隔: {args.watch}s | Ctrl+C 退出\n")
                manifest = load_manifest()
                if args.json:
                    show_json(manifest)
                else:
                    show_table(manifest)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n监控已停止")
    else:
        manifest = load_manifest()
        if args.json:
            show_json(manifest)
        else:
            print(f"=== SITNE-Walk 实验进度监控 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
            show_table(manifest)


if __name__ == "__main__":
    main()
