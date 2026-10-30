"""跨状态栏重启复用只读组件地址；进程生命周期和类型都匹配才作为候选。"""
from __future__ import annotations

import json
import os
import threading
import subprocess
import sys

from mm_assist.config import ROOT
from mm_assist.memory.process import read_ptr

_LOCK = threading.Lock()


def _birth(pid: int) -> str:
    import win32api
    import win32process
    handle = win32api.OpenProcess(0x1000, False, pid)
    try:
        return win32process.GetProcessTimes(handle)["CreationTime"].isoformat()
    finally:
        handle.Close()


def load_hint(hproc, pid: int, kind: str, klass: int) -> int:
    try:
        row = json.loads((ROOT / "logs/life_component_hints.json").read_text(encoding="utf-8"))
        if row.get("pid") != pid or row.get("birth") != _birth(pid):
            return 0
        ptr = int(row.get("components", {}).get(kind, 0))
        return ptr if ptr and read_ptr(hproc, ptr) == klass else 0
    except Exception:
        return 0


def save_hint(pid: int, kind: str, ptr: int) -> None:
    """只保存地址，不保存数量、耐久或队列成员；调用方仍须完整解码验证。"""
    try:
        birth = _birth(pid)
        path = ROOT / "logs/life_component_hints.json"
        with _LOCK:
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                row = {}
            if row.get("pid") != pid or row.get("birth") != birth:
                row = {"pid": pid, "birth": birth, "components": {}}
            row["components"][kind] = int(ptr)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(row), encoding="utf-8")
            tmp.replace(path)
    except Exception:
        pass  # 地址提示是优化；无法保存不能改变真实读数。


def prepare_hints(pid: int) -> bool:
    """首次定位交给无窗口的只读子进程，避开状态栏识别线程的扫堆争用。"""
    birth = _birth(pid)
    def ready():
        try:
            row = json.loads((ROOT / "logs/life_component_hints.json").read_text(encoding="utf-8"))
            return row.get("pid") == pid and row.get("birth") == birth and all(
                int(row.get("components", {}).get(k, 0)) > 0 for k in ("bag", "facilities"))
        except (OSError, ValueError, TypeError):
            return False
    if ready():
        return True
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mm_assist.memory.component_hints", "--warm", str(pid)],
            cwd=ROOT, capture_output=True, timeout=35,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0 and ready()
    except (OSError, subprocess.TimeoutExpired):
        return False


def _warm(pid: int) -> bool:
    # 此入口只读内存，不导入游戏执行器，不授权键鼠。
    from mm_assist.memory.process import open_process, close_handle
    from mm_assist.memory.readers.consumable import read_consumable_counts
    from mm_assist.memory.readers.facility_queue import read_facility_queues
    h = open_process(pid)
    try:
        bag = read_consumable_counts(h, pid)
        queues = read_facility_queues(h, pid)
        return bool(bag) and queues is not None
    finally:
        close_handle(h)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--warm":
        raise SystemExit("read-only usage: --warm <game-pid>")
    raise SystemExit(0 if _warm(int(sys.argv[2])) else 1)
