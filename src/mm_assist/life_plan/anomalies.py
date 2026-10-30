"""生活备料日记：动作、时间、成败；只留最近一天。"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from mm_assist.config import ROOT

LOG_REL = "logs/life_prep_diary.jsonl"
SHOT_REL = "logs/life_fail_shots"
KEEP_S = 86400.0
_LAST: dict[tuple[str, str], float] = {}
_DEDUP_S = 1.0
_IO = threading.Lock()


def log_life_anomaly(
    *,
    plan_title: str = "",
    plan_id: str = "",
    kind: str = "",
    action: str = "",
    tag: str = "",
    escaped: bool | None = None,
    extra: str = "",
    stage: str = "",
    from_node: str = "",
    ok: bool | None = None,
    elapsed_s: float | None = None,
    run_started_at: float = 0.0,
) -> None:
    now = time.time()
    key = (str(plan_id or ""), str(tag or ""))
    prev = _LAST.get(key, 0.0)
    if tag != "execute_result" and now - prev < _DEDUP_S:
        return
    _LAST[key] = now
    path = ROOT / LOG_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "ts": now,
        "plan_id": str(plan_id or ""),
        "plan": str(plan_title or ""),
        "kind": str(kind or ""),
        "action": str(action or ""),
        "tag": str(tag or ""),
        "ok": ok,
        "escaped": escaped,
        "extra": str(extra or ""),
        "stage": str(stage or ""),
        "from_node": str(from_node or ""),
        "elapsed_s": elapsed_s,
        "run_started_at": float(run_started_at),
    }
    line = json.dumps(row, ensure_ascii=False)
    cutoff = now - KEEP_S
    with _IO:
        kept: list[str] = []
        if path.is_file():
            for old in path.read_text(encoding="utf-8").splitlines():
                if not old.strip():
                    continue
                try:
                    ts = float(json.loads(old).get("ts") or 0)
                except Exception:
                    continue
                if ts >= cutoff:
                    kept.append(old)
        kept.append(line)
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def cache_fail_shot(win: Any = None, *, tag: str = "", frame: Any = None) -> str:
    """失败画面进缓存目录，只留最近一天。没有窗就只返回空路径。"""
    img = frame
    if img is None and win is not None:
        try:
            from mm_assist.platform.capture import grab_client

            img = grab_client(win)
        except Exception:
            img = None
    saved = ""
    if img is not None:
        try:
            from mm_assist.platform.capture import save_capture

            prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(tag or "fail"))[:24]
            saved = str(save_capture(img, ROOT / SHOT_REL, prefix=prefix or "fail"))
        except Exception:
            saved = ""
    cutoff = time.time() - KEEP_S
    folder = ROOT / SHOT_REL
    try:
        if folder.is_dir():
            for p in folder.iterdir():
                if not p.is_file():
                    continue
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except OSError:
                    pass
    except OSError:
        pass
    return saved
