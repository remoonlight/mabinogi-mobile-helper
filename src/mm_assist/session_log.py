"""状态栏会话日志：开窗即落盘，排障只看 latest。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from mm_assist.config import ROOT

log = logging.getLogger(__name__)

_SESSION_ATTACHED = False
LATEST_NAME = "watch_latest.log"


def watch_log_dir() -> Path:
    return ROOT / "logs"


def latest_watch_log() -> Path:
    return watch_log_dir() / LATEST_NAME


def start_watch_session_log() -> Path:
    """给 mm_assist logger 挂文件；每次开状态栏覆盖 watch_latest.log。"""
    global _SESSION_ATTACHED
    log_dir = watch_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session = log_dir / f"watch_{stamp}.log"
    latest = latest_watch_log()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    pack = logging.getLogger("mm_assist")
    pack.setLevel(logging.DEBUG)
    if _SESSION_ATTACHED:
        log.info("watch session log already on; latest=%s", latest)
        return latest
    for path, mode in ((session, "a"), (latest, "w")):
        fh = logging.FileHandler(path, mode=mode, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        pack.addHandler(fh)
    _SESSION_ATTACHED = True
    log.info("watch session log=%s latest=%s", session, latest)
    return session
