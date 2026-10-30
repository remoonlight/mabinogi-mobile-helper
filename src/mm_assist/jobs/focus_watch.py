"""只读切片占位：不跑 FocusWatch，抢焦/心跳一律中性。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_ADVICE_PATH = "logs/focus_watch.json"


def _abs_log_path(rel: str | Path) -> Path:
    from mm_assist.config import ROOT

    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def read_focus_advice(path: Path | None = None) -> dict[str, Any]:
    del path
    return {}


def agent_heartbeat_fresh(*_a, **_k) -> bool:
    return False


def agent_heartbeat_age_s(*_a, **_k) -> float | None:
    return None


def stamp_agent_input_heartbeat(*, who: str = "") -> None:
    del who
