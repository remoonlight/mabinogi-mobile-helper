"""裝水短链：空瓶检查后走日常採集水井寻找附近。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mm_assist.config import ROOT
from mm_assist.life_plan.types import ExecuteResult

EMPTY = "空瓶"
OUTPUT = "裝水的瓶子"
WELL_DETAIL = "水井"
SKILL_ID = "daily"
MAX_CLICKS = 2


def well_template_ready(*, root: Path | None = None) -> bool:
    p = (root or ROOT) / "knowledge" / "templates" / "items" / "daily" / "水井.png"
    try:
        return p.is_file() and p.stat().st_size >= 100
    except OSError:
        return False


def _count(items: dict[str, int] | None, name: str) -> int:
    if not items:
        return 0
    return int(items.get(name, 0) or 0)


def execute_fill_bottle(
    cfg: Any = None,
    win: Any = None,
    *,
    items: dict[str, int] | None = None,
    locate_well: Callable[[], dict[str, Any]] | None = None,
    do_click: Callable[[int, int], bool] | None = None,
    read_items: Callable[[], dict[str, int] | None] | None = None,
    grab: Callable[[], Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
    root: Path | None = None,
    **kwargs: Any,
) -> ExecuteResult:
    have = dict(items or {})
    if _count(have, EMPTY) < 1:
        return ExecuteResult(ok=False, status="no_empty_bottle", message="包里没有空瓶")
    before = _count(have, OUTPUT)

    if locate_well is not None:
        loc = dict(locate_well())
        if not loc.get("ok"):
            return ExecuteResult(
                ok=False,
                status=str(loc.get("error") or "well_unlocated"),
                message="找不到水井",
            )
        cx, cy = int(loc["cx"]), int(loc["cy"])

        def _after() -> dict[str, int]:
            if read_items is not None:
                return dict(read_items() or {})
            return have

        for n in range(1, MAX_CLICKS + 1):
            if on_progress:
                on_progress(f"点水井 {n}/{MAX_CLICKS}")
            if do_click is not None:
                do_click(cx, cy)
            bag = _after()
            if _count(bag, OUTPUT) > before:
                return ExecuteResult(
                    ok=True,
                    status="done",
                    message=f"裝水的瓶子 {_count(bag, OUTPUT) - before}",
                )
        return ExecuteResult(ok=False, status="fill_no_delta", message="点了水井但無净增")

    from mm_assist.life_execute.skill_nearby import execute_skill_nearby

    return execute_skill_nearby(
        cfg,
        win,
        skill_id=SKILL_ID,
        item_name=WELL_DETAIL,
        output=OUTPUT,
        items=have,
        do_click=do_click,
        read_items=read_items,
        grab=grab,
        on_progress=on_progress,
        root=root,
        **kwargs,
    )
