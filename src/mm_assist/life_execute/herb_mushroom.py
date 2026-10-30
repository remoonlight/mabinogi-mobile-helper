"""採集藥草「淨淨蘑菇」：寻找附近，成功=蘑菇净增。"""

from __future__ import annotations

from typing import Any, Callable

from mm_assist.life_execute.skill_nearby import execute_skill_nearby
from mm_assist.life_plan.types import ExecuteResult

HERB_DETAIL = "淨淨蘑菇"
SKILL_ID = "herb"


def execute_herb_mushroom(
    cfg: Any = None,
    win: Any = None,
    *,
    items: dict[str, int] | None = None,
    **kwargs: Any,
) -> ExecuteResult:
    return execute_skill_nearby(
        cfg,
        win,
        skill_id=SKILL_ID,
        item_name=HERB_DETAIL,
        output=HERB_DETAIL,
        items=items,
        **kwargs,
    )
