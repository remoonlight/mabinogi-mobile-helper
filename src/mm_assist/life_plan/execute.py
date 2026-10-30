"""计划项执行：已接线动作一律 ui_act（四参认页 + 点击）。"""

from __future__ import annotations

from typing import Any

from mm_assist.life_execute.fill_bottle import WELL_DETAIL, execute_fill_bottle
from mm_assist.life_execute.herb_mushroom import HERB_DETAIL, execute_herb_mushroom
from mm_assist.life_execute.medicine_process import execute_facility_claim, execute_facility_feed
from mm_assist.life_execute.ui_act import pump_aborted, wrap_progress
from mm_assist.life_plan.types import ExecuteResult, PlanItem

NOT_READY = "not_ready"
NOT_READY_MSG = "采集/投领执行器未就绪（不点游戏）"


def try_execute(
    item: PlanItem,
    *,
    cfg: Any = None,
    win: Any = None,
    items: dict[str, int] | None = None,
    **kwargs: Any,
) -> ExecuteResult:
    if item is None:
        return ExecuteResult(ok=False, status="empty", message="无计划")
    from mm_assist.life_plan.recipes import RETIRED_ITEMS
    if item.recipe in RETIRED_ITEMS or item.detail in RETIRED_ITEMS:
        return ExecuteResult(False, 'not_ready', '已移除的配方或材料，不执行旧请求')
    if pump_aborted():
        return ExecuteResult(ok=False, status="cancel", message="已停止操作")
    kwargs = dict(kwargs)
    kwargs["on_progress"] = wrap_progress(kwargs.get("on_progress"), title=str(item.title or ""))
    if item.kind == "repair":
        from mm_assist.life_execute.repair import execute_repair

        if cfg is None:
            return ExecuteResult(
                ok=False,
                status="need_context",
                message="修理需要游戏窗上下文",
            )
        return execute_repair(cfg, win, force=True, **kwargs)
    if item.kind == "gather":
        from mm_assist.life_execute.obtain_gather import OBTAIN_MATS, execute_obtain_gather
        from mm_assist.life_execute.skill_nearby import execute_skill_nearby
        from mm_assist.life_execute.ui_act import skill_id_for
        from mm_assist.life_plan.recipes import GATHER_NODE, SKILL_BAG_ITEM

        mat = str(item.recipe or "").strip()
        from mm_assist.life_execute.insect_obtain import INSECT_MATS, execute_insect_obtain
        if mat in INSECT_MATS:
            return execute_insect_obtain(cfg, win, mat=mat, items=items, **kwargs)
        skill = str(item.skill or "").strip()
        name = str(item.detail or "").strip()
        if not mat:
            for raw, (_sk, det) in GATHER_NODE.items():
                if det == name:
                    mat = raw
                    break
        if mat in GATHER_NODE:
            sk, det = GATHER_NODE[mat]
            if not skill:
                skill = sk
            if not name or name == mat:
                name = det
        if name == WELL_DETAIL:
            return execute_fill_bottle(cfg, win, items=items, **kwargs)
        if name == HERB_DETAIL:
            return execute_herb_mushroom(cfg, win, items=items, **kwargs)
        if mat in OBTAIN_MATS:
            return execute_obtain_gather(cfg, win, mat=mat, items=items, **kwargs)
        sid = skill_id_for(skill)
        out = SKILL_BAG_ITEM.get(name) or mat or name
        if sid and name and out:
            return execute_skill_nearby(
                cfg,
                win,
                skill_id=sid,
                item_name=name,
                output=out,
                items=items,
                **kwargs,
            )
    if item.kind == "claim" and item.facility in {"metal", "wood", "cloth", "medicine"}:
        return execute_facility_claim(
            cfg,
            win,
            facility=str(item.facility),
            items=items,
            **kwargs,
        )
    if item.kind == "feed" and item.facility in {"metal", "wood", "cloth", "medicine"}:
        return execute_facility_feed(
            cfg,
            win,
            facility=str(item.facility),
            recipe=str(item.recipe or ""),
            items=items,
            **kwargs,
        )
    return ExecuteResult(ok=False, status=NOT_READY, message=NOT_READY_MSG)


NAV_STATUSES = frozenset(
    {
        "wrong_page",
        "wrong_tab",
        "no_path",
        "nav_timeout",
        "nav_verify_timeout",
        "nav_no_move",
    }
)
SENSE_STATUSES = frozenset(
    {
        "unknown_overlay",
        "unknown_ui",
        "not_main_hud",
        "stp_busy",
        "stale_ui",
        "ui_unread",
        "click_yield",
        "need_verify",
        "facility_unknown",
        "btn_unstable",
        "btn_ambiguous",
        "stp_unknown",
    }
)


def is_nav_result(r: ExecuteResult) -> bool:
    st = str(r.status or "")
    msg = str(r.message or "")
    if st in NAV_STATUSES or st.startswith("nav_"):
        return True
    if "未到达" in msg:
        return True
    return False


def is_sense_result(r: ExecuteResult) -> bool:
    if is_nav_result(r):
        return False
    st = str(r.status or "")
    msg = str(r.message or "")
    if st in SENSE_STATUSES:
        return True
    if msg.startswith("找不到") or "未认出" in msg or "界面参数未" in msg:
        return True
    if st.startswith("找不到"):
        return True
    return False
