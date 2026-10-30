"""一键修理执行器：生产旁路 + 两次一检由外层暂停取消。"""

from __future__ import annotations

from typing import Any, Callable

from mm_assist.life_plan.types import ExecuteResult
from mm_assist.safety.input import release_mouse_buttons
from mm_assist.safety.lease import prefer_visible_clicks, production_mode
from mm_assist.tool_repair import clear_op_cancel, one_click_repair, op_cancelled


def execute_repair(
    cfg: Any,
    win: Any,
    *,
    force: bool = True,
    on_progress: Callable[[str], None] | None = None,
    **kwargs: Any,
) -> ExecuteResult:
    """状态栏/计划表「一键修理」= 一次修全部工具。force 默认 True（人手/计划已判定要修）。"""
    clear_op_cancel()
    production_mode(True)
    prefer_visible_clicks(True)
    action_issued = False

    def issued():
        nonlocal action_issued
        action_issued = True
        callback = kwargs.get('on_action_issued')
        if callback is not None:
            callback()

    try:
        at_dialog = False
        if win is not None:
            from mm_assist.life_execute.ui_act import prepare_to
            from mm_assist.platform.capture import grab_client
            from mm_assist.safety.input import click_client

            prep = prepare_to(
                "main_hud",
                grab=lambda: grab_client(win),
                click=lambda x, y: bool(click_client(win, x, y)),
                on_progress=on_progress,
                allow_dialog=True,
                title="一键修理",
            )
            if not prep.ok:
                return prep
            at_dialog = prep.message == "dialog"
        extra = {
            k: kwargs[k]
            for k in (
                "near_only",
                "skip_loc_probe",
                "nav_fn",
                "dialog_fn",
                "dur_before",
                "dur_after_fn",
                "require_durability_rose",
                "phase",
            )
            if k in kwargs
        }
        if at_dialog:
            if extra.get("phase") == "nav":
                return ExecuteResult(ok=True, status="at_dialog", message="已在修理对话")
            extra["phase"] = "dialog"
        if 'dialog_fn' not in extra:
            from mm_assist.life_execute.repair_dialog import run_repair_dialog_chain
            extra['dialog_fn'] = lambda cfg, win: run_repair_dialog_chain(
                cfg, win, on_action_issued=issued)
        _win, out = one_click_repair(
            cfg, win, force=force, on_progress=on_progress, **extra
        )
        if op_cancelled() or str(out.get("stage") or "") == "cancel":
            return ExecuteResult(ok=False, status="cancel", message="已暂停", action_issued=action_issued)
        if out.get("ok") and str(out.get("stage") or "") == "skip":
            return ExecuteResult(ok=True, status="skip", message="工具耐久无需修")
        if out.get("ok") and str(out.get("stage") or "") == "walking":
            return ExecuteResult(ok=True, status="walking", message="已点前往，等待到站")
        if out.get("ok") and str(out.get("stage") or "") == "at_dialog":
            return ExecuteResult(ok=True, status="at_dialog", message="已在修理对话")
        if out.get("ok"):
            msg = (
                "耐久已回升"
                if out.get("durability_rose")
                else "对话链已完成（耐久仅记录）"
            )
            return ExecuteResult(ok=True, status="done", message=msg, action_issued=action_issued)
        return ExecuteResult(
            ok=False,
            status=str(out.get("stage") or "fail"),
            message=str(out.get("error") or "修理失败"),
            action_issued=action_issued,
        )
    except Exception as exc:
        return ExecuteResult(ok=False, status="error", message=str(exc), action_issued=action_issued)
    finally:
        production_mode(False)
        try:
            release_mouse_buttons()
        except Exception:
            pass
