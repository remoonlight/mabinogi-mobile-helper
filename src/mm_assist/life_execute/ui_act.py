"""泵执行共用：先四参认页，再按图点击。认不到不点。"""

from __future__ import annotations

from mm_assist.life_plan.game_stop import operation_monotonic, operation_wall_time

from dataclasses import replace
from typing import Any, Callable

from mm_assist.life_execute.ui_logic import UiWhere, graph_node, hud_page_pending, recognized_click_xy, resolve_ui_node
from mm_assist.memory.readers.ui_stack import (
    stack_common_popup,
    stack_inv,
    stack_left_overlay,
)
from mm_assist.life_plan.types import ExecuteResult, PlanItem
from mm_assist.perception.state_ui import read_state_ui

WAIT_STP_S = 90.0
LEASE_CLICK_RESERVE_S = 18.0
WAIT_MENU_INSPECT_S = 8.0
HOLD_SETTLE_S = 2.0
HOLD_WATCH_S = 2.0
_FACILITY_RESULT_NEEDLE = "设施结果"
_COMMON_POPUP_LABEL = "确认弹窗"
_SKILL_ID = {
    "日常採集": "daily",
    "採集藥草": "herb",
    "伐木": "woodcut",
    "採礦": "mine",
    "剪羊毛": "shear",
    "昆蟲採集": "insect",
    "釣魚": "fish",
    "收割": "harvest",
    "鋤地": "hoe",
}


_PAGE_ZH = {
    "main_hud": "主界面",
    "system_menu": "菜单",
    "profile": "角色",
    "life_grid": "生活技能",
    "gather_skill": "生活技能",
    "gather_detail": "技能详情",
    "craft_hub": "製作",
    "facility": "设施",
    "facility_recipe": "设施",
    "inventory": "背包",
    "map_dunbarton": "地图",
    "dialog": "对话",
    "currency_wallet": "货币",
}


def inspect_is_menu() -> bool:
    """栏上「界面：」已是菜单（zh 含「菜单」或 node=system_menu）。"""
    from mm_assist.life_execute.ui_logic import read_inspect_record

    rec = read_inspect_record()
    zh = str((rec or {}).get("zh") or "").strip()
    node = str((rec or {}).get("node") or "").strip()
    return "菜单" in zh or node == "system_menu"


def inspect_is_hud_idle() -> bool:
    """栏上「界面：」为「主界面-待机」。无记录 / 「—」/ unknown / 动作中为假。"""
    rec = _inspect_rec()
    zh = str((rec or {}).get("zh") or "").strip()
    if not rec or not zh or zh in {"—", "-", "unknown"}:
        return False
    if "动作中" in zh or "unknown" in zh:
        return False
    return "主界面-待机" in zh


def inspect_is_true_hud() -> bool:
    """栏上已是真主界面（待机或动作中）。「—」/ unknown / 未采页为假。"""
    rec = _inspect_rec()
    zh = str((rec or {}).get("zh") or "").strip()
    node = str((rec or {}).get("node") or "").strip()
    if not rec or not zh or zh in {"—", "-", "unknown"}:
        return False
    if "unknown" in zh:
        return False
    if node == "main_hud":
        return True
    return "主界面-待机" in zh or "主界面-动作中" in zh


def _inspect_rec() -> dict | None:
    from mm_assist.life_execute.ui_logic import read_inspect_record

    rec = read_inspect_record()
    return rec if isinstance(rec, dict) else None


def promote_true_hud(where: UiWhere) -> UiWhere:
    """栈就绪无离开层，或栏上主界面：真主界面。假空（未就绪且栏上不是主界面）不升。"""
    if str(getattr(where, "node", "") or "") != "unknown":
        return where
    names = list(where.stack_names or ())
    if bool(where.stack_ready) and not stack_left_overlay(names):
        menu = 0 if where.menu is None else where.menu
        return replace(where, node="main_hud", menu=menu)
    if inspect_is_true_hud():
        menu = 0 if where.menu is None else where.menu
        return replace(where, node="main_hud", menu=menu)
    return where


def inspect_zh() -> str:
    """栏上「界面：」原文。点选默认只信这一行，不拿 stp 后栈串当页名。"""
    from mm_assist.life_execute.ui_logic import read_inspect_record

    rec = read_inspect_record()
    return str((rec or {}).get("zh") or "").strip()


def _stack_blob(text: str | None = None) -> str:
    if text is not None:
        return str(text or "")
    try:
        from mm_assist.perception.state_ui import read_state_ui

        view = read_state_ui()
    except Exception:
        return ""
    if view is None:
        return ""
    snap_s = str(getattr(view, "ui_stack", "") or "")
    raw = view.raw if isinstance(getattr(view, "raw", None), dict) else {}
    s = str(raw.get("ui_stack") or snap_s or "")
    if s.strip():
        return s
    names = raw.get("ui_stack_names") or ()
    return ">".join(str(x) for x in names)


def stack_has_facility_result(text: str | None = None) -> bool:
    """内存行 / 状态栏 stp 后栈串是否含「设施结果」（兼容 `>` 两侧空格）。"""
    blob = _stack_blob(text).replace(" ", "").replace("\u3000", "")
    return _FACILITY_RESULT_NEEDLE in blob


def stack_has_facility_recipe() -> bool:
    """Only a fresh, ready independent recipe popup counts as a covering layer."""
    from mm_assist.perception.state_ui import read_state_ui

    view = read_state_ui()
    raw = view.raw if view is not None and isinstance(view.raw, dict) else {}
    return bool(view is not None and view.fresh and view.pid_alive and
                raw.get("ui_stack_ready") and
                "AlteringPanelPopupUIInstance" in (raw.get("ui_stack_names") or ()))


def stack_has_common_popup(
    text: str | None = None, *, names: list[str] | tuple[str, ...] | None = None
) -> bool:
    """内存栈是否为 ButtonCommonPopup（附近没有设施 / 要前往吗）。"""
    if names:
        return stack_common_popup(list(names))
    if text is not None:
        blob = str(text or "")
        compact = blob.replace(" ", "").replace("\u3000", "")
        return "ButtonCommonPopup" in blob or _COMMON_POPUP_LABEL in compact
    try:
        from mm_assist.perception.state_ui import read_state_ui

        view = read_state_ui()
    except Exception:
        view = None
    if view is not None:
        raw = view.raw if isinstance(getattr(view, "raw", None), dict) else {}
        found = raw.get("ui_stack_names") or ()
        if stack_common_popup(list(found)):
            return True
    blob = _stack_blob().replace(" ", "").replace("\u3000", "")
    return "ButtonCommonPopup" in blob or _COMMON_POPUP_LABEL in blob


def inspect_facility_only_zh(zh: str | None = None) -> bool:
    """「界面：」为仅设施名，如 制作-加工-金属加工；含配方名则否。"""
    raw = inspect_zh() if zh is None else str(zh or "")
    text = raw.replace("界面：", "").replace("界面:", "").strip()
    text = text.replace("动作中", "").replace("（", "").replace("）", "").replace("(", "").replace(")", "")
    text = text.replace("－", "-").replace("—", "-").strip(" -")
    if not text or _FACILITY_RESULT_NEEDLE in text:
        return False
    parts = [p.strip() for p in text.split("-") if p.strip()]
    if len(parts) != 3:
        return False
    if parts[0] not in {"制作", "製作"}:
        return False
    if parts[1] != "加工":
        return False
    last = parts[2]
    if last == "设施":
        return True
    return last.endswith("加工") and last not in {"加工"}


def hold_facility_settle(
    *,
    sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], float] | None = None,
    hold_s: float | None = None,
    watch_s: float | None = None,
    read_stp: Callable[[], int | None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> ExecuteResult:
    """2+2：先稳住 hold_s，再盯 watch_s。只认仅设施名且栈无「设施结果」。仍在走不做 2+2。"""
    import time

    from mm_assist.life_execute.ui_logic import skip_wait_ui_params

    if read_stp is not None:
        try:
            if int(read_stp() or 0) == 1:
                return ExecuteResult(ok=False, status="stp_busy", message="仍在走，不做2+2")
        except (TypeError, ValueError):
            pass

    def _settled() -> bool:
        return inspect_facility_only_zh() and not stack_has_facility_result()

    sleeper = sleep_fn or time.sleep
    clock = now_fn or time.time
    hold = HOLD_SETTLE_S if hold_s is None else float(hold_s)
    watch = HOLD_WATCH_S if watch_s is None else float(watch_s)
    skip_clock = skip_wait_ui_params() and sleep_fn is None and now_fn is None
    if skip_clock:
        if _settled():
            return ExecuteResult(ok=True, status="done", message="设施页已稳住")
        return ExecuteResult(ok=False, status="settle_fail", message="未稳住仅设施名")

    if on_progress:
        on_progress("稳住设施页")
    if hold > 0:
        sleeper(hold)
    deadline = float(clock()) + watch
    while float(clock()) < deadline:
        if pump_aborted():
            return _cancel_res()
        if not inspect_facility_only_zh():
            return ExecuteResult(ok=False, status="settle_fail", message="2+2时界面不是仅设施名")
        if stack_has_facility_result():
            return ExecuteResult(ok=False, status="settle_fail", message="2+2时仍有设施结果")
        sleeper(0.15)
    if not _settled():
        return ExecuteResult(ok=False, status="settle_fail", message="未稳住仅设施名")
    return ExecuteResult(ok=True, status="done", message="设施页已稳住")


def _click_popup_and_verify(
    click: Callable[[int, int], bool], xy: tuple[int, int], label: str, *,
    stage: str, popup_names: frozenset[str], miss_status: str, miss_message: str,
    on_progress: Callable[[str], None] | None = None, root: Any = None,
) -> ExecuteResult | None:
    """Release only this stage after a fresh post-dispatch stack proves closure."""
    import time

    from mm_assist.safety.autofarm_gates import apply_stage_verify

    dispatched_at: list[float] = []
    err = click_spot(click, xy, label, on_progress=on_progress, root=root,
        stage=stage, on_dispatch=lambda: dispatched_at.append(time.time()))
    if err is not None:
        return err
    if pump_aborted():
        return _cancel_res()
    view = read_state_ui(root=root)
    raw = view.raw if view is not None and isinstance(view.raw, dict) else {}
    names = raw.get('ui_stack_names')
    try:
        stack_ts = float(raw.get('ui_stack_ts') or 0)
    except (TypeError, ValueError):
        stack_ts = 0
    if not (view is not None and view.fresh and view.pid_alive
            and raw.get('ui_stack_ready') and dispatched_at
            and stack_ts > dispatched_at[-1] and isinstance(names, (list, tuple))
            and (names or raw.get('ui_stack_empty_ok'))):
        return ExecuteResult(False, 'popup_unverified', '点后未取得新鲜就绪栈，不记弹窗关闭')
    if popup_names.intersection(names):
        return ExecuteResult(False, miss_status, miss_message)
    apply_stage_verify(stage, True)
    return None


def dismiss_facility_result(
    click: Callable[[int, int], bool],
    grab: Callable[[], Any],
    *,
    on_progress: Callable[[str], None] | None = None,
    root: Any = None,
    require: bool = True,
    present: bool | None = None,
) -> ExecuteResult | None:
    """栈含「设施结果」时点绿色「確認」。消失=成功；「界面：」不变不算故障。无命中不盲点。"""
    if not (stack_has_facility_result() if present is None else present):
        return None
    from mm_assist.life_execute.ui_logic import recognized_click_xy
    from mm_assist.perception.process_sense import locate_result_confirm
    from mm_assist.life_plan.game_stop import located_result_input, network_wait_seconds
    import time

    network_revision = network_wait_seconds()
    captured_at = time.time()
    frame = grab()
    hit = locate_result_confirm(frame, root=root)
    xy = recognized_click_xy(hit) if hit.get("ok") else None
    if xy is None:
        if require:
            return ExecuteResult(ok=False, status="no_confirm_ui", message="有设施结果但看不见確認，不盲点")
        return None
    with located_result_input(frame,xy,captured_at=captured_at,network_revision=network_revision):
        return _click_popup_and_verify(
            click,
            xy,
            "確認",
            on_progress=on_progress,
            root=root,
            stage="設施結果確認",
            popup_names=frozenset({'FacilityResultUIInstance'}),
            miss_status='confirm_miss', miss_message='点確認后仍有设施结果',
        )


def confirm_go_facility_popup(
    click: Callable[[int, int], bool],
    grab: Callable[[], Any],
    *,
    on_progress: Callable[[str], None] | None = None,
    root: Any = None,
    names: list[str] | tuple[str, ...] | None = None,
    present: bool | None = None,
) -> ExecuteResult | None:
    """前往弹窗一律点「確認」，不限定目的地；导航与投领共用。"""
    if present is None:
        present = stack_has_common_popup(names=names)
    if not present:
        return None
    from mm_assist.life_execute.ui_logic import recognized_click_xy
    from mm_assist.perception.process_sense import locate_common_popup_confirm

    frame = grab()
    hit = locate_common_popup_confirm(frame, root=root)
    xy = recognized_click_xy(hit) if hit.get("ok") else None
    if xy is None:
        return ExecuteResult(ok=False, status="no_confirm_ui", message="有确认弹窗但看不见確認，不盲点")
    from mm_assist.memory.readers.ui_stack import COMMON_POPUP_NAMES
    from mm_assist.safety.input import facility_confirm_input_scope

    def confirm_click(x: int, y: int) -> bool:
        with facility_confirm_input_scope(xy, (frame.shape[1], frame.shape[0])):
            return click(x, y)

    return _click_popup_and_verify(
        confirm_click,
        xy,
        "確認",
        on_progress=on_progress,
        root=root,
        stage="前往设施确认",
        popup_names=COMMON_POPUP_NAMES,
        miss_status='confirm_miss', miss_message='点確認后仍有确认弹窗',
    )


def wait_inspect_menu(
    *,
    timeout_s: float = WAIT_MENU_INSPECT_S,
    sleep_fn: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    """点菜单后等到栏上变成菜单。pytest 下一拍即判，不干等。"""
    import time

    from mm_assist.life_execute.ui_logic import skip_wait_ui_params

    if inspect_is_menu():
        return True
    if skip_wait_ui_params():
        return False
    sleeper = sleep_fn or time.sleep
    deadline = operation_wall_time() + float(timeout_s)
    while operation_wall_time() < deadline:
        if cancelled is not None and cancelled():
            return False
        if inspect_is_menu():
            return True
        sleeper(0.15)
    return inspect_is_menu()


def page_zh(node: str) -> str:
    key = str(node or "").strip()
    if key in _PAGE_ZH:
        return _PAGE_ZH[key]
    if key.startswith("item_info:"):
        return "物品"
    if key.startswith("obtain_method:"):
        return "获得方法"
    if key.startswith("facility"):
        return "设施"
    return key or "?"


def format_where_click(where: UiWhere | None, step: str = "") -> str:
    """计划步骤上的点选。不用认页结果。"""
    return format_act("", where, step)


def format_act(title: str, where: UiWhere | None = None, step: str = "") -> str:
    """当前动作 = 计划 title + 要点的控件。不跑物品/OCR 认页。"""
    _ = where
    t = str(title or "").strip()
    s = str(step or "").strip()
    if " | " in s:
        s = s.split(" | ", 1)[-1].strip()
    if not s or s == "认页" or s.startswith("已在"):
        return t
    if (
        s.startswith("等待")
        or s.startswith("等")
        or s.startswith("下翻")
        or s.startswith("离开")
        or s.startswith("进入")
        or s.startswith("退出")
        or s.startswith("已脱困")
        or s.startswith("失败")
        or s.startswith("导航失败")
        or s.startswith("切加工")
        or s.startswith("脱困")
        or s.startswith("再执行")
        or s.startswith("准备中")
        or "本拍不点" in s
    ):
        return f"{t} {s}".strip()
    click = {
        "打开菜单": "点菜单",
        "菜单": "点菜单",
        "打开製作": "点製作",
        "打开制作": "点製作",
        "製作": "点製作",
        "制作": "点製作",
        "打开角色": "点角色",
        "角色": "点角色",
        "打开生活技能": "点生活技能",
        "生活技能 Tab": "点生活技能",
        "六卡正式名": "点加工卡",
        "技能大格": "点技能",
        "关闭详情": "点关闭",
        "关闭/X": "点X",
        "右上 X": "点X",
        "左上 ←": "点返回",
    }.get(s)
    if click is None:
        click = s if s.startswith("点") else f"点{s}"
    return f"{t} {click}".strip()


def wrap_progress(
    on_progress: Callable[[str], None] | None,
    *,
    resolve: Callable[..., Any] | None = None,
    title: str = "",
) -> Callable[[str], None] | None:
    _ = resolve
    if on_progress is None:
        return None

    def _p(msg: str) -> None:
        on_progress(format_act(title, None, str(msg or "")))

    return _p


def skill_id_for(skill: str) -> str:
    return _SKILL_ID.get(str(skill or "").strip(), "")


def gate_ui(where: UiWhere, *, allow_dialog: bool = False) -> ExecuteResult | None:
    if where.menu not in (0, 1):
        return ExecuteResult(
            ok=False,
            status="ui_unread",
            message=f"界面参数未读，不点（{where.bits()}）",
        )
    if where.node == "unknown":
        if inspect_is_true_hud() or inspect_is_menu():
            return None
        overlay = where.menu == 1 and not stack_inv(list(where.stack_names or ()))
        return ExecuteResult(
            ok=False,
            status="unknown_overlay" if overlay else "unknown_ui",
            message=f"未认出界面，不点（{where.bits()}）",
        )
    if where.node == "dialog" and not allow_dialog:
        return ExecuteResult(
            ok=False,
            status="dialog_block",
            message=f"对话未关，不点（{where.bits()}）",
        )
    return None


def pump_aborted() -> bool:
    """停止操作已置位：当前点击链立刻停。"""
    from mm_assist.life_plan.game_stop import game_stop_interrupted
    if game_stop_interrupted():
        return True
    try:
        from mm_assist.tool_repair import op_cancelled

        if op_cancelled():
            return True
    except Exception:
        pass
    from mm_assist.life_plan.gather_repair import gathering_interrupted
    return gathering_interrupted()


def _cancel_res() -> ExecuteResult:
    return ExecuteResult(ok=False, status="cancel", message="已停止操作")


def click_spot(
    click: Callable[[int, int], bool],
    xy: tuple[int, int] | None,
    label: str,
    *,
    on_progress: Callable[[str], None] | None = None,
    wait_refresh: bool = True,
    root: Any = None,
    stage: str = "",
    on_dispatch: Callable[[], None] | None = None,
) -> ExecuteResult | None:
    from mm_assist.life_plan.game_stop import network_wait_seconds, network_input_unchanged
    if pump_aborted():
        return _cancel_res()
    if (xy is not None and network_wait_seconds() > 0
            and not network_input_unchanged(xy)):
        return ExecuteResult(False,'input_layer_wait','网络等待后目标已变化，重新认页定位后继续')
    if xy is None:
        return ExecuteResult(ok=False, status=label, message=f"找不到{label}，不点")
    def dispatch(x: int,y: int) -> bool:
        if on_progress:
            on_progress(label)
        return click(x,y)

    if stage:
        from mm_assist.safety.autofarm_gates import stage_click

        sent = stage_click(stage,dispatch,int(xy[0]),int(xy[1]))
        if sent is None:
            return ExecuteResult(
                ok=False,
                status="need_verify",
                message=f"{stage}已点两次，需先验画面",
            )
    else:
        sent = dispatch(int(xy[0]),int(xy[1]))
    if not sent:
        if pump_aborted():
            return _cancel_res()
        return ExecuteResult(ok=False, status="click_yield", message=f"{label}未点出")
    if on_dispatch is not None:
        on_dispatch()
    if wait_refresh:
        from mm_assist.life_execute.ui_logic import wait_ui_params_refresh

        if on_progress:
            on_progress("等界面刷新")
        if not wait_ui_params_refresh(root=root, cancelled=pump_aborted):
            if pump_aborted():
                return _cancel_res()
            return ExecuteResult(
                ok=False,
                status="stale_ui",
                message="点后界面参数未刷新，不继续",
            )
    return None


def click_hit(
    click: Callable[[int, int], bool],
    hit: Any,
    label: str,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> ExecuteResult | None:
    return click_spot(click, recognized_click_xy(hit), label, on_progress=on_progress)


def nav_target(item: PlanItem) -> tuple[str, str, str]:
    """返回 (ensure_node 目标, skill_id, facility)。"""
    kind = str(item.kind or "")
    if kind == "repair":
        return "main_hud", "", ""
    if kind == "gather":
        skill = skill_id_for(str(item.skill or "").strip())
        return "gather_skill", skill, ""
    if kind in {"feed", "claim"}:
        return "facility", "", str(item.facility or "").strip()
    return "", "", ""


_OVERLAY = frozenset(
    {
        "system_menu",
        "profile",
        "life_grid",
        "gather_skill",
        "gather_detail",
        "craft_hub",
        "facility",
        "facility_recipe",
        "currency_wallet",
        "map_dunbarton",
        "map_dunbarton_nerys",
    }
)


def have_layer(where: UiWhere) -> str:
    names = list(where.stack_names or ())
    if names and _dungeon_overlay_names(names):
        return "overlay"
    if names and stack_inv(names):
        return "inv"
    if graph_node(str(getattr(where, "node", "") or "")) == "inventory":
        return "inv"
    if where.dlg == 1:
        return "dlg"
    if where.menu == 0:
        return "hud"
    return "overlay"


def want_layer(target: str) -> str:
    t = str(target or "").strip()
    if t == "main_hud":
        return "hud"
    if t == "inventory":
        return "inv"
    if t == "dialog":
        return "dlg"
    return "overlay"


def _dungeon_overlay_names(names: list[str] | tuple[str, ...] | None) -> bool:
    return any("DungeonList" in n or "DungeonEntry" in n for n in (names or ()))


def layer_aligned(where: UiWhere, target: str) -> bool:
    if have_layer(where) != want_layer(target):
        return False
    node = str(getattr(where, "node", "") or "")
    if have_layer(where) == "overlay" and node == "unknown":
        return False
    if _dungeon_overlay_names(getattr(where, "stack_names", ())):
        t = graph_node(target)
        return t.startswith("stack:") and t == graph_node(node)
    return True


def _from_node(where: UiWhere) -> str:
    have = have_layer(where)
    node = str(getattr(where, "node", "") or "")
    if have == "inv":
        return "inventory"
    if have == "dlg":
        return "dialog"
    if have == "hud":
        return "main_hud"
    g = graph_node(node)
    if node in _OVERLAY or g in _OVERLAY:
        return g if g in _OVERLAY else node
    if str(node).startswith("stack:CashShop"):
        return "stack:CashShopUIState"
    if _dungeon_overlay_names(getattr(where, "stack_names", ())) or str(node).startswith("stack:"):
        return "overlay"
    return "unknown"


def prepare_to(
    target: str,
    *,
    grab: Callable[[], Any],
    click: Callable[[int, int], bool],
    on_progress: Callable[[str], None] | None = None,
    root: Any = None,
    skill_id: str = "",
    facility: str = "",
    item_name: str = "",
    title: str = "",
    allow_dialog: bool = False,
    read_stp: Callable[[], int | None] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    resolve: Callable[..., Any] | None = None,
    already_ok: Callable[[Any], bool] | None = None,
    leave_wrong_page: Callable[[Any], bool] | None = None,
) -> ExecuteResult:
    """正式导航统一沿当前页面走允许边；注入观测器保留兼容测试路径。"""
    import time

    from mm_assist.life_execute.ui_nav import (
        VERIFY_POLL_S,
        VERIFY_TOTAL_S,
        edge_stage,
        ensure_node,
        locate_edge,
        path_edges,
        note_edge_clicked,
        _verify_late_edges,
    )

    sleeper = sleep_fn or time.sleep
    from mm_assist.life_execute.ui_logic import resolve_navigation

    def _resolver(frame: Any = None) -> UiWhere:
        where = resolve_navigation(frame)
        return promote_true_hud(where)

    resolver = resolve if resolve is not None else _resolver

    def _prog(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    def _stp() -> int | None:
        if read_stp is not None:
            return read_stp()
        from mm_assist.perception.state_ui import read_state_ui

        view = read_state_ui(root=root)
        if view is None:
            return None
        try:
            return int(view.stp) if view.stp is not None else None
        except (TypeError, ValueError):
            return None

    def _align() -> ExecuteResult:
        if resolve is None:
            if pump_aborted():
                return _cancel_res()
            try:
                from mm_assist.life_execute.repair_dialog import (
                    dismiss_world_npc_menu,
                    npc_end_dialog_visible,
                    npc_goodbye_visible,
                    wait_hud_after_npc_dismiss,
                )

                fr0 = grab()
                if allow_dialog and npc_end_dialog_visible(fr0):
                    return ExecuteResult(True, 'at_node', 'dialog')
                if not allow_dialog and npc_end_dialog_visible(fr0):
                    if not dismiss_world_npc_menu(click, fr0, grab=grab, sleep_fn=sleeper):
                        return (_cancel_res() if pump_aborted() else
                                ExecuteResult(False, 'npc_end_dialog_unverified', '結束對話后未确认主界面，不点其它入口'))
                    _prog("关闭NPC对话")
                elif npc_goodbye_visible(fr0):
                    # Repair may continue in an entry dialog, but a goodbye
                    # must be dismissed before it can start map navigation.
                    if not wait_hud_after_npc_dismiss(click, grab, sleep_fn=sleeper):
                        return (_cancel_res() if pump_aborted() else
                                ExecuteResult(False, 'npc_goodbye_unverified', '告别画面未关闭，不点其它入口'))
                    _prog("关闭NPC告别")
            except Exception as exc:
                return (_cancel_res() if pump_aborted() else
                        ExecuteResult(False, 'npc_dismiss_error', str(exc)))
            deadline = operation_monotonic() + VERIFY_TOTAL_S
            where = resolver(grab())
            while (
                where.node == "unknown"
                and not inspect_is_hud_idle()
                and operation_monotonic() < deadline
            ):
                if pump_aborted():
                    return _cancel_res()
                sleeper(VERIFY_POLL_S)
                where = resolver(grab())
            names = list(getattr(where, "stack_names", ()) or ())
            if hud_page_pending(where):
                return ExecuteResult(False, "input_layer_wait", "等待HUD页面就绪，不关闭未识别画面")
            if stack_has_common_popup(names=names or None):
                go = confirm_go_facility_popup(
                    click, grab, on_progress=_prog, root=root, names=names or None
                )
                if go is not None:
                    return go
                where = resolver(grab())
            if where.node == "unknown":
                return ExecuteResult(False, "unknown_ui", "当前页面未确认，不点")
            if where.node == "dialog":
                return (ExecuteResult(True, "at_node", "dialog") if allow_dialog else
                        ExecuteResult(False, "dialog_block", "对话未关，不点"))
            if have_layer(where) == "hud" and want_layer(target) == "overlay" and _stp() == 1:
                _prog("等待停钮")
                wait_s = WAIT_STP_S
                try:
                    from mm_assist.safety.lease import lease_remaining_s

                    remain = float(lease_remaining_s() or 0.0)
                except Exception:
                    remain = 0.0
                if remain > 0:
                    wait_s = min(wait_s, max(0.0, remain - float(LEASE_CLICK_RESERVE_S)))
                if wait_s <= 0:
                    return ExecuteResult(False, "stp_busy", "停钮一直亮，不点")
                deadline = operation_wall_time() + wait_s
                while operation_wall_time() < deadline:
                    if pump_aborted():
                        return _cancel_res()
                    cur = _stp()
                    if cur is None or cur == 0:
                        where = resolver(grab())
                        break
                    sleeper(1.0)
                else:
                    return ExecuteResult(False, "stp_busy", "停钮一直亮，不点")
            return ExecuteResult(True, "layer_ok", have_layer(where))
        want = want_layer(target)
        clicked_menu = False
        for _ in range(8):
            if pump_aborted():
                return _cancel_res()
            frame_started_at = time.time()
            frame = grab()
            where = resolver(frame)
            if hud_page_pending(where):
                return ExecuteResult(False, "input_layer_wait", "等待HUD页面就绪，不关闭未识别画面")
            _verify_late_edges(where, frame_started_at=frame_started_at)
            names = list(getattr(where, "stack_names", ()) or ())
            if stack_has_common_popup(names=names or None):
                go = confirm_go_facility_popup(
                    click, grab, on_progress=_prog, root=root, names=names or None
                )
                if go is not None:
                    return go
                continue
            if inspect_is_menu() and want == "overlay":
                return ExecuteResult(ok=True, status="layer_ok", message="overlay")
            err = gate_ui(where, allow_dialog=allow_dialog)
            if err is not None:
                return err
            if allow_dialog and str(getattr(where, "node", "") or "") == "dialog":
                return ExecuteResult(ok=True, status="at_node", message="dialog")
            have = have_layer(where)
            stp_now = _stp() if have == "hud" else 0
            if have == "hud" and want == "overlay" and stp_now == 1:
                _prog("等待停钮")
                deadline = operation_wall_time() + WAIT_STP_S
                while operation_wall_time() < deadline:
                    if pump_aborted():
                        return _cancel_res()
                    cur = _stp()
                    if cur is None or cur == 0:
                        break
                    sleeper(1.0)
                else:
                    return ExecuteResult(ok=False, status="stp_busy", message="停钮一直亮，不点")
                continue
            if have == "hud" and want == "inv":
                # HUD is the bag's navigation origin, not another layer to
                # close. Let ensure_node follow and verify the registered
                # entry path; asking for a HUD->HUD exit has no graph edge.
                if not where.stack_ready:
                    return ExecuteResult(False, "unknown_ui", "当前栈未就绪，不进入背包")
                return ExecuteResult(ok=True, status="layer_ok", message=have)
            if layer_aligned(where, target):
                return ExecuteResult(ok=True, status="layer_ok", message=have)
            if have == "dlg" and not allow_dialog:
                _prog("退出对话")
                return ExecuteResult(
                    ok=False,
                    status="dialog_block",
                    message="对话未关，先退出再进指定界面",
                )
            dest = "system_menu" if have == "hud" and want == "overlay" else "main_hud"
            if have == "hud" and want == "overlay":
                _prog("进入菜单")
            elif have == "inv":
                _prog("退出背包")
            else:
                _prog("退出叠层")
            src = _from_node(where)
            if not src or src == "unknown":
                return ExecuteResult(
                    ok=False,
                    status="unknown_overlay",
                    message=f"未认出叠层，不点（{where.bits()}）",
                )
            hops = path_edges(src, dest, root=root)
            if not hops:
                return ExecuteResult(
                    ok=False,
                    status="no_path",
                    message=f"{src}→{dest} 无进入/退出边",
                )
            edge = hops[0]
            label = str((edge.get("trigger") or {}).get("label") or edge.get("id") or "切界面")
            from mm_assist.life_execute.ui_logic import is_open_menu_action, skip_click

            if skip_click(src, label, menu=getattr(where, "menu", None)):
                continue
            if is_open_menu_action(label) and clicked_menu:
                # 菜单已点开，四参滞后时再点会打到角色/製作/背包。
                continue
            xy = locate_edge(edge, frame, root=root, skill_id=skill_id, facility=facility)
            stage = edge_stage(edge, label=label)
            open_menu = is_open_menu_action(label)
            err = click_spot(
                click,
                xy,
                label,
                on_progress=None,
                wait_refresh=False,
                root=root,
                stage=stage,
                on_dispatch=lambda: note_edge_clicked(
                    str(edge.get("id") or ""), str(edge.get("to") or ""), stage=stage
                ),
            )
            if err is None and open_menu:
                clicked_menu = True
                _prog("进入菜单")
                if not wait_inspect_menu(
                    sleep_fn=sleeper if sleep_fn is not None else None,
                    cancelled=pump_aborted,
                ):
                    if pump_aborted():
                        return _cancel_res()
                    return ExecuteResult(
                        ok=False,
                        status="stale_ui",
                        message="点菜单后界面未变成菜单",
                    )
                return ExecuteResult(ok=True, status="layer_ok", message="overlay")
            if err is not None:
                return err
            if sleep_fn is None:
                from mm_assist.life_execute.ui_logic import wait_ui_params_refresh

                _prog("等界面刷新")
                if not wait_ui_params_refresh(root=root, cancelled=pump_aborted):
                    if pump_aborted():
                        return _cancel_res()
                    return ExecuteResult(
                        ok=False,
                        status="stale_ui",
                        message="点后界面参数未刷新，不继续",
                    )
            else:
                sleeper(1.0)
        return ExecuteResult(ok=False, status="layer_timeout", message="四参进退未到达指定层")

    aligned = _align()
    if not aligned.ok:
        return aligned
    if allow_dialog and aligned.message == "dialog":
        return aligned

    already = already_ok

    return ensure_node(
        target,
        grab=grab,
        click=click,
        on_progress=on_progress,
        root=root,
        skill_id=skill_id,
        facility=facility,
        **({'item_name': item_name} if item_name else {}),
        already_ok=already,
        leave_wrong_page=leave_wrong_page,
        sleep_fn=sleep_fn,
        resolve=resolver,
    )
