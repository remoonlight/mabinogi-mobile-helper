from __future__ import annotations

import pytest
import time
from types import SimpleNamespace

from mm_assist.life_execute.ui_act import (
    click_spot,
    confirm_go_facility_popup,
    format_act,
    gate_ui,
    have_layer,
    hold_facility_settle,
    inspect_facility_only_zh,
    layer_aligned,
    nav_target,
    prepare_to,
    stack_has_common_popup,
    stack_has_facility_result,
    _from_node,
)
from mm_assist.life_execute.ui_logic import UiWhere
from mm_assist.life_plan.types import ExecuteResult, PlanItem
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests


@pytest.fixture(autouse=True)
def _reset_stage_gates():
    from mm_assist.life_execute.ui_nav import reset_edge_memo_for_tests

    reset_stage_gates_for_tests()
    reset_edge_memo_for_tests()
    yield
    reset_stage_gates_for_tests()
    reset_edge_memo_for_tests()


def test_gate_ui_unread_and_unknown(monkeypatch):
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_true_hud", lambda: False)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_hud_idle", lambda: False)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_menu", lambda: False)
    unread = UiWhere("main_hud", None, 0, 0, "icon_only")
    err = gate_ui(unread)
    assert err is not None and err.status == "ui_unread"
    unknown = UiWhere("unknown", 1, 0, 0, "mem")
    err = gate_ui(unknown)
    assert err is not None
    assert err.status in {"unknown_ui", "unknown_overlay"}
    unknown_hud = UiWhere("unknown", 0, 0, 0, "mem")
    err = gate_ui(unknown_hud)
    assert err is not None and err.status == "unknown_ui"
    dlg = UiWhere("dialog", 1, 1, 0, "mem")
    assert gate_ui(dlg) is not None
    assert gate_ui(dlg, allow_dialog=True) is None


def test_gate_ui_unknown_trusts_inspect_idle(monkeypatch):
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_true_hud", lambda: True)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_menu", lambda: False)
    err = gate_ui(UiWhere("unknown", 0, 0, 0, "mem"))
    assert err is None


def test_click_spot_refuses_without_xy():
    clicks: list[tuple[int, int]] = []
    err = click_spot(lambda x, y: clicks.append((x, y)) or True, None, "获得方法")
    assert err is not None
    assert "不点" in err.message
    assert clicks == []
    assert click_spot(lambda x, y: clicks.append((x, y)) or True, (3, 4), "点A") is None
    assert clicks == [(3, 4)]


def test_click_spot_stops_when_cancelled(monkeypatch):
    from mm_assist.life_execute.ui_act import click_spot
    from mm_assist.tool_repair import clear_op_cancel, request_op_cancel

    request_op_cancel()
    try:
        clicks: list[tuple[int, int]] = []
        err = click_spot(lambda x, y: clicks.append((x, y)) or True, (8, 9), "菜单")
        assert err is not None
        assert err.status == "cancel"
        assert clicks == []
    finally:
        clear_op_cancel()


def test_nav_target_wired_kinds():
    assert nav_target(PlanItem(id="p1", kind="repair", title="修")) == ("main_hud", "", "")
    g = PlanItem(
        id="p1",
        kind="gather",
        title="采集 水井",
        skill="日常採集",
        detail="水井",
    )
    assert nav_target(g) == ("gather_skill", "daily", "")
    feed = PlanItem(
        id="p1",
        kind="feed",
        title="投料 合金鋼錠",
        facility="metal",
        recipe="合金鋼錠",
    )
    assert nav_target(feed) == ("facility", "", "metal")


def test_prepare_waits_stp_on_hud_then_overlay(monkeypatch):
    import time as time_mod
    from mm_assist.life_plan.types import ExecuteResult

    clicks: list[tuple[int, int]] = []
    now = [0.0]
    stp = {"n": 1}
    monkeypatch.setattr("mm_assist.life_execute.ui_act.WAIT_STP_S", 3.0)
    monkeypatch.setattr(time_mod, "time", lambda: now[0])
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_navigation",
        lambda _f=None: UiWhere("main_hud", 0, 0, 0, "direct"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.ensure_node",
        lambda *a, **k: ExecuteResult(True, "at_node", "facility"),
    )

    def sleep(s: float) -> None:
        now[0] += float(s)
        stp["n"] = 0

    r = prepare_to(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=sleep,
        read_stp=lambda: stp["n"],
        facility="cloth",
    )
    assert r.ok is True
    assert clicks == []
    assert now[0] >= 1.0


def test_prepare_waits_for_pending_unknown_then_proceeds_without_click(monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult

    clicks: list[tuple[int, int]] = []
    rows = iter(
        [
            UiWhere("unknown", None, None, None, "direct_pending"),
            UiWhere("main_hud", 0, 0, 0, "direct"),
        ]
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_navigation",
        lambda _f=None: next(rows),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.ensure_node",
        lambda *a, **k: ExecuteResult(True, "at_node", "facility"),
    )
    r = prepare_to(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda _s: None,
        read_stp=lambda: 0,
        facility="metal",
    )
    assert r.ok is True
    assert clicks == []


def test_prepare_pending_unknown_timeout_does_not_click(monkeypatch):
    import time as time_mod

    clicks: list[tuple[int, int]] = []
    now = [0.0]
    monkeypatch.setattr(time_mod, "monotonic", lambda: now[0])
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_hud_idle", lambda: False)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_true_hud", lambda: False)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_navigation",
        lambda _f=None: UiWhere("unknown", None, None, None, "direct_pending"),
    )
    r = prepare_to(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda s: now.__setitem__(0, now[0] + float(s)),
        read_stp=lambda: 0,
        facility="metal",
    )
    assert r.ok is False
    assert r.status == "unknown_ui"
    assert clicks == []


def test_prepare_true_hud_stack_ready_not_unknown_ui(monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult

    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_hud_idle", lambda: False)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_true_hud", lambda: False)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_navigation",
        lambda _f=None: UiWhere(
            "unknown", 0, 0, 0, "direct", stack_names=(), stack_ready=True
        ),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.ensure_node",
        lambda *a, **k: ExecuteResult(True, "at_node", "gather_skill"),
    )
    r = prepare_to(
        "gather_skill",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda _s: None,
        read_stp=lambda: 0,
        skill_id="daily",
    )
    assert r.ok is True
    assert clicks == []


def test_prepare_inspect_idle_not_unknown_ui(monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult

    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_true_hud", lambda: True)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_navigation",
        lambda _f=None: UiWhere("unknown", None, None, None, "direct_pending"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.ensure_node",
        lambda *a, **k: ExecuteResult(True, "at_node", "facility"),
    )
    r = prepare_to(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda _s: None,
        read_stp=lambda: 0,
        facility="metal",
    )
    assert r.ok is True
    assert clicks == []


def test_prepare_live_inspect_hud_while_stack_scan_pending(monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult

    rec = {
        "node": "main_hud",
        "zh": "主界面-待机",
        "stack_names": [],
        "stack_ready": True,
        "source": "stack+tpl",
    }
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **_k: rec,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_navigation",
        lambda _f=None: UiWhere("unknown", None, None, None, "direct_pending"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.ensure_node",
        lambda *a, **k: ExecuteResult(True, "at_node", "gather_skill"),
    )
    r = prepare_to(
        "gather_skill",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda _s: None,
        read_stp=lambda: 0,
        skill_id="daily",
    )
    assert r.ok is True
    assert r.status != "unknown_ui"
    assert clicks == []


def test_prepare_to_unknown_overlay_exits_not_craft(monkeypatch):
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.locate_edge",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: False,
    )
    r = prepare_to(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda _s: None,
        resolve=lambda _f=None: UiWhere("unknown", 1, 0, 0, "mem"),
        title="投料 藥品",
    )
    assert r.ok is False
    assert clicks == []
    assert r.status != "at_node"


def test_prepare_menu_not_clicked_twice_when_params_lag(monkeypatch):
    """点开菜单后四参仍写 main_hud 时，禁止再点（会打到角色/製作）。"""
    clicks: list[tuple[int, int]] = []
    n = {"i": 0}

    def resolve(_f=None):
        n["i"] += 1
        if n["i"] == 1:
            return UiWhere("main_hud", 0, 0, 0, "mem")
        return UiWhere("main_hud", 0, 0, 0, "mem")

    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.locate_edge",
        lambda *a, **k: (11, 22),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "main_hud", "zh": "主界面-待机"},
    )
    r = prepare_to(
        "gather_detail",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda _s: None,
        read_stp=lambda: 0,
        resolve=resolve,
        already_ok=lambda _f: False,
        title="采集 銅礦石",
    )
    assert clicks.count((11, 22)) == 1
    assert r.ok is False
    assert r.status == "stale_ui"


def test_prepare_hud_enters_menu_before_craft(monkeypatch):
    clicks: list[tuple[int, int]] = []
    n = {"i": 0}

    def resolve(_f=None):
        n["i"] += 1
        if n["i"] == 1:
            return UiWhere("main_hud", 0, 0, 0, "mem")
        return UiWhere("system_menu", 1, 0, 0, "mem")

    def inspect_rec(**_k):
        if clicks:
            return {"node": "system_menu", "zh": "菜单"}
        return {"node": "main_hud", "zh": "主界面-待机"}

    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.locate_edge",
        lambda *a, **k: (11, 22),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        inspect_rec,
    )
    r = prepare_to(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda _s: None,
        read_stp=lambda: 0,
        resolve=resolve,
        already_ok=lambda _f: False,
        title="投料 鋼錠",
    )
    assert clicks
    assert clicks[0] == (11, 22)


def test_prepare_inv_exits_bag(monkeypatch):
    clicks: list[tuple[int, int]] = []
    n = {"i": 0}

    def resolve(_f=None):
        n["i"] += 1
        if n["i"] == 1:
            return UiWhere("inventory", 1, 0, 0, "mem")
        return UiWhere("system_menu", 1, 0, 0, "mem")

    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.locate_edge",
        lambda *a, **k: (4, 5),
    )
    seen: list[str] = []
    prepare_to(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda _s: None,
        resolve=resolve,
        already_ok=lambda _f: False,
        on_progress=seen.append,
        title="投料 鋼錠",
    )
    assert clicks[0] == (4, 5)
    assert any("退出背包" in s for s in seen)


def test_format_act_enter_exit():
    assert format_act("投料 鋼錠", None, "进入菜单") == "投料 鋼錠 进入菜单"
    assert format_act("投料 鋼錠", None, "退出背包") == "投料 鋼錠 退出背包"
    assert format_act("投料 鋼錠", None, "等界面刷新") == "投料 鋼錠 等界面刷新"
    assert format_act("投料 鋼錠", None, "等新页") == "投料 鋼錠 等新页"
    assert format_act("采集 銅礦石", None, "等界面/找不到入口，本拍不点") == (
        "采集 銅礦石 等界面/找不到入口，本拍不点"
    )


def test_format_act_follows_plan_not_page():
    where = UiWhere("inventory", 1, 0, 0, "mem")
    text = format_act("投料 鋼錠", where, "认页")
    assert text == "投料 鋼錠"
    assert "inv=" not in text
    assert format_act("投料 鋼錠", where, "製作") == "投料 鋼錠 点製作"


def test_wrap_progress_plan_and_click():
    from mm_assist.life_execute.ui_act import wrap_progress

    seen: list[str] = []
    fn = wrap_progress(seen.append, title="投料 鋼錠")
    assert fn is not None
    fn("製作")
    assert seen == ["投料 鋼錠 点製作"]


def test_unknown_overlay_no_click(monkeypatch):
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.locate_edge",
        lambda *a, **k: (9, 9),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: False,
    )
    r = prepare_to(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda _s: None,
        resolve=lambda _f=None: UiWhere("unknown", 1, 0, 0, "mem"),
        title="投料 藥品",
    )
    assert r.ok is False
    assert r.status == "unknown_overlay"
    assert clicks == []


def test_stage_gate_third_click_blocked():
    from mm_assist.safety.autofarm_gates import allow_stage_action

    assert allow_stage_action("切页") is True
    assert allow_stage_action("切页") is True
    assert allow_stage_action("切页") is False
    reset_stage_gates_for_tests()
    clicks: list[tuple[int, int]] = []

    def do(x, y):
        clicks.append((x, y))
        return True

    assert click_spot(do, (1, 1), "切页", stage="切页", wait_refresh=False) is None
    assert click_spot(do, (2, 2), "切页", stage="切页", wait_refresh=False) is None
    err = click_spot(do, (3, 3), "切页", stage="切页", wait_refresh=False)
    assert err is not None
    assert err.status == "need_verify"
    assert clicks == [(1, 1), (2, 2)]


def test_recover_preserves_unverified_stage_gate(monkeypatch):
    from mm_assist.life_execute.ui_recover import recover_once
    from mm_assist.safety.autofarm_gates import allow_stage_action

    assert allow_stage_action("edge-stuck") is True
    assert allow_stage_action("edge-stuck") is True
    assert allow_stage_action("edge-stuck") is False
    recover_once("nav_verify_timeout")
    assert allow_stage_action("edge-stuck") is False


def test_diagnose_hud_menu_miss_no_calib():
    from mm_assist.safety.autofarm_gates import diagnose_hud_menu_miss

    assert diagnose_hud_menu_miss(peak_score=0.1) == "hud_occluded"
    assert diagnose_hud_menu_miss(peak_score=0.9) == "menu_icon_miss"
    assert isinstance(diagnose_hud_menu_miss(), str)


def test_stack_has_facility_result_spacing():
    assert stack_has_facility_result("设施结果>加工设施") is True
    assert stack_has_facility_result("设施结果 > 加工设施") is True
    assert stack_has_facility_result("加工设施") is False


def test_stack_has_common_popup_names_and_label():
    assert stack_has_common_popup("ButtonCommonPopupInstance > ButtonCommonPopupUIState") is True
    assert stack_has_common_popup("确认弹窗 > 制作") is True
    assert stack_has_common_popup("加工设施") is False


def test_confirm_go_popup_clicks_then_clears(monkeypatch):
    clicks: list[tuple[int, int]] = []
    present = {"n": True}
    monkeypatch.setattr('mm_assist.life_execute.ui_act.read_state_ui', lambda **_: SimpleNamespace(
        fresh=True, pid_alive=True, raw={'ui_stack_ready': True,
        'ui_stack_ts': time.time(), 'ui_stack_names': ['ButtonCommonPopupInstance'] if present['n'] else ['SideNavigationUIInstance']}))

    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: present["n"],
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_common_popup_confirm",
        lambda *a, **k: {"ok": True, "click_client": [1100, 980]},
    )

    def _click(x, y):
        clicks.append((x, y))
        present["n"] = False
        return True

    import numpy as np
    err = confirm_go_facility_popup(_click, lambda: np.zeros((1442, 1912, 3), dtype=np.uint8))
    assert err is None
    assert clicks == [(1100, 980)]


def test_confirm_go_popup_absent_does_not_click(monkeypatch):
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: False,
    )
    err = confirm_go_facility_popup(
        lambda x, y: clicks.append((x, y)) or True,
        lambda: object(),
    )
    assert err is None
    assert clicks == []


def test_prepare_to_common_popup_clicks_confirm(monkeypatch):
    import numpy as np
    clicks: list[tuple[int, int]] = []
    names = (
        "HudUIStepState",
        "CraftingHubUIState",
        "ButtonCommonPopupUIState",
        "ButtonCommonPopupInstance",
    )
    pages = iter(
        [
            UiWhere("overlay", 1, 0, 0, "mem", stack_names=names, stack_ready=True),
            UiWhere("system_menu", 1, 0, 0, "mem"),
        ]
    )
    present = {"n": True}
    monkeypatch.setattr('mm_assist.life_execute.ui_act.read_state_ui', lambda **_: SimpleNamespace(
        fresh=True, pid_alive=True, raw={'ui_stack_ready': True,
        'ui_stack_ts': time.time(), 'ui_stack_names': ['ButtonCommonPopupInstance'] if present['n'] else ['SideNavigationUIInstance']}))

    def _has(*_a, **k):
        if k.get("names"):
            return True
        return present["n"]

    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_common_popup", _has)
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_common_popup_confirm",
        lambda *a, **k: {"ok": True, "click_client": [1100, 980]},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.ensure_node",
        lambda *a, **k: ExecuteResult(True, "at_node", "facility"),
    )

    def _click(x, y):
        clicks.append((x, y))
        present["n"] = False
        return True

    r = prepare_to(
        "facility",
        grab=lambda: np.zeros((1442, 1912, 3), dtype=np.uint8),
        click=_click,
        sleep_fn=lambda _s: None,
        resolve=lambda _f=None: next(pages, UiWhere("system_menu", 1, 0, 0, "mem")),
        title="领取 布料",
    )
    assert clicks[0] == (1100, 980)
    assert r.ok is True


def test_inspect_facility_only_zh_recipe_vs_page():
    assert inspect_facility_only_zh("制作-加工-金属加工") is True
    assert inspect_facility_only_zh("制作-加工-金屬加工") is True
    assert inspect_facility_only_zh("制作-加工-金属加工-合金鋼錠") is False
    assert inspect_facility_only_zh("制作-加工") is False
    assert inspect_facility_only_zh("制作-加工-设施") is True
    assert inspect_facility_only_zh("主界面-待机") is False


def test_hold_facility_settle_injected_clock(monkeypatch):
    t = {"n": 0.0}

    def now() -> float:
        return t["n"]

    def sleep(s: float) -> None:
        t["n"] += float(s)

    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_facility_result",
        lambda *a, **k: False,
    )
    r = hold_facility_settle(sleep_fn=sleep, now_fn=now, hold_s=2.0, watch_s=2.0)
    assert r.ok is True
    assert t["n"] >= 4.0


def test_hold_facility_settle_skips_while_walking():
    r = hold_facility_settle(read_stp=lambda: 1, hold_s=0.0, watch_s=0.0, sleep_fn=lambda *_: None)
    assert r.ok is False
    assert r.status == "stp_busy"


def test_dismiss_confirm_fail_closed(monkeypatch):
    from mm_assist.life_execute.ui_act import dismiss_facility_result

    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_facility_result",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": False},
    )
    err = dismiss_facility_result(
        lambda x, y: clicks.append((x, y)) or True,
        lambda: object(),
    )
    assert err is not None
    assert err.status == "no_confirm_ui"
    assert clicks == []


def test_recover_esc_does_not_click_result_confirm(monkeypatch):
    from mm_assist.life_execute.ui_recover import recover_once
    from mm_assist.life_execute.ui_logic import UiWhere
    import time

    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_facility_result",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": True, "click_client": [900, 1300]},
    )
    keys = []
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    r = recover_once(
        "nav_verify_timeout",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        press_esc=lambda: keys.append(1), sleep_fn=lambda _: None,
        resolve=lambda _: UiWhere('facility', 1, 0, 0, 'fixture', stack_ready=True,
            stack_sampled_ts=time.time(), stack_names=('AlteringResultPopupUIInstance',)),
    )
    assert clicks == [] and keys == [1]
    assert r["action"] == "press_escape" and not r['ok']


def test_recover_esc_does_not_confirm_popup_business_action(monkeypatch):
    from mm_assist.life_execute.ui_recover import recover_once
    from mm_assist.life_execute.ui_logic import UiWhere
    import time

    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_facility_result",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_common_popup_confirm",
        lambda *a, **k: {"ok": True, "click_client": [1100, 980]},
    )
    keys = []
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    r = recover_once(
        "nav_verify_timeout",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        press_esc=lambda: keys.append(1), sleep_fn=lambda _: None,
        resolve=lambda _: UiWhere('facility', 1, 0, 0, 'fixture', stack_ready=True,
            stack_sampled_ts=time.time(), stack_names=('CommonPopupUIInstance',)),
    )
    assert clicks == [] and keys == [1]
    assert r["action"] == "press_escape" and not r['ok']


def test_dispatch_evidence_precedes_refresh_failure(monkeypatch):
    from mm_assist.life_execute.ui_act import click_spot
    events=[]
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.wait_ui_params_refresh',lambda **k:False)
    result=click_spot(lambda x,y:events.append('click') or True,(1,2),'加工',
        on_dispatch=lambda:events.append('dispatch'))
    assert events==['click','dispatch']
    assert result.status=='stale_ui' and not result.ok


def test_have_layer_dungeon_list_is_overlay():
    where = UiWhere(
        "main_hud",
        0,
        0,
        0,
        "mem",
        stack_names=("DungeonListUIInstance", "DungeonEntryUIInstance"),
        stack_ready=True,
    )
    assert have_layer(where) == "overlay"
    assert layer_aligned(where, "gather_skill") is False
    assert _from_node(where) == "overlay"


def test_prepare_to_goodbye_zoom_waits_hud_before_menu(monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult

    waits: list[int] = []
    monkeypatch.setattr(
        "mm_assist.life_execute.repair_dialog.dismiss_world_npc_menu",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.repair_dialog.npc_goodbye_visible",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.repair_dialog.wait_hud_after_npc_dismiss",
        lambda *_a, **_k: waits.append(1) or True,
    )
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_true_hud", lambda: True)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_navigation",
        lambda _f=None: UiWhere("main_hud", 0, 0, 0, "direct"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.ensure_node",
        lambda *a, **k: ExecuteResult(True, "at_node", "system_menu"),
    )
    r = prepare_to(
        "system_menu",
        grab=lambda: object(),
        click=lambda *_a, **_k: True,
        sleep_fn=lambda _s: None,
        read_stp=lambda: 0,
    )
    assert waits == [1]
    assert r.ok is True


def test_prepare_to_unverified_goodbye_cannot_open_next_page(monkeypatch):
    monkeypatch.setattr('mm_assist.life_execute.repair_dialog.dismiss_world_npc_menu', lambda *a, **k: False)
    monkeypatch.setattr('mm_assist.life_execute.repair_dialog.npc_goodbye_visible', lambda *a: True)
    monkeypatch.setattr('mm_assist.life_execute.repair_dialog.wait_hud_after_npc_dismiss', lambda *a, **k: False)
    monkeypatch.setattr('mm_assist.life_execute.ui_nav.ensure_node',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('unverified goodbye blocks next page')))
    out = prepare_to('system_menu', grab=lambda: object(), click=lambda *xy: True)
    assert not out.ok and out.status == 'npc_goodbye_unverified'


def test_prepare_to_goodbye_input_error_cannot_open_next_page(monkeypatch):
    monkeypatch.setattr('mm_assist.life_execute.repair_dialog.dismiss_world_npc_menu', lambda *a, **k: False)
    monkeypatch.setattr('mm_assist.life_execute.repair_dialog.npc_goodbye_visible', lambda *a: True)
    monkeypatch.setattr('mm_assist.life_execute.repair_dialog.wait_hud_after_npc_dismiss',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('click_pointer_moved')))
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation', lambda *a: UiWhere('main_hud', 0, 0, 0, 'offline'))
    monkeypatch.setattr('mm_assist.life_execute.ui_nav.ensure_node',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('rejected input blocks next page')))
    out = prepare_to('system_menu', grab=lambda: object(), click=lambda *xy: True)
    assert not out.ok and out.status == 'npc_dismiss_error'
    assert out.message == 'click_pointer_moved'
