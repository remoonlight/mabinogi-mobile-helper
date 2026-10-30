from __future__ import annotations

import threading
import time

from mm_assist.life_plan.bar_ui import (
    format_life_panel,
    format_life_plan_block,
    format_life_process_block,
    format_life_needs_block,
    format_need_items,
    format_process_status,
    request_stop_repair,
    try_start_one_click_repair,
)
from mm_assist.life_plan.controller import (
    acknowledge_done,
    get_controller,
    pause_life,
    reset_controller_for_tests,
    start_life,
)
from mm_assist.life_plan.execute import try_execute
from mm_assist.life_plan.planner import cycle_desired, plan_life
from mm_assist.life_plan.recipes import display_need_rows, explode_needs
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, QueueSlot, ToolDurability


def _snap(
    items: dict[str, int] | None = None,
    *,
    tools: tuple[ToolDurability, ...] = (),
    metal: tuple[QueueSlot, ...] = (),
    wood: tuple[QueueSlot, ...] = (),
    cloth: tuple[QueueSlot, ...] = (),
    medicine: tuple[QueueSlot, ...] = (),
    ok: bool = True,
    error: str = "",
    taken_at: float = 1.0,
) -> LifeSnapshot:
    return LifeSnapshot(
        ok=ok,
        # 10 finished of 40 leaves the original 30-unit supply scenario.
        items={"增幅蘑菇汁液": 150, "箭花": 250, "血紅藥草": 250,
               "高級混合強化秘藥(山崩)": 10,
               "滴答螢火蟲": 250, "獨角仙": 250, **dict(items or {})},
        tools=tools,
        facilities={"metal": metal, "wood": wood, "cloth": cloth, "medicine": medicine},
        error=error,
        taken_at=taken_at,
    )


def test_cycle_desired_only_contains_general_materials():
    have = {"合金鋼錠": 100, "高級木材": 100, "高級布料": 100}
    assert cycle_desired(have) == {
        "合金鋼錠": 100,
        "高級木材": 100,
        "高級布料": 100,
    }


def test_cycle_desired_stays_at_fixed_target():
    have = {
        "合金鋼錠": 100,
        "高級木材": 100,
        "高級布料": 100,
        "淨淨蘑菇汁液": 150,
    }
    assert cycle_desired(have) == {
        "合金鋼錠": 100,
        "高級木材": 100,
        "高級布料": 100,
    }


def test_plan_caps_at_ten_and_feeds_five():
    items = {
        "合金鋼錠": 0,
        "高級木材": 0,
        "高級布料": 0,
        "鋼錠": 30,
        "銅礦石": 200,
        "煤炭": 80,
    }
    plan = plan_life(_snap(items))
    assert len(plan) == 10
    feeds = [p for p in plan if p.kind == "feed"]
    assert len(feeds) == 5
    assert all(p.recipe == "合金鋼錠" and p.facility == "metal" for p in feeds)
    assert all(p.kind in {"feed", "gather"} for p in plan)
    rest = [p for p in plan if p.kind != "feed"]
    assert rest
    from mm_assist.life_plan.recipes import WIRED_GATHER_DETAILS

    assert all(p.kind == "gather" for p in rest)
    assert any(p.gather_mode == "process_mat" for p in rest)
    assert all(p.gather_mode == "process_mat" for p in rest)
    from mm_assist.life_plan.recipes import SKILL_GATHER
    assert all(p.detail in WIRED_GATHER_DETAILS for p in rest if p.gather_mode == 'process_mat')
    assert all((p.skill, p.detail) in SKILL_GATHER for p in rest if p.gather_mode == 'skill')


def test_repair_is_first_when_durability_below_one():
    plan = plan_life(
        _snap(
            {"合金鋼錠": 0, "高級木材": 80, "高級布料": 0},
            tools=(ToolDurability("十字鎬", 0.0, 100.0),),
        )
    )
    assert plan[0].kind == "repair"
    assert plan[0].title.startswith("一键修理")


def test_claim_when_three_medicine_done():
    med = tuple(QueueSlot("淨淨蘑菇汁液", True) for _ in range(3)) + (
        QueueSlot("淨淨蘑菇汁液", False),
    )
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 0,
                "高級木材": 0,
                "高級布料": 0,
                "鋼錠": 30,
                "銅礦石": 200,
                "煤炭": 80,
            },
            medicine=med,
        )
    )
    claims = [p for p in plan if p.kind == "claim"]
    assert len(claims) == 1
    assert claims[0].facility == "medicine"
    assert claims[0].title == "领取 藥品"
    assert plan[0].kind == "claim"


def test_claim_when_three_cloth_done():
    cloth = tuple(QueueSlot("高級布料", True) for _ in range(3)) + (
        QueueSlot("高級布料", False),
    )
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 100,
                "高級木材": 100,
                "高級布料": 0,
                "布料+": 9,
                "高級羊毛": 45,
                "羊毛": 24,
            },
            cloth=cloth,
        )
    )
    claims = [p for p in plan if p.kind == "claim"]
    assert claims
    assert claims[0].facility == "cloth"
    assert claims[0].title == "领取 布料"
    assert plan[0].kind == "claim"


def test_no_claim_when_two_medicine_done():
    med = (
        QueueSlot("淨淨蘑菇汁液", True),
        QueueSlot("淨淨蘑菇汁液", True),
        QueueSlot("淨淨蘑菇汁液", False),
        QueueSlot("淨淨蘑菇汁液", False),
    )
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 0,
                "高級木材": 0,
                "高級布料": 0,
                "鋼錠": 30,
                "銅礦石": 200,
                "煤炭": 80,
            },
            medicine=med,
        )
    )
    assert all(p.kind != "claim" for p in plan)


def test_full_queue_gathers_process_mat_not_other_facility():
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 0,
                "高級木材": 0,
                "高級布料": 0,
                "銅礦石": 0,
            },
            metal=tuple(QueueSlot("合金鋼錠", False) for _ in range(5)),
        )
    )
    assert all(p.kind != "feed" for p in plan)
    gathers = [p for p in plan if p.kind == "gather"]
    assert gathers
    assert any(p.gather_mode == "process_mat" for p in gathers)
    assert not any(p.gather_mode == "skill" for p in gathers)
    from mm_assist.life_plan.recipes import WIRED_GATHER_DETAILS

    assert all(p.detail in WIRED_GATHER_DETAILS for p in gathers if p.gather_mode == 'process_mat')


def test_gather_titles_use_material_not_shared_tree():
    from mm_assist.life_plan.planner import _gather_process

    sap = _gather_process({"樹液": 8, "原木": 1})
    assert sap is not None
    assert sap.title == "采集 樹液"
    assert sap.detail == "樹液"
    assert sap.recipe == "樹液"
    wood = _gather_process({"原木": 10, "樹液": 1})
    assert wood is not None
    assert wood.title == "采集 原木"
    assert wood.detail == "樹木"
    assert wood.recipe == "原木"


def test_gather_click_yield_sap_two_tiers():
    from mm_assist.life_plan.recipes import gather_click_yield

    assert gather_click_yield("樹液", sap_done=0) == 4
    assert gather_click_yield("樹液", sap_done=1) == 10
    assert gather_click_yield("淨淨蘑菇") == 40
    assert gather_click_yield("裝水的瓶子") == 2


def test_claim_when_three_metal_done_before_feed():
    metal = tuple(QueueSlot("合金鋼錠", True) for _ in range(3)) + (
        QueueSlot("合金鋼錠", False),
    )
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 0,
                "高級木材": 100,
                "高級布料": 100,
                "鋼錠": 30,
                "銅礦石": 200,
                "煤炭": 80,
            },
            metal=metal,
        )
    )
    claims = [p for p in plan if p.kind == "claim"]
    assert claims
    assert claims[0].facility == "metal"
    assert claims[0].title == "领取 金屬"
    claim_at = next(i for i, p in enumerate(plan) if p.kind == "claim")
    feed_at = next((i for i, p in enumerate(plan) if p.kind == "feed"), None)
    if feed_at is not None:
        assert claim_at < feed_at
    snap = _snap(
        {
            "合金鋼錠": 0,
            "高級木材": 0,
            "高級布料": 0,
            "原木": 40,
            "鋼錠": 30,
            "銅礦石": 200,
            "煤炭": 80,
        }
    )
    plan = plan_life(snap, blocked_facilities=frozenset({"wood"}))
    assert all(p.facility != "wood" for p in plan)


def test_failed_snapshot_yields_empty_plan():
    assert plan_life(_snap(ok=False, error="consumable_unreadable")) == []


def test_controller_pause_keeps_queue_start_replans():
    reset_controller_for_tests()
    a = start_life(
        snapshot=_snap(
            {"合金鋼錠": 12, "高級木材": 0, "高級布料": 0},
            tools=(ToolDurability("鎬", 0.0, 50.0),),
            taken_at=10.0,
        )
    )
    assert a.running
    assert a.queue[0].kind == "repair"
    paused = pause_life()
    assert paused.paused and not paused.running
    assert [x.id for x in paused.queue] == [x.id for x in a.queue]
    b = start_life(
        snapshot=_snap(
            {
                "合金鋼錠": 12,
                "高級木材": 0,
                "高級布料": 0,
                "鋼錠": 9,
                "銅礦石": 45,
                "煤炭": 24,
            },
            taken_at=11.0,
        )
    )
    assert b.running and not b.paused
    assert b.queue[0].kind != "repair"


def test_resume_keeps_queue_and_clears_fail():
    from mm_assist.life_plan.controller import note_exec_fail, resume_life

    reset_controller_for_tests()
    a = start_life(snapshot=_snap({"合金鋼錠": 0, "高級木材": 0, "高級布料": 0}))
    assert a.current is not None
    first = a.current.id
    note_exec_fail(first)
    note_exec_fail(first)
    paused = pause_life()
    assert paused.paused and paused.fail_n == 2
    r = resume_life()
    assert r.running and not r.paused
    assert r.fail_n == 0
    assert [x.id for x in r.queue] == [x.id for x in paused.queue]
    assert r.current is not None and r.current.id == first


def test_queue_empty_replans_immediately():
    reset_controller_for_tests()
    snap = _snap(
        {
            "合金鋼錠": 0,
            "高級木材": 0,
            "高級布料": 0,
            "鋼錠": 12,
            "銅礦石": 60,
            "煤炭": 32,
        },
        taken_at=1.0,
    )
    st = start_life(snapshot=snap)
    ids = [x.id for x in st.queue]
    assert ids[0] == "p1"
    assert "p11" not in ids
    for item_id in ids[:-1]:
        st = acknowledge_done(item_id, now=2.0, snapshot=snap)
    leftover = [x.id for x in st.queue]
    assert leftover == [ids[-1]]
    st = acknowledge_done(ids[-1], now=2.0, snapshot=snap)
    assert len(st.queue) == 10
    assert not any(p.gather_mode == 'skill' for p in st.queue)
    assert st.running
    new_ids = [x.id for x in st.queue]
    assert leftover[0] not in new_ids
    assert new_ids[0] != "p1"
    assert all(i not in ids for i in new_ids)


def test_skip_current_keeps_plan_numbers():
    from mm_assist.life_plan.controller import skip_current

    reset_controller_for_tests()
    st = start_life(
        snapshot=_snap(
            {
                "合金鋼錠": 0,
                "高級木材": 0,
                "高級布料": 0,
                "鋼錠": 12,
                "銅礦石": 60,
                "煤炭": 32,
            }
        )
    )
    ids = [x.id for x in st.queue]
    assert len(ids) >= 2
    assert ids[0] == "p1"
    assert "p11" not in ids
    skip_current(reason="exec_fail", action=st.current.title if st.current else "", stage="gather")
    st2 = get_controller()
    assert [x.id for x in st2.queue] == ids[1:]
    assert st2.current is not None
    assert st2.current.id == ids[1]


def test_claim_fail_skips_without_requeue(monkeypatch):
    from mm_assist.life_plan.bar_ui import _note_fail_and_continue
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    def fake_plan(_snap, start_id=1, **_k):
        n = int(start_id)
        return [
            PlanItem(
                id=f"p{n}",
                kind="claim",
                title="领取 金屬",
                facility="metal",
            ),
            PlanItem(
                id=f"p{n + 1}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
            ),
        ]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", fake_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_recover.recover_once",
        lambda *_a, **_k: {"ok": True, "action": "x", "where_node": ""},
    )
    monkeypatch.setattr("mm_assist.life_plan.anomalies.cache_fail_shot", lambda *_a, **_k: "")
    st = start_life(snapshot=snap)
    assert st.current is not None
    assert st.current.kind == "claim"
    claim_id = st.current.id
    _note_fail_and_continue(win=None, item=st.current, reason="claim_fail")
    st2 = get_controller()
    ids = [x.id for x in st2.queue]
    assert claim_id not in ids
    assert st2.current is not None
    assert st2.current.id != claim_id
    assert st2.current.kind == "gather"


def test_start_life_resets_seq_to_p1():
    reset_controller_for_tests()
    snap = _snap(
        {
            "合金鋼錠": 0,
            "高級木材": 0,
            "高級布料": 0,
            "鋼錠": 12,
            "銅礦石": 60,
            "煤炭": 32,
        }
    )
    st = start_life(snapshot=snap, now=1_700_000_000.0)
    ids = [x.id for x in st.queue]
    for item_id in ids:
        st = acknowledge_done(item_id, now=2.0, snapshot=snap)
    assert st.queue
    assert st.queue[0].id != "p1"
    st2 = start_life(snapshot=snap, now=1_700_000_010.0)
    assert st2.queue[0].id == "p1"
    text = format_life_plan_block(st2)
    assert "本次开始：" in text
    assert "计划表" in text



def test_execute_metal_feed_needs_window():
    plan = plan_life(_snap({"合金鋼錠": 0, "高級木材": 0, "高級布料": 0, "鋼錠": 30, "銅礦石": 200, "煤炭": 80}))
    feed = next(p for p in plan if p.kind == "feed" and p.facility == "metal")
    r = try_execute(feed, items={"鋼錠": 30, "銅礦石": 200, "煤炭": 80})
    assert r.ok is False
    assert r.status == "need_context"


def test_format_life_panel_lists_plan():
    reset_controller_for_tests()
    st = start_life(snapshot=_snap({"合金鋼錠": 0, "高級木材": 0, "高級布料": 0}))
    text = format_life_panel(st)
    assert "计划表" in text
    assert "当前计划" in text
    assert "当前动作" in text
    assert "生活备料（" not in text
    plan = format_life_plan_block(st)
    proc = format_life_process_block(st)
    need = format_life_needs_block(st)
    assert "计划表" in plan
    assert "必要物品" not in plan
    assert "加工状态：" in proc
    assert "必要物品：" in need
    assert "计划表" not in need
    assert "加工状态：" in text
    assert "金属：— — — — —" in text
    assert "藥品：— — — — —" in text
    assert "必要物品：" in text
    assert "合金鋼錠 0/100" in text


def test_format_process_status_codes():
    text = format_process_status(
        ok=True,
        last_scan_at=1.0,
        slots={
            "metal": (QueueSlot("合金鋼錠", True), QueueSlot("鋼錠", False)),
            "wood": (),
            "cloth": tuple(QueueSlot("高級布料", False) for _ in range(4)),
            "medicine": (
                QueueSlot("淨淨蘑菇汁液", True),
                QueueSlot("淨淨蘑菇汁液", True),
                QueueSlot("淨淨蘑菇汁液", True),
            ),
        },
    )
    assert "金属：1 2 — — —" in text
    assert "木材：— — — — —" in text
    assert "布料：2 2 2 2 —" in text
    assert "藥品：1 1 1 — —" in text
    assert "%" not in text
    assert format_process_status(ok=False, last_scan_at=0) == "加工状态：未开始（暂无进度）"
    assert format_process_status(ok=False, last_scan_at=10.0) == "加工状态：扫描失败（不使用旧数据）"


def test_format_need_items_and_display_rows():
    have = {
        "合金鋼錠": 0,
        "高級木材": 100,
        "高級布料": 100,
        "淨淨蘑菇汁液": 0,
        "紙": 3,
    }
    desired = {
        "合金鋼錠": 100,
        "高級木材": 100,
        "高級布料": 100,
        "淨淨蘑菇汁液": 180,
    }
    targets, gaps = display_need_rows(have, desired)
    names = {n for n, _, _ in gaps}
    assert ("合金鋼錠", 0, 100) in targets
    assert "紙" in names
    assert "空瓶" in names
    assert "淨淨蘑菇孢子" in names
    assert "鐵礦石" in names
    unread = format_need_items(status=None)
    assert unread == "必要物品：未读取"
    fail = format_need_items(status=False, error="consumable_unreadable")
    assert "consumable_unreadable" in fail
    text = format_need_items(status=True, have=have, taken_at=1_700_000_000.0, remain_s=12.0)
    assert "合金鋼錠 0/100" in text
    assert "高級木材 100/100" in text
    assert "紙 " not in text and "空瓶 " not in text
    assert "最长剩余：12s" in text
    assert "%" not in text


def test_one_click_repair_refuses_when_life_running():
    reset_controller_for_tests()
    start_life(snapshot=_snap({"合金鋼錠": 0, "高級木材": 0, "高級布料": 0}))
    hints: list[str] = []
    assert try_start_one_click_repair(None, lambda: None, hints.append) is False
    assert hints and "生活调度" in hints[0]


def test_one_click_repair_mutex(monkeypatch):
    reset_controller_for_tests()
    gate = threading.Event()

    def fake(*a, **k):
        gate.wait(3)
        return ExecuteResult(ok=True, status="done", message="耐久已回升")

    monkeypatch.setattr("mm_assist.life_execute.repair.execute_repair", fake)
    hints: list[str] = []
    assert try_start_one_click_repair(None, lambda: None, hints.append) is True
    assert try_start_one_click_repair(None, lambda: None, hints.append) is False
    assert any("修理中" in h for h in hints)
    gate.set()
    from mm_assist.life_plan import bar_ui as bu

    t = bu._REPAIR
    if t is not None:
        t.join(timeout=3)


def test_stop_repair_sets_cancel():
    from mm_assist.tool_repair import clear_op_cancel, op_cancelled

    clear_op_cancel()
    request_stop_repair()
    assert op_cancelled()
    clear_op_cancel()


def test_explode_needs_counts_ore():
    have = {"合金鋼錠": 0, "高級木材": 100, "高級布料": 100}
    desired = {"合金鋼錠": 100, "高級木材": 100, "高級布料": 100}
    need = explode_needs(have, desired)
    assert need["鐵礦石"] > 0
    assert need["銅礦石"] > 0
    assert "原木" not in need


def test_explode_needs_juice_bom():
    have = {"合金鋼錠": 100, "高級木材": 100, "高級布料": 100, "淨淨蘑菇汁液": 0}
    desired = {
        "合金鋼錠": 100,
        "高級木材": 100,
        "高級布料": 100,
        "淨淨蘑菇汁液": 180,
    }
    need = explode_needs(have, desired)
    assert need["淨淨蘑菇"] == 720
    assert need["紙"] == 72
    assert need["裝水的瓶子"] == 36
    assert need["空瓶"] == 36
    assert "淨淨蘑菇孢子" not in need


def test_plan_feeds_juice_when_three_full():
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 100,
                "高級木材": 100,
                "高級布料": 100,
                "淨淨蘑菇汁液": 0,
                "淨淨蘑菇孢子": 200,
                "裝水的瓶子": 40,
            }
        )
    )
    feeds = [p for p in plan if p.kind == "feed"]
    assert feeds
    assert all(p.facility == "medicine" and p.recipe == "淨淨蘑菇汁液" for p in feeds)


def test_plan_gathers_mushroom_not_paper():
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 100,
                "高級木材": 100,
                "高級布料": 100,
                "淨淨蘑菇汁液": 0,
                "淨淨蘑菇": 0,
                "紙": 0,
                "裝水的瓶子": 0,
            }
        )
    )
    gathers = [p for p in plan if p.kind == "gather"]
    assert gathers
    # Paper is required for feeding: no repeated raw-gather block without it.
    assert all(p.gather_mode == "skill" for p in gathers)
    assert all(p.detail != '紙' for p in gathers)
    assert len(gathers) == 10


def test_plan_mushroom_clicks_use_yield_40():
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 100,
                "高級木材": 100,
                "高級布料": 100,
                "淨淨蘑菇汁液": 0,
                "淨淨蘑菇": 680,
                "紙": 72,
                "裝水的瓶子": 36,
            }
        )
    )
    gathers = [p for p in plan if p.kind == "gather"]
    mush = [p for p in gathers if p.detail == "淨淨蘑菇" and p.gather_mode == 'process_mat']
    assert not mush  # 680 mushrooms already cover this facility's five free slots.
    assert sum(p.kind == 'feed' for p in plan) == 5
    assert all(p.gather_mode == 'skill' for p in gathers)


def test_plan_gathers_well_when_mushrooms_ready():
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 100,
                "高級木材": 100,
                "高級布料": 100,
                "淨淨蘑菇汁液": 0,
                "淨淨蘑菇孢子": 360,
                "淨淨蘑菇": 0,
                "紙": 0,
                "空瓶": 36,
                "裝水的瓶子": 0,
            }
        )
    )
    gathers = [p for p in plan if p.gather_mode == "process_mat"]
    assert gathers
    assert all(p.detail == "水井" for p in gathers)
    assert gathers[0].skill == "日常採集"
    assert gathers[0].gather_mode == "process_mat"
    assert "按2/次" in gathers[0].reason
    assert len(gathers) == 10
    assert '缺裝水的瓶子×5' in gathers[0].reason
    assert not any(p.gather_mode == 'skill' for p in plan)


def test_plan_well_clicks_use_yield_2():
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 100,
                "高級木材": 100,
                "高級布料": 100,
                "淨淨蘑菇汁液": 0,
                "淨淨蘑菇孢子": 360,
                "紙": 72,
                "空瓶": 10,
                "裝水的瓶子": 34,
            }
        )
    )
    wells = [p for p in plan if p.gather_mode == 'process_mat' and p.detail == "水井"]
    assert not wells  # Five slots consume five bottles, not the full target's 36.
    assert sum(p.kind == 'feed' for p in plan) == 5


def test_plan_one_well_when_no_empty_bottle():
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 100,
                "高級木材": 100,
                "高級布料": 100,
                "淨淨蘑菇汁液": 0,
                "淨淨蘑菇孢子": 360,
                "紙": 72,
                "空瓶": 0,
                "裝水的瓶子": 0,
            }
        )
    )
    wells = [p for p in plan if p.kind == "gather" and p.detail == "水井"]
    assert not wells
    assert all(p.gather_mode == 'skill' for p in plan)


def test_insert_repair_next_after_current():
    from mm_assist.life_plan.planner import insert_repair_next
    from mm_assist.life_plan.types import PlanItem

    q = [
        PlanItem(id="p1", kind="feed", title="投料 鋼錠", facility="metal", recipe="鋼錠"),
        PlanItem(id="p2", kind="gather", title="采集 銅礦脈", detail="銅礦脈"),
    ]
    out, seq = insert_repair_next(q, after_id="p1", seq=3)
    assert [x.id for x in out] == ["p1", "p3", "p2"]
    assert out[1].kind == "repair"
    assert out[1].title == "一键修理全部工具"
    assert seq == 4
    out2, seq2 = insert_repair_next(out, after_id="p1", seq=seq)
    assert [x.id for x in out2] == [x.id for x in out]
    assert seq2 == 4


def test_pump_loop_executes_current(monkeypatch):
    # This loop test has no game context. Do not let window resolution attach
    # to a locally running game; real frame readiness is tested separately.
    monkeypatch.setattr('mm_assist.life_plan.bar_ui._resolve_life_win', lambda *a: None)
    from mm_assist.life_plan.bar_ui import start_life_pump

    reset_controller_for_tests()
    n = {"c": 0}

    def fake(item, **k):
        n["c"] += 1
        return ExecuteResult(ok=True, status="done", message="ok")

    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", fake)
    start_life(
        snapshot=_snap(
            {
                "合金鋼錠": 0,
                "高級木材": 0,
                "高級布料": 0,
                "鋼錠": 30,
                "銅礦石": 200,
                "煤炭": 80,
            }
        )
    )
    start_life_pump(None, lambda: None)
    deadline = time.time() + 2
    while n["c"] == 0 and time.time() < deadline:
        time.sleep(0.02)
    pause_life()
    from mm_assist.life_plan import bar_ui
    if bar_ui._PUMP is not None:
        bar_ui._PUMP.join(timeout=2)
        assert not bar_ui._PUMP.is_alive()
    assert n["c"] >= 1


def test_repair_executor_rejects_unknown_page_without_clicking(monkeypatch):
    """未知页拒绝输入；未发Esc的等待保留当前项，不虚耗预算跳过。"""
    from types import SimpleNamespace
    import numpy as np
    from mm_assist.life_execute import repair, ui_logic
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()

    def fake_plan(_snap, start_id=1, **_k):
        return [PlanItem(id="p1", kind="repair", title="修镐"),
                PlanItem(id="p2", kind="feed", title="下一计划", facility="wood", recipe="高級木材")]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", fake_plan)
    monkeypatch.setattr(
        "mm_assist.life_plan.controller.tick_facility_progress",
        lambda **k: None,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.skip_wait_ui_params", lambda: False
    )
    monkeypatch.setattr(
        "mm_assist.perception.state_ui.read_state_ui",
        lambda **k: None,
    )
    monkeypatch.setattr(ui_logic, "resolve_navigation",
                        lambda *_a, **_k: ui_logic.UiWhere("unknown", None, None, None, "fixture"))
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_is_hud_idle", lambda: False)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_common_popup", lambda **k: False)
    monkeypatch.setattr("mm_assist.life_execute.ui_nav.VERIFY_TOTAL_S", 0)
    monkeypatch.setattr("mm_assist.platform.capture.grab_client",
                        lambda win: np.zeros((1442, 1912, 3), dtype=np.uint8))
    def forbidden(*a, **k):
        pytest.fail("unknown page must not click or start repair")
    monkeypatch.setattr("mm_assist.safety.input.click_client", forbidden)
    monkeypatch.setattr(repair, "one_click_repair", forbidden)
    monkeypatch.setattr(repair, "release_mouse_buttons", lambda: None)
    start_life(snapshot=_snap({"合金鋼錠": 0}))
    pump_life_once(SimpleNamespace(incomplete_plan_policy="pause"), object())
    assert get_controller().current is not None
    # 2026-10-06: waiting without a dispatched Esc is not two failed escapes.
    assert get_controller().current.id == "p1"
    assert '等待界面刷新' in get_controller().current_action
    from mm_assist.life_plan import bar_ui
    assert bar_ui._RECOVERY_COUNT == 0
    assert get_controller().pending_action is None


def test_pump_hud_idle_true_main_proceeds(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once

    reset_controller_for_tests()
    n = {"c": 0}

    class _V:
        family = "main_hud"
        ui = "hud"
        stp = 0
        raw = {"ui_stack_ready": True}

    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.skip_wait_ui_params", lambda: False
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "main_hud", "zh": "主界面-待机"},
    )
    monkeypatch.setattr("mm_assist.perception.state_ui.read_state_ui", lambda **k: _V())
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: n.__setitem__("c", n["c"] + 1)
        or ExecuteResult(ok=True, status="done", message="ok"),
    )
    start_life(
        snapshot=_snap(
            {
                "合金鋼錠": 0,
                "高級木材": 0,
                "高級布料": 0,
                "鋼錠": 30,
                "銅礦石": 200,
                "煤炭": 80,
            }
        )
    )
    pump_life_once(None, None)
    assert n["c"] == 1


def test_main_hud_stp0_rejects_unknown_empty_ok(monkeypatch):
    from types import SimpleNamespace

    from mm_assist.life_plan.bar_ui import _main_hud_stp0

    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record", lambda **k: {}
    )
    fake = SimpleNamespace(
        family="unknown",
        ui="hud",
        stp=0,
        raw={"ui_stack_empty_ok": True},
    )
    assert _main_hud_stp0(fake) is False
    true_hud = SimpleNamespace(
        family="main_hud",
        ui="hud",
        stp=0,
        raw={"ui_stack_ready": True},
    )
    assert _main_hud_stp0(true_hud) is False


def test_main_hud_stp0_follows_inspect_idle(monkeypatch):
    from types import SimpleNamespace

    from mm_assist.life_plan.bar_ui import _main_hud_stp0

    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "main_hud", "zh": "主界面-待机"},
    )
    fake = SimpleNamespace(
        family="unknown",
        ui="hud",
        stp=0,
        raw={"ui_stack_empty_ok": True},
    )
    assert _main_hud_stp0(fake) is True
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "main_hud", "zh": "主界面-动作中"},
    )
    assert _main_hud_stp0(fake) is False


def test_pump_sense_keeps_real_message(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})
    boiler = "等界面/找不到入口，本拍不点"

    def fake_plan(_snap, start_id=1, **_k):
        return [
            PlanItem(
                id=f"p{int(start_id)}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
                skill="採礦",
                detail="銅礦脈",
            )
        ]

    def batch_plan(snap, start_id=1, **kw):
        from dataclasses import replace
        original = fake_plan(snap, start_id=start_id, **kw)
        return [replace(original[i % len(original)], id=f"p{start_id+i}") for i in range(10)]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", batch_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(
            ok=False, status="unknown_ui", message="找不到菜单"
        ),
    )
    start_life(snapshot=snap)
    pump_life_once(None, None)
    st = get_controller()
    import json
    from mm_assist.life_plan import anomalies
    rows = [json.loads(x) for x in (anomalies.ROOT / anomalies.LOG_REL).read_text(encoding="utf-8").splitlines()]
    act = next(x["extra"] for x in rows if x["tag"] == "execute_result")
    assert "找不到菜单" in act
    assert boiler not in act
    assert st.fail_n == 0
    assert st.current is not None and st.current.id != "p1"


def test_prepare_menu_inspect_still_hud_is_sense(monkeypatch):
    from mm_assist.life_execute.ui_act import prepare_to
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.execute import is_sense_result

    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_nav.locate_edge",
        lambda *a, **k: (11, 22),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "main_hud", "zh": "主界面-待机"},
    )
    r = prepare_to(
        "facility",
        grab=lambda: object(),
        click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda _s: None,
        read_stp=lambda: 0,
        resolve=lambda _f=None: UiWhere("main_hud", 0, 0, 0, "mem"),
        already_ok=lambda _f: False,
        title="投料 鋼錠",
    )
    assert clicks == [(11, 22)]
    assert r.ok is False
    assert r.status == "stale_ui"
    assert "点菜单后界面未变成菜单" in str(r.message or "")
    assert is_sense_result(r) is True


def test_pump_claim_proceeds_when_inspect_idle(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    n = {"c": 0}

    class _V:
        family = "unknown"
        ui = "hud"
        stp = 0
        raw = {"ui_stack_empty_ok": True}

    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.skip_wait_ui_params", lambda: False
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "main_hud", "zh": "主界面-待机"},
    )
    monkeypatch.setattr("mm_assist.perception.state_ui.read_state_ui", lambda **k: _V())
    monkeypatch.setattr(
        "mm_assist.life_plan.controller.plan_life",
        lambda *_a, **k: [
            PlanItem(id="p1", kind="claim", title="领取 木材", facility="wood")
        ],
    )
    monkeypatch.setattr(
        "mm_assist.life_plan.controller.take_snapshot", lambda **k: _snap({})
    )
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: n.__setitem__("c", n["c"] + 1)
        or ExecuteResult(ok=True, status="done", message="ok"),
    )
    start_life(snapshot=_snap({}))
    pump_life_once(None, None)
    assert n["c"] == 1


def test_pump_gather_executes_off_main_hud(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    n = {"c": 0}
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    class _V:
        family = "unknown"
        ui = "hud"
        stp = 0
        raw = {"ui_stack_empty_ok": True}

    def fake_plan(_snap, start_id=1, **_k):
        return [
            PlanItem(
                id=f"p{int(start_id)}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
                skill="採礦",
                detail="銅礦脈",
            )
        ]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", fake_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.skip_wait_ui_params", lambda: False
    )
    monkeypatch.setattr("mm_assist.perception.state_ui.read_state_ui", lambda **k: _V())
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: n.__setitem__("c", n["c"] + 1)
        or ExecuteResult(ok=True, status="done", message="ok"),
    )
    start_life(snapshot=snap)
    pump_life_once(None, None)
    assert n["c"] == 1


def test_pump_feed_skips_hud_idle(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    n = {"c": 0}
    snap = _snap({"合金鋼錠": 0, "銅礦石": 200, "煤炭": 80, "鋼錠": 30})

    class _V:
        family = "unknown"
        ui = "hud"
        stp = 0
        raw = {"ui_stack_empty_ok": True}

    def fake_plan(_snap, start_id=1, **_k):
        return [
            PlanItem(
                id=f"p{int(start_id)}",
                kind="feed",
                title="投料 合金鋼錠",
                facility="metal",
                recipe="合金鋼錠",
            )
        ]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", fake_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.skip_wait_ui_params", lambda: False
    )
    monkeypatch.setattr("mm_assist.perception.state_ui.read_state_ui", lambda **k: _V())
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: n.__setitem__("c", n["c"] + 1)
        or ExecuteResult(ok=True, status="done", message="ok"),
    )
    start_life(snapshot=snap)
    pump_life_once(None, None)
    assert n["c"] == 1


def test_pump_claim_skips_hud_idle(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    n = {"c": 0}
    snap = _snap({"合金鋼錠": 0, "銅礦石": 200, "煤炭": 80, "鋼錠": 30})

    class _V:
        family = "unknown"
        ui = "hud"
        stp = 0
        raw = {"ui_stack_empty_ok": True}

    def fake_plan(_snap, start_id=1, **_k):
        return [
            PlanItem(
                id=f"p{int(start_id)}",
                kind="claim",
                title="领取 metal",
                facility="metal",
            )
        ]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", fake_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.skip_wait_ui_params", lambda: False
    )
    monkeypatch.setattr("mm_assist.perception.state_ui.read_state_ui", lambda **k: _V())
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: n.__setitem__("c", n["c"] + 1)
        or ExecuteResult(ok=True, status="done", message="ok"),
    )
    start_life(snapshot=snap)
    pump_life_once(None, None)
    assert n["c"] == 1


def test_ten_min_scan_defers_repair_detection_until_batch_end(monkeypatch):
    reset_controller_for_tests()
    from mm_assist.life_plan.controller import tick_facility_progress
    from mm_assist.life_plan.facility_sim import set_facility_read_for_tests

    monkeypatch.setattr(
        "mm_assist.life_plan.controller.take_snapshot",
        lambda **k: _snap({"鋼錠": 12}, tools=(ToolDurability("鎬", 0.0, 50.0),), taken_at=k['now']),
    )
    set_facility_read_for_tests(
        lambda: {"metal": (), "wood": (), "cloth": (), "medicine": ()}
    )
    st = start_life(
        snapshot=_snap(
            {
                "合金鋼錠": 0,
                "高級木材": 0,
                "高級布料": 0,
                "鋼錠": 12,
                "銅礦石": 60,
                "煤炭": 32,
            },
            tools=(ToolDurability("鎬", 50.0, 50.0),),
        )
    )
    assert st.current is not None
    assert st.current.kind == "feed"
    st2 = tick_facility_progress(force_scan=True)
    # The user now forbids a full read before this visible batch finishes.
    assert st2.queue == st.queue and st2.snapshot == st.snapshot
    for item in st.queue:
        acknowledge_done(item.id, now=1001)
    st2 = tick_facility_progress(now=1002)
    assert st2.current.kind == "repair"


def test_fail_skips_when_back_fails(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    recovers = []
    snap = _snap(
        {
            "合金鋼錠": 0,
            "高級木材": 0,
            "高級布料": 0,
            "銅礦石": 0,
        }
    )

    def fake_plan(_snap, start_id=1, **_k):
        n = int(start_id)
        return [
            PlanItem(
                id=f"p{n}",
                kind="feed",
                title="投料 鋼錠",
                facility="metal",
                recipe="鋼錠",
            ),
            PlanItem(
                id=f"p{n + 1}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
                skill="採礦",
                detail="銅礦脈",
            ),
        ]

    def batch_plan(snap, start_id=1, **kw):
        from dataclasses import replace
        original = fake_plan(snap, start_id=start_id, **kw)
        return [replace(original[i % len(original)], id=f"p{start_id+i}") for i in range(10)]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", batch_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_recover.recover_once",
        lambda trigger, **k: recovers.append(str(trigger)) or {"ok": False, "action": "x", "where_node": ""},
    )
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(ok=False, status="feed_fail", message="投料失败"),
    )
    st = start_life(snapshot=snap)
    assert st.current is not None
    first = st.current.id
    assert st.current.kind == "feed"
    pump_life_once(None, None)
    st1 = get_controller()
    assert st1.current is not None
    assert st1.current.id != first
    assert st1.fail_n == 0
    assert recovers == ['feed_fail', 'feed_fail']
    pump_life_once(None, None)
    st2 = get_controller()
    assert st2.paused is False
    assert st2.running is True
    assert st2.current is not None
    assert st2.current.id != first
    assert len(recovers) == 4


def test_try_execute_wiring_inventory():
    from mm_assist.life_plan.recipes import GATHER_NODE, SKILL_GATHER
    from mm_assist.life_plan.types import PlanItem

    wired_gather = {d for _m, (_s, d) in GATHER_NODE.items()}
    for _mat, (_skill, detail) in GATHER_NODE.items():
        item = PlanItem(
            id="p1",
            kind="gather",
            title=f"采集 {detail}",
            skill=_skill,
            detail=detail,
            recipe=_mat,
            gather_mode="process_mat",
        )
        r = try_execute(item, items={})
        if detail in wired_gather:
            assert r.status != "not_ready", detail
        else:
            assert r.status == "not_ready", detail
    for skill, detail in SKILL_GATHER:
        item = PlanItem(
            id="p1",
            kind="gather",
            title=f"练采集 {detail}",
            skill=skill,
            detail=detail,
            recipe=detail,
            gather_mode="skill",
        )
        assert try_execute(item).status != "not_ready", detail
    for fac in ("metal", "wood", "cloth", "medicine"):
        claim = PlanItem(id="p1", kind="claim", title="领取", facility=fac)
        assert try_execute(claim).status != "not_ready", fac


def test_life_cmd_file_start_pause(tmp_path):
    from mm_assist.life_plan.cmd_file import consume_life_cmd, request_life_cmd

    hits: list[str] = []
    request_life_cmd("start", root=tmp_path, now=100.0)
    assert consume_life_cmd(start_fn=lambda: hits.append("start"), pause_fn=lambda: hits.append("pause"), root=tmp_path, now=100.5) == "start"
    assert hits == ["start"]
    assert consume_life_cmd(start_fn=lambda: hits.append("x"), root=tmp_path, now=101.0) == ""
    request_life_cmd("pause", root=tmp_path, now=200.0)
    assert consume_life_cmd(start_fn=lambda: hits.append("start"), pause_fn=lambda: hits.append("pause"), root=tmp_path, now=200.2) == "pause"
    assert hits == ["start", "pause"]
    request_life_cmd("start", root=tmp_path, now=1.0)
    assert consume_life_cmd(start_fn=lambda: hits.append("late"), root=tmp_path, now=40.0) == ""
    assert hits == ["start", "pause"]
    request_life_cmd("resume", root=tmp_path, now=250.0)
    assert (
        consume_life_cmd(
            resume_fn=lambda: hits.append("resume"),
            root=tmp_path,
            now=250.2,
        )
        == "resume"
    )
    assert hits == ["start", "pause", "resume"]
    request_life_cmd("escape", root=tmp_path, now=300.0)
    assert (
        consume_life_cmd(
            escape_fn=lambda: hits.append("escape"),
            root=tmp_path,
            now=300.2,
        )
        == "escape"
    )
    assert hits == ["start", "pause", "resume", "escape"]


def test_set_action_format(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.controller import set_action

    reset_controller_for_tests()
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(ok=False, status="找不到菜单", message="找不到菜单"),
    )
    start_life(
        snapshot=_snap(
            {"合金鋼錠": 0, "高級木材": 0, "高級布料": 0, "銅礦石": 0},
        )
    )
    titled = set_action("投料 鋼錠", "等界面刷新")
    assert titled.current_action.startswith("投料 鋼錠")
    assert "inv=" not in titled.current_action
    assert "menu=" not in titled.current_action
    pump_life_once(None, None)
    st = get_controller()
    assert st.current is not None
    assert st.current_action.startswith(st.current.title)
    assert "inv=" not in st.current_action
    assert "menu=" not in st.current_action


def test_unissued_sense_failures_have_no_backoff(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once

    reset_controller_for_tests()
    sleeps: list[float] = []
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda s: sleeps.append(float(s)))
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(ok=False, status="找不到菜单", message="找不到菜单"),
    )
    st0 = start_life(
        snapshot=_snap(
            {"合金鋼錠": 0, "高級木材": 0, "高級布料": 0, "銅礦石": 0},
        )
    )
    assert st0.current is not None
    first = st0.current.id
    backs: list[float] = []
    batch_size = len(st0.queue)
    for index in range(batch_size):
        pump_life_once(None, None)
        st = get_controller()
        if index < batch_size - 1:
            assert st.current is not None and st.current.id != first
        else:
            assert st.current is None and st.refresh_requested
        assert st.fail_n == 0
        backs.append(st.sense_backoff_s)
    assert backs == [0.0] * batch_size
    assert sleeps == []


def test_cooldown_after_skip(monkeypatch):
    from mm_assist.life_plan.controller import (
        skip_current,
    )
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    def fake_plan(_snap, start_id=1, **_k):
        return [
            PlanItem(
                id=f"p{int(start_id)}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
                detail="銅礦石",
            )
        ]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", fake_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    st = start_life(snapshot=snap)
    assert st.current is not None
    assert st.current.recipe == "銅礦石"
    skip_current(reason="exec_fail", action="采集 銅礦石 等界面", stage="gather")
    st2 = get_controller()
    assert st2.running
    assert not hasattr(st2, "cooldown")
    assert all(x.id != st.current.id for x in st2.queue)
    text = format_life_plan_block(st2)
    assert "冷却等待" not in st2.current_action
    assert "冷却等待" not in text
    assert st2.current is None and st2.refresh_requested
    assert "等待刷新必要物品及加工状态" in text


def test_start_life_clears_cooldown_and_shows_plan(monkeypatch):
    from mm_assist.life_plan.controller import skip_current
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    def fake_plan(_snap, start_id=1, **_k):
        return [
            PlanItem(
                id=f"p{int(start_id)}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
                detail="銅礦石",
            )
        ]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", fake_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    start_life(snapshot=snap)
    skip_current(reason="exec_fail", action="采集 銅礦石 等界面", stage="gather")
    st = start_life(snapshot=snap)
    assert st.running
    assert st.queue
    assert st.current is not None
    assert "采集 銅礦石" in st.current.title
    text = format_life_plan_block(st)
    assert "计划表" in text
    assert "（空）" not in text


def test_anomaly_action_differs():
    import json

    from mm_assist.life_plan.anomalies import ROOT
    from mm_assist.life_plan.controller import skip_current

    reset_controller_for_tests()
    st = start_life(
        snapshot=_snap(
            {
                "合金鋼錠": 0,
                "高級木材": 0,
                "高級布料": 0,
                "銅礦石": 0,
            }
        )
    )
    assert st.current is not None
    plan_title = st.current.title
    skip_current(
        reason="nav_timeout",
        action="采集 銅礦石 等界面/找不到入口，本拍不点",
        stage="gather_detail",
        from_node="profile",
    )
    log = ROOT / "logs" / "life_prep_diary.jsonl"
    last = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert last["action"] != plan_title
    assert last["action"] == "采集 銅礦石 等界面/找不到入口，本拍不点"
    assert last["stage"] == "gather_detail"
    assert last["from_node"] == "profile"
    assert last["plan"] == plan_title


def test_cooled_queue_not_shown_empty(monkeypatch):
    from mm_assist.life_plan.controller import skip_current
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    def fake_plan(_snap, start_id=1, **_k):
        return [
            PlanItem(
                id=f"p{int(start_id)}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
                detail="銅礦石",
            )
        ]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", fake_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    start_life(snapshot=snap)
    skip_current(reason="exec_fail", action="采集 銅礦石", stage="gather")
    st = get_controller()
    assert st.running
    text = format_life_plan_block(st)
    assert "冷却等待" not in text


def test_not_ready_no_cooldown(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.execute import NOT_READY
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    def fake_plan(_snap, start_id=1, **_k):
        return [
            PlanItem(
                id=f"p{int(start_id)}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
                detail="銅礦石",
            )
        ]

    def batch_plan(snap, start_id=1, **kw):
        from dataclasses import replace
        original = fake_plan(snap, start_id=start_id, **kw)
        return [replace(original[i % len(original)], id=f"p{start_id+i}") for i in range(10)]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", batch_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(ok=False, status=NOT_READY, message="未就绪"),
    )
    st0 = start_life(snapshot=snap)
    assert st0.current is not None
    pump_life_once(None, None)
    st = get_controller()
    assert not hasattr(st, "cooldown")
    assert all(q.id != st0.current.id for q in st.queue)
    assert st.current is not None and st.current.id != st0.current.id
    assert st.unsupported == set()


def test_gather_empty_skill_uses_gather_node(monkeypatch):
    from mm_assist.life_plan.types import PlanItem

    called: dict = {}

    def fake_nearby(*_a, **k):
        called.update(k)
        return ExecuteResult(ok=True, status="done", message="ok")

    monkeypatch.setattr(
        "mm_assist.life_execute.skill_nearby.execute_skill_nearby",
        fake_nearby,
    )
    item = PlanItem(
        id="p1",
        kind="gather",
        title="采集 銅礦石",
        recipe="銅礦石",
        skill="",
        detail="",
        gather_mode="process_mat",
    )
    r = try_execute(item, items={})
    assert r.status != "not_ready"
    assert r.status == "done"
    assert called.get("skill_id") == "mine"
    assert called.get("item_name") == "銅礦脈"
    assert called.get("output") == "銅礦石"


def test_train_armor_tree_watches_log_bag(monkeypatch):
    from mm_assist.life_plan.live_test import scenarios
    from mm_assist.life_plan.types import PlanItem

    called: dict = {}

    def fake_nearby(*_a, **k):
        called.update(k)
        return ExecuteResult(ok=True, status="done", message="ok")

    monkeypatch.setattr(
        "mm_assist.life_execute.skill_nearby.execute_skill_nearby",
        fake_nearby,
    )
    item = scenarios()["train:甲胄樹"]
    assert item.gather_mode == "skill"
    r = try_execute(item, items={"原木": 10})
    assert r.status == "done"
    assert called.get("item_name") == "甲胄樹"
    assert called.get("output") == "原木"


def test_nav_timeout_classifies_and_skips_unissued(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.execute import is_nav_result, is_sense_result
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    def fake_plan(_snap, start_id=1, **_k):
        return [
            PlanItem(
                id=f"p{int(start_id)}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
                skill="採礦",
                detail="銅礦脈",
            )
        ]

    def batch_plan(snap, start_id=1, **kw):
        from dataclasses import replace
        original = fake_plan(snap, start_id=start_id, **kw)
        return [replace(original[i % len(original)], id=f"p{start_id+i}") for i in range(10)]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", batch_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(ok=False, status="nav_timeout", message="未到达技能详情"),
    )
    nav = ExecuteResult(ok=False, status="nav_timeout", message="超时")
    assert is_nav_result(nav) is True
    assert is_sense_result(nav) is False
    assert is_nav_result(ExecuteResult(ok=False, status="nav_verify_timeout")) is True
    assert is_sense_result(ExecuteResult(ok=False, status="nav_verify_timeout")) is False
    assert is_nav_result(ExecuteResult(ok=False, status="x", message="未到达目标页")) is True
    assert is_nav_result(ExecuteResult(ok=False, status="need_verify")) is False
    assert is_sense_result(ExecuteResult(ok=False, status="need_verify")) is True
    assert is_sense_result(ExecuteResult(ok=False, status="facility_unknown")) is True
    assert is_nav_result(ExecuteResult(ok=False, status="facility_unknown")) is False
    assert is_sense_result(ExecuteResult(ok=False, status="stale_ui")) is True
    assert is_nav_result(ExecuteResult(ok=False, status="stale_ui")) is False
    unread = ExecuteResult(ok=False, status="unknown_ui", message="认不出界面，不点、不换条")
    assert is_nav_result(unread) is False
    assert is_sense_result(unread) is True
    assert is_nav_result(ExecuteResult(ok=False, status="not_main_hud")) is False
    assert is_sense_result(ExecuteResult(ok=False, status="not_main_hud")) is True
    assert is_nav_result(ExecuteResult(ok=False, status="stp_busy")) is False
    assert is_sense_result(ExecuteResult(ok=False, status="stp_busy")) is True
    st0 = start_life(snapshot=snap)
    assert st0.current is not None
    first = st0.current.id
    pump_life_once(None, None)
    st = get_controller()
    assert st.current is not None
    assert st.current.id != first
    assert st.fail_n == 0
    assert "gather|銅礦石" not in st.unsupported


def test_unknown_ui_pump_skips_unissued_items(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    def fake_plan(_snap, start_id=1, **_k):
        n = int(start_id)
        return [
            PlanItem(
                id=f"p{n}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
                skill="採礦",
                detail="銅礦脈",
            ),
            PlanItem(
                id=f"p{n + 1}",
                kind="gather",
                title="采集 煤炭",
                recipe="煤炭",
                skill="採礦",
                detail="煤炭",
            ),
        ]

    def batch_plan(snap, start_id=1, **kw):
        from dataclasses import replace
        original = fake_plan(snap, start_id=start_id, **kw)
        return [replace(original[i % len(original)], id=f"p{start_id+i}") for i in range(10)]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", batch_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(
            ok=False, status="unknown_ui", message="认不出界面，不点、不换条"
        ),
    )
    st0 = start_life(snapshot=snap)
    assert st0.current is not None
    first = st0.current.id
    pump_life_once(None, None)
    pump_life_once(None, None)
    st = get_controller()
    assert st.current is not None
    assert st.current.id != first
    assert all(q.id not in {"p1", "p2"} for q in st.queue)
    assert st.fail_n == 0


def test_nav_fail_recovers_twice_then_skips(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    recovers: list[str] = []
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    def fake_plan(_snap, start_id=1, **_k):
        n = int(start_id)
        return [
            PlanItem(
                id=f"p{n}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
                skill="採礦",
                detail="銅礦脈",
            ),
            PlanItem(
                id=f"p{n + 1}",
                kind="gather",
                title="采集 煤炭",
                recipe="煤炭",
                skill="採礦",
                detail="煤炭",
            ),
        ]

    def batch_plan(snap, start_id=1, **kw):
        from dataclasses import replace
        original = fake_plan(snap, start_id=start_id, **kw)
        return [replace(original[i % len(original)], id=f"p{start_id+i}") for i in range(10)]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", batch_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_recover.recover_once",
        lambda trigger, **k: recovers.append(str(trigger)) or {"ok": True, "action": "x", "where_node": ""},
    )
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(ok=False, status="wrong_tab", message="停在炼金"),
    )
    st = start_life(snapshot=snap)
    assert st.current is not None
    first = st.current.id
    pump_life_once(None, None)
    st1 = get_controller()
    assert st1.current is not None
    assert st1.current.id != first
    assert st1.fail_n == 0
    assert recovers == ['wrong_tab', 'wrong_tab']
    assert all(q.id != first for q in st1.queue)
    pump_life_once(None, None)
    st2 = get_controller()
    assert st2.paused is False
    assert st2.running is True
    assert st2.current is not None
    assert st2.current.id != first
    assert len(recovers) == 4


def test_need_verify_click_gate_survives_skipping_unissued(monkeypatch):
    """跨未执行计划仍保留两次点击门闩；跳过失败项不重置门闩。"""
    from mm_assist.life_execute.ui_act import click_spot
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_controller_for_tests()
    reset_stage_gates_for_tests()
    recovers: list[str] = []
    clicks: list[tuple[int, int]] = []
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    def fake_plan(_snap, start_id=1, **_k):
        n = int(start_id)
        return [
            PlanItem(
                id=f"p{n}",
                kind="gather",
                title="采集 銅礦石",
                recipe="銅礦石",
                skill="採礦",
                detail="銅礦脈",
            ),
            PlanItem(
                id=f"p{n + 1}",
                kind="gather",
                title="采集 煤炭",
                recipe="煤炭",
                skill="採礦",
                detail="煤炭",
            ),
        ]

    def fake_try(*_a, **_k):
        err = click_spot(
            lambda x, y: clicks.append((x, y)) or True,
            (1, 1),
            "切页",
            stage="edge-stuck",
            wait_refresh=False,
        )
        if err is not None:
            return err
        return ExecuteResult(ok=False, status="nav_verify_timeout", message="未到达目标页")

    def batch_plan(snap, start_id=1, **kw):
        from dataclasses import replace
        original = fake_plan(snap, start_id=start_id, **kw)
        return [replace(original[i % len(original)], id=f"p{start_id+i}") for i in range(10)]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", batch_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_recover.recover_once",
        lambda trigger, **k: recovers.append(str(trigger)) or {"ok": True, "action": "x", "where_node": ""},
    )
    monkeypatch.setattr("mm_assist.life_plan.execute.try_execute", fake_try)
    st = start_life(snapshot=snap)
    assert st.current is not None
    first = st.current.id
    pump_life_once(None, None)
    assert get_controller().fail_n == 0
    pump_life_once(None, None)
    st2 = get_controller()
    assert recovers == ['nav_verify_timeout', 'nav_verify_timeout', 'need_verify', 'need_verify']
    assert st2.paused is False
    assert st2.running is True
    assert st2.current is not None
    assert st2.current.id != first
    pump_life_once(None, None)
    assert len(clicks) == 2


def test_gather_feed_fail_not_feed_cooldown(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    def fake_plan(_snap, start_id=1, **_k):
        return [
            PlanItem(
                id=f"p{int(start_id)}",
                kind="gather",
                title="采集 高級原木",
                recipe="高級原木",
                skill="伐木",
                detail="成材樹",
            )
        ]

    def batch_plan(snap, start_id=1, **kw):
        from dataclasses import replace
        original = fake_plan(snap, start_id=start_id, **kw)
        return [replace(original[i % len(original)], id=f"p{start_id+i}") for i in range(10)]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", batch_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(ok=False, status="feed_fail", message="投料失败"),
    )
    st0 = start_life(snapshot=snap)
    assert st0.current is not None
    assert st0.current.kind == "gather"
    first = st0.current.id
    pump_life_once(None, None)
    st = get_controller()
    assert st.current is not None
    assert st.current.id != first
    assert st.fail_n == 0
    assert not hasattr(st, "cooldown")
    assert "gather|高級原木" not in st.unsupported
    from mm_assist.life_plan.anomalies import ROOT
    import json

    log = ROOT / "logs" / "life_prep_diary.jsonl"
    last = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert last["tag"] == "feed_fail"
    assert last["kind"] == "gather"
    assert last["plan"] == "采集 高級原木"


def test_snapshot_ok_when_tools_unreadable(monkeypatch):
    from mm_assist.life_plan import snapshot as snap_mod

    monkeypatch.setattr(snap_mod, "find_game_pid", lambda: 1)
    monkeypatch.setattr(snap_mod, "open_process", lambda _pid: object())
    monkeypatch.setattr(snap_mod, "close_handle", lambda _h: None)
    monkeypatch.setattr(
        snap_mod,
        "read_consumable_counts",
        lambda *_a, **_k: {"空瓶": 2, "裝水的瓶子": 4, "合金鋼錠": 0},
    )
    monkeypatch.setattr(snap_mod, "read_equipped_tools", lambda *_a, **_k: None)
    monkeypatch.setattr(snap_mod, "read_facility_queues", lambda *_a, **_k: {})
    out = snap_mod.take_snapshot(budget_s=0)
    assert out.ok is True
    assert out.items["空瓶"] == 2
    assert out.tools == ()
    assert out.error == "tools_unreadable"
    reset_controller_for_tests()
    st = start_life(snapshot=out)
    assert st.start_blocked is False
    need = format_life_needs_block(st)
    assert "内存读取失败" not in need
    assert "空瓶" in need or "裝水的瓶子" in need or "合金鋼錠" in need


def test_snapshot_stops_after_three_seconds(monkeypatch):
    from mm_assist.life_plan import snapshot as snap_mod
    import threading

    release = threading.Event()
    finished = threading.Event()

    def _slow(_ts, **_k):
        try:
            release.wait(2)
            return snap_mod.failed_snapshot("late")
        finally:
            finished.set()

    monkeypatch.setattr(snap_mod, "_snapshot_body", _slow)
    t0 = time.perf_counter()
    out = snap_mod.take_snapshot(budget_s=0.2)
    elapsed = time.perf_counter() - t0
    assert out.ok is False
    assert out.error == "mem_budget"
    assert elapsed < 2.0
    assert snap_mod.take_snapshot(budget_s=0.01).error == "mem_busy"
    release.set()
    assert finished.wait(1)
    with snap_mod._SCAN_LOCK:
        pass


def test_start_life_snapshot_not_time_capped(monkeypatch):
    seen: dict = {}

    def fake_snap(**k):
        seen.update(k)
        return _snap({"合金鋼錠": 0, "銅礦石": 0})

    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", fake_snap)
    monkeypatch.setattr("mm_assist.life_plan.controller.tick_facility_progress", lambda **k: None)
    reset_controller_for_tests()
    st = start_life()
    assert st.start_blocked is False
    assert st.running is True
    assert float(seen.get("budget_s") or 0) == 0.0


def test_idle_stall_due_five_minutes():
    from mm_assist.life_plan.bar_ui import idle_stall_due, reset_pump_settle

    reset_pump_settle()
    assert idle_stall_due(plan_id="p1", param_key="inv=0", stp=0, now=0.0) is False
    assert idle_stall_due(plan_id="p1", param_key="inv=0", stp=0, now=299.0) is False
    assert idle_stall_due(plan_id="p1", param_key="inv=0", stp=0, now=300.0) is True
    assert idle_stall_due(plan_id="p1", param_key="inv=0", stp=1, now=301.0) is False


def test_pump_fail_uses_two_recoveries_without_extra_escape(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.types import PlanItem

    reset_controller_for_tests()
    recovers: list[str] = []
    snap = _snap({"合金鋼錠": 0, "銅礦石": 0})

    def fake_plan(_snap, start_id=1, **_k):
        n = int(start_id)
        return [
            PlanItem(id=f"p{n}", kind="gather", title="采集 銅礦石", recipe="銅礦石", skill="採礦", detail="銅礦脈"),
            PlanItem(id=f"p{n+1}", kind="gather", title="采集 煤炭", recipe="煤炭", skill="採礦", detail="煤炭"),
        ]

    monkeypatch.setattr("mm_assist.life_plan.controller.plan_life", fake_plan)
    monkeypatch.setattr("mm_assist.life_plan.controller.take_snapshot", lambda **k: snap)
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_recover.recover_once",
        lambda trigger, **k: recovers.append(str(trigger)) or {"ok": False, "action": "x", "where_node": ""},
    )
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(ok=False, status="exec_fail", message="执行失败"),
    )
    from mm_assist.life_plan.bar_ui import pump_thread_alive

    pause_life()
    deadline = time.time() + 1.0
    while pump_thread_alive() and time.time() < deadline:
        time.sleep(0.02)
    reset_controller_for_tests()
    start_life(snapshot=snap)
    assert get_controller().fail_n == 0
    pump_life_once(None, None)
    assert get_controller().fail_n == 0
    assert recovers == ['exec_fail', 'exec_fail']
    pump_life_once(None, None)
    assert len(recovers) == 4


def test_life_hotspots_still_wired():
    from mm_assist.life_execute.ui_act import nav_target
    from mm_assist.life_execute.ui_nav import path_edges
    from mm_assist.life_plan.types import PlanItem

    gather = PlanItem(
        id="g",
        kind="gather",
        title="采集 銅礦石",
        recipe="銅礦石",
        skill="採礦",
        detail="銅礦脈",
        gather_mode="process_mat",
    )
    feed = PlanItem(id="f", kind="feed", title="投料 鋼錠", facility="metal", recipe="鋼錠")
    assert nav_target(gather)[0] == "gather_skill"
    assert nav_target(feed)[0] == "facility"
    inv = [e["id"] for e in path_edges("main_hud", "inventory") or []]
    assert inv == ["main_hud_menu_system_menu", "system_menu_bag_inventory"]
    fac = [e["id"] for e in path_edges("facility", "gather_skill") or []]
    assert fac[0] == "facility_back_craft_hub"


def test_pause_discards_late_prepare_snapshot():
    from mm_assist.life_plan.bar_ui import start_life_prepare
    from mm_assist.life_plan.controller import get_controller, pause_life, reset_controller_for_tests

    reset_controller_for_tests()
    gate = threading.Event()

    def slow_snap(**_k):
        gate.wait(2)
        return _snap({"合金鋼錠": 0, "高級木材": 0, "高級布料": 0, "淨淨蘑菇汁液": 0})

    from mm_assist.tool_repair import clear_op_cancel

    clear_op_cancel()
    try:
        hints: list[str] = []
        assert start_life_prepare(None, lambda: None, snapshot_fn=slow_snap, on_hint=hints.append)
        deadline = time.time() + 1.0
        while not get_controller().preparing and time.time() < deadline:
            time.sleep(0.01)
        assert get_controller().preparing is True
        assert get_controller().running is False
        pause_life()
        gate.set()
        deadline = time.time() + 2.0
        while get_controller().preparing and time.time() < deadline:
            time.sleep(0.02)
        st = get_controller()
        assert st.running is False
        assert st.preparing is False
        assert st.paused is True
    finally:
        gate.set()
        from mm_assist.life_plan import bar_ui
        worker = bar_ui._PUMP
        if worker is not None:
            worker.join(3)
            assert not worker.is_alive()
        clear_op_cancel()


def test_old_prepare_result_does_not_cancel_new_prepare():
    from mm_assist.life_plan import controller as c

    c.reset_controller_for_tests()
    old, _ = c.mark_life_preparing()
    current, _ = c.mark_life_preparing()
    snap = _snap({})
    after_old = c.apply_prepared_snapshot(snap, old)
    assert after_old.preparing
    assert after_old.prepare_token == current
    assert c.apply_prepared_snapshot(snap, current).running


def test_prepare_exception_releases_state_and_allows_retry():
    from mm_assist.life_plan import bar_ui, controller as c

    c.reset_controller_for_tests()
    def broken(**kwargs):
        raise RuntimeError("snapshot unavailable")

    assert bar_ui.start_life_prepare(None, None, snapshot_fn=broken)
    worker = bar_ui._PUMP
    if worker is not None:
        worker.join(3)
        assert not worker.is_alive()
    state = c.get_controller()
    assert not state.preparing
    assert not state.running
    assert state.start_blocked
    assert bar_ui.start_life_prepare(None, None, snapshot_fn=lambda **k: _snap({}))
    worker = bar_ui._PUMP
    if worker is not None:
        worker.join(3)
        assert not worker.is_alive()
