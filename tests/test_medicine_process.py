from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import time
from types import SimpleNamespace

from mm_assist.config import ROOT

from mm_assist.life_execute.medicine_process import (
    execute_facility_claim,
    execute_facility_feed,
    execute_medicine_claim,
    execute_medicine_feed,
    slot_changed,
    slot_signature,
)
from mm_assist.life_plan.execute import try_execute
from mm_assist.life_execute.ui_act import stack_has_facility_recipe as actual_recipe_stack
from mm_assist.life_plan.types import PlanItem
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests


@pytest.fixture(autouse=True)
def _reset_stage_gates():
    reset_stage_gates_for_tests()
    yield
    reset_stage_gates_for_tests()


@pytest.fixture(autouse=True)
def _no_live_facility_result(monkeypatch):
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_facility_result",
        lambda *a, **k: False,
    )
    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_facility_recipe",
        lambda: False, raising=False)
    # Successful scenarios supply an independently observed closed result page.
    # Persistent-popup cases below override it with the still-visible layer.
    monkeypatch.setattr('mm_assist.life_execute.ui_act.read_state_ui',
        lambda **_: _popup_stack_view())


def _popup_stack_view(*, present=False):
    names = ['AlteringFacilityPopupUIInstance']
    if present:
        names.append('FacilityResultUIInstance')
    return SimpleNamespace(fresh=True, pid_alive=True, raw={'ui_stack_ready': True,
        'ui_stack_ts': time.time(), 'ui_stack_names': names})


def test_client_of_menu_cx_cy():
    from mm_assist.life_execute.medicine_process import _client_of

    assert _client_of({"ok": True, "cx": 1620, "cy": 55}) == (1620, 55)
    assert _client_of({"ok": True, "click_client": [10, 20]}) == (10, 20)
    assert _client_of({"ok": False, "cx": 1, "cy": 2}) is None
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    b = a.copy()
    b[90:99, 5:40] = 200
    assert slot_changed(slot_signature(a), slot_signature(b)) is True
    assert slot_changed(slot_signature(a), slot_signature(a.copy())) is False


@pytest.mark.parametrize("recipe", ["合金鋼錠", "不存在的配方", ""])
def test_medicine_feed_rejects_foreign_recipe_without_snapshot_or_click(monkeypatch, recipe):
    clicks: list[tuple[int, int]] = []

    def boom_snap(**_k):
        raise AssertionError("foreign recipe must not read snapshot")

    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", boom_snap)
    r = execute_medicine_feed(
        recipe=recipe,
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: object(),
        read_items=lambda: (_ for _ in ()).throw(AssertionError("must not read bag")),
    )
    assert r.ok is False
    assert r.status == "bad_recipe"
    assert r.message == "不是藥品配方"
    assert clicks == []


def test_feed_refuses_unknown_recipe():
    r = execute_medicine_feed(recipe="合金鋼錠", items={"淨淨蘑菇孢子": 20, "裝水的瓶子": 2})
    assert r.ok is False
    assert r.status == "bad_recipe"


def test_feed_refuses_without_mats():
    r = execute_medicine_feed(recipe="淨淨蘑菇汁液", items={"淨淨蘑菇孢子": 1, "裝水的瓶子": 0})
    assert r.ok is False
    assert r.status == "no_mats"


def test_feed_skips_when_slots_full(monkeypatch):
    from mm_assist.life_execute.medicine_process import execute_facility_feed
    from mm_assist.life_plan.types import QueueSlot

    class Snap:
        ok = True
        items = {"鐵錠": 16, "煤炭": 16}
        facilities = {"metal": tuple(QueueSlot("鋼錠", False) for _ in range(5))}
        facility_slot_caps = {'metal': 5}

    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: Snap())
    r = execute_facility_feed(
        facility="metal",
        recipe="鋼錠",
        items={"鐵錠": 16, "煤炭": 16},
        grab=lambda: None,
        do_click=lambda *_a: True,
    )
    assert r.ok is False
    assert r.status == "slots_full"


@pytest.mark.parametrize('used, caps, expected', [
    (5, {'medicine': 6}, 'error'),
    (6, {'medicine': 6}, 'slots_full'),
    (5, {}, 'facility_unknown'),
])
def test_medicine_feed_checks_reported_capacity_before_opening_facility(used, caps, expected):
    from mm_assist.life_plan.types import QueueSlot

    def grab():
        raise RuntimeError('reached facility navigation')

    result = execute_medicine_feed(
        recipe='淨淨蘑菇汁液', items={'淨淨蘑菇孢子': 20, '裝水的瓶子': 2},
        facilities={'medicine': (QueueSlot('淨淨蘑菇汁液', False),) * used},
        facility_slot_caps=caps, grab=grab, do_click=lambda *a: pytest.fail('unexpected click'),
    )
    assert result.status == expected
    if expected == 'error':
        assert result.message == 'reached facility navigation'


def test_pump_skips_slots_full(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.controller import get_controller, reset_controller_for_tests, start_life
    from mm_assist.life_plan.types import ExecuteResult, QueueSlot
    from test_life_plan import _snap

    reset_controller_for_tests()
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(ok=False, status="slots_full", message="满"),
    )
    start_life(
        snapshot=_snap(
            {"合金鋼錠": 0, "銅礦石": 40, "鐵礦石": 40, "煤炭": 40},
            metal=tuple(QueueSlot("鋼錠", False) for _ in range(2)),
        )
    )
    first = get_controller().current.id
    pump_life_once(None, None)
    st = get_controller()
    assert st.current is not None
    assert st.current.id != first
    assert "metal" in st.blocked_facilities
    assert all(not (p.kind == "feed" and p.facility == "metal") for p in st.queue)


def test_pump_need_context_skips_unissued_item(monkeypatch):
    from mm_assist.life_plan.bar_ui import pump_life_once
    from mm_assist.life_plan.controller import get_controller, reset_controller_for_tests, start_life
    from mm_assist.life_plan.types import ExecuteResult
    from test_life_plan import _snap

    reset_controller_for_tests()
    monkeypatch.setattr(
        "mm_assist.life_plan.execute.try_execute",
        lambda *a, **k: ExecuteResult(ok=False, status="need_context", message="采集需要游戏窗"),
    )
    monkeypatch.setattr("mm_assist.life_plan.bar_ui.time.sleep", lambda *_a, **_k: None)
    start_life(
        snapshot=_snap(
            {"合金鋼錠": 0, "銅礦石": 40, "鐵礦石": 40, "煤炭": 40},
        )
    )
    first = get_controller().current.id
    pump_life_once(None, None)
    st = get_controller()
    assert st.current is not None
    assert st.current.id != first
    assert all(x.id != first for x in st.queue)


def test_feed_needs_window():
    r = execute_medicine_feed(
        recipe="淨淨蘑菇汁液",
        items={"淨淨蘑菇孢子": 20, "裝水的瓶子": 2},
    )
    assert r.status == "need_context"


def test_claim_needs_window():
    r = execute_medicine_claim(items={"淨淨蘑菇汁液": 0})
    assert r.status == "need_context"


def test_try_execute_metal_feed_needs_window():
    feed = PlanItem(
        id="p3",
        kind="feed",
        title="投料 合金鋼錠",
        facility="metal",
        recipe="合金鋼錠",
    )
    r = try_execute(feed, items={"鋼錠": 9, "銅礦石": 15, "煤炭": 8})
    assert r.status == "need_context"


def test_facility_feed_refuses_wrong_pair():
    from mm_assist.life_execute.medicine_process import execute_facility_feed

    r = execute_facility_feed(facility="metal", recipe="淨淨蘑菇汁液", items={"淨淨蘑菇孢子": 20})
    assert r.ok is False
    assert r.status == "bad_recipe"
    feed = PlanItem(
        id="p1",
        kind="feed",
        title="投料 淨淨蘑菇汁液",
        facility="medicine",
        recipe="淨淨蘑菇汁液",
    )
    r = try_execute(feed, items={"淨淨蘑菇孢子": 20, "裝水的瓶子": 2})
    assert r.status == "need_context"
    claim = PlanItem(id="p2", kind="claim", title="领取 medicine", facility="medicine")
    r2 = try_execute(claim, items={"淨淨蘑菇汁液": 0})
    assert r2.status == "need_context"


def test_claim_skips_without_button(monkeypatch):
    clicks: list[tuple[int, int]] = []
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    from mm_assist.life_execute.ui_logic import UiWhere

    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.resolve_ui_node",
        lambda **k: UiWhere("unknown", 1, 0, 0, "mem"),
    )

    r = execute_medicine_claim(
        items={"淨淨蘑菇汁液": 0},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: img,
        read_items=lambda: {"淨淨蘑菇汁液": 0},
    )
    assert r.ok is False
    assert clicks == []


def test_feed_already_on_facility_skips_menu(monkeypatch):
    from mm_assist.life_execute import medicine_process
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import LifeSnapshot

    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: LifeSnapshot(ok=True))

    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "medicine"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_hub_card",
        lambda *a, **k: {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_process_recipe",
        lambda *a, **k: {"ok": True, "cx": 111, "cy": 222},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_process_button",
        lambda *a, **k: {
            "ok": True,
            "cx": 333,
            "cy": 444,
            "label": "加工",
            "at_facility": True,
        },
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem+icon"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_facility_result",
        lambda *a, **k: False,
    )
    r = execute_medicine_feed(
        recipe="淨淨蘑菇汁液",
        items={"淨淨蘑菇孢子": 20, "裝水的瓶子": 2},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"淨淨蘑菇孢子": 20, "裝水的瓶子": 2},
    )
    assert r.ok is True
    assert clicks == [(111, 222), (333, 444)]


def test_feed_ocr_false_still_clicks_when_inventory_enough(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import LifeSnapshot

    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: LifeSnapshot(ok=True))
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "medicine"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_hub_card",
        lambda *a, **k: {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_process_recipe",
        lambda *a, **k: {"ok": True, "cx": 111, "cy": 222},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_process_button",
        lambda *a, **k: {
            "ok": True,
            "cx": 333,
            "cy": 444,
            "label": "加工",
            "at_facility": True,
            "materials_ok": False,
        },
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem+icon"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )
    r = execute_medicine_feed(
        recipe="淨淨蘑菇孢子",
        items={"淨淨蘑菇": 37, "紙": 11},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"淨淨蘑菇": 37, "紙": 11},
        sleep_fn=lambda *_a, **_k: None,
        read_stp=lambda: 0,
    )
    assert r.ok is True
    assert clicks[-1] == (333, 444)


def test_feed_skips_recipe_click_when_detail_open(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import LifeSnapshot

    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: LifeSnapshot(ok=True))
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "medicine"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_hub_card",
        lambda *a, **k: {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_process_recipe",
        lambda *a, **k: {"ok": True, "cx": 860, "cy": 762},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_process_button",
        lambda *a, **k: {
            "ok": True,
            "cx": 1487,
            "cy": 1356,
            "label": "加工",
            "at_facility": True,
            "materials_ok": True,
            "materials_ocr": "所需材料37/30 11/3淨淨蘑菇孢子",
        },
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility:medicine:淨淨蘑菇孢子", 1, 0, 0, "mem+icon"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_page",
        lambda **k: "facility:medicine:淨淨蘑菇孢子",
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "facility:medicine:淨淨蘑菇孢子", "zh": "制作-加工-药品加工-净净蘑菇孢子"},
    )
    r = execute_medicine_feed(
        recipe="淨淨蘑菇孢子",
        items={"淨淨蘑菇": 37, "紙": 11},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"淨淨蘑菇": 37, "紙": 11},
        sleep_fn=lambda *_a, **_k: None,
        read_stp=lambda: 0,
    )
    assert r.ok is True
    assert (860, 762) not in clicks
    assert clicks == [(1487, 1356)]


def test_feed_iron_passes_iron_ore_variant(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import LifeSnapshot

    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: LifeSnapshot(ok=True))
    seen: dict = {}

    def _loc(*_a, **k):
        seen.update(k)
        return {"ok": True, "click_client": [1066, 396]}

    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "metal"},
    )
    monkeypatch.setattr("mm_assist.perception.process_sense.locate_process_recipe", _loc)
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_process_button",
        lambda *a, **k: {"ok": True, "click_client": [900, 900], "label": "加工", "at_facility": True},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem+icon", facility_id="metal"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )
    r = execute_facility_feed(
        facility="metal",
        recipe="鐵錠",
        items={"鐵礦石": 20},
        do_click=lambda *_a, **_k: True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"鐵礦石": 10},
    )
    assert r.ok is True
    assert seen.get("variant") == "iron_ore"
    assert seen.get("name") == "鐵錠"


@pytest.mark.parametrize('defer_completion', [False, True])
def test_claim_done_by_facility_ui_not_bag(monkeypatch, defer_completion):
    from mm_assist.life_execute.ui_logic import UiWhere

    clicks: list[tuple[int, int]] = []
    dispatch=[]
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "medicine"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": True, "click_client": [80, 1200]},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": True, "click_client": [960, 1280]},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem+icon"),
    )
    stacks = iter([False, True, False, False, False])

    def _has(_text=None):
        try:
            return bool(next(stacks))
        except StopIteration:
            return False

    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_facility_result", _has)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )
    r = execute_medicine_claim(
        items={"淨淨蘑菇汁液": 0},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"淨淨蘑菇汁液": 0},
        defer_completion=defer_completion,
        on_action_issued=lambda:dispatch.append(list(clicks)),
    )
    assert r.ok is True
    assert r.status == "done"
    assert clicks == [(80, 1200), (960, 1280)]
    assert dispatch == [[(80, 1200)]]
    assert r.action_issued is defer_completion


def test_claim_clicks_go_equip_before_claim_all(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_stage_gates_for_tests()
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "metal"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_go_equip",
        lambda *a, **k: {"ok": True, "click_client": [240, 320]},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": True, "click_client": [80, 1200]},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": True, "click_client": [960, 1280]},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility:metal", 1, 0, 0, "mem+icon", facility_id="metal"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_facility_result",
        lambda *a, **k: len(clicks) == 2,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )
    r = execute_facility_claim(
        facility="metal",
        items={"鋼錠": 0},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"鋼錠": 0},
    )
    assert r.ok is True
    assert clicks[0] == (240, 320)
    assert clicks[1] == (80, 1200)


def test_claim_reopens_facility_if_walk_closed_ui(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import ExecuteResult
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_stage_gates_for_tests()
    clicks: list[tuple[int, int]] = []
    state = {"on_fac": True, "go": True}

    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "medicine"} if state["on_fac"] else {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_go_equip",
        lambda *a, **k: (
            {"ok": True, "click_client": [240, 320]}
            if state["go"] and state["on_fac"]
            else {"ok": False}
        ),
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: (
            {"ok": True, "click_client": [80, 1200]}
            if state["on_fac"] and not state["go"]
            else {"ok": False}
        ),
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": True, "click_client": [960, 1280]},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere(
            "facility:medicine" if state["on_fac"] else "main_hud",
            1,
            0,
            0,
            "mem+icon",
            facility_id="medicine" if state["on_fac"] else "",
        ),
    )
    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_common_popup", lambda *a, **k: False)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_facility_result", lambda *a, **k: len(clicks) == 2)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.inspect_facility_only_zh", lambda *a, **k: True)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.prepare_to",
        lambda *a, **k: state.__setitem__("on_fac", True) or ExecuteResult(True, "at_node", "facility"),
    )

    def _click(x, y):
        clicks.append((x, y))
        if (x, y) == (240, 320):
            state["on_fac"] = False
            state["go"] = False
        return True

    r = execute_facility_claim(
        facility="medicine",
        items={"淨淨蘑菇孢子": 0},
        do_click=_click,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"淨淨蘑菇孢子": 0},
        read_stp=lambda: 0,
    )
    assert r.ok is True
    assert clicks[0] == (240, 320)
    assert (80, 1200) in clicks


def test_claim_arrives_when_go_equip_button_vanishes(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_stage_gates_for_tests()
    clicks: list[tuple[int, int]] = []
    away = {"on": True}
    now = [1000.0]
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.wait_ui_params_refresh',
        lambda **_: now.__setitem__(0, now[0] + 0.2) or True)
    monkeypatch.setattr("mm_assist.life_execute.ui_logic.skip_wait_ui_params", lambda: False)
    monkeypatch.setattr("mm_assist.life_execute.medicine_process.time.time", lambda: now[0])
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "wood"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_go_equip",
        lambda *a, **k: {"ok": True, "click_client": [240, 320]} if away["on"] else {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": True, "click_client": [80, 1200]},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": True, "click_client": [960, 1280]},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility:wood", 1, 0, 0, "mem+icon", facility_id="wood"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: False,
    )
    stacks = iter([False, True, False, False, False, False])

    def _has(_text=None):
        try:
            return bool(next(stacks))
        except StopIteration:
            return False

    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_facility_result", _has)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )

    def _click(x, y):
        clicks.append((x, y))
        if (x, y) == (240, 320):
            away["on"] = False
        return True

    r = execute_facility_claim(
        facility="wood",
        items={"木材": 0},
        do_click=_click,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"木材": 0},
        sleep_fn=lambda s: now.__setitem__(0, now[0] + float(s)),
        read_stp=lambda: 0,
    )
    assert r.ok is True
    assert clicks[0] == (240, 320)
    assert (80, 1200) in clicks


def test_claim_arrives_while_still_walking_if_go_equip_gone(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_stage_gates_for_tests()
    clicks: list[tuple[int, int]] = []
    away = {"on": True}
    now = [1000.0]
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.wait_ui_params_refresh',
        lambda **_: now.__setitem__(0, now[0] + 0.2) or True)
    monkeypatch.setattr("mm_assist.life_execute.ui_logic.skip_wait_ui_params", lambda: False)
    monkeypatch.setattr("mm_assist.life_execute.medicine_process.time.time", lambda: now[0])
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "cloth"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_go_equip",
        lambda *a, **k: {"ok": True, "click_client": [240, 320]} if away["on"] else {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": True, "click_client": [80, 1200]},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": True, "click_client": [960, 1280]},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility:cloth", 1, 0, 0, "mem+icon", facility_id="cloth"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: False,
    )
    stacks = iter([False, True, False, False, False, False])

    def _has(_text=None):
        try:
            return bool(next(stacks))
        except StopIteration:
            return False

    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_facility_result", _has)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )

    def _click(x, y):
        clicks.append((x, y))
        if (x, y) == (240, 320):
            away["on"] = False
        return True

    r = execute_facility_claim(
        facility="cloth",
        items={"高級布料+": 0},
        do_click=_click,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"高級布料+": 0},
        sleep_fn=lambda s: now.__setitem__(0, now[0] + float(s)),
        read_stp=lambda: 1,
    )
    assert r.ok is True
    assert clicks[0] == (240, 320)
    assert (80, 1200) in clicks


def test_claim_confirms_go_popup_then_claims(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_stage_gates_for_tests()

    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "cloth"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": True, "click_client": [80, 1200]},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_common_popup_confirm",
        lambda *a, **k: {"ok": True, "click_client": [1100, 980]},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": True, "click_client": [960, 1280]},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem+icon", facility_id="cloth"),
    )
    popup = {"n": True}

    def _popup(_text=None, **_k):
        if popup["n"]:
            popup["n"] = False
            return True
        return False

    stacks = iter([False, True, False, False, False])

    def _has(_text=None):
        try:
            return bool(next(stacks))
        except StopIteration:
            return False

    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_common_popup", _popup)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_facility_result", _has)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )
    r = execute_facility_claim(
        facility="cloth",
        items={"布料": 0},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"布料": 0},
    )
    assert r.ok is True
    assert clicks[0] == (1100, 980)
    assert (80, 1200) in clicks
    assert (400, 980) not in clicks


def test_claim_result_overlay_blocks_done(monkeypatch):
    monkeypatch.setattr('mm_assist.life_execute.ui_act.read_state_ui',
        lambda **_: _popup_stack_view(present=True))
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_stage_gates_for_tests()

    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "medicine"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": True, "click_client": [80, 1200]},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": True, "click_client": [960, 1280]},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem+icon"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_facility_result",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )
    r = execute_medicine_claim(
        items={"淨淨蘑菇汁液": 3},
        do_click=lambda *_a: True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"淨淨蘑菇汁液": 9},
    )
    assert r.ok is False
    assert r.status == "confirm_miss"


def test_claim_on_facility_page_clicks_claim_not_back(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_execute.ui_nav import reset_edge_memo_for_tests
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_stage_gates_for_tests()
    reset_edge_memo_for_tests()
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "metal"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": True, "click_client": [80, 1200]},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": True, "click_client": [960, 1280]},
    )
    where = UiWhere("facility", 1, 0, 0, "mem+icon", facility_id="metal")
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: where,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.resolve_ui_node",
        lambda **k: where,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_page",
        lambda **k: "facility:metal",
    )
    monkeypatch.setattr("mm_assist.life_execute.ui_nav.locate_edge", lambda *a, **k: (7, 8))
    stacks = iter([False, True, False, False, False])

    def _has(_text=None):
        try:
            return bool(next(stacks))
        except StopIteration:
            return False

    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_facility_result", _has)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.stack_has_common_popup",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )
    r = execute_facility_claim(
        facility="metal",
        items={"鋼錠": 0},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"鋼錠": 0},
    )
    assert r.ok is True
    assert r.status == "done"
    assert clicks[0] == (80, 1200)
    assert (7, 8) not in clicks


def test_claim_unknown_facility_does_not_click(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_execute.ui_nav import reset_edge_memo_for_tests
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_stage_gates_for_tests()
    reset_edge_memo_for_tests()

    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": True, "click_client": [80, 1200]},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_page",
        lambda **k: "",
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr("mm_assist.life_execute.ui_nav.locate_edge", lambda *a, **k: (7, 8))
    r = execute_facility_claim(
        facility="metal",
        items={"鋼錠": 0},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"鋼錠": 0},
    )
    assert r.ok is False
    assert (80, 1200) not in clicks


def test_claim_result_overlay_dismisses_before_unknown_identity(monkeypatch):
    monkeypatch.setattr('mm_assist.life_execute.ui_act.read_state_ui',
        lambda **_: _popup_stack_view(present=True))
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_execute.ui_nav import reset_edge_memo_for_tests
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_stage_gates_for_tests()
    reset_edge_memo_for_tests()
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": True, "click_client": [80, 1200]},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_result_confirm",
        lambda *a, **k: {"ok": True, "click_client": [960, 1280]},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_facility_result", lambda *a, **k: True)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_common_popup", lambda *a, **k: False)
    r = execute_facility_claim(
        facility="wood",
        items={"木材+": 0},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"木材+": 0},
    )
    assert r.ok is False
    assert r.status == "confirm_miss"
    assert clicks == [(960, 1280)]


def test_claim_known_facility_missing_button_is_no_claim_ui(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_execute.ui_nav import reset_edge_memo_for_tests
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_stage_gates_for_tests()
    reset_edge_memo_for_tests()
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "wood"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_page",
        lambda **k: "facility:wood",
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem", facility_id="wood"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem", facility_id="wood"),
    )
    r = execute_facility_claim(
        facility="wood",
        items={"木材": 0},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"木材": 0},
    )
    assert r.ok is False
    assert r.status == "no_claim_ui"
    assert "全部領取" in r.message
    assert clicks == []


def test_claim_wrong_facility_leaves_page(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_execute.ui_nav import reset_edge_memo_for_tests
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests

    reset_stage_gates_for_tests()
    reset_edge_memo_for_tests()

    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr("mm_assist.life_execute.ui_logic.resolve_navigation", lambda *a, **k: UiWhere("facility", 1, 0, 0, "direct"))
    monkeypatch.setattr("mm_assist.life_execute.ui_nav.resolve_navigation", lambda *a, **k: UiWhere("facility", 1, 0, 0, "direct"))
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": False, "facility": "wood", "expected": "metal"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": True, "click_client": [80, 1200]},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem+icon"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.resolve_ui_node",
        lambda **k: UiWhere("facility", 1, 0, 0, "mem+icon"),
    )
    monkeypatch.setattr("mm_assist.life_execute.ui_nav.locate_edge", lambda *a, **k: (7, 8))
    r = execute_facility_claim(
        facility="metal",
        items={"鋼錠": 0},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"鋼錠": 0},
        sleep_fn=lambda _s: None,
    )
    assert clicks
    assert clicks[0] == (7, 8)
    assert (80, 1200) not in clicks
    assert r.ok is False


@pytest.mark.parametrize(
    "items,facilities,snap_ok,snap_facilities,want_status,expect_snap",
    [
        ({"淨淨蘑菇孢子": 1, "裝水的瓶子": 0}, None, True, {}, "no_mats", False),
        (None, None, True, {}, "no_mats", True),
        (
            {"淨淨蘑菇孢子": 20, "裝水的瓶子": 2},
            None,
            False,
            None,
            "facility_unknown",
            True,
        ),
        (
            {"淨淨蘑菇孢子": 20, "裝水的瓶子": 2},
            "full",
            True,
            {},
            "slots_full",
            False,
        ),
    ],
)
def test_medicine_feed_facade_preserves_guards_and_forwards_options(
    monkeypatch, items, facilities, snap_ok, snap_facilities, want_status, expect_snap
):
    from mm_assist.life_plan.types import QueueSlot, SLOT_CAP

    snaps: list[int] = []

    class Snap:
        ok = snap_ok
        items = {"淨淨蘑菇孢子": 1, "裝水的瓶子": 0}
        facilities = snap_facilities
        facility_slot_caps = {'medicine': SLOT_CAP}

    def take(**_k):
        snaps.append(1)
        return Snap()

    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", take)
    fac_arg = None
    if facilities == "full":
        fac_arg = {"medicine": tuple(QueueSlot("淨淨蘑菇汁液", False) for _ in range(SLOT_CAP))}
    r = execute_medicine_feed(
        recipe="淨淨蘑菇汁液",
        items=items,
        grab=lambda: None,
        do_click=lambda *_a: True,
        sleep_fn=lambda *_a, **_k: None,
        read_stp=lambda: 0,
        facilities=fac_arg,
    )
    assert r.ok is False
    assert r.status == want_status
    if expect_snap:
        assert snaps
    else:
        assert snaps == []


def test_medicine_feed_ignores_redundant_facility_and_unknown_kwargs(monkeypatch):
    snaps: list[int] = []
    clicks: list[tuple[int, int]] = []

    def boom_snap(**_k):
        snaps.append(1)
        raise AssertionError("no_mats with items must not snapshot")

    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", boom_snap)
    r = execute_medicine_feed(
        recipe="淨淨蘑菇汁液",
        items={},
        facility="metal",
        ignored_option=True,
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: object(),
    )
    assert r.ok is False
    assert r.status == "no_mats"
    assert snaps == []
    assert clicks == []


def test_medicine_feed_forwards_run_options_through_generic_slot_guard(monkeypatch):
    from mm_assist.life_execute import medicine_process as mp
    from mm_assist.life_plan.types import QueueSlot, SLOT_CAP

    snaps: list[int] = []
    clicks: list[tuple[int, int]] = []
    seen: dict[str, object] = {}
    orig = mp._run_process

    def wrap(*a, **k):
        seen.update(k)
        return orig(*a, **k)

    monkeypatch.setattr(mp, "_run_process", wrap)
    monkeypatch.setattr(
        "mm_assist.life_plan.snapshot.take_snapshot",
        lambda **_k: snaps.append(1) or (_ for _ in ()).throw(AssertionError("slots kwargs skip snapshot")),
    )
    sleeper = lambda *_a, **_k: None
    reader = lambda: 0
    fac = {"medicine": tuple(QueueSlot("淨淨蘑菇汁液", False) for _ in range(SLOT_CAP))}
    r = execute_medicine_feed(
        recipe="淨淨蘑菇汁液",
        items={"淨淨蘑菇孢子": 20, "裝水的瓶子": 2},
        facility="metal",
        ignored_option=1,
        grab=lambda: None,
        do_click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=sleeper,
        read_stp=reader,
        facilities=fac,
    )
    assert r.ok is False
    assert r.status == "slots_full"
    assert snaps == []
    assert clicks == []
    assert seen.get("sleep_fn") is sleeper
    assert seen.get("read_stp") is reader
    assert seen.get("facilities") is fac
    assert seen.get("facility") == "medicine"


def _process_stance_bgr(name: str) -> np.ndarray:
    path = ROOT / "tests" / "fixtures" / "process_stance" / name
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def _patch_metal_alloy_ready(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import LifeSnapshot

    monkeypatch.setattr(
        "mm_assist.perception.process_sense._ocr_lines",
        lambda *a, **k: [],
    )
    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: LifeSnapshot(ok=True))
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "metal"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.locate_hub_card",
        lambda *a, **k: {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility:metal:合金鋼錠", 1, 0, 0, "mem+icon", facility_id="metal"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.inspect_facility_only_zh",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_page",
        lambda **k: "facility:metal:合金鋼錠",
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "facility:metal:合金鋼錠", "zh": "制作-加工-金属加工-合金鋼锭"},
    )


@pytest.mark.parametrize('away_pair',[
    ('away_stable.png','away_stable.png'),
    ('wood_away_anim_first.png','wood_away_anim_fresh.png'),
])
def test_feed_stale_here_then_away_travels_before_feed(monkeypatch,away_pair):
    _patch_metal_alloy_ready(monkeypatch)
    here = _process_stance_bgr("here_formal.png")
    away = _process_stance_bgr(away_pair[0])
    fresh_away = _process_stance_bgr(away_pair[1])
    arrived = _process_stance_bgr("here_stable.png")
    facility = _process_stance_bgr("facility_no_detail.png")
    frames = iter(
        [
            facility,
            here,
            here,
            away,
            fresh_away,
            arrived,
            arrived,
            arrived,
            arrived,
            arrived,
            arrived,
            facility,
        ]
    )

    def _grab():
        try:
            return next(frames)
        except StopIteration:
            return facility

    clicks: list[tuple[int, int]] = []
    labels: list[str] = []
    r = execute_facility_feed(
        facility="metal",
        recipe="合金鋼錠",
        items={"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=_grab,
        read_items=lambda: {"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        sleep_fn=lambda *_a, **_k: None,
        read_stp=lambda: 0,
        on_progress=labels.append,
    )
    assert r.ok is True
    assert clicks
    assert clicks[0] == (1487, 1356)
    assert (1520, 1355) not in clicks
    assert clicks[-1][0] in {1486, 1487}
    assert clicks[-1][1] in {1355, 1356}
    assert "前往加工" in labels
    feed_at = next(i for i, msg in enumerate(labels) if msg == "加工")
    go_at = next(i for i, msg in enumerate(labels) if msg == "前往加工")
    assert go_at < feed_at


def test_feed_stable_here_direct_feed(monkeypatch):
    _patch_metal_alloy_ready(monkeypatch)
    here = _process_stance_bgr("here_stable.png")
    clicks: list[tuple[int, int]] = []
    labels: list[str] = []
    r = execute_facility_feed(
        facility="metal",
        recipe="合金鋼錠",
        items={"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: here,
        read_items=lambda: {"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        sleep_fn=lambda *_a, **_k: None,
        read_stp=lambda: 0,
        on_progress=labels.append,
    )
    assert r.ok is True
    assert clicks == [(1486, 1355)]
    assert "前往加工" not in labels
    assert "加工" in labels


def test_feed_wrong_facility_does_not_click_process(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import LifeSnapshot

    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: LifeSnapshot(ok=True))
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": False, "facility": "wood"},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility:wood", 1, 0, 0, "mem+icon", facility_id="wood"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.prepare_to",
        lambda *a, **k: type("R", (), {"ok": False, "status": "leave_wrong", "message": "错页"})(),
    )
    r = execute_facility_feed(
        facility="metal",
        recipe="合金鋼錠",
        items={"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: _process_stance_bgr("facility_no_detail.png"),
        read_items=lambda: {"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
    )
    assert r.ok is False
    assert clicks == []


def test_wood_feed_from_live_metal_recipe_navigates_before_recipe_search(monkeypatch):
    from mm_assist.life_execute import ui_act, ui_logic, ui_nav
    from mm_assist.life_plan import node_inspect
    from mm_assist.perception import process_sense

    frame = _process_stance_bgr("metal_recipe_wood_target.png")
    monkeypatch.setattr(process_sense, "ocr_roi", lambda *a, **k: [])
    monkeypatch.setattr(node_inspect, "_ocr_blob", lambda *a, **k: "")
    node_inspect._reset_facility_fid_cache()
    assert process_sense.detect_facility_page(frame, expected="wood")["facility"] == "metal"
    assert process_sense.locate_process_button(frame)["at_facility"] is True
    monkeypatch.setattr(ui_act, "stack_has_common_popup", lambda *a, **k: False)
    monkeypatch.setattr(ui_act, "pump_aborted", lambda: False)
    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: "facility:metal:鋼錠")
    monkeypatch.setattr(ui_logic, "read_inspect_record", lambda **k: {})
    monkeypatch.setattr(ui_logic, "resolve_navigation", lambda *a, **k:
        ui_logic.UiWhere("facility:metal:鋼錠", 1, 0, 0, "mem+icon", facility_id="metal"))
    # Keep real prepare_to -> ensure_node. Stop at the first navigation edge;
    # searching wood recipes or dispatching process input on metal is a defect.
    exits = []
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: exits.append(k) or None)
    monkeypatch.setattr(process_sense, "locate_process_recipe", lambda *a, **k:
        pytest.fail("wood recipe must not be searched on the metal page"))
    clicks = []
    result = execute_facility_feed(facility="wood", recipe="高級木材",
        items={"木材+": 3, "高級原木": 100, "樹液": 8}, facilities={"wood": ()},
        grab=lambda: frame, do_click=lambda x, y: clicks.append((x, y)) or True,
        sleep_fn=lambda _: None, read_stp=lambda: 0)
    assert not result.ok
    assert exits and clicks == [], result


def test_feed_unknown_overlay_does_not_click_process(monkeypatch):
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import LifeSnapshot

    monkeypatch.setattr("mm_assist.life_plan.snapshot.take_snapshot", lambda **k: LifeSnapshot(ok=True))
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("overlay", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.resolve_ui_node",
        lambda **k: UiWhere("overlay", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.prepare_to",
        lambda *a, **k: type("R", (), {"ok": False, "status": "unknown", "message": "未知页"})(),
    )
    r = execute_facility_feed(
        facility="metal",
        recipe="合金鋼錠",
        items={"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_items=lambda: {"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
    )
    assert r.ok is False
    assert clicks == []


def test_feed_unknown_stp_does_not_click_process(monkeypatch):
    _patch_metal_alloy_ready(monkeypatch)
    here = _process_stance_bgr("here_stable.png")
    clicks: list[tuple[int, int]] = []
    r = execute_facility_feed(
        facility="metal",
        recipe="合金鋼錠",
        items={"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: here,
        read_items=lambda: {"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        sleep_fn=lambda *_a, **_k: None,
        read_stp=lambda: None,
    )
    assert r.ok is False
    assert r.status == "stp_unknown"
    assert clicks == []


def test_feed_late_cancel_does_not_click(monkeypatch):
    _patch_metal_alloy_ready(monkeypatch)
    monkeypatch.setattr("mm_assist.life_execute.ui_act.pump_aborted", lambda: True)
    clicks: list[tuple[int, int]] = []
    r = execute_facility_feed(
        facility="metal",
        recipe="合金鋼錠",
        items={"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: _process_stance_bgr("here_stable.png"),
        read_items=lambda: {"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        sleep_fn=lambda *_a, **_k: None,
        read_stp=lambda: 0,
    )
    assert r.ok is False
    assert r.status == "cancel"
    assert clicks == []


def test_feed_away_timeout_does_not_retry(monkeypatch):
    from mm_assist.life_execute import medicine_process as mp

    _patch_metal_alloy_ready(monkeypatch)
    # Offline travel has no popup; do not read the running game's stack.
    monkeypatch.setattr('mm_assist.life_execute.ui_act.stack_has_common_popup', lambda *a, **k: False)
    monkeypatch.setattr(mp, "WAIT_WALK_S", 0.0)
    away = _process_stance_bgr("away_stable.png")
    clicks: list[tuple[int, int]] = []
    labels: list[str] = []
    r = execute_facility_feed(
        facility="metal",
        recipe="合金鋼錠",
        items={"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: away,
        read_items=lambda: {"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        sleep_fn=lambda *_a, **_k: None,
        read_stp=lambda: 0,
        on_progress=labels.append,
    )
    assert r.ok is False
    assert r.status == "walk_timeout"
    assert clicks == [(1487, 1356)]
    assert "加工" not in labels
    assert labels.count("前往加工") == 1


def test_feed_cancel_during_travel_stops_observation_without_retry(monkeypatch):
    from types import SimpleNamespace
    from mm_assist.life_execute import medicine_process as mp, ui_act

    _patch_metal_alloy_ready(monkeypatch)
    # Offline travel has no popup; do not read the running game's stack.
    monkeypatch.setattr('mm_assist.life_execute.ui_act.stack_has_common_popup', lambda *a, **k: False)
    clock = [0.0]
    monkeypatch.setattr(mp, "WAIT_WALK_S", 5.0)
    monkeypatch.setattr(mp, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(ui_act, "pump_aborted", lambda: clock[0] >= 2)
    clicks = []
    result = execute_facility_feed(facility="metal", recipe="合金鋼錠",
        items={"鋼錠": 5, "銅礦石": 539, "煤炭": 786}, facilities={"metal": ()},
        grab=lambda: _process_stance_bgr("away_stable.png"),
        do_click=lambda x,y: clicks.append((x,y)) or True,
        sleep_fn=lambda s: clock.__setitem__(0, clock[0]+s), read_stp=lambda: 0)
    assert result.status == "cancel", result
    assert clock[0] < 10 and len(clicks) == 1


def test_claim_without_result_page_cannot_report_done(monkeypatch):
    monkeypatch.setattr("mm_assist.perception.process_sense.detect_facility_page",
        lambda *a, **k: {"ok": True, "facility": "metal"})
    monkeypatch.setattr("mm_assist.perception.process_sense.locate_claim_all",
        lambda *a, **k: {"ok": True, "click_client": [80, 1200]})
    monkeypatch.setattr("mm_assist.perception.process_sense.locate_go_equip", lambda *a, **k: {"ok": False})
    monkeypatch.setattr("mm_assist.life_execute.ui_act.stack_has_common_popup", lambda *a, **k: False)
    clicks = []
    result = execute_facility_claim(facility="metal", items={},
        grab=lambda: np.zeros((64,64,3), dtype=np.uint8),
        do_click=lambda x,y: clicks.append((x,y)) or True, defer_completion=True)
    assert result.status == "claim_result_missing" and not result.ok
    assert clicks == [(80, 1200)]


def test_claim_from_live_recipe_exits_layer_before_claim(monkeypatch):
    from mm_assist.life_execute import ui_act, ui_logic, ui_nav
    from mm_assist.perception import process_sense as ps

    recipe = _process_stance_bgr("metal_recipe_wood_target.png")
    overview = _process_stance_bgr("metal_standalone.png")
    clicks = []
    monkeypatch.setattr(ui_act, "pump_aborted", lambda: False)
    monkeypatch.setattr(ui_act, "stack_has_facility_recipe", lambda: not clicks, raising=False)
    monkeypatch.setattr(ui_act, "stack_has_common_popup", lambda *a, **k: False)
    monkeypatch.setattr(ui_act, "stack_has_facility_result", lambda *a, **k: len(clicks) == 2)
    monkeypatch.setattr(ui_logic, "read_inspect_record", lambda **k: {})
    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: "facility:metal")
    monkeypatch.setattr(ui_logic, "resolve_navigation", lambda *a, **k:
        ui_logic.UiWhere("facility:metal:鋼錠" if not clicks else "facility:metal", 1,0,0,"fixture"))
    close = ui_nav.locate_edge({'id': 'facility_back_craft_hub'}, recipe)
    assert close is not None
    monkeypatch.setattr(ps, "ocr_roi", lambda *a, **k: [])
    monkeypatch.setattr(ps, "locate_go_equip", lambda *a, **k: {"ok": False})
    # The underlying control matches on the dimmed recipe image too.
    monkeypatch.setattr(ps, "locate_claim_all", lambda *a, **k: {"ok": True, "click_client": [80,1200]})
    monkeypatch.setattr(ps, "locate_result_confirm", lambda *a, **k: {"ok": True, "click_client": [960,1280]})
    result = execute_facility_claim(facility="metal", items={},
        grab=lambda: recipe if not clicks else overview,
        do_click=lambda x,y: clicks.append((x,y)) or True,
        sleep_fn=lambda _: None, read_stp=lambda: 0, defer_completion=True)
    assert result.ok and clicks == [close, (80,1200), (960,1280)]


@pytest.mark.parametrize("fresh,alive,ready,names,expected", [
    (True,True,True,["AlteringPanelPopupUIInstance"],True),
    (False,True,True,["AlteringPanelPopupUIInstance"],False),
    (True,False,True,["AlteringPanelPopupUIInstance"],False),
    (True,True,False,["AlteringPanelPopupUIInstance"],False),
    (True,True,True,["AlteringFacilityPopupUIInstance"],False),
])
def test_recipe_layer_requires_fresh_independent_stack(monkeypatch, fresh,alive,ready,names,expected):
    from types import SimpleNamespace
    monkeypatch.setattr("mm_assist.perception.state_ui.read_state_ui", lambda:
        SimpleNamespace(fresh=fresh,pid_alive=alive,raw={"ui_stack_ready":ready,"ui_stack_names":names}))
    assert actual_recipe_stack() is expected


def test_feed_final_unknown_then_here_does_not_click_process(monkeypatch):
    _patch_metal_alloy_ready(monkeypatch)
    here = _process_stance_bgr("here_stable.png")
    unknown = _process_stance_bgr("facility_no_detail.png")
    frames = iter(
        [
            here,
            here,
            here,
            here,
            unknown,
            here,
            unknown,
            unknown,
        ]
    )

    def _grab():
        try:
            return next(frames)
        except StopIteration:
            return unknown

    clicks: list[tuple[int, int]] = []
    labels: list[str] = []
    r = execute_facility_feed(
        facility="metal",
        recipe="合金鋼錠",
        items={"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=_grab,
        read_items=lambda: {"鋼錠": 5, "銅礦石": 539, "煤炭": 786},
        sleep_fn=lambda *_a, **_k: None,
        read_stp=lambda: 0,
        on_progress=labels.append,
    )
    assert r.ok is False
    assert r.status in {"no_process_btn", "btn_unstable", "btn_ambiguous"}
    assert "加工" not in labels
    assert clicks == []
