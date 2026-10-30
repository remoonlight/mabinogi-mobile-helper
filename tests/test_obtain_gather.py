from __future__ import annotations

from mm_assist.life_execute.obtain_gather import _next_spot, execute_obtain_gather
from mm_assist.life_plan.execute import try_execute
from mm_assist.life_plan.types import PlanItem


def test_next_spot_obtain_chain():
    assert _next_spot("item_info:煤炭", "煤炭") == "获得方法"
    assert _next_spot("obtain_method:煤炭", "煤炭") == "煤炭礦脈"
    assert _next_spot("obtain_method:樹液", "樹液") == "伐木"
    assert _next_spot("facility:metal:合金鋼錠", "煤炭") == "煤炭"
    assert _next_spot("main_hud", "煤炭") == ""
    assert _next_spot("fine:facility_detail", "樹液") == "樹液"
    assert _next_spot("facility:wood:高級木材", "樹液") == "樹液"
    assert _next_spot("facility:wood:木材+", "樹液") == "高級木材"
    assert _next_spot("facility:wood", "樹液") == "高級木材"
    assert _next_spot("inventory", "樹液") == "樹液"
    assert _next_spot("item_info:樹液", "樹液") == "获得方法"
    assert _next_spot("facility:metal", "樹液") == ""
    assert _next_spot("facility:metal:合金鋼錠", "樹液") == ""


def test_coal_not_in_obtain_mats():
    from mm_assist.life_execute.obtain_gather import OBTAIN_MATS

    assert "煤炭" not in OBTAIN_MATS
    assert "樹液" in OBTAIN_MATS
    r = execute_obtain_gather(mat="煤炭", items={"煤炭": 0})
    assert r.status == "bad_mat"


def test_obtain_gather_needs_window():
    r = execute_obtain_gather(mat="樹液", items={"樹液": 0})
    assert r.status == "need_context"


def test_try_execute_sap_obtain():
    item = PlanItem(
        id="p1",
        kind="gather",
        title="采集 樹木",
        skill="伐木",
        detail="樹木",
        recipe="樹液",
        gather_mode="process_mat",
    )
    assert try_execute(item, items={}).status == "need_context"


def test_sap_prepare_facility_then_inventory(monkeypatch):
    import numpy as np

    from mm_assist.life_execute import obtain_gather as og
    from mm_assist.life_plan.types import ExecuteResult

    og.reset_sap_visits()
    seen: list[dict[str, object]] = []

    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.node_inspect.compute_inspect",
        lambda frame, **k: {"node": "main_hud", "xy": None},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.gather_detail_visible",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: __import__(
            "mm_assist.life_execute.ui_logic", fromlist=["UiWhere"]
        ).UiWhere("main_hud", 1, 0, 0, "mem"),
    )

    def fake_prepare(target, **kw):
        seen.append({"target": target, "facility": kw.get("facility")})
        return ExecuteResult(ok=False, status="nav_stop", message="stop")

    monkeypatch.setattr("mm_assist.life_execute.ui_act.prepare_to", fake_prepare)

    def _run():
        return execute_obtain_gather(
            mat="樹液",
            items={"樹液": 99},
            do_click=lambda _x, _y: True,
            grab=lambda: np.zeros((32, 32, 3), dtype=np.uint8),
            read_items=lambda: {"樹液": 99},
        )

    r1 = _run()
    assert r1.ok is False
    assert seen[0]["target"] == "facility"
    assert seen[0]["facility"] == "wood"

    og._sap_visits = 1
    seen.clear()
    r2 = _run()
    assert r2.ok is False
    assert seen[0]["target"] == "inventory"
    assert seen[0]["facility"] is None
    og.reset_sap_visits()


def test_sap_inventory_equipment_switches_props_once_without_recipe_fallback(monkeypatch):
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.life_execute import obtain_gather as og, ui_act, ui_logic, ui_nav
    from mm_assist.life_plan import node_inspect
    from mm_assist.perception import item_sense

    frame = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/sap_inventory_equipment.png'))
    assert frame is not None
    monkeypatch.setattr(node_inspect, '_ocr_blob', lambda *a, **k: '')
    assert node_inspect.locate_spot('inventory', '樹液', frame) is None
    monkeypatch.setattr(node_inspect, 'compute_inspect', lambda *a, **k: {'node': 'inventory'})
    where = ui_logic.UiWhere('inventory', 1, 0, 0, 'fixture')
    monkeypatch.setattr(ui_logic, 'resolve_ui_node', lambda **k: where)
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda *a, **k: where)
    monkeypatch.setattr(ui_logic, 'gather_detail_visible', lambda *a, **k: False)
    monkeypatch.setattr(item_sense, 'locate_find_nearby', lambda *a, **k: {'ok': False})
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **k: False)
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr(ui_act, 'inspect_is_hud_idle', lambda: False)
    # Real bag and tab perception, with no sap icon: tab switches once and
    # remains blocked rather than re-clicking or navigating to a recipe.
    monkeypatch.setattr(ui_act, 'prepare_to', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('second visit must stay on the bag props path')))
    clicks = []
    og._sap_visits = 1
    try:
        result = execute_obtain_gather(mat='樹液', items={'樹液': 13},
            grab=lambda: frame, do_click=lambda x, y: clicks.append((x, y)) or True,
            read_stp=lambda: 0, sleep_fn=lambda _: None)
    finally:
        og.reset_sap_visits()
    assert not result.ok
    assert result.status == 'sap_card_missing', result
    assert clicks == [(1434, 1381)]


def test_sap_selected_props_without_card_does_not_click_tab_again(monkeypatch):
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.life_execute import obtain_gather as og, ui_act, ui_logic
    from mm_assist.life_plan import node_inspect

    frame = cv2.imread(str(ROOT / 'tests/fixtures/process_stance/sap_inventory_props_selected.png'))
    monkeypatch.setattr(node_inspect, '_ocr_blob', lambda *a, **k: '')
    monkeypatch.setattr(node_inspect, 'compute_inspect', lambda *a, **k: {'node': 'inventory'})
    monkeypatch.setattr(ui_logic, 'gather_detail_visible', lambda *a, **k: False)
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    clicks = []
    og._sap_visits = 1
    try:
        result = execute_obtain_gather(mat='樹液', items={'樹液': 111},
            grab=lambda: frame, do_click=lambda x,y: clicks.append((x,y)) or True,
            sleep_fn=lambda _: None)
    finally:
        og.reset_sap_visits()
    assert result.status == 'sap_card_missing'
    assert clicks == []


def test_sap_recipe_locate_fail_no_blind_click(monkeypatch):
    import numpy as np

    from mm_assist.life_execute import obtain_gather as og

    og.reset_sap_visits()
    clicks: list[tuple[int, int]] = []
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.node_inspect.compute_inspect",
        lambda frame, **k: {"node": "fine:facility_detail", "xy": None},
    )
    monkeypatch.setattr(
        "mm_assist.life_plan.node_inspect.locate_spot",
        lambda node, spot, frame: None,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.gather_detail_visible",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: __import__(
            "mm_assist.life_execute.ui_logic", fromlist=["UiWhere"]
        ).UiWhere("facility", 1, 0, 0, "mem"),
    )

    r = execute_obtain_gather(
        mat="樹液",
        items={"樹液": 3},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((32, 32, 3), dtype=np.uint8),
        read_items=lambda: {"樹液": 3},
    )
    assert r.ok is False
    assert clicks == []
    og.reset_sap_visits()


def test_sap_woodcut_row_waits_bag_not_nearby(monkeypatch):
    import numpy as np

    from mm_assist.life_execute import obtain_gather as og

    og.reset_sap_visits()
    clicks: list[tuple[int, int]] = []
    bag = {"n": 17}
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(
        "mm_assist.life_plan.node_inspect.compute_inspect",
        lambda frame, **k: {"node": "obtain_method:wood:樹液", "xy": None},
    )
    monkeypatch.setattr(
        "mm_assist.life_plan.node_inspect.locate_spot",
        lambda node, spot, frame: (11, 22) if spot == "伐木" else None,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.gather_detail_visible",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": False},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: __import__(
            "mm_assist.life_execute.ui_logic", fromlist=["UiWhere"]
        ).UiWhere("obtain_method", 1, 0, 0, "mem"),
    )

    def _click(x, y):
        clicks.append((x, y))
        bag["n"] = 21
        return True

    ticks = {"n": 0}

    def read_stp():
        ticks["n"] += 1
        return 1 if ticks["n"] == 1 else 0

    r = execute_obtain_gather(
        mat="樹液",
        items={"樹液": 17},
        do_click=_click,
        grab=lambda: np.zeros((32, 32, 3), dtype=np.uint8),
        read_items=lambda: {"樹液": bag["n"]},
        read_stp=read_stp,
        read_page=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
        now_fn=(lambda: 0.0),
        sleep_fn=lambda _s: None,
        hold_s=0.0,
    )
    assert r.ok is True
    assert clicks == [(11, 22)]
    assert "寻找附近" not in (r.message or "")
    og.reset_sap_visits()


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += float(s)


def test_sap_nearby_fallback_reuses_current_frame_hit_without_second_lookup(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import obtain_gather as og
    from mm_assist.life_execute import ui_logic
    from mm_assist.life_plan import node_inspect
    from mm_assist.perception import item_sense
    from mm_assist.life_execute.ui_logic import UiWhere
    frame = np.full((32, 32, 3), 30, np.uint8)
    reads, clicks = [], []
    def nearby(image, **kw):
        assert image is frame
        reads.append(True)
        return {'ok': True, 'click_client': [7, 8]}
    monkeypatch.setattr(ui_logic, 'gather_detail_visible', lambda *a, **k: False)
    monkeypatch.setattr(ui_logic, 'resolve_ui_node', lambda **k: UiWhere('gather_detail', 1, 0, 0, 'fixture'))
    monkeypatch.setattr(node_inspect, 'compute_inspect', lambda *a, **k: {'node': 'gather_detail:樹木'})
    monkeypatch.setattr(item_sense, 'locate_find_nearby', nearby)
    try:
        result = og.execute_obtain_gather(mat='樹液', grab=lambda: frame,
            do_click=lambda x,y: clicks.append((x,y)) or True, defer_completion=True)
        assert result.status == 'action_pending' and result.action_issued
        assert clicks == [(7, 8)] and len(reads) == 1
    finally:
        og.reset_sap_visits()


def test_obtain_nearby_gain_while_busy_is_not_done(monkeypatch):
    """冲突：旧 _finish_nearby 净增即 done。须等曾 stp=1 后新鲜主界面待机。"""
    import numpy as np

    from mm_assist.life_execute import obtain_gather as og
    from mm_assist.life_execute.ui_logic import UiWhere

    og.reset_sap_visits()
    clock = _Clock()
    bag = {"樹液": 3}
    monkeypatch.setattr(
        "mm_assist.life_plan.node_inspect.compute_inspect",
        lambda frame, **k: {"node": "gather_detail:樹木", "xy": None},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.gather_detail_visible",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 7, "cy": 8},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("gather_detail", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.recognized_click_xy",
        lambda hit: (int(hit["cx"]), int(hit["cy"])) if hit.get("ok") else None,
    )

    def read_items():
        bag["樹液"] = 8
        return dict(bag)

    r = execute_obtain_gather(
        mat="樹液",
        items={"樹液": 3},
        do_click=lambda *_a: True,
        grab=lambda: np.zeros((32, 32, 3), dtype=np.uint8),
        read_items=read_items,
        read_stp=lambda: 1,
        read_page=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
        now_fn=clock.now,
        sleep_fn=clock.sleep,
    )
    assert r.ok is False
    assert r.status == "gather_pending"
    og.reset_sap_visits()
