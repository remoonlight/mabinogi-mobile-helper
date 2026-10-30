from __future__ import annotations

from mm_assist.life_execute.fill_bottle import execute_fill_bottle, well_template_ready
from mm_assist.life_plan.execute import try_execute
from mm_assist.life_plan.planner import plan_life
from mm_assist.life_plan.types import PlanItem
from test_life_plan import _snap


def test_well_template_ready():
    assert well_template_ready() is True


def test_fill_refuses_without_empty_bottle():
    r = execute_fill_bottle(items={"空瓶": 0, "裝水的瓶子": 0})
    assert r.ok is False
    assert r.status == "no_empty_bottle"


def test_fill_refuses_without_window():
    r = execute_fill_bottle(items={"空瓶": 5, "裝水的瓶子": 0})
    assert r.ok is False
    assert r.status == "need_context"


def test_fill_succeeds_on_bag_delta():
    clicks: list[tuple[int, int]] = []
    bag = {"空瓶": 5, "裝水的瓶子": 10}

    def locate():
        return {"ok": True, "cx": 100, "cy": 200}

    def click(x: int, y: int) -> bool:
        clicks.append((x, y))
        bag["裝水的瓶子"] += 1
        bag["空瓶"] -= 1
        return True

    r = execute_fill_bottle(
        items=dict(bag),
        locate_well=locate,
        do_click=click,
        read_items=lambda: bag,
    )
    assert r.ok is True
    assert r.status == "done"
    assert clicks == [(100, 200)]
    assert bag["裝水的瓶子"] == 11


def test_fill_stops_after_two_clicks_no_delta():
    n = {"c": 0}

    def click(x: int, y: int) -> bool:
        n["c"] += 1
        return True

    r = execute_fill_bottle(
        items={"空瓶": 2, "裝水的瓶子": 0},
        locate_well=lambda: {"ok": True, "cx": 1, "cy": 2},
        do_click=click,
        read_items=lambda: {"空瓶": 2, "裝水的瓶子": 0},
    )
    assert r.ok is False
    assert r.status == "fill_no_delta"
    assert n["c"] == 2


def test_try_execute_well_gather_uses_fill_path():
    plan = plan_life(
        _snap(
            {
                "合金鋼錠": 100,
                "高級木材": 100,
                "高級布料": 100,
                "淨淨蘑菇汁液": 0,
                "淨淨蘑菇孢子": 360,
                "紙": 0,
                "空瓶": 36,
                "裝水的瓶子": 0,
            }
        )
    )
    well = next(p for p in plan if p.kind == "gather" and p.detail == "水井")
    r = try_execute(well, items={"空瓶": 36, "裝水的瓶子": 0})
    assert r.status == "need_context"


def test_try_execute_mushroom_uses_herb_path():
    item = PlanItem(
        id="p1",
        kind="gather",
        title="采集 淨淨蘑菇",
        skill="採集藥草",
        detail="淨淨蘑菇",
        gather_mode="process_mat",
    )
    r = try_execute(item, items={"淨淨蘑菇": 10})
    assert r.status == "need_context"


def test_try_execute_cloth_claim_needs_window():
    from mm_assist.life_plan.types import PlanItem

    claim = PlanItem(id="p1", kind="claim", title="领取", facility="cloth")
    r = try_execute(claim, items={})
    assert r.status == "need_context"


def test_try_execute_coal_uses_skill_path(monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult, PlanItem

    called = {"skill": 0, "obtain": 0}

    monkeypatch.setattr(
        "mm_assist.life_execute.skill_nearby.execute_skill_nearby",
        lambda *a, **k: called.__setitem__("skill", called["skill"] + 1)
        or ExecuteResult(ok=True, status="done", message="skill"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.obtain_gather.execute_obtain_gather",
        lambda *a, **k: called.__setitem__("obtain", called["obtain"] + 1)
        or ExecuteResult(ok=True, status="done", message="obtain"),
    )
    item = PlanItem(
        id="p1",
        kind="gather",
        title="采集 煤炭礦脈",
        skill="採礦",
        detail="煤炭礦脈",
        recipe="煤炭",
        gather_mode="process_mat",
    )
    r = try_execute(item, items={})
    assert r.status == "done"
    assert called["skill"] == 1
    assert called["obtain"] == 0


def test_try_execute_iron_vein_uses_skill_path():
    from mm_assist.life_plan.types import PlanItem

    item = PlanItem(
        id="p1",
        kind="gather",
        title="采集 鐵礦脈",
        skill="採礦",
        detail="鐵礦脈",
        recipe="鐵礦石",
        gather_mode="process_mat",
    )
    r = try_execute(item, items={})
    assert r.status == "need_context"
    item = PlanItem(
        id="p1",
        kind="gather",
        title="采集 蜘蛛網",
        skill="日常採集",
        detail="蜘蛛網",
        gather_mode="skill",
    )
    r = try_execute(item)
    assert r.status == "need_context"


def test_skill_nearby_waits_stp_then_counts(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import skill_nearby as sn
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import ExecuteResult

    stp = [1, 0]
    targets: list[str] = []

    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:淨淨蘑菇')
    monkeypatch.setattr(sn, "HOLD_IDLE_S", 0.0)
    monkeypatch.setattr(sn.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_skill_item",
        lambda *a, **k: {"ok": True, "cx": 10, "cy": 20},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("main_hud", 0, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "main_hud", "zh": "主界面-待机"},
    )

    def fake_prepare(target, **_k):
        targets.append(str(target))
        return ExecuteResult(ok=True, status="at_node", message=str(target))

    monkeypatch.setattr("mm_assist.life_execute.ui_act.prepare_to", fake_prepare)

    clicks: list[tuple[int, int]] = []

    r = execute_skill_nearby(
        skill_id="herb",
        item_name="淨淨蘑菇",
        output="淨淨蘑菇",
        items={"淨淨蘑菇": 5},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_stp=lambda: stp.pop(0) if stp else 0,
        read_page=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
    )
    assert targets == ["gather_skill"]
    assert len(clicks) == 2
    assert r.ok is True
    assert r.status == "done"
    assert "4" not in r.message
    assert "待机" in r.message


def test_skill_nearby_detail_clicks_nearby_without_main_hud(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import skill_nearby as sn
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import ExecuteResult

    stp = [1, 0]
    targets: list[str] = []
    clicks: list[tuple[int, int]] = []

    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:淨淨蘑菇')
    monkeypatch.setattr(sn, "HOLD_IDLE_S", 0.0)
    monkeypatch.setattr(sn.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("gather_detail", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "gather_detail", "zh": "技能详情-淨淨蘑菇"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_skill_item",
        lambda *a, **k: {"ok": True, "cx": 10, "cy": 20},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )

    def fake_prepare(target, **_k):
        targets.append(str(target))
        return ExecuteResult(ok=True, status="at_node", message=str(target))

    monkeypatch.setattr("mm_assist.life_execute.ui_act.prepare_to", fake_prepare)

    r = execute_skill_nearby(
        skill_id="herb",
        item_name="淨淨蘑菇",
        output="淨淨蘑菇",
        items={"淨淨蘑菇": 5},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_stp=lambda: stp.pop(0) if stp else 0,
        read_page=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
    )
    assert targets == []
    assert clicks == [(30, 40)]
    assert r.ok is True
    assert r.status == "done"


def test_skill_nearby_life_grid_does_not_prepare_main_hud(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import skill_nearby as sn
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import ExecuteResult

    stp = [1, 0]
    targets: list[str] = []
    clicks: list[tuple[int, int]] = []

    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:淨淨蘑菇')
    monkeypatch.setattr(sn, "HOLD_IDLE_S", 0.0)
    monkeypatch.setattr(sn.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("life_grid", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "life_grid", "zh": "角色-生活技能"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_skill_item",
        lambda *a, **k: {"ok": True, "cx": 10, "cy": 20},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )

    def fake_prepare(target, **_k):
        targets.append(str(target))
        return ExecuteResult(ok=True, status="at_node", message=str(target))

    monkeypatch.setattr("mm_assist.life_execute.ui_act.prepare_to", fake_prepare)

    r = execute_skill_nearby(
        skill_id="herb",
        item_name="淨淨蘑菇",
        output="淨淨蘑菇",
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_stp=lambda: stp.pop(0) if stp else 0,
        read_page=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
    )
    assert targets == ["gather_skill"]
    assert len(clicks) == 2
    assert r.ok is True


def test_skill_nearby_inspect_hud_clicks_despite_resolve_unknown(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import skill_nearby as sn
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import ExecuteResult

    stp = [0, 1, 0]
    targets: list[str] = []

    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:淨淨蘑菇')
    monkeypatch.setattr(sn, "HOLD_IDLE_S", 0.0)
    monkeypatch.setattr(sn.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_skill_item",
        lambda *a, **k: {"ok": True, "cx": 10, "cy": 20},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("unknown", 0, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "main_hud", "zh": "主界面-待机"},
    )

    def fake_prepare(target, **_k):
        targets.append(str(target))
        return ExecuteResult(ok=True, status="at_node", message=str(target))

    monkeypatch.setattr("mm_assist.life_execute.ui_act.prepare_to", fake_prepare)

    clicks: list[tuple[int, int]] = []
    r = execute_skill_nearby(
        skill_id="herb",
        item_name="淨淨蘑菇",
        output="淨淨蘑菇",
        items={"淨淨蘑菇": 5},
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_stp=lambda: stp.pop(0) if stp else 0,
        read_page=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
    )
    assert r.status != "unknown_ui"
    assert targets == ["gather_skill"]
    assert len(clicks) == 2
    assert r.ok is True


def test_skill_nearby_unread_inspect_does_not_click(monkeypatch):
    import numpy as np
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_execute.ui_logic import UiWhere

    targets: list[str] = []
    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:淨淨蘑菇')
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("unknown", 0, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.prepare_to",
        lambda target, **_k: targets.append(str(target)) or None,
    )
    clicks: list[tuple[int, int]] = []
    r = execute_skill_nearby(
        skill_id="herb",
        item_name="淨淨蘑菇",
        output="淨淨蘑菇",
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_stp=lambda: 0,
    )
    assert r.ok is False
    assert r.status == "unknown_ui"
    assert "不换条" in r.message
    assert targets == []
    assert clicks == []


def test_skill_nearby_facility_recipe_prepares(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import skill_nearby as sn
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import ExecuteResult

    stp = [1, 0]
    targets: list[str] = []
    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:淨淨蘑菇')
    monkeypatch.setattr(sn, "HOLD_IDLE_S", 0.0)
    monkeypatch.setattr(sn.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_skill_item",
        lambda *a, **k: {"ok": True, "cx": 10, "cy": 20},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("facility:medicine:淨淨蘑菇孢子", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "facility:medicine:淨淨蘑菇孢子", "zh": "制作-加工-药品加工-净净蘑菇孢子"},
    )

    def fake_prepare(target, **_k):
        targets.append(str(target))
        return ExecuteResult(ok=True, status="at_node", message=str(target))

    monkeypatch.setattr("mm_assist.life_execute.ui_act.prepare_to", fake_prepare)
    clicks: list[tuple[int, int]] = []
    r = execute_skill_nearby(
        skill_id="herb",
        item_name="淨淨蘑菇",
        output="淨淨蘑菇",
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_stp=lambda: stp.pop(0) if stp else 0,
        read_page=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
    )
    assert r.status != "unknown_ui"
    assert targets == ["gather_skill"]
    assert len(clicks) == 2
    assert r.ok is True


def test_unknown_ui_is_sense_not_nav():
    from mm_assist.life_plan.execute import is_nav_result, is_sense_result
    from mm_assist.life_plan.types import ExecuteResult

    r = ExecuteResult(ok=False, status="unknown_ui", message="认不出界面，不点、不换条")
    assert is_nav_result(r) is False
    assert is_sense_result(r) is True
    busy = ExecuteResult(ok=False, status="stp_busy", message="主界面动作中，不点、不换条")
    assert is_nav_result(busy) is False
    assert is_sense_result(busy) is True


def test_skill_nearby_matching_list_skips_prepare(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import skill_nearby as sn
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import ExecuteResult

    stp = [1, 0]
    targets: list[str] = []
    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:淨淨蘑菇')
    monkeypatch.setattr(sn, "HOLD_IDLE_S", 0.0)
    monkeypatch.setattr(sn.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("gather_skill:herb", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "gather_skill:herb", "zh": "角色-生活技能-採集藥草"},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_skill_item",
        lambda *a, **k: {"ok": True, "cx": 10, "cy": 20},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.prepare_to",
        lambda target, **_k: targets.append(str(target)) or ExecuteResult(True, "at_node"),
    )
    clicks: list[tuple[int, int]] = []
    r = execute_skill_nearby(
        skill_id="herb",
        item_name="淨淨蘑菇",
        output="淨淨蘑菇",
        do_click=lambda x, y: clicks.append((x, y)) or True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_stp=lambda: stp.pop(0) if stp else 0,
        read_page=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
    )
    assert targets == []
    assert len(clicks) == 2
    assert r.ok is True


def test_skill_nearby_wrong_skill_list_still_prepares(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import skill_nearby as sn
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import ExecuteResult

    targets: list[str] = []
    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:淨淨蘑菇')
    monkeypatch.setattr(sn, "HOLD_IDLE_S", 0.0)
    monkeypatch.setattr(sn.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "gather_skill:mine", "zh": "角色-生活技能-採礦"},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("gather_skill:mine", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_skill_item",
        lambda *a, **k: {"ok": True, "cx": 10, "cy": 20},
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.prepare_to",
        lambda target, **_k: targets.append(str(target)) or ExecuteResult(True, "at_node"),
    )
    stp = [1, 0]
    execute_skill_nearby(
        skill_id="herb",
        item_name="淨淨蘑菇",
        output="淨淨蘑菇",
        do_click=lambda *_a: True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_stp=lambda: stp.pop(0) if stp else 0,
        read_page=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
        now_fn=lambda: 0.0,
        sleep_fn=lambda _s: None,
        hold_s=0.0,
    )
    assert targets == ["gather_skill"]


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += float(s)


def test_skill_nearby_gain_while_busy_is_not_done(monkeypatch):
    """冲突：旧实现 saw_stp1 后 gained>0 就 done。PRODUCT/TESTING：流程完成须 stp 曾为1再明确待机，净增不能单独完成项。"""
    import numpy as np
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import ExecuteResult

    clock = _Clock()
    bag = {"淨淨蘑菇": 5}
    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:淨淨蘑菇')
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "gather_detail", "zh": "技能详情-淨淨蘑菇"},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("gather_detail", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.prepare_to",
        lambda *_a, **_k: ExecuteResult(ok=True, status="at_node"),
    )

    def read_items():
        bag["淨淨蘑菇"] = 9
        return dict(bag)

    r = execute_skill_nearby(
        skill_id="herb",
        item_name="淨淨蘑菇",
        output="淨淨蘑菇",
        items={"淨淨蘑菇": 5},
        do_click=lambda *_a: True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_stp=lambda: 1,
        read_page=lambda: type("P", (), {"node": "main_hud", "fresh": True})(),
        read_items=read_items,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
    )
    assert r.ok is False
    assert r.status == "gather_pending"
    assert bag["淨淨蘑菇"] == 9


def test_skill_nearby_none_after_busy_is_not_idle(monkeypatch):
    """冲突：旧实现 cur!=1 把 None 当 idle。未知读数不得算待机。"""
    import numpy as np
    from mm_assist.life_execute.skill_nearby import execute_skill_nearby
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_plan.types import ExecuteResult

    clock = _Clock()
    seq = [1]

    def read_stp():
        if seq:
            return seq.pop(0)
        return None

    monkeypatch.setattr('mm_assist.life_plan.node_inspect.gather_detail_for_target',
                        lambda frame, target: 'gather_detail:淨淨蘑菇')
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.read_inspect_record",
        lambda **k: {"node": "gather_detail", "zh": "技能详情-淨淨蘑菇"},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_logic.resolve_ui_node",
        lambda **k: UiWhere("gather_detail", 1, 0, 0, "mem"),
    )
    monkeypatch.setattr(
        "mm_assist.perception.item_sense.locate_find_nearby",
        lambda *a, **k: {"ok": True, "cx": 30, "cy": 40},
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.ui_act.prepare_to",
        lambda *_a, **_k: ExecuteResult(ok=True, status="at_node"),
    )
    r = execute_skill_nearby(
        skill_id="herb",
        item_name="淨淨蘑菇",
        output="淨淨蘑菇",
        do_click=lambda *_a: True,
        grab=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
        read_stp=read_stp,
        read_page=lambda: type("P", (), {"node": "unknown", "fresh": False})(),
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        hold_s=0.0,
    )
    assert r.ok is False
    assert r.status == "gather_pending"
