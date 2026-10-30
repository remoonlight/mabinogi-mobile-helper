from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mm_assist.life_execute.ui_logic import UiWhere
from mm_assist.life_execute import ui_nav
from mm_assist.life_plan.live_test import parse_nav_detail

FIX = Path(__file__).parent / "fixtures" / "ui_benchmark"
FAIL_PNG = FIX / "menu_claim_sense_timeout.png"
ANIM_PNG = FIX / "system_menu_transition.png"
TALL_PNG = FIX / "system_menu_real.png"
OTHER_PNG = FIX / "hud_menu_lairt_claim.png"
CRAFT_EDGE_ID = "system_menu_craft_craft_hub"
# 原失败图探针：icon y435..499，匹配中心 1567,467（非代码 ROI 镜像）
EVIDENCE_CLICK = (1567, 467)


def _fail_frame():
    frame = cv2.imread(str(FAIL_PNG))
    assert frame is not None
    return frame


def _craft_from_graph():
    ui_nav._graph.cache_clear()
    hops = ui_nav.path_edges("system_menu", "craft_hub") or []
    edge = next(e for e in hops if e.get("id") == CRAFT_EDGE_ID)
    return edge


def _ensure_from_menu(frame, clicks, after_click_node="craft_hub"):
    clicked = {"n": 0}

    def resolve(_f):
        if clicked["n"]:
            return UiWhere(after_click_node, 1, 0, None, "direct")
        return UiWhere("system_menu", 1, 0, None, "direct")

    def click(x, y):
        clicks.append((int(x), int(y)))
        clicked["n"] += 1
        return True

    orig_visible = ui_nav.craft_hub_visible
    ui_nav.craft_hub_visible = lambda *_a, **_k: bool(clicked["n"])
    try:
        target, skill, facility, item_name = parse_nav_detail("craft_hub")
        return ui_nav.ensure_node(
            target,
            skill_id=skill,
            facility=facility,
            item_name=item_name,
            grab=lambda: frame,
            click=click,
            sleep_fn=lambda _s: None,
            resolve=resolve,
        )
    finally:
        ui_nav.craft_hub_visible = orig_visible


@pytest.fixture(autouse=True)
def _reset_nav():
    ui_nav.reset_edge_memo_for_tests()
    ui_nav._graph.cache_clear()
    yield
    ui_nav.reset_edge_memo_for_tests()
    ui_nav._graph.cache_clear()


def test_old_menu_roi_misses_fail_frame_and_does_not_click(monkeypatch):
    frame = _fail_frame()
    edge = {
        "id": CRAFT_EDGE_ID,
        "from": "system_menu",
        "to": "craft_hub",
        "trigger": {"type": "visible_button", "label": "製作"},
    }
    assert ui_nav.locate_edge(edge, frame) is None
    monkeypatch.setattr(ui_nav, "path_edges", lambda *a, **k: [edge])
    clicks: list[tuple[int, int]] = []
    result = _ensure_from_menu(frame, clicks)
    assert not clicks
    assert result.ok is False
    assert "製作" in str(result.message or result.status)


def test_graph_craft_roi_locates_evidence_center_and_clicks_only_that():
    frame = _fail_frame()
    edge = _craft_from_graph()
    roi = ((edge.get("trigger") or {}).get("locator") or {}).get("roi")
    assert roi == [0.80, 0.12, 0.16, 0.32] or roi == (0.80, 0.12, 0.16, 0.32)
    xy = ui_nav.locate_edge(edge, frame)
    assert xy is not None
    assert abs(xy[0] - EVIDENCE_CLICK[0]) < 25
    assert abs(xy[1] - EVIDENCE_CLICK[1]) < 25
    clicks: list[tuple[int, int]] = []
    result = _ensure_from_menu(frame, clicks)
    assert result.ok
    assert clicks == [xy]


def test_offline_menu_ignores_foreign_watch_feed_but_explicit_feed_still_works(tmp_path, monkeypatch):
    import json
    import time
    from mm_assist import config
    from mm_assist.perception import state_ui
    from mm_assist.life_execute.ui_act import stack_has_common_popup

    # Model the unrelated live feed outside this test's default feed root.
    foreign_root = tmp_path / 'foreign_watch'
    path = state_ui.state_ui_path(foreign_root)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'ts_unix':time.time(),'pid':123,
        'ui_stack_names':['ButtonCommonPopupInstance'],
        'ui_stack':'ButtonCommonPopupInstance','ui_stack_ready':True}),encoding='utf-8')
    monkeypatch.setattr(config,'ROOT',foreign_root)
    monkeypatch.setattr(state_ui,'_pid_alive',lambda pid:pid==123)

    explicit = state_ui.read_state_ui(root=foreign_root)
    assert explicit is not None and explicit.fresh and explicit.pid_alive
    assert stack_has_common_popup(names=explicit.raw['ui_stack_names'])
    assert not stack_has_common_popup()
    clicks=[]
    result=_ensure_from_menu(_fail_frame(),clicks)
    assert result.ok and clicks==[EVIDENCE_CLICK]


def test_explicit_invalid_craft_roi_refuses_instead_of_fallback():
    frame = cv2.imread(str(TALL_PNG))
    assert frame is not None
    assert ui_nav.locate_edge({"id": CRAFT_EDGE_ID}, frame) is not None
    edge = {
        "id": CRAFT_EDGE_ID,
        "from": "system_menu",
        "to": "craft_hub",
        "trigger": {
            "type": "visible_button",
            "label": "製作",
            "locator": {"roi": [0.80, 0.12, 0.16, True]},
        },
    }
    assert ui_nav.locate_edge(edge, frame) is None
    edge["trigger"]["locator"]["roi"] = [0.80, 0.12, 0.16]
    assert ui_nav.locate_edge(edge, frame) is None
    edge["trigger"]["locator"]["roi"] = [0.80, 0.12, 0.16, 1.5]
    assert ui_nav.locate_edge(edge, frame) is None
    edge["trigger"]["locator"]["roi"] = [0.80, 0.12, 0.16, 10**400]
    assert ui_nav.locate_edge(edge, frame) is None


def test_explicit_non_dict_locator_refuses_fallback_dict_without_roi_keeps_default():
    frame = cv2.imread(str(TALL_PNG))
    assert frame is not None
    fallback = {"id": CRAFT_EDGE_ID}
    assert ui_nav.locate_edge(fallback, frame) is not None
    edge = {
        "id": CRAFT_EDGE_ID,
        "from": "system_menu",
        "to": "craft_hub",
        "trigger": {"type": "visible_button", "label": "製作", "locator": None},
    }
    assert ui_nav.locate_edge(edge, frame) is None
    edge["trigger"]["locator"] = []
    assert ui_nav.locate_edge(edge, frame) is None
    edge["trigger"]["locator"] = {}
    assert ui_nav.locate_edge(edge, frame) is not None
    del edge["trigger"]["locator"]
    assert ui_nav.locate_edge(edge, frame) is not None


def test_missing_locator_uses_fallback_old_roi():
    frame = _fail_frame()
    assert ui_nav._EDGE_ROI[CRAFT_EDGE_ID] == (0.80, 0.12, 0.16, 0.22)
    assert ui_nav.locate_edge({"id": CRAFT_EDGE_ID}, frame) is None


def test_menu_animation_and_removed_icon_never_click_other_cells():
    anim = cv2.imread(str(ANIM_PNG))
    assert anim is not None
    edge = _craft_from_graph()
    assert ui_nav.locate_edge(edge, anim) is None
    frame = _fail_frame().copy()
    h, w = frame.shape[:2]
    x0, y0 = max(0, EVIDENCE_CLICK[0] - 40), max(0, EVIDENCE_CLICK[1] - 40)
    frame[y0 : min(h, y0 + 80), x0 : min(w, x0 + 80)] = 0
    assert ui_nav.locate_edge(edge, frame) is None
    if OTHER_PNG.is_file():
        other = cv2.imread(str(OTHER_PNG))
        assert other is not None
        assert ui_nav.locate_edge(edge, other) is None


def test_old_tall_menu_fixture_still_hits_visible_cell():
    frame = cv2.imread(str(TALL_PNG))
    assert frame is not None
    xy = ui_nav.locate_edge({"id": CRAFT_EDGE_ID}, frame)
    assert xy is not None
    assert abs(xy[0] - 1567) < 25
    assert 210 < xy[1] < 265


def test_blank_frame_never_clicks():
    clicks: list[tuple[int, int]] = []
    result = _ensure_from_menu(np.zeros((1442, 1912, 3), dtype=np.uint8), clicks)
    assert not clicks
    assert result.ok is False
