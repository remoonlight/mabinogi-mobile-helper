from __future__ import annotations

import pytest

from mm_assist.life_plan.types import ToolDurability
from mm_assist.perception.ocr import OcrLine
from mm_assist.tool_repair import (
    any_tool_durability_below,
    clear_op_cancel,
    clear_statusbar_equip_cache,
    detect_repair_needed,
    durability_rose,
    equip_durability_map,
    one_click_repair,
    op_cancelled,
    plan_durability_repair,
    plan_repair_dialog_entry,
    repair_flow_accepted,
    request_op_cancel,
    resolve_require_durability_rose,
    run_repair_round,
    set_statusbar_equip_cache,
)


@pytest.fixture(autouse=True)
def _isolate_repair_stage_gates(monkeypatch):
    from mm_assist.safety import autofarm_gates as gates
    monkeypatch.setattr(gates, '_stage_counts', {})
    monkeypatch.setattr(gates, '_stage_need_verify', {})


def _line(text: str, x: int = 100, y: int = 200) -> OcrLine:
    box = [[x - 10, y - 5], [x + 40, y - 5], [x + 40, y + 5], [x - 10, y + 5]]
    return OcrLine(text, 0.9, box)


def test_detect_repair_hit():
    lines = [_line("工具已損壞，請修理")]
    assert detect_repair_needed(lines) == "工具已損"


def test_detect_prefers_need_repair_over_repair():
    lines = [_line("需要修理才能繼續")]
    assert detect_repair_needed(lines) == "需要修理"


def test_detect_repair_miss():
    lines = [_line("尋找附近位置")]
    assert detect_repair_needed(lines) is None


def test_durability_rose_and_tool_below():
    clear_statusbar_equip_cache()
    assert durability_rose({"a": 0}, {"a": 10}) is True
    assert durability_rose({"a": 5}, {"a": 5}) is False
    assert durability_rose({"a": 5}, {"b": 9}) is False

    class _Snap:
        tools = [ToolDurability("镐", 0.0, 100.0)]

    m = equip_durability_map(_Snap())
    assert list(m.values()) == [0]
    set_statusbar_equip_cache(_Snap())
    probe = any_tool_durability_below(1.0, allow_mem_scan=False)
    assert probe.needed is True
    assert "镐" in probe.matched


def test_plan_skip_and_one_click():
    skip = plan_durability_repair(needed=False)
    assert skip["op"] == "skip"
    go = plan_durability_repair(needed=True, matched="镐:0")
    assert go["op"] == "one_click"
    assert "repair_dialog" in go["forbid_first"]


def test_dialog_entry_never_nameplate():
    p = plan_repair_dialog_entry(arrived=True, has_skip=True, nameplate_found=True)
    assert p["forbid_nameplate"] is True
    assert p["allow_click_nameplate"] is False
    assert p["op"] == "skip_then_find_entry"


def test_one_click_skip_when_tools_ok():
    clear_statusbar_equip_cache()

    class _Snap:
        tools = [ToolDurability("镐", 12.0, 100.0)]

    set_statusbar_equip_cache(_Snap())
    _w, out = one_click_repair(None, None, force=False)
    assert out["ok"] is True
    assert out["stage"] == "skip"


def test_run_round_requires_durability_rose():
    def nav(cfg, win, **k):
        return win, {"ok": True, "stage": "arrived"}

    def dialog(cfg, win, **k):
        return win, {"ok": True, "stage": "dialog_done"}

    _w, out = run_repair_round(
        None,
        None,
        nav_fn=nav,
        dialog_fn=dialog,
        dur_before={"tool:镐:0": 0},
        dur_after_fn=lambda: {"tool:镐:0": 0},
        require_durability_rose=True,
    )
    assert out["ok"] is False
    assert out["stage"] == "durability"

    _w, out2 = run_repair_round(
        None,
        None,
        nav_fn=nav,
        dialog_fn=dialog,
        dur_before={"tool:镐:0": 0},
        dur_after_fn=lambda: {"tool:镐:0": 80},
    )
    assert out2["ok"] is True
    assert out2["durability_rose"] is True


def test_nav_uses_map_goto(monkeypatch):
    monkeypatch.setattr(
        "mm_assist.nav.map_atlas.map_goto",
        lambda *a, **k: {"ok": False, "stage": "goto", "error": "no_game"},
    )
    _w, out = run_repair_round(
        None,
        None,
        dur_before={"t": 0},
        dur_after_fn=lambda: {"t": 0},
    )
    assert out["ok"] is False
    assert out["stage"] in {"goto", "region"}


def test_nav_phase_returns_walking_without_dialog():
    dialogs: list[int] = []

    def nav(cfg, win, **k):
        assert k.get("wait_arrive") is False
        return win, {"ok": True, "stage": "walking"}

    def dialog(cfg, win, **k):
        dialogs.append(1)
        return win, {"ok": True, "stage": "done"}

    _w, out = run_repair_round(
        None,
        None,
        nav_fn=nav,
        dialog_fn=dialog,
        phase="nav",
        dur_before={"t": 10},
        dur_after_fn=lambda: {"t": 10},
    )
    assert out["ok"] is True
    assert out["stage"] == "walking"
    assert dialogs == []


def test_repair_navigation_failure_is_not_retried_or_masked(monkeypatch):
    monkeypatch.setattr(
        "mm_assist.life_execute.repair_dialog._peek_dialog_active", lambda: False
    )
    clear_op_cancel()
    navs = []
    failure = {
        "ok": False,
        "stage": "continent_region_stuck",
        "error": "original_map_failure",
    }

    def nav(cfg, win, **kwargs):
        navs.append(kwargs)
        # A retry could succeed or replace the original error; neither is allowed.
        return win, failure if len(navs) == 1 else {"ok": True, "stage": "arrived"}

    def dialog(*args, **kwargs):
        raise AssertionError("failed navigation must not start the repair dialog")

    _win, out = run_repair_round(
        None, None, nav_fn=nav, dialog_fn=dialog, dur_before={"tool": 0}
    )
    assert len(navs) == 1
    assert out["ok"] is False
    assert out["goto"] == failure
    assert out["stage"] == "continent_region_stuck"
    assert out["error"] == "original_map_failure"


def test_dialog_phase_skips_map():
    navs: list[int] = []

    def nav(cfg, win, **k):
        navs.append(1)
        return win, {"ok": False, "stage": "goto", "error": "should_not_nav"}

    def dialog(cfg, win, **k):
        return win, {"ok": True, "stage": "done", "dlg_left": True}

    _w, out = run_repair_round(
        None,
        None,
        nav_fn=nav,
        dialog_fn=dialog,
        phase="dialog",
        dur_before={"t": 10},
        dur_after_fn=lambda: {"t": 20},
        require_durability_rose=True,
    )
    assert navs == []
    assert out["ok"] is True
    assert out["stage"] == "done"


def test_cancel_stops_round():
    clear_op_cancel()
    request_op_cancel()
    _w, out = run_repair_round(None, None)
    assert out["stage"] == "cancel"
    assert op_cancelled()
    clear_op_cancel()


def test_execute_repair_success(monkeypatch):
    from mm_assist.life_execute.repair import execute_repair

    monkeypatch.setattr(
        "mm_assist.life_execute.repair.one_click_repair",
        lambda *a, **k: (None, {"ok": True, "stage": "done", "durability_rose": True}),
    )
    from mm_assist.safety.lease import production_mode

    try:
        r = execute_repair(None, None)
        assert r.ok is True
        assert r.status == "done"
        assert "回升" in r.message
    finally:
        production_mode(False)


def test_execute_repair_drops_live_test_kwargs(monkeypatch):
    from mm_assist.life_execute.repair import execute_repair

    seen: dict[str, object] = {}

    def fake(*a, **k):
        seen.update(k)
        return None, {"ok": True, "stage": "done", "durability_rose": False}

    monkeypatch.setattr("mm_assist.life_execute.repair.one_click_repair", fake)
    from mm_assist.safety.lease import production_mode

    try:
        r = execute_repair(
            None,
            None,
            facilities={"metal": []},
            items={"鋼錠": 1},
            grab=lambda: None,
        )
        assert r.ok is True
        assert "facilities" not in seen
        assert "items" not in seen
        assert "grab" not in seen
    finally:
        production_mode(False)


def test_stamp_default_requires_durability():
    assert repair_flow_accepted() is True
    assert resolve_require_durability_rose(None) is False
    assert resolve_require_durability_rose(True) is True
    assert resolve_require_durability_rose(False) is False


def test_invalid_stamp_is_unaccepted(tmp_path):
    bad = tmp_path / "stamp.json"
    bad.write_text("{not json", encoding="utf-8")
    assert repair_flow_accepted(bad) is False
    bad.write_text('{"accepted": "true"}', encoding="utf-8")
    assert repair_flow_accepted(bad) is False
    missing = tmp_path / "nope.json"
    assert repair_flow_accepted(missing) is False


def test_accepted_dialog_ok_without_rose():
    def nav(cfg, win, **k):
        return win, {"ok": True, "stage": "arrived"}

    def dialog(cfg, win, **k):
        return win, {"ok": True, "stage": "dialog_done"}

    _w, out = run_repair_round(
        None,
        None,
        nav_fn=nav,
        dialog_fn=dialog,
        dur_before={"tool:镐:0": 0},
        dur_after_fn=lambda: {"tool:镐:0": 0},
        require_durability_rose=False,
    )
    assert out["ok"] is True
    assert out["durability_rose"] is False
    assert out["require_durability_rose"] is False


def test_dlg_left_false_still_fails():
    def nav(cfg, win, **k):
        return win, {"ok": True, "stage": "arrived"}

    def dialog(cfg, win, **k):
        return win, {"ok": True, "stage": "dialog_done", "dlg_left": False}

    _w, out = run_repair_round(
        None,
        None,
        nav_fn=nav,
        dialog_fn=dialog,
        dur_before={"tool:镐:0": 0},
        dur_after_fn=lambda: {"tool:镐:0": 80},
        require_durability_rose=False,
    )
    assert out["ok"] is False
    assert out["stage"] == "dialog"


def test_execute_repair_ok_without_rose(monkeypatch):
    from mm_assist.life_execute.repair import execute_repair
    from mm_assist.safety.lease import production_mode

    monkeypatch.setattr(
        "mm_assist.life_execute.repair.one_click_repair",
        lambda *a, **k: (None, {"ok": True, "stage": "done", "durability_rose": False}),
    )
    try:
        r = execute_repair(None, None)
        assert r.ok is True
        assert r.status == "done"
        assert "仅记录" in r.message
    finally:
        production_mode(False)


def test_repair_dialog_templates_ready():
    from mm_assist.life_execute.repair_dialog import repair_dialog_templates_ready

    assert repair_dialog_templates_ready() is True


def test_skip_template_matches_itself():
    import cv2
    import numpy as np

    from mm_assist.config import ROOT
    from mm_assist.life_execute.repair_dialog import find_skip_dialog_icon

    tpl = cv2.imread(str(ROOT / "knowledge/ui/repair/01_skip_dialog.png"))
    assert tpl is not None
    canvas = np.zeros((1440, 1920, 3), dtype=np.uint8)
    h, w = tpl.shape[:2]
    canvas[40 : 40 + h, 1600 : 1600 + w] = tpl
    hit = find_skip_dialog_icon(canvas)
    assert hit is not None
    assert hit.score >= 0.70


def test_dialog_chain_clicks_skip_first(monkeypatch):
    import numpy as np

    from mm_assist.config import AppConfig
    from mm_assist.life_execute.repair_dialog import run_repair_dialog_chain
    from mm_assist.perception.ocr import OcrLine
    from mm_assist.platform.window import WindowInfo

    win = WindowInfo(0, "t", 0, 0, 1920, 1440)
    img = np.zeros((1440, 1920, 3), dtype=np.uint8)
    skip = OcrLine(
        "跳過對話",
        0.9,
        [[1700, 40], [1860, 40], [1860, 80], [1700, 80]],
    )
    clicks: list[str] = []

    monkeypatch.setattr(
        "mm_assist.life_execute.repair_dialog._snap_frame",
        lambda *a, **k: (win, img, [skip]),
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.repair_dialog._peek_dialog_active", lambda: True
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.repair_dialog.click_client",
        lambda *a, **k: clicks.append("click") or True,
    )
    monkeypatch.setattr(
        "mm_assist.life_execute.repair_dialog.time.sleep", lambda *_a, **_k: None
    )

    _w, out = run_repair_dialog_chain(AppConfig(), win, lines=[skip], img=img)
    assert clicks, "应点过跳過對話"
    assert out["ok"] is False
    assert out.get("skipped")


def test_npc_entry_click_is_repair_row_not_shop_row():
    import cv2

    from mm_assist.config import ROOT
    from mm_assist.life_execute.repair_dialog import locate_repair_entry_click
    from mm_assist.platform.window import WindowInfo

    path = ROOT / "tests" / "fixtures" / "repair" / "nerys_entry_ocr_miss.png"
    img = cv2.imread(str(path))
    assert img is not None
    win = WindowInfo(0, "t", 0, 0, int(img.shape[1]), int(img.shape[0]))
    loc = locate_repair_entry_click(win, [], img)
    assert loc is not None
    x, y, how = loc
    assert how == "修理图标"
    assert 350 <= x <= 700
    assert 1000 <= y <= 1200, f"旧逻辑点在商店行 y=1335，修理行约 1106，实际 {y}"


def test_shop_row_ocr_is_not_repair_entry():
    from mm_assist.life_execute.repair_dialog import locate_end_dialog_click, locate_repair_entry_click
    from mm_assist.perception.ocr import OcrLine
    from mm_assist.platform.window import WindowInfo

    win = WindowInfo(0, "t", 0, 0, 1912, 1442)
    shop_repair = [
        OcrLine("修理", 0.9, [[580, 1310], [674, 1310], [674, 1370], [580, 1370]]),
    ]
    assert locate_repair_entry_click(win, shop_repair, None) is None
    end_lines = [
        OcrLine("結束對話", 0.9, [[1080, 1320], [1190, 1320], [1190, 1360], [1080, 1360]]),
    ]
    end = locate_end_dialog_click(win, end_lines, None)
    assert end is not None
    assert end[1] >= 1300


def test_repair_entry_ocr_on_six_button_row():
    from mm_assist.life_execute.repair_dialog import locate_repair_entry_click
    from mm_assist.perception.ocr import OcrLine
    from mm_assist.platform.window import WindowInfo

    win = WindowInfo(0, "t", 0, 0, 1912, 1442)
    lines = [
        OcrLine("修理", 0.9, [[430, 1080], [500, 1080], [500, 1130], [430, 1130]]),
    ]
    loc = locate_repair_entry_click(win, lines, None)
    assert loc is not None
    x, y, how = loc
    assert how == "修理胶囊"
    assert y < 1250
    assert x < 700


def test_npc_entry_menu_blocks_click_away():
    from mm_assist.life_execute.repair_dialog import npc_entry_menu_visible
    from mm_assist.perception.ocr import OcrLine

    lines = [
        OcrLine("修理", 0.9, [[200, 900], [280, 900], [280, 960], [200, 960]]),
        OcrLine("特化", 0.9, [[400, 900], [480, 900], [480, 960], [400, 960]]),
        OcrLine("解除詛咒", 0.9, [[200, 1000], [360, 1000], [360, 1060], [200, 1060]]),
    ]
    assert npc_entry_menu_visible(lines, None) is True
    assert npc_entry_menu_visible([], None) is False


def test_skip_nav_when_already_in_dialog(monkeypatch):
    monkeypatch.setattr(
        "mm_assist.life_execute.repair_dialog._peek_dialog_active", lambda: True
    )
    nav_calls: list[int] = []

    def nav(cfg, win, **k):
        nav_calls.append(1)
        return win, {"ok": True, "stage": "arrived"}

    def dialog(cfg, win, **k):
        return win, {"ok": True, "stage": "dialog_done", "dlg_left": True}

    _w, out = run_repair_round(
        None,
        None,
        nav_fn=nav,
        dialog_fn=dialog,
        dur_before={"tool:镐:0": 0},
        dur_after_fn=lambda: {"tool:镐:0": 80},
    )
    assert nav_calls == []
    assert out["ok"] is True
    assert out["goto"]["stage"] == "already_dialog"


def test_goodbye_zoom_has_no_repair_entry_icon():
    import cv2

    from mm_assist.config import ROOT
    from mm_assist.life_execute.repair_dialog import (
        find_repair_entry_icon,
        _hud_menu_xy,
    )

    path = ROOT / "tests" / "fixtures" / "repair" / "nerys_after_end_dialog.png"
    img = cv2.imread(str(path))
    assert img is not None
    assert find_repair_entry_icon(img) is None
    assert _hud_menu_xy(img) is None


def test_wait_hud_after_npc_clicks_recognized_goodbye_once_then_verifies_hud(monkeypatch):
    import cv2
    import numpy as np

    from mm_assist.config import ROOT
    from mm_assist.life_execute.repair_dialog import wait_hud_after_npc_dismiss

    goodbye = cv2.imread(str(ROOT / "tests" / "fixtures" / "repair" / "nerys_after_end_dialog.png"))
    assert goodbye is not None
    hud = np.zeros_like(goodbye)
    seq = iter([goodbye, goodbye, hud, hud])
    clicks: list[tuple[int, int]] = []

    def click(x, y):
        clicks.append((int(x), int(y)))
        return True

    def grab():
        return next(seq, hud)

    monkeypatch.setattr(
        "mm_assist.life_execute.repair_dialog._hud_menu_xy",
        lambda fr: (1800, 40) if fr is hud else None,
    )
    ok = wait_hud_after_npc_dismiss(click, grab, sleep_fn=lambda *_a, **_k: None)
    assert ok is True
    assert len(clicks) == 1, '用户要求点画面结束告别；只点已由原图模板识别的气泡'
    assert 700 < clicks[0][0] < 1210 and 100 < clicks[0][1] < 335


def test_npc_goodbye_visible_on_after_end_dialog():
    import cv2

    from mm_assist.config import ROOT
    from mm_assist.life_execute.repair_dialog import npc_goodbye_visible

    path = ROOT / "tests" / "fixtures" / "repair" / "nerys_after_end_dialog.png"
    img = cv2.imread(str(path))
    assert img is not None
    assert npc_goodbye_visible(img) is True
    six = cv2.imread(str(ROOT / "tests" / "fixtures" / "repair" / "nerys_entry_ocr_miss.png"))
    assert six is not None
    assert npc_goodbye_visible(six) is False


def test_npc_goodbye_wait_does_not_click_unrecognized_frame(monkeypatch):
    import numpy as np
    from mm_assist.life_execute import repair_dialog

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    clicks = []
    monkeypatch.setattr(repair_dialog, "find_repair_entry_icon", lambda _f: None)
    monkeypatch.setattr(repair_dialog, "_hud_menu_xy", lambda _f: None)
    assert not repair_dialog.wait_hud_after_npc_dismiss(
        lambda *xy: clicks.append(xy), lambda: frame, sleep_fn=lambda _s: None,
    )
    assert clicks == []


def test_npc_goodbye_wait_stops_on_cancel(monkeypatch):
    from mm_assist.life_execute import repair_dialog

    monkeypatch.setattr(repair_dialog, "op_cancelled", lambda: True)
    def forbidden():
        raise AssertionError("cancelled wait must not capture or click")
    assert not repair_dialog.wait_hud_after_npc_dismiss(
        lambda *_xy: forbidden(), forbidden, sleep_fn=lambda _s: forbidden(),
    )


def test_goodbye_template_is_primary_and_rejects_other_pages(monkeypatch):
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.life_execute import repair_dialog as dialog
    monkeypatch.setattr('mm_assist.perception.ocr.ocr_roi',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('no OCR main判')))
    goodbye = cv2.imread(str(ROOT / 'tests/fixtures/repair/nerys_after_end_dialog.png'))
    assert dialog.npc_goodbye_visible(goodbye)
    assert not dialog.npc_goodbye_visible(object())
    assert not dialog.npc_goodbye_visible(None)
    for rel in ('repair/nerys_entry_ocr_miss.png', 'process_stance/continent_bankor_upper.png',
                'process_stance/wood_standalone.png'):
        assert not dialog.npc_goodbye_visible(cv2.imread(str(ROOT / 'tests/fixtures' / rel)))


def test_goodbye_click_rejection_or_unchanged_page_never_retries(monkeypatch):
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.life_execute import repair_dialog as dialog
    img = cv2.imread(str(ROOT / 'tests/fixtures/repair/nerys_after_end_dialog.png'))
    for accepted in (False, True):
        clicks = []
        assert not dialog.wait_hud_after_npc_dismiss(
            lambda *xy: clicks.append(xy) or accepted, lambda: img, sleep_fn=lambda _s: None,
        )
        assert len(clicks) == 1


def test_leave_dialog_unknown_memory_without_hud_is_not_success(monkeypatch):
    import numpy as np
    from mm_assist.config import AppConfig
    from mm_assist.platform.window import WindowInfo
    from mm_assist.life_execute import repair_dialog as dialog
    win = WindowInfo(0, 'offline', 0, 0, 300, 200)
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    clicks = []
    monkeypatch.setattr(dialog, '_peek_dialog_active', lambda: None)
    monkeypatch.setattr(dialog, '_grab_frame', lambda *a: (win, img))
    monkeypatch.setattr(dialog, 'click_away', lambda *a: clicks.append('blind'))
    monkeypatch.setattr(dialog.time, 'sleep', lambda _s: None)
    _, left = dialog._wait_leave_dialog(AppConfig(), win)
    assert not left
    assert clicks == []


def test_goodbye_cancel_after_capture_has_no_input(monkeypatch):
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.life_execute import repair_dialog as dialog
    img = cv2.imread(str(ROOT / 'tests/fixtures/repair/nerys_after_end_dialog.png'))
    stopped = [False]
    monkeypatch.setattr(dialog, 'op_cancelled', lambda: stopped[0])
    def grab():
        stopped[0] = True
        return img
    clicks = []
    assert not dialog.wait_hud_after_npc_dismiss(lambda *xy: clicks.append(xy), grab)
    assert clicks == []


def test_goodbye_unchanged_across_calls_keeps_two_action_gate(monkeypatch):
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.life_execute import repair_dialog as dialog
    img = cv2.imread(str(ROOT / 'tests/fixtures/repair/nerys_after_end_dialog.png'))
    clicks = []
    for _ in range(3):
        assert not dialog.wait_hud_after_npc_dismiss(
            lambda *xy: clicks.append(xy) or True, lambda: img, sleep_fn=lambda _s: None,
        )
    assert len(clicks) == 2


def test_end_dialog_matches_complete_label_and_rejects_neighbor_and_other_pages():
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.life_execute import repair_dialog as dialog
    for name in ('nerys_repair_done_end.png', 'nerys_entry_ocr_miss.png'):
        img = cv2.imread(str(ROOT/'tests/fixtures/repair'/name))
        hit = dialog.find_end_dialog_btn(img)
        assert hit is not None
        assert 1100 < hit.cx < 1180 and 1310 < hit.cy < 1380
        img[1300:1380, 1050:1210] = 0  # Leave shop / chat buttons untouched.
        assert dialog.find_end_dialog_btn(img) is None
    img = cv2.imread(str(ROOT/'tests/fixtures/repair/nerys_after_end_dialog.png'))
    assert dialog.find_end_dialog_btn(img) is None


@pytest.mark.parametrize('accepted', [True, False])
def test_end_dialog_waits_animation_then_dispatches_once(monkeypatch, accepted):
    import cv2
    import numpy as np
    from mm_assist.config import ROOT, AppConfig
    from mm_assist.platform.window import WindowInfo
    from mm_assist.life_execute import repair_dialog as dialog
    img = cv2.imread(str(ROOT/'tests/fixtures/repair/nerys_repair_done_end.png'))
    win = WindowInfo(0,'offline',0,0,img.shape[1],img.shape[0])
    frames = iter([np.zeros_like(img), img])
    monkeypatch.setattr(dialog, '_grab_frame', lambda *a: (win,next(frames)))
    monkeypatch.setattr(dialog.time,'sleep',lambda _: None)
    monkeypatch.setattr(dialog,'click_away',lambda *a: pytest.fail('no blank result click'))
    clicks = []
    monkeypatch.setattr(dialog,'click_client',lambda w,x,y,**k: clicks.append((x,y)) or accepted)
    _, ended = dialog._wait_and_end_dialog(AppConfig(),win)
    assert bool(ended) == accepted
    assert len(clicks) == 1 and clicks[0][0] > 1100


def test_end_dialog_cancel_after_capture_has_no_input(monkeypatch):
    import cv2
    from mm_assist.config import ROOT, AppConfig
    from mm_assist.platform.window import WindowInfo
    from mm_assist.life_execute import repair_dialog as dialog
    img = cv2.imread(str(ROOT/'tests/fixtures/repair/nerys_repair_done_end.png'))
    win = WindowInfo(0,'offline',0,0,img.shape[1],img.shape[0])
    stopped = [False]
    def grab(*a):
        stopped[0] = True
        return win,img
    monkeypatch.setattr(dialog,'_grab_frame',grab)
    monkeypatch.setattr(dialog,'op_cancelled',lambda: stopped[0])
    monkeypatch.setattr(dialog,'click_client',lambda *a,**k: pytest.fail('cancelled end'))
    assert dialog._wait_and_end_dialog(AppConfig(),win)[1] is None


@pytest.mark.parametrize('phase', ['all', 'nav'])
def test_execute_repair_recognized_npc_menu_does_not_reopen_map(monkeypatch, phase):
    import cv2
    from mm_assist.config import ROOT, AppConfig
    from mm_assist.platform.window import WindowInfo
    from mm_assist.life_execute.repair import execute_repair
    from mm_assist.life_execute.ui_logic import UiWhere
    from mm_assist.life_execute import ui_act
    frame = cv2.imread(str(ROOT/'tests/fixtures/repair/nerys_repair_done_end.png'))
    win = WindowInfo(7, 'offline', 0, 0, frame.shape[1], frame.shape[0])
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda _: frame)
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.resolve_navigation',
                        lambda _: UiWhere('main_hud', 0, 0, 0, 'fixture'))
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **_: False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda **_: False)
    monkeypatch.setattr('mm_assist.life_execute.repair_dialog._peek_dialog_active', lambda: False)
    monkeypatch.setattr('mm_assist.safety.input.click_client',
                        lambda *a, **k: pytest.fail('stay in the recognized repair menu'))
    calls = []
    def dialog(cfg, w):
        calls.append('dialog')
        return w, {'ok': True, 'dlg_left': True}
    result = execute_repair(AppConfig(), win, phase=phase, dur_before={},
        dur_after_fn=lambda: {}, dialog_fn=dialog,
        nav_fn=lambda *a, **k: pytest.fail('recognized dialog must not open the map'))
    assert result.ok
    assert result.status == ('at_dialog' if phase == 'nav' else 'done')
    assert calls == ([] if phase == 'nav' else ['dialog'])


def test_npc_menu_exit_unchanged_keeps_two_action_gate_and_never_uses_ocr(monkeypatch):
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.life_execute import repair_dialog as dialog
    frame = cv2.imread(str(ROOT/'tests/fixtures/repair/nerys_repair_done_end.png'))
    monkeypatch.setattr('mm_assist.perception.ocr.ocr_roi',
                        lambda *a, **k: pytest.fail('no OCR exit authorization'))
    clicks = []
    for _ in range(3):
        assert not dialog.dismiss_world_npc_menu(
            lambda *xy: clicks.append(xy) or True, frame, grab=lambda: frame,
            sleep_fn=lambda _: None)
    assert len(clicks) == 2


def test_npc_menu_exit_without_complete_label_never_clicks(monkeypatch):
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.life_execute import repair_dialog as dialog
    frame = cv2.imread(str(ROOT/'tests/fixtures/repair/nerys_repair_done_end.png'))
    frame[1300:1380, 1050:1210] = 0
    assert not dialog.dismiss_world_npc_menu(
        lambda *_: pytest.fail('missing label accepts no input'), frame, grab=lambda: frame)


def test_npc_menu_exit_input_rejection_blocks_following_navigation(monkeypatch):
    import cv2
    from mm_assist.config import ROOT
    from mm_assist.life_execute import ui_act, ui_logic
    frame = cv2.imread(str(ROOT/'tests/fixtures/repair/nerys_repair_done_end.png'))
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda *_: pytest.fail('exit failed'))
    clicks = []
    result = ui_act.prepare_to('main_hud', grab=lambda: frame,
        click=lambda *xy: clicks.append(xy) or False, sleep_fn=lambda _: None)
    assert result.status == 'npc_end_dialog_unverified'
    assert len(clicks) == 1
