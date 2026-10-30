from __future__ import annotations

from types import SimpleNamespace

from mm_assist.jobs.state_watch import (
    format_job_content,
    format_job_status,
    format_mem_panel,
    format_warm_progress,
)


def test_format_job_content_is_short_and_named():
    text = format_job_content(
        {
            "title": "菜单源帧",
            "need_ui": "主界面菜单可见",
            "action": "readonly_ui_fine",
            "status": "await_start",
            "args": {"entry_id": "menu", "pair_role": "source"},
        }
    )
    assert text == "菜单 · 源帧 · 菜单源帧 · 主界面菜单可见"
    assert "await_start" not in text
    assert "readonly_ui_fine" not in text


def test_format_job_content_idle_is_empty():
    assert format_job_content({"status": "idle"}) == "无"


def test_format_job_status_hides_internal_codes():
    waiting = format_job_status({"status": "await_start", "action": "readonly_ui_fine"})
    running = format_job_status({"status": "queued"})
    done = format_job_status({"status": "ok"})
    assert waiting == "等待采集"
    assert running == "采集中"
    assert done == "已完成"
    for text in (waiting, running, done):
        assert "await_start" not in text
        assert "queued" not in text


def test_format_job_status_asks_to_confirm_marker():
    job = {
        "id": "job-1",
        "action": "readonly_ui_fine",
        "method": "B",
        "status": "ok",
        "args": {"entry_id": "menu", "pair_role": "source", "capture_id": "menu-1"},
        "result": {
            "locator": {
                "feature_ok": False,
                "position_ok": True,
                "accepted": False,
                "provisional": True,
                "click": {"screen": [1500, 80], "client": [1200, 40]},
            }
        },
    }
    assert format_job_status(job) == "请拖动获取框后点确认"
    assert format_job_status(job, overlay_available=False) == "获取框不可用"
    job["result"]["overlay_evidence"] = {"ok": False}
    assert format_job_status(job) == "请拖动获取框后点确认"
    job["result"]["overlay_confirmation"] = {"decision": "correct"}
    assert format_job_status(job) == "位置正确"
    job["result"]["overlay_confirmation"] = {"decision": "wrong"}
    assert format_job_status(job) == "位置错误"


def test_format_mem_panel_shows_xy_and_params():
    snap = SimpleNamespace(
        mem_status_label="已读",
        mem_xy=(12.5, 34.0),
        ui_label="主界面",
        geo="都柏林",
        ui_params="inv=0 menu=1 dlg=0 stp=0",
        ui_relation="生活技能界面",
    )
    ms = SimpleNamespace(
        menu_open=True,
        dialog_active=False,
        map_open=False,
        profile_open=False,
        in_combat=False,
        combat_mode_ui=False,
    )
    text = format_mem_panel(snap, ms)
    assert "内存：已读" in text
    assert "坐标：12.5, 34.0" in text
    assert "地点：都柏林" in text
    assert "stp=0" in text
    assert "inv=" not in text
    assert "menu=" not in text
    assert not any(ln.startswith("参数：") for ln in text.splitlines())
    assert not any(ln.startswith("细分：") for ln in text.splitlines())
    assert "界面：" not in text
    assert "菜单开" not in text
    assert "背包关" not in text
    assert "对话关" not in text
    assert "动作" not in text
    assert "关系：" not in text


def test_format_mem_panel_reads_mem_scene_field_names():
    snap = SimpleNamespace(
        mem_status_label="正常",
        mem_xy=(76.1, -31.2),
        ui_label="未知",
        mem_ui="非战斗",
        geo="",
        ui_params="inv=0 menu=0 dlg=0 stp=1",
        ui_relation="主界面",
    )
    ms = SimpleNamespace(
        menu_open=False,
        dialog_active=False,
        map_open=False,
        profile_open=False,
        in_combat=False,
        combat_mode_ui=False,
    )
    text = format_mem_panel(snap, ms)
    assert "stp=1" in text
    assert "界面：" not in text
    assert "非战斗" not in text
    assert "菜单关" not in text
    assert "关系：" not in text
    assert not any(ln.startswith("参数：") for ln in text.splitlines())
    assert not any(ln.startswith("细分：") for ln in text.splitlines())


def test_format_mem_panel_reads_snapshot_mem_xy():
    snap = SimpleNamespace(
        mem_status_label="正常",
        mem_xy=(76.1, -31.2),
        ui_label="非战斗",
        geo="",
        ui_params="inv=0 menu=0 dlg=0 stp=0",
    )
    ms = SimpleNamespace(
        menu_open=False,
        dialog_active=False,
        map_open=False,
        profile_open=False,
        in_combat=False,
        combat_mode_ui=False,
    )
    text = format_mem_panel(snap, ms)
    assert "内存：正常" in text
    assert "stp=0" in text
    assert "坐标：76.1, -31.2" in text
    assert "菜单关" not in text
    assert not any(ln.startswith("参数：") for ln in text.splitlines())


def test_format_mem_panel_waiting():
    assert format_mem_panel(None) == "内存：等待首拍…"


def test_format_mem_panel_hides_nav_param():
    snap = SimpleNamespace(
        mem_status_label="正常",
        mem_xy=(0.0, 0.0),
        ui_label="主界面",
        geo="",
        ui_params="inv=0 menu=1 dlg=0 stp=0 nav=背包",
    )
    text = format_mem_panel(snap)
    assert not any(ln.startswith("参数：") for ln in text.splitlines())
    assert "nav=" not in text
    assert "menu=1" not in text
    assert "stp=0" in text


def test_format_warm_progress():
    assert format_warm_progress(3, 11) == "读取 3/11"
    assert format_warm_progress(0, 0) == "读取 0/0"
    assert format_warm_progress(0, 11) == "读取 0/11"


def test_format_mem_panel_omits_stp_when_params_blank():
    snap = SimpleNamespace(
        mem_status_label="正常",
        mem_xy=(1.0, 2.0),
        geo="",
        ui_params="—",
        ui_stack="—",
    )
    text = format_mem_panel(snap)
    assert "坐标：" in text
    assert "1.0, 2.0" in text
    assert "stp=" in text
    assert "界面参数" not in text
    assert "\n" not in text


def test_format_mem_panel_hides_origin_junk_xy():
    snap = SimpleNamespace(
        mem_status_label="正常",
        mem_xy=(-0.0, 0.0),
        geo="冰霜峡谷",
        ui_params="stp=1",
    )
    text = format_mem_panel(snap)
    assert "坐标：—" in text
    assert "-0.0" not in text
    assert "0.0, 0.0" not in text
    assert "地点：冰霜峡谷" in text
    snap.mem_xy = (12.5, -3.0)
    text = format_mem_panel(snap)
    assert "坐标：12.5, -3.0" in text


def test_format_mem_panel_appends_stack_after_stp():
    snap = SimpleNamespace(
        mem_status_label="已读",
        mem_xy=(1.0, 2.0),
        geo="",
        ui_params="stp=0",
        ui_stack="菜单 > TutorialTooltipUIState",
    )
    text = format_mem_panel(snap)
    assert text.endswith("stp=0  菜单 > TutorialTooltipUIState")
    assert "界面参数" not in text


def test_format_mem_panel_keeps_xy_and_stp_when_stack_unready():
    snap = SimpleNamespace(
        mem_status_label="正常",
        mem_xy=(3.0, 4.0),
        geo="杜巴頓",
        ui_params="stp=0",
        ui_stack="—",
    )
    text = format_mem_panel(snap)
    assert "坐标：3.0, 4.0" in text
    assert "地点：杜巴頓" in text
    assert "stp=0" in text
    assert "界面参数" not in text
    assert "菜单" not in text


def test_warm_base_limits_are_geo_and_map():
    from mm_assist.memory.readers.scene import WARM_BASE_LIMITS

    assert set(WARM_BASE_LIMITS) == {
        "GameSpaceAPI",
        "ChannelIdentifierComponent",
        "CurrentViewingMapUIComponent",
    }


def test_warm_progress_caps_until_params_ready():
    from mm_assist.memory.readers.scene import Il2CppSceneReader, MemSceneInfo, WARM_BASE_LIMITS

    r = Il2CppSceneReader()
    r._has_live_obj = lambda name: True  # type: ignore[method-assign]
    r._cache = None
    loaded, total = r.warm_load_progress()
    assert total == len(WARM_BASE_LIMITS)
    assert loaded == total - 1
    r._cache = MemSceneInfo(status="ok")
    assert r.warm_load_progress()[0] == total - 1
    r._cache.display_name = "都柏林"
    assert r.warm_load_progress()[0] == total


def test_viewing_map_counts_loaded_when_resolved_empty():
    from mm_assist.memory.readers.scene import Il2CppSceneReader, MemSceneInfo

    r = Il2CppSceneReader()
    r._h = object()  # type: ignore[assignment]
    r._klasses = {"CurrentViewingMapUIComponent": 0x100000}
    r._objs = {"CurrentViewingMapUIComponent": []}
    r._resolved = True
    r._cache = MemSceneInfo(status="ok", display_name="庫漢")
    assert r._has_live_obj("CurrentViewingMapUIComponent") is True


def test_heap_scan_calls_on_first_hit():
    import inspect

    from mm_assist.memory import il2cpp
    from mm_assist.memory.readers.scene import Il2CppSceneReader

    src = inspect.getsource(il2cpp._scan_multi)
    assert "on_first_hit" in src
    src2 = inspect.getsource(Il2CppSceneReader._collect_many)
    assert "on_first_hit" in src2


def test_clear_state_ui_unlinks_inspect_page(tmp_path):
    from mm_assist.perception.state_ui import clear_state_ui, state_ui_path

    logs = tmp_path / "logs"
    logs.mkdir()
    su = state_ui_path(tmp_path)
    insp = su.with_name("ui_inspect.json")
    su.write_text("{}", encoding="utf-8")
    insp.write_text('{"node": "main_hud"}\n', encoding="utf-8")
    clear_state_ui(tmp_path)
    assert not su.exists()
    assert not insp.exists()


def test_format_mem_panel_has_stack_line():
    snap = SimpleNamespace(
        mem_status_label="已读",
        mem_xy=(1.0, 2.0),
        geo="都柏林",
        ui_params="inv=0 menu=1 dlg=0 stp=0",
        ui_stack="对话 > HudUIStepState > SkipButtonUIState > SkipCircleUIState > DialogueInteractionUIState",
        ui_relation="制作界面",
    )
    text = format_mem_panel(snap)
    assert "stp=0" in text
    assert "对话 > HudUIStepState" in text
    assert "界面参数" not in text
    assert "inv=" not in text
    assert "内存：已读" in text
    assert "坐标：1.0, 2.0" in text
    assert "地点：都柏林" in text


def test_format_mem_panel_stack_placeholder():
    snap = SimpleNamespace(
        mem_status_label="已读",
        mem_xy=(0.0, 0.0),
        geo="",
        ui_params="inv=0 menu=0 dlg=0 stp=0",
        ui_stack="",
    )
    text = format_mem_panel(snap)
    assert "stp=0" in text
    assert "界面参数" not in text
    assert "\n" not in text


def test_format_mem_panel_shows_stack_line():
    snap = SimpleNamespace(
        mem_status_label="已读",
        mem_xy=(1.0, 2.0),
        geo="",
        ui_params="stp=0",
        ui_stack="菜单 > TutorialTooltipUIState",
    )
    text = format_mem_panel(snap)
    assert "stp=0" in text
    assert "菜单 > TutorialTooltipUIState" in text
    assert "界面参数" not in text
