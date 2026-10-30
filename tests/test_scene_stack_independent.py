from types import SimpleNamespace

from mm_assist.config import load_config
from mm_assist.memory.readers.ui_stack import UiStackSnap
from mm_assist.perception.scene import SceneMonitor
from mm_assist.perception.state_ui import build_state_ui_payload


def test_background_startup_uses_bounded_base_warmup(monkeypatch):
    from mm_assist.memory.readers.scene import Il2CppSceneReader
    reader = Il2CppSceneReader()
    calls = []
    monkeypatch.setattr(reader, "warm_bundle", lambda *a, **k: calls.append(k) or (None, {}))
    monkeypatch.setattr(reader, "read", lambda *a, **k: calls.append({"unbounded": True}))
    reader.schedule_resolve(hwnd=123)
    reader._bg.join(timeout=1)
    assert not reader._bg.is_alive()
    assert not reader.resolving
    assert calls == [{"max_wall_s": 4.5}]


def test_inventory_stack_survives_missing_location_reader(monkeypatch):
    mon = SceneMonitor(load_config())
    mon._mem = SimpleNamespace(read=lambda **k: None, last_error="pending", resolving=False)
    mon._scene_mem = SimpleNamespace(read=lambda **k: None, _cache=None, _pause_resolve=False, resolving=False)
    monkeypatch.setattr(mon, "_stack_snap", lambda: UiStackSnap(names=["EquipmentInventoryUIInstance"], ready=True, line="背包"))
    for name in ("_refresh_icons", "_maybe_refresh_action", "_maybe_refresh_menu_nav"):
        monkeypatch.setattr(mon, name, lambda **k: None)
    snap = mon.tick_lean()
    payload = build_state_ui_payload(snap, mon.last_ms)
    assert payload["ui_stack_ready"] is True
    assert payload["ui_stack_names"] == ["EquipmentInventoryUIInstance"]
    assert mon.last_ms.display_name == ""
