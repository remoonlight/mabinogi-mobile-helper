from mm_assist.memory.readers.scene import Il2CppSceneReader
import pytest


def _reader(monkeypatch):
    from mm_assist.memory.readers import scene

    reader = Il2CppSceneReader()
    reader._h = object()
    reader._klasses = {"GameSpaceAPI": 0x100000000, "GameSpaceSheetData": 0x200000000}
    ghost = 0x300000000
    live = 0x400000000
    data = 0x500000000
    ptrs = {ghost: reader._klasses["GameSpaceAPI"], live: reader._klasses["GameSpaceAPI"],
            live + 0x18: data, data: reader._klasses["GameSpaceSheetData"]}
    ptrs.update({ghost + i * 0x100: reader._klasses["GameSpaceAPI"] for i in range(96)})
    monkeypatch.setattr(scene, "read_ptr", lambda h, addr: ptrs.get(addr))
    monkeypatch.setattr(reader, "_read_display_from_gamespace", lambda addr: "杜巴頓" if addr == data else "")
    return scene, reader, ghost, live


def test_location_rescue_filters_ghosts_before_hit_limit(monkeypatch):
    scene, reader, ghost, live = _reader(monkeypatch)
    calls = []

    def scan(h, klass, *, max_hits, should_stop, max_wall_s, hit_filter=None, resume=False):
        assert resume is True
        calls.append((max_hits, max_wall_s))
        hits = []
        for obj in [ghost + i * 0x100 for i in range(96)] + [live]:
            if should_stop():
                break
            if hit_filter is None or hit_filter("_", obj):
                hits.append(obj)
                if len(hits) >= max_hits:
                    break
        return hits

    monkeypatch.setattr(scene, "scan_klass_objects", scan)
    reader._prime_gamespace_api(max_wall_s=2.0)
    assert reader._objs["GameSpaceAPI"] == [live]
    assert reader._display_name_cache == "杜巴頓"
    assert calls == [(1, 2.0)]


def test_location_rescue_does_not_retain_invalid_cached_candidate(monkeypatch):
    scene, reader, ghost, live = _reader(monkeypatch)
    reader._objs["GameSpaceAPI"] = [ghost]
    monkeypatch.setattr(scene, "scan_klass_objects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no rescan")))
    reader._prime_gamespace_api(rescan=False)
    assert reader._objs["GameSpaceAPI"] == []
    assert reader._display_name_cache == ""


def test_location_rescue_reuses_valid_cached_candidate(monkeypatch):
    scene, reader, ghost, live = _reader(monkeypatch)
    reader._objs["GameSpaceAPI"] = [live]
    monkeypatch.setattr(scene, "scan_klass_objects", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no rescan")))
    reader._prime_gamespace_api(rescan=False)
    assert reader._objs["GameSpaceAPI"] == [live]
    assert reader._display_name_cache == "杜巴頓"


@pytest.mark.parametrize("seeded,budget", [(1, 4.5), (0, 6.0)])
def test_warmup_location_rescue_shares_total_deadline(monkeypatch, seeded, budget):
    from mm_assist.memory.readers import scene
    reader = Il2CppSceneReader()
    reader._h, reader._pid, reader._ga = object(), 1, 0x100000000
    reader._klasses = {name: 0x100000000 + i * 0x100 for i, name in enumerate(scene.WARM_BASE_LIMITS)}
    now, calls = [0.0], []
    monkeypatch.setattr(scene.time, "perf_counter", lambda: now[0])
    monkeypatch.setenv("MM_STATE_WARM_COLD_S", "6")
    monkeypatch.setattr(reader, "_ensure_proc", lambda **k: True)
    monkeypatch.setattr(reader, "_resolve_klasses", lambda: True)
    monkeypatch.setattr(reader, "_apply_warm_seeds", lambda k: seeded)
    monkeypatch.setattr(reader, "_save_warm_seeds", lambda *a, **k: None)
    monkeypatch.setattr(reader, "_read_once", lambda: scene.MemSceneInfo(status="ok", display_name=reader._display_name_cache))
    def scan(*a, **k):
        calls.append(("main", k["max_wall_s"]))
        now[0] += k["max_wall_s"]
        return {}
    def geo(*, max_wall_s):
        calls.append(("geo", max_wall_s))
        now[0] += max_wall_s
        reader._display_name_cache = "杜巴頓"
    monkeypatch.setattr(scene, "scan_klass_objects_multi", scan)
    monkeypatch.setattr(reader, "_ensure_display_name", geo)
    info, _ = reader.warm_bundle(max_wall_s=4.5)
    assert now[0] == pytest.approx(budget)
    assert [name for name, _ in calls] == ["main", "geo"]
    assert [seconds for _, seconds in calls] == pytest.approx([budget * 0.4, budget * 0.6])
    assert info.display_name == "杜巴頓"


@pytest.mark.parametrize("flag", ["_pause_resolve", "_resolving", "_ui_refreshing"])
def test_warmup_does_not_start_competing_map_scan(monkeypatch, flag):
    from mm_assist.memory.readers import scene
    reader = Il2CppSceneReader()
    reader._h = object()
    reader._resolved = True
    reader._display_name_cache = "杜巴頓"
    setattr(reader, flag, True)
    monkeypatch.setattr(scene.threading, "Thread", lambda **k: pytest.fail("competing heap scan"))
    reader._schedule_map_rescan()
    reader._maybe_refresh_ui_objs(lean=True)
    assert not reader._map_rescanning
