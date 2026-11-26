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


def framework_reader(monkeypatch):
    scene, reader, ghost, live = _reader(monkeypatch)
    root_class, static, root = 0x600000000, 0x700000000, 0x800000000
    reader._ga = 0x900000000
    reader._cfg['typeinfo_rva']['FrameworkSingleton'] = 123
    reader._cfg['offsets'].update({'Il2CppClass.static_fields':0xb8,
        'FrameworkSingleton.static.instance':0,'FrameworkSingleton.GameSpaceAPI':0xb0})
    original = scene.read_ptr
    pointers = {root_class+0xb8:static,static:root,root:root_class,root+0xb0:live}
    monkeypatch.setattr(scene,'resolve_klass',lambda *a:root_class)
    monkeypatch.setattr(scene,'read_ptr',lambda h,p:pointers[p] if p in pointers else original(h,p))
    return reader,pointers,live


def test_current_framework_root_supplies_location_without_heap_scan(monkeypatch):
    from mm_assist.memory.readers import scene
    reader,pointers,live = framework_reader(monkeypatch)
    monkeypatch.setattr(scene,'scan_klass_objects',lambda *a,**k:pytest.fail('stable static root must not heap scan'))
    reader._prime_gamespace_api()
    assert reader._objs['GameSpaceAPI'] == [live]
    assert reader._display_name_cache == '杜巴頓'


def test_warm_bundle_prefers_current_framework_over_cached_old_place(monkeypatch):
    from mm_assist.memory.readers import scene
    reader,pointers,live = framework_reader(monkeypatch)
    reader._pid = 1
    monkeypatch.setattr(reader,'_ensure_proc',lambda **k:True)
    monkeypatch.setattr(reader,'_resolve_klasses',lambda:True)
    def seeds(klasses):
        reader._display_name_cache = '旧地点'
        return 1
    monkeypatch.setattr(reader,'_apply_warm_seeds',seeds)
    monkeypatch.setattr(reader,'_save_warm_seeds',lambda *a,**k:None)
    monkeypatch.setattr(reader,'_read_once',lambda:scene.MemSceneInfo(status='ok',display_name=reader._display_name_cache))
    def scan(h,klasses,**kw):
        assert 'GameSpaceAPI' not in klasses
        return {}
    monkeypatch.setattr(scene,'scan_klass_objects_multi',scan)
    info,_ = reader.warm_bundle()
    assert info.display_name == '杜巴頓'
    assert reader._objs['GameSpaceAPI'] == [live]


@pytest.mark.parametrize('broken', [0x6000000b8,0x700000000,0x800000000,0x8000000b0])
def test_current_framework_root_rejects_missing_or_replaced_link(monkeypatch,broken):
    reader,pointers,live = framework_reader(monkeypatch)
    pointers[broken] = 0
    assert reader._framework_gamespace_api() == 0


def test_current_framework_root_rejects_late_replacement(monkeypatch):
    from mm_assist.memory.readers import scene
    reader,pointers,live = framework_reader(monkeypatch)
    original = scene.read_ptr
    calls = [0]
    def changed(h,p):
        if p == 0x8000000b0:
            calls[0] += 1
            if calls[0] > 1: return 0
        return original(h,p)
    monkeypatch.setattr(scene,'read_ptr',changed)
    assert reader._framework_gamespace_api() == 0


@pytest.mark.parametrize('fault', ['', 'token', 'namespace', 'name'])
def test_framework_root_binds_uninitialized_typeinfo_only_with_exact_type_identity(monkeypatch,fault):
    from mm_assist.memory.readers import scene, ui_stack
    reader,pointers,live = framework_reader(monkeypatch)
    original = scene.read_ptr
    actual_api,actual_data = reader._klasses['GameSpaceAPI'],reader._klasses['GameSpaceSheetData']
    reader._klasses.update(GameSpaceAPI=0x20000003,GameSpaceSheetData=0x20000005)
    reader._cfg['framework_type_tokens'] = {
        'GameSpaceAPI':{'token':0x20000003,'namespace':'A','name':'GameSpaceAPI'},
        'GameSpaceSheetData':{'token':0x20000005,'namespace':'B','name':'GameSpaceSheetData'}}
    if fault == 'token':reader._klasses['GameSpaceAPI'] = 0x20000007
    text_pointers={actual_api+0x10:1,actual_api+0x18:2,actual_data+0x10:3,actual_data+0x18:4}
    strings={1:'Other' if fault=='name' else 'GameSpaceAPI',2:'C' if fault=='namespace' else 'A',3:'GameSpaceSheetData',4:'B'}
    monkeypatch.setattr(scene,'read_ptr',lambda h,p:text_pointers[p] if p in text_pointers else original(h,p))
    monkeypatch.setattr(ui_stack,'read_c_string',lambda h,p:strings.get(p,''))
    result = reader._framework_gamespace_api()
    assert result == (0 if fault else live)
    if not fault:
        assert reader._klasses['GameSpaceAPI'] == actual_api
        assert reader._klasses['GameSpaceSheetData'] == actual_data
    else:
        assert reader._klasses['GameSpaceSheetData'] == 0x20000005


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
