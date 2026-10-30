"""A fresh crafting shell defers optional legacy map heap scans, never map facts."""
from types import SimpleNamespace as NS
import time
import pytest
from mm_assist.memory.readers import scene, ui_stack


@pytest.mark.parametrize('case',['craft','map','map_over_craft','covered','unready','stale','other_pid','error','missing','future'])
def test_optional_map_scan_yields_only_to_current_known_crafting_shell(monkeypatch,case):
    snapshot=ui_stack.UiStackSnap(names=['CraftingHubPopupUIInstance'],ready=True,sampled_ts=time.time())
    owner=NS(_pid=1,_last_snap=snapshot)
    if case=='map':snapshot.names=['MapUIInstance']
    if case=='map_over_craft':snapshot.names.append('MapUIInstance')
    if case=='covered':snapshot.names.append('UnknownPopup')
    if case=='unready':snapshot.ready=False
    if case=='stale':snapshot.sampled_ts-=5
    if case=='future':snapshot.sampled_ts+=5
    if case=='other_pid':owner._pid=2
    if case=='error':snapshot.error='stack_read_reused'
    if case=='missing':owner=None
    monkeypatch.setattr(ui_stack,'_SHARED_READER',owner)
    reader=NS(_closed=False,_map_rescanning=False,_h=object(),_pause_resolve=False,
        _resolving=False,_ui_refreshing=False,_display_name_cache='杜巴頓',_next_map_sync=0.,_pid=1)
    started=[]
    monkeypatch.setattr(scene.threading,'Thread',lambda **k:NS(start=lambda:started.append(k['name'])))
    scene.Il2CppSceneReader._schedule_map_rescan(reader)
    assert bool(started) is (case!='craft')
    assert reader._map_rescanning is (case!='craft')
