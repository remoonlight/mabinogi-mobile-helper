"""A retained town breadcrumb must not hide the current loaded world map."""
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import AppConfig
from mm_assist.memory.readers import ui_stack
from mm_assist.nav import map_atlas as maps
from mm_assist.platform.window import WindowInfo

FIX = Path(__file__).parent / 'fixtures/map_memory'


def observation(place='world_bankor_20261014'):
    evidence = json.loads((FIX/f'{place}.json').read_text('utf-8'))
    ui = evidence['ui']
    frame = cv2.imdecode(np.fromfile(FIX/f'{place}.png', np.uint8), 1)
    snap = ui_stack.UiStackSnap(names=ui['ui_stack_names'], ready=True,
        map_identity=tuple(ui['map_identity']), map_view=ui['map_view'], sampled_ts=time.time())
    return frame, snap


def test_retained_world_pixels_localize_with_existing_geometry_gates():
    from mm_assist.nav import map_localize
    frame, _ = observation()
    atlas, _ = map_localize.load_continent_atlas(maps.maps_root(AppConfig()))
    result = map_localize.localize_viewport_zoom_search(frame, atlas, min_score=.35)
    assert result.ok and result.note.startswith('partial_features SIFT')
    assert result.score >= .70 and .54 < result.zoom < .55


def test_emain_failure_frame_localizes_against_updated_atlas():
    from mm_assist.nav import map_localize

    frame = cv2.imdecode(np.fromfile(FIX/'world_emain_failure_20261126.png', np.uint8), 1)
    atlas, _ = map_localize.load_continent_atlas(maps.maps_root(AppConfig()))
    result = map_localize.localize_viewport_zoom_search(frame, atlas, min_score=.35)
    assert result.ok and result.note.startswith('partial_features SIFT')
    assert result.score >= .70 and .54 < result.zoom < .55
    assert result.match_x == pytest.approx(-48.4, abs=3)
    assert result.match_y == pytest.approx(57.2, abs=3)


@pytest.mark.parametrize('place', ['world_bankor_20261014', 'world_emain_20261126'])
def test_loaded_world_resource_reaches_independent_localization_without_reclick(monkeypatch, place):
    frame, snap = observation(place)
    win = WindowInfo(0, 'offline', 0, 0, frame.shape[1], frame.shape[0])
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader',
        lambda: SimpleNamespace(read=lambda **kw: replace(snap, sampled_ts=time.time())))
    assert maps._current_map_parameters(frame, win) == 'map_continent'
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **kw: (win, frame))
    monkeypatch.setattr(maps, 'snap', lambda *a, **kw: (win, frame, [], None))
    monkeypatch.setattr(maps, '_click', lambda *a, **kw: pytest.fail('world is already loaded; no ascent'))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr('mm_assist.nav.map_localize.drag_until_near',
        lambda **kw: (_ for _ in ()).throw(RuntimeError('independent localization reached')))
    with pytest.raises(RuntimeError, match='independent localization reached'):
        maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=[], img=frame)


@pytest.mark.parametrize('fault', ['wrong_resource', 'wrong_title', 'town', 'stale', 'unready', 'transition'])
def test_world_proof_rejects_invalid_or_changing_memory(monkeypatch, fault):
    frame, snap = observation()
    fact = dict(snap.map_view)
    if fault == 'wrong_resource': fact['resource_name'] = 'OtherWorld'
    if fault == 'wrong_title': fact['titles'] = {'world': '其他大陸'}
    if fault == 'town': fact.update(map_type=2, resource_name='Bankor')
    snap = replace(snap, map_view=fact, ready=fault != 'unready')
    reads = 0
    def read(**kw):
        nonlocal reads
        reads += 1
        return replace(snap, sampled_ts=time.time()-(10 if fault == 'stale' else 0),
            map_identity=(1,2,3) if fault == 'transition' and reads % 2 == 0 else snap.map_identity)
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: SimpleNamespace(read=read))
    win = WindowInfo(0, 'offline', 0, 0, frame.shape[1], frame.shape[0])
    assert maps._current_map_parameters(frame, win) != 'map_continent'


@pytest.mark.parametrize('boundary', ['sample', 'drag'])
@pytest.mark.parametrize('replacement', ['town', 'registered_world'])
def test_resource_change_during_localization_stops_before_input(monkeypatch, boundary, replacement):
    frame, snap = observation()
    changed = False
    def read(**kw):
        fact = snap.map_view
        if changed:
            if replacement == 'town':
                fact = {**fact, 'resource_name': 'Bankor', 'map_type': 2}
            else:
                fact = {**fact, 'resource_name': 'WorldMap@v2'}
        return replace(snap, map_view=fact, sampled_ts=time.time())
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: SimpleNamespace(read=read))
    win = WindowInfo(0, 'offline', 0, 0, frame.shape[1], frame.shape[0])
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **kw: (win, frame))
    monkeypatch.setattr(maps, '_click', lambda *a, **kw: pytest.fail('no ascent'))
    monkeypatch.setattr(maps, '_drag_map', lambda *a, **kw: pytest.fail('changed resource cannot drag'))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    def localize(**kw):
        nonlocal changed
        if boundary == 'drag':
            assert kw['snap_fn']() is frame
        changed = True
        if boundary == 'sample':
            kw['snap_fn']()
        else:
            kw['drag_fn'](100, 100)
    monkeypatch.setattr('mm_assist.nav.map_localize.drag_until_near', localize)
    with pytest.raises(maps.GuardError, match='地图资源已变化'):
        maps.continent_pixel_enter_dubarton(AppConfig(), win, lines=[], img=frame)
