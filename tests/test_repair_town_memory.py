"""Repair navigation must consume the same fresh town identity as recognition."""
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from mm_assist.config import AppConfig
from mm_assist.memory.readers import ui_stack
from mm_assist.nav import map_atlas as maps, map_localize as localize
from mm_assist.platform.window import WindowInfo

FIX = Path(__file__).parent / 'fixtures/map_memory'


def recorded(name):
    row = json.loads((FIX / f'{name}.json').read_text(encoding='utf-8'))
    frame = cv2.imdecode(np.fromfile(FIX / f'{name}.png', np.uint8), 1)
    ui = row['ui']
    snap = ui_stack.UiStackSnap(names=ui['ui_stack_names'], ready=True,
        map_identity=tuple(ui['map_identity']), map_view=ui['map_view'])
    return row, frame, snap


@pytest.fixture
def navigation(monkeypatch):
    row, town, town_snap = recorded('dunbarton_enter_20261021')
    _, world, world_snap = recorded('world_bankor_20261014')
    assert town.shape == world.shape == (1440, 1920, 3)
    state = NS(frame=town, snap=town_snap, clicks=[], arrivals=[])
    win = WindowInfo(0, 'offline', 0, 0, 1920, 1440)
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: NS(
        read=lambda **kw: replace(state.snap, sampled_ts=time.time())))
    monkeypatch.setattr(maps, 'dismiss_pre_map_blockers', lambda cfg, w: w)
    monkeypatch.setattr(maps, 'load_whitelist', lambda cfg: {})
    monkeypatch.setattr('mm_assist.nav.compat.in_production_mode', lambda: True)
    monkeypatch.setattr(maps, 'open_map', lambda *a, **kw: maps.MapSnapshot(
        state='map', lines=[], blob=''))
    monkeypatch.setattr(maps, '_foreground_grab', lambda *a, **kw: (win, state.frame))
    monkeypatch.setattr(maps, '_grab_map_frame', lambda *a, **kw: (win, state.frame))
    monkeypatch.setattr(maps, 'snap', lambda *a, **kw: (win, state.frame, [], None))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr(maps, '_click', lambda *a, **kw: pytest.fail('no breadcrumb click'))
    monkeypatch.setattr(maps, '_drag_map', lambda *a, **kw: pytest.fail('no native drag'))
    def arrive(*a, **kw):
        state.arrivals.append(kw)
        return {'ok': True, 'stage': 'npc_navigation_boundary'}
    monkeypatch.setattr(maps, '_map_goto_npc_arrive', arrive)
    return NS(state=state, win=win, town=town, town_snap=town_snap,
              world=world, world_snap=world_snap, row=row)


def goto(win):
    return maps._map_goto_once(AppConfig(), win, region='杜巴頓', npc='奈麗絲',
        timeout_s=1, idle_abort_s=1, npc_only=True, ban_smith_outside_town=True)


def test_failure_fixture_is_original_and_shared_recognition_proves_town(navigation):
    n = navigation
    assert hashlib.sha256((FIX/'dunbarton_enter_20261021.png').read_bytes()).hexdigest() == n.row['sha256']
    assert maps._current_map_parameters(n.town, n.win) == 'map_dunbarton'


def test_already_loaded_town_reaches_npc_navigation_without_ascent(navigation, monkeypatch):
    monkeypatch.setattr(maps, 'continent_pixel_enter_dubarton',
                        lambda *a, **kw: pytest.fail('fresh named town must not ascend'))
    result = goto(navigation.win)
    assert result['stage'] == 'npc_navigation_boundary'
    assert len(navigation.state.arrivals) == 1
    assert navigation.state.arrivals[0]['allow_smith'] is False


def test_continent_entry_already_in_named_town_never_clicks(navigation, monkeypatch):
    monkeypatch.setattr(maps, 'click_ulara_text_and_ocr',
                        lambda *a: pytest.fail('already in town; no ascent or atlas'))
    result = maps.continent_pixel_enter_dubarton(AppConfig(), navigation.win,
                                               lines=[], img=navigation.town)
    assert result['ok'] and result['stage'] == 'already_town'
    assert navigation.state.arrivals == []  # A map identity is no NPC/repair result.


def enter_from_world(n, monkeypatch):
    n.state.frame, n.state.snap = n.world, n.world_snap
    monkeypatch.setattr(localize, 'drag_until_near', lambda **kw: NS(
        ok=True, loc=NS(ok=True, score=.9), note='offline-localized'))
    monkeypatch.setattr(localize, 'atlas_xy_to_client_norm', lambda *a: (.452, .406))
    monkeypatch.setattr(maps, '_find_dubarton_castle_norm', lambda *a: None)
    def click(*a, **kw):
        n.state.clicks.append(a[1:])
        n.state.frame, n.state.snap = n.town, n.town_snap
    monkeypatch.setattr(maps, '_click_norm', click)
    monkeypatch.setattr(maps, '_wait_action_effect', lambda *a, **kw:
                        (n.win, n.state.frame, True, 17.48))


def test_city_click_accepts_fresh_named_town_when_ocr_is_empty(navigation, monkeypatch):
    n = navigation
    enter_from_world(n, monkeypatch)
    result = maps.continent_pixel_enter_dubarton(AppConfig(), n.win,
                                               lines=[], img=n.world)
    assert result['ok'] and result['level'] == 'town'
    assert len(n.state.clicks) == 1 and n.state.arrivals == []


def test_entire_map_route_rechecks_town_before_npc_navigation(navigation, monkeypatch):
    n = navigation
    enter_from_world(n, monkeypatch)
    result = goto(n.win)
    assert result['stage'] == 'npc_navigation_boundary'
    assert len(n.state.clicks) == len(n.state.arrivals) == 1
    assert n.state.arrivals[0]['allow_smith'] is False


def test_proven_town_never_seeks_a_second_city_click(navigation, monkeypatch):
    n = navigation
    enter_from_world(n, monkeypatch)
    def castle(*a):
        if n.state.frame is n.town:
            pytest.fail('town is proven; do not seek or re-click the city marker')
        return None
    monkeypatch.setattr(maps, '_find_dubarton_castle_norm', castle)
    result = maps.continent_pixel_enter_dubarton(AppConfig(), n.win, lines=[], img=n.world)
    assert result['ok'] and len(n.state.clicks) == 1


def test_wrong_loaded_resource_after_city_click_is_not_town_success(navigation, monkeypatch):
    n = navigation
    enter_from_world(n, monkeypatch)
    n.town_snap = replace(n.town_snap, map_view={**n.town_snap.map_view,
        'resource_name': 'Bankor'})
    result = maps.continent_pixel_enter_dubarton(AppConfig(), n.win, lines=[], img=n.world)
    assert not result['ok'] and result['stage'] == 'continent_enter'
    assert len(n.state.clicks) == 1 and n.state.arrivals == []


def test_city_result_cannot_authorize_npc_after_resource_changes(navigation, monkeypatch):
    n = navigation
    enter_from_world(n, monkeypatch)
    captures = []
    def snap(*a, **kw):
        captures.append(1)
        if len(captures) == 2:
            n.state.snap = replace(n.town_snap, map_view={**n.town_snap.map_view,
                'resource_name': 'Bankor', 'titles': {'town': '班克爾'}})
        return n.win, n.state.frame, [], None
    monkeypatch.setattr(maps, 'snap', snap)
    result = goto(n.win)
    assert not result['ok'] and result['stage'] == 'continent_enter'
    assert len(n.state.clicks) == 1 and n.state.arrivals == []


@pytest.mark.parametrize('fault', ['stale', 'unready', 'covered', 'identity_change',
    'resource_change', 'wrong_resource', 'wrong_title', 'wrong_type', 'reused'])
def test_invalid_town_memory_never_reaches_npc_without_visual_evidence(navigation, monkeypatch, fault):
    n = navigation
    calls = []
    base = n.town_snap
    fact = dict(base.map_view)
    if fault == 'wrong_resource': fact['resource_name'] = 'Bankor'
    if fault == 'wrong_title': fact['titles'] = {'town': '班克爾'}
    if fault == 'wrong_type': fact['map_type'] = 1
    base = replace(base, map_view=fact, ready=fault != 'unready',
        names=base.names + (['UnknownPopup'] if fault == 'covered' else []),
        error='stack_read_reused' if fault == 'reused' else '')
    def read(**kw):
        calls.append(1)
        return replace(base, sampled_ts=time.time()-(10 if fault == 'stale' else 0),
            map_identity=(1, 2, 3) if fault == 'identity_change' and len(calls)%2 == 0 else base.map_identity,
            map_view={**fact, 'resource_name': 'Bankor'}
                if fault == 'resource_change' and len(calls)%2 == 0 else fact)
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: NS(read=read))
    # This scenario has no independent visual map identity. Existing image/OCR
    # fallback is separately covered; malformed memory cannot authorize it.
    monkeypatch.setattr('mm_assist.life_plan.node_inspect._inspect_map_how', lambda *a: None)
    monkeypatch.setattr(maps, 'continent_pixel_enter_dubarton',
                        lambda *a, **kw: {'ok': False, 'stage': 'unproven'})
    result = goto(n.win)
    assert not result['ok'] and n.state.arrivals == [] and n.state.clicks == []
