"""Real dialog pixels must stop Life, including normal mode and pending actions."""
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests/fixtures/game_stop'


def module():
    return importlib.import_module('mm_assist.life_plan.game_stop')


def pixels(name='both_1920.png'):
    return cv2.imdecode(np.fromfile(FIX / name, np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize('only', ['runtime', 'disconnect', 'both'])
@pytest.mark.parametrize('scale', [1, .5, .385416667])
def test_either_dialog_alone_is_sufficient(only, scale):
    frame = pixels()
    if only == 'runtime':
        frame[490:955, 570:1350] = 30
    if only == 'disconnect':
        frame[45:305, 630:1290] = 30
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    assert set(module().detect_game_stop(frame)) == ({only} if only != 'both' else {'runtime', 'disconnect'})


def test_user_window_client_crop_also_matches():
    # User's original window includes a 22px title bar; pixels below are client.
    frame = pixels('user_window.png')[22:577, :740]
    assert set(module().detect_game_stop(frame)) == {'runtime', 'disconnect'}


@pytest.mark.parametrize('which', ['title', 'button', 'none'])
def test_partial_or_common_confirm_is_not_dialog_identity(which):
    frame = pixels()
    if which != 'title':
        frame[80:155, 830:1090] = 30
        frame[555:605, 710:1205] = 30
    if which != 'button':
        frame[208:245, 925:997] = 30
        frame[835:875, 925:997] = 30
    assert module().detect_game_stop(frame) == ()


def test_registration_retains_original_and_crop_hashes():
    catalog = json.loads((ROOT / 'knowledge/templates/game_stop/catalog.json').read_text('utf-8'))
    assert hashlib.sha256((FIX / 'both_1920.png').read_bytes()).hexdigest() == catalog['source_sha256']
    frame = pixels()
    for dialog in catalog['dialogs']:
        for anchor in dialog['anchors']:
            path = ROOT / 'knowledge/templates/game_stop' / anchor['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == anchor['sha256']
            x1,y1,x2,y2 = anchor['crop']
            assert np.array_equal(cv2.imdecode(np.fromfile(path,np.uint8),1), frame[y1:y2,x1:x2])


@pytest.mark.parametrize('name',['cloth_facility_misread_20261013.png',
    'insect_items_20261011.png','herb_list_failed_20261008.png','repair_hud_failure_20261011.png'])
def test_other_actual_pages_are_not_stop_dialogs(name):
    path=ROOT/'tests/fixtures/process_stance'/name
    frame=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
    assert module().detect_game_stop(frame)==()


@pytest.fixture
def life(monkeypatch):
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()
    bar_ui.reset_recovery_for_tests()
    current = PlanItem('p1','feed','投料 布料', facility='cloth', recipe='布料')
    tail = PlanItem('p2','claim','领取金屬',facility='metal')
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:[current,tail])
    controller.start_life(snapshot=LifeSnapshot(ok=True),now=1000)
    monkeypatch.setattr(bar_ui,'_execution_page_ready',lambda _:True)
    monkeypatch.setattr('mm_assist.life_plan.action_finish.read_completion_stp',
                        lambda: SimpleNamespace(fresh=True, stp=0))
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',
                        lambda *a,**k:pytest.fail('dialog must never trigger recovery'))
    monkeypatch.setattr(module(),'_capture',lambda *a:pixels())
    yield current,tail
    controller.reset_controller_for_tests()


@pytest.mark.parametrize('kind',['feed','claim','gather','repair'])
@pytest.mark.parametrize('tracking',[False,True])
def test_pump_stops_preserving_current_and_tail(life,monkeypatch,kind,tracking):
    from dataclasses import replace
    current,tail = life
    current = replace(current,kind=kind)
    controller._STATE.current = current
    controller._STATE.queue[0] = current
    controller._STATE.pause_on_failure = tracking
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
                        lambda *a,**k:pytest.fail('no Life action beneath dialog'))
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=123))
    state = controller.get_controller()
    assert state.paused and not state.running
    assert state.current == current and state.queue == [current,tail]
    assert '遊戲已執行過久' in state.current_action
    assert '連線已中斷' in state.current_action


def test_pending_action_is_preserved_not_acknowledged(life):
    current,tail = life
    controller.wait_action_completion(current.id,1000,ExecuteResult(True,'done',action_issued=True),actions_complete=True)
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=123))
    state = controller.get_controller()
    assert state.paused and state.pending_action is not None
    assert state.pending_action.actions_complete and state.queue == [current,tail]


def test_popup_appearing_in_executor_aborts_before_input(life,monkeypatch):
    m = module()
    frames = iter([np.full((720,960,3),30,np.uint8),pixels()])
    monkeypatch.setattr(m,'_capture',lambda *a:next(frames))
    clicks=[]
    def execute(*a,**k):
        from mm_assist.safety.input import _guard_before_input
        _guard_before_input('click',SimpleNamespace(hwnd=123),method='sendinput',client_xy=(10,10))
        clicks.append((10,10))
        return ExecuteResult(True,'done')
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',execute)
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=123))
    assert clicks == [] and controller.get_controller().paused


def test_late_old_capture_cannot_pause_new_start(life,monkeypatch):
    m = module()
    def capture(*a):
        controller.start_life(snapshot=LifeSnapshot(ok=True),now=2000)
        return pixels()
    monkeypatch.setattr(m,'_capture',capture)
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=123))
    state=controller.get_controller()
    assert state.life_started_at==2000 and state.running and not state.paused


def test_user_pause_during_capture_is_not_overwritten(life,monkeypatch):
    def capture(*a):
        controller.pause_life()
        return pixels()
    monkeypatch.setattr(module(),'_capture',capture)
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=123))
    state=controller.get_controller()
    assert state.paused and state.current_action == '已暂停：投料 布料'


def test_bad_capture_stops_without_calling_it_no_dialog(life,monkeypatch):
    controller._STATE.pause_on_failure = True  # Normal capture failure holds, then skips after verification.
    monkeypatch.setattr(module(),'_capture',lambda *a:None)
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=123))
    assert controller.get_controller().paused
    assert '未能核验' in controller.get_controller().current_action


def test_removed_dialog_does_not_auto_resume(life,monkeypatch):
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=123))
    monkeypatch.setattr(module(),'_capture',lambda *a:np.full((720,960,3),30,np.uint8))
    bar_ui.pump_life_once(None,SimpleNamespace(hwnd=123))
    assert controller.get_controller().paused
