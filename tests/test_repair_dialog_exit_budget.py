"""A last authorized goodbye needs one independent read, never another input."""
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import pytest

from mm_assist.config import AppConfig
from mm_assist.life_execute import repair_dialog as dialog
from mm_assist.memory.readers import ui_stack
from mm_assist.platform.window import WindowInfo
from mm_assist.safety import autofarm_gates as gates

FIX = Path(__file__).parent / 'fixtures/repair'


@pytest.fixture
def exit_scene(monkeypatch):
    row = json.loads((FIX/'leave_dialog_hud_20261021.json').read_text(encoding='utf-8'))
    images = {name: cv2.imread(str(FIX/file)) for name, file in {
        'goodbye': 'nerys_after_end_dialog.png',
        'hud': 'leave_dialog_hud_20261021.png'}.items()}
    assert all(im is not None for im in images.values())
    state = NS(clock=100., page='goodbye', grabs=[], clicks=[], reads=0,
        grab_cost=3.5, click_cost=2.3, fault='', cancelled=False,
        cancel_on_read=False, read_error=False, row=row, images=images)
    monkeypatch.setattr(dialog.time, 'monotonic', lambda: state.clock)
    monkeypatch.setattr(dialog.time, 'sleep', lambda seconds: setattr(state, 'clock', state.clock+seconds))
    monkeypatch.setattr(dialog, 'op_cancelled', lambda: state.cancelled)
    def grab(cfg, win):
        state.grabs.append((state.page, state.clock))
        state.clock += state.grab_cost if state.page == 'goodbye' else .2
        image = images[state.page]
        return WindowInfo(0, 'offline', 0, 0, image.shape[1], image.shape[0]), image
    def click(win, x, y, **kw):
        state.clicks.append((state.clock, x, y))
        state.clock += state.click_cost
        state.page = 'hud'
        return True
    def read(**kw):
        state.reads += 1
        if state.read_error:
            raise RuntimeError('independent stack unavailable')
        if state.cancel_on_read:
            state.cancelled = True
        names = list(row['ui']['ui_stack_names'])
        if state.fault == 'dialog': names += ['DialogueInteractionUIState']
        if state.fault == 'blocked': names += ['ClickShieldUIState']
        if state.fault == 'unknown': names += ['UnknownPopup']
        state.clock += .15
        return ui_stack.UiStackSnap(names=names, ready=state.fault != 'unready',
            sampled_ts=time.time()-(10 if state.fault == 'stale' else 0)
                +(10 if ((state.fault == 'future_first' and state.reads%2 == 1)
                         or (state.fault == 'future_second' and state.reads%2 == 0)) else 0),
            error='stack_read_reused' if state.fault == 'reused' else '')
    monkeypatch.setattr(dialog, '_grab_frame', grab)
    monkeypatch.setattr(dialog, 'click_client', click)
    monkeypatch.setattr(ui_stack, 'shared_ui_stack_reader', lambda: NS(read=read))
    monkeypatch.setattr(dialog, '_peek_dialog_active', lambda: True)
    gates.reset_stage_gates_for_tests()
    # The end action was already dispatched by the caller; independent return
    # evidence is required before either end/goodbye gate can be cleared.
    assert gates.stage_click('npc_end_dialog', lambda *a: True, 1, 1)
    yield state
    gates.reset_stage_gates_for_tests()


def wait(state):
    frame = state.images['goodbye']
    win = WindowInfo(0, 'offline', 0, 0, frame.shape[1], frame.shape[0])
    return dialog._wait_leave_dialog(AppConfig(), win, timeout_s=6)[1]


def test_registered_current_failure_hud_remains_independent_evidence(exit_scene):
    s = exit_scene
    assert hashlib.sha256((FIX/'leave_dialog_hud_20261021.png').read_bytes()).hexdigest() == s.row['sha256']
    assert dialog._hud_menu_xy(s.images['hud']) is not None
    assert not dialog.npc_dialog_visible(s.images['hud'])


def test_last_in_budget_goodbye_gets_one_readonly_post_input_verification(exit_scene):
    s = exit_scene
    assert wait(s)
    assert len(s.clicks) == 1 and s.clicks[0][0] < 106
    assert [page for page, _ in s.grabs] == ['goodbye', 'hud']
    assert s.reads == 2  # Real resolve_navigation pairs the current stack.
    assert 106 < s.clock < 107
    assert gates._stage_counts['npc_goodbye_dismiss'] == 0
    assert gates._stage_counts['npc_end_dialog'] == 0


def test_last_regular_poll_can_only_add_one_readonly_result_poll(exit_scene, monkeypatch):
    s = exit_scene
    s.grab_cost, s.click_cost = 0., .1
    def grab(cfg, win):
        if len(s.grabs) < 23:
            s.grabs.append(('pending', s.clock))
            return win, None
        s.grabs.append((s.page, s.clock))
        return win, s.images[s.page]
    monkeypatch.setattr(dialog, '_grab_frame', grab)
    assert wait(s)
    assert len(s.clicks) == 1 and s.clicks[0][0] < 106
    assert len(s.grabs) == 25 and [p for p, _ in s.grabs[-2:]] == ['goodbye', 'hud']
    assert s.reads == 2


@pytest.mark.parametrize('fault', ['dialog', 'blocked', 'unknown', 'stale', 'unready', 'reused',
                                 'future_first', 'future_second'])
def test_final_read_never_accepts_unknown_or_old_stack_or_replays(exit_scene, fault):
    s = exit_scene
    s.fault = fault
    assert not wait(s)
    assert len(s.clicks) == 1
    assert [page for page, _ in s.grabs] == ['goodbye', 'hud']
    assert gates._stage_counts['npc_goodbye_dismiss'] == 1
    assert gates._stage_counts['npc_end_dialog'] == 1


@pytest.mark.parametrize('fault', ['dialog', 'blocked', 'stale', 'unready'])
def test_hud_pixels_alone_never_clear_gates_before_stack_verification(exit_scene, fault):
    s = exit_scene
    s.grab_cost, s.click_cost, s.fault = .1, .1, fault
    assert not wait(s)
    assert len(s.clicks) == 1
    assert gates._stage_counts['npc_goodbye_dismiss'] == 1
    assert gates._stage_counts['npc_end_dialog'] == 1


def test_slow_initial_capture_cannot_authorize_goodbye_after_deadline(exit_scene):
    s = exit_scene
    s.grab_cost = 6.1
    assert not wait(s)
    assert s.clicks == [] and s.reads == 0
    assert len(s.grabs) == 1
    assert gates._stage_counts['npc_end_dialog'] == 1


def test_pause_during_final_paired_read_never_clears_or_replays(exit_scene):
    s = exit_scene
    s.cancel_on_read = True
    assert not wait(s)
    assert len(s.clicks) == 1 and len(s.grabs) == 2
    assert gates._stage_counts['npc_goodbye_dismiss'] == 1
    assert gates._stage_counts['npc_end_dialog'] == 1
    s.cancelled = s.cancel_on_read = False
    assert wait(s)
    assert len(s.clicks) == 1 and len(s.grabs) == 3
    assert gates._stage_counts['npc_goodbye_dismiss'] == 0
    assert gates._stage_counts['npc_end_dialog'] == 0


def test_read_error_after_final_input_preserves_pending_gates(exit_scene):
    s = exit_scene
    s.read_error = True
    with pytest.raises(RuntimeError, match='independent stack unavailable'):
        wait(s)
    assert len(s.clicks) == 1 and len(s.grabs) == 2
    assert gates._stage_counts['npc_goodbye_dismiss'] == 1
    assert gates._stage_counts['npc_end_dialog'] == 1
    s.read_error = False
    assert wait(s)
    assert len(s.clicks) == 1 and len(s.grabs) == 3
    assert gates._stage_counts['npc_goodbye_dismiss'] == 0
    assert gates._stage_counts['npc_end_dialog'] == 0
