"""2026-11-08: two center clicks, one verified escape, then repeat NPC exit."""
import time
import hashlib
import json
from types import SimpleNamespace as NS

import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import repair_dialog as dialog, ui_logic
from mm_assist.life_execute.ui_recover import recover_once as real_recover
from mm_assist.life_plan import bar_ui, controller
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem
from mm_assist.safety import autofarm_gates as gates


@pytest.fixture
def scene(monkeypatch):
    goodbye = cv2.imread(str(ROOT/'tests/fixtures/repair/nerys_after_end_dialog.png'))
    hud = cv2.imread(str(ROOT/'tests/fixtures/repair/leave_dialog_hud_20261021.png'))
    assert goodbye is not None and hud is not None
    s = NS(clock=100., page='goodbye', cancelled=False, events=[], clicks=0,
           escapes=0, close_on=3, fault='', images={'goodbye': goodbye, 'hud': hud})
    gates.reset_stage_gates_for_tests()
    monkeypatch.setattr(dialog.time, 'monotonic', lambda: s.clock)
    monkeypatch.setattr(dialog.time, 'time', lambda: 1794100000. + s.clock)
    monkeypatch.setattr(dialog, 'op_cancelled', lambda: s.cancelled)
    def sleep(seconds):
        s.events.append(('wait', seconds))
        s.clock += seconds
    def grab():
        s.events.append(('grab', s.page, s.clock))
        return s.images[s.page]
    def click(x, y):
        s.events.append(('click', x, y, s.clock))
        s.clicks += 1
        if s.clicks == s.close_on:
            s.page = 'hud'
        return True
    def recover(*a, **kw):
        s.events.append(('escape', s.clock))
        s.escapes += 1
        if s.fault == 'pause':
            s.cancelled = True
        if s.fault == 'exception':
            raise RuntimeError('stack unavailable')
        return dict(ok=False, status='escape_unverified', esc_sent=True,
                    sent_at=time.time()-2., waiting=s.fault == 'unready')
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', recover)
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda frame:
        ui_logic.UiWhere('dialog', 0, 1, 0, 'fixture', stack_ready=s.fault!='unready',
                        stack_sampled_ts=time.time()))
    s.grab, s.click, s.sleep = grab, click, sleep
    token = dialog._GOODBYE_ESCAPE_AT.set(0.0)
    yield s
    dialog._GOODBYE_ESCAPE_AT.reset(token)
    gates.reset_stage_gates_for_tests()


def test_two_center_clicks_then_recovery_then_original_exit_loop(scene):
    s = scene
    assert dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=s.sleep,
                                            continuous=True)
    assert (s.clicks, s.escapes) == (3, 1)
    actions = [e[0] for e in s.events if e[0] in ('click', 'escape')]
    assert actions == ['click', 'click', 'escape', 'click']
    h, w = s.images['goodbye'].shape[:2]
    assert all(e[1:3] == (round(w*.5), round(h*.5)) for e in s.events if e[0]=='click')
    for i, event in enumerate(s.events):
        if event[0] != 'click':
            continue
        after = next(e for e in s.events[i+1:] if e[0]=='grab')
        assert after[2] - event[3] >= 1 - 1e-8
    assert gates._stage_counts['npc_goodbye_dismiss'] == 0


def test_first_center_click_closes_without_escape(scene):
    s = scene
    s.close_on = 1
    assert dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=s.sleep,
                                            continuous=True)
    assert (s.clicks, s.escapes) == (1, 0)


def test_escape_cancellation_keeps_emitted_center_budget(scene):
    s = scene
    s.close_on, s.fault = 99, 'pause'
    assert not dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=s.sleep,
                                                continuous=True)
    assert s.clicks == 2 and s.escapes == 1
    assert gates._stage_counts['npc_goodbye_dismiss'] == 2


def test_recovery_exception_never_refunds_or_sends_third_center(scene):
    s = scene
    s.close_on, s.fault = 99, 'exception'
    with pytest.raises(RuntimeError, match='stack unavailable'):
        dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=s.sleep,
                                         continuous=True)
    assert s.clicks == 2 and gates._stage_counts['npc_goodbye_dismiss'] == 2


def test_bounded_exit_keeps_two_click_budget_across_calls(scene):
    s = scene
    s.close_on = 99
    for _ in range(3):
        assert not dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=s.sleep)
    assert s.clicks == 2 and s.escapes == 0


@pytest.mark.parametrize('name', ['goodbye_stuck_20261108.png', 'dialog_block_20261108.png'])
def test_actual_current_failure_authorizes_only_current_client_center(monkeypatch, name):
    frame = cv2.imread(str(ROOT/'tests/fixtures/repair'/name))
    assert frame is not None
    monkeypatch.setattr('mm_assist.perception.ocr.ocr_roi',
                        lambda *a, **kw: pytest.fail('OCR cannot authorize goodbye'))
    assert dialog.locate_npc_goodbye_click(frame) == (960, 720)
    assert dialog._hud_menu_xy(frame) is None


def test_new_variant_preserves_source_crop_and_original_thresholds():
    spec = json.loads((ROOT/'knowledge/ui/repair/06_goodbye.json').read_text('utf-8'))
    assert spec['threshold'] == .85 and spec['min_margin'] == .05
    variant = spec['template_variants'][0]
    original, template = ROOT/variant['source'], ROOT/variant['template']
    assert hashlib.sha256(original.read_bytes()).hexdigest() == variant['source_sha256']
    assert hashlib.sha256(template.read_bytes()).hexdigest() == variant['template_sha256']
    source, crop = cv2.imread(str(original)), cv2.imread(str(template))
    x0, y0, x1, y1 = variant['crop_box']
    assert (source[y0:y1, x0:x1] == crop).all()
    assert spec['click_in_client'] == [.5, .5]


@pytest.mark.parametrize('event', ['pause', 'new_start', 'lease', 'f8'])
def test_wait_after_first_center_respects_stop_or_new_generation(scene, monkeypatch, event):
    from mm_assist.safety import input as native, lease
    s = scene
    s.close_on = 99
    stopped = [False]
    def sleep(seconds):
        s.clock += seconds
        stopped[0] = True
        if event == 'pause':
            s.cancelled = True
    if event == 'lease':
        monkeypatch.setattr(lease, 'in_test_mode', lambda: True)
        monkeypatch.setattr(lease, 'current_lease', lambda: NS(alive=lambda: not stopped[0]))
    if event == 'f8':
        def abort():
            if stopped[0]:
                raise native.AbortRequested('F8')
        monkeypatch.setattr(native, 'check_abort', abort)
        with pytest.raises(native.AbortRequested, match='F8'):
            dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=sleep, continuous=True)
    else:
        assert not dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=sleep,
            continuous=True, cancelled=lambda: event=='new_start' and stopped[0])
    assert s.clicks == 1 and s.escapes == 0
    assert gates._stage_counts['npc_goodbye_dismiss'] == 1


def test_cycle_uses_real_recovery_before_third_center(scene, monkeypatch):
    from mm_assist.life_execute import ui_recover
    from mm_assist.safety import input as native
    s = scene
    monkeypatch.setattr(ui_recover, 'recover_once', real_recover)
    def escape(**kw):
        s.events.append(('escape', s.clock))
        s.escapes += 1
    monkeypatch.setattr(native, 'press_escape', escape)
    monkeypatch.setattr(native, 'release_mouse_buttons', lambda: None)
    monkeypatch.setattr('mm_assist.life_plan.live_test.inspect_nav_watch_ready',
                        lambda **kw: ({}, ''))
    assert dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=s.sleep,
                                            continuous=True, win=NS(hwnd=7))
    assert [e[0] for e in s.events if e[0] in ('click', 'escape')] == [
        'click', 'click', 'escape', 'click']
    assert (s.clicks, s.escapes) == (3, 1)


def test_missing_game_context_never_sends_global_escape(scene, monkeypatch):
    from mm_assist.life_execute import ui_recover
    s = scene
    s.close_on = 99
    monkeypatch.setattr(ui_recover, 'recover_once', real_recover)
    monkeypatch.setattr('mm_assist.safety.input.press_escape',
                        lambda **kw: pytest.fail('no global Esc fallback'))
    assert not dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=s.sleep,
                                                continuous=True)
    assert s.clicks == 2 and gates._stage_counts['npc_goodbye_dismiss'] == 2


def test_exception_after_escape_dispatch_preserves_pending_input_on_resume(scene, monkeypatch):
    s = scene
    def broken(*a, **kw):
        kw['on_dispatch'](time.time())
        raise RuntimeError('post-Esc observation failed')
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', broken)
    with pytest.raises(RuntimeError, match='post-Esc observation failed'):
        dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=s.sleep,
                                         continuous=True)
    stamp = dialog._GOODBYE_ESCAPE_AT.get()
    assert stamp > 0 and s.clicks == 2
    seen = []
    def observe(*a, **kw):
        seen.append(kw['after_input_at'])
        s.cancelled = True
        return dict(ok=False, waiting=True, esc_sent=False, sent_at=stamp)
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', observe)
    assert not dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=s.sleep,
                                                continuous=True)
    assert seen == [stamp] and s.clicks == 2


@pytest.mark.parametrize('fault', ['unready', 'old_stack'])
def test_unverified_recovery_observes_without_third_center(scene, monkeypatch, fault):
    s = scene
    s.close_on = 99
    calls = []
    def recover(*a, **kw):
        calls.append(kw['after_input_at'])
        if len(calls) == 4:
            s.cancelled = True
        return dict(ok=False, status='page_not_ready', waiting=True,
                    esc_sent=not kw['after_input_at'], sent_at=10.)
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', recover)
    monkeypatch.setattr(ui_logic, 'resolve_navigation', lambda frame:
        ui_logic.UiWhere('dialog', 0, 1, 0, 'fixture', stack_ready=fault!='unready',
                        stack_sampled_ts=9. if fault=='old_stack' else time.time()))
    assert not dialog.wait_hud_after_npc_dismiss(s.click, s.grab, sleep_fn=s.sleep,
                                                continuous=True)
    assert s.clicks == 2 and calls == [0., 10., 10., 10.]
    assert gates._stage_counts['npc_goodbye_dismiss'] == 2


def test_goodbye_timeout_keeps_running_and_observes_same_escape(monkeypatch):
    controller.reset_controller_for_tests()
    clock = [100.]
    item = PlanItem('p1', 'gather', '练采集 黃豆')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **kw: [item])
    monkeypatch.setattr(bar_ui.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda win: True)
    monkeypatch.setattr('mm_assist.life_execute.ui_act.pump_aborted', lambda: False)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **kw: '')
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    calls = []
    def recover(*a, **kw):
        calls.append(kw['after_input_at'])
        if not kw['after_input_at']:
            kw['on_dispatch'](1001.)
        return dict(ok=False, status='page_not_ready', waiting=True,
                    esc_sent=not kw['after_input_at'], sent_at=1001.)
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', recover)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
                        lambda *a, **kw: ExecuteResult(False, 'npc_goodbye_unverified'))
    try:
        bar_ui.pump_life_once(None, None)
        clock[0] += 61
        bar_ui.pump_life_once(None, None)
        st = controller.get_controller()
        assert st.running and not st.paused and st.current == item
        assert calls == [0., 1001.] and bar_ui._RECOVERY_AFTER == 1001.
    finally:
        controller.reset_controller_for_tests()
