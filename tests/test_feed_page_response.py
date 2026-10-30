"""A dispatched feed must leave the recipe before its flow can complete.

External native UI facts are controlled; no game reads or inputs occur here.
Business deductions/queues remain a separate acceptance check.
"""
import copy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import medicine_process as process, ui_act, ui_logic
from mm_assist.memory.readers.facility_ui import select_facility_view
from test_facility_memory import sample


@pytest.mark.parametrize('case', ['closed', 'delayed', 'unchanged', 'changed_pair',
                                  'wrong_facility', 'remote', 'pause', 'exception',
                                  'black', 'missing_image'])
def test_feed_requires_fresh_uncovered_same_facility_without_reclick(monkeypatch, case):
    fact = select_facility_view(sample('metal_arrived')['render'], 0)
    state = {'clicked': False, 'clock': 0., 'reads': 0}
    clicks, issued = [], []
    frame = cv2.imread(str(ROOT/'tests/fixtures/process_stance/metal_standalone.png'))
    assert frame is not None

    def grab():
        if state['clicked']:
            if case == 'black':
                return np.zeros_like(frame)
            if case == 'missing_image':
                return None
        return frame

    def memory():
        if not state['clicked']:
            return copy.deepcopy(fact)
        state['reads'] += 1
        if case == 'exception':
            raise RuntimeError('offline post-feed read failed')
        if case in ('unchanged', 'pause') or case == 'delayed' and state['reads'] <= 4:
            # Active-facility reader rejects an extra recipe/unknown player.
            return None
        if case == 'changed_pair' and state['reads'] % 2:
            return None
        fresh = copy.deepcopy(fact)
        if case == 'wrong_facility':
            fresh['facility'] = 'wood'
        if case == 'remote':
            fresh['buttons']['go_equip']['present'] = True
        return fresh

    def click(*xy):
        clicks.append(xy)
        state['clicked'] = True
        return True

    def sleep(seconds):
        state['clock'] += seconds

    monkeypatch.setattr(process.time, 'monotonic', lambda: state['clock'])
    monkeypatch.setattr(process, '_recipe_detail_open', lambda *a: True)
    monkeypatch.setattr(ui_act, 'pump_aborted',
                        lambda: case == 'pause' and state['clicked'])
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda **k: False)
    monkeypatch.setattr('mm_assist.perception.process_sense.locate_process_button',
                        lambda *a, **k: {'ok': True, 'label': '加工',
                                        'at_facility': True, 'click_client': [1492, 1353]})
    monkeypatch.setattr(ui_logic, 'wait_ui_params_refresh', lambda **k: True)
    result = process._run_process(None, facility='metal', recipe='鐵錠', mode='feed',
        items={'鐵礦石': 187}, do_click=click, read_items=None, grab=grab,
        on_progress=None, root=None, sleep_fn=sleep, read_stp=lambda: 0,
        facilities={'metal': ()}, defer_completion=True,
        on_action_issued=lambda: issued.append(True), memory_reader=memory)

    assert clicks == [(1492, 1353)] and issued == [True]
    assert result.action_issued, 'post-dispatch failure must retain the major action fact'
    assert result.ok is (case in ('closed', 'delayed'))
    if case == 'pause':
        assert result.status == 'cancel'
    elif case not in ('closed', 'delayed'):
        assert result.status == 'feed_page_unverified'
    assert state['clock'] <= 2.1


def test_debug_feed_unverified_preserves_plan_and_major_action_on_resume(monkeypatch):
    from mm_assist.life_plan import controller, bar_ui
    from mm_assist.life_plan.types import LifeSnapshot, PlanItem, ExecuteResult
    controller.reset_controller_for_tests()
    plan = PlanItem('p1', 'feed', '投料 鐵錠', facility='metal', recipe='鐵錠')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [plan])
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda *a: True)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly', lambda **k: None)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda **k: 'offline.png')
    calls = []
    def execute(*a, **k):
        calls.append(True)
        k['on_action_issued']()
        return ExecuteResult(False, 'feed_page_unverified', action_issued=True)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    try:
        controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000, pause_on_failure=True)
        bar_ui.pump_life_once(None, SimpleNamespace(hwnd=0))
        st = controller.get_controller()
        assert st.paused and st.current == plan and st.queue == [plan]
        assert st.pending_action is not None and not st.pending_action.actions_complete
        controller.resume_life(pause_on_failure=True)
        bar_ui.pump_life_once(None, SimpleNamespace(hwnd=0))
        assert calls == [True], 'resume must not replay the dispatched feed'
        assert controller.get_controller().current == plan
    finally:
        controller.reset_controller_for_tests()
