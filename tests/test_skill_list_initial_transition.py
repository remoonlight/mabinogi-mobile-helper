"""p14 original pixels: first paired stack changes while a skill list opens."""
import hashlib
import json
from types import SimpleNamespace

import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import skill_list, skill_nearby, ui_act, ui_logic
from mm_assist.memory.readers import ui_stack
from mm_assist.perception import item_sense
from mm_assist.safety import autofarm_gates as gates


FIXTURE = ROOT / 'tests/fixtures/life_faults_20261025/herb_initial_transition.png'


def test_fixture_is_the_unmodified_failure_frame():
    meta = json.loads(FIXTURE.with_suffix('.json').read_text('utf-8'))
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == meta['source_sha256']
    assert meta['major_action_issued'] is False and meta['list_drags_issued'] == 0


@pytest.mark.parametrize('initial', ['transition', 'unready'])
def test_real_executor_reobserves_initial_stack_then_verifies_rows_before_drag(monkeypatch, initial):
    frame = cv2.imread(str(FIXTURE))
    meta = json.loads(FIXTURE.with_suffix('.json').read_text('utf-8'))
    initial_hit = item_sense.locate_skill_item(frame, skill_id='herb', name='增幅蘑菇')
    assert not initial_hit['ok'] and initial_hit['scroll_plan']['direction'] == 'down'
    # Reuse recognition of identical fixture pixels, but capture distinct frames
    # and run the real paired-stack resolver on every validation.
    monkeypatch.setattr(item_sense, 'locate_skill_item', lambda *a, **k: dict(initial_hit))
    clock, reads, captures, observations, drags = [1000.], [], [], [], []
    monkeypatch.setattr(skill_list.time, 'time', lambda: clock[0])
    monkeypatch.setattr(skill_list, 'operation_monotonic', lambda: clock[0])
    monkeypatch.setattr(skill_list.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0]+dt))
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr(ui_logic, 'read_inspect_record', lambda **k:
                        {'node': 'gather_skill:herb', 'zh': '生活技能'})
    monkeypatch.setattr(ui_act, 'prepare_to', lambda *a, **k: pytest.fail('do not replay navigation'))

    def read(**kwargs):
        assert kwargs.get('force') is True
        clock[0] += .01
        n = len(reads)
        names = meta['observed_before_stack'] if initial == 'transition' and n == 0 else meta['observed_after_stack']
        snap = ui_stack.UiStackSnap(ready=not(initial == 'unready' and n == 0),
                                   names=names, sampled_ts=clock[0])
        reads.append(snap)
        return snap

    reader = SimpleNamespace(read=read)
    def resolve(image):
        where = ui_logic.resolve_navigation(image, reader=reader)
        observations.append(where)
        return where
    def grab():
        clock[0] += .05
        fresh = frame.copy()
        captures.append(fresh)
        return fresh
    def drag(*args, **kwargs):
        assert kwargs['end_hold_s'] == .4
        assert len(captures) >= 3
        assert all(w.node == 'gather_skill:herb' and w.stack_ready for w in observations[-2:])
        drags.append(args)
        return False  # No actual dispatch; no target or main-action click.
    gates.reset_stage_gates_for_tests()
    result = skill_nearby.execute_skill_nearby(skill_id='herb', item_name='增幅蘑菇', output='增幅蘑菇',
        grab=grab, do_click=lambda *a: pytest.fail('target remains below viewport'),
        do_drag=drag, resolve=resolve, defer_completion=True)
    assert observations[0].source == ('direct_transition' if initial == 'transition' else 'direct_pending')
    assert result.status == 'drag_rejected' and len(drags) == 1
    assert not result.action_issued and not skill_list._pending
    assert gates._stage_counts.get('详情物下翻:herb:增幅蘑菇', 0) == 0


@pytest.mark.parametrize('interrupt', ['pause', 'new_start', 'late_pause', 'exception'])
def test_initial_refresh_preserves_unverified_gesture_when_interrupted(monkeypatch, interrupt):
    from mm_assist.life_plan import controller
    from mm_assist.life_plan.types import LifeSnapshot, PlanItem
    controller.reset_controller_for_tests()
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [PlanItem('p1', 'repair', 'offline')])
    owner = controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    stage = '详情物下翻:herb:增幅蘑菇'
    pending = (999., (8, 8), {'藥草': 7})
    skill_list._pending['herb', '增幅蘑菇'] = pending
    gates._stage_counts[stage] = 2
    gates._stage_need_verify[stage] = True
    calls = []
    frame = cv2.imread(str(FIXTURE))
    def cancelled():
        state = controller.get_controller()
        return not state.running or state.life_started_at != owner.life_started_at
    def validate(*args):
        calls.append(1)
        if len(calls) == 1:
            if interrupt == 'pause': controller.pause_life()
            if interrupt == 'new_start':
                controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000)
                # Explicit start initializes new gates. The late old caller
                # must not erase reservations subsequently owned by that run.
                gates._stage_counts[stage] = 2
                gates._stage_need_verify[stage] = True
            return False
        if interrupt == 'exception': raise RuntimeError('read failed')
        controller.pause_life()
        return True
    args = dict(skill_id='herb', name='增幅蘑菇', captured_at=1000.,
        grab=lambda: frame.copy(), validate=validate, cancelled=cancelled,
        drag=lambda *a: pytest.fail('do not dispatch'), sleep=lambda _: None)
    try:
        if interrupt == 'exception':
            with pytest.raises(RuntimeError, match='read failed'):
                skill_list.seek_item(frame, {'ok': False}, **args)
        else:
            _, _, error = skill_list.seek_item(frame, {'ok': False}, **args)
            assert error.status == 'cancel'
        assert skill_list._pending['herb', '增幅蘑菇'] == pending
        assert gates._stage_counts[stage] == 2 and gates._stage_need_verify[stage]
    finally:
        controller.reset_controller_for_tests()


@pytest.mark.parametrize('bad', ['wrong_skill', 'unready', 'old_stack', 'covered'])
def test_initial_refresh_never_uses_a_later_invalid_page(monkeypatch, bad):
    frame = cv2.imread(str(FIXTURE))
    clock, captures = [1000.], []
    monkeypatch.setattr(skill_list.time, 'time', lambda: clock[0])
    monkeypatch.setattr(skill_list.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0]+dt))
    monkeypatch.setattr(skill_list, 'operation_monotonic', lambda: clock[0])
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr(ui_logic, 'read_inspect_record', lambda **k:
                        {'node': 'gather_skill:herb', 'zh': '生活技能'})
    monkeypatch.setattr(item_sense, 'locate_skill_item', lambda *a, **k:
                        {'ok': True, 'click_client': [1200, 900]})
    def grab():
        clock[0] += .1
        captures.append(clock[0])
        return frame.copy()
    def resolve(image):
        return SimpleNamespace(node='gather_skill:mine' if bad == 'wrong_skill' else
            'unknown' if bad == 'covered' else 'gather_skill:herb', stack_ready=bad != 'unready',
            stack_sampled_ts=clock[0]-10 if bad == 'old_stack' else clock[0]+.01)
    result = skill_nearby.execute_skill_nearby(skill_id='herb', item_name='增幅蘑菇', output='增幅蘑菇',
        grab=grab, do_click=lambda *a: pytest.fail('no invalid-page click'),
        do_drag=lambda *a, **k: pytest.fail('no invalid-page drag'), resolve=resolve,
        defer_completion=True)
    assert result.status == 'item_list_unknown' and not result.action_issued
    assert len(captures) > 1 and clock[0] <= 1006.6
    assert not skill_list._pending
