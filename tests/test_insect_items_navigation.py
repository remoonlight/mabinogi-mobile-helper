"""Permanent page-only statusbar calibration: never starts insect acquisition."""
from types import SimpleNamespace
import pytest
from test_insect_obtain_flow import harness
from mm_assist.life_execute import insect_obtain as flow, ui_act
from mm_assist.life_plan.types import ExecuteResult
from mm_assist.life_plan import live_test
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    flow.reset_insect_visits_for_tests(); reset_stage_gates_for_tests()
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    yield
    flow.reset_insect_visits_for_tests(); reset_stage_gates_for_tests()


def test_items_calibration_is_registered_in_existing_statusbar_catalog():
    assert live_test.scenarios()['nav:insect_items'].detail == 'insect_items'


@pytest.mark.parametrize('initial', ['hud', 'obtain', 'card', 'recipe', 'bag'])
def test_entry_only_reaches_items_then_stops_before_any_recipe_or_primary_action(monkeypatch, initial):
    from mm_assist.perception import ocr
    from mm_assist.perception.ocr import OcrLine
    args, clicks = harness(monkeypatch, [initial, 'make', 'items'])
    original = args['resolve']
    def resolve(frame):
        result = original(frame)
        if result.node == 'unknown': result.node = 'craft_make'; result.stack_ready = True
        return result
    args['resolve'] = resolve
    monkeypatch.setattr(ocr, 'ocr_roi', lambda *a, **kw: [OcrLine('道具', .99, [[10,10],[20,10],[20,20],[10,20]])])
    targets = []
    def nav(target, **kw):
        targets.append(target); args['do_click'](1, 2)
        return ExecuteResult(True, 'page_matched')
    monkeypatch.setattr(ui_act, 'prepare_to', nav)
    flow._visits['滴答螢火蟲'] = 1  # Calibration never chooses the second-visit bag route.
    result = flow.execute_insect_obtain(mat='滴答螢火蟲', entry_only=True, **args)
    assert result.ok and result.status == 'page_matched' and not result.action_issued
    assert targets == ['craft_make'] and len(clicks) == 2
    assert flow._visits == {'滴答螢火蟲': 1} and not flow._source_after


def test_already_items_entry_only_is_zero_input_and_does_not_search_recipe(monkeypatch):
    from mm_assist.perception import ocr
    args, clicks = harness(monkeypatch, ['items'])
    args['resolve'] = lambda f: SimpleNamespace(node='craft_shell',stack_ready=True,stack_sampled_ts=1001)
    monkeypatch.setattr(ocr, 'ocr_roi', lambda *a, **kw: pytest.fail('recipe OCR must not run'))
    result = flow.execute_insect_obtain(mat='滴答螢火蟲', entry_only=True, **args)
    assert result.ok and not clicks and not result.action_issued


def test_items_calibration_does_not_clear_previously_issued_primary_evidence(monkeypatch):
    args,clicks=harness(monkeypatch,['hud'])
    flow._source_after['滴答螢火蟲']=999
    flow._source_attempts['滴答螢火蟲']=2
    monkeypatch.setattr(ui_act,'prepare_to',lambda *a,**kw:ExecuteResult(False,'cancel'))
    result=flow.execute_insect_obtain(mat='滴答螢火蟲',entry_only=True,**args)
    assert result.status=='cancel' and not clicks
    assert flow._source_after['滴答螢火蟲']==999 and flow._source_attempts['滴答螢火蟲']==2


@pytest.mark.parametrize('mutation', ['old', 'not_ready', 'wrong_node', 'wrong_pixels'])
def test_independent_final_items_proof_rejects_stale_unknown_and_wrong_pages(monkeypatch, mutation):
    from mm_assist.perception import insect_obtain_sense as sense
    monkeypatch.setattr(sense, 'phase', lambda *a, **kw: '' if mutation=='wrong_pixels' else 'items')
    where = SimpleNamespace(node='inventory' if mutation=='wrong_node' else 'craft_shell',
        stack_ready=mutation!='not_ready',stack_sampled_ts=99 if mutation=='old' else 101)
    assert live_test.insect_items_navigation_node(object(), where, captured_at=100) == ''


def test_independent_final_items_proof_uses_current_frame_and_stack(monkeypatch):
    from mm_assist.perception import insect_obtain_sense as sense
    monkeypatch.setattr(sense, 'phase', lambda *a, **kw: 'items')
    where = SimpleNamespace(node='craft_shell',stack_ready=True,stack_sampled_ts=101)
    assert live_test.insect_items_navigation_node(object(), where, captured_at=100) == 'insect_items'


@pytest.mark.parametrize('stp,fresh,matched,expected', [
    (1,True,True,'blocked'), (None,True,True,'blocked'), (0,False,True,'blocked'),
    (0,True,True,'passed'), (0,True,False,'failed')])
def test_real_statusbar_scene_gates_idle_and_independently_verifies_final_page(monkeypatch,tmp_path,stp,fresh,matched,expected):
    from contextlib import nullcontext
    from mm_assist.life_plan import action_finish
    monkeypatch.setattr(live_test,'ROOT',tmp_path)
    monkeypatch.setattr(live_test.Evidence,'capture',lambda *a,**kw:None)
    monkeypatch.setattr(live_test,'probe_focus',lambda h:{'kind':'game','is_game':True})
    monkeypatch.setattr(live_test,'bounded_stacks',lambda:[])
    monkeypatch.setattr('mm_assist.jobs.bar.require_state_watch_running',lambda:{})
    monkeypatch.setattr(live_test,'inspect_nav_watch_ready',lambda **kw:({},''))
    monkeypatch.setattr('mm_assist.safety.lease.bounded_test_lease',lambda **kw:nullcontext())
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.read_inspect_record',lambda:{'node':'main_hud','stp':0})
    monkeypatch.setattr(action_finish,'read_completion_stp',lambda:SimpleNamespace(stp=stp,fresh=fresh))
    rows=iter([{'node':'main_hud','stack_ready':True},
        {'node':'insect_items' if matched else 'craft_make','stack_ready':True}])
    monkeypatch.setattr(live_test,'wait_start_ui',lambda *a,**kw:next(rows))
    monkeypatch.setattr(live_test,'wait_hud_idle',lambda read,**kw: {} if read().get('node')=='unknown' else read())
    calls=[]
    def entry(*a,**kw):
        calls.append(kw)
        assert kw['entry_only'] is True
        return ExecuteResult(True,'page_matched')
    monkeypatch.setattr(flow,'execute_insect_obtain',entry)
    report=live_test.run_scenario(None,SimpleNamespace(hwnd=1),'nav:insect_items')
    assert report['status']==expected,report.get('reason')
    assert len(calls)==(1 if stp==0 and fresh else 0)


def test_entry_only_pause_never_dispatches(monkeypatch):
    args,clicks=harness(monkeypatch,['make','items'])
    monkeypatch.setattr(ui_act,'pump_aborted',lambda:True)
    result=flow.execute_insect_obtain(mat='滴答螢火蟲',entry_only=True,**args)
    assert result.status=='cancel' and not clicks
