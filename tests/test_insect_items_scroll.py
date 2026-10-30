"""Retained first viewport drives one proven list gesture, never a blind retry."""
from types import SimpleNamespace
import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import insect_obtain as flow, ui_act
from mm_assist.perception import insect_obtain_sense as sense, ocr
from mm_assist.perception.ocr import OcrLine
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests


@pytest.mark.parametrize('mat', ['滴答螢火蟲', '獨角仙'])
def test_phase_reuses_template_results_only_within_one_observation(monkeypatch, mat):
    from collections import Counter
    image = cv2.imread(str(ROOT/'tests/fixtures/insect_obtain/items_failed_20261014.png'))
    assert image is not None
    original = sense.locate
    calls = []
    def locate(frame, key, **kw):
        calls.append(key)
        return original(frame, key, **kw)
    monkeypatch.setattr(sense, 'locate', locate)
    assert sense.phase(image, mat) == 'items'
    assert max(Counter(calls).values()) == 1, 'one phase observation must not match a template twice'
    calls.clear()
    image[:] = 0  # Same buffer, new content: no match can survive the next call.
    assert sense.phase(image, mat) == ''
    assert calls and max(Counter(calls).values()) == 1


@pytest.fixture
def scenario(monkeypatch):
    flow.reset_insect_visits_for_tests()
    reset_stage_gates_for_tests()
    first = cv2.imread(str(ROOT/'tests/fixtures/insect_obtain/items_failed_20261014.png'))
    assert first is not None and sense.phase(first, '滴答螢火蟲') == 'items'
    original_locate = sense.locate
    ticks, state = [1000.0], {'view': 0}
    images = [first, first.copy()]
    drags, clicks = [], []
    def locate(image, key, **kw):
        # A second actual scrolled viewport is acquired after loading live.
        # Inject only its moved landmark, preserving the original recognizer.
        if key == 'items_high_bandage' and state['view']:
            return (1047, 760)
        return original_locate(image, key, **kw)
    monkeypatch.setattr(sense, 'locate', locate)
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: state.get('cancel', False))
    monkeypatch.setattr(ocr, 'ocr_roi', lambda *a, **kw: [] if not state['view'] else
        [OcrLine('連擊強化秘藥', .99, [[1300,900],[1480,900],[1480,930],[1300,930]])])
    def resolve(image):
        return SimpleNamespace(node=state.get('node', 'craft_make'),
            stack_ready=not state.get('unready', False),
            stack_sampled_ts=999 if state.get('stale') else ticks[0]+.001)
    def drag(x0,y0,x1,y1,**kw):
        drags.append((x0,y0,x1,y1,kw))
        state['view'] = 1
        return True
    def sleep(seconds):
        ticks[0] += seconds
    args = dict(grab=lambda: images[state['view']], do_drag=drag,
        do_click=lambda *xy: clicks.append(xy) or False, resolve=resolve,
        sleep_fn=sleep, now_fn=lambda: ticks[0])
    yield args, state, drags, clicks
    flow.reset_insect_visits_for_tests()
    reset_stage_gates_for_tests()


def test_missing_potion_first_pages_down_then_relocates_on_fresh_view(scenario):
    args,state,drags,clicks=scenario
    result=flow.execute_insect_obtain(mat='滴答螢火蟲', **args)
    assert len(drags)==1 and clicks==[(1390,915)]
    x0,y0,x1,y1,kw=drags[0]
    assert x0==x1 and y0>y1 and kw['end_hold_s']==.4
    assert not result.ok and not result.action_issued  # Potion click deliberately refused.


@pytest.mark.parametrize('problem',['wrong_page','stale','pause','unchanged','exception','refused'])
def test_bad_after_view_never_clicks_or_repeats_scroll(scenario, problem):
    args,state,drags,clicks=scenario
    def drag(*xy,**kw):
        drags.append(xy)
        if problem=='wrong_page':state['node']='craft_hub'
        if problem=='stale':state['stale']=True
        if problem=='pause':state['cancel']=True
        if problem=='exception':raise RuntimeError('after mouse down')
        return problem!='refused'
    args['do_drag']=drag
    result=flow.execute_insect_obtain(mat='滴答螢火蟲', **args)
    assert not result.ok and not result.action_issued and len(drags)==1 and not clicks
    if problem not in {'refused','pause'}:
        # Clearing general click gates/pausing cannot erase this issued gesture.
        reset_stage_gates_for_tests()
        state.pop('stale',None);state.pop('node',None)
        flow.execute_insect_obtain(mat='滴答螢火蟲', **args)
        assert len(drags)==1 and not clicks


def test_entry_only_page_calibration_cannot_scroll_or_choose_recipe(scenario):
    args,state,drags,clicks=scenario
    result=flow.execute_insect_obtain(mat='滴答螢火蟲', entry_only=True, **args)
    assert result.ok and result.status=='page_matched' and not drags and not clicks


def test_no_scroll_anchor_never_blind_drags(scenario, monkeypatch):
    args,state,drags,clicks=scenario
    original=sense.locate
    monkeypatch.setattr(sense,'locate',lambda image,key,**kw:
        None if key=='items_recovery_header' else original(image,key,**kw))
    result=flow.execute_insect_obtain(mat='滴答螢火蟲', **args)
    assert not result.ok and not drags and not clicks


def test_target_moving_after_paging_is_not_clicked(scenario, monkeypatch):
    args,state,drags,clicks=scenario
    readings=[]
    def labels(*a,**kw):
        if not state['view']:return []
        readings.append(1)
        x=1300 if len(readings)==1 else 1320
        return [OcrLine('連擊強化秘藥',.99,[[x,900],[x+180,900],[x+180,930],[x,930]])]
    monkeypatch.setattr(ocr,'ocr_roi',labels)
    result=flow.execute_insect_obtain(mat='滴答螢火蟲',**args)
    assert result.status=='insect_items_moving' and len(drags)==1 and not clicks


@pytest.mark.parametrize('change',[{'x':True},{'end_y':.9},{'max_drags':4},{'list_roi':[.1,.1,2,.8]}])
def test_invalid_drag_configuration_cannot_supply_coordinates(monkeypatch,tmp_path,change):
    import json
    original=ROOT/'knowledge/ui/config/process_navigation.json'
    data=json.loads(original.read_text('utf-8'))
    data['insect_items_list_down'].update(change)
    config=tmp_path/'knowledge/ui/config/process_navigation.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(data),'utf-8')
    monkeypatch.setattr(sense,'locate',lambda *a,**kw:(10,20))
    image=cv2.imread(str(ROOT/'tests/fixtures/insect_obtain/items_failed_20261014.png'))
    with pytest.raises(ValueError):sense.items_down_points(image,root=tmp_path)


def test_three_viewports_each_require_new_landmark_progress_before_next_drag(scenario,monkeypatch):
    args,state,drags,clicks=scenario
    original=sense.locate
    def locate(image,key,**kw):
        if key=='items_high_bandage':return (1047,1258-489*state['view'])
        if key=='items_excellent_bandage':
            return (840,1840-489*state['view']) if state['view'] else None
        return original(image,key,**kw)
    monkeypatch.setattr(sense,'locate',locate)
    monkeypatch.setattr(ocr,'ocr_roi',lambda *a,**kw: [] if state['view']<3 else
        [OcrLine('連擊強化秘藥',.99,[[1300,900],[1480,900],[1480,930],[1300,930]])])
    first=args['grab']()
    args['grab']=lambda:first.copy()
    def drag(*xy,**kw):
        assert '滴答螢火蟲' not in flow._items_scroll_after
        state['view']+=1;drags.append(xy);return True
    args['do_drag']=drag
    result=flow.execute_insect_obtain(mat='滴答螢火蟲',**args)
    assert len(drags)==3 and clicks==[(1390,915)] and not result.action_issued
