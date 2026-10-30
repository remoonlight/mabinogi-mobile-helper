"""Retained sources and input sequencing for per-material insect obtain flow."""
from types import SimpleNamespace
import hashlib
import json
from PIL import Image

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import insect_obtain as flow
from mm_assist.life_plan.types import ExecuteResult
from mm_assist.perception import insect_obtain_sense as sense
from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests


def image(key):
    return cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/insect_obtain' / (key + '.png'), dtype=np.uint8), 1)


@pytest.fixture(autouse=True)
def clean():
    flow.reset_insect_visits_for_tests()
    reset_stage_gates_for_tests()
    yield
    flow.reset_insect_visits_for_tests()
    reset_stage_gates_for_tests()


@pytest.mark.parametrize('mat,prefix,other', [
    ('獨角仙', 'beetle', 'firefly'), ('滴答螢火蟲', 'firefly', 'beetle'),
])
def test_original_recipe_and_obtain_pages_distinguish_materials(mat, prefix, other):
    assert sense.phase(image(prefix + '_recipe'), mat) == 'recipe'
    assert sense.phase(image(prefix + '_obtain'), mat) == 'obtain'
    assert sense.phase(image(other + '_recipe'), mat) == ''
    assert sense.phase(image(other + '_obtain'), mat) == ''
    xy = sense.locate(image(prefix + '_obtain'), 'insect_entry', roi=(.52,.18,.40,.28))
    assert xy is not None
    # Beetle also offers woodcutting in row one. Never take the first row.
    assert (420 < xy[1] < 500) if mat == '獨角仙' else (300 < xy[1] < 390)
    recipe = image(prefix + '_recipe')
    xy = sense.material_xy(recipe, mat, inventory=False)
    assert xy is not None and 1450 < xy[0] < 1590
    assert sense.material_xy(image(other + '_recipe'), mat, inventory=False) is None


def harness(monkeypatch, phases, *, stale=False):
    ticks = [1000.0]
    index = [0]
    clicks = []
    frames = [object() for _ in phases]
    monkeypatch.setattr(sense, 'phase', lambda f, mat, **kw: phases[frames.index(f)])
    monkeypatch.setattr(sense, 'locate', lambda *a, **kw: (10, 20))
    # Object frames model the legacy card sequence. The all-key locator stub
    # cannot establish the newly registered bag layout; retained-frame tests
    # cover that layout and its required parent stack independently.
    monkeypatch.setattr(sense, 'bag_card_visible', lambda *a, **kw: False)
    monkeypatch.setattr(sense, 'material_xy', lambda *a, **kw: (30, 40))
    def resolve(f):
        p = phases[frames.index(f)]
        return SimpleNamespace(node={'recipe':'item_info','card':'item_info','obtain':'obtain_method',
            'bag':'inventory','make':'craft_make','hud':'main_hud'}.get(p,'unknown'),
            stack_ready=p != 'unknown', stack_sampled_ts=999 if stale and index[0] > 0 else ticks[0]+.001)
    def click(x, y):
        clicks.append((x,y))
        index[0] = min(index[0]+1, len(frames)-1)
        return True
    def sleep(n):
        ticks[0] += n
    return dict(grab=lambda: frames[index[0]], do_click=click, resolve=resolve,
        sleep_fn=sleep, now_fn=lambda:ticks[0]), clicks


def test_first_recipe_material_then_obtain_source_and_deferred_completion(monkeypatch):
    from mm_assist.life_execute import ui_act
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    args, clicks = harness(monkeypatch, ['recipe','card','obtain','hud'])
    issued = []
    result = flow.execute_insect_obtain(mat='獨角仙', defer_completion=True,
        on_action_issued=lambda:issued.append(True), **args)
    assert result.status == 'action_pending' and result.action_issued
    assert len(clicks) == 3 and issued == [True]
    assert flow._visits == {'獨角仙': 1}


def test_second_visit_uses_inventory_icon_and_still_recognizes_card(monkeypatch):
    from mm_assist.life_execute import ui_act
    from mm_assist.perception import inventory_sense
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr(inventory_sense, 'props_tab_selected', lambda *a, **kw: True)
    flow._visits['滴答螢火蟲'] = 1
    args, clicks = harness(monkeypatch, ['hud','bag','card','obtain','hud'])
    targets = []
    def nav(target, **kw):
        targets.append(target)
        args['do_click'](1,2)
        return ExecuteResult(True,'page_matched')
    monkeypatch.setattr(ui_act, 'prepare_to', nav)
    result = flow.execute_insect_obtain(mat='滴答螢火蟲', defer_completion=True, **args)
    assert result.status == 'action_pending' and targets == ['inventory']
    assert len(clicks) == 4 and flow._visits['滴答螢火蟲'] == 2


def test_old_stack_after_material_click_does_not_click_next_or_reset_stage(monkeypatch):
    from mm_assist.life_execute import ui_act
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    args, clicks = harness(monkeypatch, ['recipe','card','obtain'], stale=True)
    result = flow.execute_insect_obtain(mat='獨角仙', defer_completion=True, **args)
    assert result.status == 'insect_refresh_unknown' and len(clicks) == 1
    assert not flow._visits


def test_source_callback_exception_retains_issued_input_and_second_visit(monkeypatch):
    from mm_assist.life_execute import ui_act
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    args, clicks = harness(monkeypatch, ['obtain','hud'])
    def callback():
        raise RuntimeError('pause after dispatch')
    result = flow.execute_insect_obtain(mat='獨角仙', defer_completion=True,
        on_action_issued=callback, **args)
    assert result.status == 'error' and result.action_issued and len(clicks) == 1
    assert flow._visits == {'獨角仙': 1} and '獨角仙' in flow._source_after


def test_first_visit_navigates_make_and_visible_recipe_without_crafting(monkeypatch):
    from mm_assist.life_execute import ui_act
    from mm_assist.perception import ocr
    from mm_assist.perception.ocr import OcrLine
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    args, clicks = harness(monkeypatch, ['hud','make','items','recipe','card','obtain','hud'])
    # Items is independently confirmed by source pixels; OCR only locates labels.
    base_resolve = args['resolve']
    def resolve(frame):
        result = base_resolve(frame)
        if result.node == 'unknown':
            result.node = 'craft_make'
            result.stack_ready = True
        return result
    args['resolve'] = resolve
    monkeypatch.setattr(ocr, 'ocr_roi', lambda *a, **kw: [
        OcrLine('道具', .99, [[10,10],[20,10],[20,20],[10,20]]),
        OcrLine('強擊強化秘藥', .99, [[30,30],[40,30],[40,40],[30,40]])])
    targets = []
    def nav(target, **kw):
        targets.append(target)
        args['do_click'](1,2)
        return ExecuteResult(True,'page_matched')
    monkeypatch.setattr(ui_act, 'prepare_to', nav)
    result = flow.execute_insect_obtain(mat='獨角仙', defer_completion=True, **args)
    assert result.status == 'action_pending' and targets == ['craft_make']
    assert len(clicks) == 6  # Navigation, 道具, recipe, material, method, insect source.


def test_pause_before_entry_does_not_input_or_change_visit(monkeypatch):
    from mm_assist.life_execute import ui_act
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: True)
    args, clicks = harness(monkeypatch, ['obtain','hud'])
    assert flow.execute_insect_obtain(mat='獨角仙', **args).status == 'cancel'
    assert not clicks and not flow._visits


def test_unknown_or_wrong_material_never_clicks_obtain_row(monkeypatch):
    from mm_assist.life_execute import ui_act
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    args, clicks = harness(monkeypatch, ['unknown'])
    args['resolve'] = lambda f: SimpleNamespace(node='obtain_method', stack_ready=True, stack_sampled_ts=1001)
    result = flow.execute_insect_obtain(mat='獨角仙', **args)
    assert result.status == 'insect_material_unknown' and not clicks


@pytest.mark.parametrize('message,issued', [('after mouse down', True), ('click_focus_lost', False)])
def test_source_input_exception_only_refunds_proven_before_down_refusal(monkeypatch, message, issued):
    from mm_assist.life_execute import ui_act
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    args, _ = harness(monkeypatch, ['obtain'])
    def click(x,y):
        raise RuntimeError(message)
    args['do_click'] = click
    result = flow.execute_insect_obtain(mat='獨角仙', defer_completion=True, **args)
    assert result.status == 'error' and result.action_issued == issued
    assert bool(flow._visits) == issued


def test_unverified_source_inputs_keep_budget_across_calls_and_material_plan_changes(monkeypatch):
    from mm_assist.life_execute import ui_act
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    args, clicks = harness(monkeypatch, ['obtain'])
    for _ in range(2):
        assert flow.execute_insect_obtain(mat='獨角仙', defer_completion=True, **args).action_issued
    result = flow.execute_insect_obtain(mat='獨角仙', defer_completion=True, **args)
    assert result.status == 'need_verify' and len(clicks) == 2
    assert flow._visits == {'獨角仙': 2}


def test_registered_pixels_keep_original_window_and_exact_client_crop():
    folder = ROOT / 'knowledge/templates/insect_obtain'
    catalog = json.loads((folder / 'catalog.json').read_text(encoding='utf-8'))
    for row in catalog['rows'].values():
        source = ROOT / row['source_frame']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_sha256']
        x,y,w,h = row['game_frame_crop']
        with Image.open(source) as original:
            client = original.crop((x,y,x+w,y+h))
            cx,cy,cw,ch = row['crop_client_px']
            target = folder / row['file']
            with Image.open(target) as actual:
                assert np.array_equal(np.array(actual), np.array(client.crop((cx,cy,cx+cw,cy+ch))))
            assert hashlib.sha256(target.read_bytes()).hexdigest() == row['template_sha256']


def test_hiding_insect_entry_keeps_other_methods_unusable():
    original = image('beetle_obtain')
    config = json.loads((ROOT / 'knowledge/templates/insect_obtain/catalog.json').read_text(encoding='utf-8'))
    x,y,w,h = config['rows']['insect_entry']['crop_client_px']
    original[y:y+h,x:x+w] = 0
    assert sense.phase(original,'獨角仙') == 'obtain'
    assert sense.locate(original,'insect_entry') is None


def test_pause_new_start_reset_of_shared_gates_cannot_bypass_source_budget(monkeypatch):
    from mm_assist.life_execute import ui_act
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    args, clicks = harness(monkeypatch, ['obtain'])
    for _ in range(2):
        reset_stage_gates_for_tests()  # Same reset the controller performs on pause/start.
        assert flow.execute_insect_obtain(mat='獨角仙', defer_completion=True, **args).action_issued
    reset_stage_gates_for_tests()
    assert flow.execute_insect_obtain(mat='獨角仙', defer_completion=True, **args).status == 'need_verify'
    assert len(clicks) == 2


def test_reset_of_shared_gates_cannot_bypass_unverified_material_entry(monkeypatch):
    from mm_assist.life_execute import ui_act
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    args, clicks = harness(monkeypatch, ['recipe'])
    for _ in range(2):
        reset_stage_gates_for_tests()
        assert flow.execute_insect_obtain(mat='獨角仙', **args).status == 'insect_refresh_unknown'
    reset_stage_gates_for_tests()
    assert flow.execute_insect_obtain(mat='獨角仙', **args).status == 'need_verify'
    assert len(clicks) == 2
