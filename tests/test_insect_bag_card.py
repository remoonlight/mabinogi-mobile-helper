"""A later bag card has a higher title than the potion material card."""
import time
from types import SimpleNamespace

import cv2
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import insect_obtain as flow, ui_act, ui_logic
from mm_assist.life_plan import node_inspect
from mm_assist.memory.readers.ui_stack import UiStackSnap
from mm_assist.perception import insect_obtain_sense as sense
from mm_assist.safety import autofarm_gates as gates

NAMES = ['TutorialTooltipUIState', 'EquipmentInventoryUIInstance',
         'SizeAdjustmentSender', 'DetailCardUIInstance']


def pixels():
    return cv2.imread(str(ROOT/'tests/fixtures/insect_obtain/firefly_bag_card_20261021.png'))


def where(frame, *, names=NAMES, ready=True, age=0, sampled_at=None):
    reader = SimpleNamespace(read=lambda **_:UiStackSnap(ready=ready,
        names=list(names), sampled_ts=(time.time() if sampled_at is None else sampled_at)-age))
    return ui_logic.resolve_navigation(frame, reader=reader)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    flow.reset_insect_visits_for_tests()
    gates.reset_stage_gates_for_tests()
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda:False)
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **kw:[])
    yield
    flow.reset_insect_visits_for_tests()
    gates.reset_stage_gates_for_tests()


def test_native_bag_card_has_independent_full_title_and_method(monkeypatch):
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi',
        lambda *a, **kw:pytest.fail('registered bag title and method must not need OCR'))
    assert sense.phase(pixels(), '滴答螢火蟲') == 'card'
    assert sense.phase(pixels(), '獨角仙') == ''
    assert sense.locate(pixels(), 'obtain_button') == (744, 867)
    assert where(pixels()).node == 'item_info:滴答螢火蟲'


def test_bag_card_disappearing_during_inspection_cannot_verify_old_pixels():
    samples = iter([UiStackSnap(ready=True, names=list(NAMES), sampled_ts=time.time()),
        UiStackSnap(ready=True, names=NAMES[:-1], sampled_ts=time.time())])
    result = ui_logic.resolve_navigation(pixels(),
        reader=SimpleNamespace(read=lambda **_:next(samples)))
    assert result.node == 'unknown' and result.source == 'direct_transition'


def test_real_bag_card_verifies_prior_icon_then_only_attempts_method():
    stage = '虫料入口:滴答螢火蟲:滴答螢火蟲'
    flow._visits['滴答螢火蟲'] = 1
    flow._entry_after[stage] = ('card', time.time()-1)
    flow._entry_attempts[stage] = 1
    assert gates.allow_stage_action(stage)
    clicks = []
    result = flow.execute_insect_obtain(mat='滴答螢火蟲', grab=pixels, resolve=where,
        do_click=lambda *xy:clicks.append(xy) or False, sleep_fn=lambda _:None)
    assert clicks == [(744, 867)]
    assert stage not in flow._entry_after and gates._stage_counts[stage] == 0
    assert not result.ok and not result.action_issued and not flow._source_attempts
    assert flow._visits == {'滴答螢火蟲': 1}


@pytest.mark.parametrize('case', ['hidden_title', 'hidden_method', 'wrong_material',
                                  'unready', 'stale', 'covered', 'wrong_context'])
def test_bag_card_without_independent_current_proof_never_clicks(case):
    frame, names = pixels(), list(NAMES)
    if case == 'hidden_title': frame[334:386,675:910] = 0
    if case == 'hidden_method': frame[844:892,675:820] = 0
    if case == 'covered': names.append('UnknownPopup')
    if case == 'wrong_context': names.remove('EquipmentInventoryUIInstance')
    ticks = [time.time()]
    clicks = []
    result = flow.execute_insect_obtain(mat='獨角仙' if case=='wrong_material' else '滴答螢火蟲',
        grab=lambda:frame, resolve=lambda image:where(image, names=names,
            ready=case!='unready', age=10 if case=='stale' else 0, sampled_at=ticks[0]+.001),
        do_click=lambda *xy:clicks.append(xy) or False,
        sleep_fn=lambda n:ticks.__setitem__(0,ticks[0]+n), now_fn=lambda:ticks[0])
    assert not result.ok and not result.action_issued and not clicks


def test_bag_title_is_lossless_registered_crop_without_changing_old_card_region():
    import hashlib
    from PIL import Image
    row = sense._catalog(ROOT)['rows']['firefly_bag_card_title']
    old = sense._catalog(ROOT)['rows']['firefly_card_title']
    assert old['roi'] == [.28,.24,.24,.2]
    assert old['threshold'] == row['threshold'] == .94
    assert old['min_margin'] == row['min_margin'] == .04
    source, target = ROOT/row['source_frame'], ROOT/'knowledge/templates/insect_obtain'/row['file']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_sha256']
    assert hashlib.sha256(target.read_bytes()).hexdigest() == row['template_sha256']
    x,y,w,h = row['crop_client_px']
    assert Image.open(source).crop((x,y,x+w,y+h)).tobytes() == Image.open(target).tobytes()


@pytest.mark.parametrize('has_title', [True, False])
def test_method_header_still_requires_the_registered_target_title(monkeypatch, has_title):
    # Contract only: this is not a fabricated live method-page fixture.
    anchors = {'obtain_header', 'firefly_bag_card_title'} if has_title else {'obtain_header'}
    monkeypatch.setattr(sense, 'locate', lambda frame, key, **_: (1,2) if key in anchors else None)
    assert sense.phase(object(), '滴答螢火蟲') == ('obtain' if has_title else '')
    assert sense.phase(object(), '獨角仙') == ''
