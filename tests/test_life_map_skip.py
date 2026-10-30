"""watch21852: failed Laert map left 70 subsequent plans timing out."""
import hashlib
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import ui_logic, ui_recover
from mm_assist.life_plan import bar_ui, controller, node_inspect
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem


def frame(name):
    return cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/process_stance' / name,
                                    dtype=np.uint8), cv2.IMREAD_COLOR)


def stack_reader(names=(), ready=True):
    return SimpleNamespace(read=lambda **kw: SimpleNamespace(names=names, ready=ready,
        empty_confirmed=not names, craft_category=None, sampled_ts=time.time()))


@pytest.mark.parametrize('name', [
    'continent_region_stuck_20261012T200513Z.png',
    'input_page_timeout_20261012T200720Z.png',
    'laert_repair_stuck_20261012.png',
])
@pytest.mark.parametrize('names', [(), ('TutorialTooltipUIState',)])
def test_failed_map_and_previous_view_are_maps_without_ocr(monkeypatch, name, names):
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi',
                        lambda *a, **kw: pytest.fail('map identity requires pixels'))
    where = ui_logic.resolve_navigation(frame(name), reader=stack_reader(names))
    assert where.node == 'map' and where.stack_ready


def test_failed_map_reaches_real_pump_recovery_then_continues_same_item(monkeypatch):
    controller.reset_controller_for_tests()
    bar_ui.reset_pump_settle()
    item = PlanItem('p1', 'feed', '投料 布料+', facility='cloth', recipe='布料+')
    following = PlanItem('p2', 'gather', '下一项')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **kw: [item, following])
    monkeypatch.setattr('mm_assist.memory.readers.ui_stack.shared_ui_stack_reader', stack_reader)
    images = [frame('input_page_timeout_20261012T200720Z.png'),
              frame('map_open_stale_point.png')]
    monkeypatch.setattr('mm_assist.platform.capture.grab_client', lambda win: images[0])
    monkeypatch.setattr('mm_assist.life_plan.action_finish.read_completion_stp',
        lambda: SimpleNamespace(stp=0, fresh=True, timestamp=time.time()))
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    calls = []
    def escape():
        calls.append('esc')
        images.pop(0)
    real_recover = ui_recover.recover_once
    monkeypatch.setattr(ui_recover, 'recover_once',
        lambda *a, **kw: real_recover(*a, **kw, press_esc=escape, sleep_fn=lambda _: None))
    def execute(actual, **kw):
        calls.append(actual.id)
        return (ExecuteResult(False, 'nav_verify_timeout') if len(images) == 2
                else ExecuteResult(True, 'done'))
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    try:
        bar_ui.pump_life_once(None, SimpleNamespace(hwnd=7))
        assert calls == ['p1', 'esc', 'p1']
        assert controller.get_controller().current == following
        assert bar_ui._RECOVERY_COUNT == 1
    finally:
        controller.reset_controller_for_tests()
        bar_ui.reset_pump_settle()


@pytest.mark.parametrize('case', ['no_title', 'no_close', 'wrong_size', 'unready',
                                 'blocked', 'extra_layer', 'hud', 'facility'])
def test_current_title_variant_does_not_relax_page_safety(case):
    image = frame('input_page_timeout_20261012T200720Z.png')
    names, ready = ('TutorialTooltipUIState',), True
    if case == 'no_title': image[:140, :450] = 0
    if case == 'no_close': image[:170, 1730:] = 0
    if case == 'wrong_size': image = image[:, :-1]
    if case == 'unready': ready = False
    if case == 'blocked': names += ('ClickShieldUIState',)
    if case == 'extra_layer': names += ('EquipmentRepairUIInstance',)
    if case == 'hud': image = frame('repair_hud_failure_20261011.png')
    if case == 'facility': image = frame('medicine_list_20261011.png')
    where = ui_logic.resolve_navigation(image, reader=stack_reader(names, ready))
    assert not where.node.startswith('map')


def test_registered_title_variant_is_exact_native_crop_with_original_thresholds():
    spec = json.loads((ROOT / 'knowledge/templates/map/coarse_map_title.json').read_text(encoding='utf-8'))
    variant = spec['template_variants'][0]
    source = ROOT / variant['source_frame']
    template = ROOT / variant['template']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == variant['source_sha256']
    assert hashlib.sha256(template.read_bytes()).hexdigest() == variant['template_sha256']
    image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), 1)
    x, y, w, h = variant['crop_client_px']
    assert np.array_equal(image[y:y+h, x:x+w], cv2.imdecode(np.fromfile(template, dtype=np.uint8), 1))
    assert spec['threshold'] == .97 and spec['min_margin'] == .02
    assert variant['source_size'] == [1920, 1440]
