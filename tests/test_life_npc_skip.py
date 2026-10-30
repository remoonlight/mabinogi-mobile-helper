"""11:35 p72 NPC exit must not poison subsequent plans' shared menu gate."""
import hashlib
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.life_execute import repair_dialog, ui_act, ui_logic, ui_nav
from mm_assist.safety import autofarm_gates as gates


def frame(name):
    image = cv2.imread(str(ROOT / 'tests/fixtures/repair' / name))
    assert image is not None
    return image


def reader(names=(), ready=True):
    return SimpleNamespace(read=lambda **_: SimpleNamespace(names=names, ready=ready,
        empty_confirmed=not names, craft_category=None, sampled_ts=time.time()))


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    monkeypatch.setattr('mm_assist.perception.ocr.ocr_roi',
                        lambda *a, **k: pytest.fail('NPC exit requires independent pixels'))
    yield
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()


@pytest.mark.parametrize('name', ['nerys_end_20261017.png', 'nerys_menu_timeout_20261017.png'])
def test_recorded_end_label_is_recognized_without_lowering_threshold(name):
    image = frame(name)
    hit = repair_dialog.find_end_dialog_btn(image)
    assert hit is not None and hit.score >= .88 and hit.margin >= .05
    assert 1070 <= hit.cx < 1220 and 1310 <= hit.cy < 1380
    assert repair_dialog.npc_end_dialog_visible(image)
    image[1300:1380, 1050:1220] = 0
    assert repair_dialog.find_end_dialog_btn(image) is None
    assert not repair_dialog.npc_end_dialog_visible(image)


@pytest.mark.parametrize('name', ['nerys_end_20261017.png', 'nerys_menu_timeout_20261017.png',
                                'nerys_after_end_dialog.png'])
@pytest.mark.parametrize('names', [(), ('HudUIStepState',), ('TutorialTooltipUIState',)])
def test_npc_pixels_override_hud_shell_and_reject_menu_coordinate(name, names):
    image = frame(name)
    where = ui_logic.resolve_navigation(image, reader=reader(names))
    assert where.node == 'dialog' and where.stack_ready and where.dlg == 1
    assert ui_nav._hud_xy(image, 'menu', root=None) is None


@pytest.mark.parametrize('names,ready', [((), False), (('ToyDimLayerUIInstance',), True),
    (('EquipmentInventoryUIInstance',), True)])
def test_npc_pixels_cannot_override_unready_or_another_page(names, ready):
    where = ui_logic.resolve_navigation(frame('nerys_end_20261017.png'), reader=reader(names, ready))
    assert where.node != 'dialog'


def test_actual_hud_still_has_menu_and_remains_hud():
    image = frame('nerys_hud_after_skip_20261017.png')
    assert not repair_dialog.npc_end_dialog_visible(image)
    assert not repair_dialog.npc_goodbye_visible(image)
    assert ui_logic.resolve_navigation(image, reader=reader()).node == 'main_hud'
    assert ui_nav._hud_xy(image, 'menu', root=None) is not None


@pytest.mark.parametrize('accepted', [False, True])
def test_recorded_exit_failure_never_clicks_menu_or_resets_unverified_gate(monkeypatch, accepted):
    image = frame('nerys_end_20261017.png')
    monkeypatch.setattr(ui_logic, 'resolve_navigation',
                        lambda *_: pytest.fail('must finish NPC exit first'))
    clicks = []
    for _ in range(3):
        result = ui_act.prepare_to('system_menu', grab=lambda: image,
            click=lambda *xy: clicks.append(xy) or accepted, sleep_fn=lambda _: None)
        assert result.status == 'npc_end_dialog_unverified'
    assert len(clicks) == (2 if accepted else 3)
    assert all(1070 <= x < 1220 and 1310 <= y < 1380 for x, y in clicks)
    assert 'main_hud_menu_system_menu' not in gates._stage_counts


def test_recorded_frame_manifest_preserves_original_pixels():
    manifest = json.loads((ROOT/'tests/fixtures/repair/life_skip_20261017.json').read_text(encoding='utf-8'))
    for spec in manifest['frames']:
        assert hashlib.sha256((ROOT/spec['fixture']).read_bytes()).hexdigest() == spec['sha256']
        image = cv2.imread(str(ROOT/spec['fixture']))
        assert [image.shape[1], image.shape[0]] == spec['client_size']


def test_native_end_variant_is_exact_registered_crop_and_size_limited():
    spec = json.loads((ROOT/'knowledge/ui/repair/05b_end_dialog_label.json').read_text(encoding='utf-8'))
    assert spec['threshold'] == .88 and spec['min_margin'] == .05
    variant = spec['template_variants'][0]
    source = ROOT / variant['source']
    template = ROOT / variant['template']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == variant['source_sha256']
    assert hashlib.sha256(template.read_bytes()).hexdigest() == variant['template_sha256']
    image = cv2.imread(str(source))
    x0, y0, x1, y1 = variant['crop_box']
    assert np.array_equal(image[y0:y1, x0:x1], cv2.imread(str(template)))
    assert repair_dialog.find_end_dialog_btn(image[:, :-1]) is None


def test_real_exit_and_navigation_leave_next_plans_menu_budget_available(monkeypatch):
    images = {'end': frame('nerys_end_20261017.png'),
              'goodbye': frame('nerys_after_end_dialog.png'),
              'hud': frame('nerys_hud_after_skip_20261017.png'),
              'menu': cv2.imread(str(ROOT/'tests/fixtures/ui_benchmark/system_menu_real.png'))}
    page = ['end']
    clicks = []
    shared = SimpleNamespace(read=lambda **_: reader(
        ('SideNavigationUIInstance',) if page[0] == 'menu' else ()).read())
    monkeypatch.setattr('mm_assist.memory.readers.ui_stack.shared_ui_stack_reader', lambda: shared)
    monkeypatch.setattr(ui_act, 'inspect_is_hud_idle', lambda **_: False)
    monkeypatch.setattr(ui_act, 'inspect_is_true_hud', lambda **_: False)
    monkeypatch.setattr(ui_act, 'pump_aborted', lambda: False)
    monkeypatch.setattr(ui_act, 'stack_has_common_popup', lambda **_: False)
    monkeypatch.setattr(ui_act, 'stack_has_facility_result', lambda **_: False)

    def click(x, y):
        clicks.append((page[0], x, y))
        page[0] = {'end': 'goodbye', 'goodbye': 'hud', 'hud': 'menu'}[page[0]]
        return True

    for _ in range(3):
        page[0] = 'end'
        result = ui_act.prepare_to('system_menu', grab=lambda: images[page[0]],
            click=click, read_stp=lambda: 0, sleep_fn=lambda _: None)
        assert result.ok and page[0] == 'menu'
        assert gates._stage_counts['main_hud_menu_system_menu'] == 0
        assert gates._stage_counts['npc_end_dialog'] == 0
        assert not ui_nav._PENDING_EDGES
    assert [p for p, _, _ in clicks] == ['end', 'goodbye', 'hud'] * 3


def test_recorded_dialog_does_not_clear_preexisting_menu_input(monkeypatch):
    stage = 'main_hud_menu_system_menu'
    for _ in range(2):
        assert gates.stage_click(stage, lambda *_: True, 1625, 102)
        ui_nav.note_edge_clicked(stage, 'system_menu', stage=stage)
    where = ui_logic.resolve_navigation(frame('nerys_end_20261017.png'), reader=reader())
    ui_nav._verify_late_edges(where, frame_started_at=time.time())
    assert gates._stage_counts[stage] == 2 and stage in ui_nav._PENDING_EDGES
    assert gates.stage_click(stage, lambda *_: pytest.fail('no third menu click'), 1625, 102) is None
