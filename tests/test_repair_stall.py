"""2026-10-12 p51: map with HUD-only stack, then 95 minutes of unknown wait."""
import hashlib
import json
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT, AppConfig
from mm_assist.life_execute import ui_logic
from mm_assist.life_plan import bar_ui, controller, node_inspect
from mm_assist.life_plan.types import ExecuteResult, LifeSnapshot, PlanItem
from mm_assist.nav import map_atlas
from mm_assist.platform.window import WindowInfo


def image(name='laert_repair_stuck_20261012.png'):
    return cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/process_stance' / name,
                                    dtype=np.uint8), cv2.IMREAD_COLOR)


def reader(names=('TutorialTooltipUIState',), ready=True):
    return SimpleNamespace(read=lambda **kw: SimpleNamespace(names=names, ready=ready,
        empty_confirmed=False, craft_category=None, sampled_ts=time.time()))


def test_actual_failed_map_and_hud_stack_authorize_one_fixed_ascent(monkeypatch):
    frame = image()
    monkeypatch.setattr('mm_assist.memory.readers.ui_stack.shared_ui_stack_reader', reader)
    monkeypatch.setattr(node_inspect, '_ocr_lines_roi', lambda *a, **kw: pytest.fail('no OCR page proof'))
    where = ui_logic.resolve_navigation(frame)
    assert where.node == 'map' and where.stack_ready
    clicks = []
    monkeypatch.setattr(map_atlas, '_click', lambda win, x, y, **kw: clicks.append((x, y)))
    win = WindowInfo(0, 'offline', 0, 0, 1920, 1440)
    assert map_atlas._click_ulara_breadcrumb([], win, frame=frame)
    assert clicks == [(116, 58)]


def test_unchanged_current_region_never_authorizes_continent_drag(monkeypatch):
    from mm_assist.perception.ocr import OcrLine
    win = WindowInfo(0, 'offline', 0, 0, 1920, 1440)
    frame = image()
    lines = [OcrLine('歐拉大陸 / 萊爾特丘陵', .99, [[85,38],[450,38],[450,80],[85,80]])]
    monkeypatch.setattr('mm_assist.memory.readers.ui_stack.shared_ui_stack_reader', reader)
    monkeypatch.setattr(map_atlas, '_grab_map_frame', lambda *a, **kw: (win, frame))
    monkeypatch.setattr(map_atlas, 'snap', lambda *a, **kw: (win, frame, lines, None))
    monkeypatch.setattr(map_atlas, '_wait_action_effect', lambda *a, **kw: (win, frame, False, 0))
    monkeypatch.setattr('mm_assist.nav.map_localize.drag_until_near',
                        lambda *a, **kw: pytest.fail('region pixels do not authorize atlas dragging'))
    clicks = []
    monkeypatch.setattr(map_atlas, '_click', lambda win, x, y, **kw: clicks.append((x, y)))
    result = map_atlas.continent_pixel_enter_dubarton(AppConfig(), win, lines=lines, img=frame)
    assert not result['ok'] and result['stage'] == 'continent_region_stuck'
    assert clicks == [(116, 58)]


@pytest.mark.parametrize('case', ['no_title', 'no_close', 'wrong_size', 'hud', 'facility', 'unready', 'blocked', 'extra_layer'])
def test_map_pixels_never_promote_wrong_or_unready_page(monkeypatch, case):
    frame = image()
    names, ready = ('TutorialTooltipUIState',), True
    if case == 'no_title': frame[:140, :450] = 0
    if case == 'no_close': frame[:170, 1730:] = 0
    if case == 'wrong_size': frame = frame[:, :-1]
    if case == 'hud': frame = image('repair_hud_failure_20261011.png')
    if case == 'facility': frame = image('medicine_list_20261011.png')
    if case == 'unready': ready = False
    if case == 'blocked': names += ('ClickShieldUIState',)
    if case == 'extra_layer': names += ('EquipmentRepairUIInstance',)
    where = ui_logic.resolve_navigation(frame, reader=reader(names, ready))
    assert not where.node.startswith('map')


@pytest.fixture
def recovery(monkeypatch):
    controller.reset_controller_for_tests()
    clock = [10.0]
    monkeypatch.setattr(bar_ui.time, 'monotonic', lambda: clock[0])
    queue = [PlanItem('p1', 'repair', '一键修理'), PlanItem('p2', 'repair', '下一项')]
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **kw: list(queue))
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda win: True)
    monkeypatch.setattr('mm_assist.life_execute.ui_act.pump_aborted', lambda: False)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **kw: '')
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    events = []
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',
        lambda *a, **kw: events.append('execute') or ExecuteResult(False, 'continent_region_stuck', '原始升层失败'))
    state = dict(status='page_not_ready', sent=False)
    def recover(*a, **kw):
        events.append('observe')
        sent_at = kw['after_input_at']
        if state['sent'] and not sent_at:
            kw['on_dispatch'](1001.0)
            sent_at = 1001.0
            events.append('esc')
        return dict(ok=False, waiting=True, status=state['status'],
                    esc_sent=state['sent'] and not kw['after_input_at'], sent_at=sent_at)
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', recover)
    yield clock, queue, events, state
    controller.reset_controller_for_tests()


def test_unknown_repair_wait_expires_without_input_or_executor_replay(recovery):
    clock, queue, events, _ = recovery
    bar_ui.pump_life_once(None, None)
    assert controller.get_controller().current == queue[0]
    clock[0] += 59
    bar_ui.pump_life_once(None, None)
    assert controller.get_controller().current == queue[0]
    clock[0] += 1
    bar_ui.pump_life_once(None, None)
    assert controller.get_controller().current == queue[1]
    assert events.count('execute') == 1 and 'esc' not in events
    assert bar_ui._RECOVERY_COUNT == 0


def test_known_busy_does_not_expire_or_consume_escape(recovery):
    clock, queue, events, state = recovery
    state['status'] = 'stp_busy'
    for delta in [0, 61, 600]:
        clock[0] += delta
        bar_ui.pump_life_once(None, None)
        assert controller.get_controller().current == queue[0]
    assert events.count('execute') == 1 and 'esc' not in events


def test_repair_recovery_uses_same_isolated_capture_as_map_navigation(recovery, monkeypatch):
    _, queue, _, _ = recovery
    frame = image()
    cfg = AppConfig()
    win = WindowInfo(7, 'offline', 0, 0, 1920, 1440)
    seen = []
    monkeypatch.setattr(map_atlas, '_grab_navigation_client',
                        lambda actual_cfg, actual_win: seen.append((actual_cfg, actual_win)) or frame)
    def recover(*a, **kw):
        assert kw['grab']() is frame
        return dict(ok=False, waiting=True, status='page_not_ready', esc_sent=False, sent_at=0)
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', recover)
    result = bar_ui._execute_with_recovery(lambda *a, **kw: ExecuteResult(False, 'continent_region_stuck'),
                                          queue[0], cfg=cfg, win=win)
    assert result.status == 'recovery_wait' and seen == [(cfg, win)]


def test_unverified_sent_escape_keeps_running_without_refunding_input(recovery):
    clock, queue, events, state = recovery
    state['sent'] = True
    bar_ui.pump_life_once(None, None)
    clock[0] += 60
    bar_ui.pump_life_once(None, None)
    st = controller.get_controller()
    assert st.running and not st.paused and st.current == queue[0]
    assert bar_ui._RECOVERY_COUNT == 1 and bar_ui._RECOVERY_AFTER == 1001
    bar_ui.pump_life_once(None, None)
    assert events.count('esc') == 1 and events.count('execute') == 1


def test_timeout_late_result_cannot_pause_a_new_generation(recovery, monkeypatch):
    clock, queue, _, state = recovery
    state['sent'] = True
    bar_ui.pump_life_once(None, None)
    def log(**kw):
        if kw.get('tag') == 'recovery_page_timeout':
            controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly', log)
    clock[0] += 60
    bar_ui.pump_life_once(None, None)
    st = controller.get_controller()
    assert st.life_started_at == 2000 and st.running and not st.paused
    assert st.current == queue[0]


def test_post_dispatch_observation_exception_is_bounded_without_new_escape(recovery, monkeypatch):
    clock, queue, events, state = recovery
    state['sent'] = True
    bar_ui.pump_life_once(None, None)
    def broken(*a, **kw):
        raise RuntimeError('frame read failed')
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', broken)
    clock[0] += 60
    bar_ui.pump_life_once(None, None)
    assert controller.get_controller().running and not controller.get_controller().paused
    assert controller.get_controller().current == queue[0]
    assert events.count('esc') == 1 and bar_ui._RECOVERY_COUNT == 1


def test_missing_coarse_map_asset_leaves_normal_hud_recognition_available(monkeypatch):
    from mm_assist.perception import map_header
    def missing(name):
        raise FileNotFoundError(name)
    monkeypatch.setattr(map_header, '_spec', missing)
    assert not map_header.coarse_map_visible(image())


@pytest.mark.parametrize('event', ['pause', 'new_start'])
def test_wait_clock_resets_on_pause_or_new_generation(recovery, event):
    clock, queue, events, _ = recovery
    bar_ui.pump_life_once(None, None)
    clock[0] += 600
    if event == 'pause':
        controller.pause_life()
        controller.resume_life()
    else:
        controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000)
    bar_ui.pump_life_once(None, None)
    assert controller.get_controller().current == queue[0]
    assert events.count('execute') == (1 if event == 'pause' else 2)


def test_current_map_heading_has_exact_registered_source():
    path = ROOT / 'knowledge/templates/map/coarse_map_title.json'
    row = json.loads(path.read_text(encoding='utf-8'))
    original = ROOT / row['source_frame']
    assert hashlib.sha256(original.read_bytes()).hexdigest() == row['source_sha256']
    template = ROOT / row['template']
    assert hashlib.sha256(template.read_bytes()).hexdigest() == row['template_sha256']
    x, y, w, h = row['crop_client_px']
    assert np.array_equal(cv2.imdecode(np.fromfile(template, dtype=np.uint8), 1), image()[y:y+h, x:x+w])
