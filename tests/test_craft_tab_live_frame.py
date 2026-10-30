"""The Oct 5 processing hub must not be mistaken for unselected 製作."""
import time
from pathlib import Path

import cv2
import pytest

from mm_assist.life_plan import node_inspect as ni
from mm_assist.life_execute import ui_nav
from mm_assist.life_execute.ui_logic import UiWhere
from mm_assist.safety import autofarm_gates as gates


FRAME = Path(__file__).parent / 'fixtures/process_stance/craft_process_gate_20261005.png'


def test_original_processing_hub_uses_cyan_selected_tab(monkeypatch):
    monkeypatch.setattr(ni, '_craft_title_blob', lambda _: pytest.fail('OCR is not page proof'))
    frame = cv2.imread(str(FRAME))
    assert frame is not None
    assert ni._selected_craft_tab(frame) == 'craft_hub'
    assert ni._craft_shell_node(frame, category=2) == 'craft_hub'
    assert ni.refine_inspect_node(frame, sn='fine:craft_hub', category=2) == 'craft_hub'


def test_hub_observation_verifies_prior_tab_without_third_click(monkeypatch):
    gates.reset_stage_gates_for_tests()
    ui_nav.reset_edge_memo_for_tests()
    stage = 'craft_make_tab_craft_hub'
    try:
        assert gates.allow_stage_action(stage) and gates.allow_stage_action(stage)
        assert not gates.allow_stage_action(stage)
        now = time.time()
        ui_nav._PENDING_EDGES[stage] = (stage, 'craft_hub', now - 1)
        ui_nav.note_edge_clicked(stage, 'craft_hub')
        frame = cv2.imread(str(FRAME))
        node = ni._craft_shell_node(frame)
        observation = UiWhere(node, 1, 0, 0, 'stack+tpl', stack_ready=True,
            stack_names=('CraftingUIInstance',), stack_sampled_ts=now)
        monkeypatch.setattr('mm_assist.life_execute.ui_act.stack_has_common_popup', lambda **_: False)
        monkeypatch.setattr('mm_assist.life_execute.ui_act.stack_has_facility_result', lambda **_: False)
        result = ui_nav.ensure_node('craft_hub', grab=lambda: frame, resolve=lambda _: observation,
            click=lambda *_: pytest.fail('already processing hub'), sleep_fn=lambda _: None)
        assert result.ok
        assert gates._stage_counts[stage] == 0 and stage not in ui_nav._PENDING_EDGES
    finally:
        gates.reset_stage_gates_for_tests()
        ui_nav.reset_edge_memo_for_tests()
