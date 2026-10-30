"""A retained dungeon result must not expose the underlying HUD or facilities."""
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.memory.readers import ui_stack as stack
from mm_assist.life_execute import ui_logic, ui_nav
from mm_assist.life_plan import node_inspect


def pixels():
    return cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/life_faults_20261023/dungeon_result.png',
                                   dtype=np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize('names', [
    ['ClearResultUIInstance'], ['ClearResultUIState'], ['ClearRankUIState'],
    ['HudUIStepState', 'ClearResultUIState', 'ClearRankUIState',
     'TutorialTooltipUIState', 'ClearResultUIInstance'],
    ['AlteringFacilityPopupUIInstance', 'ClearResultUIInstance'],
    ['DialogueInteractionUIState', 'ClearResultUIInstance'],
])
def test_result_is_a_known_overlay_even_when_only_instance_is_available(names):
    assert stack.stack_node(names, ready=True) == 'overlay'
    assert stack.stack_kind(names, ready=True) == 'overlay'
    assert stack.left_main_hud(names) is True
    assert not stack.stack_hud_only(names)


def test_native_result_pixels_and_current_single_instance_cannot_publish_main_hud():
    view = SimpleNamespace(ui_stack_names=['ClearResultUIInstance'],
        ui_stack_ready=True, ui_stack_ts=time.time())
    result = node_inspect.compute_inspect(pixels(), view=view)
    assert result['node'] == 'overlay'


def test_direct_navigation_does_not_find_menu_or_click_under_result():
    reader = SimpleNamespace(read=lambda **kw: stack.UiStackSnap(ready=True,
        names=['ClearResultUIInstance'], sampled_ts=time.time()))
    image = pixels()
    def resolve(frame):
        return ui_logic.resolve_navigation(frame, reader=reader)
    where = resolve(image)
    assert where.node == 'overlay'
    clicked = []
    result = ui_nav.ensure_node('facility', facility='medicine', grab=lambda:image,
        resolve=resolve, click=lambda *xy: clicked.append(xy), sleep_fn=lambda _:None)
    assert not result.ok and not clicked


def test_result_gone_does_not_break_existing_hud_or_facility_classification():
    assert stack.stack_node(['TutorialTooltipUIState'], ready=True) == 'main_hud'
    assert stack.stack_node(['AlteringFacilityPopupUIInstance'], ready=True) == 'fine:facility'
    assert stack.stack_node(['ClearResultUIInstance'], ready=False) == ''
