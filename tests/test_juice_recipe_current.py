"""Full caption evidence for the actual visible juice row; no game IO."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.perception import process_sense as sense


def original():
    return cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/life_faults_20261024/medicine_juice_list.png',
                                  dtype=np.uint8), cv2.IMREAD_COLOR)


def fact(frame, facility='medicine'):
    return {'facility': facility, 'identity': [1,2,3,4,5], 'window': 5,
            'view_root': [3,4], 'screen_size': [frame.shape[1], frame.shape[0]]}


def test_visible_full_juice_caption_locates_without_ocr_spelling_guess(monkeypatch):
    frame = original()
    monkeypatch.setattr(sense, '_gray_from_ocr', lambda *_: False)
    monkeypatch.setattr(sense, '_ocr_lines', lambda *_: pytest.fail('recipe text must match retained pixels'))
    result = sense.locate_process_recipe(frame, facility='medicine', name='淨淨蘑菇汁液',
                                         memory_view=fact(frame))
    assert result['ok'] and result['src'] == 'template'
    x, y = result['click_client']
    assert 1194 <= x <= 1360 and 998 <= y <= 1045
    assert result['peak'] >= .94


@pytest.mark.parametrize('case', ['missing_caption', 'missing_suffix', 'wrong_facility', 'invalid_memory'])
def test_full_juice_caption_does_not_relax_other_page_or_partial_name_guards(monkeypatch, case):
    frame = original()
    memory = fact(frame)
    if case == 'missing_caption':
        frame[998:1045,1194:1360] = 25
    elif case == 'missing_suffix':
        frame[998:1045,1304:1360] = 25
    elif case == 'wrong_facility':
        memory = fact(frame, 'metal')
    else:
        memory = {'error': 'memory_unavailable'}
    monkeypatch.setattr(sense, '_ocr_lines', lambda *_: [])
    result = sense.locate_process_recipe(frame, facility='medicine', name='淨淨蘑菇汁液', memory_view=memory)
    assert not result['ok'] and not result.get('click_client')
