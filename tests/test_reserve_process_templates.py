"""Existing unmodified game frames must independently locate new medicine rows."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import pytest

from mm_assist.config import ROOT
from mm_assist.perception import process_sense as sense

NAMES = ('增幅蘑菇孢子', '增幅蘑菇汁液', '咻咻蘑菇孢子', '咻咻蘑菇汁液')


def frame(name):
    return cv2.imdecode(np.fromfile(ROOT / 'tests/fixtures/process_stance' / name, dtype=np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize('name', NAMES)
def test_real_medicine_source_and_negative_frames_without_ocr(monkeypatch, name):
    monkeypatch.setattr(sense, '_ocr_lines', lambda *a, **kw: [])
    original = frame('medicine_standalone.png')
    hit = sense.locate_process_recipe(original, facility='medicine', name=name)
    assert hit['ok'] and hit['src'] == 'template'
    path = sense.recipe_template_path('medicine', name)
    row = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    source = ROOT / row['source_frame']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_sha256']
    x, y, w, h = row['crop_client_px']
    crop = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(crop, original[y:y+h, x:x+w])
    assert x <= hit['click_client'][0] < x+w and y <= hit['click_client'][1] < y+h
    masked = original.copy()
    masked[y:y+h, x:x+w] = 0
    assert not sense.locate_process_recipe(masked, facility='medicine', name=name)['ok']
    for other in ('wood_standalone.png', 'metal_standalone.png', 'cloth_standalone.png', 'facility_no_detail.png'):
        assert not sense.locate_process_recipe(frame(other), facility='medicine', name=name)['ok']


@pytest.mark.parametrize('value', ['nan', 'invalid', 1.01, None])
def test_invalid_recipe_threshold_cannot_relax_pixel_gate(monkeypatch, value):
    monkeypatch.setattr(sense, '_ocr_lines', lambda *a, **kw: [])
    row = {**sense._find_recipe('medicine', NAMES[0]), 'threshold': value}
    monkeypatch.setattr(sense, '_find_recipe', lambda *a, **kw: row)
    result = sense.locate_process_recipe(frame('medicine_standalone.png'), facility='medicine', name=NAMES[0])
    assert not result['ok'] and result['error'] == 'invalid_template_threshold'
