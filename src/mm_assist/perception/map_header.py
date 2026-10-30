"""Read-only map breadcrumb and registered region header evidence."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from mm_assist.config import ROOT
from mm_assist.perception.list_ocr import box_center_norm, normalize_text
from mm_assist.platform.match import match_template

_ULARA = re.compile(r'(?:歐|欧)拉(?:(?:大|太)(?:陸|陆|睦|隆))?')


def breadcrumb_has_secondary_place(text: str) -> bool:
    text = normalize_text(text or '')
    for sep in ('／', '/', '·', '|'):
        if sep in text:
            right = text.split(sep, 1)[-1].strip()
            if right and not right.startswith(('欧', '歐')):
                return True
    match = _ULARA.search(text)
    if match is not None:
        tail = text[match.end():].strip('/／·| ')
        return tail not in {'', '地圖', '地图'}
    return False


def title_has_secondary_place(lines, win) -> bool:
    titles = []
    for line in lines:
        center = box_center_norm(line, win.width, win.height)
        if center and center[0] < .34 and center[1] < .10:
            titles.append(str(line.text or '').strip())
    if any(breadcrumb_has_secondary_place(text) for text in titles):
        return True
    keys = ('歐拉', '欧拉', '大睦', '大隆', '太睦')
    if not any(any(key in text for key in keys) for text in titles):
        return False
    return any(text.strip('/／·| ') not in {'', '地圖', '地图', '大陸', '大陆'}
               and not any(key in text for key in keys) for text in titles)


@lru_cache(maxsize=2)
def _spec(name):
    return json.loads((ROOT / 'knowledge/templates/map' / f'{name}.json').read_text(encoding='utf-8'))


def _hit(frame, name):
    if frame is None or getattr(frame, 'size', 0) == 0:
        return None
    spec = _spec(name)
    for candidate in (spec, *spec.get('template_variants', ())):
        if (candidate is not spec and candidate.get('source_size')
                and list(frame.shape[1::-1]) != candidate['source_size']):
            continue
        hit = match_template(frame, ROOT / candidate['template'], roi=tuple(spec['roi']),
                             threshold=spec['threshold'], min_margin=spec['min_margin'])
        if hit is not None:
            return hit
    return None


def _map_close_visible(frame) -> bool:
    if frame is None or getattr(frame, 'size', 0) == 0:
        return False
    return match_template(frame, ROOT / 'knowledge/templates/map/close_x.png',
                          roi=(.9, 0, .1, .12), threshold=.80) is not None


def cloud_region_visible(frame) -> bool:
    """Registered complete cloud-region name plus map chrome; never overview."""
    return _hit(frame, 'cloud_region_title') is not None and _map_close_visible(frame)


def locate_ulara_breadcrumb(frame):
    """Return the independently visible left label, without an OCR coordinate."""
    hit = _hit(frame, 'ulara_breadcrumb_label')
    if hit is None or not _map_close_visible(frame):
        return None
    return int(hit.cx), int(hit.cy)


def coarse_map_visible(frame) -> bool:
    """Native retained full breadcrumb plus map close button proves only a map."""
    if frame is None or getattr(frame, 'size', 0) == 0:
        return False
    try:
        spec = _spec('coarse_map_title')
        if list(frame.shape[1::-1]) != spec['source_size']:
            return False
        return _hit(frame, 'coarse_map_title') is not None and _map_close_visible(frame)
    except (OSError, KeyError, TypeError, ValueError):
        # An absent optional proof cannot disrupt normal HUD recognition.
        return False
