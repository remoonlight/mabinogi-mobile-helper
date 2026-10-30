"""Only retained-frame pixels identify insect cards and the obtain-method row."""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from mm_assist.config import ROOT
from mm_assist.platform.match import match_template

PREFIX = {'獨角仙': 'beetle', '滴答螢火蟲': 'firefly'}


def items_stack_allowed(names) -> bool:
    """已知製作壳状态／教程不遮住道具子窗；其它覆盖层仍拒绝。"""
    return set(names or ()) <= {'CraftingHubPopupUIInstance', 'CraftingFacilityPopupUIInstance',
        'CraftingHubUIState', 'HudUIStepState', 'TutorialTooltipUIState',
        'SizeAdjustmentSender', 'UILayerHideState'}


def bag_card_stack_allowed(names) -> bool:
    names = set(names or ())
    return ({'EquipmentInventoryUIInstance', 'DetailCardUIInstance'} <= names
            and names <= {'EquipmentInventoryUIInstance', 'DetailCardUIInstance',
                          'TutorialTooltipUIState', 'SizeAdjustmentSender',
                          'HudUIStepState', 'EquipmentInventoryUIState',
                          'DetailCardUIState', 'UILayerHideState'})


@lru_cache(maxsize=4)
def _catalog(root: Path) -> dict:
    return json.loads((root / 'knowledge/templates/insect_obtain/catalog.json').read_text(encoding='utf-8'))


def region(key: str, *, root: Path | None = None):
    return tuple(_catalog(Path(root or ROOT))['regions'][key])


def locate(frame: Any, key: str, *, roi=None, root: Path | None = None):
    if frame is None:
        return None
    base = Path(root or ROOT)
    try:
        cfg = _catalog(base)
        sizes = [cfg['client_dimensions']] + [
            row['dimensions'] for row in cfg.get('client_variants', [])]
        if list(frame.shape[1::-1]) not in sizes:
            return None
        row = cfg['rows'][key]
        for candidate in [row,*row.get('variants',[])]:
            hit = match_template(frame, base / 'knowledge/templates/insect_obtain' / candidate['file'],
                roi=tuple(roi or row['roi']), threshold=float(row['threshold']), min_margin=float(row['min_margin']))
            if hit is not None:
                return int(hit.cx),int(hit.cy)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return None


def phase(frame: Any, mat: str, *, root: Path | None = None) -> str:
    prefix = PREFIX.get(mat)
    if prefix is None:
        return ''
    # Local to this observation: absent titles are reused too, and a new call
    # always rematches its pixels, even when a capture buffer is reused.
    hits = {}
    def located(key):
        if key not in hits:
            hits[key] = locate(frame, key, root=root)
        return hits[key]
    if located('obtain_header'):
        return ('obtain' if (located(prefix + '_card_title')
                            or located(prefix + '_bag_card_title')) else '')
    if located(prefix + '_recipe_title'):
        return 'recipe'
    card_titles = {p for p in PREFIX.values()
                   if (located(p + '_card_title')
                       or located(p + '_bag_card_title'))}
    if card_titles:
        # A material card covers the list. Missing its method control or seeing
        # another material must not fall through to the underlying items title.
        return ('card' if card_titles == {prefix}
                and located('obtain_button') else '')
    if located('items_title'):
        # A wrong potion popup is still covering the items list.
        if any(located(p + '_recipe_title')
               for p in PREFIX.values()):
            return ''
        return 'items'
    return ''


def bag_card_visible(frame: Any, mat: str, *, root: Path | None = None) -> bool:
    prefix = PREFIX.get(mat)
    return bool(prefix and locate(frame, prefix + '_bag_card_title', root=root)
                and locate(frame, 'obtain_button', root=root))


def material_xy(frame: Any, mat: str, *, inventory: bool, root: Path | None = None):
    prefix = PREFIX.get(mat)
    if prefix is None:
        return None
    roi = region('inventory_material' if inventory else 'recipe_material', root=root)
    return locate(frame, prefix + '_material_icon', roi=roi, root=root)


def items_down_anchor(frame: Any, *, root: Path | None = None):
    for key in ('items_excellent_bandage','items_high_bandage'):
        xy = locate(frame,key,root=root)
        if xy is not None:
            return key,xy
    return None


def items_down_points(frame: Any, *, root: Path | None = None, first: bool = True):
    """Only registered pre-potion landmarks authorize the next viewport."""
    anchor = items_down_anchor(frame,root=root)
    if (anchor is None or (first and anchor[0] != 'items_excellent_bandage'
            and locate(frame,'items_recovery_header',root=root) is None)):
        return None
    base = Path(root or ROOT)
    spec = json.loads((base/'knowledge/ui/config/process_navigation.json').read_text('utf-8'))['insect_items_list_down']
    x,y0,y1 = spec['x'],spec['start_y'],spec['end_y']
    rx,ry,rw,rh = spec['list_roi']
    if (spec['max_drags'] != 3 or any(isinstance(v,bool) or not isinstance(v,(int,float))
            or not math.isfinite(v) for v in (x,y0,y1,rx,ry,rw,rh))
            or not 0 <= rx < x < rx+rw <= 1 or not 0 <= ry < y1 < y0 < ry+rh <= 1):
        raise ValueError('invalid insect items-list drag configuration')
    h,w = frame.shape[:2]
    return int(w*x),int(h*y0),int(w*x),int(h*y1)
