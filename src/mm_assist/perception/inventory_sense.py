"""Read-only bag controls, located from retained-frame templates."""
import json
from pathlib import Path
from typing import Any

from mm_assist.config import ROOT
from mm_assist.platform.match import match_template


def locate_props_tab(frame: Any, *, root: Path | None = None) -> tuple[int, int] | None:
    return _props_tab_hit(frame, root=root, selected=False)


def props_tab_selected(frame: Any, *, root: Path | None = None) -> bool:
    return _props_tab_hit(frame, root=root, selected=True) is not None


def _props_tab_hit(frame: Any, *, root: Path | None, selected: bool) -> tuple[int, int] | None:
    folder = (root or ROOT) / 'knowledge/templates/inventory'
    if frame is None:
        return None
    try:
        cfg = json.loads((folder / 'tab_props.json').read_text(encoding='utf-8'))
        hit = match_template(frame, folder / cfg['selected_file' if selected else 'file'], roi=tuple(cfg['roi']),
            threshold=float(cfg['threshold']), min_margin=float(cfg['min_margin']))
        if selected and hit is not None:
            import cv2
            import numpy as np

            label = frame[hit.y:hit.y+hit.h, hit.x:hit.x+hit.w]
            if label.ndim != 3 or label.size == 0:
                return None
            hsv = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
            orange = cv2.inRange(hsv, np.array(cfg['selected_hsv_min'], dtype=np.uint8),
                                np.array(cfg['selected_hsv_max'], dtype=np.uint8))
            if float(np.count_nonzero(orange))/orange.size < float(cfg['selected_min_color_fraction']):
                return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (int(hit.cx), int(hit.cy)) if hit is not None else None
