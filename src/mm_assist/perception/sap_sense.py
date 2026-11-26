"""Captured controls for the 樹液 route through 柔軟木材."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from mm_assist.config import ROOT
from mm_assist.platform.match import match_template

RECIPE = '柔軟木材'
RECIPE_PAGE = f'facility:wood:{RECIPE}'
CARD_PAGE = 'item_info:樹液'
OBTAIN_PAGE = 'obtain_method:樹液'
_ASSETS = Path('knowledge/templates/process/wood/sap_navigation')


@lru_cache(maxsize=4)
def _catalog(folder: Path) -> dict:
    path = folder / 'catalog.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def locate(frame, control: str, *, root: Path | None = None) -> tuple[int, int] | None:
    if not getattr(frame, 'size', 0):
        return None
    folder = (root or ROOT) / _ASSETS
    spec = _catalog(folder).get(control)
    if spec is None:
        return None
    path = folder / spec['file']
    if not path.is_file():
        return None
    hit = match_template(frame, path, roi=tuple(spec['roi']), threshold=.97, min_margin=.02)
    return (hit.cx, hit.cy) if hit is not None else None
