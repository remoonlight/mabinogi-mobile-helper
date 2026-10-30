"""合成繁体 UI 基准图（离线 holdout；不冒充实机黄金帧）。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mm_assist.config import ROOT

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "ui_benchmark"
MANIFEST_NAME = "manifest.json"

# 简繁一旦互替会改语义，OCR 必须保真
TRAD_MARKERS = ("製", "藥", "採", "礦", "鋤", "鐮", "裝", "尋", "開")
SIMP_INTRUDERS = ("制", "药", "采", "矿", "锄", "镰", "装", "寻", "开")

_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msjh.ttc"),
    Path(r"C:\Windows\Fonts\msjhbd.ttc"),
    Path(r"C:\Windows\Fonts\mingliu.ttc"),
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size, index=0)
            except OSError:
                continue
    return ImageFont.load_default()


def _paint_bgr(
    w: int,
    h: int,
    *,
    bg: tuple[int, int, int],
    texts: list[tuple[str, int, int, int, tuple[int, int, int]]],
    boxes: list[tuple[int, int, int, int, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    for x, y, bw, bh, color in boxes or []:
        draw.rectangle([x, y, x + bw, y + bh], fill=color)
    for text, x, y, size, color in texts:
        draw.text((x, y), text, fill=color, font=_font(size))
    rgb = np.asarray(img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ensure_synthetic_fixtures(*, root: Path | None = None) -> Path:
    """写出合成 PNG + manifest。已存在且完整则跳过。"""
    out = (root or ROOT) / "tests" / "fixtures" / "ui_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []

    hud = _paint_bgr(
        640,
        360,
        bg=(28, 32, 40),
        boxes=[
            (20, 20, 72, 72, (90, 70, 40)),
            (500, 20, 110, 48, (50, 90, 70)),
            (20, 280, 160, 56, (40, 50, 90)),
        ],
        texts=[
            ("選單", 28, 40, 22, (255, 255, 255)),
            ("背包", 520, 28, 22, (255, 255, 255)),
            ("生活技能", 28, 292, 20, (255, 255, 255)),
        ],
    )
    _imwrite(out / "hud_entries.png", hud)
    samples.append(
        {
            "id": "hud_entries",
            "split": "train",
            "family": "main_hud",
            "file": "hud_entries.png",
            "width": 640,
            "height": 360,
            "dpi": 100,
            "synthetic": True,
            "needles": ["選單", "背包", "生活技能"],
            "clickable": [[20, 20, 92, 92], [500, 20, 610, 68], [20, 280, 180, 336]],
            "forbidden": [[300, 140, 380, 220]],
            "template_roi": [20, 20, 72, 72],
        }
    )

    skill = _paint_bgr(
        640,
        360,
        bg=(18, 22, 28),
        boxes=[(80, 80, 480, 48, (35, 40, 55)), (80, 150, 480, 48, (35, 40, 55))],
        texts=[
            ("採礦", 100, 88, 28, (240, 230, 200)),
            ("採集藥草", 100, 158, 28, (240, 230, 200)),
        ],
    )
    _imwrite(out / "skill_list.png", skill)
    samples.append(
        {
            "id": "skill_list",
            "split": "holdout",
            "family": "life_skill",
            "file": "skill_list.png",
            "width": 640,
            "height": 360,
            "dpi": 100,
            "synthetic": True,
            "needles": ["採礦", "採集藥草"],
            "clickable": [[80, 80, 560, 128], [80, 150, 560, 198]],
            "forbidden": [[0, 0, 40, 40]],
            "template_roi": [80, 80, 160, 48],
        }
    )

    gather = _paint_bgr(
        640,
        360,
        bg=(22, 20, 18),
        boxes=[
            (360, 250, 240, 52, (45, 95, 70)),
            (40, 250, 220, 52, (80, 50, 40)),
        ],
        texts=[
            ("礦脈", 48, 80, 32, (220, 210, 180)),
            ("尋找附近", 70, 260, 22, (255, 255, 255)),
            ("開始採集", 390, 260, 22, (255, 255, 255)),
        ],
    )
    _imwrite(out / "gather_actions.png", gather)
    samples.append(
        {
            "id": "gather_actions",
            "split": "holdout",
            "family": "gather",
            "file": "gather_actions.png",
            "width": 640,
            "height": 360,
            "dpi": 100,
            "synthetic": True,
            "needles": ["礦脈", "尋找附近", "開始採集"],
            "clickable": [[360, 250, 600, 302], [40, 250, 260, 302]],
            "forbidden": [[40, 250, 260, 302]],
            "note": "forbidden=前往加工占位；開始採集才可点",
            "template_roi": [360, 250, 240, 52],
        }
    )

    craft = _paint_bgr(
        640,
        360,
        bg=(30, 26, 36),
        texts=[
            ("製作", 40, 30, 30, (255, 240, 220)),
            ("藥品加工", 40, 90, 26, (230, 220, 200)),
            ("修理已佩戴的裝備", 40, 160, 22, (230, 220, 200)),
            ("鶴嘴鋤 鐮刀", 40, 220, 22, (200, 210, 180)),
        ],
    )
    _imwrite(out / "craft_repair.png", craft)
    samples.append(
        {
            "id": "craft_repair",
            "split": "holdout",
            "family": "craft",
            "file": "craft_repair.png",
            "width": 640,
            "height": 360,
            "dpi": 100,
            "synthetic": True,
            "needles": ["製作", "藥品加工", "裝備", "鋤", "鐮"],
            "clickable": [[40, 30, 160, 80]],
            "forbidden": [[400, 280, 620, 350]],
            "template_roi": [40, 30, 120, 50],
        }
    )

    neg = _paint_bgr(
        640,
        360,
        bg=(12, 12, 12),
        texts=[("戰鬥中", 240, 160, 28, (180, 40, 40))],
    )
    _imwrite(out / "negative_combat.png", neg)
    samples.append(
        {
            "id": "negative_combat",
            "split": "holdout",
            "family": "combat",
            "file": "negative_combat.png",
            "width": 640,
            "height": 360,
            "dpi": 100,
            "synthetic": True,
            "needles": ["戰鬥中"],
            "clickable": [],
            "forbidden": [[0, 0, 640, 360]],
            "template_roi": [240, 160, 160, 40],
        }
    )

    manifest = {
        "schema": "ui_benchmark.v1",
        "locale": "zh-Hant",
        "source": "synthetic_offline",
        "note": "合成帧仅用于离线相对比较；不得宣称实机感知够用。实机黄金帧须方法 B。",
        "trad_markers": list(TRAD_MARKERS),
        "simp_intruders": list(SIMP_INTRUDERS),
        "samples": samples,
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out


def _imwrite(path: Path, bgr: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raw = path.as_posix()
        cv2.imencode(".png", bgr)[1].tofile(raw)


def load_manifest(root: Path | None = None) -> dict:
    ensure_synthetic_fixtures(root=root)
    base = (root or ROOT) / "tests" / "fixtures" / "ui_benchmark"
    return json.loads((base / MANIFEST_NAME).read_text(encoding="utf-8"))


def load_sample_bgr(sample: dict, *, root: Path | None = None) -> np.ndarray:
    base = (root or ROOT) / "tests" / "fixtures" / "ui_benchmark"
    path = base / str(sample["file"])
    if not path.is_file():
        ensure_synthetic_fixtures(root=root)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img
