"""界面态指纹：ROI 哈希 + MemScene 字段序列化 / 样本匹配。

标定产物：`knowledge/ui/ui_mode_samples/<label>/*.json`
调用方：`ui_mode_calibrate` · `scene_status.classify_ui_mode`
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mm_assist.memory.readers.scene import MemSceneInfo
from mm_assist.perception.rois import HUD_SEEDS

log = logging.getLogger(__name__)

SAMPLES_REL = Path("knowledge/ui/ui_mode_samples")

# 标定优先 6 态（业务名 → UiKind）；地图拆成 town/region，总地图=continent
PRIORITY_CALIB_LABELS: list[tuple[str, str]] = [
    ("hud", "非战斗"),
    ("combat", "战斗"),
    ("inventory", "物品"),
    ("map_town", "地图·城镇"),
    ("map_region", "地图·地区/地城"),
    ("map_continent", "总地图"),
    ("chat", "聊天"),
]

# 指纹 ROI：HUD 种子 + 中心大面板（背包/地图）
_FP_ROIS: list[tuple[str, tuple[float, float, float, float]]] = [
    (s["id"], tuple(s["roi"]))  # type: ignore[misc]
    for s in HUD_SEEDS
]
_FP_ROIS.extend(
    [
        ("panel.center", (0.20, 0.12, 0.60, 0.76)),
        ("panel.left", (0.02, 0.20, 0.28, 0.55)),
        ("panel.top", (0.25, 0.02, 0.50, 0.12)),
    ]
)

MEM_FIELDS = (
    "map_open",
    "minimap_visible",
    "dialog_active",
    "repair_list_active",
    "repair_modal_open",
    "menu_open",
    "quest_open",
    "chat_open",
    "chat_focus",
    "chat_fullscreen",
    "craft_hub_category",
    "side_nav_open",
    "profile_open",
    "in_combat",
    "combat_mode_ui",
    "combat_state",
    "map_type",
    "map_type_name",
    "channel_sheet_id",
    "display_name",
    "in_dungeon",
    "in_normal_field",
    "changing_channel",
    "source_hits",
    "status",
)


def default_samples_dir(knowledge_root: Path | None = None) -> Path:
    if knowledge_root is None:
        return Path.cwd() / SAMPLES_REL
    return Path(knowledge_root) / "ui" / "ui_mode_samples"


def mem_to_dict(ms: MemSceneInfo | None) -> dict[str, Any]:
    if ms is None:
        return {"status": "none"}
    out: dict[str, Any] = {}
    for k in MEM_FIELDS:
        if hasattr(ms, k):
            out[k] = getattr(ms, k)
    return out


def _crop_roi(img: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray | None:
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return None
    x, y, rw, rh = roi
    x0 = max(0, min(w - 1, int(x * w)))
    y0 = max(0, min(h - 1, int(y * h)))
    x1 = max(x0 + 1, min(w, int((x + rw) * w)))
    y1 = max(y0 + 1, min(h, int((y + rh) * h)))
    return img[y0:y1, x0:x1]


def ahash_hex(gray: np.ndarray, size: int = 8) -> str:
    """平均哈希 → 16 hex（64 bit）。"""
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    mean = float(np.mean(small))
    bits = (small.astype(np.float32) >= mean).astype(np.uint8).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


def compute_fingerprints(img: np.ndarray | None) -> dict[str, str]:
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return {}
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    out: dict[str, str] = {}
    for name, roi in _FP_ROIS:
        crop = _crop_roi(gray, roi)
        if crop is None or crop.size == 0:
            continue
        out[name] = ahash_hex(crop)
    # 全图粗指纹
    out["full"] = ahash_hex(gray)
    return out


def hamming_hex(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 64
    try:
        x = int(a, 16) ^ int(b, 16)
    except ValueError:
        return 64
    return int(x.bit_count())


def fingerprint_distance(a: dict[str, str], b: dict[str, str]) -> float:
    """加权平均汉明距离；缺键忽略。越小越像。"""
    keys = set(a) & set(b)
    if not keys:
        return 999.0
    weights = {"full": 0.5, "hud.chat": 1.5, "panel.center": 1.2}
    total_w = 0.0
    total_d = 0.0
    for k in keys:
        w = float(weights.get(k, 1.0))
        total_w += w
        total_d += w * hamming_hex(a[k], b[k])
    return total_d / max(total_w, 1e-6)


@dataclass
class SampleMatch:
    label: str
    distance: float
    path: str


def load_sample_files(samples_dir: Path | None = None) -> list[dict[str, Any]]:
    root = samples_dir or default_samples_dir()
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for p in sorted(root.glob("*/*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.debug("skip sample %s: %s", p, exc)
            continue
        if not isinstance(raw, dict):
            continue
        raw["_path"] = str(p)
        rows.append(raw)
    return rows


def match_fingerprint(
    fps: dict[str, str],
    *,
    samples_dir: Path | None = None,
    max_distance: float = 18.0,
    prefer_labels: set[str] | None = None,
) -> SampleMatch | None:
    """在样本库中找最近指纹；超过阈值返回 None。"""
    if not fps:
        return None
    best: SampleMatch | None = None
    for row in load_sample_files(samples_dir):
        label = str(row.get("label") or "").strip()
        if not label:
            continue
        if prefer_labels is not None and label not in prefer_labels:
            continue
        fp = row.get("fingerprints") or {}
        if not isinstance(fp, dict):
            continue
        dist = fingerprint_distance(fps, {str(k): str(v) for k, v in fp.items()})
        if dist > max_distance:
            continue
        if best is None or dist < best.distance:
            best = SampleMatch(label=label, distance=dist, path=str(row.get("_path") or ""))
    return best


def chat_roi_hint(img: np.ndarray | None) -> bool:
    """轻量：左下聊天 ROI 亮度/对比相对全图偏亮或有结构 → 可能聊天展开。"""
    if img is None or img.size == 0:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    # hud.chat ROI
    x0, y0 = int(0.0104 * w), int(0.9306 * h)
    x1, y1 = int(0.2344 * w), int(0.9862 * h)
    crop = gray[y0:y1, x0:x1]
    if crop.size < 16:
        return False
    # 输入条展开时常更亮、边缘更多
    mean = float(np.mean(crop))
    std = float(np.std(crop))
    return mean > 55 and std > 18


def build_sample_payload(
    *,
    label: str,
    ms: MemSceneInfo | None,
    img: np.ndarray | None,
    game_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "label": label,
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mem": mem_to_dict(ms),
        "fingerprints": compute_fingerprints(img),
        "game_frame": game_frame or {},
    }


def write_sample(
    payload: dict[str, Any],
    *,
    samples_dir: Path | None = None,
) -> Path:
    root = samples_dir or default_samples_dir()
    label = str(payload.get("label") or "unknown")
    dest_dir = root / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_dir / f"{ts}.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dest
