"""系统菜单四键模板：冻帧后人手框选覆盖。方法 B，不向游戏发键鼠。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mm_assist.config import ROOT, AppConfig

MENU_REL = Path("knowledge") / "templates" / "menu"


@dataclass(frozen=True)
class MenuButtonSpec:
    id: str
    label: str
    filename: str
    rel_dir: str = "knowledge/templates/menu"
    need_ui: str = "请打开系统菜单"
    role: str = ""


MENU_BUTTONS: tuple[MenuButtonSpec, ...] = (
    MenuButtonSpec("character", "角色", "icon_character.png"),
    MenuButtonSpec("inventory", "背包", "icon_inventory.png"),
    MenuButtonSpec("craft", "制作", "icon_crafting.png"),
    MenuButtonSpec("close_x", "右上X", "icon_close_x.png"),
)

PAGE_BUTTONS: tuple[MenuButtonSpec, ...] = (
    MenuButtonSpec(
        "inventory_close",
        "关闭背包",
        "close.png",
        "knowledge/templates/inventory",
        "请打开背包，套住关闭钮",
    ),
    MenuButtonSpec(
        "profile_close",
        "关闭或X",
        "close.png",
        "knowledge/templates/profile",
        "请打开人物界面，套住关闭/X",
    ),
    MenuButtonSpec(
        "profile_life_tab",
        "生活技能Tab",
        "life_tab.png",
        "knowledge/templates/profile",
        "人物界面仍开着，套住生活技能 Tab",
    ),
    MenuButtonSpec(
        "life_grid_close",
        "关闭生活技能",
        "close.png",
        "knowledge/templates/life_grid",
        "请打开生活技能大格，套住关闭",
    ),
    MenuButtonSpec(
        "life_grid_skill_cell",
        "技能大格",
        "skill_cell.png",
        "knowledge/templates/life_grid",
        "生活技能大格仍开着，套住任意一个技能格（伐木/采矿等，22 格共用详情）",
    ),
    MenuButtonSpec(
        "gather_detail_close",
        "关闭详情",
        "close.png",
        "knowledge/templates/gather_detail",
        "请打开某个技能详情，套住关闭详情",
    ),
    MenuButtonSpec(
        "gather_detail_find_nearby",
        "寻找附近",
        "find_nearby.png",
        "knowledge/templates/gather_detail",
        "技能详情已选详情物，套住「寻找附近位置」按钮",
    ),
    MenuButtonSpec(
        "gather_detail_help",
        "问号帮助",
        "help.png",
        "knowledge/templates/gather_detail",
        "技能详情页，套住问号/说明按钮",
    ),
    MenuButtonSpec(
        "craft_hub_close",
        "枢纽右上X",
        "close_x.png",
        "knowledge/templates/craft_hub",
        "请打开制作加工枢纽，套住右上 X",
    ),
    MenuButtonSpec(
        "craft_card_metal",
        "金屬卡",
        "card_metal.png",
        "knowledge/templates/craft_hub",
        "加工档可见六卡时，套住金屬",
    ),
    MenuButtonSpec(
        "craft_card_wood",
        "木材卡",
        "card_wood.png",
        "knowledge/templates/craft_hub",
        "套住木材",
    ),
    MenuButtonSpec(
        "craft_card_leather",
        "皮革卡",
        "card_leather.png",
        "knowledge/templates/craft_hub",
        "套住皮革",
    ),
    MenuButtonSpec(
        "craft_card_cloth",
        "布料卡",
        "card_cloth.png",
        "knowledge/templates/craft_hub",
        "套住布料",
    ),
    MenuButtonSpec(
        "craft_card_medicine",
        "藥品卡",
        "card_medicine.png",
        "knowledge/templates/craft_hub",
        "套住藥品",
    ),
    MenuButtonSpec(
        "craft_card_food",
        "食材卡",
        "card_food.png",
        "knowledge/templates/craft_hub",
        "套住食材",
    ),
    MenuButtonSpec(
        "facility_back",
        "左上返回",
        "back.png",
        "knowledge/templates/facility",
        "请打开任一加工设施页，套住左上 ←（此页无顶栏 X）",
    ),
    MenuButtonSpec(
        "map_close_x",
        "地图右上X",
        "close_x.png",
        "knowledge/templates/map",
        "请打开杜巴顿区域地图，套住右上 X",
    ),
    MenuButtonSpec(
        "map_nerys",
        "奈丽丝头像",
        "nerys.png",
        "knowledge/templates/map",
        "杜巴顿地图仍开着，只套奈丽丝头像，不要点前往",
    ),
)

# C2 练采集：状态下拉九门（藥水調製/料理已取消，不采）
TRAIN_SKILL_CELLS: tuple[MenuButtonSpec, ...] = tuple(
    MenuButtonSpec(
        sid,
        label,
        f"{sid}.png",
        "knowledge/templates/life_grid",
        f"生活技能大格仍开着，套住「{label}」格",
    )
    for sid, label in (
        ("woodcut", "伐木"),
        ("mine", "採礦"),
        ("herb", "採集藥草"),
        ("shear", "剪羊毛"),
        ("harvest", "收割"),
        ("hoe", "鋤地"),
        ("insect", "昆蟲採集"),
        ("fish", "釣魚"),
        ("daily", "日常採集"),
    )
)


def menu_template_dir(*, root: Path | None = None) -> Path:
    return (root or ROOT) / MENU_REL


def menu_template_path(spec: MenuButtonSpec, *, root: Path | None = None) -> Path:
    return menu_template_dir(root=root) / spec.filename


def clamp_box(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    ih, iw = int(img.shape[0]), int(img.shape[1])
    x = max(0, min(int(x), max(0, iw - 1)))
    y = max(0, min(int(y), max(0, ih - 1)))
    w = max(4, min(int(w), iw - x))
    h = max(4, min(int(h), ih - y))
    return x, y, w, h


def crop_bgr(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    x, y, w, h = clamp_box(img, x, y, w, h)
    crop = np.ascontiguousarray(img[y : y + h, x : x + w])
    if crop.size == 0:
        raise ValueError("裁切为空")
    return crop


def save_menu_button_crop(
    img: np.ndarray,
    spec: MenuButtonSpec,
    box: tuple[int, int, int, int],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    x, y, w, h = clamp_box(img, *box)
    crop = crop_bgr(img, x, y, w, h)
    out_dir = (root or ROOT) / spec.rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / spec.filename
    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise RuntimeError(f"PNG 编码失败 {png}")
    tmp = png.with_suffix(".png.tmp")
    tmp.write_bytes(encoded.tobytes())
    tmp.replace(png)
    sidecar = {
        "id": spec.id,
        "label": spec.label,
        "file": spec.filename,
        "rel_dir": spec.rel_dir,
        "need_ui": spec.need_ui,
        "client_px": {"x": x, "y": y, "w": w, "h": h},
    }
    if spec.role:
        sidecar["role"] = spec.role
    json_path = png.with_suffix(".json")
    json_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "id": spec.id, "png": str(png), "json": str(json_path), **sidecar}


def run_menu_button_capture(cfg: AppConfig, *, interactive: bool = True) -> dict[str, Any]:
    if not interactive:
        return {"ok": True, "saved": [], "need_ui": "系统菜单已开"}
    from mm_assist.jobs.menu_crop_ui import run_overlay_session

    saved = run_overlay_session(cfg, MENU_BUTTONS)
    return {
        "ok": True,
        "need_ui": "系统菜单已开：角色、背包、制作、右上X 可见",
        "count": len(saved),
        "saved": saved,
    }


def detail_item_buttons(*, root: Path | None = None) -> tuple[MenuButtonSpec, ...]:
    from mm_assist.knowledge_paths import GATHER_TIERS_JSON

    path = GATHER_TIERS_JSON if root is None else Path(root) / "knowledge/ui/config/living_skill_gather_tiers.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[MenuButtonSpec] = []
    for skill in data.get("skills") or []:
        if not isinstance(skill, dict):
            continue
        sid = str(skill.get("id") or "").strip()
        skill_label = str(skill.get("ui_label") or sid)
        for tier in skill.get("tiers") or []:
            if not isinstance(tier, dict):
                continue
            for item in tier.get("items") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                rel = str(item.get("template") or "").replace("\\", "/")
                if not sid or not name or not rel:
                    continue
                parent = str(Path(rel).parent).replace("\\", "/")
                fname = Path(rel).name
                out.append(
                    MenuButtonSpec(
                        f"item_{sid}_{name}",
                        name,
                        fname,
                        parent,
                        f"请打开「{skill_label}」详情右栏，套住「{name}」",
                    )
                )
    return tuple(out)


PROCESS_NEED_UI = {
    "metal": "请打开金屬加工设施页，列表可见目标行时套住该行（左侧列表，不是详情右栏）",
    "wood": "请打开木材加工设施页，列表可见目标行时套住该行（左侧列表，不是详情右栏）",
    "cloth": "请打开布料加工设施页，列表可见目标行时套住该行（左侧列表，不是详情右栏）",
}


def process_page_buttons(*, root: Path | None = None) -> tuple[MenuButtonSpec, ...]:
    path = (root or ROOT) / "knowledge" / "templates" / "process" / "catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[MenuButtonSpec] = []
    facilities = data.get("facilities") or {}
    for fid in ("metal", "wood", "cloth"):
        need = PROCESS_NEED_UI[fid]
        for row in facilities.get(fid) or []:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id") or "").strip()
            label = str(row.get("label") or row.get("name") or "").strip()
            fname = str(row.get("file") or "").strip()
            rel = str(row.get("rel_dir") or f"knowledge/templates/process/{fid}").replace("\\", "/")
            if not rid or not fname or not label:
                continue
            hint = need
            if "鐵錠(鐵礦石)" in label:
                hint = "套住「鐵錠(鐵礦石)」"
            out.append(MenuButtonSpec(rid, label, fname, rel, f"{hint}：{label}"))
    actions = data.get("actions") or {}
    for row in actions.get("process_btn") or []:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "").strip()
        fname = str(row.get("file") or "").strip()
        label = str(row.get("label") or rid)
        if not rid or not fname:
            continue
        out.append(
            MenuButtonSpec(
                rid,
                label,
                fname,
                "knowledge/templates/process",
                str(row.get("need_ui") or "套住加工行动钮，勿点投料"),
            )
        )
    for row in actions.get("claim_all") or []:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "").strip()
        fname = str(row.get("file") or "").strip()
        if not rid or not fname:
            continue
        out.append(
            MenuButtonSpec(
                rid,
                str(row.get("label") or "全部領取"),
                fname,
                "knowledge/templates/process",
                str(row.get("need_ui") or "套住全部領取"),
            )
        )
    return tuple(out)


_GATHER_SKILL_ANCHOR_GATES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("woodcut", "伐木", ("樹木", "粗壮樹")),
    ("mine", "採礦", ("礦脈", "煤炭礦脈")),
    ("herb", "採集藥草", ("藥草", "淨淨蘑菇")),
    ("shear", "剪羊毛", ("羊", "捲毛羊")),
    ("harvest", "收割", ("小麥", "玉米", "黃豆")),
    ("hoe", "鋤地", ("馬鈴薯", "洋蔥", "防風草")),
    ("insect", "昆蟲採集", ("光群", "溫暖的光群")),
    ("fish", "釣魚", ("小釣魚場", "南側城牆釣魚場")),
)


def gather_skill_anchor_buttons() -> tuple[MenuButtonSpec, ...]:
    out: list[MenuButtonSpec] = []
    for sid, gate, names in _GATHER_SKILL_ANCHOR_GATES:
        need = (
            f"角色-生活技能-{gate} 右栏列表，不要点寻找附近，套住图标+物名"
        )
        for name in names:
            out.append(
                MenuButtonSpec(
                    f"gather_skill_{sid}_{name}",
                    name,
                    f"{name}.png",
                    f"knowledge/templates/gather_skill/{sid}",
                    f"{need}「{name}」",
                    "page_anchor",
                )
            )
    return tuple(out)


_CRAFT_TAB_SPECS: tuple[tuple[str, str, str], ...] = (
    ("quick", "快速製作", "tab_quick"),
    ("process", "加工", "tab_process"),
    ("make", "製作", "tab_make"),
    ("alchemy", "鍊金術", "tab_alchemy"),
)


def craft_tab_buttons() -> tuple[MenuButtonSpec, ...]:
    out: list[MenuButtonSpec] = []
    for stem_id, label, stem in _CRAFT_TAB_SPECS:
        on_need = (
            f"製作已开；先点「{label}」让这一栏变青、其它变灰；"
            "红框只套这一栏图标+栏名（不要套到右边内容）；"
            f"加工若有红点仍整套；不要点六卡、不要进设施。套住「{label}」选中(青)"
        )
        off_need = (
            f"製作已开；先点别的栏，让「{label}」变灰；"
            "红框只套这一栏图标+栏名（不要套到右边内容）；"
            f"不要点六卡、不要进设施。套住「{label}」未选(灰)"
        )
        out.append(
            MenuButtonSpec(
                f"craft_tab_{stem_id}_on",
                f"{label}·选中(青)",
                f"{stem}_on.png",
                "knowledge/templates/craft_hub",
                on_need,
                "tab_on",
            )
        )
        out.append(
            MenuButtonSpec(
                f"craft_tab_{stem_id}_off",
                f"{label}·未选(灰)",
                f"{stem}_off.png",
                "knowledge/templates/craft_hub",
                off_need,
                "tab_off",
            )
        )
    return tuple(out)


_FACILITY_TITLE_SPECS: tuple[tuple[str, str], ...] = (
    ("metal", "金屬加工"),
    ("wood", "木材加工"),
    ("leather", "皮革加工"),
    ("cloth", "布料加工"),
    ("medicine", "藥品加工"),
    ("food", "食材加工"),
)


def facility_title_buttons() -> tuple[MenuButtonSpec, ...]:
    out: list[MenuButtonSpec] = []
    for fid, label in _FACILITY_TITLE_SPECS:
        out.append(
            MenuButtonSpec(
                f"facility_title_{fid}",
                f"{label}·标题",
                f"title_{fid}.png",
                "knowledge/templates/facility",
                (
                    f"已进「{label}」设施页；红框只套左上标题「{label}」"
                    "（不要套六卡、不要套配方列表）"
                ),
                "page_anchor",
            )
        )
    return tuple(out)


def select_page_buttons(ids: list[str] | tuple[str, ...] | None = None) -> tuple[MenuButtonSpec, ...]:
    items = detail_item_buttons()
    process = process_page_buttons()
    anchors = gather_skill_anchor_buttons()
    craft_tabs = craft_tab_buttons()
    facility_titles = facility_title_buttons()
    catalog = (
        PAGE_BUTTONS
        + TRAIN_SKILL_CELLS
        + items
        + process
        + anchors
        + craft_tabs
        + facility_titles
    )
    if not ids:
        return PAGE_BUTTONS
    if list(ids) == ["items"]:
        return items
    if list(ids) == ["process"]:
        return process
    if list(ids) == ["gather-anchor"]:
        return anchors
    if list(ids) == ["craft-tabs"]:
        return craft_tabs
    if list(ids) == ["facility-titles"]:
        return facility_titles
    by_id = {s.id: s for s in catalog}
    out = tuple(by_id[i] for i in ids if i in by_id)
    if not out:
        raise ValueError(f"未知 ui-button id={list(ids)}")
    return out


def run_page_button_capture(
    cfg: AppConfig,
    *,
    interactive: bool = True,
    ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    buttons = select_page_buttons(ids)
    if not interactive:
        return {"ok": True, "saved": [], "need_ui": buttons[0].need_ui}
    from mm_assist.jobs.menu_crop_ui import run_overlay_session

    saved = run_overlay_session(cfg, buttons)
    return {
        "ok": True,
        "need_ui": "按提示换页后拖红框；N 跳过",
        "count": len(saved),
        "saved": saved,
    }
