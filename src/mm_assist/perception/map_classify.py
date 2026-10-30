"""地图/HUD OCR 只读分类。从 map_atlas 抽出，不含导航点击。"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np

from mm_assist.perception.list_ocr import box_center_norm, normalize_text
from mm_assist.perception.ocr import OcrLine
from mm_assist.platform.window import WindowInfo

MapState = Literal[
    "hud",
    "continent",
    "region",
    "npc_card",
    "traveling",
    "dialog",
    "unknown",
]

CONTINENT_NEEDLES = (
    "歐拉大陸",
    "欧拉大陆",
    "歐拉大陆",
    "欧拉大陸",
    "欧拉大睦",  # OCR 误识
    "歐拉大睦",
)
GOTO_NEEDLES = ("前往此處", "前往此处", "前往此", "前往奈", "前往")
STOP_NEEDLES = (
    "停止",
    "取消移動",
    "取消移动",
    "停止移動",
    "停止移动",
    "取消自動",
    "取消自动",
    "取消移",
    "停止移",
    # OCR 常见残片
    "停正",
    "停走",
    "取消動",
)
DIALOG_NEEDLES = ("對話", "对话", "交談", "修理", "修復", "修复", "购买", "購買", "武器商人", "带著清爽", "帶著清爽", "再見")

# 地图 chrome：不当作地区名
_CHROME = (
    "地圖",
    "地图",
    "歐拉大陸",
    "欧拉大陆",
    "歐拉",
    "欧拉",
    "地下城",
    "世界",
    "關閉",
    "关闭",
    "返回",
    "搜尋",
    "搜索",
    "推薦",
    "推荐",
    "Home",
    "ESC",
    "End",
    "前往",
    "距離",
    "距离",
    "治療所",
    "治疗所",
    "教会",
    "教會",
    "武器店",
    "鐵匠",
    "铁匠",
    "旅馆",
    "旅館",
    "銀行",
    "银行",
    "商店",
)

_REGION_HINTS = (
    "杜巴頓",
    "杜巴顿",
    "杜巴頔",  # OCR
    "堤爾克那",
    "堤尔克那",
    "佛格斯",
    "艾明馬夏",
    "艾明马夏",
    "歐絲拉",
    "欧丝拉",
    "班克爾",
    "班克尔",
    "萊比",
    "莱比",
    "提爾",
    "提尔",
    "艾明",
    "史卡哈",
    "肯尼斯",
    "都柏林",
    "埃门",
    "北門",
    "北门",
)

def _blob(lines: list[OcrLine]) -> str:
    return normalize_text("".join(L.text for L in lines))
def _is_inventory_blob(b: str) -> bool:
    """装备/背包面板（勿当地图 town）。"""
    marks = ("自動配戴", "自动配戴", "武器符", "防具符", "飾品符", "饰品符", "寶石·符文", "宝石·符文")
    return sum(1 for k in marks if k in b) >= 2


def _is_craft_ui_blob(b: str) -> bool:
    """生活工具/製作面板（遮住小地图，开图前须关掉）。"""
    marks = (
        "所需材料",
        "前往製作",
        "前往制作",
        "建立任務",
        "建立任务",
        "生活工具",
        "製作成功率",
        "制作成功率",
        "未持有",
        "多用途製作台",
        "多用途制作台",
    )
    if sum(1 for k in marks if k in b) >= 2:
        return True
    # 顶栏「工具」+ 分类页
    if ("工具" in b[:40] or "工具" in b[:80]) and any(
        k in b for k in ("全部", "生活工具", "改良型", "藥草鋤", "药草锄")
    ):
        return True
    return False


def _is_npc_dialog_blob(b: str) -> bool:
    """NPC 对话层（遮住 HUD/小地图）。"""
    if any(
        k in b
        for k in (
            "跳過對話",
            "跳过对话",
            "跳過对話",
            "跳對话",
            "跳对话",
            "跳對話",
            "跳对話",
        )
    ):
        return True
    if any(k in b for k in ("結束對話", "结束对话", "結束对話", "糙束")):
        return True
    # 对话选项菜单（无跳過时也要关）
    if any(k in b for k in ("随便聊聊", "隨便聊聊")) and any(
        k in b for k in ("商店", "結束", "结束", "修理")
    ):
        return True
    # 仅有对白气泡、无地图/任务栏强信号
    if "對話" in b or "对话" in b or "對话" in b:
        if not any(k in b for k in ("每週", "每周", "目標", "地图", "地圖", "歐拉")):
            return True
    return False


def _is_quest_offer_blob(b: str) -> bool:
    """兼职/任务接取层：接受 + 之後再做（遮住小地图）。OCR 常糊成 兼藏/之筱再做。"""
    has_later = ("再做" in b) or any(
        k in b for k in ("之後再", "之后再", "以後再", "以后再", "之筱再", "之筱")
    )
    has_accept = "接受" in b
    has_part = any(k in b for k in ("兼職", "兼职", "兼联", "兼藏", "兼職", "兼联"))
    return (has_later and has_accept) or (has_part and has_accept)


_QUEST_LATER_NEEDLES = (
    "之後再做",
    "之后再做",
    "以後再做",
    "以后再做",
    "之筱再做",
    "之筱再",
    "之後再",
    "之后再",
    "再做",
)
# 1920×1440 底栏「之後再做」近似中心（接受在左）
_QUEST_LATER_CAL_XY = (1105, 1285)


_SKIP_DIALOG_CLICK_NEEDLES = (
    "跳過對話",
    "跳过对话",
    "跳過对話",
    "跳對话",
    "跳对话",
    "跳對話",
    "跳对話",
    "跳過",
    "跳过",
    "跳對",
    "跳对",
    "結束對話",
    "结束对话",
    "結束对話",
    "糙束對话",
    "糙束對話",
    "糙束",
    "結束",
    "结束",
)
# 1920×1440 右上「跳過對話」近似中心（OCR 常糊成跳對话）
_SKIP_DIALOG_CAL_XY = (1768, 78)
# 对话底栏「結束對話」常见区
_END_DIALOG_CAL_XY = (1766, 1243)


def _is_town_directory_blob(b: str) -> bool:
    """城镇「主要」设施目录/图例（非世界地图；勿当 town 详图）。"""
    if "主要" not in b:
        return False
    marks = (
        "帽子開關",
        "帽子开关",
        "成長指南",
        "成长指南",
        "愛心商店",
        "爱心商店",
        "任務佈告",
        "任務怖告",
        "证物交换",
    )
    return sum(1 for k in marks if k in b) >= 2


def _map_opened_blob(b: str) -> bool:
    """强信号：勿用任务栏「地下城」误判。"""
    if _is_inventory_blob(b):
        return False
    if _is_craft_ui_blob(b):
        return False
    if _is_town_directory_blob(b):
        return False
    if _looks_hud_chrome(b):
        return False
    if any(k in b for k in ("歐拉大陸", "欧拉大陆", "世界地圖", "世界地图")):
        return True
    if "地圖" in b or "地图" in b:
        # 排除任务/聊天里偶发「地下城」而无地图标题
        if "Home" in b and "ESC" in b and "歐拉" not in b and "欧拉" not in b:
            # HUD 仍可见系统钮时，通常地图未盖住全屏
            if "理科" in b or "王牌" in b:
                return False
        return True
    return False


def _looks_hud_chrome(b: str) -> bool:
    """主界面 HUD（地图未盖住）：Home/ESC + 侧栏/任务噪声，勿当地图大陆层。"""
    if "Home" not in b or "ESC" not in b:
        return False
    if any(k in b for k in ("歐拉大陸", "欧拉大陆", "欧拉大睦", "歐拉大睦")):
        return False
    # 侧栏/技能栏常见字
    return any(
        k in b
        for k in (
            "王牌",
            "理科",
            "疾風",
            "疾风",
            "LEV",
            "每週目標",
            "每周目标",
            "每凋目標",
            "每调目標",
            "每遇自",
        )
    )


def classify_state(lines: list[OcrLine], img: np.ndarray | None = None) -> MapState:
    """五态 + dialog；实机文案可在 world_map.md 增补后同步 needles。"""
    b = _blob(lines)
    texts = [normalize_text(L.text) for L in lines]

    if any(n in b for n in DIALOG_NEEDLES) and (
        "選擇" in b or "选择" in b or "確認" in b or "确认" in b or "關閉" in b or "关闭" in b
    ):
        # 弱启发：对话常与选项并存；停走优先于对话误判
        if not any(n in b for n in STOP_NEEDLES):
            if "修理" in b or "對話" in b or "对话" in b or "交談" in b:
                return "dialog"

    if any(n in b for n in STOP_NEEDLES):
        return "traveling"

    if any(n in b for n in GOTO_NEEDLES) and (
        any(a in b for a in ("奈麗", "奈丽", "武器店", "鐵匠", "截匠", "杜巴"))
        or _has_nearby_name_card(texts)
    ):
        return "npc_card"

    if any(n in b for n in GOTO_NEEDLES):
        return "npc_card"

    has_continent_title = any(n in b for n in CONTINENT_NEEDLES) or "欧拉大睦" in b or "歐拉大睦" in b
    has_map = _map_opened_blob(b)
    regionish = sum(1 for h in _REGION_HINTS if h in b)
    facility = any(
        k in b
        for k in (
            "鐵匠",
            "铁匠",
            "截匠",
            "旅館",
            "旅馆",
            "學校",
            "学校",
            "乳牛",
            "北門",
            "北门",
            "防具",
            "皮革加工",
        )
    )
    # 面包屑「歐拉大陸 / 杜巴頓」+ 设施 = 地区图；点歐拉大陸后才是全图
    if has_map or has_continent_title:
        if facility or ("兼職" in b or "兼职" in b):
            if regionish <= 2 and (facility or "杜巴" in b):
                return "region"
        if regionish >= 3:
            return "continent"
        if has_continent_title and not facility:
            return "continent"
        if facility:
            return "region"
        return "continent"

    # HUD：有小地图区但无大地图标题（弱）
    if img is not None:
        h, w = img.shape[:2]
        corner = img[0 : max(1, h // 6), int(w * 0.85) : w]
        if float(corner.std()) > 8 and not has_map:
            return "hud"

    if not has_map and regionish == 0:
        return "hud"
    return "unknown"


def _has_nearby_name_card(texts: list[str]) -> bool:
    return any(len(t) >= 2 and len(t) <= 8 and re.search(r"[\u4e00-\u9fff]", t) for t in texts)
def _map_level(b: str, lines: list[OcrLine], win: WindowInfo) -> str:
    """town | region | continent | unknown — 实机分层。"""
    if _is_inventory_blob(b) or _is_town_directory_blob(b):
        return "unknown"
    # HUD 任务文案常含「地下城/莱比」等，禁止据此判 continent
    if _looks_hud_chrome(b) and not _map_opened_blob(b):
        return "unknown"
    fac = any(
        k in b
        for k in ("鐵匠", "铁匠", "截匠", "旅館", "旅馆", "雜貨", "杂货", "學校", "乳牛", "食材")
    )
    titles = [
        L.text
        for L in lines
        if box_center_norm(L, win.width, win.height)
        and box_center_norm(L, win.width, win.height)[1] < 0.12
    ]
    title_blob = "".join(titles)
    # 顶栏杜巴 + 设施芯片 = town（优先于任务栏「莱比/地下城」噪声）
    if fac and "杜巴" in title_blob:
        return "town"
    # 多城/地下城标签 = 大陆中景（即使 OCR 到「杜巴顿」地名也不算 town）
    world_marks = sum(
        1
        for k in (
            "莱比",
            "萊比",
            "希德",
            "冰霜",
            "堤爾",
            "堤尔",
            "库漢",
            "庫漢",
            "库汉",
            "地下城",
            "狩場",
            "狩场",
            "雪原",
            "伐木",
        )
        if k in b
    )
    other = sum(1 for k in ("堤爾", "堤尔", "佛格", "艾明", "班克", "歐絲", "提爾克那") if k in b)
    only_u = bool(titles) and all("杜巴" not in t for t in titles) and any(
        "欧" in t or "歐" in t for t in titles
    )
    # 面包屑「歐拉大陸／狼之森林」= 地区详图，禁 only_u 误升 continent（08-20 atlas score≈0.22）
    from mm_assist.perception.map_header import title_has_secondary_place
    title_secondary = title_has_secondary_place(lines, win)
    if title_secondary and not any(
        k in title_blob for k in _CONTINENT_CITY_KEYS
    ):
        return "region"
    if world_marks >= 2:
        return "continent"
    if other >= 1 or (only_u and not title_secondary):
        return "continent"
    # 标题停在他城（库漢等）且无杜巴 → 非杜巴頓 town
    if any(k in title_blob for k in ("库漢", "庫漢", "库汉", "庫汉")) and "杜巴" not in title_blob:
        return "continent" if world_marks >= 1 or only_u else "region"
    # 设施芯片 + 顶栏含杜巴 + 非多城 = town（禁止仅凭画面中城名「杜巴顿」判 town）
    if fac and "杜巴" not in title_blob and "杜巴" not in b and world_marks < 1:
        return "town"
    if "北門" in b or "北门" in b or "杜巴顿" in b or "杜巴頓" in b or "杜巴頔" in b:
        # 标题明确杜巴且无多城 → region/town 边缘，偏 region
        if "杜巴" in title_blob and world_marks == 0:
            return "region"
        return "region"
    if _map_opened_blob(b):
        return "region"
    return "unknown"


def _breadcrumb_has_secondary_place(text: str) -> bool:
    """顶栏「歐拉…／狼之森林」等：斜杠后还有地名 → 合并面包屑，须点左段歐拉。"""
    t = text or ""
    for sep in ("／", "/", "·", "|"):
        if sep not in t:
            continue
        right = t.split(sep, 1)[-1].strip()
        if right and not right.startswith(("欧", "歐")):
            return True
    return False

_CONTINENT_CITY_KEYS: tuple[str, ...] = (
    "杜巴頓",
    "杜巴顿",
    "杜巴頔",
    "佛格",
    "艾明",
    "班克",
    "歐絲",
    "欧丝",
    "库漢",
    "庫漢",
    "库汉",
)



def snap(cfg, win, *, min_conf: float = 0.30, save_prefix: str | None = None):
    """只读截屏 + OCR。"""
    from mm_assist.platform.capture import grab_client, save_capture
    from mm_assist.platform.window import refresh_window
    from mm_assist.perception.ocr import ocr_image

    win = refresh_window(win)
    img = grab_client(win, method=cfg.window.capture_method)
    lines = ocr_image(img, min_confidence=min_conf)
    path = None
    if save_prefix:
        path = save_capture(img, cfg.resolve(cfg.paths.captures), prefix=save_prefix)
    return win, img, lines, path
