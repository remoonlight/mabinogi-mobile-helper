"""生活短链认页。

resolve_navigation 用当前帧和即时只读栈，并做前后一致性核对。
resolve_ui_node / observe 是兼容与显示态观测。
模板可见性辅助函数只认图，不输入。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm_assist.config import ROOT
from mm_assist.life_plan.execution_records import trace_navigation
from mm_assist.memory.readers.ui_stack import (
    left_main_hud,
    input_blocking_layers,
    stack_compare_key,
    stack_dlg,
    stack_hud_only,
    stack_left_overlay,
    stack_node,
)
from mm_assist.perception.state_ui import parse_ui_param_bit, read_state_ui, state_ui_path

KNOWN_NODES = (
    "main_hud",
    "system_menu",
    "inventory",
    "map",
    "map_dunbarton",
    "map_continent",
    "profile",
    "life_grid",
    "gather_skill",
    "gather_detail",
    "craft_hub",
    "craft_quick",
    "craft_make",
    "craft_alchemy",
    "facility",
    "facility_recipe",
    "currency_wallet",
    "dialog",
    "dialog_skip",
    "item_info",
    "obtain_method",
    "craft_shell",
    "craft_items",
)

GATHER_NODES = frozenset(
    {
        "main_hud",
        "system_menu",
        "profile",
        "life_grid",
        "gather_skill",
        "gather_detail",
        "unknown",
    }
)
FEED_NODES = frozenset(
    {
        "main_hud",
        "system_menu",
        "craft_hub",
        "craft_quick",
        "craft_make",
        "craft_items",
        "craft_alchemy",
        "facility",
        "facility_recipe",
        "currency_wallet",
        "unknown",
    }
)

CLOSE_TEMPLATES: tuple[Path, ...] = (
    Path("knowledge") / "templates" / "menu" / "icon_close_x.png",
    Path("knowledge") / "templates" / "inventory" / "close.png",
    Path("knowledge") / "templates" / "profile" / "close.png",
    Path("knowledge") / "templates" / "life_grid" / "close.png",
    Path("knowledge") / "templates" / "gather_detail" / "close.png",
    Path("knowledge") / "templates" / "craft_hub" / "close_x.png",
    Path("knowledge") / "templates" / "facility" / "back.png",
    Path("knowledge") / "templates" / "map" / "close_x.png",
)

_ID_TEMPLATES: tuple[tuple[str, Path], ...] = (
    ("life_grid", Path("knowledge") / "templates" / "life_grid" / "daily.png"),
    ("life_grid", Path("knowledge") / "templates" / "life_grid" / "herb.png"),
    ("life_grid", Path("knowledge") / "templates" / "life_grid" / "woodcut.png"),
    ("profile", Path("knowledge") / "templates" / "profile" / "life_tab.png"),
    ("facility", Path("knowledge") / "templates" / "facility" / "back.png"),
)

_MENU_TRIPLE: tuple[Path, ...] = (
    Path("knowledge") / "templates" / "menu" / "icon_character.png",
    Path("knowledge") / "templates" / "menu" / "icon_inventory.png",
    Path("knowledge") / "templates" / "menu" / "icon_crafting.png",
)

PROFILE_TAB_NEEDLES = ("我的資訊", "我的信息", "戰鬥技能", "战斗技能", "生活技能", "職業", "职业")
LIFE_TITLE_NEEDLES = ("生活技能",)
CRAFT_TITLE_ROI = (0.02, 0.00, 0.45, 0.12)
LIFE_TITLE_ROI = (0.16, 0.00, 0.52, 0.12)
PROFILE_TAB_ROI = (0.00, 0.08, 0.20, 0.55)

CLOSE_OCR_ROI = (0.78, 0.00, 0.22, 0.16)
CLOSE_OCR_NEEDLES = ("關閉", "关闭", "閉じる")


@dataclass(frozen=True)
class UiWhere:
    node: str
    menu: int | None
    dlg: int | None
    stp: int | None
    source: str
    skill_id: str = ""
    craft_tab: str = ""
    process_stance: str = ""
    overlay: str = ""
    facility_id: str = ""
    stack_names: tuple[str, ...] = ()
    stack_ready: bool = False
    cost_ms: dict[str, float] = field(default_factory=dict)
    stack_sampled_ts: float = 0.0

    @property
    def known(self) -> bool:
        return graph_node(self.node) in KNOWN_NODES

    def bits(self) -> str:
        def _b(k: str, v: int | None) -> str:
            return f"{k}={v}" if v in (0, 1) else f"{k}=?"

        return " ".join(
            (_b("menu", self.menu), _b("dlg", self.dlg), _b("stp", self.stp))
        )


@dataclass(frozen=True)
class Observation:
    where: UiWhere
    stamps: tuple[float, float]  # from ui_param_stamps
    frame: Any = None


def _stack_from_view(view: Any) -> tuple[list[str] | None, bool]:
    raw = getattr(view, "raw", None)
    if isinstance(raw, dict) and "ui_stack_names" in raw:
        names = [str(x) for x in (raw.get("ui_stack_names") or [])]
        return names, bool(raw.get("ui_stack_ready"))
    names = getattr(view, "ui_stack_names", None)
    if names is not None:
        return [str(x) for x in names], bool(getattr(view, "ui_stack_ready", True))
    return None, False


def _bits_from_view(view: Any) -> tuple[int | None, int | None, int | None]:
    params = str(getattr(view, "params", "") or "")
    stp = parse_ui_param_bit(params, "stp")
    if stp is None:
        try:
            raw_stp = getattr(view, "stp", None)
            stp = int(raw_stp) if raw_stp is not None else None
        except (TypeError, ValueError):
            stp = None
    names, ready = _stack_from_view(view)
    if names is not None:
        if not ready:
            return None, None, stp
        left = left_main_hud(names)
        return (
            1 if left else 0,
            1 if stack_dlg(names) else 0,
            stp,
        )
    menu = parse_ui_param_bit(params, "menu")
    dlg = parse_ui_param_bit(params, "dlg")
    if dlg is None and getattr(view, "dialog_active", None) is True:
        dlg = 1
    if menu is None and getattr(view, "menu_open", None) is True:
        menu = 1
    return menu, dlg, stp


INSPECT_PAGE = "ui_inspect.json"
INSPECT_MAX_AGE_S = 16.0
WAIT_PARAM_S = 40.0
LAST_UI_WAIT: dict[str, Any] = {}


def inspect_page_path(root: Path | None = None) -> Path:
    p = state_ui_path(root)
    return p.with_name(INSPECT_PAGE)


def _file_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _file_ts(path: Path) -> float:
    try:
        return float(_file_json(path).get("ts_unix") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def ui_param_stamps(*, root: Path | None = None) -> tuple[float, float]:
    """(四参落盘 ts, 界面：落盘 ts)。缺文件为 0。"""
    return _file_ts(state_ui_path(root)), _file_ts(inspect_page_path(root))


def _ui_stack_sample_ts(*, root: Path | None = None) -> float:
    """落盘 `ui_stack_ts`。缺字段 / 非正数 = 未就绪（不得当两次不同采样）。"""
    try:
        return float(_file_json(state_ui_path(root)).get("ui_stack_ts") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def ui_param_key(*, root: Path | None = None) -> tuple[str, str]:
    """对拍键：stp|去Hud栈名 + 界面：节点。"""
    su = _file_json(state_ui_path(root))
    rec = read_inspect_record(root=root)
    params = str(su.get("ui_params") or "").strip()
    stp = parse_ui_param_bit(params, "stp")
    if stp is None:
        try:
            stp = int(su["stp"]) if su.get("stp") is not None else None
        except (TypeError, ValueError):
            stp = None
    stp_tok = f"stp={stp}" if stp in (0, 1) else "stp=?"
    names = [str(x) for x in (su.get("ui_stack_names") or [])]
    if su.get("ui_stack_ready") or names:
        key = f"{stp_tok}|{stack_compare_key(names)}"
    else:
        key = params
    node = str(rec.get("node") or "").strip()
    return key, node


def skip_wait_ui_params() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def observe(
    *,
    frame: Any = None,
    root: Path | None = None,
    resolve: Any = None,
) -> Observation:
    """一次观测 = resolve_ui_node + stamps。"""
    if resolve is not None:
        where = resolve(frame)
    else:
        where = resolve_ui_node(frame=frame, root=root)
    return Observation(where=where, stamps=ui_param_stamps(root=root), frame=frame)


def wait_ui_params_refresh(
    *,
    root: Path | None = None,
    sleep_fn: Any = None,
    timeout_s: float = WAIT_PARAM_S,
    cancelled: Any = None,
    pre_wait_s: float = 1.0,
    before: tuple[str, str] | None = None,
    require_change: bool = False,
) -> bool:
    """点后先等 pre_wait_s，再等到落盘两拍 key 一致且 ui_stack_ts 为两次不同采样。"""
    import time

    t0 = time.monotonic()
    LAST_UI_WAIT.clear()
    LAST_UI_WAIT.update(outcome="pending", timeout_s=float(timeout_s), pre_wait_s=float(pre_wait_s or 0))

    def _done(outcome: str, ok: bool) -> bool:
        LAST_UI_WAIT.update(outcome=outcome, elapsed_s=round(time.monotonic() - t0, 3))
        return ok

    if skip_wait_ui_params():
        return _done("skip_pytest", True)
    sleeper = sleep_fn or time.sleep
    if pre_wait_s and float(pre_wait_s) > 0:
        sleeper(float(pre_wait_s))
    su0, in0 = ui_param_stamps(root=root)
    if su0 <= 0.0 and in0 <= 0.0:
        return _done("no_stamps", True)
    from mm_assist.life_plan.game_stop import operation_wall_time
    deadline = operation_wall_time() + float(timeout_s)
    prev: tuple[str, str] | None = None
    prev_stack_ts = 0.0
    while operation_wall_time() < deadline:
        if cancelled is not None and cancelled():
            return _done("cancel", False)
        su, ins = ui_param_stamps(root=root)
        if su0 > 0.0:
            bumped = su > su0
        else:
            bumped = ins > in0
        if bumped:
            key = ui_param_key(root=root)
            stack_ts = _ui_stack_sample_ts(root=root)
            if require_change and before is not None and key == before:
                prev = key
                prev_stack_ts = stack_ts
                su0, in0 = su, ins
                sleeper(0.05)
                continue
            if (
                prev is not None
                and key == prev
                and prev_stack_ts > 0.0
                and stack_ts > 0.0
                and prev_stack_ts != stack_ts
            ):
                return _done("two_beat", True)
            prev = key
            prev_stack_ts = stack_ts
            su0, in0 = su, ins
        sleeper(0.05)
    return _done("timeout", False)


def write_inspect_page(
    node: str,
    *,
    zh: str = "",
    root: Path | None = None,
    stack_names: list[str] | tuple[str, ...] | None = None,
    stack_ready: bool | None = None,
    source: str = "",
    cost: dict | None = None,
) -> None:
    """状态栏「界面：」落盘。内容变则全量写；内容不变也刷新 ts_unix+pid，避免 16s 老化。cost 不参与 same。"""
    import time

    path = inspect_page_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    node_s = str(node or "").strip()
    zh_s = str(zh or "").strip()
    names = [str(x) for x in (stack_names or [])]
    ready_b = bool(stack_ready)
    src = str(source or "").strip()
    prev: dict[str, Any] = {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            prev = raw
    except (OSError, json.JSONDecodeError, TypeError):
        prev = {}
    same = (
        str(prev.get("node") or "").strip() == node_s
        and str(prev.get("zh") or "").strip() == zh_s
        and [str(x) for x in (prev.get("stack_names") or [])] == names
        and bool(prev.get("stack_ready")) == ready_b
        and str(prev.get("source") or "").strip() == src
    )
    payload: dict[str, Any] = {
        "ts_unix": time.time(),
        "pid": int(os.getpid()),
        "node": node_s,
        "zh": zh_s,
        "stack_names": names,
        "stack_ready": ready_b,
        "source": src,
    }
    if isinstance(cost, dict):
        payload["cost"] = cost
    if same:
        payload["node"] = node_s
        payload["zh"] = zh_s
        payload["stack_names"] = names
        payload["stack_ready"] = ready_b
        payload["source"] = src
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_inspect_page(*, root: Path | None = None, now: float | None = None) -> str:
    rec = read_inspect_record(root=root, now=now)
    return str(rec.get("node") or "").strip()


def read_inspect_record(*, root: Path | None = None, now: float | None = None) -> dict[str, Any]:
    import time

    path = inspect_page_path(root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    if raw.get("ts_unix") is None:
        return {}
    try:
        ts = float(raw.get("ts_unix") or 0.0)
    except (TypeError, ValueError):
        return {}
    if ts <= 0.0:
        return {}
    age = max(0.0, float(now if now is not None else time.time()) - ts)
    if age > INSPECT_MAX_AGE_S:
        return {}
    if "pid" in raw:
        try:
            pid = int(raw.get("pid"))
        except (TypeError, ValueError):
            return {}
        if pid != int(os.getpid()):
            return {}
    return raw


CRAFT_SHELL_NODES = frozenset(
    {"craft_hub", "craft_quick", "craft_make", "craft_alchemy", "craft_shell"}
)


def graph_node(node: str) -> str:
    """界面参数细名 → 切页图节点。例 facility:metal → facility；facility:metal:鋼錠 → facility_recipe。"""
    n = str(node or "").strip()
    if n.startswith("profile:"):
        return "profile"
    if n.startswith("map"):
        return "map_dunbarton"
    if n == "facility_recipe" or n.startswith("facility_recipe:"):
        return "facility_recipe"
    if n.startswith("facility:") and len(n.split(":")) >= 3:
        return "facility_recipe"
    if n.startswith("facility"):
        return "facility"
    if n.startswith("gather_skill"):
        return "gather_skill"
    if n.startswith("gather_detail"):
        return "gather_detail"
    if n.startswith("item_info"):
        return "item_info"
    if n.startswith("obtain_method"):
        return "obtain_method"
    if n.startswith("dialog_skip"):
        return "dialog_skip"
    if n.startswith("fine:"):
        return n.split(":", 1)[1]
    if n.startswith("stack:"):
        return n
    return n


def hud_page_pending(where: UiWhere) -> bool:
    """A ready HUD shell without an identified page is not a closable overlay."""
    return (graph_node(where.node) in {"overlay", "unknown"}
            and bool(where.stack_ready)
            and stack_hud_only(where.stack_names))


def is_craft_shell(node: str) -> bool:
    """製作壳四 Tab（加工/快速/制作/炼金）；不含设施页。"""
    return graph_node(node) in CRAFT_SHELL_NODES


def facility_id_of(node: str) -> str:
    n = str(node or "").strip()
    if not n.startswith("facility:"):
        return ""
    rest = n.split(":", 1)[1]
    return rest.split(":", 1)[0].strip()


def _empty_hud_ok(view: Any) -> bool:
    """空栈已确认，或栏上栈串就是「主界面」。"""
    raw = getattr(view, "raw", None) if view is not None else None
    if not isinstance(raw, dict):
        return False
    if bool(raw.get("ui_stack_empty_ok")):
        return True
    line = str(raw.get("ui_stack") or "").replace(" ", "").replace("\u3000", "")
    return line == "主界面"


def node_from_mem(view: Any) -> tuple[str, str]:
    """当前有效内存关联优先；缺失时只认栈，开窗参数不证明当前分页。"""
    tab = detect_craft_tab(None, view, root=None)
    names, ready = _stack_from_view(view)
    if not ready:
        return "unknown", tab
    from mm_assist.life_plan.node_inspect import _profile_node_from_view, _map_node_from_view, _craft_node_from_view, map_node_from_type, _view_field
    profile = _profile_node_from_view(view)
    if profile:
        return profile, tab
    map_node = _map_node_from_view(view)
    if map_node:
        return map_node, tab
    craft_node = _craft_node_from_view(view)
    if craft_node:
        return craft_node, _CRAFT_NODE_TO_TAB[craft_node]
    cat = getattr(view, "craft_hub_category", None) if view is not None else None
    try:
        cat_i = int(cat) if cat is not None else None
    except (TypeError, ValueError):
        cat_i = None
    sn = stack_node(
        list(names or []),
        ready=True,
        craft_category=cat_i,
        empty_ok=_empty_hud_ok(view),
    )
    if not sn:
        return "unknown", tab
    return sn, tab


def recognized_click_xy(hit: Any, *, min_score: float = 0.72) -> tuple[int, int] | None:
    """未认到（模板/OCR ok）一律不给坐标，禁止用 ROI 中心冒充。"""
    if hit is None:
        return None
    if isinstance(hit, dict):
        if hit.get("ok") is False:
            return None
        recognized = hit.get("ok") is True or hit.get("feature_ok") is True or hit.get("accepted") is True
        if not recognized:
            return None
        cc = hit.get("click_client")
        if isinstance(cc, (list, tuple)) and len(cc) >= 2:
            xy = (int(cc[0]), int(cc[1]))
        elif "cx" in hit and "cy" in hit:
            xy = (int(hit["cx"]), int(hit["cy"]))
        else:
            click = hit.get("click") if isinstance(hit.get("click"), dict) else None
            client = click.get("client") if click else None
            if isinstance(client, (list, tuple)) and len(client) >= 2:
                xy = (int(client[0]), int(client[1]))
            else:
                return None
    else:
        score = getattr(hit, "score", None)
        if score is not None:
            try:
                if float(score) < float(min_score):
                    return None
            except (TypeError, ValueError):
                return None
        cx = getattr(hit, "cx", None)
        cy = getattr(hit, "cy", None)
        if cx is None or cy is None:
            return None
        xy = (int(cx), int(cy))
    if xy[0] == 0 and xy[1] == 0:
        return None
    return xy


def _hit_xy(hit: Any) -> tuple[int, int] | None:
    return recognized_click_xy(hit)


def _match(
    frame: Any,
    rel: Path,
    *,
    root: Path | None,
    threshold: float = 0.72,
    roi: tuple[float, float, float, float] | None = None,
) -> Any:
    from mm_assist.platform.match import match_template

    path = (root or ROOT) / rel
    if frame is None or not path.is_file():
        return None
    return match_template(frame, path, threshold=threshold, roi=roi)


def _close_rels_for(node: str) -> tuple[Path, ...]:
    """这一层该点哪张关闭/返回图。製作壳禁扫设施 back，避免点进木材卡。"""
    n = str(node or "").strip()
    if n == "overlay":
        return (Path("knowledge") / "templates" / "menu" / "icon_close_x.png",)
    g = graph_node(n)
    if is_craft_shell(n):
        return (Path("knowledge") / "templates" / "craft_hub" / "close_x.png",)
    if g in ("facility", "facility_recipe"):
        return (Path("knowledge") / "templates" / "facility" / "back.png",)
    if g == "inventory":
        return (Path("knowledge") / "templates" / "inventory" / "close.png",)
    if g == "profile":
        return (Path("knowledge") / "templates" / "profile" / "close.png",)
    if g == "life_grid":
        return (
            Path("knowledge") / "templates" / "life_grid" / "close.png",
            Path("knowledge") / "templates" / "profile" / "close.png",
        )
    if g in ("gather_detail", "gather_skill"):
        return (Path("knowledge") / "templates" / "gather_detail" / "close.png",)
    if g == "map_dunbarton":
        return (Path("knowledge") / "templates" / "map" / "close_x.png",)
    if g == "system_menu":
        return (Path("knowledge") / "templates" / "menu" / "icon_close_x.png",)
    return tuple(p for p in CLOSE_TEMPLATES if "facility" not in p.as_posix())


def locate_close(
    frame: Any,
    *,
    root: Path | None = None,
    allow_ocr: bool = True,
    node: str = "",
) -> tuple[int, int] | None:
    """先关闭/返回模板，再右上角 OCR「關閉」。"""
    for rel in _close_rels_for(node):
        xy = _hit_xy(_match(frame, rel, root=root))
        if xy is not None:
            return xy
    if not allow_ocr or frame is None:
        return None
    try:
        from mm_assist.perception.ocr import ocr_roi
    except Exception:
        return None
    try:
        lines = ocr_roi(frame, CLOSE_OCR_ROI, min_confidence=0.45)
    except Exception:
        return None
    for line in lines:
        text = str(getattr(line, "text", "") or "")
        if not any(n in text for n in CLOSE_OCR_NEEDLES):
            continue
        box = getattr(line, "box", None) or []
        xs = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
        ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
        if xs and ys:
            return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
        return None
    return None


def facility_page_visible(
    frame: Any, *, expected: str | None = None, root: Path | None = None
) -> bool:
    if frame is None:
        return False
    try:
        from mm_assist.perception.process_sense import detect_facility_page
    except Exception:
        return False
    try:
        return bool(detect_facility_page(frame, expected=expected, root=root).get("ok"))
    except Exception:
        return False


def currency_wallet_visible(frame: Any, *, root: Path | None = None) -> bool:
    if frame is None:
        return False
    try:
        from mm_assist.perception.item_sense import compact_zh
        from mm_assist.perception.ocr import ocr_roi
    except Exception:
        return False
    try:
        lines = ocr_roi(frame, (0.00, 0.00, 0.55, 0.16), min_confidence=0.32)
    except Exception:
        return False
    blob = compact_zh("".join(getattr(ln, "text", "") or "" for ln in lines))
    return "持有的貨幣" in blob or "持有的货币" in blob


def craft_hub_visible(frame: Any, *, root: Path | None = None) -> bool:
    if frame is None:
        return False
    try:
        from mm_assist.perception.process_sense import _any_hub_card
    except Exception:
        return False
    try:
        return bool(_any_hub_card(frame, root=root))
    except Exception:
        return False


def gather_detail_visible(frame: Any, *, root: Path | None = None) -> bool:
    """详情页只认「尋找附近」图标。OCR 会在菜单/加工页空转几十秒。"""
    if frame is None:
        return False
    try:
        hit = _match(
            frame,
            Path("knowledge") / "templates" / "gather_detail" / "find_nearby.png",
            root=root,
            threshold=0.70,
            roi=(0.0, 0.38, 1.0, 0.60),
        )
        return _hit_xy(hit) is not None
    except Exception:
        return False


def skill_overview_visible(
    frame: Any, *, skill_id: str = "", root: Path | None = None
) -> bool:
    """点了技能格、尚未点详情物：tiers 列表已出。"""
    if frame is None or gather_detail_visible(frame, root=root):
        return False
    sid = str(skill_id or "").strip()
    if not sid:
        return False
    try:
        from mm_assist.perception.item_sense import items_for_skill, locate_skill_item
    except Exception:
        return False
    for name in items_for_skill(sid, root=root)[:5]:
        try:
            if locate_skill_item(frame, skill_id=sid, name=name, root=root).get("ok"):
                return True
        except Exception:
            continue
    return False


def _ocr_blob(frame: Any, roi: tuple[float, float, float, float]) -> str:
    if frame is None:
        return ""
    try:
        from mm_assist.perception.item_sense import compact_zh
        from mm_assist.perception.ocr import ocr_roi
    except Exception:
        return ""
    try:
        lines = ocr_roi(frame, roi, min_confidence=0.35)
    except Exception:
        return ""
    return compact_zh("".join(str(getattr(ln, "text", "") or "") for ln in lines))


def system_menu_visible(frame: Any, *, root: Path | None = None) -> bool:
    """三角标认菜单。三张不全时两张也算（製作图常漏，否则会当成加工页不点）。"""
    if frame is None:
        return False
    try:
        hits = sum(
            1
            for rel in _MENU_TRIPLE
            if _hit_xy(_match(frame, rel, root=root, threshold=0.70)) is not None
        )
        return hits >= 2
    except Exception:
        return False


def profile_shell_visible(frame: Any, *, root: Path | None = None) -> bool:
    if frame is None:
        return False
    try:
        blob = _ocr_blob(frame, PROFILE_TAB_ROI)
        hits = sum(1 for n in PROFILE_TAB_NEEDLES if n and n in blob)
        if hits >= 2:
            return True
        return (
            _hit_xy(
                _match(
                    frame,
                    Path("knowledge") / "templates" / "profile" / "life_tab.png",
                    root=root,
                    threshold=0.55,
                )
            )
            is not None
        )
    except Exception:
        return False


def life_grid_visible(frame: Any, *, root: Path | None = None) -> bool:
    if frame is None:
        return False
    try:
        title = _ocr_blob(frame, LIFE_TITLE_ROI)
        if any(n in title for n in LIFE_TITLE_NEEDLES):
            return True
        for node, rel in _ID_TEMPLATES:
            if node != "life_grid":
                continue
            if _hit_xy(_match(frame, rel, root=root, threshold=0.70)) is not None:
                return True
    except Exception:
        return False
    return False


def detect_gather_skill_id(frame: Any, *, root: Path | None = None) -> str:
    if frame is None or gather_detail_visible(frame, root=root):
        return ""
    try:
        from mm_assist.perception.item_sense import iter_tier_items
    except Exception:
        return ""
    seen: set[str] = set()
    try:
        rows = iter_tier_items(root=root)
    except Exception:
        return ""
    for row in rows:
        sid = str(row.get("skill_id") or "").strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        if skill_overview_visible(frame, skill_id=sid, root=root):
            return sid
    return ""


def recipe_action_state(frame: Any, *, root: Path | None = None) -> dict[str, Any] | None:
    if frame is None:
        return None
    try:
        from mm_assist.perception.process_sense import locate_process_button
    except Exception:
        return None
    try:
        hit = locate_process_button(frame, root=root)
    except Exception:
        return None
    if not isinstance(hit, dict) or hit.get("ok") is not True:
        return None
    label = str(hit.get("label") or "")
    if label not in ("前往加工", "加工"):
        return None
    if hit.get("at_facility") is False or label == "前往加工":
        stance = "away"
    elif hit.get("at_facility") is True or label == "加工":
        stance = "here"
    else:
        stance = ""
    return {"label": label, "stance": stance}


_CRAFT_NODE_TO_TAB = {
    "craft_quick": "quick",
    "craft_hub": "process",
    "craft_make": "make",
    "craft_alchemy": "alchemy",
}
_CRAFT_TAB_TO_NODE = {v: k for k, v in _CRAFT_NODE_TO_TAB.items()}


def detect_craft_tab(frame: Any, view: Any = None, *, root: Path | None = None) -> str:
    from mm_assist.life_plan.node_inspect import _craft_node_from_view
    current = _craft_node_from_view(view)
    if current:
        return _CRAFT_NODE_TO_TAB[current]
    if frame is not None:
        from mm_assist.life_plan.node_inspect import _selected_craft_tab

        node = _selected_craft_tab(frame)
        tab = _CRAFT_NODE_TO_TAB.get(node, "")
        if tab:
            return tab
    return ""


def _refine_menu1(frame: Any, *, root: Path | None) -> str:
    if gather_detail_visible(frame, root=root):
        return "gather_detail"
    if recipe_action_state(frame, root=root) is not None:
        return "facility_recipe"
    if facility_page_visible(frame, root=root):
        return "facility"
    if detect_gather_skill_id(frame, root=root):
        return "gather_skill"
    if system_menu_visible(frame, root=root):
        return "system_menu"
    tab = detect_craft_tab(frame, root=root)
    if tab:
        return _CRAFT_TAB_TO_NODE.get(tab, "craft_hub")
    if craft_hub_visible(frame, root=root):
        return "craft_hub"
    if life_grid_visible(frame, root=root):
        return "life_grid"
    if currency_wallet_visible(frame, root=root):
        return "currency_wallet"
    for node, rel in _ID_TEMPLATES:
        if _hit_xy(_match(frame, rel, root=root, threshold=0.70)) is not None:
            return node
    return "unknown"


def _hud_entry_ok(frame: Any, entry_id: str, *, root: Path | None) -> bool:
    if frame is None:
        return False
    try:
        from mm_assist.perception.hud_entry_locator import locate_hud_entry
    except Exception:
        return False
    try:
        cap = json.loads(
            ((root or ROOT) / "knowledge" / "ui" / "contracts" / "main_hud_entry_capture.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return False
    entry = next((e for e in cap.get("entries") or [] if e.get("id") == entry_id), None)
    if not isinstance(entry, dict):
        return False
    loc = locate_hud_entry(entry, frame, root=root)
    return loc.get("feature_ok") is True


def _main_hud_confirmed(frame: Any, *, root: Path | None) -> bool:
    return any(
        _hud_entry_ok(frame, eid, root=root) for eid in ("menu", "bag", "craft", "stop")
    )


@trace_navigation
def resolve_navigation(frame: Any = None, *, reader: Any = None) -> UiWhere:
    """导航即时观测；不等待状态栏显示或 inspect 文件刷新。"""
    from types import SimpleNamespace
    from time import monotonic
    from mm_assist.memory.readers.ui_stack import shared_ui_stack_reader
    from mm_assist.life_plan.node_inspect import compute_inspect

    if reader is None:
        reader = shared_ui_stack_reader()
    t0 = monotonic()
    before = reader.read(force=True)
    t1 = monotonic()
    if not before.ready or frame is None:
        return UiWhere("unknown", None, None, None, "direct_pending")
    if input_blocking_layers(before.names):
        return UiWhere("unknown", None, None, None, "direct_input_blocked",
                       stack_names=tuple(before.names), stack_ready=True)
    view = SimpleNamespace(ui_stack_names=before.names, ui_stack_ready=before.ready,
                           ui_stack_empty_ok=before.empty_confirmed,
                           craft_hub_category=before.craft_category,
                           ui_stack_ts=getattr(before, "sampled_ts", 0.0),
                           ui_stack_error=getattr(before, "error", ""),
                           profile_category=getattr(before, "profile_category", None),
                           profile_identity=getattr(before, "profile_identity", None),
                           craft_current_category=getattr(before, "craft_current_category", None),
                           craft_identity=getattr(before, "craft_identity", None),
                           map_identity=getattr(before, "map_identity", None),
                           map_view=getattr(before, "map_view", None))
    # The inventory instance and its known shell companions already identify
    # the page. Avoid OCR of every item-card/obtain-method region on each poll.
    # Additional instances still take the full overlay recognition path.
    from mm_assist.memory.readers.ui_stack import INVENTORY_NAMES
    simple_inventory = (stack_node(list(before.names), ready=True) == "inventory"
        and set(before.names) <= (INVENTORY_NAMES | {
            "SizeAdjustmentSender", "SideNavigationUIInstance", "SideNavigationUIState",
            "UILayerHideState", "HudUIStepState"}))
    rec = {"node": "inventory"} if simple_inventory else compute_inspect(frame, view=view)
    t2 = monotonic()
    after = reader.read(force=True)
    t3 = monotonic()
    if not after.ready or stack_compare_key(before.names) != stack_compare_key(after.names):
        return UiWhere("unknown", None, None, None, "direct_transition")
    if rec.get("map_mem"):
        from mm_assist.life_plan.node_inspect import _map_node_from_view
        after_view = SimpleNamespace(ui_stack_names=after.names, ui_stack_ready=after.ready,
            ui_stack_ts=getattr(after, "sampled_ts", 0.0),
            ui_stack_error=getattr(after, "error", ""),
            map_identity=getattr(after, "map_identity", None),
            map_view=getattr(after, "map_view", None))
        if (not _map_node_from_view(after_view)
                or after_view.map_identity != view.map_identity
                or after_view.map_view != view.map_view):
            return UiWhere("unknown", None, None, None, "direct_transition")
    if rec.get("profile_mem"):
        # All four tabs share class names: compare current association and value too.
        from mm_assist.life_plan.node_inspect import _profile_node_from_view
        after_view = SimpleNamespace(ui_stack_names=after.names, ui_stack_ready=after.ready,
            ui_stack_ts=getattr(after, "sampled_ts", 0.0),
            profile_category=getattr(after, "profile_category", None),
            profile_identity=getattr(after, "profile_identity", None))
        if (not _profile_node_from_view(after_view)
                or after_view.profile_category != view.profile_category
                or after_view.profile_identity != view.profile_identity):
            return UiWhere("unknown", None, None, None, "direct_transition")
    if rec.get("craft_mem"):
        from mm_assist.life_plan.node_inspect import _craft_node_from_view
        after_view = SimpleNamespace(ui_stack_names=after.names, ui_stack_ready=after.ready,
            ui_stack_ts=getattr(after, "sampled_ts", 0.0), ui_stack_error=getattr(after, "error", ""),
            craft_current_category=getattr(after, "craft_current_category", None),
            craft_identity=getattr(after, "craft_identity", None))
        if (not _craft_node_from_view(after_view)
                or after_view.craft_current_category != view.craft_current_category
                or after_view.craft_identity != view.craft_identity):
            return UiWhere("unknown", None, None, None, "direct_transition")
    names = list(after.names)
    node = str(rec.get("node") or "unknown")
    if node == "profile" and rec.get("source") == "stack":
        node = "unknown"  # The profile instance can precede its visible contents.
    hud_stack = bool(
        after.ready
        and after.empty_confirmed
        and not names
        and not stack_left_overlay(names)
    )
    if hud_stack and (not node or node == "unknown"):
        node = "main_hud"
    elif node == "main_hud" and not hud_stack and not _main_hud_confirmed(frame, root=None):
        node = "unknown"
    return UiWhere(node, int(left_main_hud(names)), int(stack_dlg(names) or node == "dialog"), None,
                   "direct+mem" if rec.get("craft_mem") else "direct",
                   skill_id=node.split(":", 1)[1] if node.startswith("gather_skill:") else "",
                   facility_id=facility_id_of(node), stack_names=tuple(names), stack_ready=True,
                   cost_ms={"stack_before": round((t1-t0)*1000, 1), "recognize": round((t2-t1)*1000, 1),
                            "stack_after": round((t3-t2)*1000, 1)},
                   stack_sampled_ts=min(float(getattr(before, "sampled_ts", 0.0)),
                                        float(getattr(after, "sampled_ts", 0.0))))


def resolve_ui_node(
    *,
    frame: Any = None,
    root: Path | None = None,
    view: Any = None,
) -> UiWhere:
    """先读状态栏界面参数。不截屏扫图。不改写四参。"""
    v = view if view is not None else read_state_ui(root=root)

    def pack(
        node: str,
        source: str,
        *,
        menu: int | None = None,
        dlg: int | None = None,
        stp: int | None = None,
        skill_id: str = "",
        craft_tab: str = "",
        process_stance: str = "",
        overlay: str = "",
        facility_id: str = "",
        stack_names: tuple[str, ...] = (),
        stack_ready: bool = False,
    ) -> UiWhere:
        return UiWhere(
            node=node,
            menu=menu,
            dlg=dlg,
            stp=stp,
            source=source,
            skill_id=skill_id,
            craft_tab=craft_tab,
            process_stance=process_stance,
            overlay=overlay,
            facility_id=facility_id,
            stack_names=stack_names,
            stack_ready=stack_ready,
        )

    if v is None:
        return pack("unknown", "mem_pending")
    _ = frame

    menu, dlg, stp = _bits_from_view(v)
    names, ready = _stack_from_view(v)
    from mm_assist.life_plan.node_inspect import _profile_node_from_view, _map_node_from_view, _craft_node_from_view
    memory_node = _profile_node_from_view(v) or _map_node_from_view(v) or _craft_node_from_view(v)
    sn = memory_node or stack_node(list(names or []), ready=bool(ready))
    node, tab = node_from_mem(v)
    insp = read_inspect_page(root=root)
    rec = read_inspect_record(root=root)
    zh = str((rec or {}).get("zh") or "")
    if not memory_node and ("主界面-待机" in zh or "主界面-动作中" in zh):
        node = "main_hud"
    fid = ""
    if insp and not memory_node:
        gi = graph_node(insp)
        allow = False
        if ready and sn:
            fine = (
                sn.startswith("fine:")
                or sn == "overlay"
                or sn in ("craft_shell", "craft_hub", "life_grid")
                # NewProfileUIInstance is shared by all life-skill subpages.
                or (sn == "profile" and gi in ("profile", "life_grid", "gather_skill", "gather_detail"))
            )
            same = stack_compare_key(list(rec.get("stack_names") or [])) == stack_compare_key(
                list(names or [])
            )
            allow = fine and same and bool(rec)
        else:
            allow = False
        if allow and gi in KNOWN_NODES and gi != "main_hud":
            gm = graph_node(node)
            if gi != "currency_wallet" or gm == "currency_wallet":
                node = insp
                fid = facility_id_of(insp)
                tab = tab or ""
    src = "mem" if memory_node else ("inspect" if insp and node == insp else "mem")
    overlay = "currency_wallet" if graph_node(node) == "currency_wallet" else ""
    return pack(
        node,
        src,
        menu=menu,
        dlg=dlg,
        stp=stp,
        craft_tab=tab,
        overlay=overlay,
        facility_id=fid,
        stack_names=tuple(str(x) for x in (names or [])),
        stack_ready=bool(ready),
    )


def ui_settled(*, root: Path | None = None, resolve: Any = None) -> bool:
    """栈就绪且 node 在已知集合（非 unknown）则 True。"""
    resolver = resolve or (lambda frame=None: resolve_ui_node(frame=frame, root=root))
    try:
        where = resolver(None)
    except TypeError:
        try:
            where = resolver()
        except Exception:
            return False
    except Exception:
        return False
    if where is None:
        return False
    if not bool(getattr(where, "stack_ready", False)):
        return False
    node = graph_node(str(getattr(where, "node", "") or ""))
    return bool(node) and node != "unknown" and node in KNOWN_NODES


def is_open_menu_action(action: str) -> bool:
    """图边 label 是「菜单」，文案还有打开菜单/点菜单。"""
    a = str(action or "").strip()
    return a in {"打开菜单", "菜单", "点菜单"}


def skip_click(node: str, action: str, *, menu: int | None = None) -> bool:
    """已在目标页则不要再点入口。menu=1 禁止再点 HUD 菜单。"""
    n = str(node or "")
    a = str(action or "")
    if is_open_menu_action(a):
        if menu == 1:
            return True
        return n in {
            "system_menu",
            "profile",
            "life_grid",
            "gather_skill",
            "gather_detail",
            "inventory",
            "facility",
            "facility_recipe",
            "currency_wallet",
            "map_dunbarton",
        } or is_craft_shell(n) or n.startswith("facility") or n.startswith("gather_")
    if a == "技能大格" or a in {
        "点日常採集",
        "点採集藥草",
        "点伐木",
        "点採礦",
        "点剪羊毛",
        "点收割",
        "点鋤地",
        "点昆蟲採集",
        "点釣魚",
    }:
        return graph_node(n) != "life_grid"
    if a in {"打开角色", "角色", "点角色"}:
        return n in {"profile", "life_grid", "gather_skill", "gather_detail"}
    if a in {"打开生活技能", "生活技能", "生活技能 Tab", "点生活技能"}:
        return n in {"life_grid", "gather_skill", "gather_detail"}
    if a in {"打开製作", "打开制作", "製作", "点製作"}:
        return n in {"craft_hub", "facility", "facility_recipe", "currency_wallet"} or n.startswith(
            "facility"
        )
    if a.startswith("点") and a.endswith("卡"):
        return graph_node(n) in {"facility", "facility_recipe"} or n.startswith("facility")
    return False


def gather_node_ok(node: str) -> bool:
    return str(node or "unknown") in GATHER_NODES


def feed_node_ok(node: str) -> bool:
    return str(node or "unknown") in FEED_NODES
