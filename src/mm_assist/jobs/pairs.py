"""成对采证辅助（方法 B · docs/TESTING.md）。

`open_perceive_sample` / `open_test_job(method=B)` → status=`await_start`。
须用户在状态栏点【开始采集】后才执行；本模块不 Sleep、不连开两态。
成对：每一态分别 open 一次并走完按钮。

A4 空转熔断为协议层；感知标定走 B；勿把标定塞进每次 LIVE A。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mm_assist.jobs.bar import open_test_job

# pair_kind → 两帧 role / need_ui / title（零键鼠只读）
_PAIR_SPECS: dict[str, list[dict[str, str]]] = {
    "stop": [
        {
            "pair_role": "moving",
            "need_ui": "非战斗界面·动作中（停止钮可见 stp=1）",
            "title": "成对·动作中",
        },
        {
            "pair_role": "stopped",
            "need_ui": "非战斗界面·非动作（停止钮消失 stp=0）",
            "title": "成对·非动作",
        },
    ],
    "combat": [
        {
            "pair_role": "noncombat",
            "need_ui": "非战斗界面（清主界面；勿信旧 cbt/cs；记参数行差分）",
            "title": "成对·非战斗",
        },
        {
            "pair_role": "combat",
            "need_ui": "战斗界面（勿信旧 cbt/cs；记与非战斗帧差分字段）",
            "title": "成对·战斗",
        },
    ],
    "map": [
        {
            "pair_role": "map_closed",
            "need_ui": "已关地圖回主界面（map=0）；勿停在背包/对话/製作界面",
            "title": "成对·关图",
        },
        {
            "pair_role": "map_open",
            "need_ui": "已开地圖（map=1；可见地图画布）；人物界面须关（勿 prf=1）",
            "title": "成对·开图",
        },
    ],
    "map_profile": [
        {
            "pair_role": "map_open",
            "need_ui": "已开地圖（map=1；人物界面关 prf≠1）；勿停在背包/对话",
            "title": "成对·地图",
        },
        {
            "pair_role": "profile_open",
            "need_ui": "已开人物界面（prf=1；地圖关 map=0）；勿停在对话/製作界面",
            "title": "成对·人物",
        },
    ],
    # MapType 层采证（逐帧 B；须 map_open 真开图再信 mt）
    "map_layers": [
        {
            "pair_role": "map_closed",
            "need_ui": "已关地圖回非战斗界面（清界面）；对照基线",
            "title": "图层·关图HUD",
        },
        {
            "pair_role": "map_town",
            "need_ui": "开图·城镇详图（MapType≈2；如杜巴城内；勿欧拉总图）",
            "title": "图层·城镇",
        },
        {
            "pair_role": "map_continent",
            "need_ui": "开图·欧拉大陆总图（MapType≈1；大陆画布；顶栏歐拉大陸）",
            "title": "图层·大陆",
        },
        {
            "pair_role": "map_dungeon",
            "need_ui": "开图·地下城层（MapType≈3；若当前进得去）",
            "title": "图层·地下城",
        },
        {
            "pair_role": "map_moongate",
            "need_ui": "开图·月亮门层（MapType≈4；若当前进得去）",
            "title": "图层·月亮门",
        },
    ],
    # 菜单图标候选对拍（MapMarker/Shop；须 probe_menu=1；成对差分 ≠ 够用）
    "menu_icons": [
        {
            "pair_role": "hud_menu0",
            "need_ui": "非战斗界面：menu=0 inv=0；可见菜单入口；勿开背包/对话/製作界面/地圖",
            "title": "菜单图·清HUD",
        },
        {
            "pair_role": "esc_root",
            "need_ui": "已开菜单：可见角色/背包/製作/交易所等；menu=1",
            "title": "菜单图·ESC根",
        },
        {
            "pair_role": "life_leaves",
            "need_ui": "人物界面→生活技能列表：可见伐木/采矿/药草等（勿停在主界面）",
            "title": "菜单图·生活叶子",
        },
        {
            "pair_role": "craft_tabs",
            "need_ui": "製作界面·加工档可见（金屬/木材/皮革/布料/藥品/食材）；menu=1",
            "title": "菜单图·加工页签",
        },
        {
            "pair_role": "inv_open",
            "need_ui": "背包已开：inv=1 且 menu=1；可见背包界面",
            "title": "菜单图·背包开",
        },
    ],
    # 制作关↔开（hub 落 panel_probe；不上栏；≠够用）
    "craft": [
        {
            "pair_role": "craft_closed",
            "need_ui": "已关製作界面回非战斗界面（清主界面；勿停菜单/背包/对话/地圖）",
            "title": "成对·关制作",
        },
        {
            "pair_role": "craft_open",
            "need_ui": "已开製作界面·加工档（可见金屬等六卡）；勿停菜单根；记 hub 与 nav 差分",
            "title": "成对·开制作",
        },
    ],
    # 对话关↔开（粗 dlg 已上栏；icon 坐标未钉；≠够用）
    "dlg": [
        {
            "pair_role": "dlg_closed",
            "need_ui": "非对话主界面（dlg 应 0；清主界面；勿停修理/背包）",
            "title": "成对·非对话",
        },
        {
            "pair_role": "dlg_open",
            "need_ui": "已开 NPC 对话（可见选项/結束；dlg 应 1 且 menu=1）；记参数行与截图像素锚点",
            "title": "成对·对话",
        },
    ],
}

PAIR_KINDS = frozenset(_PAIR_SPECS)


def _utc_pair_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def perceive_pair_job_specs(
    pair_kind: str,
    *,
    pair_id: str | None = None,
) -> list[dict[str, Any]]:
    """返回各帧 need_ui / title / args（pair_id, pair_role）；不执行、不 Sleep。

    pair_kind ∈ {"stop","combat","map","map_profile","map_layers","menu_icons","craft","dlg"}。
    调用方在用户确认各态后再分别 ``open_test_job`` 或 ``open_perceive_sample``。
    """
    kind = str(pair_kind or "").strip().lower()
    if kind not in _PAIR_SPECS:
        raise ValueError(f"未知 pair_kind={pair_kind!r}；已知 {sorted(PAIR_KINDS)}")
    pid = str(pair_id).strip() if pair_id else _utc_pair_id(kind)
    out: list[dict[str, Any]] = []
    for frame in _PAIR_SPECS[kind]:
        role = frame["pair_role"]
        args: dict[str, Any] = {
            "pair_id": pid,
            "pair_role": role,
        }
        if kind == "menu_icons":
            args["probe_menu"] = 1
        out.append(
            {
                "action": "readonly_ui_fine",
                "method": "B",
                "need_ui": frame["need_ui"],
                "title": frame["title"],
                "args": args,
            }
        )
    return out


def open_perceive_sample(
    *,
    pair_id: str,
    pair_role: str,
    need_ui: str,
    title: str | None = None,
    path: Path | None = None,
    probe_menu: bool = False,
) -> dict[str, Any]:
    """开一帧 method=B readonly_ui_fine → await_start（等人点开始采集）。

    禁止 Sleep 冒充等人；成对第二帧须再 open 并再走按钮。
    """
    role = str(pair_role or "").strip()
    pid = str(pair_id or "").strip()
    if not pid:
        raise ValueError("open_perceive_sample 须提供 pair_id")
    if not role:
        raise ValueError("open_perceive_sample 须提供 pair_role")
    need = str(need_ui or "").strip()
    if not need:
        raise ValueError("open_perceive_sample 须标明 need_ui")
    args: dict[str, Any] = {"pair_id": pid, "pair_role": role}
    if probe_menu:
        args["probe_menu"] = 1
    return open_test_job(
        "readonly_ui_fine",
        method="B",
        need_ui=need,
        title=str(title or f"perceive·{role}"),
        args=args,
        path=path,
    )


# pair_kind → 归档 JSON 文件名（role → stem）
_PAIR_ARCHIVE_STEMS: dict[str, dict[str, str]] = {
    "craft": {"craft_closed": "craft_closed", "craft_open": "craft_open"},
}


def pair_capture_dir(pair_id: str, *, root: Path | None = None) -> Path:
    """成对采证归档目录 ``knowledge/captures/{pair_id}/``。"""
    from mm_assist.config import ROOT

    pid = str(pair_id or "").strip()
    if not pid:
        raise ValueError("pair_capture_dir 须提供 pair_id")
    base = root or ROOT
    return base / "knowledge" / "captures" / pid


def load_pair_job_json(path: Path) -> dict[str, Any]:
    """读单帧归档 job JSON（含 result/panel_probe）。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"归档不存在: {p}")
    import json

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"归档须为 object: {p}")
    return data


def load_craft_pair_jobs(
    pair_id: str,
    *,
    root: Path | None = None,
    closed_path: Path | None = None,
    open_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """加载製作关↔开两帧归档。"""
    if closed_path is not None and open_path is not None:
        return load_pair_job_json(closed_path), load_pair_job_json(open_path)
    cap = pair_capture_dir(pair_id, root=root)
    stems = _PAIR_ARCHIVE_STEMS["craft"]
    closed_p = cap / f"{stems['craft_closed']}.json"
    open_p = cap / f"{stems['craft_open']}.json"
    missing = [str(x) for x in (closed_p, open_p) if not x.is_file()]
    if missing:
        raise FileNotFoundError(
            f"製作成对归档不全 pair_id={pair_id!r}；缺: {', '.join(missing)}"
        )
    return load_pair_job_json(closed_p), load_pair_job_json(open_p)


def validate_craft_pair_jobs(
    closed_job: dict[str, Any],
    open_job: dict[str, Any],
) -> dict[str, Any]:
    """离线对拍製作关↔开；永不 enough=True。"""
    from mm_assist.safety.autofarm_gates import assess_craft_pair_captures

    return assess_craft_pair_captures(closed_job, open_job)


def validate_sample_pair(
    pair_kind: str,
    *,
    pair_id: str | None = None,
    root: Path | None = None,
    closed_path: Path | None = None,
    open_path: Path | None = None,
) -> dict[str, Any]:
    """离线校验成对采证归档（零键鼠 · 不读活内存）。

    pair_kind 目前仅 ``craft``；须两帧 JSON 均在 ``knowledge/captures/{pair_id}/``
    或由 ``closed_path``/``open_path`` 显式指定。
    """
    kind = str(pair_kind or "").strip().lower()
    if kind not in _PAIR_ARCHIVE_STEMS:
        raise ValueError(
            f"validate_sample_pair 未知 pair_kind={pair_kind!r}；"
            f"已知 {sorted(_PAIR_ARCHIVE_STEMS)}"
        )
    pid = str(pair_id or "").strip() or None
    if not pid and (closed_path is None or open_path is None):
        raise ValueError("validate_sample_pair 须提供 pair_id 或 closed_path+open_path")

    out: dict[str, Any] = {
        "ok": False,
        "pair_kind": kind,
        "pair_id": pid,
        "error": None,
    }
    try:
        if kind == "craft":
            closed_job, open_job = load_craft_pair_jobs(
                pid or "",
                root=root,
                closed_path=closed_path,
                open_path=open_path,
            )
            assess = validate_craft_pair_jobs(closed_job, open_job)
            out.update(
                {
                    "ok": True,
                    "pair_id": pid
                    or str(
                        (closed_job.get("args") or {}).get("pair_id")
                        or (open_job.get("args") or {}).get("pair_id")
                        or ""
                    ).strip()
                    or None,
                    "closed_job_id": closed_job.get("id"),
                    "open_job_id": open_job.get("id"),
                    "assess": assess,
                    "pair_ok": bool(assess.get("pair_ok")),
                    "enough": False,
                    "fail_tags": list(assess.get("fail_tags") or []),
                    "open_signal": assess.get("open_signal"),
                }
            )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}:{exc}"
    return out


def format_validate_pair_report(result: dict[str, Any]) -> str:
    """人话短表（stdout；完整 JSON 用 --json）。"""
    lines = [
        "# validate-pair",
        f"pair_kind={result.get('pair_kind')} pair_id={result.get('pair_id')}",
    ]
    if result.get("error"):
        lines.append(f"error: {result['error']}")
        return "\n".join(lines) + "\n"
    assess = result.get("assess") if isinstance(result.get("assess"), dict) else {}
    lines.extend(
        [
            f"pair_ok={result.get('pair_ok')} enough=False（硬闸）",
            f"menu_flip={assess.get('menu_flip')} hub_open_ok={assess.get('hub_open_ok')} "
            f"visual_open_ok={assess.get('visual_open_ok')}",
            f"closed_hub={assess.get('closed_hub')} open_hub={assess.get('open_hub')}",
        ]
    )
    sig = assess.get("open_signal") if isinstance(assess.get("open_signal"), dict) else {}
    if sig:
        lines.append(
            f"open_signal: source={sig.get('source')} trusted={sig.get('trusted')} "
            f"fail_open={sig.get('fail_open')}"
        )
    tags = result.get("fail_tags") or []
    if tags:
        lines.append("fail_tags: " + ", ".join(str(t) for t in tags))
    return "\n".join(lines) + "\n"
