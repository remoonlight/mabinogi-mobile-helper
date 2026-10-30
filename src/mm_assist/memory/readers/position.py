"""IL2CPP 只读坐标：优先 MapComponent.PlayerPosition，备选 Transform XZ。

配置：`knowledge/ui/config/il2cpp_position.json`
探测：`python -m mm_assist.tools.il2cpp_position_probe`
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mm_assist.memory.il2cpp import (
    module_base,
    parse_int,
    plausible_xy,
    read_u8,
    read_xy,
    read_xyz,
    resolve_klass,
    scan_klass_objects_multi,
)
from mm_assist.memory.process import close_handle, find_game_pid, open_process, read_ptr, read_identity

log = logging.getLogger(__name__)

from mm_assist.knowledge_paths import POSITION_JSON

DEFAULT_REL = POSITION_JSON
from mm_assist.config import ROOT as _REPO_ROOT

_POS_SEED_FILE = _REPO_ROOT / "logs" / "il2cpp_pos_seeds.json"

# map_component 与 transform 偏差过大 → 视为绑错实例（米）
_SOURCE_MISMATCH_M = 8.0
# primary 连续不动、transform 明显移动 → 丢缓存重绑
_STUCK_EPS_M = 0.35
_MOVE_EPS_M = 1.5
_STUCK_TICKS = 3
# 堆扫常见占位/垃圾实例贴原点，但仍过 plausible_xy（如 0.05）
_ORIGIN_JUNK_M = 0.5
_SCAN_MAX_HITS = 24
# 冷启动无种子时单源堆扫常要 8s+；4.5s 会稳定 instance_not_found
_RESOLVE_WALL_S = 12.0
# 绑定坐标长期不变且与粗定位也无对齐时 → 重扫（站桩正确时粗定位会一起静，不误伤）
_FREEZE_REBIND_S = 5.0
_COLD_MOVE_WAIT_S = 0.2
_COLD_MOVE_EPS_M = 0.4


def xy_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def near_origin(
    xy: tuple[float, float], *, eps_m: float = _ORIGIN_JUNK_M
) -> bool:
    return abs(xy[0]) < eps_m and abs(xy[1]) < eps_m


def filter_origin_junk(
    candidates: list[tuple[int, float, float]],
    *,
    eps_m: float = _ORIGIN_JUNK_M,
) -> list[tuple[int, float, float]]:
    """有非原点候选时丢掉贴原点垃圾；否则原样返回。"""
    if not candidates:
        return candidates
    kept = [c for c in candidates if not near_origin((c[1], c[2]), eps_m=eps_m)]
    return kept or candidates


def pick_closest_xy(
    candidates: list[tuple[int, float, float]],
    ref_xy: tuple[float, float] | None,
) -> int | None:
    """多实例时选距 ref 最近的 obj；无 ref 则取第一个。"""
    if not candidates:
        return None
    if ref_xy is None:
        return candidates[0][0]
    best = min(candidates, key=lambda c: xy_dist((c[1], c[2]), ref_xy))
    return best[0]


def pick_player_xy(
    candidates: list[tuple[int, float, float]],
    ref_xy: tuple[float, float] | None,
    *,
    agree_xy: tuple[float, float] | None = None,
) -> int | None:
    """选玩法坐标：剔原点；优先跟 agree（粗定位）对齐；再跟上一帧 ref。

    禁止「离原点最近」——会误绑野外静止实体（实机曾钉死 -49,46）。
    """
    pool = filter_origin_junk(candidates)
    if not pool:
        return None
    if agree_xy is not None and not near_origin(agree_xy):
        return pick_closest_xy(pool, agree_xy)
    if ref_xy is not None and not near_origin(ref_xy):
        return pick_closest_xy(pool, ref_xy)
    if len(pool) == 1:
        return pool[0][0]
    return None


def sources_disagree(
    primary_xy: tuple[float, float],
    transform_xy: tuple[float, float],
    *,
    mismatch_m: float = _SOURCE_MISMATCH_M,
) -> bool:
    return xy_dist(primary_xy, transform_xy) > mismatch_m

_DEFAULT_SOURCES = {
    "map_component": {
        "typeinfo_rva": 0x7B571C8,
        "xy_offset": 0x10,
        "layout": "xy_ff",
    },
    "transform_xz": {
        "typeinfo_rva": 0x7B42D08,
        "xyz_offset": 0x10,
        "layout": "xz_fff",
    },
    "inaccurate": {
        "typeinfo_rva": 0x7B3EFB8,
        "xy_offset": 0x10,
        "has_offset": 0x1C,
        "layout": "xy_ff",
    },
}


def default_path(knowledge_root: Path | None = None) -> Path:
    if knowledge_root is None:
        return POSITION_JSON
    return Path(knowledge_root) / "ui" / "config" / "il2cpp_position.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or default_path()
    cfg: dict[str, Any] = {
        "module": "GameAssembly.dll",
        "primary": "map_component",
        "sources": {k: dict(v) for k, v in _DEFAULT_SOURCES.items()},
    }
    if not p.is_file():
        return cfg
    raw = json.loads(p.read_text(encoding="utf-8"))
    cfg["module"] = str(raw.get("module") or cfg["module"])
    cfg["primary"] = str(raw.get("primary") or cfg["primary"])
    src_in = raw.get("sources") or {}
    for name, spec in src_in.items():
        if not isinstance(spec, dict):
            continue
        base = dict(cfg["sources"].get(name) or {})
        for k, v in spec.items():
            if k in ("typeinfo_rva", "xy_offset", "xyz_offset", "has_offset"):
                base[k] = parse_int(v)
            else:
                base[k] = v
        cfg["sources"][name] = base
    # 兼容旧 typeinfo_rva 块
    ti = raw.get("typeinfo_rva") or {}
    if "MapComponent" in ti and "map_component" in cfg["sources"]:
        cfg["sources"]["map_component"]["typeinfo_rva"] = parse_int(ti["MapComponent"])
    if "ControlObjectTransformComponent" in ti and "transform_xz" in cfg["sources"]:
        cfg["sources"]["transform_xz"]["typeinfo_rva"] = parse_int(
            ti["ControlObjectTransformComponent"]
        )
    if "ControlObjectLocationInaccurateUpTo1mComponent" in ti:
        cfg["sources"]["inaccurate"]["typeinfo_rva"] = parse_int(
            ti["ControlObjectLocationInaccurateUpTo1mComponent"]
        )
    return cfg


@dataclass
class Il2CppPos:
    x: float
    y: float
    addr: int
    pid: int
    source: str
    sc_addr: int | None = None
    age_ms: float = 0.0
    z: float | None = None


class Il2CppPositionReader:
    """多源坐标：map_component → transform_xz → inaccurate。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_path()
        self._cfg = load_config(self.path)
        self._pid: int | None = None
        self._h = None
        self._ga: int | None = None
        self._klasses: dict[str, int] = {}
        self._obj: dict[str, int] = {}  # source -> object addr
        self._last_err: str = ""
        self._lock = threading.RLock()
        self._resolving = False
        self._bg: threading.Thread | None = None
        self._closed = False
        self._prev_primary_xy: tuple[float, float] | None = None
        self._prev_transform_xy: tuple[float, float] | None = None
        self._stuck_n = 0
        self._need_rescan = False
        self._last_good: Il2CppPos | None = None
        self._freeze_xy: tuple[float, float] | None = None
        self._freeze_since: float = 0.0

    @property
    def last_error(self) -> str:
        return self._last_err

    @property
    def resolving(self) -> bool:
        return self._resolving

    @property
    def cached(self) -> bool:
        return bool(self._obj)

    def reload(self) -> None:
        self._cfg = load_config(self.path)
        self.close()
        self._closed = False

    def close(self) -> None:
        self._closed = True
        bg = self._bg
        if bg is not None and bg.is_alive() and threading.current_thread() is not bg:
            bg.join(timeout=2.0)
        with self._lock:
            if self._h is not None:
                try:
                    close_handle(self._h)
                except Exception:
                    pass
            self._h = None
            self._pid = None
            self._ga = None
            self._klasses.clear()
            self._obj.clear()
            self._bg = None
            self._prev_primary_xy = None
            self._prev_transform_xy = None
            self._stuck_n = 0
            self._need_rescan = False
            self._last_good = None
            self._freeze_xy = None
            self._freeze_since = 0.0

    def _ensure_proc(self, hwnd: int | None = None) -> bool:
        if self._h is not None and self._pid:
            try:
                read_identity(self._h, self._pid)
            except OSError as exc:
                self.close()
                self._last_err = str(exc)
                return False
        pid = find_game_pid(hwnd=hwnd)
        if not pid:
            self._last_err = "process_not_found"
            with self._lock:
                if self._h is not None:
                    try:
                        close_handle(self._h)
                    except Exception:
                        pass
                self._h = None
                self._pid = None
                self._ga = None
                self._klasses.clear()
                self._obj.clear()
            return False
        if self._h is None or self._pid != pid:
            with self._lock:
                if self._h is not None:
                    try:
                        close_handle(self._h)
                    except Exception:
                        pass
                self._h = None
                self._ga = None
                self._klasses.clear()
                self._obj.clear()
                try:
                    self._h = open_process(pid)
                    self._pid = pid
                    self._closed = False
                except OSError as e:
                    self._last_err = f"open_failed:{e}"
                    return False
        return True

    def _resolve_klasses(self) -> bool:
        assert self._h is not None and self._pid is not None
        ga = module_base(self._pid, str(self._cfg["module"]))
        if not ga:
            self._last_err = "gameassembly_not_found"
            return False
        self._ga = ga
        self._klasses.clear()
        for name, spec in (self._cfg.get("sources") or {}).items():
            rva = int(spec.get("typeinfo_rva") or 0)
            klass = resolve_klass(self._h, ga, rva)
            if klass:
                self._klasses[name] = klass
        if not self._klasses:
            self._last_err = "typeinfo_uninitialized"
            return False
        return True

    def _order(self) -> list[str]:
        primary = str(self._cfg.get("primary") or "map_component")
        names = list((self._cfg.get("sources") or {}).keys())
        ordered = [primary] + [n for n in names if n != primary]
        # 稳定优先序
        prefer = ["map_component", "transform_xz", "inaccurate"]
        rest = [n for n in ordered if n not in prefer]
        return [n for n in prefer if n in ordered] + rest

    def _read_from_obj(self, name: str, obj: int) -> Il2CppPos | None:
        assert self._h is not None and self._pid is not None
        klass = self._klasses.get(name)
        if not klass or read_ptr(self._h, obj) != klass:
            return None
        spec = self._cfg["sources"][name]
        layout = str(spec.get("layout") or "xy_ff")
        if layout == "xz_fff":
            off = int(spec.get("xyz_offset") or 0x10)
            xyz = read_xyz(self._h, obj + off)
            if not xyz:
                return None
            x, _y, z = xyz
            if not plausible_xy(x, z):
                return None
            return Il2CppPos(
                x=x, y=z, z=_y, addr=obj + off, pid=self._pid, source=name
            )
        off = int(spec.get("xy_offset") or 0x10)
        if "has_offset" in spec:
            has = read_u8(self._h, obj + int(spec["has_offset"]))
            if has is None or has == 0:
                return None
        xy = read_xy(self._h, obj + off)
        if not xy or not plausible_xy(xy[0], xy[1]):
            return None
        return Il2CppPos(
            x=xy[0], y=xy[1], addr=obj + off, pid=self._pid, source=name
        )

    def _read_cached_named(self, name: str) -> Il2CppPos | None:
        obj = self._obj.get(name)
        if not obj:
            return None
        pos = self._read_from_obj(name, obj)
        if pos is None:
            self._obj.pop(name, None)
        return pos

    def _note_primary_stuck(
        self,
        primary_xy: tuple[float, float],
        transform_xy: tuple[float, float] | None,
    ) -> bool:
        """primary 钉死且 transform 在动 → True（应丢 map 缓存）。"""
        prev_p = self._prev_primary_xy
        prev_t = self._prev_transform_xy
        self._prev_primary_xy = primary_xy
        if transform_xy is not None:
            self._prev_transform_xy = transform_xy
        if prev_p is None or transform_xy is None or prev_t is None:
            self._stuck_n = 0
            return False
        primary_still = xy_dist(primary_xy, prev_p) < _STUCK_EPS_M
        transform_moved = xy_dist(transform_xy, prev_t) >= _MOVE_EPS_M
        if primary_still and transform_moved:
            self._stuck_n += 1
        else:
            self._stuck_n = 0
        return self._stuck_n >= _STUCK_TICKS

    def _note_freeze(self, xy: tuple[float, float]) -> bool:
        """同一坐标钉太久 → 可能绑到静止实体，应重扫。"""
        now = time.monotonic()
        if self._freeze_xy is None or xy_dist(xy, self._freeze_xy) >= _STUCK_EPS_M:
            self._freeze_xy = xy
            self._freeze_since = now
            return False
        return (now - self._freeze_since) >= _FREEZE_REBIND_S

    def _collect_cands(
        self, name: str, objs: list[int]
    ) -> list[tuple[int, float, float]]:
        cands: list[tuple[int, float, float]] = []
        for obj in objs:
            if self._closed:
                break
            pos = self._read_from_obj(name, obj)
            if pos:
                cands.append((obj, pos.x, pos.y))
        return cands

    def _pick_by_movement(
        self, name: str, cands: list[tuple[int, float, float]]
    ) -> int | None:
        """冷启动无 agree/ref：短等再读，优先选明显移动的实例。"""
        pool = filter_origin_junk(cands)
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0][0]
        time.sleep(_COLD_MOVE_WAIT_S)
        moved: list[tuple[int, float]] = []
        for obj, x, y in pool:
            if self._closed:
                break
            pos2 = self._read_from_obj(name, obj)
            if not pos2:
                continue
            d = xy_dist((x, y), (pos2.x, pos2.y))
            if d >= _COLD_MOVE_EPS_M:
                moved.append((obj, d))
        if moved:
            return max(moved, key=lambda t: t[1])[0]
        return pool[0][0]

    def _try_cached(self) -> Il2CppPos | None:
        """优先 map_component；与 transform/粗定位严重不符或钉死时丢弃错实例。"""
        map_pos = self._read_cached_named("map_component")
        xf_pos = self._read_cached_named("transform_xz")
        ina_pos = self._read_cached_named("inaccurate")
        xf_xy = (xf_pos.x, xf_pos.y) if xf_pos else None
        ina_xy = (ina_pos.x, ina_pos.y) if ina_pos else None

        def _drop_bad(source: str, xy: tuple[float, float], reason: str) -> None:
            log.info(
                "il2cpp pos drop %s %s xy=(%.2f,%.2f)",
                source,
                reason,
                xy[0],
                xy[1],
            )
            self._obj.pop(source, None)
            self._stuck_n = 0
            self._need_rescan = True
            self._freeze_xy = None

        if ina_xy is not None and not near_origin(ina_xy):
            if map_pos is not None and sources_disagree((map_pos.x, map_pos.y), ina_xy):
                _drop_bad("map_component", (map_pos.x, map_pos.y), "vs inaccurate")
                map_pos = None
            if xf_xy is not None and sources_disagree(xf_xy, ina_xy):
                _drop_bad("transform_xz", xf_xy, "vs inaccurate")
                self._obj.pop("transform_xz", None)
                xf_pos = None
                xf_xy = None

        if map_pos is not None:
            mxy = (map_pos.x, map_pos.y)
            bad = False
            if xf_xy is not None and sources_disagree(mxy, xf_xy):
                log.info(
                    "il2cpp pos drop stale map_component "
                    "map=(%.2f,%.2f) xf=(%.2f,%.2f)",
                    mxy[0],
                    mxy[1],
                    xf_xy[0],
                    xf_xy[1],
                )
                bad = True
            elif self._note_primary_stuck(mxy, xf_xy):
                log.info(
                    "il2cpp pos drop stuck map_component xy=(%.2f,%.2f)",
                    mxy[0],
                    mxy[1],
                )
                bad = True
            elif self._note_freeze(mxy) and (
                ina_xy is None or near_origin(ina_xy)
            ):
                log.info(
                    "il2cpp pos drop frozen map_component xy=(%.2f,%.2f)",
                    mxy[0],
                    mxy[1],
                )
                bad = True
            if bad:
                self._obj.pop("map_component", None)
                self._stuck_n = 0
                self._need_rescan = True
                self._freeze_xy = None
                map_pos = None
            elif near_origin(mxy):
                map_pos = None
            else:
                return map_pos

        if xf_pos is not None and xf_xy is not None:
            if near_origin(xf_xy):
                xf_pos = None
            elif self._note_freeze(xf_xy) and (
                ina_xy is None or near_origin(ina_xy)
            ):
                _drop_bad("transform_xz", xf_xy, "frozen")
            else:
                return xf_pos

        for name in self._order():
            if name in ("map_component", "transform_xz"):
                continue
            pos = self._read_cached_named(name)
            if pos:
                return pos
        return None

    def _resolve_instances(self, *, max_wall_s: float | None = None) -> bool:
        if self._h is None or self._closed:
            self._last_err = "closed" if self._closed else "read_failed"
            return False
        wall = float(max_wall_s if max_wall_s is not None else _RESOLVE_WALL_S)
        t0 = time.perf_counter()
        if self._apply_pos_seeds():
            # 种子可用则先试读；够用就跳过堆扫
            if self._try_cached() is not None:
                log.info("il2cpp pos seeds hit (skip heap) wall=%.2fs", time.perf_counter() - t0)
                self._freeze_xy = None
                self._freeze_since = 0.0
                self._save_pos_seeds()
                return True
        found_any = False
        order = [n for n in self._order() if self._klasses.get(n)]
        klasses = {n: self._klasses[n] for n in order}
        remain = max(0.3, wall - (time.perf_counter() - t0))
        res = scan_klass_objects_multi(
            self._h,
            klasses,
            max_hits=_SCAN_MAX_HITS,
            chunk=4 * 1024 * 1024,
            should_stop=lambda: self._closed,
            max_wall_s=remain,
            resume=True,
        )
        ref_xy = self._prev_primary_xy or self._prev_transform_xy

        ina_xy: tuple[float, float] | None = None
        ina_cands = self._collect_cands("inaccurate", list(res.get("inaccurate") or []))
        ina_pick = pick_player_xy(ina_cands, ref_xy)
        if ina_pick is None and ina_cands:
            ina_pick = self._pick_by_movement("inaccurate", ina_cands)
        if ina_pick is not None:
            pos = self._read_from_obj("inaccurate", ina_pick)
            if pos:
                self._obj["inaccurate"] = ina_pick
                ina_xy = (pos.x, pos.y)
                found_any = True
                log.info(
                    "il2cpp pos source=%s obj=0x%X xy=(%.2f,%.2f) n=%d",
                    "inaccurate",
                    ina_pick,
                    pos.x,
                    pos.y,
                    len(ina_cands),
                )

        xf_xy: tuple[float, float] | None = None
        xf_cands = self._collect_cands(
            "transform_xz", list(res.get("transform_xz") or [])
        )
        xf_pick = pick_player_xy(xf_cands, ref_xy, agree_xy=ina_xy)
        if xf_pick is None and xf_cands:
            xf_pick = self._pick_by_movement("transform_xz", xf_cands)
        if xf_pick is not None:
            pos = self._read_from_obj("transform_xz", xf_pick)
            if pos:
                self._obj["transform_xz"] = xf_pick
                xf_xy = (pos.x, pos.y)
                found_any = True
                log.info(
                    "il2cpp pos source=%s obj=0x%X xy=(%.2f,%.2f) n=%d",
                    "transform_xz",
                    xf_pick,
                    pos.x,
                    pos.y,
                    len(xf_cands),
                )

        map_ref = xf_xy or ina_xy or ref_xy
        map_cands = self._collect_cands(
            "map_component", list(res.get("map_component") or [])
        )
        map_pick = pick_player_xy(map_cands, map_ref, agree_xy=ina_xy or xf_xy)
        if map_pick is None and map_cands:
            map_pick = self._pick_by_movement("map_component", map_cands)
        if map_pick is not None:
            pos = self._read_from_obj("map_component", map_pick)
            if pos:
                if (
                    xf_xy is not None
                    and near_origin((pos.x, pos.y))
                    and not near_origin(xf_xy)
                ):
                    log.info(
                        "il2cpp pos skip origin map_component "
                        "xy=(%.2f,%.2f) xf=(%.2f,%.2f)",
                        pos.x,
                        pos.y,
                        xf_xy[0],
                        xf_xy[1],
                    )
                else:
                    self._obj["map_component"] = map_pick
                    found_any = True
                    log.info(
                        "il2cpp pos source=%s obj=0x%X xy=(%.2f,%.2f) n=%d",
                        "map_component",
                        map_pick,
                        pos.x,
                        pos.y,
                        len(map_cands),
                    )

        for name in order:
            if name in ("transform_xz", "inaccurate", "map_component"):
                continue
            if self._closed:
                break
            cands = self._collect_cands(name, list(res.get(name) or []))
            pick = pick_player_xy(cands, map_ref or ref_xy, agree_xy=ina_xy)
            if pick is None:
                continue
            pos = self._read_from_obj(name, pick)
            if not pos:
                continue
            self._obj[name] = pick
            found_any = True
            log.info(
                "il2cpp pos source=%s obj=0x%X xy=(%.2f,%.2f) n=%d",
                name,
                pick,
                pos.x,
                pos.y,
                len(cands),
            )
        if found_any:
            self._freeze_xy = None
            self._freeze_since = 0.0
            self._save_pos_seeds()
            log.info(
                "il2cpp pos resolve done wall=%.2fs sources=%s",
                time.perf_counter() - t0,
                list(self._obj),
            )
        else:
            self._last_err = "instance_not_found"
        return found_any

    def _pos_seed_key(self) -> str:
        return f"{int(self._pid or 0)}:{hex(int(self._ga or 0))}"

    def _apply_pos_seeds(self) -> bool:
        """复用上次坐标对象指针；成功写入 self._obj 返回 True。"""
        if self._h is None or not self._pid or not self._ga:
            return False
        try:
            if not _POS_SEED_FILE.is_file():
                return False
            raw = json.loads(_POS_SEED_FILE.read_text(encoding="utf-8"))
            entry = raw.get(self._pos_seed_key()) if isinstance(raw, dict) else None
            if not isinstance(entry, dict):
                return False
        except Exception as exc:
            log.debug("pos seed load: %s", exc)
            return False
        restored = 0
        for name, klass in self._klasses.items():
            a = entry.get(name)
            if a is None:
                continue
            try:
                addr = int(a, 0) if isinstance(a, str) else int(a)
            except (TypeError, ValueError):
                continue
            if not addr or read_ptr(self._h, addr) != klass:
                continue
            if self._read_from_obj(name, addr) is None:
                continue
            self._obj[name] = addr
            restored += 1
        return restored > 0

    def _save_pos_seeds(self) -> None:
        if not self._obj or not self._pid or not self._ga:
            return
        try:
            _POS_SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
            raw: dict = {}
            if _POS_SEED_FILE.is_file():
                try:
                    raw = json.loads(_POS_SEED_FILE.read_text(encoding="utf-8"))
                except Exception:
                    raw = {}
            if not isinstance(raw, dict):
                raw = {}
            raw[self._pos_seed_key()] = {
                name: hex(int(addr)) for name, addr in self._obj.items()
            }
            if len(raw) > 8:
                keys = list(raw.keys())[-8:]
                raw = {k: raw[k] for k in keys}
            tmp = _POS_SEED_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(_POS_SEED_FILE)
        except Exception as exc:
            log.debug("pos seed save: %s", exc)

    def schedule_resolve(self, hwnd: int | None = None) -> None:
        with self._lock:
            if self._resolving or self._closed:
                return
            self._resolving = True
            self._last_err = "resolving"

        def _work() -> None:
            try:
                if not self._closed:
                    self.read(hwnd=hwnd, force_rescan=False, allow_scan=True)
            except Exception as exc:
                log.debug("il2cpp background resolve: %s", exc)
                self._last_err = "read_failed"
            finally:
                self._resolving = False

        self._bg = threading.Thread(target=_work, name="il2cpp-resolve", daemon=True)
        self._bg.start()

    def read_all_sources(self, hwnd: int | None = None) -> dict[str, Il2CppPos]:
        """探测用：同步扫并返回所有可读源。"""
        out: dict[str, Il2CppPos] = {}
        if not self._ensure_proc(hwnd=hwnd):
            return out
        with self._lock:
            if not self._resolve_klasses():
                return out
        self._resolve_instances()
        for name in self._order():
            obj = self._obj.get(name)
            if not obj:
                continue
            pos = self._read_from_obj(name, obj)
            if pos:
                out[name] = pos
        return out

    def read(
        self,
        hwnd: int | None = None,
        *,
        force_rescan: bool = False,
        allow_scan: bool = True,
    ) -> Il2CppPos | None:
        t0 = time.perf_counter()
        if self._closed:
            self._last_err = "closed"
            return None
        if not self._ensure_proc(hwnd=hwnd):
            return None
        # lean：短超时，避免后台堆扫偶发占锁把状态栏拖成数百 ms 一拍
        if allow_scan:
            self._lock.acquire()
            got = True
        else:
            got = self._lock.acquire(timeout=0.05)
        if not got:
            self._last_err = "busy"
            lg = self._last_good
            if lg is not None:
                lg.age_ms = (time.perf_counter() - t0) * 1000.0
            return lg
        try:
            if self._ga is None or not self._klasses or force_rescan:
                if not self._resolve_klasses():
                    return None
            pos = None if force_rescan else self._try_cached()
            need = self._need_rescan
            if need:
                self._need_rescan = False
            if pos is not None and not need:
                self._last_err = ""
                pos.age_ms = (time.perf_counter() - t0) * 1000.0
                self._last_good = pos
                return pos
            # 绑错后立刻有 fallback 坐标也返回，并后台重扫 map
            fallback = pos
        finally:
            self._lock.release()
        if need and not allow_scan:
            self.schedule_resolve(hwnd=hwnd)
            out = fallback if fallback is not None else self._last_good
            if out is not None:
                self._last_err = ""
                out.age_ms = (time.perf_counter() - t0) * 1000.0
                self._last_good = out
                return out
        if not allow_scan:
            out = fallback if fallback is not None else self._last_good
            if out is not None:
                # 有最近成功坐标就先吐出；后台可再补扫，勿把状态栏打回「定位中」
                if self._resolving:
                    self._last_err = ""
                else:
                    self.schedule_resolve(hwnd=hwnd)
                    self._last_err = ""
                out.age_ms = (time.perf_counter() - t0) * 1000.0
                self._last_good = out
                return out
            if self._resolving:
                self._last_err = "resolving"
            else:
                self.schedule_resolve(hwnd=hwnd)
                self._last_err = "resolving"
            return None
        with self._lock:
            pos = self._try_cached()
            if pos is not None and not need:
                self._last_err = ""
                pos.age_ms = (time.perf_counter() - t0) * 1000.0
                self._last_good = pos
                return pos
            if self._ga is None or not self._klasses:
                if not self._resolve_klasses():
                    return fallback
        ok = self._resolve_instances()
        with self._lock:
            if not ok:
                self._last_err = self._last_err or "read_failed"
                return fallback
            pos = self._try_cached()
            if pos is None:
                self._last_err = "read_failed"
                return fallback
            self._last_err = ""
            pos.age_ms = (time.perf_counter() - t0) * 1000.0
            self._last_good = pos
            return pos


_READER: Il2CppPositionReader | None = None


def get_il2cpp_reader(path: Path | None = None) -> Il2CppPositionReader:
    global _READER
    if _READER is None:
        _READER = Il2CppPositionReader(path)
    return _READER
