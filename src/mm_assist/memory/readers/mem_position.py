"""角色世界坐标：优先 IL2CPP 指针链，回退固定地址。

配置：
- `knowledge/ui/config/il2cpp_position.json`（主路径）
- `knowledge/ui/mem_position.json`（差分扫描回退，本切片可缺）

探测：`python -m mm_assist.tools.il2cpp_position_probe`
扫描：`python -m mm_assist.tools.mem_position_scan`
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mm_assist.memory.readers.position import Il2CppPositionReader, default_path as il2cpp_path
from mm_assist.memory.process import (
    close_handle,
    find_game_pid,
    open_process,
    read_floats,
)

log = logging.getLogger(__name__)

DEFAULT_REL = Path("knowledge/ui/mem_position.json")


@dataclass
class MemPos:
    x: float
    y: float
    addr: int
    pid: int
    z: float | None = None
    source: str = "fixed"
    age_ms: float = 0.0
    layout: str = "xy_ff"

    def as_xy(self) -> tuple[float, float]:
        return self.x, self.y

    def as_tuple(self) -> tuple[float, float] | tuple[float, float, float]:
        if self.layout == "xyz_fff" and self.z is not None:
            return self.x, self.y, self.z
        return self.x, self.y

    def fmt(self, nd: int = 0) -> str:
        if nd <= 0:
            if self.layout == "xyz_fff" and self.z is not None:
                return f"({int(round(self.x))}, {int(round(self.y))}, {int(round(self.z))})"
            return f"({int(round(self.x))}, {int(round(self.y))})"
        if self.layout == "xyz_fff" and self.z is not None:
            return f"({self.x:.{nd}f}, {self.y:.{nd}f}, {self.z:.{nd}f})"
        return f"({self.x:.{nd}f}, {self.y:.{nd}f})"


def default_path(knowledge_root: Path | None = None) -> Path:
    if knowledge_root is None:
        return Path.cwd() / DEFAULT_REL
    return Path(knowledge_root) / "ui" / "mem_position.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or default_path()
    if not p.is_file():
        return {
            "version": 1,
            "process": "MabinogiMobile.exe",
            "layout": "xy_ff",
            "addr": None,
            "alloc_base": None,
            "offset_in_alloc": None,
            "pid": None,
            "verified_at": None,
            "sample": None,
            "note": "未锁定；请运行 mem_position_scan 并走动角色做差分",
        }
    return json.loads(p.read_text(encoding="utf-8"))


def save_config(cfg: dict[str, Any], path: Path | None = None) -> Path:
    p = path or default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def resolve_addr(cfg: dict[str, Any]) -> int | None:
    raw = cfg.get("addr")
    if raw is None or raw == "":
        return None
    if isinstance(raw, int):
        return raw
    s = str(raw).strip().lower()
    if s.startswith("0x"):
        return int(s, 16)
    return int(s)


class MemPositionReader:
    """优先 IL2CPP SystemContainers 链；失败再读固定地址。"""

    def __init__(
        self,
        path: Path | None = None,
        *,
        il2cpp: Path | None = None,
        prefer_il2cpp: bool = True,
    ) -> None:
        self.path = path or default_path()
        self._cfg = load_config(self.path)
        self._pid: int | None = None
        self._h = None
        self._addr: int | None = resolve_addr(self._cfg)
        self._layout: str = str(self._cfg.get("layout") or "xy_ff")
        self._last_err: str = ""
        self._prefer_il2cpp = prefer_il2cpp
        self._il2cpp = Il2CppPositionReader(il2cpp or il2cpp_path())

    @property
    def last_error(self) -> str:
        return self._last_err

    @property
    def locked(self) -> bool:
        return self._prefer_il2cpp or self._addr is not None

    @property
    def layout(self) -> str:
        return self._layout

    def reload(self) -> None:
        self._cfg = load_config(self.path)
        self._addr = resolve_addr(self._cfg)
        self._layout = str(self._cfg.get("layout") or "xy_ff")
        self._il2cpp.reload()
        self.close()

    def close(self) -> None:
        self._il2cpp.close()
        if self._h is not None:
            try:
                close_handle(self._h)
            except Exception:
                pass
            self._h = None
            self._pid = None

    def _ensure_fixed(self, hwnd: int | None = None) -> bool:
        pid = find_game_pid(hwnd=hwnd)
        if not pid:
            self._last_err = "process_not_found"
            self.close()
            return False
        if self._h is None or self._pid != pid:
            if self._h is not None:
                try:
                    close_handle(self._h)
                except Exception:
                    pass
                self._h = None
            try:
                self._h = open_process(pid)
                self._pid = pid
            except OSError as e:
                self._last_err = f"open_failed:{e}"
                return False
        if self._addr is None:
            self._last_err = "addr_not_locked"
            return False
        cfg_pid = self._cfg.get("pid")
        if cfg_pid and int(cfg_pid) != pid:
            self._last_err = "pid_changed_rescan"
            return False
        return True

    def _read_fixed(self, hwnd: int | None = None) -> MemPos | None:
        t0 = time.perf_counter()
        if not self._ensure_fixed(hwnd=hwnd):
            return None
        assert self._h is not None and self._addr is not None and self._pid is not None
        n = 3 if self._layout == "xyz_fff" else 2
        vals = read_floats(self._h, self._addr, n)
        if vals is None:
            self._last_err = "read_failed"
            return None
        x, y = float(vals[0]), float(vals[1])
        z = float(vals[2]) if n == 3 else None
        check = (x, y) if z is None else (x, y, z)
        if not all(abs(v) < 1e8 for v in check):
            self._last_err = "value_out_of_range"
            return None
        if abs(x) < 1e-4 and abs(y) < 1e-4:
            self._last_err = "xy_zero"
            return None
        self._last_err = ""
        return MemPos(
            x=x,
            y=y,
            z=z,
            addr=self._addr,
            pid=self._pid,
            source="fixed",
            age_ms=(time.perf_counter() - t0) * 1000.0,
            layout=self._layout if n == 3 else "xy_ff",
        )

    @property
    def resolving(self) -> bool:
        return bool(self._prefer_il2cpp and self._il2cpp.resolving)

    def read(self, hwnd: int | None = None, *, allow_scan: bool = True) -> MemPos | None:
        """allow_scan=False：状态栏用；只读 IL2CPP 缓存，未命中则后台扫。

        后台定位期间不回退固定地址，避免过期绝对地址显示假坐标。
        """
        if self._prefer_il2cpp:
            ip = self._il2cpp.read(hwnd=hwnd, allow_scan=allow_scan)
            if ip is not None:
                self._last_err = ""
                return MemPos(
                    x=ip.x,
                    y=ip.y,
                    addr=ip.addr,
                    pid=ip.pid,
                    source=ip.source,
                    age_ms=ip.age_ms,
                    layout="xy_ff",
                )
            il_err = self._il2cpp.last_error
            # 状态栏路径：绝不回退过期绝对地址；定位中/失败都如实返回
            if not allow_scan:
                if self._il2cpp.resolving or il_err in ("resolving", "cache_miss"):
                    self._last_err = "resolving"
                else:
                    self._last_err = il_err or "il2cpp_failed"
                return None
            fixed = self._read_fixed(hwnd=hwnd)
            if fixed is not None:
                return fixed
            self._last_err = il_err or self._last_err or "il2cpp_and_fixed_failed"
            return None
        return self._read_fixed(hwnd=hwnd)


_READER: MemPositionReader | None = None


def get_reader(path: Path | None = None) -> MemPositionReader:
    global _READER
    if _READER is None:
        _READER = MemPositionReader(path)
    return _READER


def read_world_pos(path: Path | None = None) -> MemPos | None:
    return get_reader(path).read()
