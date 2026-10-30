"""只读进程内存原语（禁止写入/注入）。"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import struct
import threading
from typing import Callable, Iterator

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
MEM_COMMIT = 0x1000
MEM_PRIVATE = 0x20000
PAGE_NOACCESS = 0x01
PAGE_GUARD = 0x100
PAGE_READONLY = 0x02
PAGE_READWRITE = 0x04
PAGE_WRITECOPY = 0x08
PAGE_EXECUTE_READ = 0x20
PAGE_EXECUTE_READWRITE = 0x40
PAGE_EXECUTE_WRITECOPY = 0x80

READABLE = {
    PAGE_READONLY,
    PAGE_READWRITE,
    PAGE_WRITECOPY,
    PAGE_EXECUTE_READ,
    PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY,
}

WRITABLE = {PAGE_READWRITE, PAGE_EXECUTE_READWRITE, PAGE_WRITECOPY, PAGE_EXECUTE_WRITECOPY}

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.CloseHandle.argtypes = [wt.HANDLE]
kernel32.CloseHandle.restype = wt.BOOL
kernel32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
kernel32.CreateToolhelp32Snapshot.restype = wt.HANDLE
for _module_fn in (kernel32.Module32FirstW, kernel32.Module32NextW):
    _module_fn.argtypes = [wt.HANDLE, ctypes.c_void_p]
    _module_fn.restype = wt.BOOL

# 少一次 ctypes 参数编解码；只读扫描热路径。
kernel32.ReadProcessMemory.argtypes = [
    wt.HANDLE,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.ReadProcessMemory.restype = wt.BOOL
kernel32.VirtualQueryEx.argtypes = [
    wt.HANDLE,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
kernel32.VirtualQueryEx.restype = ctypes.c_size_t

PAGE_SIZE = 4096
# 线程内复用 RPM 缓冲，避免每读一次就 ``(c_char * n)()``。
_RPM_TLS = threading.local()
_RPM_TLS_MAX = 4 * 1024 * 1024
# 相邻已提交区合并上限：再大就留给堆扫按块读，避免单区吞掉并行。
COALESCE_MAX_SPAN = 16 * 1024 * 1024


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wt.DWORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wt.DWORD),
        ("Protect", wt.DWORD),
        ("Type", wt.DWORD),
    ]


def find_pid_by_name(name: str) -> int | None:
    """按映像名找 PID；容忍带/不带 .exe。"""
    want = name.lower()
    want_bare = want[:-4] if want.endswith(".exe") else want
    try:
        import psutil

        for p in psutil.process_iter(["pid", "name"]):
            n = (p.info.get("name") or "").lower()
            bare = n[:-4] if n.endswith(".exe") else n
            if n == want or bare == want_bare:
                return int(p.info["pid"])
    except Exception:
        pass

    TH32CS_SNAPPROCESS = 0x00000002

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wt.DWORD),
            ("cntUsage", wt.DWORD),
            ("th32ProcessID", wt.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wt.DWORD),
            ("cntThreads", wt.DWORD),
            ("th32ParentProcessID", wt.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wt.DWORD),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == wt.HANDLE(-1).value:
        return None
    pe = PROCESSENTRY32W()
    pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    ok = kernel32.Process32FirstW(snap, ctypes.byref(pe))
    while ok:
        n = pe.szExeFile.lower()
        bare = n[:-4] if n.endswith(".exe") else n
        if n == want or bare == want_bare:
            pid = int(pe.th32ProcessID)
            kernel32.CloseHandle(snap)
            return pid
        ok = kernel32.Process32NextW(snap, ctypes.byref(pe))
    kernel32.CloseHandle(snap)
    return None


def pid_from_hwnd(hwnd: int) -> int | None:
    if not hwnd:
        return None
    try:
        import win32process

        _tid, pid = win32process.GetWindowThreadProcessId(int(hwnd))
        return int(pid) or None
    except Exception:
        pass
    pid = wt.DWORD(0)
    user32.GetWindowThreadProcessId(ctypes.c_void_p(int(hwnd)), ctypes.byref(pid))
    return int(pid.value) or None


def find_game_pid(*, hwnd: int | None = None, title_substr: str = "") -> int | None:
    """优先窗口句柄 → 进程名 → 按标题枚举窗口。"""
    if hwnd:
        p = pid_from_hwnd(hwnd)
        if p:
            return p
    p = find_pid_by_name("MabinogiMobile.exe") or find_pid_by_name("MabinogiMobile")
    if p:
        return p
    try:
        import win32gui

        keys = [title_substr] if title_substr else ["Mabinogi", "瑪奇", "Mobile"]
        found: list[int] = []

        def _cb(h: int, _: object) -> bool:
            if not win32gui.IsWindowVisible(h):
                return True
            title = win32gui.GetWindowText(h) or ""
            if any(k and k in title for k in keys):
                # 避免误匹配其它 Mobile 应用：优先含 瑪奇/Mabinogi
                if not any(k in title for k in ("瑪奇", "Mabinogi", "mabinogi")):
                    if "Mobile" in title and "瑪奇" not in title:
                        return True
                pid = pid_from_hwnd(h)
                if pid:
                    found.append(pid)
            return True

        win32gui.EnumWindows(_cb, None)
        if found:
            return found[0]
    except Exception:
        pass
    return None


def process_handle_alive(handle) -> bool:
    """Check the bound process itself; PID reuse cannot revive an exited handle."""
    if not handle:
        return False
    code = wt.DWORD()
    query = kernel32.GetExitCodeProcess
    query.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
    query.restype = wt.BOOL
    return bool(query(handle, ctypes.byref(code))) and code.value == 259


def open_process(pid: int):
    h = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
    if not h:
        err = ctypes.get_last_error()
        from mm_assist.memory.access_diagnostic import ProcessReadDenied
        from mm_assist.memory.diagnostics import note_access_error
        exc = ProcessReadDenied(pid, err)
        note_access_error(pid, exc.diagnostic)
        raise exc
    try:
        from mm_assist.memory.client_build import verify_client
        verify_client(h, pid)
    except Exception:
        kernel32.CloseHandle(h)
        raise
    return h


def process_birth(handle) -> int:
    """Creation FILETIME from the bound handle, independent of PID reuse."""
    created, exited, kernel, user = (wt.FILETIME() for _ in range(4))
    fn = kernel32.GetProcessTimes
    fn.argtypes = [wt.HANDLE] + [ctypes.POINTER(wt.FILETIME)] * 4
    fn.restype = wt.BOOL
    if not fn(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
        raise OSError(ctypes.get_last_error(), 'GetProcessTimes failed')
    return (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)


def read_identity(handle, pid: int) -> dict:
    from mm_assist.memory.client_build import bound_identity
    return bound_identity(handle, pid)


def coalesce_regions(
    regions: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...],
    *,
    max_span: int = COALESCE_MAX_SPAN,
) -> list[tuple[int, int, int, int]]:
    """合并相邻、同 Type 的已提交区，减少 RPM 次数。

    不跨空洞、不混 ``MEM_PRIVATE`` / ``MEM_IMAGE``。``max_span`` 防止合成超大区。
    VirtualQuery 已按 Protect 拆区；此处只把过滤后仍相邻的可读写私有块拼回去。
    """
    out: list[tuple[int, int, int, int]] = []
    cap = int(max_span) if max_span and max_span > 0 else COALESCE_MAX_SPAN
    for base, size, prot, typ in regions:
        if size <= 0:
            continue
        if out:
            pb, ps, pp, pt = out[-1]
            if typ == pt and base == pb + ps and ps + size <= cap:
                out[-1] = (pb, ps + size, pp, pt)
                continue
        out.append((int(base), int(size), int(prot), int(typ)))
    return out


def iter_regions(
    hproc,
    *,
    private_only: bool = False,
    writable_only: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[tuple[int, int, int, int]]:
    """yield (base, size, protect, type)。只枚举 MEM_COMMIT + 可读、非 GUARD。"""
    mbi = MEMORY_BASIC_INFORMATION()
    addr = 0
    max_addr = 0x00007FFFFFFFFFFF
    while addr < max_addr:
        if should_stop is not None and should_stop():
            break
        n = kernel32.VirtualQueryEx(
            hproc, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)
        )
        if not n:
            break
        base = int(mbi.BaseAddress or 0)
        size = int(mbi.RegionSize or 0)
        if size == 0:
            break
        prot = int(mbi.Protect)
        typ = int(mbi.Type)
        if (
            mbi.State == MEM_COMMIT
            and (prot & 0xFF) in READABLE
            and not (prot & PAGE_GUARD)
            and not (prot & PAGE_NOACCESS)
        ):
            if private_only and typ != MEM_PRIVATE:
                pass
            elif writable_only and (prot & 0xFF) not in WRITABLE:
                pass
            else:
                yield base, size, prot, typ
        next_addr = base + size
        if next_addr <= addr:
            break
        addr = next_addr


def _rpm_dest(size: int) -> tuple[bytearray, object]:
    """线程本地 bytearray + ctypes 视图；下次 read 会覆盖内容。"""
    cap = int(getattr(_RPM_TLS, "cap", 0) or 0)
    ba = getattr(_RPM_TLS, "ba", None)
    ctype = getattr(_RPM_TLS, "ctype", None)
    if ba is None or cap < size:
        n = max(size, PAGE_SIZE)
        n = (n + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)
        ba = bytearray(n)
        ctype = (ctypes.c_char * n).from_buffer(ba)
        _RPM_TLS.ba = ba
        _RPM_TLS.ctype = ctype
        _RPM_TLS.cap = n
    return ba, ctype


def read_mem_into(hproc, addr: int, buf, size: int) -> int:
    """读入已有缓冲，返回实读字节数；失败为 0。"""
    if size <= 0:
        return 0
    nread = ctypes.c_size_t(0)
    ok = bool(
        kernel32.ReadProcessMemory(
            hproc, ctypes.c_void_p(addr), buf, size, ctypes.byref(nread)
        )
    )
    got = int(nread.value)
    if not ok or got <= 0:
        return 0
    return got


def read_chunk(hproc, addr: int, size: int) -> tuple[bytearray, int] | None:
    """读入 TLS 缓冲，不另拷成 bytes。返回 (buf, 实读长度)。"""
    if size <= 0:
        return None
    if size > _RPM_TLS_MAX:
        ba = bytearray(size)
        ctype = (ctypes.c_char * size).from_buffer(ba)
        got = read_mem_into(hproc, addr, ctype, size)
        if got <= 0:
            return None
        return ba, got
    ba, ctype = _rpm_dest(size)
    got = read_mem_into(hproc, addr, ctype, size)
    if got <= 0:
        return None
    return ba, got


def read_mem(hproc, addr: int, size: int) -> bytes | None:
    chunk = read_chunk(hproc, addr, size)
    if not chunk:
        return None
    ba, got = chunk
    return bytes(ba[:got])


def read_floats(hproc, addr: int, count: int = 3) -> tuple[float, ...] | None:
    raw = read_mem(hproc, addr, 4 * count)
    if not raw or len(raw) < 4 * count:
        return None
    return struct.unpack("<" + "f" * count, raw[: 4 * count])


def read_ptr(hproc, addr: int) -> int | None:
    """读 8 字节小端指针（x64）。"""
    raw = read_mem(hproc, addr, 8)
    if not raw or len(raw) < 8:
        return None
    return int.from_bytes(raw, "little")


def read_u8(hproc, addr: int) -> int | None:
    raw = read_mem(hproc, addr, 1)
    if not raw:
        return None
    return raw[0]


def read_u32(hproc, addr: int) -> int | None:
    raw = read_mem(hproc, addr, 4)
    if not raw or len(raw) < 4:
        return None
    return int.from_bytes(raw, "little")


def decode_utf16_le(raw: bytes | None) -> str | None:
    """UTF-16LE → str。非法/代理项不进 JSON：解不出就 None，不猜。"""
    if not raw:
        return None
    try:
        text = raw.decode("utf-16-le", errors="strict")
    except UnicodeError:
        text = raw.decode("utf-16-le", errors="replace")
    text = text.rstrip("\x00")
    if not text:
        return None
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return text


def process_modules(pid: int) -> dict[str, tuple[int, str]]:
    """Module addresses and actual loaded paths from the selected process."""
    TH32CS_SNAPMODULE = 0x00000008
    TH32CS_SNAPMODULE32 = 0x00000010

    class MODULEENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wt.DWORD),
            ("th32ModuleID", wt.DWORD),
            ("th32ProcessID", wt.DWORD),
            ("GlblcntUsage", wt.DWORD),
            ("ProccntUsage", wt.DWORD),
            ("modBaseAddr", ctypes.c_void_p),
            ("modBaseSize", wt.DWORD),
            ("hModule", wt.HMODULE),
            ("szModule", ctypes.c_wchar * 256),
            ("szExePath", ctypes.c_wchar * 260),
        ]

    snap = kernel32.CreateToolhelp32Snapshot(
        TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, wt.DWORD(pid)
    )
    invalid = ctypes.c_void_p(-1).value
    if not snap or snap == invalid:
        return {}
    me = MODULEENTRY32W()
    me.dwSize = ctypes.sizeof(MODULEENTRY32W)
    ok = kernel32.Module32FirstW(snap, ctypes.byref(me))
    found = {}
    try:
        while ok:
            name = str(me.szModule)
            name = {'gameassembly.dll': 'GameAssembly.dll', 'unityplayer.dll': 'UnityPlayer.dll'}.get(name.lower(), name)
            value = (int(me.modBaseAddr or 0), str(me.szExePath))
            if name in found and found[name] != value:
                if name in {'GameAssembly.dll', 'UnityPlayer.dll'}:
                    raise OSError('ambiguous loaded module name: '+name)
                # Vendor SDKs can legitimately load the same basename from two
                # directories. They do not identify the client layout. Keep the
                # ambiguity explicit without blocking unrelated client reads.
                value = (0, '')
            found[name] = value
            ok = kernel32.Module32NextW(snap, ctypes.byref(me))
    finally:
        kernel32.CloseHandle(snap)
    return found


def find_module_base(pid: int, module_name: str) -> int | None:
    want = module_name.lower().removesuffix('.dll')
    for name, (base, _path) in process_modules(pid).items():
        if name.lower().removesuffix('.dll') == want:
            return base or None
    return None


def close_handle(h) -> None:
    from mm_assist.memory.client_build import forget_handle
    forget_handle(h)
    kernel32.CloseHandle(h)
