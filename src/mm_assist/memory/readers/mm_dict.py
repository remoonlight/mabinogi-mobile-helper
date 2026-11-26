"""MMDictionary 只读遍历。偏移来自各组件 json，不猜新值。"""

from __future__ import annotations

import struct
from collections.abc import Iterator

from mm_assist.memory.process import read_mem, read_ptr, read_u32

_MAX = 4096


def _looks_ascii_qword(p: int) -> bool:
    """堆扫假指针常把 ASCII 当 qword；6 个以上可打印字节则丢掉。"""
    b = int(p).to_bytes(8, "little")
    return sum(1 for x in b if 0x20 <= x <= 0x7E) >= 6


def iter_ptr_values(hproc, dict_ptr: int, offs: dict[str, int]) -> Iterator[int]:
    if not dict_ptr:
        return
    count = int(read_u32(hproc, dict_ptr + offs["count"]) or 0)
    if count <= 0 or count > _MAX:
        return
    arr = read_ptr(hproc, dict_ptr + offs["values"])
    if not arr:
        return
    max_len = int(read_u32(hproc, arr + offs["arr_max"]) or 0)
    vec = arr + offs["arr_vec"]
    n = min(count, max_len, _MAX)
    for i in range(n):
        p = read_ptr(hproc, vec + i * 8)
        if p and p > 0x10000 and not _looks_ascii_qword(p):
            yield int(p)


def iter_key_ptr_pairs(
    hproc, dict_ptr: int, offs: dict[str, int]
) -> Iterator[tuple[int, int]]:
    if not dict_ptr:
        return
    count = int(read_u32(hproc, dict_ptr + offs["count"]) or 0)
    if count <= 0 or count > _MAX:
        return
    keys = read_ptr(hproc, dict_ptr + offs["keys"])
    vals = read_ptr(hproc, dict_ptr + offs["values"])
    if not keys or not vals:
        return
    klen = int(read_u32(hproc, keys + offs["arr_max"]) or 0)
    vlen = int(read_u32(hproc, vals + offs["arr_max"]) or 0)
    kvec = keys + offs["arr_vec"]
    vvec = vals + offs["arr_vec"]
    n = min(count, klen, vlen, _MAX)
    for i in range(n):
        key = read_u32(hproc, kvec + i * 4)
        val = read_ptr(hproc, vvec + i * 8)
        if val and val > 0x10000:
            yield int(key or 0), int(val)


def _read_i32(hproc, addr: int) -> int | None:
    raw = read_mem(hproc, addr, 4)
    if not raw or len(raw) < 4:
        return None
    return struct.unpack("<i", raw)[0]


def iter_key_int_pairs(
    hproc, dict_ptr: int, offs: dict[str, int]
) -> Iterator[tuple[int, int]]:
    """EnumKey/u32 键 + int 值；值按 i32、stride 4，不当指针。"""
    if not dict_ptr:
        return
    count = int(read_u32(hproc, dict_ptr + offs["count"]) or 0)
    if count <= 0 or count > _MAX:
        return
    keys = read_ptr(hproc, dict_ptr + offs["keys"])
    vals = read_ptr(hproc, dict_ptr + offs["values"])
    if not keys or not vals:
        return
    klen = int(read_u32(hproc, keys + offs["arr_max"]) or 0)
    vlen = int(read_u32(hproc, vals + offs["arr_max"]) or 0)
    kvec = keys + offs["arr_vec"]
    vvec = vals + offs["arr_vec"]
    n = min(count, klen, vlen, _MAX)
    for i in range(n):
        key = read_u32(hproc, kvec + i * 4)
        val = _read_i32(hproc, vvec + i * 4)
        if key is None or val is None:
            continue
        if 0 <= key <= 5 and 0 <= int(val) <= 32:
            yield key, int(val)
