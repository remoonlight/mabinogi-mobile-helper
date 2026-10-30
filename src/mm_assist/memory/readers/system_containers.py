"""SystemContainers 单例只读。TypeInfo RVA 由调用方 json 提供，不改 A 档。"""

from __future__ import annotations

from collections.abc import Iterator

from mm_assist.memory.il2cpp import resolve_klass, scan_klass_objects
from mm_assist.memory.process import read_ptr


def iter_system_containers_objs(hproc, ga: int, typeinfo_rva: int) -> Iterator[int]:
    if not ga or not typeinfo_rva:
        return
    klass = resolve_klass(hproc, ga, typeinfo_rva)
    if not klass:
        return
    for obj in scan_klass_objects(hproc, klass, max_hits=8, max_wall_s=3.0):
        if read_ptr(hproc, obj) == klass:
            yield int(obj)


def get_system_containers_obj(hproc, ga: int, typeinfo_rva: int) -> int | None:
    for obj in iter_system_containers_objs(hproc, ga, typeinfo_rva):
        return obj
    return None


def read_sc_component(
    hproc, ga: int, sc_rva: int, field_off: int, expected_klass: int
) -> int | None:
    """在多份 SC 堆命中里，挑字段 klass 对得上的那份。"""
    if not expected_klass:
        return None
    for sc in iter_system_containers_objs(hproc, ga, sc_rva):
        comp = read_ptr(hproc, sc + int(field_off))
        if not comp:
            continue
        if read_ptr(hproc, comp) == expected_klass:
            return int(comp)
    return None
