# -*- coding: utf-8 -*-
from __future__ import annotations

import ctypes

from mm_assist.memory.process import read_chunk, read_mem, read_mem_into


def test_read_chunk_reuses_buffer(monkeypatch):
    from mm_assist.memory import process as mr

    def fake_rpm(hproc, addr, dest, size, nread_ptr):
        ctypes.cast(nread_ptr, ctypes.POINTER(ctypes.c_size_t)).contents.value = size
        ctypes.memset(dest, 0xAB, size)
        return 1

    monkeypatch.setattr(mr.kernel32, "ReadProcessMemory", fake_rpm)
    buf = (ctypes.c_char * 16)()
    got = read_mem_into(1, 0x1000, buf, 16)
    assert got == 16
    assert buf[0] == b"\xab"
    chunk = read_chunk(1, 0x1000, 8)
    assert chunk is not None
    ba, n = chunk
    assert n == 8
    assert ba[0] == 0xAB
    raw = read_mem(1, 0x1000, 8)
    assert raw == b"\xab" * 8
