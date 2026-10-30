"""Bounded direct UI reads yield heap scans; all test workers must exit."""
import threading
import pytest
from mm_assist.memory import il2cpp as mem


def test_active_heap_scan_yields_and_new_scan_does_not_compete(monkeypatch):
    entered=threading.Event();finished=threading.Event()
    def scan(*args,should_stop,**kwargs):
        entered.set()
        while not should_stop():
            if finished.wait(0.001):break
        return {'A':[]}
    monkeypatch.setattr(mem,'_scan_multi',scan)
    worker=threading.Thread(target=lambda:mem.scan_klass_objects_multi(1,{'A':0x10000},max_wall_s=1))
    worker.start()
    try:
        assert entered.wait(1)
        with mem.ui_memory_read_slot(timeout_s=0.25) as acquired:
            assert acquired
            worker.join(1);assert not worker.is_alive()
            monkeypatch.setattr(mem,'_scan_multi',lambda *a,**k:pytest.fail('competing heap read'))
            assert mem.scan_klass_objects_multi(1,{'A':0x10000},max_wall_s=0.01)=={'A':[]}
    finally:
        finished.set();worker.join(2)
    assert not worker.is_alive() and not mem._UI_MEMORY_READ_PENDING.is_set()


def test_direct_timeout_and_exception_release_pending_state():
    mem._HEAP_READ_LOCK.acquire()
    try:
        with mem.ui_memory_read_slot(timeout_s=0.01) as acquired:assert not acquired
        assert not mem._UI_MEMORY_READ_PENDING.is_set()
    finally:mem._HEAP_READ_LOCK.release()
    with pytest.raises(RuntimeError):
        with mem.ui_memory_read_slot(timeout_s=0.1) as acquired:
            assert acquired;raise RuntimeError('cancelled caller')
    assert not mem._UI_MEMORY_READ_PENDING.is_set()
    assert mem._HEAP_READ_LOCK.acquire(blocking=False)
    mem._HEAP_READ_LOCK.release()


def test_one_waiter_timing_out_does_not_clear_another_active_request():
    results=[]
    def other():
        with mem.ui_memory_read_slot(timeout_s=0.01) as acquired:results.append(acquired)
    with mem.ui_memory_read_slot(timeout_s=0.1) as acquired:
        assert acquired
        worker=threading.Thread(target=other);worker.start();worker.join(1)
        assert not worker.is_alive() and results==[False]
        assert mem._UI_MEMORY_READ_PENDING.is_set()
    assert not mem._UI_MEMORY_READ_PENDING.is_set()
