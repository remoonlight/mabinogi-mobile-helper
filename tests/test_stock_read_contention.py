"""Current feed stock must progress while background work yields or inspects pixels."""
import threading
from types import SimpleNamespace

import pytest

from mm_assist.life_plan import snapshot
from mm_assist.memory import il2cpp as mem


def test_stock_request_yields_scan_before_waiting_for_outer_read_lock(monkeypatch):
    from mm_assist.memory.readers import consumable

    entered = threading.Event()
    stop = threading.Event()
    yielded = threading.Event()
    outcomes = []
    reads, closed = [], []

    def read_items(h, pid, *, allow_scan):
        assert (h, pid, allow_scan) == (7, 1234, False)
        reads.append(1)
        return {'鐵礦石': 100, '煤炭': 100}

    def scan(*args, should_stop, **kwargs):
        entered.set()
        while not stop.wait(.001):
            if should_stop():
                yielded.set()
                break
        return {'A': []}

    def watch():
        with snapshot.heavy_read_lock:
            outcomes.append(mem.scan_klass_objects_multi(1, {'A': 0x10000}, max_wall_s=2))

    monkeypatch.setattr(mem, '_scan_multi', scan)
    monkeypatch.setattr(snapshot, 'find_game_pid', lambda: 1234)
    monkeypatch.setattr(snapshot, 'open_process', lambda pid: 7)
    monkeypatch.setattr(snapshot, 'close_handle', closed.append)
    monkeypatch.setattr(consumable, 'read_consumable_counts', read_items)
    worker = threading.Thread(target=watch)
    worker.start()
    try:
        assert entered.wait(1)
        assert snapshot.refresh_items(budget_s=.25) == {'鐵礦石': 100, '煤炭': 100}
        assert yielded.is_set(), '库存请求必须在外层锁等待前通知扫描让行'
    finally:
        stop.set()
        worker.join(2)
    assert not worker.is_alive() and outcomes == [{'A': []}]
    assert reads == [1, 1] and closed == [7]
    assert not snapshot.snapshot_in_progress() and not snapshot.heavy_read_lock.locked()
    assert not mem._UI_MEMORY_READ_PENDING.is_set()


@pytest.mark.parametrize('mode', ['memory', 'close', 'exception'])
def test_slow_pixel_inspection_does_not_occupy_stock_read_lock(monkeypatch, mode):
    import tkinter.ttk as ttk
    from mm_assist.life_plan import node_inspect as ni

    callbacks, workers, published, written = [], [], [], []
    entered = threading.Event()
    stop = threading.Event()
    finish = threading.Event()

    class Widget:
        def __init__(self, *a, **k): pass
        def pack(self, *a, **k): pass
        def __setitem__(self, *a): pass
        def bind(self, *a): pass
        def set(self, *a): pass
        def configure(self, **k): pass

    def compute(*a, **k):
        entered.set()
        assert finish.wait(2)
        if mode == 'exception':
            raise RuntimeError('offline inspection failure')
        return {'node': 'menu', 'stack_ready': True, 'stack_names': ['SystemMenuUIState']}

    real_thread = threading.Thread
    def thread(*a, **k):
        worker = real_thread(*a, **k)
        workers.append(worker)
        return worker

    monkeypatch.setattr(ttk, 'Combobox', Widget)
    monkeypatch.setattr('mm_assist.perception.ocr.ocr_roi', lambda *a, **k: [])
    monkeypatch.setattr(ni, '_view_or_state', lambda *a: None)
    monkeypatch.setattr(ni, '_stack_sn_from_mem', lambda *a: 'inventory')
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.write_inspect_page', lambda node, **k: written.append(node))
    monkeypatch.setattr(ni, 'compute_inspect', compute)
    monkeypatch.setattr(ni.threading, 'Thread', thread)
    apply_memory = ni.attach_node_inspect(SimpleNamespace(after=lambda ms, fn: callbacks.append((ms, fn))),
        tk_mod=SimpleNamespace(Frame=Widget, StringVar=Widget, Label=Widget),
        get_win=lambda: None, grab=lambda: pytest.fail('no frame needed'), mark=None,
        stop=stop, post_ui=lambda key, fn, *args: published.append((fn, args)))
    try:
        next(fn for ms, fn in callbacks if ms == 200)()
        assert entered.wait(1)
        assert snapshot.refresh_items(budget_s=.01) is None
        if mode == 'memory':
            apply_memory(SimpleNamespace(ui_stack_names=['InventoryUIState'], ui_stack_ready=True),
                         SimpleNamespace(ui_params='stp=0'))
            next(fn for ms, fn in callbacks if ms == 200)()
            assert len(workers) == 1, '内存展示不能清除仍在执行的识别任务'
        elif mode == 'close':
            stop.set()
        finish.set()
        for worker in workers:
            worker.join(2)
            assert not worker.is_alive()
        for fn, args in published:
            fn(*args)
        if mode == 'memory':
            assert written == ['inventory'], '旧识别不能覆盖较新的内存界面事实'
        elif mode == 'close':
            assert not published and not written
        else:
            assert written == ['']  # 异常发布未知，下一拍仍可工作。
            next(fn for ms, fn in callbacks if ms == 200)()
            assert len(workers) == 2
    finally:
        stop.set()
        finish.set()
        for worker in workers:
            worker.join(2)
            assert not worker.is_alive()
    assert not snapshot.heavy_read_lock.locked()


def test_stock_wait_does_not_cancel_an_active_full_snapshot(monkeypatch):
    entered, finish, yielded = threading.Event(), threading.Event(), threading.Event()

    def scan(*args, should_stop, **kwargs):
        entered.set()
        while not finish.wait(.001):
            if should_stop():
                yielded.set()
                break
        return {'A': []}

    def full_read():
        with snapshot._SCAN_LOCK, snapshot.heavy_read_lock:
            mem.scan_klass_objects_multi(1, {'A': 0x10000}, max_wall_s=2)

    monkeypatch.setattr(mem, '_scan_multi', scan)
    worker = threading.Thread(target=full_read)
    worker.start()
    try:
        assert entered.wait(1)
        with pytest.raises(snapshot.ItemsReadBusy, match='snapshot'):
            snapshot.refresh_items(budget_s=.02)
        assert not yielded.is_set(), '整包快照必须保持单份串行，不能被库存锁等待取消'
        assert not mem._UI_MEMORY_READ_PENDING.is_set()
    finally:
        finish.set()
        worker.join(2)
    assert not worker.is_alive()


@pytest.mark.parametrize('lock', [snapshot._SCAN_LOCK, snapshot.heavy_read_lock, mem._HEAP_READ_LOCK])
def test_timed_out_stock_request_releases_priority_without_another_reader(monkeypatch, lock):
    monkeypatch.setattr(snapshot, 'open_process', lambda *a: pytest.fail('锁忙不能开始读取'))
    with mem.ui_memory_read_request():
        lock.acquire()
        try:
            with pytest.raises(snapshot.ItemsReadBusy):
                snapshot.refresh_items(budget_s=.001)
        finally:
            lock.release()
        # A nested timeout must not withdraw another caller's request.
        assert mem._UI_MEMORY_READ_PENDING.is_set()
    assert not mem._UI_MEMORY_READ_PENDING.is_set()
    assert not snapshot.snapshot_in_progress() and not snapshot.heavy_read_lock.locked()
    assert not mem._HEAP_READ_LOCK.locked()
