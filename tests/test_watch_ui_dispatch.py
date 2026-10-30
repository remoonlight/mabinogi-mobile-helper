from types import SimpleNamespace
import threading
import time

from mm_assist.jobs import state_watch as sw


def test_watch_keeps_publishing_while_tk_does_not_drain(monkeypatch):
    """真实 watch 主循环：Tk 忙碌不能堵采样、预热回调或就绪落盘。"""
    owner = threading.get_ident()
    stopped = threading.Event()
    published = threading.Event()
    workers = []
    foreign_tk_calls = []
    ready = []

    class Widget:
        def __init__(self, *a, **k): pass
        def pack(self, *a, **k): pass
        def set(self, *a):
            assert threading.get_ident() == owner

    class Root(Widget):
        def title(self, *a): pass
        def attributes(self, *a): pass
        def update_idletasks(self): pass
        def winfo_id(self): return 0
        def protocol(self, _name, fn): self.close = fn
        def after(self, *a):
            if threading.get_ident() != owner:
                foreign_tk_calls.append(a)
                stopped.wait(1)
        def destroy(self): pass
        def mainloop(self):
            try:
                assert published.wait(2), '采样被界面线程阻塞'
            finally:
                self.close()
                stopped.set()

    class Monitor:
        last_ms = None
        def __init__(self, *a, **k): pass
        def set_warm_progress(self, fn): self.progress = fn
        def tick_lean(self, w):
            from mm_assist.life_plan.action_finish import read_completion_stp
            sample = read_completion_stp()
            assert sample.stp == 1 and sample.fresh and sample.generation == 7
            self.progress(3, 3)
            return SimpleNamespace()
        def warm_load_progress(self): return 3, 3
        def _close_stop_pixel(self): pass
        def read_stop_sample(self): return {'stp': 1, 'timestamp': time.time(), 'generation': 7}

    real_thread = threading.Thread
    def thread(*a, **k):
        t = real_thread(*a, **k)
        workers.append(t)
        return t

    monkeypatch.setattr(sw.threading, 'Thread', thread)
    monkeypatch.setattr(sw, 'tk', SimpleNamespace(Tk=Root, Frame=Widget, StringVar=Widget, Label=Widget))
    monkeypatch.setattr(sw, 'SceneMonitor', Monitor)
    monkeypatch.setattr(sw, 'try_acquire_state_watch_singleton', lambda: True)
    monkeypatch.setattr('mm_assist.session_log.start_watch_session_log', lambda: None)
    monkeypatch.setattr(sw, 'hide_own_console', lambda: None)
    monkeypatch.setattr(sw, 'ensure_game_frame', lambda *a, **k: None)
    monkeypatch.setattr('win32gui.GetAncestor', lambda *a: 0)
    monkeypatch.setattr('mm_assist.safety.input.register_input_hud', lambda *a: None)
    monkeypatch.setattr('mm_assist.safety.input.release_mouse_buttons', lambda: None)
    monkeypatch.setattr(sw, 'ClickMarkOverlay', lambda *a: SimpleNamespace(destroy=lambda: None))
    monkeypatch.setattr('mm_assist.life_plan.node_inspect.attach_node_inspect', lambda *a, **k: None)
    monkeypatch.setattr('mm_assist.life_plan.bar_ui.attach_life_plan_controls', lambda *a, **k: None)
    monkeypatch.setattr(sw, 'JobWatcher', lambda *a, **k: SimpleNamespace(poll_once=lambda: None))
    monkeypatch.setattr(sw, 'load_job', lambda: {})
    monkeypatch.setattr(sw, 'refresh_window', lambda w: w)
    monkeypatch.setattr(sw, 'write_state_watch_heartbeat', lambda: None)
    monkeypatch.setattr(sw, 'write_state_ui', lambda *a: None)
    def write_ready(*a, **k):
        if k.get('tick_completed') and not k.get('busy'):
            ready.append(time.monotonic())
            if len(ready) >= 3:
                published.set()
    monkeypatch.setattr(sw, 'write_watch_ready', write_ready)
    for name in ('clear_state_watch_heartbeat', 'clear_state_ui', 'clear_watch_ready'):
        monkeypatch.setattr(sw, name, lambda: None)
    try:
        assert sw.run_state_watch(SimpleNamespace(), SimpleNamespace(hwnd=0), interval_s=.05) == 0
    finally:
        stopped.set()
        for t in workers:
            t.join(3)
            assert not t.is_alive()
    assert not foreign_tk_calls
    assert len(ready) >= 3


def test_mailbox_latest_close_late_result_and_callback_error():
    from mm_assist.platform.tk_mailbox import TkMailbox

    owner = threading.get_ident()
    scheduled, applied = [], []
    stop = threading.Event()
    def after(_ms, callback):
        assert threading.get_ident() == owner
        scheduled.append(callback)
    box = TkMailbox(SimpleNamespace(after=after), stop=stop)
    def apply(value):
        assert threading.get_ident() == owner
        applied.append(value)
    def fail():
        raise RuntimeError('bad display callback')
    def publish():
        box.post('frame', apply, 'old')
        box.post('error', fail)
        box.post('frame', apply, 'new')
        box.post('warm', apply, 'warm')
    t = threading.Thread(target=publish)
    t.start()
    t.join(2)
    assert not t.is_alive()
    assert applied == []
    scheduled.pop(0)()
    assert applied == ['new', 'warm']
    assert len(scheduled) == 1  # 异常不杀掉下一拍刷新。
    box.post('frame', apply, 'pending at close')
    stop.set()
    box.post('frame', apply, 'late after close')
    scheduled.pop(0)()
    assert applied == ['new', 'warm']
    assert not scheduled
    # 新窗口有独立停旗／邮箱，旧窗口的迟到结果不能进入新窗口。
    new_box = TkMailbox(SimpleNamespace(after=after), stop=threading.Event())
    box.post('frame', apply, 'old generation')
    new_box.post('frame', apply, 'new generation')
    scheduled.pop(0)()
    assert applied == ['new', 'warm', 'new generation']


def test_inspect_worker_releases_read_lock_without_waiting_for_tk(monkeypatch):
    from mm_assist.life_plan import node_inspect as ni
    from mm_assist.life_plan.snapshot import heavy_read_lock
    import tkinter.ttk as ttk

    owner = threading.get_ident()
    callbacks, workers, written = [], [], []
    stop = threading.Event()
    class Widget:
        def __init__(self, *a, **k): pass
        def pack(self, *a, **k): pass
        def __setitem__(self, *a): pass
        def bind(self, *a): pass
        def set(self, *a):
            assert threading.get_ident() == owner
        def configure(self, **k): pass
    def after(ms, fn):
        assert threading.get_ident() == owner, '工作线程不能等待Tk，尤其在持有读锁时'
        callbacks.append((ms, fn))
    monkeypatch.setattr(ttk, 'Combobox', Widget)
    monkeypatch.setattr('mm_assist.perception.ocr.ocr_roi', lambda *a, **k: [])
    monkeypatch.setattr(ni, '_view_or_state', lambda *a: None)
    monkeypatch.setattr(ni, 'compute_inspect', lambda *a, **k:
                        {'node': 'menu', 'stack_ready': True, 'stack_names': ['SystemMenuUIState']})
    monkeypatch.setattr('mm_assist.life_execute.ui_logic.write_inspect_page', lambda node, **k: written.append(node))
    real_thread = threading.Thread
    def thread(*a, **k):
        t = real_thread(*a, **k)
        workers.append(t)
        return t
    monkeypatch.setattr(ni.threading, 'Thread', thread)
    ni.attach_node_inspect(SimpleNamespace(after=after),
                           tk_mod=SimpleNamespace(Frame=Widget, StringVar=Widget, Label=Widget),
                           get_win=lambda: None, grab=lambda: None, mark=None, stop=stop)
    schedule = next(fn for ms, fn in callbacks if ms == 200)
    schedule()
    for t in workers:
        t.join(2)
        assert not t.is_alive()
    assert not written  # UI尚未取队列，但读取锁已释放。
    assert heavy_read_lock.acquire(timeout=.1)
    heavy_read_lock.release()
    drain = next(fn for ms, fn in callbacks if ms == 100)
    drain()
    assert written == ['menu']
    stop.set()
