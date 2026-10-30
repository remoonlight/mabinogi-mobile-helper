"""Native controls exercise the real controller without game reads or input."""
import tkinter as tk

import pytest

from mm_assist.life_plan import bar_ui, controller, execution_records as records
from mm_assist.life_plan.types import LifeSnapshot, PlanItem


@pytest.fixture(scope='module')
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def panel(monkeypatch, tk_root):
    root = tk.Toplevel(tk_root)
    root.withdraw()
    controller.reset_controller_for_tests()
    first = PlanItem('p1', 'gather', '采集', recipe='高級羊毛')
    following = PlanItem('p2', 'gather', '采集', recipe='高級羊毛')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [first, following])
    monkeypatch.setattr(bar_ui, 'pump_thread_alive', lambda: False)
    monkeypatch.setattr(bar_ui, 'repair_busy', lambda: False)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: '')
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly', lambda **k: None)
    def prepare(*a, **kwargs):
        token, _ = controller.mark_life_preparing(pause_on_failure=kwargs['pause_on_failure'])
        controller.apply_prepared_snapshot(LifeSnapshot(ok=True), token, now=1000)
        return True
    monkeypatch.setattr(bar_ui, 'start_life_prepare', prepare)
    bar_ui.attach_life_plan_controls(root, tk_mod=tk)
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)
    buttons = {w.cget('text'): w for w in descendants(root) if isinstance(w, tk.Button)}
    yield root, buttons, first, following
    controller.reset_controller_for_tests()
    root.destroy()


def test_debug_button_failure_resume_preserves_run_queue_and_tracking(panel):
    root, buttons, first, following = panel
    buttons['生活備料（調試）'].invoke()
    started = controller.get_controller().life_started_at
    for _ in range(2):
        state = controller.get_controller()
        assert state.running and state.pause_on_failure
        assert state.life_started_at == started and state.queue == [first, following]
        bar_ui._note_fail_and_continue(win=None, item=first, reason='no_process_btn')
        state = controller.get_controller()
        assert state.paused and state.current == first and state.queue == [first, following]
        buttons['恢复'].invoke()
    assert controller.get_controller().pause_on_failure
    buttons['暂停'].invoke()
    buttons['生活备料'].invoke()
    assert not controller.get_controller().pause_on_failure


@pytest.mark.parametrize('tracking', [False, True])
def test_native_start_buttons_select_recovery_skip_or_pause(panel, monkeypatch, tracking):
    from mm_assist.life_plan.types import ExecuteResult
    root, buttons, first, following = panel
    calls, recoveries = [], []
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda _: True)
    def execute(item, **kwargs):
        calls.append(item.id)
        return ExecuteResult(False, 'no_process_btn' if item == first else 'cancel')
    def recover(*a, **kwargs):
        stamp = 1001+len(recoveries)
        kwargs['on_dispatch'](stamp)
        recoveries.append(stamp)
        # Inject independent post-Esc verification; actual input is forbidden.
        return {'ok': True, 'esc_sent': True, 'sent_at': stamp}
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', recover)
    bar_ui.reset_pump_settle()
    bar_ui.reset_recovery_for_tests()
    buttons['生活備料（調試）' if tracking else '生活备料'].invoke()
    run = controller.get_controller().life_started_at
    bar_ui.pump_life_once(None, None)
    state = controller.get_controller()
    assert state.life_started_at == run and state.pause_on_failure is tracking
    if tracking:
        assert calls == [first.id] and recoveries == []
        assert state.paused and state.current == first and state.queue == [first, following]
        buttons['恢复'].invoke()
        assert controller.get_controller().pause_on_failure
    else:
        assert calls == [first.id]*3 and len(recoveries) == 2
        assert state.running and not state.paused and state.current == following
        assert first not in state.queue and state.pending_action is None
        bar_ui.pump_life_once(None, None)
        assert calls == [first.id]*3+[following.id]
    bar_ui.reset_pump_settle()
    bar_ui.reset_recovery_for_tests()


def test_records_can_scroll_to_old_plans_and_detail(panel, monkeypatch):
    root, _, _, _ = panel
    rows = [dict(pid=1, run_started_at=1000, plan_id=f'p{i}', plan='采集',
        status='paused', started_at=1000+i, elapsed_s=1, stp_wait_s=0, paused_s=0)
        for i in range(100)]
    from types import SimpleNamespace
    monkeypatch.setattr(records, '_get', lambda: SimpleNamespace(read=lambda: rows))
    monkeypatch.setattr(records, 'format_detail', lambda row: '\n'.join(str(i) for i in range(200)))
    records.show_records(root)
    window = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
    window.update()
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)
    widgets = list(descendants(window))
    from tkinter import ttk
    tree = next(w for w in widgets if isinstance(w, ttk.Treeview))
    detail = next(w for w in widgets if isinstance(w, tk.Text))
    bars = [w for w in widgets if isinstance(w, ttk.Scrollbar) and str(w.cget('orient')) == 'vertical']
    assert len(bars) == 2
    list_bar = next(w for w in bars if w.master == tree.master)
    window.tk.call(list_bar.cget('command'), 'moveto', 1.0)
    window.update()
    assert tree.yview()[0] > 0.5
    tree.selection_set(tree.get_children()[-1])
    tree.event_generate('<<TreeviewSelect>>')
    window.update()
    detail_bar = next(w for w in bars if w.master == detail.master)
    window.tk.call(detail_bar.cget('command'), 'moveto', 1.0)
    assert detail.yview()[0] > 0.5 and detail.cget('state') == 'disabled'
    window.destroy()
