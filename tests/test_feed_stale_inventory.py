"""p9: claimed cloth exists in memory but the batch's old stock is insufficient."""
import pytest
from mm_assist.life_plan.execute import try_execute
from mm_assist.life_plan.types import PlanItem, ExecuteResult
from mm_assist.life_execute import medicine_process as process
from mm_assist.life_plan import snapshot


@pytest.mark.parametrize('fresh,expected', [({'布料':5,'羊毛':99},'observed'),
    ({'布料':2,'羊毛':99},'no_mats'), (None,'items_unknown')])
def test_dispatch_rechecks_short_stock_before_any_navigation(monkeypatch,fresh,expected):
    calls=[]
    monkeypatch.setattr('mm_assist.life_execute.ui_act.pump_aborted',lambda:False)
    monkeypatch.setattr(process,'_run_process',lambda *a,**k:
        calls.append(k['items']) or ExecuteResult(True,'observed','read evidence'))
    item=PlanItem('p9','feed','投料 布料+',facility='cloth',recipe='布料+')
    reads=[]
    result=try_execute(item,items={'布料':2,'羊毛':99},
        read_items=lambda:reads.append(1) or fresh,do_click=lambda *a:pytest.fail('input'))
    assert result.status==expected and reads==[1]
    assert calls==([fresh] if expected=='observed' else [])


def test_pause_during_stock_refresh_does_not_navigate(monkeypatch):
    paused=[False]
    monkeypatch.setattr('mm_assist.life_execute.ui_act.pump_aborted',lambda:paused[0])
    monkeypatch.setattr(process,'_run_process',lambda *a,**k:pytest.fail('navigation'))
    def read():paused[0]=True;return {'布料':5,'羊毛':99}
    result=try_execute(PlanItem('p9','feed','投料 布料+',facility='cloth',recipe='布料+'),
        items={},read_items=read)
    assert result.status=='cancel'


def test_current_stock_read_is_paired_and_never_heap_scans(monkeypatch):
    from mm_assist.memory.readers import consumable
    closed=[];reads=[]
    monkeypatch.setattr(snapshot,'find_game_pid',lambda:1234)
    monkeypatch.setattr(snapshot,'open_process',lambda pid:7)
    monkeypatch.setattr(snapshot,'close_handle',closed.append)
    def read(h,pid,*,allow_scan=True):
        assert (h,pid,allow_scan)==(7,1234,False)
        reads.append(1);return {'布料':5,'羊毛':99}
    monkeypatch.setattr(consumable,'read_consumable_counts',read)
    assert snapshot.refresh_items()=={'布料':5,'羊毛':99}
    assert reads==[1,1] and closed==[7]


@pytest.mark.parametrize('values',[[{'布料':5},{'布料':2}],[None,None]])
def test_changed_or_unreadable_stock_is_unknown_and_unlocks(monkeypatch,values):
    from mm_assist.memory.readers import consumable
    monkeypatch.setattr(snapshot,'find_game_pid',lambda:1234)
    monkeypatch.setattr(snapshot,'open_process',lambda pid:7)
    monkeypatch.setattr(snapshot,'close_handle',lambda h:None)
    values=iter(values)
    monkeypatch.setattr(consumable,'read_consumable_counts',lambda *a,**k:next(values))
    assert snapshot.refresh_items() is None
    assert not snapshot.snapshot_in_progress() and not snapshot.heavy_read_lock.locked()


def test_busy_stock_reader_does_not_create_another_read(monkeypatch):
    monkeypatch.setattr(snapshot,'open_process',lambda *a:pytest.fail('second read'))
    snapshot._SCAN_LOCK.acquire()
    try:
        with pytest.raises(snapshot.ItemsReadBusy):
            snapshot.refresh_items(budget_s=0.001)
    finally:snapshot._SCAN_LOCK.release()


@pytest.mark.parametrize('tracking', [False, True])
def test_pump_stock_lock_wait_preserves_plan_without_recovery_or_input(monkeypatch, tracking):
    from types import SimpleNamespace
    from mm_assist.life_plan import controller, bar_ui
    from mm_assist.life_plan.types import LifeSnapshot

    controller.reset_controller_for_tests()
    plan = PlanItem('p2', 'feed', '投料 布料+', facility='cloth', recipe='布料+')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [plan])
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda win: True)
    monkeypatch.setattr(bar_ui, '_note_fail_and_continue', lambda **k: pytest.fail('读锁忙不当业务失败'))
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once', lambda *a, **k: pytest.fail('只读等待，不能Esc'))
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: 'offline.png')
    monkeypatch.setattr(snapshot, 'open_process', lambda *a: pytest.fail('不得另开读取'))
    monkeypatch.setattr(process, '_run_process', lambda *a, **k: pytest.fail('库存未确认，不能导航'))
    controller.start_life(snapshot=LifeSnapshot(ok=True, items={'布料': 2, '羊毛': 99}),
                          now=1000, pause_on_failure=tracking)
    real_refresh = snapshot.refresh_items
    monkeypatch.setattr(snapshot, 'refresh_items', lambda: real_refresh(budget_s=.001))
    snapshot.heavy_read_lock.acquire()
    try:
        bar_ui.pump_life_once(None, SimpleNamespace(hwnd=0))
    finally:
        snapshot.heavy_read_lock.release()
    st = controller.get_controller()
    assert st.running and not st.paused and st.current == plan and st.queue == [plan]
    assert st.pending_action is None and st.pause_on_failure == tracking
    assert '库存读取' in st.current_action
    # 锁释放后只凭本次实际双读数据继续；不靠stp变化唤醒，不记一次已发动作。
    fresh = {'布料': 5, '羊毛': 99}
    observed = []
    monkeypatch.setattr(snapshot, 'refresh_items', lambda: dict(fresh))
    monkeypatch.setattr(process, '_run_process', lambda *a, **k:
                        observed.append(k['items']) or ExecuteResult(True, 'done'))
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=0))
    assert observed == [fresh]
    assert not controller.get_controller().queue


@pytest.mark.parametrize('interrupt', ['pause', 'new_start'])
def test_stock_wait_late_result_cannot_change_paused_or_new_run(monkeypatch, interrupt):
    from mm_assist.life_plan import controller, bar_ui
    from mm_assist.life_plan.types import LifeSnapshot
    from types import SimpleNamespace

    controller.reset_controller_for_tests()
    plan = PlanItem('p2', 'feed', '投料 布料+', facility='cloth', recipe='布料+')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [plan])
    monkeypatch.setattr(bar_ui, '_execution_page_ready', lambda win: True)
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000, pause_on_failure=True)
    interrupted = []
    def read():
        if interrupt == 'pause':
            controller.pause_life()
        else:
            controller.start_life(snapshot=LifeSnapshot(ok=True), now=2000, pause_on_failure=True)
        interrupted.append(controller.get_controller().current_action)
        raise snapshot.ItemsReadBusy('heavy')
    monkeypatch.setattr(snapshot, 'refresh_items', read)
    monkeypatch.setattr(process, '_run_process', lambda *a, **k: pytest.fail('不能导航'))
    bar_ui.pump_life_once(None, SimpleNamespace(hwnd=0))
    st = controller.get_controller()
    assert st.current == plan and st.pending_action is None
    if interrupt == 'pause': assert st.paused and not st.running
    else: assert st.life_started_at == 2000 and st.running and st.current_action == interrupted[0]


@pytest.mark.parametrize('failure',['exception','pid_changed','deadline'])
def test_stock_failure_releases_all_locks(monkeypatch,failure):
    from mm_assist.memory.readers import consumable
    from mm_assist.memory import il2cpp
    calls=[];closed=[];clock=[10.0]
    monkeypatch.setattr(snapshot.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(snapshot,'find_game_pid',lambda:4321 if failure=='pid_changed' and len(calls)==2 else 1234)
    monkeypatch.setattr(snapshot,'open_process',lambda pid:7)
    monkeypatch.setattr(snapshot,'close_handle',closed.append)
    def read(*a,**k):
        calls.append(1)
        if failure=='exception':raise RuntimeError('unreadable')
        if failure=='deadline':clock[0]+=1
        return {'布料':5}
    monkeypatch.setattr(consumable,'read_consumable_counts',read)
    assert snapshot.refresh_items() is None
    assert closed==[7] and not snapshot.snapshot_in_progress() and not snapshot.heavy_read_lock.locked()
    assert not il2cpp._UI_MEMORY_READ_PENDING.is_set()


def test_missing_direct_bag_does_not_fall_back_to_heap(monkeypatch):
    from mm_assist.memory.readers import consumable as bag
    monkeypatch.setattr(bag,'_BAG_PID',0);monkeypatch.setattr(bag,'_BAG_COMP',0)
    monkeypatch.setattr(bag,'module_base',lambda *a:1)
    monkeypatch.setattr(bag,'resolve_klass',lambda *a:0x10000)
    monkeypatch.setattr(bag,'load_hint',lambda *a:0)
    monkeypatch.setattr(bag,'read_sc_component',lambda *a:None)
    monkeypatch.setattr(bag,'scan_klass_objects',lambda *a,**k:pytest.fail('heap fallback'))
    assert bag.read_consumable_counts(7,1234,allow_scan=False) is None
