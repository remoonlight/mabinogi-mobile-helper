"""Failed items may not be rebuilt from the same material observation."""
from dataclasses import replace

import pytest


@pytest.mark.parametrize('scan_started,expected',[(1009,10),(1011,20)])
def test_scan_during_feed_keeps_reservation_but_later_scan_replaces_it(monkeypatch,scan_started,expected):
    first = PlanItem('p1','feed','汁液',facility='medicine',recipe='淨淨蘑菇汁液')
    following = replace(first,id='p2')
    snap = LifeSnapshot(ok=True,items={'淨淨蘑菇孢子':15,'裝水的瓶子':7},taken_at=1000)
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:[first,following])
    controller.start_life(snapshot=snap,now=1000)
    controller.acknowledge_done(first.id,now=1010)
    # Duplicate acknowledgement must not reserve the same feed twice.
    controller.acknowledge_done(first.id,now=1010.5)
    assert controller.execution_items(controller.get_controller())['淨淨蘑菇孢子'] == 5
    monkeypatch.setattr(controller,'take_snapshot',lambda **k:replace(snap,
        items={'淨淨蘑菇孢子':20,'裝水的瓶子':8},taken_at=scan_started))
    refreshed = controller.tick_facility_progress(now=1601)
    assert controller.execution_items(refreshed)['淨淨蘑菇孢子'] == 5
    controller.acknowledge_done(following.id, now=1602)
    refreshed = controller.tick_facility_progress(now=1603)
    expected -= 10  # the second feed is also reserved until a newer scan
    assert controller.execution_items(refreshed)['淨淨蘑菇孢子'] == expected
    restarted = controller.start_life(snapshot=snap,now=2000)
    assert controller.execution_items(restarted)['淨淨蘑菇孢子'] == 15


def test_completed_feed_cost_is_unavailable_to_next_plan_until_new_scan(monkeypatch):
    from mm_assist.life_plan.types import ExecuteResult
    first = PlanItem('p1','feed','汁液',facility='medicine',recipe='淨淨蘑菇汁液')
    following = PlanItem('p2','feed','汁液',facility='medicine',recipe='淨淨蘑菇汁液')
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:[first,following])
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly',lambda **k:None)
    snap = LifeSnapshot(ok=True,items={'淨淨蘑菇孢子':15,'裝水的瓶子':7},taken_at=1000)
    controller.start_life(snapshot=snap,now=1000)
    seen = []
    def execute(item,**kwargs):
        seen.append(kwargs['items'])
        return ExecuteResult(True,'done',action_issued=True)
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute',execute)
    bar_ui.pump_life_once(None,None)
    bar_ui.pump_life_once(None,None)
    assert seen[0]['淨淨蘑菇孢子'] == 15
    assert seen[1]['淨淨蘑菇孢子'] == 5 and seen[1]['裝水的瓶子'] == 6
    assert snap.items['淨淨蘑菇孢子'] == 15  # read evidence is never rewritten
    controller.reset_controller_for_tests()

from mm_assist.life_plan import controller, bar_ui
from mm_assist.life_plan.types import LifeSnapshot, PlanItem


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    controller.reset_controller_for_tests()
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly', lambda **k: None)
    monkeypatch.setattr('mm_assist.life_plan.snapshot.refresh_tools', lambda: ())
    yield
    controller.reset_controller_for_tests()


@pytest.mark.parametrize('durability', [0, 1, 10, None, 'empty'])
def test_failed_training_checks_fresh_tools_before_next_action(monkeypatch, durability):
    from mm_assist.life_plan import snapshot
    from mm_assist.life_plan.types import ExecuteResult, ToolDurability
    targets = {'合金鋼錠':100, '高級木材':100, '高級布料':100, '淨淨蘑菇汁液':150,
               '高級混合強化秘藥(山崩)':100,
               '增幅蘑菇汁液':150, '箭花':250, '血紅藥草':250, '滴答螢火蟲':250, '獨角仙':250}
    original = controller.start_life(snapshot=LifeSnapshot(ok=True, items=targets,
        tools=(ToolDurability('舊讀數', 10),)), now=1000).queue
    order = []
    tools = (None if durability is None else () if durability == 'empty'
             else (ToolDurability('採集工具', durability),))
    def read():
        order.append(('read', controller.get_controller().current.id))
        return tools
    def execute(item, **kwargs):
        order.append(('execute', item.kind, item.id))
        return ExecuteResult(False, 'fixture_fail', '采集动作失败')
    monkeypatch.setattr(snapshot, 'refresh_tools', read)
    monkeypatch.setattr(snapshot, 'take_snapshot', lambda **k: pytest.fail('no bag scan'))
    monkeypatch.setattr('mm_assist.life_plan.execute.try_execute', execute)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot',
                        lambda *a, **k: order.append(('shot',)) or None)
    monkeypatch.setattr('mm_assist.life_execute.ui_recover.recover_once',
        lambda *a, **k: order.append(('recover',)) or {'ok': False})
    bar_ui.pump_life_once(None, None)
    st = controller.get_controller()
    # Updated user requirement: preserve the same item for two recoveries;
    # final failure still reads durability once before selecting the next item.
    assert order == [('execute', 'gather', original[0].id), ('shot',),
                     ('recover',), ('recover',), ('shot',), ('read', original[0].id)]
    assert st.running and not st.paused
    training = [p for p in st.queue if p.gather_mode == 'skill']
    assert [(p.skill, p.detail, p.recipe) for p in training] == [(p.skill, p.detail, p.recipe) for p in original[1:]]
    if durability != 0:
        assert [p.id for p in training] == [p.id for p in original[1:]]
    assert [int(p.id[1:]) for p in st.queue] == sorted(int(p.id[1:]) for p in st.queue)
    assert (st.current.kind == 'repair') == (durability == 0)
    assert len([p for p in st.queue if p.kind == 'repair']) == (1 if durability == 0 else 0)
    assert st.training_counts == {}


@pytest.mark.parametrize('durability', [0, 10])
def test_tracking_training_failure_checks_tools_and_preserves_failed_item(monkeypatch, durability):
    from mm_assist.life_plan import snapshot
    from mm_assist.life_plan.types import ToolDurability
    targets = {'合金鋼錠':100, '高級木材':100, '高級布料':100, '淨淨蘑菇汁液':150,
               '高級混合強化秘藥(山崩)':100,
               '增幅蘑菇汁液':150, '箭花':250, '血紅藥草':250, '滴答螢火蟲':250, '獨角仙':250}
    st = controller.start_life(snapshot=LifeSnapshot(ok=True, items=targets), now=1000,
                               pause_on_failure=True)
    calls = []
    tools = (ToolDurability('採集工具', durability),)
    monkeypatch.setattr(snapshot, 'refresh_tools', lambda: calls.append('read') or tools)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: None)
    bar_ui._note_fail_and_continue(win=None, item=st.current, reason='fixture_fail')
    after = controller.get_controller()
    assert calls == ['read']
    assert after.paused and after.current == st.current and after.queue == st.queue
    assert after.snapshot.tools == tools


@pytest.mark.parametrize('interruption', ['pause', 'start'])
def test_failure_tools_late_result_cannot_change_paused_or_new_run(monkeypatch, interruption):
    from mm_assist.life_plan import snapshot
    from mm_assist.life_plan.types import ToolDurability
    item = PlanItem('p1', 'gather', '练采集', skill='日常採集', detail='蜘蛛網', gather_mode='skill')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item])
    initial = LifeSnapshot(ok=True, tools=(ToolDurability('原工具', 10),))
    controller.start_life(snapshot=initial, now=1000)
    calls = []
    def read():
        calls.append('read')
        if interruption == 'pause':
            controller.pause_life()
        else:
            controller.start_life(snapshot=initial, now=2000)
        return (ToolDurability('舊代結果', 0),)
    monkeypatch.setattr(snapshot, 'refresh_tools', read)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: None)
    bar_ui._note_fail_and_continue(win=None, item=item, reason='fixture_fail')
    after = controller.get_controller()
    assert calls == ['read']
    assert after.snapshot.tools == initial.tools and after.queue == [item]
    assert after.paused == (interruption == 'pause')


@pytest.mark.parametrize('kind,mode', [('feed', None), ('gather', 'process_mat')])
def test_other_plan_failures_do_not_read_tools(monkeypatch, kind, mode):
    from mm_assist.life_plan import snapshot
    item = PlanItem('p1', kind, '计划', gather_mode=mode)
    following = PlanItem('p2', 'gather', '下一项')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item, following])
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000)
    monkeypatch.setattr(snapshot, 'refresh_tools', lambda: pytest.fail('only failed training checks tools'))
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: None)
    bar_ui._note_fail_and_continue(win=None, item=item, reason='fixture_fail')
    assert controller.get_controller().current == following


def test_old_periodic_tools_cannot_override_failure_check(monkeypatch):
    from mm_assist.life_plan import snapshot
    from mm_assist.life_plan.types import ToolDurability
    targets = {'合金鋼錠':100, '高級木材':100, '高級布料':100, '淨淨蘑菇汁液':150,
               '高級混合強化秘藥(山崩)':100,
               '增幅蘑菇汁液':150, '箭花':250, '血紅藥草':250, '滴答螢火蟲':250, '獨角仙':250}
    original = LifeSnapshot(ok=True, items=targets, taken_at=1000,
                            tools=(ToolDurability('工具', 10),))
    st = controller.start_life(snapshot=original, now=1000)
    low = (ToolDurability('工具', 0),)
    monkeypatch.setattr(snapshot, 'refresh_tools', lambda: low)
    monkeypatch.setattr(controller.time, 'time', lambda: 1002)
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: None)
    bar_ui._note_fail_and_continue(win=None, item=st.current, reason='fixture_fail')
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(original, taken_at=1001))
    after = controller.tick_facility_progress(now=1003, force_scan=True)
    assert after.snapshot.tools == low and after.current.kind == 'repair'


@pytest.mark.parametrize('item', [
    PlanItem('p1', 'feed', '高級布料', facility='cloth', recipe='高級布料'),
    PlanItem('p1', 'repair', '修理'),
    PlanItem('p1', 'gather', '采集', recipe='高級羊毛'),
])
def test_skip_keeps_queued_duplicate_but_does_not_generate_another(monkeypatch, item):
    monkeypatch.setattr(controller.time, 'time', lambda: 1000.5)
    duplicate = replace(item, id='p2')
    snap = LifeSnapshot(ok=True, taken_at=1000)
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item, duplicate])
    monkeypatch.setattr(controller, 'training_item', lambda *a, **k: item)
    controller.start_life(snapshot=snap, now=1000)
    assert controller.skip_current(reason='failure').current == duplicate
    st = controller.skip_current(reason='failure')
    assert st.running and st.current is None and st.queue == []
    # Exhaustion requests a fresh observation immediately, before 600 seconds.
    reads = []
    def read(**kwargs):
        reads.append(kwargs)
        return replace(snap, taken_at=kwargs['now'])
    monkeypatch.setattr(controller, 'take_snapshot', read)
    st = controller.tick_facility_progress(now=1001)
    assert len(reads) == 1 and reads[0]['include_tools'] is True
    assert st.current is not None and st.current.kind == item.kind


def test_successful_training_can_continue_and_new_start_clears_failure(monkeypatch):
    item = PlanItem('p1', 'gather', '训练', skill='草藥採集', gather_mode='skill')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item])
    monkeypatch.setattr(controller, 'training_item', lambda *a, **k: item)
    snap = LifeSnapshot(ok=True)
    controller.start_life(snapshot=snap, now=1000)
    st = controller.acknowledge_done(item.id)
    assert st.current is None and st.refresh_requested
    monkeypatch.setattr(controller, 'take_snapshot', lambda **k: replace(snap, taken_at=k['now']))
    st = controller.tick_facility_progress(now=controller.time.time()+1)
    assert st.current is not None  # A fresh scan permits the next batch.
    for _ in range(20):
        if st.current is None:
            break
        st = controller.skip_current(reason='failure')
    assert st.current is None
    assert controller.start_life(snapshot=snap, now=2000).current == item


def _record_failure(monkeypatch, *, reason, extra, tracking):
    """同一份前置及失败留证，模式差异由两个测试的独立断言表达。"""
    item = PlanItem('p1', 'gather', '采集', recipe='高級羊毛')
    following = replace(item, id='p2')
    monkeypatch.setattr(controller, 'plan_life', lambda *a, **k: [item, following])
    monkeypatch.setattr('mm_assist.life_plan.anomalies.cache_fail_shot', lambda *a, **k: '')
    records = []
    monkeypatch.setattr('mm_assist.life_plan.anomalies.log_life_anomaly',lambda **k:records.append(k))
    controller.start_life(snapshot=LifeSnapshot(ok=True), now=1000, pause_on_failure=tracking)
    bar_ui._note_fail_and_continue(win=None, item=item, reason=reason, extra=extra)
    return controller.get_controller(), item, following, records[-1]


@pytest.mark.parametrize('reason,extra', [('not_foreground', ''),
    ('error', 'click_pointer_moved: yield to mouse input'), ('human_busy', ''),
    ('error','click_focus_lost: cannot foreground game'), ('error','focus_click_timeout'),
    ('error','click_hud_occluded: no visible space outside game'),
    ('click_ui_blocked','系统弹层未退出，不点底层'),
    ('error','click_ui_blocked: ToyTransparentLayerUIInstance'),
    ('error','click_ui_not_ready: 当前界面栈未就绪'), ('btn_unstable',''),
    ('nav_verify_timeout','')])
def test_normal_failure_skips_current_and_preserves_each_reason(monkeypatch, reason, extra):
    st, item, following, record = _record_failure(monkeypatch, reason=reason, extra=extra, tracking=False)
    assert st.running and not st.paused
    assert st.queue == [following] and st.current == following
    assert record['tag'] == reason
    assert record['extra'] == extra
    assert record['plan_id'] == item.id


@pytest.mark.parametrize('reason,extra', [('not_foreground', ''),
    ('error', 'click_focus_lost: cannot foreground game'), ('nav_verify_timeout', '')],
    ids=['input-refused', 'input-error-with-detail', 'navigation-failed'])
def test_tracking_failure_keeps_current_and_original_evidence(monkeypatch, reason, extra):
    # 跟踪在失败原因分类之前直接暂停；保留空/非空详情及业务/输入失败。
    st, item, following, record = _record_failure(monkeypatch, reason=reason, extra=extra, tracking=True)
    assert st.paused and not st.running
    assert st.queue == [item,following] and st.current == item
    assert record['tag'] == 'tracking_failure_paused'
    assert record['stage'] == reason
    assert record['extra'] == extra
    assert record['plan_id'] == item.id


def test_normal_resume_and_start_clear_tracking(monkeypatch):
    item = PlanItem('p1','gather','采集',recipe='高級羊毛')
    snap = LifeSnapshot(ok=True)
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:[item])
    controller.start_life(snapshot=snap,now=1000,pause_on_failure=True)
    controller.pause_life()
    assert not controller.resume_life().pause_on_failure
    controller.pause_life()
    assert controller.resume_life(pause_on_failure=True).pause_on_failure
    assert not controller.start_life(snapshot=snap,now=2000).pause_on_failure


@pytest.mark.parametrize('cmd',['start','resume'])
@pytest.mark.parametrize('tracking',[False,True])
def test_ipc_mode_flows_to_controller(tmp_path,monkeypatch,cmd,tracking):
    from mm_assist.life_plan.cmd_file import request_life_cmd,consume_life_cmd
    snap = LifeSnapshot(ok=True)
    item = PlanItem('p1','gather','采集',recipe='高級羊毛')
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:[item])
    controller.start_life(snapshot=snap,now=50,pause_on_failure=True)
    controller.pause_life()
    request_life_cmd(cmd,root=tmp_path,now=100,pause_on_failure=tracking)
    def start(**kwargs):
        token,_ = controller.mark_life_preparing(**kwargs)
        controller.apply_prepared_snapshot(snap,token,now=100)
    assert consume_life_cmd(root=tmp_path,now=100,start_fn=start,
        resume_fn=controller.resume_life) == cmd
    st = controller.get_controller()
    assert st.running and st.pause_on_failure == tracking
    assert ('测试跟踪：失败时暂停' in bar_ui.format_life_plan_block(st)) == tracking


def test_late_preparation_cannot_restore_old_tracking_mode(monkeypatch):
    item = PlanItem('p1','gather','采集',recipe='高級羊毛')
    monkeypatch.setattr(controller,'plan_life',lambda *a,**k:[item])
    old,_ = controller.mark_life_preparing(pause_on_failure=True)
    new,_ = controller.mark_life_preparing()
    controller.apply_prepared_snapshot(LifeSnapshot(ok=True),new,now=200)
    st = controller.apply_prepared_snapshot(LifeSnapshot(ok=True),old,now=100)
    assert not st.pause_on_failure and st.life_started_at == 200
