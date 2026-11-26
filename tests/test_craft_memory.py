"""Craft shell skip regressions: current keeper, not opening state or old pixels."""
from types import SimpleNamespace as NS
import time

import pytest

from mm_assist.life_plan import node_inspect as inspect
from mm_assist.memory.readers import ui_stack as stack


def view(category=1, *, names=None, **updates):
    values = dict(ui_stack_names=names or ["CraftingHubUIInstance", "CraftingHubPopupUIInstance"],
        ui_stack_ready=True, ui_stack_ts=time.time(), ui_stack_error="",
        craft_hub_category=0, craft_current_category=category,
        craft_identity=(0x50000, 0x60000, 0x70000, 0x80000, 0x90000))
    values.update(updates)
    return NS(**values)


@pytest.mark.parametrize("category,node", [(0,"craft_quick"),(1,"craft_hub"),
                                            (2,"craft_make"),(3,"craft_alchemy")])
def test_current_craft_page_bypasses_old_opening_parameter_and_pixels(monkeypatch, category, node):
    def forbidden(*a, **k):
        pytest.fail("active current craft page should not scan pixels/OCR")
    monkeypatch.setattr(inspect, "refine_inspect_node", forbidden)
    v = view(category)
    result = inspect.compute_inspect(object(), view=v)
    assert result["node"] == node
    assert result["craft_mem"] and result["source"] == "stack+mem"
    assert result["cost"]["tpl_n"] == result["cost"]["ocr_n"] == 0
    assert not inspect.inspect_needs_grab("fine:craft_hub", v)
    assert inspect._stack_sn_from_mem(v) == node


def test_craft_tab_helper_uses_current_memory_before_pixels(monkeypatch):
    from mm_assist.life_execute import ui_logic
    monkeypatch.setattr(inspect, "_selected_craft_tab", lambda _: pytest.fail("current tab is verified"))
    assert ui_logic.detect_craft_tab(object(), view(2)) == "make"
    assert ui_logic.detect_craft_tab(None, view(1)) == "process"


def test_opening_category_and_retained_label_do_not_prove_current_tab(monkeypatch):
    from mm_assist.life_execute import ui_logic
    monkeypatch.setattr(inspect, "_selected_craft_tab", lambda _: "")
    old = view(craft_current_category=None, craft_identity=None, craft_tab="加工")
    assert ui_logic.detect_craft_tab(None, old) == ""
    assert ui_logic.detect_craft_tab(object(), old) == ""


@pytest.mark.parametrize("updates", [dict(craft_current_category=None),
    dict(craft_current_category=True),dict(craft_current_category=4),dict(craft_identity=None),
    dict(craft_identity=(0x50000,)),dict(ui_stack_ready=False),dict(ui_stack_ts=0),
    dict(ui_stack_ts=time.time()-10),dict(ui_stack_ts=time.time()+10),
    dict(ui_stack_error="stack_read_reused"),dict(fresh=False),dict(pid_alive=False),
    dict(ui_stack_names=["CraftingHubUIState"]),
    dict(ui_stack_names=["CraftingHubPopupUIInstance","AlteringFacilityPopupUIInstance"]),
    dict(ui_stack_names=["CraftingHubPopupUIInstance","DetailCardUIInstance"]),
    dict(ui_stack_names=["CraftingHubPopupUIInstance","UnknownPopup"])])
def test_invalid_or_covered_craft_fact_does_not_replace_visual_proof(updates):
    v = view(**updates)
    assert inspect._craft_node_from_view(v) == ""
    assert inspect.inspect_needs_grab("fine:craft_hub", v)


def test_same_stack_switch_or_reopen_is_not_a_stable_craft_observation():
    from mm_assist.life_execute.ui_logic import resolve_navigation
    def snap(category=1, identity=(0x50000,0x60000,0x70000,0x80000,0x90000)):
        return stack.UiStackSnap(names=view().ui_stack_names, ready=True,
            craft_current_category=category, craft_identity=identity, sampled_ts=time.time())
    for other in (snap(2),snap(identity=(0xa0000,0x60000,0x70000,0x80000,0x90000)),snap(None)):
        samples=iter([snap(),other])
        result=resolve_navigation(object(),reader=NS(read=lambda **k:next(samples)))
        assert result.node == "unknown" and result.source == "direct_transition"
    samples=iter([snap(),snap()])
    result=resolve_navigation(object(),reader=NS(read=lambda **k:next(samples)))
    assert result.node == "craft_hub" and result.source == "direct+mem"


def test_missing_stack_clears_current_craft_fact():
    from mm_assist.memory.readers.scene import MemSceneInfo
    from mm_assist.perception.scene import apply_stack_bits
    from mm_assist.perception.state_ui import build_state_ui_payload
    ms=MemSceneInfo()
    live=stack.UiStackSnap(names=view().ui_stack_names,ready=True,
        craft_current_category=2,craft_identity=view().craft_identity,sampled_ts=time.time())
    apply_stack_bits(ms,live)
    result=build_state_ui_payload(NS(),ms)
    assert result["craft_current_category"] == 2
    assert inspect._stack_sn_from_mem(ms) == "craft_make"
    for missing in (stack.UiStackSnap(ready=False),
                    stack.UiStackSnap(names=live.names,ready=True,error="stack_read_reused")):
        apply_stack_bits(ms,missing)
        assert ms.craft_current_category is None and ms.craft_identity is None


def test_late_visual_task_does_not_restore_old_craft_page():
    current=inspect.compute_inspect(None,view=view(2))
    old=dict(node="craft_quick",stack_names=view().ui_stack_names,stack_ready=True,
             source="stack+tpl",stack_sampled_ts=time.time()-1,craft_how="tpl")
    held,_=inspect.hold_inspect_payload(current,old)
    assert held["node"] == "craft_make"


def test_current_memory_craft_page_beats_retained_display(monkeypatch):
    from mm_assist.life_execute import ui_logic
    monkeypatch.setattr(ui_logic,"read_inspect_page",lambda **k:"craft_shell")
    monkeypatch.setattr(ui_logic,"read_inspect_record",lambda **k:{"node":"craft_shell","zh":"制作",
        "stack_names":view().ui_stack_names})
    assert ui_logic.resolve_ui_node(view=view(2)).node == "craft_make"


@pytest.mark.parametrize("target", ["craft_hub","craft_make","facility","gather_skill"])
def test_identified_coarse_craft_shell_has_explicit_safe_navigation(target):
    from mm_assist.life_execute.ui_nav import path_edges
    edges=path_edges("craft_shell",target)
    assert edges and edges[0]["from"] == "craft_shell"
    assert edges[0]["to"] in {"craft_hub","craft_make","main_hud"}
    assert all(edge["kind"]=="allowed" for edge in edges)


def test_memory_verified_craft_destination_does_not_require_card_ocr(monkeypatch):
    from mm_assist.life_execute import ui_logic, ui_nav
    monkeypatch.setattr(ui_nav,"craft_hub_visible",lambda *a,**k:pytest.fail("no image gate on memory-verified tab"))
    where=ui_logic.UiWhere("craft_hub",1,0,0,"direct+mem",stack_names=tuple(view().ui_stack_names),
        stack_ready=True,stack_sampled_ts=time.time())
    result=ui_nav.ensure_node("craft_hub",grab=lambda:object(),click=lambda *a:pytest.fail("already there"),
        resolve=lambda frame:where,sleep_fn=lambda _:None)
    assert result.ok


@pytest.mark.parametrize("observed", ["craft_shell", "craft_make"])
def test_tab_click_waits_for_target_without_clearing_gate_or_clicking_again(monkeypatch, observed):
    from mm_assist.life_execute import ui_logic, ui_nav, ui_act
    from mm_assist.safety.autofarm_gates import reset_stage_gates_for_tests
    reset_stage_gates_for_tests()
    ui_nav._EDGE_MEMO.clear()
    ui_nav._PENDING_EDGES.clear()
    clicks = []
    monkeypatch.setattr(ui_nav, "locate_edge", lambda *a, **k: (11, 22))
    monkeypatch.setattr(ui_act, "stack_has_common_popup", lambda **k: False)
    monkeypatch.setattr(ui_act, "stack_has_facility_result", lambda: False)
    def resolve(_):
        return ui_logic.UiWhere(observed if clicks else "craft_quick", 1, 0, 0, "direct",
            stack_names=tuple(view().ui_stack_names), stack_ready=True, stack_sampled_ts=time.time())
    try:
        result = ui_nav.ensure_node("craft_hub", grab=lambda: object(),
            click=lambda *xy: clicks.append(xy) or True, resolve=resolve, sleep_fn=lambda _: None)
        assert result.status == "nav_verify_timeout"
        assert clicks == [(11, 22)]
        assert ui_nav._PENDING_EDGES
    finally:
        reset_stage_gates_for_tests()
        ui_nav._EDGE_MEMO.clear()
        ui_nav._PENDING_EDGES.clear()


def craft_memory(monkeypatch):
    """Actual dump offsets, MMList version differs from count, active window backlinks."""
    reader=stack.Il2CppUiStackReader()
    # This replay retains the pre-update window bytes and all original guards.
    reader._off.update({'UIStackingPlayer.instance':0x20,'UnityMMMLWindow.closing':0x80,
        'UnityMMMLWindow.root':0x98,'UnityMMMLWindow.root_id':0xa0,
        'UnityMMMLWindow.id':0x130,'UnityMMMLWindow.scope':0x1a8})
    reader._h,reader._ga=object(),0x900000
    pointers={0x9000b8:0x910000,0x910050:0x10000,0x10018:0x20000,0x20010:0x30000,
        0x30020:0x40000,0x40020:0x50000,0x50048:0x60000,0x500c8:0x80000,
        0x500b8:0x90000,0x500c0:0xa0000,0x50028:0x90000,0x80018:0x90000,
        0x80020:0x60000,0x90060:0xa0000,0x60030:0x70000,0x90010:0xb0000,
        0x90058:0xc0000,0xc0050:0xd0000,0xd0010:0xe0000,0xd0018:0xf0000,
        0xf0020:0x110000,0x110098:0xb0000,0x1101a8:0x120000}
    values={0x20018:86,0x2001c:1,0x30018:4,0x90068:1,0x7001c:2,
        0x90018:7,0x90040:1,0x90044:9,0xd0034:1,0xe0018:4,0xf0018:4,
        0xe0020:9,0x110080:0,0x120020:0,0x1100a0:7,0x110130:9}
    names={0x10000:"UIStackingService",0x40000:"UIStackingPlayer",0x50000:"CraftingHubPopupUIInstance",
        0x60000:"CraftingHubUIComponent",0x80000:"CraftingHubPopupUIViewModelComponent",
        0x90000:"MMMLUIHandle",0xa0000:"MMMLUIViewModel`2",0x70000:"CraftingHubUIKeeper",
        0xb0000:"XForest",0xc0000:"MMMLService",0x110000:"UnityMMMLWindow",0x120000:"UIResourceScope"}
    monkeypatch.setattr(stack,"resolve_klass",lambda *a:0x900000)
    monkeypatch.setattr(stack,"read_ptr",lambda h,p:pointers.get(p))
    monkeypatch.setattr(stack,"read_u32",lambda h,p:values.get(p))
    monkeypatch.setattr(stack,"read_u8",lambda h,p:values.get(p))
    monkeypatch.setattr(stack,"klass_simple_name",lambda h,p,o:names.get(p,""))
    return reader,pointers,values,names


def test_bounded_active_craft_reader_uses_keeper_and_rejects_closed_window(monkeypatch):
    reader,pointers,values,names=craft_memory(monkeypatch)
    expected=(0x50000,0x60000,0x70000,0x80000,0x90000)
    assert reader._active_craft(0x10000)==(2,expected)
    values[0x30018]=64  # Retained capacity does not mean 64 active players.
    assert reader._active_craft(0x10000)==(2,expected)
    values[0x30018]=4
    values[0x7001c]=1
    assert reader._active_craft(0x10000)==(1,expected)
    values[0x90018]=values[0x1100a0]=None
    assert reader._active_craft(0x10000) is None
    values[0x90018]=values[0x1100a0]=7
    for address,bad in [(0x90068,0),(0x7001c,4),(0x2001c,5),(0x90040,0),
                        (0x110080,1),(0x120020,1),(0x110130,10),(0x1100a0,8)]:
        original=values[address];values[address]=bad
        assert reader._active_craft(0x10000) is None
        values[address]=original
    for address in (0x50048,0x500b8,0x80020,0x50028,0x110098):
        original=pointers[address];pointers[address]=0
        assert reader._active_craft(0x10000) is None
        pointers[address]=original
    names[0x70000]="WrongKeeper"
    assert reader._active_craft(0x10000) is None


def test_active_craft_rejects_concurrent_switch_and_covered_popup(monkeypatch):
    reader,pointers,values,names=craft_memory(monkeypatch)
    calls=0
    def changing(h,addr):
        nonlocal calls
        if addr==0x7001c:
            calls+=1
            return 2 if calls==1 else 1
        return values.get(addr)
    monkeypatch.setattr(stack,"read_u32",changing)
    assert reader._active_craft(0x10000) is None
    monkeypatch.setattr(stack,"read_u32",lambda h,p:values.get(p))
    values[0x2001c]=2;pointers[0x30028]=0x130000;pointers[0x130020]=0x140000
    names[0x130000]="UIStackingPlayer";names[0x140000]="AlteringFacilityPopupUIInstance"
    assert reader._active_craft(0x10000) is None


def test_current_craft_change_updates_display_without_weak_three_frame_hold():
    before=inspect.compute_inspect(None,view=view(0))
    after=inspect.compute_inspect(None,view=view(2))
    held,_=inspect.hold_inspect_payload(before,after)
    assert held["node"] == "craft_make"
