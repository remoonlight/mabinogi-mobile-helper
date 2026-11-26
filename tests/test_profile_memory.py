"""Current profile tabs: active instance, fresh facts, and same-stack transitions."""
from types import SimpleNamespace as NS
import time

import pytest

from mm_assist.life_plan import node_inspect as inspect
from mm_assist.memory.readers import ui_stack as stack


def view(category=0, *, names=None, ts=None, identity=(0x50000, 0x60000)):
    return NS(ui_stack_names=names or ["NewProfileUIInstance"], ui_stack_ready=True,
              ui_stack_empty_ok=False, ui_stack_ts=time.time() if ts is None else ts,
              profile_category=category, profile_identity=identity)


@pytest.mark.parametrize("category,node", [(0, "profile:info"), (1, "profile:combat"), (2, "life_grid"),
                                          (3, "profile:career")])
def test_current_profile_tab_without_pixels(monkeypatch, category, node):
    def forbidden(*a, **k):
        pytest.fail("verified profile shell must not use image recognition")
    for name in ("_tmpl", "_ocr_blob", "_gather_detail_visible", "_life_grid_confirmed"):
        monkeypatch.setattr(inspect, name, forbidden)
    v = view(category)
    result = inspect.compute_inspect(object(), view=v)
    assert result["node"] == node
    assert result["source"] == "stack+mem"
    assert result["cost"]["tpl_n"] == result["cost"]["ocr_n"] == 0
    assert not inspect.inspect_needs_grab("profile", v)


@pytest.mark.parametrize("updates", [dict(profile_category=4), dict(profile_category=-1),
    dict(profile_category=True), dict(profile_category=None), dict(ui_stack_ready=False),
    dict(ui_stack_ts=0), dict(ui_stack_ts=time.time()-10), dict(ui_stack_ts=time.time()+10),
    dict(fresh=False), dict(pid_alive=False), dict(profile_identity=None),
    dict(ui_stack_names=["NewProfileUIState"]),
    dict(ui_stack_names=["NewProfileUIInstance", "UnknownPopup"]),
    dict(ui_stack_names=["NewProfileUIInstance", "LivingSkillUIInstance"]),
    dict(ui_stack_names=["NewProfileUIInstance", "DetailCardUIInstance"])])
def test_unverified_profile_fact_keeps_visual_path(updates):
    v = view()
    vars(v).update(updates)
    assert inspect.inspect_needs_grab("profile", v)
    assert inspect._profile_node_from_view(v) == ""


def test_active_profile_field_reads_mutable_vm_not_opening_parameter(monkeypatch):
    reader = stack.Il2CppUiStackReader()
    reader._off['UIStackingPlayer.instance'] = 0x20  # Retained historical bytes.
    reader._h = object()
    reader._ga = 0x90000
    # Actual MMList uses version@18 and size@1c, not System.List._size@18.
    ptrs = {0x900b8: 0xa0000, 0xa0050: 0x10000,
            0x10018: 0x20000, 0x20010: 0x30000, 0x30020: 0x40000,
            0x40020: 0x50000, 0x50068: 0x60000}
    values = {0x20018: 86, 0x2001c: 1, 0x30018: 4, 0x60030: 1,
              0x50010: 0}  # Opening state says 0; mutable current tab says 1.
    names = {0x10000: "UIStackingService", 0x40000: "UIStackingPlayer",
             0x50000: "NewProfileUIInstance", 0x60000: "ProfileUIViewModelComponent"}
    monkeypatch.setattr(stack, "read_ptr", lambda h, p: ptrs.get(p))
    monkeypatch.setattr(stack, "resolve_klass", lambda *a: 0x90000)
    monkeypatch.setattr(stack, "read_u32", lambda h, p: values.get(p))
    monkeypatch.setattr(stack, "klass_simple_name", lambda h, p, o: names.get(p, ""))
    assert reader._active_profile(0x10000) == (1, (0x50000, 0x60000))
    values[0x60030] = 3
    assert reader._active_profile(0x10000) == (3, (0x50000, 0x60000))
    # A closing / overwritten / malformed current association cannot yield an old tab.
    for address, bad in [(0x50068, 0), (0x40020, 0), (0x30020, 0)]:
        original = ptrs[address]
        ptrs[address] = bad
        assert reader._active_profile(0x10000) is None
        ptrs[address] = original
    values[0x2001c] = 5
    assert reader._active_profile(0x10000) is None
    values[0x2001c] = 1
    names[0x60000] = "WrongComponent"
    assert reader._active_profile(0x10000) is None


def test_same_stack_navigation_rechecks_tab_and_instance():
    from mm_assist.life_execute.ui_logic import resolve_navigation
    def snap(category, identity=(0x50000, 0x60000)):
        return stack.UiStackSnap(names=["NewProfileUIInstance"], ready=True,
            profile_category=category, profile_identity=identity, sampled_ts=time.time())
    for after in (snap(1), snap(0, (0x70000, 0x80000)), snap(None)):
        samples = iter([snap(0), after])
        reader = NS(read=lambda **k: next(samples))
        assert resolve_navigation(object(), reader=reader).source == "direct_transition"
    samples = iter([snap(0), snap(0)])
    assert resolve_navigation(object(), reader=NS(read=lambda **k: next(samples))).node == "profile:info"


def test_missing_or_reused_stack_clears_profile_fact():
    from mm_assist.memory.readers.scene import MemSceneInfo
    from mm_assist.perception.scene import apply_stack_bits
    from mm_assist.perception.state_ui import build_state_ui_payload
    ms = MemSceneInfo()
    live = stack.UiStackSnap(names=["NewProfileUIInstance"], ready=True,
        profile_category=1, profile_identity=(0x50000, 0x60000), sampled_ts=time.time())
    apply_stack_bits(ms, live)
    payload = build_state_ui_payload(NS(), ms)
    assert payload["profile_category"] == 1
    assert inspect._stack_sn_from_mem(ms) == "profile:combat"
    for missing in (stack.UiStackSnap(ready=False),
                    stack.UiStackSnap(names=live.names, ready=True, error="stack_read_reused")):
        apply_stack_bits(ms, missing)
        assert ms.profile_category is None
        assert ms.profile_identity is None


def test_late_visual_result_cannot_overwrite_new_memory_tab():
    now = time.time()
    current = inspect.compute_inspect(None, view=view(1, ts=now))
    old = dict(node="profile:info", stack_names=["NewProfileUIInstance"],
               stack_ready=True, source="stack+tpl", stack_sampled_ts=now-1)
    held, _ = inspect.hold_inspect_payload(current, old)
    assert held["node"] == "profile:combat"
    new = inspect.compute_inspect(None, view=view(3, ts=now+.01))
    held, _ = inspect.hold_inspect_payload(current, new)
    assert held["node"] == "profile:career"


@pytest.mark.parametrize("old_node,old_zh", [("main_hud", "主界面-待机"),
                                           ("gather_skill:daily", "日常採集")])
def test_current_memory_tab_wins_over_retained_inspect(monkeypatch, old_node, old_zh):
    from mm_assist.life_execute import ui_logic
    monkeypatch.setattr(ui_logic, "read_inspect_page", lambda **k: old_node)
    monkeypatch.setattr(ui_logic, "read_inspect_record", lambda **k: {
        "node": old_node, "zh": old_zh, "stack_names": ["NewProfileUIInstance"]})
    assert ui_logic.resolve_ui_node(view=view(2)).node == "life_grid"


def test_read_mutation_and_duplicate_active_profiles_are_unknown(monkeypatch):
    reader = stack.Il2CppUiStackReader()
    reader._off['UIStackingPlayer.instance'] = 0x20  # Retained historical bytes.
    reader._h, reader._ga = object(), 0x90000
    ptrs = {0x900b8: 0xa0000, 0xa0050: 0x10000, 0x10018: 0x20000,
            0x20010: 0x30000, 0x30020: 0x40000, 0x40020: 0x50000, 0x50068: 0x60000}
    values = {0x20018: 86, 0x2001c: 1, 0x30018: 4, 0x60030: 2}
    names = {0x10000: "UIStackingService", 0x40000: "UIStackingPlayer",
             0x50000: "NewProfileUIInstance", 0x60000: "ProfileUIViewModelComponent"}
    monkeypatch.setattr(stack, "resolve_klass", lambda *a: 0x90000)
    monkeypatch.setattr(stack, "read_ptr", lambda h, p: ptrs.get(p))
    monkeypatch.setattr(stack, "klass_simple_name", lambda h, p, o: names.get(p, ""))
    calls = 0
    def changing(h, addr):
        nonlocal calls
        if addr == 0x20018:
            calls += 1
            return calls
        return values.get(addr)
    monkeypatch.setattr(stack, "read_u32", changing)
    assert reader._active_profile(0x10000) is None
    monkeypatch.setattr(stack, "read_u32", lambda h, p: values.get(p))
    # Second active profile: ambiguous. Same enum does not make it safe.
    values[0x2001c] = 2
    ptrs.update({0x30028: 0x41000, 0x41020: 0x51000, 0x51068: 0x61000})
    names.update({0x41000: "UIStackingPlayer", 0x51000: "NewProfileUIInstance",
                  0x61000: "ProfileUIViewModelComponent"})
    assert reader._active_profile(0x10000) is None
    # Actual captured daily list adds LivingSkillUIInstance while category stays 2.
    names[0x51000] = "LivingSkillUIInstance"
    assert reader._active_profile(0x10000) is None
