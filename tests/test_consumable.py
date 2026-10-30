from __future__ import annotations

from mm_assist.memory.readers import consumable as bag_mod


def test_bag_scan_skips_empty_hits_and_caches(monkeypatch):
    bag_mod.reset_bag_cache_for_tests()
    klass = 0x1000
    real = 0x2000
    ghost = 0x3000
    bags = {ghost: {}, real: {"空瓶": 3, "裝水的瓶子": 1}}
    seen: list[int] = []

    monkeypatch.setattr(bag_mod, "module_base", lambda *_a, **_k: 1)
    monkeypatch.setattr(bag_mod, "resolve_klass", lambda *_a, **_k: klass)
    monkeypatch.setattr(bag_mod, "parse_int", lambda v: int(str(v), 0) if v else 0)
    monkeypatch.setattr(bag_mod, "read_sc_component", lambda *_a, **_k: None)
    monkeypatch.setattr(bag_mod, "read_ptr", lambda _h, addr: klass if addr in (ghost, real) else 0)
    monkeypatch.setattr(bag_mod, "_cfg", lambda: {
        "module": "GameAssembly.dll",
        "typeinfo_rva": {
            "ConsumableInventoryComponent": "0x1",
            "ConsumableStack": "0x2",
            "SystemContainers": "0x3",
        },
        "offsets": {"SystemContainers.ConsumableInventoryComponent": "0x0",
                    "ConsumableInventoryComponent.ConsumableStacks": "0x10"},
    })
    monkeypatch.setattr(
        bag_mod,
        "_decode_bag",
        lambda _h, comp, _off, _sk: dict(bags.get(comp, {})),
    )

    def fake_scan(_h, _klass, *, max_hits=4, max_wall_s=None, hit_filter=None):
        hits = []
        for addr in (ghost, real):
            if hit_filter is not None and not hit_filter("_", addr):
                continue
            hits.append(addr)
            if len(hits) >= max_hits:
                break
        seen.extend(hits)
        return hits

    monkeypatch.setattr(bag_mod, "scan_klass_objects", fake_scan)
    first = bag_mod.read_consumable_counts(object(), 11)
    assert first == {"空瓶": 3, "裝水的瓶子": 1}
    assert seen == [real]
    monkeypatch.setattr(bag_mod, "read_sc_component", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("warm read must not scan SC")))
    bags[real]["空瓶"] = 2
    second = bag_mod.read_consumable_counts(object(), 11)
    assert second == {"空瓶": 2, "裝水的瓶子": 1}
    assert seen == [real]
