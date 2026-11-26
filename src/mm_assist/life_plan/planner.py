"""纯函数计划器：最多 10 条，不含窗口/键鼠。"""

from __future__ import annotations

from dataclasses import replace

from mm_assist.life_plan.recipes import (
    CHAIN_RAWS,
    FEED_ORDER,
    GATHER_NODE,
    gather_click_yield,
    RECIPES,
    LEGACY_QUEUE_OUTPUTS,
    SKILL_GATHER,
    LIFE_TRAINING_GATHER,
    LIFE_TRAINING_LEVEL_CAPS,
    TRAINING_UPGRADES,
    WIRED_GATHER_DETAILS,
    explode_needs,
    material_recipes,
    can_afford,
    batches_for,
)
from mm_assist.life_plan import recipes as recipe_registry
from mm_assist.life_plan.types import (
    CLAIM_DONE_MIN,
    CYCLE_SIZES,
    DURABILITY_REPAIR_BELOW,
    FACILITY_ORDER,
    FACILITY_TARGET,
    PLAN_CAP,
    TARGETS,
    FacilityId,
    LifeSnapshot,
    PlanItem,
    facility_slot_cap,
)

_FAC_ZH = {"metal": "金屬", "wood": "木材", "cloth": "布料", "medicine": "藥品"}


def cycle_desired(have: dict[str, int]) -> dict[str, int]:
    """三类通用加工料定额；药品需求由成品储备反推。"""
    return {t: CYCLE_SIZES[t] for t in TARGETS}


def necessary_desired(have: dict[str, int]) -> dict[str, int]:
    from mm_assist.life_plan.elixir_reserve import RESERVE_GOALS, mountain_targets

    return {**cycle_desired(have), **RESERVE_GOALS, **mountain_targets(have=have)}


def _batch_out(recipe: str) -> int:
    rec = RECIPES.get(recipe) or material_recipes().get(recipe)
    return int(rec.output_n) if rec is not None else LEGACY_QUEUE_OUTPUTS.get(recipe, ('', 3))[1]


def _virtual_have(snap: LifeSnapshot) -> dict[str, int]:
    have = {k: int(v) for k, v in snap.items.items()}
    for slots in snap.facilities.values():
        for slot in slots:
            have[slot.recipe] = have.get(slot.recipe, 0) + _batch_out(slot.recipe)
    return have


def needs_repair(snap: LifeSnapshot) -> bool:
    return any(float(t.durability) < DURABILITY_REPAIR_BELOW for t in snap.tools)


def _needs_repair(snap: LifeSnapshot) -> bool:
    return needs_repair(snap)


def _needed_batches(facility: FacilityId, projected: dict[str, int], desired: dict[str, int]) -> dict[str, int]:
    """Expand every branch; projected outputs satisfy demand but are not spendable."""
    stock = dict(projected)
    counts = dict.fromkeys(FEED_ORDER[facility], 0)

    def require(name: str, amount: int) -> None:
        rec = RECIPES.get(name)
        if rec is None or rec.facility != facility:
            return
        use = min(amount, max(0, stock.get(name, 0)))
        stock[name] = stock.get(name, 0) - use
        batch = batches_for(amount - use, rec.output_n)
        counts[name] += batch
        stock[name] += batch * rec.output_n - (amount - use)
        for mat, qty in rec.inputs.items():
            require(mat, batch * qty)

    for name, amount in desired.items():
        if name in counts:
            require(name, amount)
    return counts


def _pick_feed(facility: FacilityId, have: dict[str, int], desired: dict[str, int],
               *, projected: dict[str, int] | None = None) -> str:
    needed = _needed_batches(facility, projected if projected is not None else have, desired)
    for recipe in FEED_ORDER[facility]:
        if (needed[recipe] > 0 and can_afford(recipe, have)
                and recipe_registry.feed_registered(recipe)):
            return recipe
    return ""


def _slot_free(
    snap: LifeSnapshot,
    fac: FacilityId,
    extra_free: dict[str, int] | None = None,
    extra_occ: dict[str, int] | None = None,
) -> int:
    cap = facility_slot_cap(snap.facility_slot_caps, fac)
    if cap is None:
        return 0
    occupied = len(snap.facilities.get(fac) or ())
    extra = int((extra_free or {}).get(fac, 0) or 0)
    planned = int((extra_occ or {}).get(fac, 0) or 0)
    return max(0, cap - occupied + extra - planned)


def _slot_material_needs(fac: FacilityId, free: int, available: dict[str, int],
                         projected: dict[str, int], desired: dict[str, int]) -> dict[str, int]:
    needed = _needed_batches(fac, projected, desired)
    for recipe in FEED_ORDER[fac]:
        if not recipe_registry.feed_registered(recipe):
            continue
        count = min(free, needed[recipe])
        if count <= 0:
            continue
        for mat, qty in RECIPES[recipe].inputs.items():
            if mat not in CHAIN_RAWS[fac]:
                # Missing paper/intermediates cannot be fixed by raw gathering.
                count = min(count, available.get(mat, 0) // qty)
            elif mat == '裝水的瓶子':
                count = min(count, (available.get(mat, 0) + 2 * available.get('空瓶', 0)) // qty)
        if count <= 0:
            continue
        return {mat: count * qty - available.get(mat, 0)
                for mat, qty in RECIPES[recipe].inputs.items()
                if mat in CHAIN_RAWS[fac] and count * qty > available.get(mat, 0)}
    return {}


def _gather_process(needs: dict[str, int]) -> PlanItem | None:
    if not needs:
        return None
    mat = max(needs, key=lambda k: needs[k])
    node = GATHER_NODE.get(mat)
    if not node:
        return None
    skill, detail = node
    yield_n = gather_click_yield(mat)
    extra = f"（按{yield_n}/次）" if yield_n else ""
    return PlanItem(
        id="",
        kind="gather",
        title=f"采集 {mat}",
        skill=skill,
        detail=detail,
        recipe=mat,
        gather_mode="process_mat",
        reason=f"缺{mat}×{needs[mat]}{extra}",
    )


def training_item(snap: LifeSnapshot, counts: dict[str, int] | None = None) -> PlanItem | None:
    """Each eligible skill gets a turn; fresh lower levels break round ties."""
    counts = counts or {}
    levels = {s.name: s.level for s in snap.skills}
    eligible = [pair for pair in LIFE_TRAINING_GATHER
                if levels.get(pair[0], 0) < LIFE_TRAINING_LEVEL_CAPS[pair[0]]]
    if not eligible:
        return None
    skill, detail = min(eligible, key=lambda pair: (
        counts.get(pair[0], 0), levels.get(pair[0], 99)))
    _, detail = max((tier for tier in TRAINING_UPGRADES.get(skill, ())
                     if levels.get(skill, 0) >= tier[0]),
                    key=lambda tier: tier[0], default=(0, detail))
    return PlanItem('', 'gather', f'练采集 {detail}', recipe=detail, skill=skill,
                    detail=detail, gather_mode='skill',
                    reason='本批同一目标；批间均衡轮换，同轮低等级优先')


def training_target_allowed(item: PlanItem, snap: LifeSnapshot, *, automatic: bool = False) -> bool:
    level = next((s.level for s in snap.skills if s.name == item.skill), 0)
    if automatic and level >= LIFE_TRAINING_LEVEL_CAPS.get(item.skill, 0):
        return False
    pair = (item.skill, item.detail)
    if pair in SKILL_GATHER:
        return True
    return any(item.detail == detail and level >= threshold
               for threshold, detail in TRAINING_UPGRADES.get(item.skill, ()))


def group_gather_items(items: list[PlanItem]) -> list[PlanItem]:
    """Group a new batch by gathered item, retaining action slots and ascending IDs."""
    groups: dict[str, list[PlanItem]] = {}
    for item in items:
        if item.kind == 'gather':
            groups.setdefault(item.recipe or item.detail, []).append(item)
    gathers = iter(item for group in groups.values() for item in group)
    return [replace(next(gathers), id=item.id) if item.kind == 'gather' else item
            for item in items]


def plan_life(
    snap: LifeSnapshot,
    *,
    start_id: int = 1,
    blocked_facilities: frozenset[str] | None = None,
    training_counts: dict[str, int] | None = None,
    reserved_feeds: dict[str, int] | None = None,
    training_target: PlanItem | None = None,
) -> list[PlanItem]:
    if not snap.ok:
        return []
    items: list[PlanItem] = []
    seq = start_id
    projected = _virtual_have(snap)
    for recipe, count in (reserved_feeds or {}).items():
        projected[recipe] = projected.get(recipe, 0) + count * _batch_out(recipe)
    have = dict(snap.items)
    desired = necessary_desired(projected)
    blocked = frozenset(blocked_facilities or ())

    def push(item: PlanItem) -> None:
        nonlocal seq
        if len(items) >= PLAN_CAP:
            return
        items.append(
            PlanItem(
                id=f"p{seq}",
                kind=item.kind,
                title=item.title,
                facility=item.facility,
                recipe=item.recipe,
                skill=item.skill,
                detail=item.detail,
                gather_mode=item.gather_mode,
                reason=item.reason,
            )
        )
        seq += 1

    if _needs_repair(snap):
        push(
            PlanItem(
                id="",
                kind="repair",
                title="一键修理全部工具",
                reason="工具耐久<1",
            )
        )

    for fac in FACILITY_ORDER:
        slots = snap.facilities.get(fac, ())
        n_done = sum(1 for s in slots if s.done)
        if n_done >= CLAIM_DONE_MIN and len(items) < PLAN_CAP:
            push(
                PlanItem(
                    id="",
                    kind="claim",
                    title=f"领取 {_FAC_ZH.get(fac, fac)}",
                    facility=fac,
                    recipe="",
                    reason=f"{fac} 已完成{n_done}格",
                )
            )

    extra_free: dict[str, int] = {}
    for it in items:
        if it.kind != "claim" or not it.facility:
            continue
        fac = str(it.facility)
        extra_free[fac] = sum(1 for s in (snap.facilities.get(fac) or ()) if s.done)
        for slot in snap.facilities.get(fac, ()):
            if slot.done:
                have[slot.recipe] = have.get(slot.recipe, 0) + _batch_out(slot.recipe)
    extra_occ: dict[str, int] = {f: 0 for f in FACILITY_ORDER}
    for recipe, count in (reserved_feeds or {}).items():
        # Archived recipes still identify already-issued, unscanned occupancy.
        # They never become enabled feed recipes again.
        rec = RECIPES.get(recipe) or material_recipes().get(recipe)
        fac = rec.facility if rec is not None else LEGACY_QUEUE_OUTPUTS[recipe][0]
        extra_occ[fac] += count

    def free_slots(fac: FacilityId) -> int:
        return _slot_free(snap, fac, extra_free, extra_occ)

    # Finish all immediately affordable feeding before planning any gathering.
    for fac in FACILITY_ORDER:
        if fac in blocked:
            continue
        free = free_slots(fac)
        while free > 0 and len(items) < PLAN_CAP:
            recipe = _pick_feed(fac, have, desired, projected=projected)
            if not recipe:
                break
            push(
                PlanItem(
                    id="",
                    kind="feed",
                    title=f"投料 {recipe}",
                    facility=fac,
                    recipe=recipe,
                    reason=f"{fac} 空位{free}",
                )
            )
            for mat, qty in RECIPES[recipe].inputs.items():
                have[mat] = have.get(mat, 0) - qty
                projected[mat] = projected.get(mat, 0) - qty
            # Planned output contributes to the goal, but is not feedable stock.
            projected[recipe] = projected.get(recipe, 0) + _batch_out(recipe)
            extra_occ[fac] = extra_occ.get(fac, 0) + 1
            free -= 1

    processing_supply = False
    for fac in FACILITY_ORDER:
        if fac in blocked or len(items) >= PLAN_CAP:
            continue
        raw_needs = _slot_material_needs(fac, free_slots(fac), have, projected, desired)
        g = _gather_process(raw_needs) if raw_needs else None
        if g is not None and g.detail in WIRED_GATHER_DETAILS:
            # Establish the eligible supply targets from actual stock.
            push(g)
            processing_supply = True

    # One targeted reserve action, then refresh actual stock before replanning.
    from mm_assist.life_plan.elixir_reserve import mountain_targets

    from mm_assist.life_plan.acquisition import acquisition_plan
    reserve_routes = acquisition_plan(projected, mountain_targets(have=projected))
    # The original four processing chains have slot-sized gathering above.
    # Additional registered reserve sources get a targeted supply action,
    # while unsupported materials/processing stay visible for human preparation.
    scheduled = {p.recipe for p in items if p.gather_mode == 'process_mat'}
    reserve_needs = {row['name']: row['missing'] for row in reserve_routes['gather']
                  if row['ready'] and row['name'] not in scheduled
                  and row['name'] in {'增幅蘑菇', '咻咻蘑菇', '箭花', '血紅藥草', '滴答螢火蟲', '獨角仙'}}
    reserve_gather = _gather_process(reserve_needs)
    if reserve_gather is not None and len(items) < PLAN_CAP:
        push(reserve_gather)

    # Distribute this batch's ten slots across known supply targets, then group
    # each material. Expected gather output must never fund a future feed action.
    supply = [p for p in items if p.gather_mode == 'process_mat']
    if processing_supply and supply:
        index = 0
        while len(items) < PLAN_CAP:
            push(supply[index % len(supply)])
            index += 1
        return group_gather_items(items[:PLAN_CAP])

    training = (training_target if training_target is not None
                and training_target_allowed(training_target, snap, automatic=True)
                else training_item(snap, training_counts))
    while training is not None and len(items) < PLAN_CAP:
        push(training)
    return group_gather_items(items[:PLAN_CAP])


def insert_repair_next(queue: list[PlanItem], *, after_id: str, seq: int) -> tuple[list[PlanItem], int]:
    """当前这条做完后插入一键修理。队里已有修理则不动。"""
    if any(x.kind == "repair" for x in queue):
        return list(queue), seq
    repair = PlanItem(
        id=f"p{int(seq)}",
        kind="repair",
        title="一键修理全部工具",
        reason="工具耐久<1",
    )
    nxt = int(seq) + 1
    out: list[PlanItem] = []
    placed = False
    for item in queue:
        out.append(item)
        if item.id == after_id:
            out.append(repair)
            placed = True
    if not placed:
        out = [repair, *queue]
    return out[:PLAN_CAP], nxt


def apply_done(queue: list[PlanItem], item_id: str) -> list[PlanItem]:
    return [x for x in queue if x.id != item_id]
