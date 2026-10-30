"""Stock-aware acquisition routes using only registered gathering/feeding."""
from __future__ import annotations

from mm_assist.life_plan.recipes import (
    FILL_FROM, GATHER_NODE, RECIPES, display_need_rows, material_recipes,
    feed_registered,
)


def acquisition_plan(have: dict[str, int], desired: dict[str, int] | None = None) -> dict[str, list[dict]]:
    """Keep unavailable automatic processing distinct from manual stock items.

    Quantities deduct bag stock. Callers may pass projected stock for planning,
    but must label that separately from independently observed inventory.
    """
    if desired is None:
        from mm_assist.life_plan.planner import necessary_desired
        desired = necessary_desired(have)
    targets, gaps = display_need_rows(have, desired)
    rows = {name: {'name': name, 'have': got, 'missing': short} for name, got, short in gaps}
    for name, got, want in targets:
        if got < want:
            rows[name] = {'name': name, 'have': got, 'missing': want - got}
    known = material_recipes()
    result = {key: [] for key in ('gather', 'processing', 'manual_materials', 'manual_processing', 'manual_crafting')}
    from mm_assist.life_plan.elixir_reserve import MOUNTAIN_NAME, observed_recipes
    for name, row in rows.items():
        if name == MOUNTAIN_NAME:
            recipe = observed_recipes()[name]
            result['manual_crafting'].append({**row, 'batch_output': recipe['output_n'],
                'ingredients': {i['name']: i['amount'] for i in recipe['ingredients']}})
        elif name in known:
            rec = known[name]
            result['processing' if name in RECIPES and feed_registered(name) else 'manual_processing'].append({
                **row, 'facility': rec.facility, 'batch_output': rec.output_n,
                'ingredients': dict(rec.inputs),
            })
        elif name in GATHER_NODE:
            skill, detail = GATHER_NODE[name]
            fill = FILL_FROM.get(name)
            result['gather'].append({**row, 'skill': skill, 'detail': detail,
                'ready': fill is None or have.get(fill[0], 0) > 0})
        else:
            result['manual_materials'].append(row)
    return result
