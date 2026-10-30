"""Mountain reserve quantities from the observed recipe cache; no game IO."""
from collections import Counter
from functools import lru_cache
import json

from mm_assist.config import ROOT

MOUNTAIN_COUNT = 40
MOUNTAIN_NAME = '高級混合強化秘藥(山崩)'
RESERVE_GOALS = {MOUNTAIN_NAME: MOUNTAIN_COUNT}


@lru_cache(maxsize=1)
def observed_recipes() -> dict[str, dict]:
    data = json.loads((ROOT / 'knowledge/items/mountain_elixir_materials.json').read_text('utf-8'))
    recipes = {row['name']: row for row in data['steps']}
    if len(recipes) != len(data['steps']):
        raise ValueError('ambiguous mountain recipe')
    for row in recipes.values():
        if row['output_n'] <= 0 or any(i['amount'] <= 0 for i in row['ingredients']):
            raise ValueError('invalid mountain recipe quantity')
    return recipes


def mountain_targets(count: int = MOUNTAIN_COUNT, *, have: dict[str, int] | None = None) -> dict[str, int]:
    """Deduct crafted stock, aggregate shared demand, and stop at processing."""
    recipes = observed_recipes()
    order = []
    visited = set()
    active = set()

    def visit(name):
        row = recipes.get(name)
        if row is None or row['kind'] != 'CraftingRecipe' or name in visited:
            return
        if name in active:
            raise ValueError('mountain recipe cycle')
        active.add(name)
        for ingredient in row['ingredients']:
            visit(ingredient['name'])
        active.remove(name)
        visited.add(name)
        order.append(name)

    if count < 0:
        raise ValueError('negative mountain count')
    visit(MOUNTAIN_NAME)
    demand = Counter({MOUNTAIN_NAME: int(count)})
    stock = {name: max(0, int(qty)) for name, qty in (have or {}).items()}
    for name in reversed(order):
        row = recipes[name]
        short = max(0, demand[name] - stock.get(name, 0))
        batches = (short + row['output_n'] - 1) // row['output_n']
        for ingredient in row['ingredients']:
            demand[ingredient['name']] += batches * ingredient['amount']
    targets = {}
    for name, amount in demand.items():
        if name in visited or amount <= 0:
            continue
        row = recipes.get(name)
        if row is not None and stock.get(name, 0) < amount:
            amount = ((amount + row['output_n'] - 1) // row['output_n']) * row['output_n']
        targets[name] = amount
    return targets
