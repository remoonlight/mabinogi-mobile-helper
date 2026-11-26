"""四件加工链件数。数值来自 process_catalog.md 台服已见行，不是玩法调参。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from mm_assist.life_plan.types import FacilityId

OUTPUT_PER_BATCH = 3

# Tombstones reject stale queues/IPC; these are not executable recipes/routes.
RETIRED_ITEMS = frozenset({'魔力起爆劑', '蘊含魔力的石頭', '魔力藥草', '閃閃發亮的苔蘚'})
# Minimal historical output metadata for already-issued occupancy only.
LEGACY_QUEUE_OUTPUTS: dict[str, tuple[FacilityId, int]] = {'魔力起爆劑': ('medicine', 10)}


@dataclass(frozen=True)
class Recipe:
    facility: FacilityId
    output: str
    output_n: int
    inputs: dict[str, int]


RECIPES: dict[str, Recipe] = {
    "鐵錠": Recipe("metal", "鐵錠", 3, {"鐵礦石": 10}),
    "鋼錠": Recipe("metal", "鋼錠", 3, {"鐵錠": 3, "煤炭": 4}),
    "合金鋼錠": Recipe("metal", "合金鋼錠", 3, {"鋼錠": 3, "銅礦石": 15, "煤炭": 8}),
    "木材": Recipe("wood", "木材", 3, {"原木": 10}),
    "木材+": Recipe("wood", "木材+", 3, {"木材": 3, "樹液": 4}),
    "高級木材": Recipe("wood", "高級木材", 3, {"木材+": 3, "高級原木": 15, "樹液": 8}),
    "布料": Recipe("cloth", "布料", 3, {"羊毛": 10}),
    "布料+": Recipe("cloth", "布料+", 3, {"布料": 3, "羊毛": 4}),
    "高級布料": Recipe("cloth", "高級布料", 3, {"布料+": 3, "高級羊毛": 15, "羊毛": 8}),
    "淨淨蘑菇孢子": Recipe("medicine", "淨淨蘑菇孢子", 15, {"淨淨蘑菇": 30, "紙": 3}),
    "淨淨蘑菇汁液": Recipe(
        "medicine", "淨淨蘑菇汁液", 5, {"淨淨蘑菇孢子": 10, "裝水的瓶子": 1}
    ),
    "增幅蘑菇孢子": Recipe("medicine", "增幅蘑菇孢子", 15, {"增幅蘑菇": 30, "紙": 3}),
    "咻咻蘑菇孢子": Recipe("medicine", "咻咻蘑菇孢子", 15, {"咻咻蘑菇": 30, "紙": 3}),
    "咻咻蘑菇汁液": Recipe("medicine", "咻咻蘑菇汁液", 5, {"咻咻蘑菇孢子": 10, "裝水的瓶子": 1}),
    "增幅蘑菇汁液": Recipe("medicine", "增幅蘑菇汁液", 5, {"增幅蘑菇孢子": 10, "咻咻蘑菇汁液": 10}),
}

FEED_ORDER: dict[FacilityId, tuple[str, ...]] = {
    "metal": ("合金鋼錠", "鋼錠", "鐵錠"),
    "wood": ("高級木材", "木材+", "木材"),
    "cloth": ("高級布料", "布料+", "布料"),
    "medicine": ("淨淨蘑菇汁液", "增幅蘑菇汁液", "咻咻蘑菇汁液",
                 "淨淨蘑菇孢子", "增幅蘑菇孢子", "咻咻蘑菇孢子"),
}

# 链上原料 → 采集详情物（模板名）与技能
GATHER_NODE: dict[str, tuple[str, str]] = {
    "鐵礦石": ("採礦", "鐵礦脈"),
    "銅礦石": ("採礦", "銅礦脈"),
    "煤炭": ("採礦", "煤炭礦脈"),
    "原木": ("伐木", "樹木"),
    "樹液": ("加工详情", "樹液"),
    "高級原木": ("伐木", "成材樹"),
    "羊毛": ("剪羊毛", "羊"),
    "高級羊毛": ("剪羊毛", "捲毛羊"),
    "淨淨蘑菇": ("採集藥草", "淨淨蘑菇"),
    "增幅蘑菇": ("採集藥草", "增幅蘑菇"),
    "咻咻蘑菇": ("採集藥草", "咻咻蘑菇"),
    "箭花": ("採集藥草", "箭花"),
    "血紅藥草": ("採集藥草", "血紅藥草"),
    # User-confirmed obtain methods have a 昆蟲採集 row, not a named light group.
    "滴答螢火蟲": ("昆蟲採集", "獲得方法"),
    "獨角仙": ("昆蟲採集", "獲得方法"),
    # 产出裝水的瓶子：生活技能→日常採集→水井→寻找附近。不是主界面点世界井。
    "裝水的瓶子": ("日常採集", "水井"),
}

CHAIN_RAWS: dict[FacilityId, frozenset[str]] = {
    "metal": frozenset({"鐵礦石", "銅礦石", "煤炭"}),
    "wood": frozenset({"原木", "樹液", "高級原木"}),
    "cloth": frozenset({"羊毛", "高級羊毛"}),
    "medicine": frozenset({"淨淨蘑菇", "增幅蘑菇", "咻咻蘑菇", "裝水的瓶子"}),
}

# 人买空瓶；日常採集水井寻找附近 → 裝水的瓶子（计划按 1 空瓶→1 装水；2026-08-26 实机旁证 1→2）
FILL_FROM: dict[str, tuple[str, int]] = {"裝水的瓶子": ("空瓶", 1)}

# 计划器按「点一次」估件数（人物采集连采口径，不是玩法调参）
GATHER_CLICK_YIELD: dict[str, int] = {"淨淨蘑菇": 40, "裝水的瓶子": 2}


def gather_click_yield(mat: str) -> int | None:
    return GATHER_CLICK_YIELD.get(mat)

# 汁液链上非采集料：人买，不进 GATHER_NODE
BUY_MATS: frozenset[str] = frozenset({"紙", "空瓶"})

# 计划泵已接线的详情物（GATHER_NODE 全开）
WIRED_GATHER_DETAILS: frozenset[str] = frozenset(d for _m, (_s, d) in GATHER_NODE.items())

SKILL_GATHER: tuple[tuple[str, str], ...] = (
    ("收割", "黃豆"),
    ("鋤地", "防風草"),
    ("日常採集", "蜘蛛網"),
    ("伐木", "甲胄樹"),
    ("採礦", "白銅礦脈"),
    ("採集藥草", "淨淨蘑菇"),
    ("剪羊毛", "捲毛羊"),
    ("昆蟲採集", "冰冷的光群"),
)

# 台服 S1 上限见 life_caps_20261125_105607.json。釣魚不在计划目录。
LIFE_TRAINING_GATHER = SKILL_GATHER
LIFE_TRAINING_LEVEL_CAPS = {
    skill: 30 if skill == '日常採集' else 35 for skill, _ in LIFE_TRAINING_GATHER
}

# 新目录中的升级目标；灰色未解锁原图不作为可执行模板。
TRAINING_UPGRADES: dict[str, tuple[tuple[int, str], ...]] = {
    '日常採集': ((20, '蜘蛛絲團'),),
    '伐木': ((25, '幽暮樹'), (30, '雷擊樹')),
    '採礦': ((25, '銀礦脈'), (30, '隕鐵礦脈')),
    '採集藥草': ((20, '傷痕花'), (25, '增幅蘑菇'), (30, '柔柔蘑菇')),
    '剪羊毛': ((25, '烏雲羊'), (30, '雲絮羊')),
    '昆蟲採集': ((20, '荒蕪的昆蟲群'), (25, '荒涼的昆蟲群')),
    '收割': ((20, '稻米'),),
    '鋤地': ((20, '南瓜'),),
}


def fishing_plan(item) -> bool:
    return item.kind == 'gather' and str(item.skill or '') in {'釣魚', '钓鱼', 'fish', 'fishing'}

# 练采集详情物 ≠ 背包短名时：甲胄樹进原木档。
SKILL_BAG_ITEM: dict[str, str] = {
    "甲胄樹": "原木",
}


def batches_for(need: int, per: int = OUTPUT_PER_BATCH) -> int:
    if need <= 0:
        return 0
    return (int(need) + per - 1) // per


def feed_registered(name: str) -> bool:
    """Only current processing recipes may enter the input pipeline."""
    return name in RECIPES


@lru_cache(maxsize=1)
def material_recipes() -> dict[str, Recipe]:
    """Known processing BOM for reserves, distinct from enabled feed recipes."""
    from mm_assist.life_plan.elixir_reserve import observed_recipes

    recipes = dict(RECIPES)
    for name, row in observed_recipes().items():
        if row['kind'] == 'Altering' and name not in recipes:
            recipes[name] = Recipe('medicine', name, row['output_n'],
                                   {i['name']: i['amount'] for i in row['ingredients']})
    return recipes


def display_need_rows(
    have: dict[str, int], desired: dict[str, int]
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """展示用：目标 (名, 有, 本轮目标)；缺口 (名, 有, 缺)。含中间档与紙/空瓶。不改 explode_needs。"""
    bag = {k: int(v) for k, v in have.items()}
    h = dict(bag)
    gaps: dict[str, int] = {}
    recipes = material_recipes()

    def take(name: str, n: int) -> int:
        got = int(h.get(name, 0) or 0)
        use = min(got, n)
        h[name] = got - use
        return n - use

    def add_gap(name: str, n: int) -> None:
        if n > 0:
            gaps[name] = gaps.get(name, 0) + n

    def explode_recipe(output: str, want: int) -> None:
        if output not in recipes:
            add_gap(output, take(output, want))
            return
        rec = recipes[output]
        short = take(output, want)
        if short > 0 and output not in desired:
            add_gap(output, short)
        b = batches_for(short, rec.output_n)
        for mat, qty in rec.inputs.items():
            req = b * qty
            if mat in recipes:
                explode_recipe(mat, req)
            else:
                short_mat = take(mat, req)
                add_gap(mat, short_mat)
                fill = FILL_FROM.get(mat)
                if fill is not None:
                    src, n = fill
                    add_gap(src, take(src, short_mat * n))

    for item, tgt in desired.items():
        explode_recipe(item, int(tgt))
    targets = [
        (name, int(bag.get(name, 0) or 0), int(want))
        for name, want in desired.items()
    ]
    gap_rows = [
        (name, int(bag.get(name, 0) or 0), int(short))
        for name, short in gaps.items()
        if short > 0
    ]
    return targets, gap_rows


def display_need_totals(have: dict[str, int], desired: dict[str, int]) -> list[tuple[str, int, int]]:
    """One inventory/total-demand row per item, deducting stock once per BOM."""
    stock = {name: max(0, int(qty)) for name, qty in have.items()}
    required = dict.fromkeys(desired, 0)
    recipes = material_recipes()

    def take(name: str, amount: int) -> int:
        use = min(stock.get(name, 0), amount)
        stock[name] = stock.get(name, 0) - use
        return amount - use

    def require(name: str, amount: int) -> None:
        if amount <= 0:
            return
        required[name] = required.get(name, 0) + amount
        short = take(name, amount)
        recipe = recipes.get(name)
        if recipe is not None:
            batches = batches_for(short, recipe.output_n)
            for mat, qty in recipe.inputs.items():
                require(mat, batches * qty)
        elif short and name in FILL_FROM:
            source, qty = FILL_FROM[name]
            require(source, short * qty)

    for name, amount in desired.items():
        require(name, int(amount))
    return [(name, int(have.get(name, 0)), amount) for name, amount in required.items()]


def explode_needs(have: dict[str, int], desired: dict[str, int]) -> dict[str, int]:
    """反推本轮还缺的中间档与原料（已扣包内数量，可为 0）。"""
    h = dict(have)
    need: dict[str, int] = {}
    recipes = material_recipes()

    def take(name: str, n: int) -> int:
        got = int(h.get(name, 0) or 0)
        use = min(got, n)
        h[name] = got - use
        return n - use

    def add(name: str, n: int) -> None:
        if n > 0:
            need[name] = need.get(name, 0) + n

    def explode_recipe(output: str, want: int) -> None:
        if output not in recipes:
            add(output, take(output, want))
            return
        rec = recipes[output]
        short = take(output, want)
        b = batches_for(short, rec.output_n)
        for mat, qty in rec.inputs.items():
            req = b * qty
            if mat in recipes:
                explode_recipe(mat, req)
            else:
                short_mat = take(mat, req)
                add(mat, short_mat)
                fill = FILL_FROM.get(mat)
                if fill is not None:
                    src, n = fill
                    add(src, take(src, short_mat * n))

    for item, tgt in desired.items():
        explode_recipe(item, int(tgt))
    return {k: v for k, v in need.items() if v > 0}


def can_afford(recipe_name: str, have: dict[str, int]) -> bool:
    rec = RECIPES[recipe_name]
    return all(int(have.get(mat, 0) or 0) >= qty for mat, qty in rec.inputs.items())
