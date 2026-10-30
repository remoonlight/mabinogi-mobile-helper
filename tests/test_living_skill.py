from __future__ import annotations

from mm_assist.memory.il2cpp import parse_int
from mm_assist.memory.readers.living_skill import (
    plausible_growth,
    plausible_name,
    score_component,
    unpack_living_skill_data,
)


def _off() -> dict[str, int]:
    return {
        "LivingSkillData.LivingSkillSheetId": parse_int("0x0"),
        "LivingSkillData.LivingSkillGrowth": parse_int("0x4"),
        "LivingSkillData.Exp": parse_int("0x8"),
        "LivingSkillData.BonusExpressionExp": parse_int("0x10"),
        "LivingSkillData.stride": parse_int("0x18"),
    }


def test_plausible_name_and_growth():
    assert plausible_name("伐木")
    assert not plausible_name("")
    assert not plausible_name("Woodcutting")
    assert plausible_growth(1) and plausible_growth(99)
    assert not plausible_growth(0) and not plausible_growth(100)


def test_score_prefers_typical_count_and_dirty():
    assert score_component(0, dirty=True) == -1
    assert score_component(20, dirty=True) == 35
    assert score_component(20, dirty=False) == 30
    assert score_component(3, dirty=False) == 3


def test_unpack_stride_24():
    raw = (
        (42).to_bytes(4, "little", signed=True)
        + (22).to_bytes(4, "little", signed=True)
        + (9796).to_bytes(8, "little", signed=True)
        + (0).to_bytes(8, "little", signed=True)
    )
    assert unpack_living_skill_data(raw, _off()) == (42, 22, 9796, 0)
    assert unpack_living_skill_data(raw[:8], _off()) is None
