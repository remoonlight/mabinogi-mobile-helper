"""生活技能短名+等级+exp 只读。配置 knowledge/ui/config/il2cpp_living_skill.json。"""

from __future__ import annotations

import json
import struct
from typing import Any, TypedDict

from mm_assist.knowledge_paths import ui_config
from mm_assist.memory.il2cpp import (
    module_base,
    parse_int,
    read_il2cpp_string,
    resolve_klass,
    scan_klass_objects,
)
from mm_assist.memory.component_hints import load_hint, save_hint
from mm_assist.memory.evidence import root_observation
from mm_assist.memory.process import read_mem, read_ptr, read_u32

_MAX_DICT = 64
_MAX_SHEETS = 96


class LivingSkill(TypedDict):
    name: str
    level: int
    exp: int
    bonus_exp: int
    sheet_id: int


def _cfg() -> dict[str, Any]:
    return json.loads(ui_config("il2cpp_living_skill.json").read_text(encoding="utf-8"))


def _i32(hproc, addr: int) -> int | None:
    raw = read_mem(hproc, addr, 4)
    if not raw or len(raw) < 4:
        return None
    return int(struct.unpack("<i", raw)[0])


def _i64(hproc, addr: int) -> int | None:
    raw = read_mem(hproc, addr, 8)
    if not raw or len(raw) < 8:
        return None
    return int(struct.unpack("<q", raw)[0])


def plausible_name(name: str) -> bool:
    if not name or len(name) > 32:
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in name)


def plausible_growth(level: int) -> bool:
    return 1 <= int(level) <= 99


def score_component(n_valid: int, *, dirty: bool) -> int:
    """条目约 15–25 加分；levelDirtySource 非空再加分。"""
    n = int(n_valid)
    if n <= 0:
        return -1
    score = n
    if 15 <= n <= 25:
        score += 10
    if dirty:
        score += 5
    return score


def unpack_living_skill_data(raw: bytes, off: dict[str, int]) -> tuple[int, int, int, int] | None:
    """从 stride 缓冲区解 SheetId / Growth / Exp / Bonus。off 为已 parse 的字节偏移。"""
    stride = int(off["LivingSkillData.stride"])
    if not raw or len(raw) < stride:
        return None
    sid_o = int(off["LivingSkillData.LivingSkillSheetId"])
    grow_o = int(off["LivingSkillData.LivingSkillGrowth"])
    exp_o = int(off["LivingSkillData.Exp"])
    bonus_o = int(off["LivingSkillData.BonusExpressionExp"])
    if max(sid_o + 4, grow_o + 4, exp_o + 8, bonus_o + 8) > len(raw):
        return None
    sheet_id = int(struct.unpack_from("<i", raw, sid_o)[0])
    growth = int(struct.unpack_from("<i", raw, grow_o)[0])
    exp = int(struct.unpack_from("<q", raw, exp_o)[0])
    bonus = int(struct.unpack_from("<q", raw, bonus_o)[0])
    return sheet_id, growth, exp, bonus


def _read_unique(hproc, obj: int, field_off: int) -> str:
    us = read_ptr(hproc, obj + field_off)
    if not us:
        return ""
    direct = read_il2cpp_string(hproc, us, max_chars=32)
    if direct:
        return direct
    inner = read_ptr(hproc, us)
    return read_il2cpp_string(hproc, inner, max_chars=32) or "" if inner else ""


def _sheet_names(hproc, sheet_klass: int, off: dict[str, int]) -> dict[int, str]:
    names: dict[int, str] = {}
    if not sheet_klass:
        return names

    def _named_sheet(_type: str, obj: int) -> bool:
        if read_ptr(hproc, obj) != sheet_klass:
            return False
        sid = _i32(hproc, obj + off["LivingSkillSheetData.Id"])
        name = _read_unique(hproc, obj, off["LivingSkillSheetData.DisplayName"])
        if sid is None or sid <= 0 or not plausible_name(name):
            return False
        names[int(sid)] = name
        return True

    # 堆上大量未初始化表行会先占满 max_hits；只计有中文短名的行。
    scan_klass_objects(
        hproc, sheet_klass, max_hits=_MAX_SHEETS, max_wall_s=12.0, hit_filter=_named_sheet
    )
    return names


def _decode_state(
    hproc, state: int, off: dict[str, int], names: dict[int, str]
) -> list[LivingSkill]:
    count = int(read_u32(hproc, state + off["MMDictionary.count"]) or 0)
    if count <= 0 or count > _MAX_DICT:
        return []
    arr = read_ptr(hproc, state + off["MMDictionary.values"])
    if not arr:
        return []
    max_len = int(read_u32(hproc, arr + off["Il2CppArray.max_length"]) or 0)
    vec = arr + off["Il2CppArray.vector"]
    stride = int(off["LivingSkillData.stride"])
    n = min(count, max_len, _MAX_DICT)
    out: list[LivingSkill] = []
    for i in range(n):
        raw = read_mem(hproc, vec + i * stride, stride)
        parsed = unpack_living_skill_data(raw or b"", off)
        if parsed is None:
            continue
        sheet_id, growth, exp, bonus = parsed
        if not plausible_growth(growth):
            continue
        if exp < 0 or exp > 10**12 or bonus < 0 or bonus > 10**12:
            continue
        name = names.get(int(sheet_id), "")
        if not plausible_name(name):
            continue
        out.append(
            {
                "name": name,
                "level": int(growth),
                "exp": int(exp),
                "bonus_exp": int(bonus),
                "sheet_id": int(sheet_id),
            }
        )
    out.sort(key=lambda r: (-r["level"], r["name"]))
    return out


def _decode_component(
    hproc, comp: int, off: dict[str, int], names: dict[int, str]
) -> tuple[list[LivingSkill], int]:
    state = read_ptr(hproc, comp + off["CharacterLivingSkillComponent.LivingSkillState"])
    if not state:
        return [], -1
    skills = _decode_state(hproc, state, off, names)
    dirty = bool(read_ptr(hproc, comp + off["CharacterLivingSkillComponent.levelDirtySource"]))
    return skills, score_component(len(skills), dirty=dirty)


def read_living_skills(hproc, pid: int) -> list[LivingSkill] | None:
    cfg = _cfg()
    ga = module_base(pid, str(cfg.get("module") or "GameAssembly.dll"))
    if not ga:
        return None
    off = {k: parse_int(v) for k, v in cfg["offsets"].items()}
    comp_rva = parse_int(cfg["typeinfo_rva"]["CharacterLivingSkillComponent"])
    sheet_rva = parse_int(cfg["typeinfo_rva"]["LivingSkillSheetData"])
    klass = resolve_klass(hproc, ga, comp_rva)
    sheet_klass = resolve_klass(hproc, ga, sheet_rva)
    if not klass or not sheet_klass:
        return None
    names = _sheet_names(hproc, sheet_klass, off)
    if not names:
        return None
    state_off = off["CharacterLivingSkillComponent.LivingSkillState"]
    count_off = off["MMDictionary.count"]

    def _decode_live(obj: int) -> tuple[list[LivingSkill], int]:
        if read_ptr(hproc, obj) != klass:
            return [], -1
        with root_observation('skills', hproc, obj, (0, off['CharacterLivingSkillComponent.LivingSkillState']), accepted=False) as root:
            result = _decode_component(hproc, obj, off, names)
            root.accepted = bool(result[0]) and result[1] >= 0
            return result

    def _plausible_comp(_name: str, obj: int) -> bool:
        skills, score = _decode_live(obj)
        return score >= 0 and bool(skills)

    hinted = load_hint(hproc, pid, "living_skill", klass)
    if hinted:
        skills, score = _decode_live(hinted)
        if score >= 0 and skills:
            return skills
    hits = scan_klass_objects(
        hproc, klass, max_hits=4, max_wall_s=12.0, hit_filter=_plausible_comp
    )
    best: list[LivingSkill] = []
    best_score = -1
    best_comp = 0
    for comp in hits:
        skills, score = _decode_live(comp)
        if score > best_score:
            best, best_score, best_comp = skills, score, int(comp)
    if best_score < 0:
        return None
    if best_comp:
        save_hint(pid, "living_skill", best_comp)
    return best
