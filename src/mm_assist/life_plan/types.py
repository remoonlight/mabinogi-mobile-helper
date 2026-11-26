"""生活调度计划类型。不含键鼠。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FacilityId = Literal["metal", "wood", "cloth", "medicine"]
PlanKind = Literal["repair", "claim", "feed", "gather"]
GatherMode = Literal["process_mat", "skill"]

TARGETS = ("合金鋼錠", "高級木材", "高級布料")
FACILITY_ORDER: tuple[FacilityId, ...] = ("metal", "wood", "cloth", "medicine")
FACILITY_TARGET: dict[FacilityId, str] = {
    "metal": "合金鋼錠",
    "wood": "高級木材",
    "cloth": "高級布料",
    "medicine": "淨淨蘑菇汁液",
}
SLOT_CAP = 5  # 旧快照及未传容量的调用沿用原口径。
CLAIM_DONE_MIN = 3
FACILITY_SCAN_INTERVAL_S = 600.0
PLAN_CAP = 10
CYCLE_SIZE = 100
CYCLE_SIZES: dict[str, int] = {
    "合金鋼錠": 100,
    "高級木材": 100,
    "高級布料": 100,
}
SNAPSHOT_TTL_S = 600.0
DURABILITY_REPAIR_BELOW = 1.0


def facility_slot_cap(caps: dict[FacilityId, int] | None, facility: FacilityId) -> int | None:
    """旧调用未传容量时沿用原值。实时读数缺项时，容量未知。"""
    return SLOT_CAP if caps is None else caps.get(facility)


@dataclass(frozen=True)
class ToolDurability:
    name: str
    durability: float
    max_durability: float = 0.0


@dataclass(frozen=True)
class QueueSlot:
    recipe: str
    done: bool
    progress_ratio: float | None = None
    remain_s: float | None = None
    sheet_id: int = 0
    start_ms: int = 0
    end_ms: int = 0


@dataclass(frozen=True)
class SkillGrowth:
    name: str
    level: int
    exp: int


@dataclass(frozen=True)
class LifeSnapshot:
    ok: bool
    items: dict[str, int] = field(default_factory=dict)
    tools: tuple[ToolDurability, ...] = ()
    facilities: dict[FacilityId, tuple[QueueSlot, ...]] = field(default_factory=dict)
    skills: tuple[SkillGrowth, ...] = ()
    error: str = ""
    taken_at: float = 0.0
    evidence: dict = field(default_factory=dict)
    facility_slot_caps: dict[FacilityId, int] | None = None


@dataclass(frozen=True)
class PlanItem:
    id: str
    kind: PlanKind
    title: str
    facility: FacilityId | None = None
    recipe: str = ""
    skill: str = ""
    detail: str = ""
    gather_mode: GatherMode | None = None
    reason: str = ""


@dataclass(frozen=True)
class ExecuteResult:
    ok: bool
    status: str
    message: str = ""
    action_issued: bool = False
    action_started: bool = False
