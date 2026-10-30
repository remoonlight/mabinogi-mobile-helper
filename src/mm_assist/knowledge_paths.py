"""当前仓 knowledge 路径真源。运行时 JSON 相对仓根，不随 Python 包搬家。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

UI = ROOT / "knowledge" / "ui"
CONFIG = UI / "config"
CONTRACTS = UI / "contracts"
REPORTS = UI / "reports"
TEMPLATES = ROOT / "knowledge" / "templates"


def ui_config(name: str) -> Path:
    return CONFIG / name


def ui_contract(name: str) -> Path:
    return CONTRACTS / name


SCENE_JSON = ui_config("il2cpp_scene.json")
POSITION_JSON = ui_config("il2cpp_position.json")
FRAME_JSON = ui_config("game_frame.json")
UI_PRIORITY_JSON = ui_config("il2cpp_ui_priority.json")
UI_STACK_JSON = ui_config("il2cpp_ui_stack.json")
BUTTON_HITS_JSON = ui_config("il2cpp_button_hits.json")
PERCEPTION_PROFILE_JSON = ui_config("ui_perception_profile.json")
TRANSITION_GRAPH_JSON = ui_contract("ui_transition_graph.json")
TRANSITION_ACCEPT_JSON = ui_contract("transition_accept_manifest.json")
HUD_ENTRY_JSON = ui_contract("main_hud_entry_capture.json")
GATHER_TIERS_JSON = ui_config("living_skill_gather_tiers.json")
TRANSITION_REVIEW = REPORTS / "ui_transition_review.md"
HUD_ENTRY_REVIEW = REPORTS / "main_hud_entry_review.md"
PERCEPTION_BENCHMARK = REPORTS / "ui_perception_benchmark.md"
MEMORY_BENCHMARK = REPORTS / "memory_acquisition_benchmark.md"
