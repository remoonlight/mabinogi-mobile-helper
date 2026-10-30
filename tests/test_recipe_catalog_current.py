"""Offline integrity checks for the TW reference catalogs, never game IO."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_processing_release_state_is_not_the_august_visibility_sample():
    data = load("knowledge/items/altering_process_catalog.json")
    rows = [row for group in data["by_facility"].values() for row in group]
    assert len(rows) == data["recipe_count"] == 88
    assert len({row["altering_sheet_id"] for row in rows}) == 88
    by_id = {row["altering_sheet_id"]: row for row in rows}
    assert by_id[826686065]["tw_status"] == "released"
    assert by_id[1125390553]["tw_status"] == "released"
    assert by_id[972233435]["display_name"] == "麵"
    assert by_id[962391520]["tw_status"] == "feature_pending"
    assert by_id[2049639717]["tw_status"] == "hidden"
    assert data["tw_released_count"] == sum(r["tw_status"] == "released" for r in rows) == 57
    assert data["tw_feature_pending_count"] == 5
    assert data["tw_hidden_count"] == 26
    index = load("knowledge/life_loop/process_db_index.json")
    assert len(index["entries"]) == 88
    for entry in index["entries"]:
        row = by_id[entry["altering_sheet_id"]]
        assert entry["tw_status"] == row["tw_status"]
        assert entry["display_name"] == row["display_name"]


def test_crafting_catalog_covers_all_loaded_rows_with_explicit_release_filters():
    data = load("knowledge/items/craft_recipes.json")
    rows = data["recipes"]
    assert len(rows) == data["recipe_count"] == 847
    assert len({r["sheet_id"] for r in rows}) == 847
    assert sum(r["tw_status"] == "released" for r in rows) == data["tw_released_count"] == 487
    assert sum(r["tw_status"] == "hidden" for r in rows) == data["tw_hidden_count"] == 313
    assert sum(r["tw_status"] == "quest_only" for r in rows) == data["tw_quest_only_count"] == 47
    for row in rows:
        if row["tw_status"] != "released":
            continue
        assert row["display_type"] == "Show"
        assert row["feature_switch"] == 0
        assert row["facility_level"] <= 4
        assert row["display_name"]
        assert row["living_skill"]
        assert row["schedule"]["start_unix_ms"] <= data["as_of_unix_ms"] < row["schedule"]["end_unix_ms"]
        assert all(i["amount"] > 0 and i["name"] for i in row["ingredients"])
    by_id = {r["sheet_id"]: r for r in rows}
    mountain = by_id[1197217502]
    assert mountain["display_name"] == "高級混合強化秘藥(山崩)"
    assert mountain["recipe_unlock_skill_level"] == 24
    assert mountain["consumable_results"][0]["amount"] == 2
    assert len(mountain["ingredients"]) == 5
    chair = by_id[1986935917]
    assert chair["schedule"]["start_unix_ms"] == 1792533600000
    assert chair["tw_status"] == "released"
