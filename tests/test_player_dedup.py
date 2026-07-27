"""Tests for build_player_dict case-variant duplicate merging (fix O4).

The batter list is built from the raw ALPB /players payload, which contains
genuine case-variant duplicates of the same human (e.g. "flores, santiago"
and "Flores, Santiago"). These must collapse to a single dropdown entry, while
players who merely share a name on different teams must stay separate.
"""

import json
from pathlib import Path

import app

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "data" / "players" / "Southern_Maryland_Blue_Crabs.json"
)


def _records_matching(substr: str) -> list:
    """Pull the literal player records whose name contains `substr` (ci)."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [
        p for p in payload["data"]
        if substr.casefold() in (p.get("player_name") or "").casefold()
    ]


# ── Real fixtures from the cached team file ──────────────────────────────

def test_flores_case_variants_collapse_to_one_entry():
    records = _records_matching("flores, santiago")
    # Sanity: the fixture really does contain both a lower- and proper-cased row.
    names = sorted(p["player_name"] for p in records)
    assert names == ["Flores, Santiago", "flores, santiago"]

    result = app.build_player_dict(records)

    assert len(result) == 1, result
    (entry,) = result.values()
    # Properly-cased variant wins the label; all-lowercase is rejected.
    assert entry["batter_name"] == "Flores, Santiago"
    assert entry["label"].startswith("Flores, Santiago")


def test_demeritte_trailing_space_variants_collapse():
    records = _records_matching("demeritte, travis")
    assert len(records) == 2  # "Demeritte, Travis" and "Demeritte, Travis "
    result = app.build_player_dict(records)
    assert len(result) == 1
    (entry,) = result.values()
    assert entry["batter_name"] == "Demeritte, Travis"
    assert entry["batter_hand"] == "R"


# ── Synthetic same-team case/hand merge ──────────────────────────────────

def test_synthetic_bates_switch_hand_wins_and_pid_stable():
    pid_a, pid_b = "pid-bates-A", "pid-bates-B"
    records = [
        {"player_id": pid_a, "player_name": "Bates, Austin",
         "player_batting_handedness": "Right", "team_name": "Test Team"},
        {"player_id": pid_b, "player_name": "bates, austin",
         "player_batting_handedness": "Switch", "team_name": "Test Team"},
    ]
    result = app.build_player_dict(records)

    assert len(result) == 1
    # First data-bearing pid is kept as the stable key.
    assert pid_a in result
    assert pid_b not in result
    entry = result[pid_a]
    # Proper-cased label, switch hand beats the one-sided value.
    assert entry["batter_name"] == "Bates, Austin"
    assert entry["batter_hand"] == "S"
    assert entry["label"] == "Bates, Austin (S)"


def test_populated_hand_beats_missing():
    records = [
        {"player_id": "p1", "player_name": "Gomez, Luis",
         "player_batting_handedness": None, "team_name": "Test Team"},
        {"player_id": "p2", "player_name": "gomez, luis",
         "player_batting_handedness": "Left", "team_name": "Test Team"},
    ]
    result = app.build_player_dict(records)
    assert len(result) == 1
    (entry,) = result.values()
    assert entry["batter_hand"] == "L"


# ── Different teams must NOT merge ───────────────────────────────────────

def test_same_name_different_teams_not_merged():
    records = [
        {"player_id": "p1", "player_name": "Smith, John",
         "player_batting_handedness": "Right", "team_name": "Team A"},
        {"player_id": "p2", "player_name": "smith, john",
         "player_batting_handedness": "Left", "team_name": "Team B"},
    ]
    result = app.build_player_dict(records)
    assert len(result) == 2
    assert {"p1", "p2"} == set(result)


# ── Fallback constant shape is untouched ─────────────────────────────────

def test_fallback_batters_shape_unchanged():
    for entry in app.BATTERS.values():
        assert set(entry) == {"label", "batter_name", "batter_hand"}
