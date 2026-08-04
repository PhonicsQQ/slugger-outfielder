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


# ── Same-name records stay separate but self-describing ──────────────────

def test_duplicate_name_labels_carry_team():
    records = [
        {"player_id": "p1", "player_name": "Smith, John",
         "player_batting_handedness": "Right", "team_name": "Team A"},
        {"player_id": "p2", "player_name": "Smith, John",
         "player_batting_handedness": "Left", "team_name": "Team B"},
    ]
    result = app.build_player_dict(records)
    assert len(result) == 2
    assert result["p1"]["label"] == "Smith, John (R) — Team A"
    assert result["p2"]["label"] == "Smith, John (L) — Team B"


def test_unique_name_label_is_not_qualified():
    records = [
        {"player_id": "p1", "player_name": "Doe, Jane",
         "player_batting_handedness": "Right", "team_name": "Team A"},
    ]
    result = app.build_player_dict(records)
    assert result["p1"]["label"] == "Doe, Jane (R)"


def test_case_variant_merge_label_unqualified():
    pid_a, pid_b = "pid-bates-A", "pid-bates-B"
    records = [
        {"player_id": pid_a, "player_name": "Bates, Austin",
         "player_batting_handedness": "Right", "team_name": "Test Team"},
        {"player_id": pid_b, "player_name": "bates, austin",
         "player_batting_handedness": "Switch", "team_name": "Test Team"},
    ]
    result = app.build_player_dict(records)
    # The case-variant merge collapses to one group, so it takes no qualifier.
    assert len(result) == 1
    assert result[pid_a]["label"] == "Bates, Austin (S)"


def test_traded_player_same_name_two_teams_stays_split():
    records = [
        {"player_id": "p-arocho-york", "player_name": "Arocho, Jeremy",
         "player_batting_handedness": "Switch", "team_name": "York Revolution"},
        {"player_id": "p-arocho-lan", "player_name": "Arocho, Jeremy",
         "player_batting_handedness": "Switch", "team_name": "Lancaster Stormers"},
        {"player_id": "p-deaza-smd", "player_name": "De Aza, Alejandro",
         "player_batting_handedness": "Switch",
         "team_name": "Southern Maryland Blue Crabs"},
        {"player_id": "p-deaza-sta", "player_name": "De Aza, Alejandro",
         "player_batting_handedness": "Left",
         "team_name": "Staten Island FerryHawks"},
    ]
    result = app.build_player_dict(records)
    assert len(result) == 4
    assert {"p-arocho-york", "p-arocho-lan",
            "p-deaza-smd", "p-deaza-sta"} == set(result)
    assert result["p-arocho-york"]["team_name"] == "York Revolution"
    assert result["p-arocho-lan"]["team_name"] == "Lancaster Stormers"
    assert result["p-deaza-smd"]["team_name"] == "Southern Maryland Blue Crabs"
    assert result["p-deaza-sta"]["team_name"] == "Staten Island FerryHawks"
    # Same name, same hand — only the team suffix tells the two rows apart.
    assert result["p-arocho-york"]["label"] != result["p-arocho-lan"]["label"]
    assert result["p-arocho-york"]["label"].endswith(" — York Revolution")
    assert result["p-arocho-lan"]["label"].endswith(" — Lancaster Stormers")


def test_blank_team_record_is_not_merged_into_its_rostered_twin():
    # Folding a blank-team record into its rostered twin costs the player his
    # spray history: measured over the live payload it makes 7 players render
    # nothing at all (both hands fall under MIN_QUALIFYING_BALLS).
    records = [
        {"player_id": "pid-blank", "player_name": "Blackwell, Benjamin",
         "player_batting_handedness": "Switch", "team_name": ""},
        {"player_id": "pid-york", "player_name": "Blackwell, Benjamin",
         "player_batting_handedness": "Right", "team_name": "York Revolution"},
    ]
    result = app.build_player_dict(records)
    assert len(result) == 2
    assert {"pid-blank", "pid-york"} == set(result)
    assert result["pid-blank"]["batter_hand"] == "S"
    assert result["pid-york"]["batter_hand"] == "R"
    assert result["pid-blank"]["label"] == \
        "Blackwell, Benjamin (S) — no team listed"
    assert result["pid-york"]["label"] == \
        "Blackwell, Benjamin (R) — York Revolution"


# ── Fallback constant shape is untouched ─────────────────────────────────

def test_fallback_batters_shape_unchanged():
    for entry in app.BATTERS.values():
        assert set(entry) == {"label", "batter_name", "batter_hand"}
