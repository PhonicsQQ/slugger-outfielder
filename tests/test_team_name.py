"""Tests for team_name emission in build_player_dict (fix O2).

The team dropdown on the frontend derives its option list from each batter's
team_name, so build_player_dict must surface a normalized team_name for every
entry: stripped when present, empty string when blank or missing.
"""

import app


def test_team_name_emitted_and_stripped():
    records = [
        {"player_id": "p1", "player_name": "Doe, Jane",
         "player_batting_handedness": "Right", "team_name": "  York Revolution "},
    ]
    (entry,) = app.build_player_dict(records).values()
    assert entry["team_name"] == "York Revolution"


def test_blank_team_name_becomes_empty_string():
    records = [
        {"player_id": "p1", "player_name": "Doe, Jane",
         "player_batting_handedness": "Right", "team_name": ""},
    ]
    (entry,) = app.build_player_dict(records).values()
    assert entry["team_name"] == ""


def test_missing_team_name_becomes_empty_string():
    records = [
        {"player_id": "p1", "player_name": "Doe, Jane",
         "player_batting_handedness": "Right"},
    ]
    (entry,) = app.build_player_dict(records).values()
    assert entry["team_name"] == ""


def test_merged_entry_takes_the_rostered_team():
    """12 of the 75 duplicate-name groups span two populated teams and 63 span a
    real team versus a teamless record, so a pooled entry moves team buckets and
    a coach's team PDF roster changes with it. The primary is the team-bearing
    member with the most recent tracked pitch, pinned here rather than left to
    dict ordering.
    """
    records = [
        {"player_id": "p-blank", "player_name": "Whalen, Brady",
         "player_batting_handedness": "Right", "team_name": ""},
        {"player_id": "p-lan", "player_name": "Whalen, Brady",
         "player_batting_handedness": "Right", "team_name": "Lancaster Stormers"},
    ]
    identities = {
        "p-blank": {"games": frozenset({"g1"}), "days": frozenset({"2024-05-01"}),
                    "team_code": "GAS", "sides": {"R": ("R", 1.0, 40),
                                                  "L": ("R", 1.0, 20)},
                    "balls": {"R": 30, "L": 20},
                    "truncated": False, "error": False},
        "p-lan": {"games": frozenset({"g2"}), "days": frozenset({"2025-05-01"}),
                  "team_code": "LAN", "sides": {"R": ("R", 1.0, 60),
                                                "L": ("R", 1.0, 25)},
                  "balls": {"R": 40, "L": 25},
                  "truncated": False, "error": False},
    }
    result = app.build_player_dict(records, identities=identities)
    assert len(result) == 1
    entry = result["p-lan"]
    assert entry["team_name"] == "Lancaster Stormers"
    assert entry["merged_ids"] == ["p-lan", "p-blank"]


def _identity(days, team_code, side="R"):
    return {"games": frozenset({team_code}), "days": frozenset(days),
            "team_code": team_code,
            "sides": {"R": (side, 1.0, 40), "L": (side, 1.0, 40)},
            "balls": {"R": 30, "L": 30}, "truncated": False, "error": False}


def test_merged_entry_takes_the_most_recent_team():
    """A pooled hitter is filed under the club he currently plays for, not the
    first one the roster feed happens to list.

    Measured on the live 431-entry roster, seven pooled hitters have two
    populated teams and the feed lists the stale one first — Arocho, Jeremy
    reads York Revolution (last tracked 2025-10-01) ahead of the Lancaster
    Stormers record that runs through 2026-06-16. Filing him under York is not
    cosmetic: a pooled name loses its team qualifier, so he disappears from the
    Lancaster dropdown filter and the Lancaster team PDF, and his label says
    nothing about which club he is on. The fingerprint already carries the dates.
    """
    records = [
        {"player_id": "p-yor", "player_name": "Arocho, Jeremy",
         "player_batting_handedness": "Switch", "team_name": "York Revolution"},
        {"player_id": "p-lan", "player_name": "Arocho, Jeremy",
         "player_batting_handedness": "Switch", "team_name": "Lancaster Stormers"},
    ]
    identities = {
        "p-yor": _identity(["2025-09-04", "2025-10-01"], "YOR"),
        "p-lan": _identity(["2026-04-30", "2026-06-16"], "LAN"),
    }
    result = app.build_player_dict(records, identities=identities)
    assert len(result) == 1
    entry = result["p-lan"]
    assert entry["team_name"] == "Lancaster Stormers"
    assert entry["merged_ids"] == ["p-lan", "p-yor"]
    # The pooled name is unambiguous now, so no qualifier is left to carry it.
    assert entry["label"] == "Arocho, Jeremy (S)"


def test_team_name_from_first_seen_record_in_group():
    # A case-variant merge keeps the first-seen team's original casing.
    records = [
        {"player_id": "p1", "player_name": "Roe, Sam",
         "player_batting_handedness": "Right", "team_name": "Test Team"},
        {"player_id": "p2", "player_name": "roe, sam",
         "player_batting_handedness": "Switch", "team_name": "Test Team"},
    ]
    result = app.build_player_dict(records)
    assert len(result) == 1
    (entry,) = result.values()
    assert entry["team_name"] == "Test Team"
