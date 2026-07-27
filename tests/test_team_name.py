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
