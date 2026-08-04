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


def _fp(games=(), days=(), team_code=None, side="L", n=40, balls=60,
        truncated=False, error=False) -> dict:
    """A probe_player_identity fingerprint, one-sided against both hands."""
    return {
        "games": frozenset(games),
        "days": frozenset(days),
        "team_code": team_code,
        "sides": {"R": (side, 1.0, n), "L": (side, 1.0, n)},
        "balls": {"R": balls // 2, "L": balls - balls // 2},
        "truncated": truncated,
        "error": error,
    }


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

def test_same_name_different_teams_not_merged_without_identity_evidence():
    """No fingerprints supplied ⇒ every cross-team merge is refused.

    Merging is proof-gated, not name-gated: build_player_dict called without
    ``identities`` has nothing but the shared name to go on, and a shared name
    is not evidence. Case (b) is measured, not hypothetical — "mccarthy, ryan"
    pids 2561d7bf (Lexington) and 55d691d8 (Southern Maryland) share 7 game_ids
    and measure opposite batting sides, so they are two men. Pooling them on
    name alone would splice a lefty and a righty into one bimodal cloud whose
    fitted centroid lands where neither hitter puts the ball.
    """
    records = [
        {"player_id": "p1", "player_name": "Smith, John",
         "player_batting_handedness": "Right", "team_name": "Team A"},
        {"player_id": "p2", "player_name": "smith, john",
         "player_batting_handedness": "Left", "team_name": "Team B"},
    ]
    result = app.build_player_dict(records)
    assert len(result) == 2
    assert {"p1", "p2"} == set(result)
    # An unpooled entry stands on exactly its own record.
    assert result["p1"]["merged_ids"] == ["p1"]
    assert result["p2"]["merged_ids"] == ["p2"]


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


def test_traded_player_stays_split_without_identity_evidence():
    """Without fingerprints the traded-player case is indistinguishable from
    the two-different-men case, so both stay split and the label carries the
    team. Refusal is the default; see the companion test below for what it
    takes to pool them."""
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


def test_traded_player_merges_when_fingerprints_prove_one_human():
    """The real De Aza shape: Staten Island 2024-06-15..07-23, then Southern
    Maryland from 2025-04-25 — no shared game, no shared day, and both records
    measure Left against both pitcher hands. Measured live, the two carry 375
    and 39 batted balls, so leaving them split shows a coach 9% of the evidence
    on one of the two sheets."""
    records = [
        {"player_id": "p-deaza-smd", "player_name": "De Aza, Alejandro",
         "player_batting_handedness": "Switch",
         "team_name": "Southern Maryland Blue Crabs"},
        {"player_id": "p-deaza-sta", "player_name": "De Aza, Alejandro",
         "player_batting_handedness": "Left",
         "team_name": "Staten Island FerryHawks"},
    ]
    identities = {
        "p-deaza-smd": _fp(games=["g-smd-1", "g-smd-2"], days=["2025-04-25"],
                           team_code="SMD", side="L"),
        "p-deaza-sta": _fp(games=["g-sta-1"], days=["2024-06-15"],
                           team_code="STA_YAN", side="L"),
    }
    result = app.build_player_dict(records, identities=identities)

    assert len(result) == 1
    (entry,) = result.values()
    # Primary is the first record carrying a team, and the pool is ordered.
    assert entry["merged_ids"] == ["p-deaza-smd", "p-deaza-sta"]
    assert entry["sample_note"] == "sample pooled from 2 roster records"
    # One entity ⇒ the name is no longer ambiguous ⇒ no team suffix.
    assert entry["label"] == "De Aza, Alejandro (S)"


def test_blank_team_record_is_not_merged_into_its_rostered_twin_without_evidence():
    # Refusal is the default when nothing proves the two records are one human.
    # NOTE: the earlier rationale for this split — that folding a blank-team
    # record costs the player his spray history — was inverted. The union POOLS
    # both records' history rather than discarding one, and pooling is what
    # rescues those renders: measured live, dead (player, pitcher-hand) chart
    # slices inside the duplicate-name groups fall from 121 of 312 to 16 of 154.
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


def test_blank_team_record_merges_into_rostered_twin_when_proven():
    """63 of the 75 duplicate-name groups are a real team versus a record the
    feed left teamless, and 138 of 431 dropdown entries have no team at all.
    When the fingerprints prove one human the pooled entry takes the ROSTERED
    team, so it leaves the "No Team Listed" bucket (measured 138 → 76)."""
    records = [
        {"player_id": "pid-blank", "player_name": "Blackwell, Benjamin",
         "player_batting_handedness": "Switch", "team_name": ""},
        {"player_id": "pid-york", "player_name": "Blackwell, Benjamin",
         "player_batting_handedness": "Right", "team_name": "York Revolution"},
    ]
    identities = {
        "pid-blank": _fp(games=["g1"], days=["2024-05-01"], team_code="WES_POW"),
        "pid-york": _fp(games=["g2"], days=["2025-05-01"], team_code="YOR"),
    }
    result = app.build_player_dict(records, identities=identities)

    assert len(result) == 1
    assert "pid-york" in result
    entry = result["pid-york"]
    assert entry["team_name"] == "York Revolution"
    assert entry["merged_ids"] == ["pid-york", "pid-blank"]
    assert entry["label"] == "Blackwell, Benjamin (S)"


# ── Fallback constant shape is untouched ─────────────────────────────────

def test_fallback_batters_shape_unchanged():
    for entry in app.BATTERS.values():
        assert set(entry) == {"label", "batter_name", "batter_hand"}
