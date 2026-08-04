"""Tests for the same-name merge rule (``_merge_refusal``) and group closure.

One human who appears under more than one team_name used to become two dropdown
entries with different player_ids, each charted from only that record's batted
balls — measured live on 2026-08-04, 75 duplicate name groups covering 81
redundant entries of 431, and a coach picking "Sandoval, Ariel" positioned an
outfield off 106 of that hitter's 243 tracked balls with nothing on the sheet
saying so.

Pooling them is only safe when the two records are provably one human. They may
instead be two different men sharing a name, and that failure is worse: it
splices two spray clouds into one whose fitted centroid lands where neither
hitter puts the ball, and nothing on the sheet could reveal it. So merging is
refused by default and allowed only on positive proof from the pitch feed —
no shared game, no shared calendar day under two clubs, and measured batting
side that agrees *and* actually decides.

The rule is exercised directly on hand-built fingerprints, so these tests never
touch the network, matplotlib, or the data loader. The shapes are the real ones
measured from /api/players/<pid>/spray.
"""

import app


# ── helpers ────────────────────────────────────────────────────────────────

def _fp(games=(), days=(), team_code=None, sides=None,
        truncated=False, error=False, balls=(30, 30)) -> dict:
    """A probe_player_identity fingerprint.

    ``sides`` is {pitcher_hand: (batting_side, purity, n)}; the default is a
    clean one-sided lefty against both hands.
    """
    return {
        "games": frozenset(games),
        "days": frozenset(days),
        "team_code": team_code,
        "sides": sides or {"R": ("L", 1.0, 40), "L": ("L", 1.0, 40)},
        "balls": {"R": balls[0], "L": balls[1]},
        "truncated": truncated,
        "error": error,
    }


def _one_sided(side, n=40, purity=1.0):
    return {"R": (side, purity, n), "L": (side, purity, n)}


# ── 1. no evidence is refusal, not weak evidence ───────────────────────────

def test_missing_fingerprint_refuses():
    # Also the state before the identity pass completes, in JSON-loader and
    # synthetic modes, and for any pid whose identity fetch failed outright.
    assert app._merge_refusal(None, _fp()) == "missing evidence"
    assert app._merge_refusal(_fp(), None) == "missing evidence"
    assert app._merge_refusal(None, None) == "missing evidence"


# ── 2. a capped window means "cannot tell", never "no collision found" ─────

def test_truncated_fingerprint_refuses():
    """A fetch that hit its page cap has an unordered, partial view of the
    record, so it can hide the very shared game that proves two humans.

    Inert today by construction — the deepest duplicate-name record is
    terry/e35d1276 at 535 batted balls against a 1000-row page — but load
    bearing if the fingerprint is ever refactored back onto the 500-row
    qualifying probe, whose PROBE_LIMIT is a cap on TOTAL InPlay rows (upstream
    ignores the pitcher_throws filter), which e35d1276 already exceeds.
    """
    assert app._merge_refusal(_fp(truncated=True), _fp()) == "truncated evidence"
    assert app._merge_refusal(_fp(), _fp(truncated=True)) == "truncated evidence"
    assert app._merge_refusal(_fp(error=True), _fp()) == "truncated evidence"


# ── 3. one human cannot appear twice inside one game ───────────────────────

def test_shared_game_refuses():
    # The real McCarthy collision: 2561d7bf (Lexington) and 55d691d8 (Southern
    # Maryland) share 7 game_ids and measure opposite sides.
    a = _fp(games=["g-1", "g-2", "g-3"], team_code="LEX_LEG",
            sides=_one_sided("L"))
    b = _fp(games=["g-3", "g-9"], team_code="SMD", sides=_one_sided("R"))
    assert app._merge_refusal(a, b) == "shared game"


# ── 4. …nor suit up for two clubs on one calendar day ──────────────────────

def test_same_day_under_two_clubs_refuses():
    a = _fp(games=["g-1"], days=["2025-06-01"], team_code="LAN")
    b = _fp(games=["g-2"], days=["2025-06-01"], team_code="YOR")
    assert app._merge_refusal(a, b) == "same day, two clubs"


def test_same_day_under_the_same_club_is_not_a_collision():
    """Pinned to the 2024-07-05 Hagerstown re-ingest: the feed carries that game
    under game_ids 4ec96986 and 1bdc88e2, and for campagna, joe the two records
    hold byte-identical Trackman — (1, 'Bottom', PA 3, pitch 6, exit 98.38211,
    angle 11.49186, direction -27.448734) under both. Four pairs share exactly
    one calendar day this way, all Hagerstown. Same-day-same-club is an ingest
    artifact, not two humans, so it must never become a tripwire."""
    a = _fp(games=["4ec96986"], days=["2024-07-05"], team_code="HAG_FLY")
    b = _fp(games=["1bdc88e2"], days=["2024-07-05"], team_code="HAG_FLY")
    assert app._merge_refusal(a, b) is None


def test_shared_day_refuses_when_the_club_is_unknown():
    """An unequal team_code is not what makes the day damning — a *known* equal
    one is what makes it innocent. probe_player_identity returns team_code=None
    whenever a record spans more than one batter_team_code or carries none, so
    comparing the two codes for inequality lets two None records skip the check
    entirely: ambiguity would disable the tripwire instead of tripping it.

    That is precisely the feed change the clause exists to survive. Today every
    duplicate-group pid carries one clean code, so this is latent — two different
    men, each traded mid-season, sharing 28 calendar days under different clubs
    and no game_id, would have pooled into one spray chart.
    """
    a = _fp(games=["g-1"], days=["2025-06-01", "2025-06-02"], team_code=None,
            sides=_one_sided("R"))
    b = _fp(games=["g-2"], days=["2025-06-01"], team_code=None,
            sides=_one_sided("R"))
    assert app._merge_refusal(a, b) == "same day, two clubs"

    # One side known is still not a positive match.
    assert app._merge_refusal(_fp(days=["2025-06-01"], team_code="LAN"),
                              _fp(days=["2025-06-01"])) == "same day, two clubs"


def test_unknown_club_with_no_shared_day_still_merges():
    # The clause only bites on a shared day; a missing team_code alone is not a
    # refusal, or the 52 disjoint-span pairs would never pool.
    a = _fp(games=["g-1"], days=["2024-05-01"], team_code=None)
    b = _fp(games=["g-2"], days=["2025-05-01"], team_code=None)
    assert app._merge_refusal(a, b) is None


# ── 5. interleaved stints are a loan, not a collision ──────────────────────

def test_nested_stint_with_no_shared_day_still_merges():
    """The real Alonso, Alan shape: Southern Maryland 2024-04-25..06-27, then
    Gastonia 2024-06-28..2025-07-20, with a Hagerstown record sitting entirely
    inside the Gastonia span (2025-06-05..06-28) — a window in which Gastonia
    played no games at all. Zero shared game_ids, zero shared days, side agrees:
    one man on loan. Interior-to-span is NOT a disqualifier; a shared calendar
    day under two clubs is."""
    gastonia = _fp(games=["g-gas-1", "g-gas-2"],
                   days=["2024-06-28", "2025-05-27", "2025-07-16"],
                   team_code="GAS")
    hagerstown = _fp(games=["g-hag-1"],
                     days=["2025-06-05", "2025-06-28"], team_code="HAG_FLY")
    assert app._merge_refusal(gastonia, hagerstown) is None


# ── 6-7. measured batting side has to agree, and has to decide ─────────────

def test_contradicting_measured_side_refuses():
    a = _fp(days=["2024-05-01"], team_code="LEX_LEG", sides=_one_sided("L"))
    b = _fp(days=["2025-05-01"], team_code="SMD", sides=_one_sided("R"))
    assert app._merge_refusal(a, b) == "batting side conflict"


def test_no_decidable_side_refuses():
    """52 of the merged pairs never overlap in time, so the game and day tests
    have zero discriminating power on them. Side agreement is the only thing
    left standing between one traded man and two different men, so a pair that
    cannot decide a single platoon hand is refused rather than merged on the
    absence of a contradiction."""
    thin = _fp(sides={"R": ("L", 1.0, 4), "L": ("L", 1.0, 3)})
    assert app._merge_refusal(thin, _fp()) == "no corroborating side evidence"

    # Enough swings, but genuinely two-sided within the platoon — undecided.
    muddy = _fp(sides={"R": ("L", 0.55, 40), "L": ("L", 0.60, 40)})
    assert app._merge_refusal(muddy, _fp()) == "no corroborating side evidence"


def test_one_decided_hand_is_enough():
    # 18 of the 85 pooled pairs decide exactly one platoon hand; 67 decide both.
    a = _fp(sides={"R": ("L", 1.0, 40), "L": ("L", 1.0, 3)})
    b = _fp(sides={"R": ("L", 1.0, 55), "L": ("R", 1.0, 2)})
    assert app._merge_refusal(a, b) is None


# ── 8. a genuine switch hitter is compared per platoon, not on one side ────

def test_switch_hitter_is_not_split_by_the_side_test():
    # Bats left against RHP and right against LHP on BOTH records — comparing
    # raw dominant side would call this a conflict; comparing per platoon hand
    # (which is how the charts are drawn anyway) agrees twice over.
    a = _fp(games=["g-a"], days=["2024-05-01"],
            sides={"R": ("L", 1.0, 60), "L": ("R", 1.0, 25)})
    b = _fp(games=["g-b"], days=["2025-05-01"],
            sides={"R": ("L", 1.0, 40), "L": ("R", 1.0, 18)})
    assert app._merge_refusal(a, b) is None


# ── 9. roster handedness is never consulted ────────────────────────────────

def test_roster_handedness_is_ignored_in_favour_of_measured_side():
    """The stakeholder's complaint was a hitter listed once as "R" and once as
    "S". Measured, the "S" is feed noise and it does not mean left: de aza's S
    record measures Left, sandoval's S record measures Right, and in all 16
    conflicting-hand groups the measured side matches the sibling record. So
    contradicting roster flags must not block a merge, and agreeing ones must
    not rescue one.
    """
    records = [
        {"player_id": "p-s", "player_name": "Sandoval, Ariel",
         "player_batting_handedness": "Switch", "team_name": ""},
        {"player_id": "p-r", "player_name": "Sandoval, Ariel",
         "player_batting_handedness": "Right", "team_name": "Lancaster Stormers"},
    ]
    agreeing = {
        "p-s": _fp(games=["g1"], days=["2026-04-21"], team_code="WES_POW",
                   sides=_one_sided("R")),
        "p-r": _fp(games=["g2"], days=["2025-04-25"], team_code="LAN",
                   sides=_one_sided("R")),
    }
    assert len(app.build_player_dict(records, identities=agreeing)) == 1

    # Same roster flag on both records, contradicting measurement → refused.
    both_right = [
        {"player_id": "p-a", "player_name": "Doe, Sam",
         "player_batting_handedness": "Right", "team_name": "Team A"},
        {"player_id": "p-b", "player_name": "Doe, Sam",
         "player_batting_handedness": "Right", "team_name": "Team B"},
    ]
    contradicting = {
        "p-a": _fp(games=["g1"], team_code="A", sides=_one_sided("L")),
        "p-b": _fp(games=["g2"], team_code="B", sides=_one_sided("R")),
    }
    assert len(app.build_player_dict(both_right, identities=contradicting)) == 2


# ── 10. group closure: one proven collision refuses the whole name ─────────

def test_group_closure_refuses_every_pair_once_one_pair_collides():
    """The real McCarthy, Ryan group. 2561d7bf (Lexington, measures Left) and
    55d691d8 (Southern Maryland, measures Right) share 7 game_ids; 55d691d8 and
    61169326 (High Point, Left) share 5. The Lexington/High Point pair passes
    every check on its own — no shared game, no shared day, both Left, disjoint
    spans — and is most likely one lefty who moved clubs.

    It is refused anyway. Once a name is proven to cover two humans, "these two
    never collided" stops implying "same human", because the base rate that made
    absence-of-collision informative is gone. Closure keeps the invariant
    auditable in one line — an entity never spans a group containing a proven
    collision — and makes grouping order-independent. Cost: 1 group, 3 of 431
    entries, each of which gets a quantitative disclosure instead.
    """
    records = [
        {"player_id": "p-lex", "player_name": "McCarthy, Ryan",
         "player_batting_handedness": "Switch", "team_name": ""},
        {"player_id": "p-smd", "player_name": "McCarthy, Ryan",
         "player_batting_handedness": "Right",
         "team_name": "Southern Maryland Blue Crabs"},
        {"player_id": "p-hp", "player_name": "McCarthy, Ryan",
         "player_batting_handedness": "Left", "team_name": "High Point Rockers"},
    ]
    identities = {
        "p-lex": _fp(games=["g-shared-a", "g-lex"], team_code="LEX_LEG",
                     sides=_one_sided("L"), balls=(69, 14)),
        "p-smd": _fp(games=["g-shared-a", "g-shared-b"], team_code="SMD",
                     sides=_one_sided("R"), balls=(111, 32)),
        "p-hp": _fp(games=["g-shared-b", "g-hp"], team_code="HP",
                    sides=_one_sided("L"), balls=(69, 15)),
    }
    # The clean pair on its own would merge…
    assert app._merge_refusal(identities["p-lex"], identities["p-hp"]) is None

    # …but not inside a group that contains a proven collision.
    result = app.build_player_dict(records, identities=identities)
    assert len(result) == 3
    assert {"p-lex", "p-smd", "p-hp"} == set(result)
    for entry in result.values():
        assert entry["merged_ids"] == [entry["player_id"]]
        # Refusal is not silence — every partial sheet says how much it is
        # missing and where the rest of it is.
        assert entry["sample_note"].startswith("partial sample — ")
    assert result["p-lex"]["sample_note"] == (
        "partial sample — 227 more tracked balls under separate McCarthy, Ryan "
        "records (Southern Maryland Blue Crabs, High Point Rockers)")


def test_group_closure_is_order_independent():
    records = [
        {"player_id": "p-a", "player_name": "Roe, Pat",
         "player_batting_handedness": "Right", "team_name": "Team A"},
        {"player_id": "p-b", "player_name": "Roe, Pat",
         "player_batting_handedness": "Right", "team_name": "Team B"},
        {"player_id": "p-c", "player_name": "Roe, Pat",
         "player_batting_handedness": "Right", "team_name": "Team C"},
    ]
    identities = {
        "p-a": _fp(games=["g-a"], team_code="A", sides=_one_sided("R")),
        "p-b": _fp(games=["g-b"], team_code="B", sides=_one_sided("R")),
        "p-c": _fp(games=["g-b"], team_code="C", sides=_one_sided("R")),
    }
    forward = app.build_player_dict(records, identities=identities)
    backward = app.build_player_dict(list(reversed(records)), identities=identities)
    assert len(forward) == 3
    assert set(forward) == set(backward)


# ── 11. a whole entry says nothing ─────────────────────────────────────────

def test_unambiguous_entry_carries_no_sample_note():
    records = [
        {"player_id": "p1", "player_name": "Doe, Jane",
         "player_batting_handedness": "Right", "team_name": "Team A"},
    ]
    (entry,) = app.build_player_dict(records).values()
    assert entry["sample_note"] == ""
    assert entry["merged_ids"] == ["p1"]
