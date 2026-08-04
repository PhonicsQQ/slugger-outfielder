"""Tests for the pooled spray fetch (``_fetch_union_spray``).

Once two roster records are proven to be one hitter, the chart has to be drawn
from both of their batted balls — the render path previously fetched exactly one
player_id, which is why a coach picking "Sandoval, Ariel" positioned an outfield
off 106 of that hitter's 243 tracked balls.

Pooling introduces two failure modes this module pins: the same batted ball
arriving under two ids (the feed re-ingested a 2024-07-05 Hagerstown game under
two game_ids), and one member of the pool failing to load, which would recreate
the original partial-sample bug wearing a "merged" badge.

``fetch_player_spray`` is monkeypatched, so these tests never hit the network.
"""

import pytest

import app


# ── helpers ────────────────────────────────────────────────────────────────

def _row(**over):
    """A pitch row carrying only the fields the signature reads."""
    row = {
        "date": "2024-07-05", "inning": 1, "top_or_bottom": "Bottom",
        "pa_of_inning": 3, "pitch_of_pa": 6, "pitcher_id": "pit-1",
        "exit_speed": 98.38211, "angle": 11.49186,
        "direction": -27.448734, "distance": 331.0,
    }
    row.update(over)
    return row


@pytest.fixture
def calls(monkeypatch):
    """Record every fetch_player_spray call; serve rows from ``by_pid``."""
    seen = []
    by_pid = {}

    def _fake(player_id, pitcher_hand=None, start_date=None,
              end_date=None, limit=5000):
        seen.append(player_id)
        result = by_pid.get(player_id, [])
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(app, "fetch_player_spray", _fake)
    monkeypatch.setattr(app, "_union_ids", {})
    monkeypatch.setattr(app, "_sample_notes", {})
    return seen, by_pid


# ── 1. an unpooled batter costs exactly what it costs today ────────────────

def test_unpooled_batter_issues_one_call_and_returns_rows_untouched(calls):
    seen, by_pid = calls
    by_pid["solo"] = [_row(), _row(exit_speed=91.0)]

    rows, contributed = app._fetch_union_spray("solo", pitcher_hand="R")

    assert seen == ["solo"]
    assert rows == by_pid["solo"]
    assert contributed == ["solo"]


def test_pooled_batter_fetches_every_member(calls):
    seen, by_pid = calls
    app._union_ids.update({"a": ["a", "b"], "b": ["a", "b"]})
    by_pid["a"] = [_row()]
    by_pid["b"] = [_row(date="2025-05-01", pitcher_id="pit-2")]

    rows, contributed = app._fetch_union_spray("a", pitcher_hand="R")

    assert seen == ["a", "b"]
    assert len(rows) == 2
    assert contributed == ["a", "b"]


def test_a_folded_pid_resolves_to_the_whole_pool(calls):
    # A stale client holding the folded id must still get the whole hitter
    # rather than half of one.
    seen, by_pid = calls
    app._union_ids.update({"a": ["a", "b"], "b": ["a", "b"]})
    by_pid["a"] = [_row()]
    by_pid["b"] = [_row(date="2025-05-01")]

    _, contributed = app._fetch_union_spray("b")
    assert contributed == ["a", "b"]


# ── 2. the same ball under two ids is drawn once ───────────────────────────

def test_cross_pid_duplicate_rows_collapse(calls):
    """The literal Campagna, Joe values: the 2024-07-05 Hagerstown game arrived
    under game_ids 4ec96986 and 1bdc88e2 with identical Trackman to 5 decimals,
    so a pooled chart would otherwise draw those balls twice and weight the
    centroid toward them."""
    seen, by_pid = calls
    app._union_ids.update({"a": ["a", "b"]})
    by_pid["a"] = [_row()]
    by_pid["b"] = [_row(), _row(exit_speed=88.1)]

    rows, _ = app._fetch_union_spray("a")

    assert len(rows) == 2
    assert [r["exit_speed"] for r in rows] == [98.38211, 88.1]


def test_rows_differing_only_in_exit_speed_are_both_kept(calls):
    seen, by_pid = calls
    app._union_ids.update({"a": ["a", "b"]})
    by_pid["a"] = [_row()]
    by_pid["b"] = [_row(exit_speed=98.38212)]

    rows, _ = app._fetch_union_spray("a")
    assert len(rows) == 2


def test_null_physics_rows_are_still_pinned_by_the_remaining_fields(calls):
    # exit_speed/angle come through null on some rows; the six context fields
    # (date, inning, half, PA, pitch, pitcher) still identify the pitch.
    seen, by_pid = calls
    app._union_ids.update({"a": ["a", "b"]})
    blank = _row(exit_speed=None, angle=None)
    by_pid["a"] = [blank]
    by_pid["b"] = [dict(blank), _row(exit_speed=None, angle=None, pitch_of_pa=7)]

    rows, _ = app._fetch_union_spray("a")
    assert len(rows) == 2


def test_within_pid_duplicates_are_left_alone(calls):
    """The feed already carries 296 duplicate rows *within* single ids. Dropping
    those would silently shift 36 existing charts for a reason unrelated to
    pooling, so the first id's rows are kept verbatim."""
    seen, by_pid = calls
    by_pid["solo"] = [_row(), _row()]

    rows, _ = app._fetch_union_spray("solo")
    assert len(rows) == 2


# ── 3. a partial pool says so ──────────────────────────────────────────────

def test_failed_member_still_returns_the_other_rows_and_reports_the_gap(calls):
    seen, by_pid = calls
    app._union_ids.update({"a": ["a", "b"]})
    by_pid["a"] = [_row()]
    by_pid["b"] = RuntimeError("upstream 502")

    rows, contributed = app._fetch_union_spray("a")

    assert len(rows) == 1
    assert contributed == ["a"]
    # A silent partial pool would be the original bug wearing a merged badge.
    assert app._render_sample_note("a", contributed) == (
        "partial sample — 1 pooled record(s) could not be loaded")


def test_empty_member_is_reported_the_same_way(calls):
    seen, by_pid = calls
    app._union_ids.update({"a": ["a", "b"]})
    app._sample_notes["a"] = "sample pooled from 2 roster records"
    by_pid["a"] = [_row()]
    by_pid["b"] = []

    rows, contributed = app._fetch_union_spray("a")
    assert contributed == ["a"]
    note = app._render_sample_note("a", contributed)
    assert note.startswith("sample pooled from 2 roster records — ")
    assert "could not be loaded" in note


def test_whole_pool_reports_only_what_the_entry_claims(calls):
    seen, by_pid = calls
    app._union_ids.update({"a": ["a", "b"]})
    app._sample_notes.update({"a": "sample pooled from 2 roster records"})
    by_pid["a"] = [_row()]
    by_pid["b"] = [_row(date="2025-05-01")]

    _, contributed = app._fetch_union_spray("a")
    assert (app._render_sample_note("a", contributed)
            == "sample pooled from 2 roster records")


def test_unpooled_batter_has_no_note(calls):
    seen, by_pid = calls
    by_pid["solo"] = [_row()]
    _, contributed = app._fetch_union_spray("solo")
    assert app._render_sample_note("solo", contributed) == ""


# ── 4. /api/compute pools the union and returns the disclosure ─────────────

def test_api_compute_pools_the_union_and_returns_the_note(calls, monkeypatch):
    """The qualifying gate applies to the POOLED sample, which is the whole
    point: measured live, dead (player, pitcher-hand) chart slices inside the
    duplicate-name groups fall from 121 of 312 to 16 of 154."""
    import pandas as pd

    seen, by_pid = calls
    app._union_ids.update({"a": ["a", "b"]})
    app._sample_notes.update({"a": "sample pooled from 2 roster records"})
    by_pid["a"] = [_row()]
    by_pid["b"] = [_row(date="2025-05-01")]

    monkeypatch.setattr(app, "USE_API_ADAPTER", True)
    monkeypatch.setattr(app, "USE_JSON_LOADER", False)
    monkeypatch.setattr(app, "MIN_QUALIFYING_BALLS", 1)
    monkeypatch.setattr(app, "make_plot_with_image",
                        lambda *a, **k: "stub-png")
    monkeypatch.setattr(app, "resolve_batter_meta",
                        lambda *a, **k: {"label": "De Aza, Alejandro (S)",
                                         "batter_hand": "S"})
    import data_loader
    monkeypatch.setattr(
        data_loader, "parse_spray_to_dataframe",
        lambda rows: pd.DataFrame({
            "x": [1.0] * len(rows), "y": [2.0] * len(rows),
            "hang_time": [None] * len(rows), "outcome": [None] * len(rows),
        }))

    resp = app.app.test_client().post(
        "/api/compute", json={"batter_id": "a", "pitcher_hand": "RHP"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert seen == ["a", "b"]
    assert data["sample_note"] == "sample pooled from 2 roster records"


# ── 5. _publish_unions keys every member of a pool ─────────────────────────

def test_publish_unions_keys_primary_and_folded_ids(calls):
    batters = {
        "a": {"merged_ids": ["a", "b"],
              "sample_note": "sample pooled from 2 roster records"},
        "c": {"merged_ids": ["c"], "sample_note": ""},
    }
    app._publish_unions(batters)

    assert app._union_ids["a"] == ["a", "b"]
    assert app._union_ids["b"] == ["a", "b"]
    assert app._union_ids["c"] == ["c"]
    assert app._sample_notes["b"] == "sample pooled from 2 roster records"
    assert "c" not in app._sample_notes


# ── 6. the list never folds a record the chart cannot reach ────────────────

def test_identity_pass_publishes_nothing_until_it_has_finished(calls, monkeypatch):
    """The dropdown folds records using _identity_cache while the chart pools
    ids using _union_ids, so a fingerprint must not become visible before its
    pool does.

    Writing fingerprints into _identity_cache inside the fetch loop opened a
    window — the whole pass, 156 paced upstream fetches, on every worker start —
    where De Aza listed once under Southern Maryland, charted off 375 of his 414
    tracked balls with an empty sample_note, and the 39-ball FerryHawks record
    was neither pooled nor reachable: the original bug, with the missing half
    now deleted from the list. A pass that dies partway must leave every record
    separate, which is what its caller already reports.
    """
    monkeypatch.setattr(app, "_identity_cache", {})
    monkeypatch.setattr(app, "USE_API_ADAPTER", True)
    monkeypatch.setattr(app, "_cache_ready", True)
    monkeypatch.setattr(app, "_players_with_data_cache", {"p-smd": True, "p-sta": True})

    fps = {
        "p-smd": {"games": frozenset({"g1"}), "days": frozenset({"2024-05-01"}),
                  "team_code": "SMD", "sides": {"R": ("L", 1.0, 40),
                                                "L": ("L", 1.0, 40)},
                  "balls": {"R": 30, "L": 30}, "truncated": False, "error": False},
        "p-sta": {"games": frozenset({"g2"}), "days": frozenset({"2025-05-01"}),
                  "team_code": "STA_YAN", "sides": {"R": ("L", 1.0, 40),
                                                    "L": ("L", 1.0, 40)},
                  "balls": {"R": 30, "L": 30}, "truncated": False, "error": False},
    }
    players = [
        {"player_id": "p-smd", "player_name": "De Aza, Alejandro",
         "player_batting_handedness": "Switch",
         "team_name": "Southern Maryland Blue Crabs"},
        {"player_id": "p-sta", "player_name": "de aza, alejandro",
         "player_batting_handedness": "Left", "team_name": "Staten Island FerryHawks"},
    ]

    seen_mid_pass = []

    def _probe(pid):
        # What a request landing mid-pass would resolve: the fold must not have
        # happened yet, and _union_ids must not be half-written either.
        entries = app.build_player_dict(players, identities=app._identity_cache)
        seen_mid_pass.append((len(entries), dict(app._union_ids)))
        return fps[pid]

    monkeypatch.setattr(app, "probe_player_identity", _probe)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    app._resolve_identities(players)

    assert seen_mid_pass == [(2, {}), (2, {})]
    # …and once it lands, both surfaces agree.
    assert len(app.build_player_dict(players, identities=app._identity_cache)) == 1
    assert app._union_ids["p-sta"] == ["p-sta", "p-smd"]


def test_cache_status_reports_merge_readiness_separately(monkeypatch):
    """The probe finishes before the identity pass, so /api/cache-status has to
    say so: a frontend that stops polling on `ready` keeps the pre-merge list
    forever and shows both halves of a pooled hitter, each rendering the same
    union under a different team.
    """
    monkeypatch.setattr(app, "fetch_players", lambda limit=5000: [])
    monkeypatch.setattr(app, "_cache_ready", True)
    monkeypatch.setattr(app, "_merge_ready", False)

    client = app.app.test_client()
    body = client.get("/api/cache-status").get_json()
    assert body["ready"] is True and body["merged"] is False

    monkeypatch.setattr(app, "_merge_ready", True)
    assert client.get("/api/cache-status").get_json()["merged"] is True
