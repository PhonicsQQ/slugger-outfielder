"""Warm-up runs once, the pool is read once, and a typo is a 400.

Three leftovers from the 2026-08-04 bug hunt whose verification agents were
killed before they finished. Each is reproduced here before being fixed:

  * a cold page load started the 2072-player probe TWICE — ``/`` renders the
    template and the React app then fetches ``/api/batters``, and both requests
    land inside the ~0.5s before the first probe result is recorded, so both
    pass the ``not _cache_ready and not _players_with_data_cache`` gate;
  * the render read ``_union_ids`` once for the fetch and again for the
    disclosure, several seconds apart, so an identity pass landing in between
    made the note describe a pool the chart never drew from;
  * ``?limit=abc`` reached ``int()`` bare, so a typed query string was reported
    as a server fault with a Python message in the body.
"""

from __future__ import annotations

import threading
import time

import pytest

import app


# ── 1. One warm-up per container ─────────────────────────────────────────────

@pytest.fixture
def probe_recorder(monkeypatch):
    """Count real warm-up passes, with the upstream work stubbed out.

    Real threads are used — faking ``threading.Thread`` would fake it for the
    test itself, since ``app.threading`` is the same module object.
    """
    passes: list[int] = []

    monkeypatch.setattr(app, "_probe_started", False)
    monkeypatch.setattr(app, "_cache_ready", False)
    monkeypatch.setattr(app, "_merge_ready", False)
    monkeypatch.setattr(app, "_players_with_data_cache", {})
    monkeypatch.setattr(app, "_probe_one_player", lambda pid: True)
    monkeypatch.setattr(app, "_resolve_identities", lambda batch: passes.append(1))
    return passes


def _await_warmup(timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app._merge_ready:
            return
        time.sleep(0.05)
    raise AssertionError("warm-up never finished")


def test_two_cold_requests_start_one_probe(probe_recorder) -> None:
    """`/` and `/api/batters` both arrive before the first result is recorded."""
    players = [{"player_id": f"p{i}"} for i in range(2)]

    app._start_background_probe(players)   # index()
    app._start_background_probe(players)   # api_batters(), ~300ms later
    _await_warmup()

    assert probe_recorder == [1], (
        f"{len(probe_recorder)} warm-up passes ran — every cold container "
        "would probe the whole 2072-player roster twice"
    )


def test_a_burst_of_requests_starts_one_probe(probe_recorder) -> None:
    """Same guarantee when the requests are genuinely simultaneous."""
    players = [{"player_id": f"p{i}"} for i in range(2)]
    callers = [
        threading.Thread(target=app._start_background_probe, args=(players,))
        for _ in range(8)
    ]
    for t in callers:
        t.start()
    for t in callers:
        t.join()
    _await_warmup()

    assert probe_recorder == [1]


def test_a_failed_probe_can_be_retried(probe_recorder, monkeypatch) -> None:
    """Once-per-container must not mean once-ever after a crash.

    Before the latch, a pass that died left `_cache_ready` False and the cache
    empty, so the next request simply started another. That has to stay true.
    """
    boom = {"fail": True}

    def flaky(pid: str) -> bool:
        if boom["fail"]:
            raise RuntimeError("upstream refused")
        return True

    monkeypatch.setattr(app, "_probe_one_player", flaky)
    players = [{"player_id": "p0"}]

    app._start_background_probe(players)
    deadline = time.monotonic() + 10
    while app._probe_started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not app._probe_started, "a dead pass left the latch closed forever"

    boom["fail"] = False
    app._start_background_probe(players)
    _await_warmup()

    assert probe_recorder == [1], "the retry never ran"


# ── 2. The disclosure describes the pool the chart actually used ─────────────

def test_the_note_counts_the_pool_the_fetch_used(monkeypatch) -> None:
    """An identity pass landing mid-fetch must not rewrite the disclosure."""
    monkeypatch.setattr(app, "_union_ids", {"a": ["a"]})
    monkeypatch.setattr(app, "_sample_notes", {})

    pool_ids = app._pool_for("a")

    # The identity pass publishes while the (seconds-long) fetch is in flight.
    monkeypatch.setattr(app, "_union_ids", {"a": ["a", "b"], "b": ["a", "b"]})
    monkeypatch.setattr(
        app, "_sample_notes",
        {"a": "sample pooled from 2 roster records",
         "b": "sample pooled from 2 roster records"},
    )

    note = app._render_sample_note("a", contributed=["a"], pool_ids=pool_ids)

    assert "could not be loaded" not in note, (
        "the chart drew the only record it was asked for, but the note "
        f"reported a failure: {note!r}"
    )


def test_a_genuinely_missing_record_is_still_disclosed(monkeypatch) -> None:
    """The fix must not silence the disclosure it exists for."""
    monkeypatch.setattr(app, "_union_ids", {"a": ["a", "b"]})
    monkeypatch.setattr(app, "_sample_notes", {"a": "sample pooled from 2 roster records"})

    pool_ids = app._pool_for("a")
    note = app._render_sample_note("a", contributed=["a"], pool_ids=pool_ids)

    assert "1 pooled record(s) could not be loaded" in note


def test_the_fetch_uses_the_pool_it_was_handed(monkeypatch) -> None:
    """The handed-in pool wins over whatever `_union_ids` says right now."""
    fetched: list[str] = []

    def fake_fetch(player_id, **kwargs):
        fetched.append(player_id)
        return [{"date": "2024-07-05", "distance": 300.0, "pitcher_id": player_id}]

    monkeypatch.setattr(app, "fetch_player_spray", fake_fetch)
    monkeypatch.setattr(app, "_union_ids", {"a": ["a", "b", "c"]})

    rows, contributed = app._fetch_union_spray("a", pool_ids=["a", "b"])

    assert fetched == ["a", "b"], "record 'c' was published after this fetch began"
    assert contributed == ["a", "b"]
    assert len(rows) == 2


# ── 3. A mistyped query string is the client's error ─────────────────────────

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app, "USE_API_ADAPTER", True)
    monkeypatch.setattr(app, "USE_JSON_LOADER", False)
    app.app.config["TESTING"] = True
    return app.app.test_client()


@pytest.mark.parametrize("query", ["limit=abc", "page=xyz", "limit=1.5"])
def test_non_integer_paging_is_a_400(client, monkeypatch, query) -> None:
    monkeypatch.setattr(app, "fetch_ballparks", lambda **kwargs: [])

    resp = client.get(f"/api/ballparks?{query}")

    assert resp.status_code == 400
    assert "whole number" in resp.get_json()["error"]


def test_valid_paging_still_reaches_the_adapter(client, monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(
        app, "fetch_ballparks",
        lambda **kwargs: seen.update(kwargs) or [{"ballpark_id": "x"}],
    )

    resp = client.get("/api/ballparks?limit=7&page=3")

    assert resp.status_code == 200
    assert seen["limit"] == 7 and seen["page"] == 3


def test_omitted_paging_keeps_its_default(client, monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(
        app, "fetch_ballparks",
        lambda **kwargs: seen.update(kwargs) or [],
    )

    client.get("/api/ballparks")

    assert seen["limit"] == 50 and seen["page"] == 1


def test_a_real_adapter_failure_is_still_a_500(client, monkeypatch) -> None:
    """Narrowing to 400 must not swallow genuine faults."""
    def boom(**kwargs):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(app, "fetch_ballparks", boom)

    assert client.get("/api/ballparks?limit=10").status_code == 500
