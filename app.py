# app.py — SLUGGER Outfield Positioning Optimizer
# -*- coding: utf-8 -*-

import io
import base64
import sys
import os
import re
import logging
import threading
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

from dotenv import load_dotenv
load_dotenv()

# ── Windows UTF-8 fix ────────────────────────────────────
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "buffer"):
            setattr(
                sys,
                stream.name,
                io.TextIOWrapper(stream.buffer, encoding="utf-8",
                                 errors="replace", line_buffering=True),
            )

# ── Core imports ─────────────────────────────────────────
from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Arc, Patch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ═════════════════════════════════════════════════════════
#  SUB-PATH MOUNTING (behind the shared ALB)
# ═════════════════════════════════════════════════════════
# When hosted at e.g. https://www.alpb-analytics.com/widgets/outfielder/, the
# load balancer forwards the full path. Strip the prefix so the existing routes
# ("/", "/api/...") match unchanged. Defaults to "" → no-op for local dev and
# the legacy PythonAnywhere host. The frontend reads the same value (injected as
# `url_prefix`) to prefix its fetch() calls.
URL_PREFIX = os.getenv("URL_PREFIX", "").rstrip("/")


class _PrefixMiddleware:
    def __init__(self, wsgi_app, prefix):
        self.wsgi_app = wsgi_app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path.startswith(self.prefix):
            environ["PATH_INFO"] = path[len(self.prefix):] or "/"
            environ["SCRIPT_NAME"] = self.prefix
        return self.wsgi_app(environ, start_response)


if URL_PREFIX:
    app.wsgi_app = _PrefixMiddleware(app.wsgi_app, URL_PREFIX)


@app.context_processor
def _inject_url_prefix():
    return {"url_prefix": URL_PREFIX}


@app.route("/healthz")
def healthz():
    """Lightweight liveness probe for the load balancer."""
    return {"status": "ok"}, 200


# ═════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════

LAST_CSV_PATH = "optimized_positions.csv"
DEFAULT_BACKGROUND = "img/background.png"

BATTERS: Dict[str, Dict] = {
    "dickerson_L": {
        "label": "Corey Dickerson (L)",
        "batter_name": "Corey Dickerson",
        "batter_hand": "L",
    },
    "dickerson_R": {
        "label": "Corey Dickerson (R)",
        "batter_name": "Corey Dickerson",
        "batter_hand": "R",
    },
}

# Pixel calibration for background image spray mapping
SPRAY_PIXEL_CONFIG = {
    "outfield_top_px":    720,
    "outfield_bottom_px": 930,
    "home_x_px":          1170,
    "home_y_px":          1600,
    "lf_pole_x_px":       248,
    "lf_pole_y_px":       848,
    "rf_pole_x_px":       2114,
    "rf_pole_y_px":       854,
    "dist_min":           150.0,
    "dist_max":           400.0,
    "dir_min":            -38.0,
    "dir_max":            38.0,
}

OUTCOME_COLORS = {
    "OUT": "#bdbdbd", "SINGLE": "#42a5f5", "DOUBLE": "#e040fb",
    "TRIPLE": "#ffa726", "HOMERUN": "#ef5350",
    "1B": "#42a5f5", "2B": "#e040fb", "3B": "#ffa726", "HR": "#ef5350",
}

# ── Depth-based weighting config ─────────────────────────
# Deeper balls carry more positional leverage — mispositioning
# on a deep fly is catastrophic, while shallow bloopers barely
# matter.  These thresholds control both the optimizer weight
# and the visualization outcome caps.
DEPTH_WEIGHT_CONFIG = {
    "shallow_cutoff": 200.0,   # feet — below this is a blooper
    "deep_cutoff":    260.0,   # feet — above this is a deep fly
    "shallow_weight":   0.5,
    "medium_weight":    1.0,
    "deep_weight":      4.0,   # ← bump this to push fielders deeper
}

# ── Coach-standard alignment constraints ─────────────────
# The depth-weighted centroid can drop a fielder somewhere a coach would
# never stand: pulled hard off the alignment line, buried too shallow/deep,
# or stacked on top of a neighbour. Coaches align outfielders around three
# anchors — LF ≈ −27°, CF ≈ 0°, RF ≈ +27° — and read a placement as "crazy"
# when it sits more than ±13° from its anchor, is closer than 18° to a
# neighbour, or falls outside a sane depth band. We sanity-clamp each raw
# placement back into these windows before it is drawn.
#
# Note the anchor±13 windows (e.g. LF −40..−14) run past the ±38 pixel
# mapping range, so each window is *intersected* with (dir_min, dir_max):
# LF collapses to −38..−14, CF to −13..13, RF to 14..38 — three disjoint
# bands. Disjointness is what makes left→right ordering structural rather
# than something the separation pass has to defend.
OF_CONSTRAINTS_ENABLED = os.getenv("OF_CONSTRAINTS_ENABLED", "true").lower() == "true"
OF_ANGLE_ANCHORS = {"LF": -27.0, "CF": 0.0, "RF": 27.0}
OF_MAX_ANGLE_DEVIATION = 13.0
OF_DEPTH_BOUNDS = {"LF": (240.0, 325.0), "CF": (260.0, 345.0), "RF": (240.0, 325.0)}
OF_MIN_SEPARATION_DEG = 18.0
OF_CLAMP_NOTE_MIN_DELTA = (0.5, 5.0)   # (deg, ft) — deltas above this count as "engaged"
DEPTH_POSITION_PERCENTILE = 30  # moved from make_plot_with_image local (line 648)


def _angle_window(name: str) -> Tuple[float, float]:
    """Anchor ±deviation intersected with the (dir_min, dir_max) mapping range."""
    anchor = OF_ANGLE_ANCHORS[name]
    cfg = SPRAY_PIXEL_CONFIG
    lo = max(anchor - OF_MAX_ANGLE_DEVIATION, cfg["dir_min"])
    hi = min(anchor + OF_MAX_ANGLE_DEVIATION, cfg["dir_max"])
    return (lo, hi)


def _validate_of_constraints() -> None:
    """Fail fast at import if the constraint config is internally inconsistent."""
    cfg = SPRAY_PIXEL_CONFIG
    order = ["LF", "CF", "RF"]
    anchors = [OF_ANGLE_ANCHORS[n] for n in order]
    if not (anchors[0] < anchors[1] < anchors[2]):
        raise ValueError(f"OF_ANGLE_ANCHORS must strictly increase LF<CF<RF: {anchors}")

    windows = {n: _angle_window(n) for n in order}
    for n in order:
        lo, hi = windows[n]
        if lo >= hi:
            raise ValueError(f"angle window for {n} empty after intersection: {(lo, hi)}")
    # Adjacent windows must stay disjoint so ordering is structural.
    for a, b in zip(order, order[1:]):
        if windows[a][1] >= windows[b][0]:
            raise ValueError(
                f"angle windows {a} {windows[a]} and {b} {windows[b]} overlap")

    dmin, dmax = cfg["dist_min"], cfg["dist_max"]
    for n in order:
        lo, hi = OF_DEPTH_BOUNDS[n]
        if not (dmin <= lo < hi <= dmax):
            raise ValueError(
                f"OF_DEPTH_BOUNDS[{n}]={(lo, hi)} must sit inside ({dmin}, {dmax})")


_validate_of_constraints()


# ═════════════════════════════════════════════════════════
#  DATA SOURCE DETECTION
# ═════════════════════════════════════════════════════════

USE_API_MODE_ENV = os.getenv("USE_API_MODE", "false")
USE_API_MODE = USE_API_MODE_ENV.lower() == "true"

# JSON loader
USE_JSON_LOADER = False
if not USE_API_MODE:
    try:
        from data_loader import (
            load_players,
            get_player_spray_dataframe,
            get_unique_players_with_spray_data,
            filter_players_by_handedness,
            parse_spray_to_dataframe,
        )
        USE_JSON_LOADER = True
    except ImportError:
        pass

# API adapter
USE_API_ADAPTER = False
MIN_QUALIFYING_BALLS = 15
try:
    from adapter import (
        fetch_ballparks, fetch_games, fetch_player_spray,
        fetch_players, probe_player_identity, MIN_QUALIFYING_BALLS,
    )
    USE_API_ADAPTER = True
except ImportError:
    pass

# Excel-based optimizer
USE_EXCEL_ALGORITHM = False
try:
    from optimizer import optimize_outfield_excel
    USE_EXCEL_ALGORITHM = True
except ImportError:
    pass

log.info("=" * 60)
log.info("Data mode: %s",
         "API Adapter" if (USE_API_ADAPTER and not USE_JSON_LOADER)
         else "JSON Loader" if USE_JSON_LOADER
         else "Synthetic Fallback")
log.info("Min qualifying balls: %d", MIN_QUALIFYING_BALLS)
log.info("=" * 60)


# ═════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═════════════════════════════════════════════════════════

def normalize_hand(raw: str) -> str:
    """Normalize batting/pitching handedness to single letter.
    L = Left, R = Right, S = Switch, U = Unknown/missing."""
    raw = str(raw).strip().upper()
    if raw in ("LEFT", "L"):
        return "L"
    if raw in ("RIGHT", "R"):
        return "R"
    if raw in ("SWITCH", "S"):
        return "S"
    return "U"


def is_valid_player_name(name: str) -> bool:
    """Return True if name is usable for display."""
    name = name.strip()
    if not name or len(name) < 2:
        return False
    if name.startswith(",") or name == ",":
        return False
    if not re.search(r"[a-zA-Z0-9]", name):
        return False
    return True


def _uppercase_count(name: str) -> int:
    """Count uppercase letters — used to prefer a properly-cased label over
    an all-lowercase duplicate of the same name."""
    return sum(1 for c in name if c.isupper())


def _merge_hand(existing: str, incoming: str) -> str:
    """Merge two normalized handedness codes for the same player.

    Prefer a populated value over 'U' (unknown/missing), and prefer 'S'
    (switch) over a one-sided 'L'/'R' when both are populated.
    """
    if existing == incoming:
        return existing
    if existing == "U":
        return incoming
    if incoming == "U":
        return existing
    if "S" in (existing, incoming):
        return "S"
    # Two differing one-sided values (L vs R) for the same name are unexpected;
    # keep the first-seen value for stability.
    return existing


def _team_qualifier(team: str) -> str:
    """Suffix that tells two same-name roster records apart in the dropdown.
    The feed leaves team_name blank on a large slice of records, so say so
    plainly rather than leaving two identical rows."""
    return f" — {team}" if team else " — no team listed"


# Two records only corroborate each other on a pitcher hand where both carry
# enough measured swings to be one-sided. The constants sit on a measured
# plateau rather than a cliff: min_n 5 or 10 × purity 0.90 or 0.95 all merge
# the same 74 name groups; refusal only starts at min_n 20 (71) and 25 (61).
_SIDE_MIN_BALLS = 10
_SIDE_PURITY = 0.90


def _merge_refusal(fa: Optional[Dict], fb: Optional[Dict]) -> Optional[str]:
    """Why two same-name records must NOT be pooled, or None to pool them.

    Merging is refused by default and allowed only on positive proof, because
    the two failure modes are not symmetric: leaving one traded hitter split
    shows a coach half a spray chart, while splicing two different men produces
    a confident alignment where neither of them hits the ball — undetectable on
    the sheet. Clauses run most-damning first so the reported reason is the one
    that actually settles it.
    """
    if not fa or not fb:
        return "missing evidence"
    if fa.get("truncated") or fb.get("truncated") or fa.get("error") or fb.get("error"):
        return "truncated evidence"
    # One human cannot appear twice inside one game_id. Measured: this is the
    # tripwire with a proven positive (McCarthy, Ryan — 7 shared games on one
    # pair, 5 on another).
    if fa["games"] & fb["games"]:
        return "shared game"
    # …nor suit up for two clubs on one calendar day. Same-day-same-club is an
    # ingest artifact (the 2024-07-05 Hagerstown game arrived under two
    # game_ids with byte-identical Trackman), so the club test is required —
    # but it has to be a *positive* match. team_code is None whenever a record
    # spans more than one club, which is exactly what a feed that stopped
    # minting one id per stint would produce, so an unknown club reads as
    # "cannot tell" and the shared day still refuses.
    same_club = bool(fa.get("team_code")) and fa.get("team_code") == fb.get("team_code")
    if not same_club and (fa["days"] & fb["days"]):
        return "same day, two clubs"

    # Measured batting side, per platoon hand so a genuine switch hitter is not
    # split. Roster handedness is never consulted: it is contradicted by the
    # data on the very records at issue, and an "S" tag does not mean left.
    decided = 0
    for hand in ("R", "L"):
        side_a, purity_a, n_a = fa["sides"].get(hand, (None, 0.0, 0))
        side_b, purity_b, n_b = fb["sides"].get(hand, (None, 0.0, 0))
        if (n_a < _SIDE_MIN_BALLS or n_b < _SIDE_MIN_BALLS
                or purity_a < _SIDE_PURITY or purity_b < _SIDE_PURITY):
            continue
        if side_a != side_b:
            return "batting side conflict"
        decided += 1
    if not decided:
        # 52 of the merged pairs never overlap in time, so the collision tests
        # have no power on them and side agreement is the only thing standing
        # between one traded man and two different men.
        return "no corroborating side evidence"
    return None


def _last_tracked_day(group: Dict) -> str:
    """Latest date this roster record has a tracked pitch on, "" when unknown.

    Dates arrive ISO-formatted, so a plain string max orders them. Used to pick
    which of a pooled hitter's records names his current club.
    """
    fp = group.get("fp")
    days = fp.get("days") if fp else None
    return max(days) if days else ""


def _sample_note(members: List[Dict], refused: List[Dict]) -> str:
    """One-line disclosure telling a coach whether the sheet is whole.

    ``members`` are the roster records pooled into this entry and ``refused``
    the same-name records the evidence would not let us pool in; both carry the
    ``name``/``team``/``balls`` keys build_player_dict assembles. A pooled entry
    says how many records it stands on, a partial one says how much tracked
    evidence is sitting under the record it could not join, and a whole entry
    says nothing.
    """
    if len(members) > 1:
        return f"sample pooled from {len(members)} roster records"
    if not refused:
        return ""
    extra = sum(m.get("balls", 0) for m in refused)
    if not extra:
        return ""
    teams = ", ".join(m["team"] or "no team listed" for m in refused)
    article = "a separate" if len(refused) == 1 else "separate"
    noun = "record" if len(refused) == 1 else "records"
    return (f"partial sample — {extra} more tracked balls under {article} "
            f"{members[0]['name']} {noun} ({teams})")


def build_player_dict(players: list,
                      identities: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
    """
    Convert a list of player records into the batter dict format used by the
    frontend.

    Records are grouped by (casefolded name, casefolded team) so case-variant
    duplicates of the same human on the same team collapse into a single entry
    (e.g. "flores, santiago" + "Flores, Santiago"). Within a group the properly-
    cased variant wins the label, handedness merges (populated beats missing,
    switch beats a one-sided L/R), and the first data-bearing player_id is kept
    as the stable id — the probe filter has already dropped non-qualifying pids
    upstream.

    Records sharing a casefolded name across *teams* — including a record the
    feed left without a team — are one hitter's stints only when the pitch-feed
    fingerprints in ``identities`` (keyed by player_id, see
    adapter.probe_player_identity) prove it: no shared game, no shared day under
    two clubs, and measured batting side that agrees and actually decides. One
    refusal anywhere inside a name group refuses the whole group, because a name
    proven to cover two humans can no longer treat "these two never collided" as
    evidence of one. ``identities=None`` therefore reproduces the pre-merge
    output exactly — refusal is the literal default when no evidence exists.

    Every entry carries ``merged_ids`` (ordered, primary first, always non-empty)
    so the render path can pool the union, and ``sample_note``, which is empty
    only when the entry is whole.
    """
    groups: Dict[Tuple[str, str], Dict] = {}
    order: List[Tuple[str, str]] = []
    for p in players:
        pid = p.get("player_id")
        if not pid:
            continue
        name = (p.get("player_name") or "").strip()
        if not is_valid_player_name(name):
            continue
        hand = normalize_hand(p.get("player_batting_handedness") or "")
        team = (p.get("team_name") or "").strip()
        gkey = (name.casefold(), team.casefold())

        group = groups.get(gkey)
        if group is None:
            groups[gkey] = {
                "pid": pid,
                "name": name,
                "hand": hand,
                "team": team,
                "dropped": [],
            }
            order.append(gkey)
            continue

        # Duplicate of an already-seen human on the same team — merge in place.
        group["dropped"].append(str(pid))
        if _uppercase_count(name) > _uppercase_count(group["name"]):
            group["name"] = name
        group["hand"] = _merge_hand(group["hand"], hand)

    # ── Cross-team merge pass ────────────────────────────────────────────
    # Attach each group's fingerprint and its qualifying-ball count, then ask
    # _merge_refusal about every pair inside a casefolded-name bucket. A bucket
    # collapses to one entity only when every pair clears; one refusal keeps the
    # whole bucket split (see the docstring), so grouping is order-independent.
    fingerprints = identities or {}
    for gkey in order:
        group = groups[gkey]
        fp = fingerprints.get(str(group["pid"]))
        group["fp"] = fp
        group["balls"] = sum(fp["balls"].values()) if fp else 0

    buckets: Dict[str, List[Tuple[str, str]]] = {}
    for gkey in order:
        buckets.setdefault(gkey[0], []).append(gkey)

    merged_bucket: Dict[str, bool] = {}
    for cname, bucket in buckets.items():
        members = [groups[k] for k in bucket]
        refusal = None
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                refusal = _merge_refusal(a["fp"], b["fp"])
                if refusal:
                    break
            if refusal:
                break
        merged_bucket[cname] = len(members) > 1 and refusal is None
        if len(members) > 1 and refusal:
            log.info("Kept %d records for %s separate: %s",
                     len(members), members[0]["name"], refusal)

    # Primary pid = the team-bearing member seen most recently in the pitch
    # feed, so a merged entity leaves the "no team listed" bucket *and* is filed
    # under the club the hitter actually plays for. Roster order says nothing
    # about which stint is current, and a pooled name loses its team qualifier,
    # so a stale primary would drop the hitter out of his own team's dropdown
    # filter and team PDF with nothing on the label to show it. max() keeps the
    # first-seen record on a tie.
    primaries: Dict[str, Tuple[str, str]] = {}
    for cname, bucket in buckets.items():
        rostered = [k for k in bucket if groups[k]["team"]]
        primaries[cname] = (max(rostered, key=lambda k: _last_tracked_day(groups[k]))
                            if rostered else bucket[0])

    entries: List[Tuple[Tuple[str, str], Dict, List[Dict], List[Dict]]] = []
    for gkey in order:
        cname = gkey[0]
        bucket = buckets[cname]
        members = [groups[k] for k in bucket]
        if merged_bucket[cname]:
            # One entity, emitted at the primary record's position in the list.
            if gkey != primaries[cname]:
                continue
            primary = groups[primaries[cname]]
            pooled = [primary] + [m for m in members if m is not primary]
            entries.append((gkey, primary, pooled, []))
        else:
            group = groups[gkey]
            entries.append(
                (gkey, group, [group], [m for m in members if m is not group]))

    result: Dict[str, Dict] = {}
    # Recomputed after merging, so a name that collapsed to one entity loses the
    # now-redundant team suffix and only genuinely ambiguous names keep it.
    name_groups: Dict[str, int] = {}
    for gkey, _, _, _ in entries:
        name_groups[gkey[0]] = name_groups.get(gkey[0], 0) + 1
    for gkey, group, pooled, refused in entries:
        name, pid = group["name"], group["pid"]
        hand = group["hand"]
        # Same preference as the same-team merge: properly-cased label wins,
        # populated handedness beats missing, switch beats a one-sided L/R.
        for m in pooled[1:]:
            if _uppercase_count(m["name"]) > _uppercase_count(name):
                name = m["name"]
            hand = _merge_hand(hand, m["hand"])
        if group["dropped"]:
            log.info(
                "Merged duplicate player record(s) for %s (%s): kept pid %s, "
                "dropped %s",
                name, hand, pid, ", ".join(group["dropped"]),
            )
        merged_ids = [str(m["pid"]) for m in pooled]
        if len(merged_ids) > 1:
            log.info("Pooled %s (%s) from %d roster records: %s",
                     name, hand, len(merged_ids), ", ".join(merged_ids))
        label = f"{name} ({hand})"
        if name_groups[gkey[0]] > 1:
            label += _team_qualifier(group["team"])
        result[str(pid)] = {
            "label": label,
            "batter_name": name,
            "batter_hand": hand,
            "player_id": pid,
            "team_name": group["team"],
            "merged_ids": merged_ids,
            "sample_note": _sample_note(pooled, refused),
        }
    return result


def resolve_batter_meta(batter_id: str,
                        client_name: Optional[str] = None,
                        client_hand: Optional[str] = None) -> Dict:
    """
    Resolve a batter's display metadata.
    Priority: client-provided → API lookup → fallback.
    """
    if client_name and client_name.strip():
        name = client_name.strip()
        hand = normalize_hand(client_hand or "R")
    elif USE_API_ADAPTER:
        try:
            players = fetch_players(limit=5000)
            match = next((p for p in players
                          if p.get("player_id") == batter_id), None)
            if match:
                name = match.get("player_name") or f"Player {batter_id[:8]}"
                hand = normalize_hand(
                    match.get("player_batting_handedness") or "R")
            else:
                name, hand = f"Player {batter_id[:8]}", "R"
        except Exception:
            name, hand = f"Player {batter_id[:8]}", "R"
    else:
        name, hand = f"Player {batter_id[:8]}", "R"

    return {
        "label": f"{name} ({hand})",
        "batter_name": name,
        "batter_hand": hand,
        "player_id": batter_id,
    }


def _prepare_synthetic(batter_id: str, pitcher_hand: str):
    """
    Build synthetic spray DataFrame + positions for fallback mode.
    Returns (df, positions_drawn).
    """
    df_drawn = generate_spray(batter_id, pitcher_hand)
    df = df_drawn.copy()
    df["x"] = (df_drawn["x"] - 150) * 0.5
    df["y"] = (df_drawn["y"] - 200) * 2.0
    df["hang_time"] = 3.0
    positions_drawn = optimize_outfield(df_drawn)
    df_drawn = assign_distance_based_outcomes(df_drawn, positions_drawn)
    df["outcome"] = df_drawn["outcome"].values[: len(df)]
    return df, positions_drawn


def _get_depth_weight(distance: float) -> float:
    """Return depth-based importance weight for a batted ball distance."""
    cfg = DEPTH_WEIGHT_CONFIG
    if distance <= 0 or np.isnan(distance):
        return cfg["medium_weight"]
    if distance < cfg["shallow_cutoff"]:
        return cfg["shallow_weight"]
    elif distance > cfg["deep_cutoff"]:
        return cfg["deep_weight"]
    return cfg["medium_weight"]


# ═════════════════════════════════════════════════════════
#  OUTFIELD PLACEMENT CONSTRAINTS
# ═════════════════════════════════════════════════════════

def pixel_to_angle_dist(px: float, py: float) -> Tuple[float, float]:
    """Exact inverse of the forward (angle, dist) → pixel mapping.

    Mirrors lines 609-628: distance from vertical band position, then angle
    from horizontal position between the two pole lines *recomputed at this py*.
    """
    cfg = SPRAY_PIXEL_CONFIG
    top, bottom = cfg["outfield_top_px"], cfg["outfield_bottom_px"]
    dist = cfg["dist_min"] + (bottom - py) / (bottom - top) * (cfg["dist_max"] - cfg["dist_min"])

    x_left = (cfg["home_x_px"]
              + (py - cfg["home_y_px"])
              * (cfg["lf_pole_x_px"] - cfg["home_x_px"])
              / (cfg["lf_pole_y_px"] - cfg["home_y_px"]))
    x_right = (cfg["home_x_px"]
               + (py - cfg["home_y_px"])
               * (cfg["rf_pole_x_px"] - cfg["home_x_px"])
               / (cfg["rf_pole_y_px"] - cfg["home_y_px"]))

    angle = cfg["dir_min"] + (px - x_left) / (x_right - x_left) * (cfg["dir_max"] - cfg["dir_min"])
    return angle, dist


def angle_dist_to_pixel(angle: float, dist: float) -> Tuple[float, float]:
    """Forward (angle, dist) → pixel mapping in float (no int(), no clip).

    Mirrors lines 609-628 exactly except for the rounding/clipping the display
    path applies. ``x_left``/``x_right`` are recomputed at the new ``pixel_y`` by
    construction, so it is the true inverse of ``pixel_to_angle_dist``.
    """
    cfg = SPRAY_PIXEL_CONFIG
    top, bottom = cfg["outfield_top_px"], cfg["outfield_bottom_px"]
    depth_frac = (dist - cfg["dist_min"]) / (cfg["dist_max"] - cfg["dist_min"])
    pixel_y = bottom - depth_frac * (bottom - top)

    dir_frac = (angle - cfg["dir_min"]) / (cfg["dir_max"] - cfg["dir_min"])
    x_left = (cfg["home_x_px"]
              + (pixel_y - cfg["home_y_px"])
              * (cfg["lf_pole_x_px"] - cfg["home_x_px"])
              / (cfg["lf_pole_y_px"] - cfg["home_y_px"]))
    x_right = (cfg["home_x_px"]
               + (pixel_y - cfg["home_y_px"])
               * (cfg["rf_pole_x_px"] - cfg["home_x_px"])
               / (cfg["rf_pole_y_px"] - cfg["home_y_px"]))
    pixel_x = x_left + dir_frac * (x_right - x_left)
    return pixel_x, pixel_y


def compute_raw_positions(balls_pixel: list) -> Dict[str, Tuple[float, float]]:
    """Depth-weighted lateral centroid + depth-percentile Y per LF/CF/RF third.

    Behaviour-identical extraction of the former inline block (lines 650-671).
    ``balls_pixel`` entries are (pixel_x, pixel_y, color, distance_ft).
    """
    optimized_pixel: Dict[str, Tuple[float, float]] = {}
    if balls_pixel:
        sorted_dots = sorted(balls_pixel, key=lambda d: d[0])
        third = max(1, len(sorted_dots) // 3)
        for name, dots in [("LF", sorted_dots[:third]),
                           ("CF", sorted_dots[third:2 * third]),
                           ("RF", sorted_dots[2 * third:])]:
            if dots:
                # Weighted centroid for lateral position
                total_w = 0.0
                wx_sum = 0.0
                for (px, py, _, dist) in dots:
                    w = _get_depth_weight(dist)
                    wx_sum += w * px
                    total_w += w
                avg_x = wx_sum / total_w

                # Percentile for depth — lower pixel_y = deeper
                ys = [py for (_, py, _, _) in dots]
                depth_y = float(np.percentile(ys, DEPTH_POSITION_PERCENTILE))

                optimized_pixel[name] = (avg_x, depth_y)
    return optimized_pixel


def _enforce_separation(
    left: str,
    right: str,
    angles: Dict[str, float],
    left_floor: float | None = None,
) -> None:
    """Push an adjacent (left, right) pair to ≥ OF_MIN_SEPARATION_DEG apart.

    Symmetric push of need/2 each, re-clipped to each window; any residual left
    by a window edge is pushed onto whichever side is not pinned. ``left_floor``
    tightens the left element's lower bound so enforcing this pair can never
    re-violate an already-satisfied pair to its left. With the default disjoint
    windows a full fix is always reachable — the log.warning is a safety net
    for a mis-tuned config.
    """
    llo, lhi = _angle_window(left)
    if left_floor is not None:
        llo = max(llo, left_floor)
    rlo, rhi = _angle_window(right)
    la, ra = angles[left], angles[right]
    gap = ra - la
    if gap >= OF_MIN_SEPARATION_DEG:
        return

    need = OF_MIN_SEPARATION_DEG - gap
    half = need / 2.0
    la2 = min(max(la - half, llo), lhi)
    ra2 = min(max(ra + half, rlo), rhi)

    if ra2 - la2 < OF_MIN_SEPARATION_DEG:
        residual = OF_MIN_SEPARATION_DEG - (ra2 - la2)
        eps = 1e-9
        left_pinned = la2 <= llo + eps
        right_pinned = ra2 >= rhi - eps
        if not right_pinned:
            ra2 = min(max(ra2 + residual, rlo), rhi)
        elif not left_pinned:
            la2 = min(max(la2 - residual, llo), lhi)
        if (ra2 - la2) < OF_MIN_SEPARATION_DEG - 1e-6:
            log.warning(
                "OF separation %s/%s could not reach %.1f° (windows too tight): "
                "gap=%.2f°", left, right, OF_MIN_SEPARATION_DEG, ra2 - la2)

    angles[left], angles[right] = la2, ra2


def compute_constrained_positions(
    raw_positions: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Dict]]:
    """Sanity-clamp raw fielder placements into coach alignment windows.

    Returns (clamped_positions, report) where report[name] carries the signed
    angle/distance deltas and an ``engaged`` flag (moved beyond the note
    thresholds). When disabled or given no fielders, the input is returned
    unchanged with an empty report.
    """
    if not OF_CONSTRAINTS_ENABLED or not raw_positions:
        return raw_positions, {}

    angle_th, dist_th = OF_CLAMP_NOTE_MIN_DELTA

    # Polar-convert the known fielders; unknown keys pass through untouched.
    raw_polar: Dict[str, Tuple[float, float]] = {}
    passthrough: Dict[str, Tuple[float, float]] = {}
    for name, (px, py) in raw_positions.items():
        if name in OF_ANGLE_ANCHORS:
            raw_polar[name] = pixel_to_angle_dist(px, py)
        else:
            passthrough[name] = (px, py)

    # Per-fielder clamp: angle into its window, distance into its depth band.
    angles: Dict[str, float] = {}
    dists: Dict[str, float] = {}
    for name, (a, d) in raw_polar.items():
        lo, hi = _angle_window(name)
        angles[name] = min(max(a, lo), hi)
        dlo, dhi = OF_DEPTH_BOUNDS[name]
        dists[name] = min(max(d, dlo), dhi)

    # Separation over adjacent present pairs, left to right. The second pair
    # gets a floor at LF's final angle + the minimum separation so pushing CF
    # left for RF's sake can never undo the already-satisfied (LF, CF) pair.
    if "LF" in angles and "CF" in angles:
        _enforce_separation("LF", "CF", angles)
    if "CF" in angles and "RF" in angles:
        floor = (
            angles["LF"] + OF_MIN_SEPARATION_DEG if "LF" in angles else None
        )
        _enforce_separation("CF", "RF", angles, left_floor=floor)

    # Back-convert and build the adjustment report.
    clamped: Dict[str, Tuple[float, float]] = dict(passthrough)
    report: Dict[str, Dict] = {}
    for name, (a0, d0) in raw_polar.items():
        a1, d1 = angles[name], dists[name]
        clamped[name] = angle_dist_to_pixel(a1, d1)
        d_angle = a1 - a0
        d_dist = d1 - d0
        engaged = abs(d_angle) > angle_th or abs(d_dist) > dist_th
        report[name] = {"engaged": engaged, "d_angle": d_angle, "d_dist": d_dist}
        if engaged:
            log.info("OF constraint engaged for %s: Δangle=%.1f° Δdist=%.1fft",
                     name, d_angle, d_dist)
    return clamped, report


# ═════════════════════════════════════════════════════════
#  SYNTHETIC SPRAY GENERATION
# ═════════════════════════════════════════════════════════

def generate_spray(batter_id: str, pitcher_hand: str) -> pd.DataFrame:
    """Generate synthetic spray clusters for demo/fallback."""
    bhand = BATTERS[batter_id]["batter_hand"] if batter_id in BATTERS else "R"
    seed = abs(hash(batter_id + "_" + pitcher_hand)) % (2**32)
    rng = np.random.default_rng(seed)
    n = 150

    cluster_map = {
        ("L", "RHP"): [(200,320,30,30,45),(170,290,25,35,35),(150,340,20,25,25),(120,280,30,30,25),(90,310,25,25,20)],
        ("L", "LHP"): [(150,310,35,30,40),(180,290,25,35,30),(110,300,30,30,30),(200,330,20,20,25),(80,320,20,25,25)],
        ("R", "LHP"): [(100,320,30,30,45),(130,290,25,35,35),(150,340,20,25,25),(180,280,30,30,25),(210,310,25,25,20)],
        ("R", "RHP"): [(150,310,35,30,35),(120,290,25,35,30),(190,300,30,30,30),(100,330,20,20,25),(210,320,20,25,30)],
    }
    clusters = cluster_map.get((bhand, pitcher_hand), cluster_map[("R", "RHP")])

    xs, ys = [], []
    for cx, cy, sx, sy, count in clusters:
        xs.append(rng.normal(cx, sx, count))
        ys.append(rng.normal(cy, sy, count))

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(x))
    x = np.clip(x[idx][:n], 50, 250)
    y = np.clip(y[idx][:n], 230, 400)

    return pd.DataFrame({"x": x, "y": y})


# ═════════════════════════════════════════════════════════
#  BASIC OPTIMIZER (grid search)
# ═════════════════════════════════════════════════════════

def optimize_outfield(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Brute-force grid search for LF / CF / RF positions."""
    lf_grid = [(x, y) for x in range(70, 120, 10) for y in range(260, 330, 10)]
    cf_grid = [(x, y) for x in range(120, 180, 10) for y in range(310, 380, 10)]
    rf_grid = [(x, y) for x in range(180, 230, 10) for y in range(260, 330, 10)]

    bx, by = df["x"].to_numpy(), df["y"].to_numpy()
    best_score, best = float("inf"), {}

    for lf in lf_grid:
        dlf = np.hypot(bx - lf[0], by - lf[1])
        for cf in cf_grid:
            dcf = np.hypot(bx - cf[0], by - cf[1])
            for rf in rf_grid:
                drf = np.hypot(bx - rf[0], by - rf[1])
                score = np.minimum(np.minimum(dlf, dcf), drf).sum()
                if score < best_score:
                    best_score = score
                    best = {"LF": lf, "CF": cf, "RF": rf}
    return best


def assign_distance_based_outcomes(
    df: pd.DataFrame,
    positions: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """Label each ball as OUT / SINGLE / DOUBLE by proximity to fielders."""
    bx, by = df["x"].to_numpy(), df["y"].to_numpy()
    dists = [np.hypot(bx - fx, by - fy) for _, (fx, fy) in positions.items()]
    min_dist = np.minimum.reduce(dists)
    p65, p90 = np.percentile(min_dist, 65), np.percentile(min_dist, 90)

    df = df.copy()
    df["outcome"] = np.where(
        min_dist <= p65, "OUT",
        np.where(min_dist <= p90, "SINGLE", "DOUBLE"),
    )
    return df


# ═════════════════════════════════════════════════════════
#  VISUALIZATION — Drawn field (fallback)
# ═════════════════════════════════════════════════════════

def _find_outcome_col(df: pd.DataFrame) -> str:
    """Find or create an outcome column in df."""
    for c in df.columns:
        if c.lower() in ("result", "outcome", "event"):
            return c
    rng = np.random.default_rng(0)
    df["outcome"] = rng.choice(
        ["1B", "2B", "3B", "OUT"], size=len(df),
        p=[0.55, 0.25, 0.03, 0.17],
    )
    return "outcome"


# Measured purity of a (player, pitcher-hand) panel is either >=0.93 or <=0.65 —
# nothing lands in between — so the threshold sits in an empty band.
BATTING_SIDE_AGREEMENT = 0.90


def _batting_side_note(df: pd.DataFrame, batter_label: str) -> str:
    """Chart-title suffix naming the side of the plate the hitter swung from.

    A switch hitter bats left against a RHP and right against a LHP, so the
    vs-RHP and vs-LHP charts already are the two permutations — only the
    disclosure was missing. The side is read from the batter_side column of the
    balls actually drawn rather than from the roster handedness, because an "S"
    tag routinely goes stale when a hitter converts mid-season. A genuinely
    two-sided sample lists the side he is using now first, so the sheet is not
    read off a retired swing.
    """
    if not batter_label.rstrip().endswith("(S)"):
        return ""
    if df.empty or "batter_side" not in df.columns:
        return ""
    # Keep the mask index-aligned to df — the date lookup below needs it.
    sides = df["batter_side"].map(normalize_hand)
    sides = sides[sides.isin(("L", "R"))]
    if sides.empty:
        return ""

    counts = sides.value_counts()
    if counts.iloc[0] / len(sides) >= BATTING_SIDE_AGREEMENT:
        return f" — batting {'LH' if counts.index[0] == 'L' else 'RH'}"

    # Genuinely two-sided sample — lead with the most recently used side.
    order = list(counts.index)
    if "date" in df.columns:
        # The feed leaves date unset on some rows, so drop the nulls before
        # comparing — max() on a mixed str/NaN object column raises TypeError.
        last = {s: df.loc[sides.index[sides == s], "date"].dropna().max()
                for s in order}
        dated = [s for s in order if pd.notna(last[s])]
        dated.sort(key=lambda s: last[s], reverse=True)
        order = dated + [s for s in order if pd.isna(last[s])]
    return " — batting " + " / ".join(
        f"{'LH' if s == 'L' else 'RH'} ({counts[s]})" for s in order)


def make_plot(
    df: pd.DataFrame,
    positions: Optional[Dict[str, Tuple[float, float]]],
    batter_label: str,
    pitcher_hand: str,
    sample_note: str = "",
) -> str:
    """Draw a synthetic field and overlay spray data. Returns base64 PNG.

    ``sample_note`` is the whole-sample disclosure and is drawn as a footnote —
    never appended to ``batter_label``, which the batting-side note reads.
    """
    outcome_col = _find_outcome_col(df)
    spray_colors = df[outcome_col].map(
        lambda v: OUTCOME_COLORS.get(str(v).upper(), "white"))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#144d14")

    home = (150, 200)
    left_line, right_line = (60, 250), (240, 250)

    # Fence arc
    fence_r = 180
    ax.add_patch(Arc(home, fence_r*2, fence_r*2,
                     theta1=22, theta2=158,
                     edgecolor="white", linewidth=2, zorder=1))

    # Outfield grass
    pts = [(home[0] + fence_r*np.cos(np.radians(a)),
            home[1] + fence_r*np.sin(np.radians(a)))
           for a in np.linspace(22, 158, 30)]
    ax.add_patch(Polygon([left_line] + pts + [right_line],
                         closed=True, facecolor="#1c6b1c",
                         edgecolor="none", zorder=0))

    # Infield dirt
    ax.add_patch(Arc(home, 190, 190, theta1=22, theta2=158,
                     edgecolor="#c49a6c", linewidth=25, zorder=2))

    # Diamond
    ax.add_patch(Polygon([(150,200),(170,220),(150,240),(130,220)],
                         closed=True, facecolor="#c49a6c",
                         edgecolor="white", linewidth=2, zorder=3))

    # Baselines + centerline
    for end in (left_line, right_line):
        ax.plot([home[0], end[0]], [home[1], end[1]],
                color="white", linewidth=2, zorder=4)
    ax.plot([150, 150], [250, 380], color="white",
            linestyle="--", linewidth=1.2, alpha=0.6, zorder=4)

    # Spray dots
    ax.scatter(df["x"], df["y"], c=spray_colors, s=30,
               alpha=0.8, edgecolor="none", zorder=5)

    # Fielder markers
    box = 12
    for name, (cx, cy) in (positions or {}).items():
        ax.add_patch(Rectangle((cx - box/2, cy - box/2), box, box,
                               linewidth=2, edgecolor="red",
                               facecolor="none", zorder=7))
        ax.scatter(cx, cy, c="red", s=70, zorder=8)
        ax.text(cx, cy + box + 3, name, color="red",
                fontsize=10, ha="center", va="bottom",
                weight="bold", zorder=9)

    ax.set_xlim(40, 260)
    ax.set_ylim(200, 420)
    ax.axis("off")

    # Sample-completeness footnote, bottom-left (light on the drawn grass).
    if sample_note:
        ax.text(42, 202, sample_note, fontsize=7, color="#dddddd",
                ha="left", va="bottom", zorder=9)

    ax.set_title(f"{batter_label} vs {pitcher_hand}" +
                 _batting_side_note(df, batter_label),
                 color="white", fontsize=16, pad=12)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ═════════════════════════════════════════════════════════
#  VISUALIZATION — Background image (production)
# ═════════════════════════════════════════════════════════

def make_plot_with_image(
    df: pd.DataFrame,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    batter_label: str = "Test Player",
    pitcher_hand: str = "RHP",
    background_image_path: str = DEFAULT_BACKGROUND,
    sample_note: str = "",
) -> str:
    """
    Render spray chart over a real ballpark photo.
    Falls back to make_plot() if image/config unavailable.

    ``sample_note`` states whether the chart stands on this hitter's whole
    tracked sample; it is drawn as a footnote and is deliberately kept out of
    ``batter_label``, which _batting_side_note pattern-matches on.
    """
    from PIL import Image

    # Outcome colours
    outcome_col = _find_outcome_col(df)
    spray_colors = df[outcome_col].map(
        lambda v: OUTCOME_COLORS.get(str(v).upper(), "#ffffff"))

    # Load background
    try:
        img = Image.open(background_image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)
    except Exception as e:
        log.warning("Background image failed (%s) — using drawn field.", e)
        return make_plot(df, positions, batter_label, pitcher_hand, sample_note)

    # Load outfield region manager
    outfield_manager = None
    try:
        from outfield_region import OutfieldRegionManager
        config_path = "outfield_region_config.json"
        if Path(config_path).exists():
            outfield_manager = OutfieldRegionManager(config_path)
    except Exception as e:
        log.warning("OutfieldRegionManager load failed: %s", e)

    if not outfield_manager:
        return make_plot(df, positions, batter_label, pitcher_hand, sample_note)

    img_h, img_w = img_array.shape[:2]
    cfg = SPRAY_PIXEL_CONFIG

    # Figure setup
    dpi = 150
    fig, ax = plt.subplots(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    ax.imshow(img_array, origin="upper", zorder=0)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)

    # ── Map each ball to pixel coordinates via direction + distance ──
    # Store as 4-tuple: (pixel_x, pixel_y, color, distance_ft)
    # so depth info is available for centroid weighting and outcome caps.
    balls_pixel = []
    for idx, row in df.iterrows():
        if pd.isna(row["x"]) or pd.isna(row["y"]):
            continue
        dir_val, dist_val = row.get("direction"), row.get("distance")
        if dir_val is None or dist_val is None or pd.isna(dir_val) or pd.isna(dist_val):
            continue

        dist_f = float(dist_val)

        depth_frac = np.clip(
            (dist_f - cfg["dist_min"]) / (cfg["dist_max"] - cfg["dist_min"]),
            0.0, 1.0)
        pixel_y = int(cfg["outfield_bottom_px"]
                      - depth_frac * (cfg["outfield_bottom_px"] - cfg["outfield_top_px"]))

        dir_frac = np.clip(
            (float(dir_val) - cfg["dir_min"]) / (cfg["dir_max"] - cfg["dir_min"]),
            0.0, 1.0)

        x_left = (cfg["home_x_px"]
                  + (pixel_y - cfg["home_y_px"])
                  * (cfg["lf_pole_x_px"] - cfg["home_x_px"])
                  / (cfg["lf_pole_y_px"] - cfg["home_y_px"]))
        x_right = (cfg["home_x_px"]
                   + (pixel_y - cfg["home_y_px"])
                   * (cfg["rf_pole_x_px"] - cfg["home_x_px"])
                   / (cfg["rf_pole_y_px"] - cfg["home_y_px"]))

        pixel_x = int(x_left + dir_frac * (x_right - x_left))
        pixel_x = max(0, min(img_w - 1, pixel_x))
        pixel_y = max(0, min(img_h - 1, pixel_y))

        if pixel_y < cfg["outfield_top_px"] or pixel_y > cfg["outfield_bottom_px"]:
            continue

        color = spray_colors.iloc[idx] if idx < len(spray_colors) else "#ffffff"
        balls_pixel.append((pixel_x, pixel_y, color, dist_f))

    # ── Fielder positioning: weighted X + depth percentile Y ──
    # Lateral (X): depth-weighted centroid. Depth (Y): the
    # DEPTH_POSITION_PERCENTILE-th percentile of ball pixel_y in the zone
    # (lower pixel_y = closer to the fence). The raw centroid is then sanity-
    # clamped into coach alignment windows before anything is drawn, so the
    # outcome recolour below scores dots against the *displayed* fielders.
    optimized_pixel = compute_raw_positions(balls_pixel)
    optimized_pixel, clamp_report = compute_constrained_positions(optimized_pixel)

    # ── Depth-aware outcome reassignment ─────────────────
    # Shallow balls (< shallow_cutoff) are capped at SINGLE.
    # Deep balls get full OUT / SINGLE / DOUBLE range.
    dcfg = DEPTH_WEIGHT_CONFIG
    if balls_pixel and optimized_pixel:
        fp = list(optimized_pixel.values())
        min_dists = np.array([
            min(np.hypot(px - fx, py - fy) for fx, fy in fp)
            for px, py, _, _ in balls_pixel
        ])
        p65 = np.percentile(min_dists, 65)
        p90 = np.percentile(min_dists, 90)
        oc = {"OUT": OUTCOME_COLORS["OUT"],
              "SINGLE": OUTCOME_COLORS["SINGLE"],
              "DOUBLE": OUTCOME_COLORS["DOUBLE"]}

        new_balls = []
        for i, (px, py, _, dist) in enumerate(balls_pixel):
            # Base outcome from fielder proximity
            if min_dists[i] <= p65:
                base = "OUT"
            elif min_dists[i] <= p90:
                base = "SINGLE"
            else:
                base = "DOUBLE"

            # Depth cap: shallow balls can never be doubles
            if dist < dcfg["shallow_cutoff"] and base == "DOUBLE":
                outcome = "SINGLE"
            else:
                outcome = base

            new_balls.append((px, py, oc[outcome]))

        balls_pixel = new_balls

    # Draw spray dots
    for entry in balls_pixel:
        px, py, color = entry[0], entry[1], entry[2]
        ax.scatter(px, py, s=40, c=color, alpha=0.7,
                   edgecolor="white", linewidth=0.5, zorder=5)

    # Draw fielder markers
    bw = 8
    for name, (px, py) in optimized_pixel.items():
        ax.add_patch(Rectangle((px - bw/2, py - bw/2), bw, bw,
                               linewidth=2, edgecolor="red",
                               facecolor="yellow", alpha=0.7, zorder=7))
        ax.scatter(px, py, c="red", s=60, edgecolor="white",
                   linewidth=0.8, zorder=8)

    # Footnote when any fielder was nudged onto a standard alignment.
    if any(r.get("engaged") for r in clamp_report.values()):
        ax.text(img_w - 24, img_h - 20,
                "* fielder positions adjusted to standard alignment",
                fontsize=7, color="#555555", ha="right", va="bottom", zorder=9)

    # Mirrored on the left: whether this is the hitter's whole tracked sample.
    if sample_note:
        ax.text(24, img_h - 20, sample_note,
                fontsize=7, color="#555555", ha="left", va="bottom", zorder=9)

    # Legend
    ax.legend(handles=[
        Patch(facecolor=OUTCOME_COLORS["OUT"], label="OUT"),
        Patch(facecolor=OUTCOME_COLORS["SINGLE"], label="SINGLE"),
        Patch(facecolor=OUTCOME_COLORS["DOUBLE"], label="DOUBLE"),
    ], loc="upper right", framealpha=0.9, fontsize=10)

    ax.axis("off")
    ax.set_title(f"{batter_label} vs {pitcher_hand}" +
                 _batting_side_note(df, batter_label),
                 color="black", fontsize=16, pad=12, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, facecolor="white",
                edgecolor="none", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ═════════════════════════════════════════════════════════
#  DATA LOADING — unified per-pitcher-hand helper
# ═════════════════════════════════════════════════════════

# Fields that pin one pitch. Used only to drop the same batted ball when it
# reaches us under two player_ids — the feed re-ingested one 2024-07-05 game
# under two game_ids, so a pooled chart would otherwise draw those balls twice.
_SPRAY_SIGNATURE_FIELDS = (
    "date", "inning", "top_or_bottom", "pa_of_inning", "pitch_of_pa",
    "pitcher_id", "exit_speed", "angle", "direction", "distance",
)


def _spray_signature(row: Dict) -> Tuple:
    return tuple(row.get(f) for f in _SPRAY_SIGNATURE_FIELDS)


def _fetch_union_spray(batter_id: str,
                       pitcher_hand: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       limit: int = 1000) -> Tuple[List[Dict], List[str]]:
    """
    Fetch spray data for every roster record proven to be this hitter.

    Mirrors fetch_player_spray's signature and degrades to exactly one call for
    a batter that was never pooled (which is every batter until the identity
    pass has run), so nothing changes for the 278 unmerged entries.

    Rows from the first id are kept verbatim — including the duplicate rows the
    feed already carries within a single id, so an unmerged chart never shifts —
    and each later id contributes only pitches not already drawn.

    Returns ``(rows, contributed_ids)``; an id missing from ``contributed_ids``
    returned nothing, so the caller can disclose a partial pool rather than
    quietly showing one again.
    """
    ids = _union_ids.get(str(batter_id), [str(batter_id)])

    rows: List[Dict] = []
    contributed: List[str] = []
    seen = set()

    for pid in ids:
        try:
            part = fetch_player_spray(
                player_id=pid, pitcher_hand=pitcher_hand,
                start_date=start_date, end_date=end_date, limit=limit)
        except Exception:
            log.exception("Pooled spray fetch failed for %s", pid)
            continue

        if not part:
            log.warning("Pooled record %s returned no rows for %s", pid, batter_id)
            continue

        fresh = [r for r in part if _spray_signature(r) not in seen]
        seen.update(_spray_signature(r) for r in fresh)
        rows.extend(fresh)
        contributed.append(pid)

    if len(ids) > 1:
        log.info("Pooled spray for %s: %d rows from %d/%d records",
                 batter_id, len(rows), len(contributed), len(ids))
    return rows, contributed


def _render_sample_note(batter_id: str, contributed: List[str]) -> str:
    """Disclosure for one rendered chart: what the entry claims, plus whatever
    the pool actually failed to load on this render."""
    note = _sample_notes.get(str(batter_id), "")
    ids = _union_ids.get(str(batter_id), [str(batter_id)])
    missing = [i for i in ids if i not in contributed]
    if missing:
        short = f"{len(missing)} pooled record(s) could not be loaded"
        note = f"{note} — {short}" if note else f"partial sample — {short}"
    return note


def load_spray_and_render(
    batter_id: str,
    pitcher_hand_label: str,
    client_batter_name: Optional[str] = None,
    client_batter_hand: Optional[str] = None,
    background_image_path: str = DEFAULT_BACKGROUND,
) -> Tuple[Optional[str], str]:
    """
    Load spray data for one batter + pitcher hand and render a chart.

    Returns:
        (base64_png, batter_label)  on success
        (None, error_message)       on failure
    """
    sample_note = ""

    # ── API adapter mode ──
    if USE_API_ADAPTER and not USE_JSON_LOADER:
        pitcher_letter = pitcher_hand_label.replace("HP", "").upper()
        spray_data, contributed = _fetch_union_spray(
            batter_id, pitcher_hand=pitcher_letter,
            start_date=None, end_date=None, limit=1000)
        sample_note = _render_sample_note(batter_id, contributed)

        if not spray_data:
            return None, f"No spray data for {pitcher_hand_label}"

        from data_loader import parse_spray_to_dataframe
        df = parse_spray_to_dataframe(spray_data)
        if df.empty:
            return None, f"No qualifying outfield data for {pitcher_hand_label}"

        df = df.dropna(subset=["x", "y"])
        if len(df) < MIN_QUALIFYING_BALLS:
            return None, (f"Only {len(df)} balls for {pitcher_hand_label} "
                          f"(need {MIN_QUALIFYING_BALLS})")

        df = df.copy()
        df["hang_time"] = df["hang_time"].fillna(3.0)
        df["outcome"] = df["outcome"].fillna("OUT")

        meta = resolve_batter_meta(batter_id, client_batter_name, client_batter_hand)
        batter_label = meta["label"]

    # ── JSON loader mode ──
    elif USE_JSON_LOADER:
        players_with_data = get_unique_players_with_spray_data()
        player_ids = {p.get("player_id") for p in players_with_data}

        if batter_id in player_ids:
            selected = next(p for p in players_with_data
                            if p.get("player_id") == batter_id)
            name = selected.get("player_name", "Unknown")
            hand = normalize_hand(selected.get("player_batting_handedness") or "")
            batter_label = f"{name} ({hand})"

            df = get_player_spray_dataframe(batter_id)
            if df.empty or df.dropna(subset=["x", "y"]).shape[0] < MIN_QUALIFYING_BALLS:
                df, _ = _prepare_synthetic("dickerson_R", pitcher_hand_label)
            else:
                df = df.dropna(subset=["x", "y"])
                df["hang_time"] = df["hang_time"].fillna(3.0)
                df["outcome"] = df["outcome"].fillna("OUT")

        elif batter_id in BATTERS:
            batter_label = BATTERS[batter_id]["label"]
            df, _ = _prepare_synthetic(batter_id, pitcher_hand_label)
        else:
            return None, "Unknown batter"

    # ── Synthetic fallback ──
    else:
        if batter_id not in BATTERS:
            return None, "Unknown batter"
        batter_label = BATTERS[batter_id]["label"]
        df, _ = _prepare_synthetic(batter_id, pitcher_hand_label)

    # Render chart
    img_b64 = make_plot_with_image(
        df, positions=None, batter_label=batter_label,
        pitcher_hand=pitcher_hand_label,
        background_image_path=background_image_path,
        sample_note=sample_note)

    return img_b64, batter_label


# ═════════════════════════════════════════════════════════
#  BACKGROUND PLAYER PROBE CACHE
# ═════════════════════════════════════════════════════════

_players_with_data_cache: dict = {}
_cache_ready: bool = False

# Pitch-feed fingerprints keyed by player_id, and what the merge concluded from
# them. _union_ids maps EVERY pid in a pooled group — primary and folded — onto
# the same ordered id list, so a stale client holding a folded pid still renders
# the whole hitter instead of half of one. All three are published in one swap
# at the end of the identity pass and are empty until then, so the dropdown can
# never conclude a pool the render path cannot yet fetch; every path degrades to
# its pre-merge behaviour in the meantime.
_identity_cache: dict = {}
_union_ids: Dict[str, List[str]] = {}
_sample_notes: Dict[str, str] = {}
# False until the identity pass has finished (or failed) — the dropdown is only
# final once this is set, so the frontend must not stop polling on _cache_ready.
_merge_ready: bool = False


def _probe_one_player(player_id: str) -> bool:
    try:
        from adapter import probe_player_has_data
        return probe_player_has_data(player_id)
    except Exception:
        return True


def _start_background_probe(players: list) -> None:
    """Probe players for qualifying data in a background thread."""
    batch = players[:5000]

    def run():
        global _players_with_data_cache, _cache_ready
        import time

        def probe(player):
            pid = player.get("player_id")
            if not pid:
                return pid, False
            time.sleep(0.5)
            return pid, _probe_one_player(pid)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed(
                {pool.submit(probe, p): p for p in batch}
            ):
                pid, result = future.result()
                if pid:
                    _players_with_data_cache[pid] = result

        _cache_ready = True
        found = sum(1 for v in _players_with_data_cache.values() if v)
        log.info("Probe complete: %d/%d players confirmed", found, len(batch))

        # Only now, so the dropdown never re-groups mid-warm-up and yanks the
        # user's selection out from under him.
        global _merge_ready
        try:
            _resolve_identities(batch)
        except Exception:
            log.exception("Identity pass failed — records stay separate")
        finally:
            # Either way the list is as final as it is going to get, so the
            # frontend stops polling instead of waiting on a pass that died.
            _merge_ready = True

    threading.Thread(target=run, daemon=True).start()
    log.info("Background probe started for %d players", len(batch))


def _resolve_identities(players: list) -> None:
    """Fingerprint the duplicate-name players so the merge has evidence.

    Scoped to names that survive the qualifying probe on more than one record —
    measured, 156 of 2072 players — because those are the only records the merge
    can ever pool, and the scoping is what keeps this at one extra upstream call
    per candidate instead of one per player. Paced like the probe it follows.

    Fingerprints accumulate in a local dict and reach _identity_cache only once
    the whole pass has run: the dropdown merges off _identity_cache while the
    render pools off _union_ids, so publishing a fingerprint early would let the
    list fold two records into one entry that still charts a single record, with
    no disclosure and the sibling entry gone — the original half-sample bug with
    the other half made unreachable. A pass that dies partway therefore leaves
    every record separate, which is what the caller already reports.
    """
    import time

    qualifying = _filter_by_qualifying_cache(players)
    seen_names: Dict[str, int] = {}
    for p in qualifying:
        name = (p.get("player_name") or "").strip().casefold()
        if name:
            seen_names[name] = seen_names.get(name, 0) + 1

    targets = [
        p for p in qualifying
        if seen_names.get((p.get("player_name") or "").strip().casefold(), 0) > 1
    ]
    log.info("Identity pass: fingerprinting %d duplicate-name records",
             len(targets))

    fingerprints: Dict[str, Dict] = {}
    for p in targets:
        pid = p.get("player_id")
        if not pid:
            continue
        time.sleep(0.5)
        try:
            fingerprint = probe_player_identity(pid)
        except Exception:
            log.exception("Identity probe failed for %s", pid)
            continue
        if fingerprint:
            fingerprints[str(pid)] = fingerprint

    _publish_identities(fingerprints,
                        build_player_dict(qualifying, identities=fingerprints))


# ═════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════

def _filter_by_qualifying_cache(players: list) -> list:
    """
    Drop players the probe has confirmed as having <MIN_QUALIFYING_BALLS
    vs either pitcher hand.

    While the probe is still running (cache not ready), unprobed players
    pass through so the dropdown is populated immediately and narrows
    progressively. Once the probe finishes, only confirmed-qualifying
    players remain.
    """
    if _cache_ready:
        # Probe done — strict filter: must be explicitly True in cache
        return [
            p for p in players
            if _players_with_data_cache.get(p.get("player_id")) is True
        ]
    # Probe still running — let unprobed players through for now
    return [
        p for p in players
        if _players_with_data_cache.get(p.get("player_id"), True)
    ]


def _publish_unions(batters: dict) -> None:
    """Publish what the merge concluded, for the render path to read.

    Both maps are rebuilt whole and swapped in, so a render either sees the
    complete previous conclusion or the complete new one, never a half-written
    pool.
    """
    global _union_ids, _sample_notes

    unions: Dict[str, List[str]] = {}
    notes: Dict[str, str] = {}
    for pid, entry in batters.items():
        ids = list(entry.get("merged_ids") or [str(pid)])
        note = entry.get("sample_note") or ""
        for member in ids:
            unions[str(member)] = ids
            if note:
                notes[str(member)] = note

    _union_ids, _sample_notes = unions, notes
    pooled = sum(1 for e in batters.values() if len(e.get("merged_ids") or []) > 1)
    log.info("Identity pass complete: %d entries, %d pooled from >1 record",
             len(batters), pooled)


def _publish_identities(fingerprints: Dict[str, Dict], batters: dict) -> None:
    """Hand the finished identity pass to the dropdown and the render path.

    The pools go out first and the fingerprints second, because the two are read
    by different code: build_player_dict folds records using _identity_cache
    while _fetch_union_spray pools ids using _union_ids. Publishing the pools
    first means the only window that exists is the harmless one — an id already
    resolves to its whole pool before any entry has folded — never the one where
    the list has folded a record the chart cannot reach.
    """
    global _identity_cache

    _publish_unions(batters)
    _identity_cache = fingerprints


@app.route("/")
def index():
    """Render the main page with the player dropdown."""
    if USE_JSON_LOADER:
        try:
            batters = build_player_dict(get_unique_players_with_spray_data())
            if batters:
                return render_template("index.html", batters=batters)
        except Exception:
            log.exception("JSON loader failed")

    elif USE_API_ADAPTER:
        try:
            players = fetch_players(limit=5000)
            if players:
                if not _cache_ready and not _players_with_data_cache:
                    _start_background_probe(players)
                # Filter out players known to have <15 balls
                batters = build_player_dict(
                    _filter_by_qualifying_cache(players), identities=_identity_cache)
                if batters:
                    return render_template("index.html", batters=batters)
        except Exception:
            log.exception("API player fetch failed")

    return render_template("index.html", batters=BATTERS)


@app.route("/api/batters")
def api_batters():
    """Return the current batter list as JSON (used by React frontend)."""
    if USE_JSON_LOADER:
        try:
            batters = build_player_dict(get_unique_players_with_spray_data())
            if batters:
                return jsonify({"ok": True, "batters": batters})
        except Exception:
            pass
    elif USE_API_ADAPTER:
        try:
            players = fetch_players(limit=5000)
            if players:
                # Kick off the probe on first call if not yet started
                if not _cache_ready and not _players_with_data_cache:
                    _start_background_probe(players)
                batters = build_player_dict(
                    _filter_by_qualifying_cache(players), identities=_identity_cache)
                if batters:
                    return jsonify({"ok": True, "batters": batters})
        except Exception:
            pass
    return jsonify({"ok": True, "batters": BATTERS})


@app.route("/api/cache-status")
def api_cache_status():
    """Return background probe progress for the frontend to poll."""
    try:
        players = fetch_players(limit=5000)
    except Exception:
        return jsonify({"ready": False, "batters": {}, "probed": 0, "total": 0})

    confirmed = [
        p for p in players
        if p.get("player_id") and _players_with_data_cache.get(p["player_id"], False)
    ]
    batters = build_player_dict(confirmed, identities=_identity_cache)

    return jsonify({
        "ready": _cache_ready,
        # The probe finishes before the identity pass, so "ready" alone still
        # describes a list that is about to fold its duplicate records. A client
        # that stops polling here keeps both halves of a pooled hitter on screen,
        # rendering the same union twice under two teams.
        "merged": _merge_ready,
        "batters": batters,
        "probed": len(_players_with_data_cache),
        "total": len(players),
    })


@app.route("/api/compute", methods=["POST"])
def api_compute():
    """Main computation endpoint: load data → optimize → render chart."""
    try:
        payload = request.get_json(force=True)
        batter_id = payload.get("batter_id")
        pitcher_hand = payload.get("pitcher_hand", "RHP")
        bg_path = payload.get("background_image_path", DEFAULT_BACKGROUND)
        client_name = payload.get("batter_name")
        client_hand = payload.get("batter_hand")

        sample_note = ""

        # ── API adapter mode ──
        if USE_API_ADAPTER and not USE_JSON_LOADER:
            pitcher_letter = pitcher_hand.replace("HP", "").upper() if pitcher_hand else "R"
            spray_data, contributed = _fetch_union_spray(
                batter_id, pitcher_hand=pitcher_letter,
                start_date=None, end_date=None, limit=1000)
            sample_note = _render_sample_note(batter_id, contributed)

            if not spray_data:
                return jsonify({"ok": False, "error": "No spray data available."}), 404

            from data_loader import parse_spray_to_dataframe
            df = parse_spray_to_dataframe(spray_data)
            if df.empty:
                return jsonify({"ok": False, "error": "No qualifying outfield balls found."}), 404

            df = df.dropna(subset=["x", "y"])
            if len(df) < MIN_QUALIFYING_BALLS:
                return jsonify({
                    "ok": False,
                    "error": f"Only {len(df)} balls (need {MIN_QUALIFYING_BALLS})."
                }), 404

            df = df.copy()
            df["hang_time"] = df["hang_time"].fillna(3.0)
            df["outcome"] = df["outcome"].fillna("OUT")
            meta = resolve_batter_meta(batter_id, client_name, client_hand)
            positions_drawn = None

        # ── JSON loader mode ──
        elif USE_JSON_LOADER:
            player_ids = {
                p.get("player_id")
                for p in get_unique_players_with_spray_data()
            }

            if batter_id in player_ids:
                selected = next(p for p in get_unique_players_with_spray_data()
                                if p.get("player_id") == batter_id)
                name = selected.get("player_name", "Unknown")
                hand = normalize_hand(selected.get("player_batting_handedness") or "")
                meta = {"label": f"{name} ({hand})",
                        "batter_name": name, "batter_hand": hand,
                        "player_id": batter_id}

                df = get_player_spray_dataframe(batter_id)
                if df.empty or df.dropna(subset=["x", "y"]).shape[0] < MIN_QUALIFYING_BALLS:
                    df, positions_drawn = _prepare_synthetic("dickerson_R", pitcher_hand)
                else:
                    df = df.dropna(subset=["x", "y"])
                    df["hang_time"] = df["hang_time"].fillna(3.0)
                    df["outcome"] = df["outcome"].fillna("OUT")
                    positions_drawn = None

            elif batter_id in BATTERS:
                meta = BATTERS[batter_id]
                df, positions_drawn = _prepare_synthetic(batter_id, pitcher_hand)
            else:
                return jsonify({"ok": False, "error": "Unknown batter"}), 400

        # ── Synthetic fallback ──
        else:
            if batter_id not in BATTERS:
                return jsonify({"ok": False, "error": "Unknown batter"}), 400
            meta = BATTERS[batter_id]
            df, positions_drawn = _prepare_synthetic(batter_id, pitcher_hand)

        # Save CSV
        if positions_drawn:
            pd.DataFrame.from_dict(
                positions_drawn, orient="index", columns=["X", "Y"]
            ).to_csv(LAST_CSV_PATH)

        # Render
        img_b64 = make_plot_with_image(
            df, positions=None, batter_label=meta["label"],
            pitcher_hand=pitcher_hand, background_image_path=bg_path,
            sample_note=sample_note)

        return jsonify({
            "ok": True,
            "batter_id": batter_id,
            "batter_label": meta["label"],
            "batter_hand": meta["batter_hand"],
            "pitcher_hand": pitcher_hand,
            "positions": positions_drawn or {},
            "image_base64": img_b64,
            "sample_note": sample_note,
            "download_url": "/download",
        })

    except Exception as e:
        log.exception("api_compute failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/download")
def download():
    """Serve the last optimized-positions CSV."""
    if not pd.io.common.file_exists(LAST_CSV_PATH):
        return "Run an optimization first.", 404
    return send_file(LAST_CSV_PATH, as_attachment=True)


# ═════════════════════════════════════════════════════════
#  SLUGGER API PASS-THROUGH ENDPOINTS
# ═════════════════════════════════════════════════════════

def _require_api_adapter():
    if not USE_API_ADAPTER:
        return jsonify({"success": False, "error": "API adapter not available"}), 503
    return None


@app.route("/api/ballparks", methods=["GET"])
def api_ballparks():
    err = _require_api_adapter()
    if err:
        return err
    try:
        data = fetch_ballparks(
            ballpark_name=request.args.get("ballpark_name"),
            city=request.args.get("city"),
            state=request.args.get("state"),
            limit=int(request.args.get("limit", 50)),
            page=int(request.args.get("page", 1)),
            order=request.args.get("order", "ASC"))
        return jsonify({"success": True, "data": data, "count": len(data)})
    except Exception as e:
        log.exception("api_ballparks failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/games", methods=["GET"])
def api_games():
    err = _require_api_adapter()
    if err:
        return err
    try:
        data = fetch_games(
            ballpark_name=request.args.get("ballpark_name"),
            team_name=request.args.get("team_name"),
            start_date=request.args.get("start_date"),
            end_date=request.args.get("end_date"),
            limit=int(request.args.get("limit", 50)),
            page=int(request.args.get("page", 1)),
            order=request.args.get("order", "DESC"))
        return jsonify({"success": True, "data": data, "count": len(data)})
    except Exception as e:
        log.exception("api_games failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/players/<player_id>/spray", methods=["GET"])
def api_player_spray(player_id: str):
    err = _require_api_adapter()
    if err:
        return err
    try:
        data = fetch_player_spray(
            player_id=player_id,
            pitcher_hand=request.args.get("pitcher_hand"),
            start_date=request.args.get("start_date"),
            end_date=request.args.get("end_date"),
            limit=int(request.args.get("limit", 5000)))
        return jsonify({"success": True, "data": data, "count": len(data)})
    except Exception as e:
        log.exception("api_player_spray failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/optimize/<player_id>", methods=["GET", "POST"])
def api_optimize_and_visualize(player_id: str):
    """Full optimization pipeline via the API adapter."""
    err = _require_api_adapter()
    if err:
        return err
    try:
        if request.method == "POST":
            payload = request.get_json(force=True) or {}
            pitcher_hand = payload.get("pitcher_hand", "R")
            start_date = payload.get("start_date")
            end_date = payload.get("end_date")
            bg_path = payload.get("background_image_path", DEFAULT_BACKGROUND)
        else:
            pitcher_hand = request.args.get("pitcher_hand", "R")
            start_date = request.args.get("start_date")
            end_date = request.args.get("end_date")
            bg_path = request.args.get("background_image_path", DEFAULT_BACKGROUND)

        spray_data, contributed = _fetch_union_spray(
            player_id, pitcher_hand=pitcher_hand,
            start_date=start_date, end_date=end_date, limit=1000)
        if not spray_data:
            return jsonify({"success": False, "error": "No spray data found"}), 404

        from data_loader import parse_spray_to_dataframe
        df = parse_spray_to_dataframe(spray_data)
        if df.empty:
            return jsonify({"success": False, "error": "Failed to parse spray data"}), 400

        df = df.dropna(subset=["x", "y", "hang_time"])
        if len(df) < MIN_QUALIFYING_BALLS:
            return jsonify({
                "success": False,
                "error": f"Insufficient data: {len(df)} rows (need {MIN_QUALIFYING_BALLS})"
            }), 400

        from mlb_to_logical_converter import convert_dataframe_mlb_to_logical
        from excel_grid_to_logical_converter import convert_optimizer_positions_to_logical
        from outfield_region import OutfieldRegionManager

        df_logical = convert_dataframe_mlb_to_logical(df, mlb_x_col="x", mlb_y_col="y")
        positions_logical = convert_optimizer_positions_to_logical(
            optimize_outfield_excel(df_logical))

        label = f"Player {player_id[:8]}"
        img_b64 = make_plot_with_image(
            df, positions=positions_logical, batter_label=label,
            pitcher_hand="RHP" if pitcher_hand.upper() == "R" else "LHP",
            background_image_path=bg_path,
            sample_note=_render_sample_note(player_id, contributed))

        mgr = OutfieldRegionManager("outfield_region_config.json")
        positions_pixel = {
            n: (float(mgr.logical_to_pixel((lx, ly))[0]),
                float(mgr.logical_to_pixel((lx, ly))[1]))
            for n, (lx, ly) in positions_logical.items()
        }

        return jsonify({
            "success": True,
            "image_base64": img_b64,
            "positions": positions_pixel,
            "positions_logical": {k: (float(v[0]), float(v[1]))
                                  for k, v in positions_logical.items()},
            "data_count": len(df),
            "batter_label": label,
            "player_id": player_id,
            "pitcher_hand": pitcher_hand,
        })
    except Exception as e:
        log.exception("api_optimize_and_visualize failed")
        return jsonify({"success": False, "error": str(e)}), 500


# ═════════════════════════════════════════════════════════
#  PDF SCOUTING REPORT
# ═════════════════════════════════════════════════════════

def _draw_player_report_page(c, page_w, page_h, batter_id,
                             client_name=None, client_hand=None,
                             img_format="png"):
    """Draw one player's vs-RHP/vs-LHP report onto canvas ``c``.

    Renders the vs RHP (top) and vs LHP (bottom) spray charts stacked on the
    current page. Does NOT call ``c.showPage()`` or ``c.save()`` so callers can
    append this page to a multi-page document (the client-side team merge builds
    a document one player-page at a time).

    ``img_format`` is "png" (default, embedded losslessly) or "jpeg" (re-encoded
    at quality 85 to shrink multi-page team reports).

    Returns ``(drawn, rhp_label_or_err, lhp_label_or_err)``. When neither hand
    has data, ``drawn`` is False and nothing is drawn.
    """
    from reportlab.lib.utils import ImageReader
    from PIL import Image as PILImage

    rhp_img, rhp_label = load_spray_and_render(
        batter_id, "RHP", client_name, client_hand)
    lhp_img, lhp_label = load_spray_and_render(
        batter_id, "LHP", client_name, client_hand)

    if rhp_img is None and lhp_img is None:
        return False, rhp_label, lhp_label

    player_name = rhp_label or lhp_label or "Unknown Player"
    margin = 24

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(page_w / 2, page_h - 32,
                        f"SLUGGER Scouting Report — {player_name}")

    # Chart layout
    header_bottom = page_h - 46
    gap = 8
    chart_w = page_w - 2 * margin
    chart_h = (header_bottom - margin - gap) / 2

    def draw_chart(img_b64, y_bottom, label):
        if img_b64 is None:
            c.setFont("Helvetica", 12)
            c.setFillColorRGB(0.5, 0.5, 0.5)
            c.drawCentredString(page_w / 2, y_bottom + chart_h / 2,
                                f"No data available ({label})")
            c.setFillColorRGB(0, 0, 0)
            return
        img_bytes = base64.b64decode(img_b64)
        pil = PILImage.open(io.BytesIO(img_bytes))
        if img_format == "jpeg":
            rgb = pil.convert("RGB") if pil.mode != "RGB" else pil
            jbuf = io.BytesIO()
            rgb.save(jbuf, format="JPEG", quality=85)
            jbuf.seek(0)
            reader = ImageReader(jbuf)
        else:
            reader = ImageReader(io.BytesIO(img_bytes))
        scale = min(chart_w / pil.width, chart_h / pil.height)
        dw, dh = pil.width * scale, pil.height * scale
        c.drawImage(reader,
                    margin + (chart_w - dw) / 2,
                    y_bottom + (chart_h - dh) / 2,
                    width=dw, height=dh)

    top_y = header_bottom - chart_h
    draw_chart(rhp_img, top_y, "vs RHP")
    draw_chart(lhp_img, top_y - gap - chart_h, "vs LHP")

    return True, rhp_label, lhp_label


@app.route("/api/pdf/<batter_id>", methods=["GET"])
def api_pdf(batter_id: str):
    """
    Generate a portrait 8.5x11 PDF with vs RHP (top) and
    vs LHP (bottom) spray charts stacked vertically.

    ``?img=jpeg`` re-encodes the embedded charts to shrink the file (used by the
    client-side team-report merge); any other value falls back to PNG.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError:
        return jsonify({
            "ok": False,
            "error": "reportlab not installed — run: pip install reportlab"
        }), 500

    try:
        client_name = request.args.get("batter_name")
        client_hand = request.args.get("batter_hand")
        img_format = request.args.get("img", "png").lower()
        if img_format not in ("png", "jpeg"):
            img_format = "png"

        page_w, page_h = letter  # 612 x 792 pt
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf, pagesize=letter)

        drawn, rhp_label, lhp_label = _draw_player_report_page(
            c, page_w, page_h, batter_id,
            client_name, client_hand, img_format)

        if not drawn:
            return jsonify({
                "ok": False,
                "error": f"No data. RHP: {rhp_label}. LHP: {lhp_label}"
            }), 404

        c.save()
        buf.seek(0)

        player_name = rhp_label or lhp_label or "Unknown Player"
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", player_name.split("(")[0].strip())
        return send_file(buf, mimetype="application/pdf",
                         as_attachment=True,
                         download_name=f"SLUGGER_{safe}_report.pdf")

    except Exception as e:
        log.exception("api_pdf failed")
        return jsonify({"ok": False, "error": str(e)}), 500


# ═════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)