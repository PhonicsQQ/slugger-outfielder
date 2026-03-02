# data_loader.py – JSON-file-based data loader (development/testing use only)
#
# Note:
#   This module is intended ONLY for development and testing.
#   For distribution and production use, replace this with adapter.py
#   (which fetches real data from the SLUGGER API).

import json
import logging
import math
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd

log = logging.getLogger(__name__)

# Base data directory
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# -------------------------------------------------------
# Basic load functions
# -------------------------------------------------------

def load_teams() -> List[Dict]:
    """Load team list from local JSON."""
    teams_file = DATA_DIR / "teams" / "teams.json"
    if not teams_file.exists():
        return []
    with open(teams_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("data", [])


def load_players(team_name: Optional[str] = None,
                 batting_handedness: Optional[str] = None) -> List[Dict]:
    """
    Load all players from JSON files.

    Args:
        team_name: Optional team filter
        batting_handedness: Optional filter ("Right", "Left", "Switch")

    Returns:
        List[Dict]: List of player dictionaries
    """
    players = []
    players_dir = DATA_DIR / "players"

    if not players_dir.exists():
        return []

    for json_file in players_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                team_players = data.get("data", [])

                if team_name:
                    team_players = [
                        p for p in team_players
                        if p.get("team_name") == team_name
                    ]

                if batting_handedness:
                    team_players = [
                        p for p in team_players
                        if p.get("player_batting_handedness") == batting_handedness
                    ]

                players.extend(team_players)

        except Exception as e:
            log.error(f"Error loading {json_file}: {e}")
            continue

    return players


def load_spray_data(player_id: str) -> Optional[List[Dict]]:
    """
    Load raw spray JSON for a given player.

    Args:
        player_id: Player UUID

    Returns:
        Optional[List[Dict]]: List of spray entries, or None if not found
    """
    spray_file = DATA_DIR / "spray" / f"{player_id}.json"

    if not spray_file.exists():
        return None

    try:
        with open(spray_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("data", [])
    except Exception as e:
        log.error(f"Error loading spray data for {player_id}: {e}")
        return None


def load_games() -> List[Dict]:
    """Load game list from JSON."""
    games_file = DATA_DIR / "games" / "games.json"
    if not games_file.exists():
        return []
    with open(games_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("data", [])


def load_ballparks() -> List[Dict]:
    """Load ballpark list from JSON."""
    ballparks_file = DATA_DIR / "ballparks" / "ballparks.json"
    if not ballparks_file.exists():
        return []
    with open(ballparks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("data", [])


# -------------------------------------------------------
# Spray data → DataFrame transformation
# -------------------------------------------------------

# Outcome labels included in optimization & visualization.
# TRIPLE and HOMERUN are intentionally excluded from outfield
# positioning analysis (they land in specific non-repositionable zones).
VALID_OUTCOMES = {"OUT", "SINGLE", "DOUBLE"}

# Hit types that are definitely not outfield balls
EXCLUDE_HIT_TYPES = {"groundball", "bunt"}


def _normalize_pitch_call(value: Any) -> str:
    """
    Normalize pitch_call to a canonical lowercase string.

    Handles API variants like "InPlay", "in_play", "In Play", "inplay".
    All of these collapse to "inplay".
    """
    if not value:
        return ""
    return str(value).strip().lower().replace("_", "").replace(" ", "")


def _normalize_hit_type(value: Any) -> str:
    """Normalize auto_hit_type / tagged_hit_type to lowercase stripped string."""
    if not value:
        return ""
    return str(value).strip().lower()


def _safe_float(value) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_landing_from_trajectory(item: Dict) -> Optional[tuple]:
    """
    Compute ball landing position (in feet) from Trackman trajectory
    polynomial coefficients.

    Trackman trajectory model:
        x(t) = xc0 + xc1*t + xc2*t²
        y(t) = yc0 + yc1*t + yc2*t²

    Where:
        xc0/yc0 = initial position at contact (feet from home plate)
        xc1/yc1 = initial velocity (ft/s)
        xc2/yc2 = acceleration/gravity component (ft/s²)
        t = time in seconds (we use hang_time as the evaluation point)

    The raw coefficient values (especially xc2) are on a completely
    different scale from feet. Using them directly as x,y coordinates
    causes all dots to collapse into a tiny cluster. This function
    evaluates the polynomial to produce real feet coordinates.

    Args:
        item: raw spray record dict

    Returns:
        (x_feet, y_feet, z) or None if insufficient data
    """
    xc0 = _safe_float(item.get("hit_trajectory_xc0"))
    xc1 = _safe_float(item.get("hit_trajectory_xc1"))
    xc2 = _safe_float(item.get("hit_trajectory_xc2"))
    yc0 = _safe_float(item.get("hit_trajectory_yc0"))
    yc1 = _safe_float(item.get("hit_trajectory_yc1"))
    yc2 = _safe_float(item.get("hit_trajectory_yc2"))

    # We need all 6 coefficients to compute a meaningful position
    if any(v is None for v in [xc0, xc1, xc2, yc0, yc1, yc2]):
        return None

    # Evaluation time: use hang_time if available, else estimate
    t = _safe_float(item.get("hang_time"))
    if t is None or t <= 0:
        dist = _safe_float(item.get("distance"))
        if dist and dist > 0:
            t = max(2.0, min(6.0, dist / 75.0))
        else:
            t = 4.0

    # Evaluate trajectory polynomial
    x = xc0 + xc1 * t + xc2 * t * t
    y = yc0 + yc1 * t + yc2 * t * t

    # Sanity check: result should be in plausible outfield range
    if abs(x) > 500 or y < -50 or y > 600:
        log.debug(
            f"Trajectory polynomial produced out-of-range position: "
            f"x={x:.1f}, y={y:.1f} (t={t:.1f}s) — skipping"
        )
        return None

    z_val = _safe_float(item.get("hit_trajectory_zc1"))
    return (x, y, z_val)


def parse_spray_to_dataframe(spray_data: List[Dict]) -> pd.DataFrame:
    """
    Convert raw spray JSON records into a clean DataFrame.

    Coordinate extraction priority:
        1. direction + distance    (trigonometric — always correct field-plane x,y)
        2. position_at_110_x + distance  (lateral from trackman, depth from distance)
        3. trajectory polynomial   (evaluate xc0+xc1*t+xc2*t² at hang_time)
        4. SKIP the record         (no usable coordinate source)

    IMPORTANT — WHY position_at_110_y CANNOT BE USED AS DEPTH:
        position_at_110_y is the ball's HEIGHT above the ground at 110 ft
        from home plate (typically 0–50 ft). It is NOT the ball's depth
        from home plate. Using it as y produces dots clustered in a
        horizontal band near the infield. The correct depth coordinate
        comes from the "distance" field.

    IMPORTANT — WHY RAW TRAJECTORY COEFFICIENTS ARE NOT USED DIRECTLY:
        hit_trajectory_xc2/yc2 are polynomial acceleration coefficients
        (typically ±10). They are NOT positions in feet. Using them
        directly collapses all dots into a tiny cluster.

    Filtering (NaN-safe — missing values are NOT treated as failures):
        • pitch_call must normalize to "inplay"  (flexible matching)
        • auto_hit_type / tagged_hit_type must NOT be groundball or bunt
          (if the field is missing/null, we keep the record)
        • launch angle > 10°  (only excluded if angle is explicitly ≤ 10)
        • distance ≥ 150 ft  (only excluded if distance is explicitly < 150)
        • outcome must be OUT, SINGLE, or DOUBLE

    Returns:
        DataFrame with columns: x, y, z, distance, hang_time, outcome,
        batter_id, pitcher_throws, date, angle, direction, pitch_call, etc.
    """
    if not spray_data:
        return pd.DataFrame()

    records = []
    coord_sources = {"dir_dist": 0, "pos110": 0, "trajectory": 0, "skipped": 0}

    for item in spray_data:
        x, y, z = None, None, None

        # -----------------------------------------------------------
        # Priority 1: direction + distance (trigonometric)
        # Always produces correct field-plane coordinates in feet.
        # direction = spray angle in degrees, distance = total distance in ft.
        # x = distance * sin(direction)  → lateral position
        # y = distance * cos(direction)  → depth from home plate
        # -----------------------------------------------------------
        if item.get("direction") is not None and item.get("distance") is not None:
            direction = item.get("direction")
            distance = item.get("distance")
            try:
                direction = float(direction)
                distance = float(distance)
                if distance > 0:
                    rad = math.radians(direction)
                    x = distance * math.sin(rad)
                    y = distance * math.cos(rad)
                    z = item.get("position_at_110_z") or item.get("hit_trajectory_zc1")
                    coord_sources["dir_dist"] += 1
            except (TypeError, ValueError):
                pass

        # -----------------------------------------------------------
        # Priority 2: position_at_110_x (lateral) + distance (depth)
        # position_at_110_x = lateral position at 110 ft from home (feet)
        # position_at_110_y = HEIGHT of ball at 110 ft (NOT depth!)
        # So we use "distance" for depth instead of position_at_110_y.
        # -----------------------------------------------------------
        if x is None and item.get("position_at_110_x") is not None and item.get("distance") is not None:
            try:
                pos_x = float(item["position_at_110_x"])
                dist = float(item["distance"])
                if dist > 0:
                    x = pos_x
                    y = dist  # depth from home plate in feet
                    z = item.get("position_at_110_z")
                    coord_sources["pos110"] += 1
            except (TypeError, ValueError):
                pass

        # -----------------------------------------------------------
        # Priority 3: Evaluate trajectory polynomial at hang_time
        # x(t) = xc0 + xc1*t + xc2*t²  gives actual position in feet
        # DO NOT use raw xc2/yc2 values as coordinates — they are
        # acceleration coefficients, not positions.
        # -----------------------------------------------------------
        if x is None:
            result = _compute_landing_from_trajectory(item)
            if result is not None:
                x, y, z = result
                coord_sources["trajectory"] += 1

        # If we still have no coordinates, skip this record entirely.
        if x is None:
            coord_sources["skipped"] += 1

        # ---------------------------------------------------
        # Outcome classification
        # ---------------------------------------------------
        play_result = (item.get("play_result") or "").strip().upper()
        outs_on_play = item.get("outs_on_play", 0) or 0
        distance_val = item.get("distance")
        runs_scored = item.get("runs_scored", 0)

        if play_result in ("SINGLE", "1B"):
            outcome = "SINGLE"
        elif play_result in ("DOUBLE", "2B"):
            outcome = "DOUBLE"
        elif play_result in ("TRIPLE", "3B"):
            outcome = "TRIPLE"
        elif play_result in ("HOMERUN", "HOME_RUN", "HR"):
            outcome = "HOMERUN"
        elif play_result in (
            "OUT", "ERROR", "FIELDERSCHOICE", "SACRIFICE"
        ) or outs_on_play > 0:
            outcome = "OUT"
        else:
            if distance_val is not None:
                try:
                    d = float(distance_val)
                    if d >= 400:
                        outcome = "HOMERUN"
                    elif d >= 300:
                        outcome = "DOUBLE"
                    elif d >= 200:
                        outcome = "SINGLE"
                    else:
                        outcome = "OUT"
                except (TypeError, ValueError):
                    outcome = "OUT"
            else:
                outcome = "OUT"

        hang_time = item.get("hang_time")

        records.append({
            "x": x,
            "y": y,
            "z": z,
            "distance": distance_val,
            "hang_time": hang_time,
            "outcome": outcome,
            "batter_id": item.get("batter_id"),
            "batter_side": item.get("batter_side"),
            "pitcher_throws": item.get("pitcher_throws"),
            "pitcher_id": item.get("pitcher_id"),
            "date": item.get("date"),
            "game_id": item.get("game_id"),
            "exit_speed": item.get("exit_speed"),
            "angle": item.get("angle"),
            "direction": item.get("direction"),
            "outs_on_play": outs_on_play,
            "runs_scored": runs_scored,
            "play_result": play_result,
            "pitch_call": item.get("pitch_call", ""),
            "auto_hit_type": item.get("auto_hit_type", ""),
            "tagged_hit_type": item.get("tagged_hit_type", ""),
        })

    log.info(
        f"[parse_spray] Coordinate sources: "
        f"direction+distance={coord_sources['dir_dist']}, "
        f"position_at_110_x+distance={coord_sources['pos110']}, "
        f"trajectory_polynomial={coord_sources['trajectory']}, "
        f"skipped={coord_sources['skipped']}"
    )

    df = pd.DataFrame(records)
    n_start = len(df)
    log.info(f"[parse_spray] Raw records: {n_start}")

    # Step 0: Drop rows with no x/y coordinates at all
    df = df.dropna(subset=["x", "y"])
    log.info(f"[parse_spray] After coordinate check: {len(df)}/{n_start}")

    # ---------------------------------------------------
    # Step 1: pitch_call must be "inplay" (flexible matching)
    # ---------------------------------------------------
    if "pitch_call" in df.columns:
        pitch_norm = df["pitch_call"].apply(_normalize_pitch_call)
        before = len(df)
        df = df[pitch_norm == "inplay"]
        log.info(f"[parse_spray] After pitch_call filter: {len(df)}/{before}")
    else:
        log.warning("[parse_spray] No pitch_call column — skipping in-play filter")

    # ---------------------------------------------------
    # Step 2: Exclude definite ground balls and bunts
    # ---------------------------------------------------
    if "auto_hit_type" in df.columns:
        before = len(df)
        auto_norm = df["auto_hit_type"].apply(_normalize_hit_type)
        df = df[~auto_norm.isin(EXCLUDE_HIT_TYPES) | (auto_norm == "")]
        log.info(f"[parse_spray] After auto_hit_type filter: {len(df)}/{before}")

    if "tagged_hit_type" in df.columns:
        before = len(df)
        tagged_norm = df["tagged_hit_type"].apply(_normalize_hit_type)
        df = df[~tagged_norm.isin(EXCLUDE_HIT_TYPES) | (tagged_norm == "")]
        log.info(f"[parse_spray] After tagged_hit_type filter: {len(df)}/{before}")

    # ---------------------------------------------------
    # Step 3: Outcome filter
    # ---------------------------------------------------
    if "outcome" in df.columns:
        before = len(df)
        df = df[df["outcome"].isin(VALID_OUTCOMES)]
        log.info(f"[parse_spray] After outcome filter (OUT/SINGLE/DOUBLE): {len(df)}/{before}")

    # ---------------------------------------------------
    # Step 4: Launch angle — NaN-safe
    # ---------------------------------------------------
    if "angle" in df.columns:
        before = len(df)
        angle_ok = df["angle"].isna() | (pd.to_numeric(df["angle"], errors="coerce") > 10)
        df = df[angle_ok]
        log.info(f"[parse_spray] After angle filter (>10 or NaN): {len(df)}/{before}")

    # ---------------------------------------------------
    # Step 5: Distance — NaN-safe
    # ---------------------------------------------------
    if "distance" in df.columns:
        before = len(df)
        dist_ok = df["distance"].isna() | (pd.to_numeric(df["distance"], errors="coerce") >= 150)
        df = df[dist_ok]
        log.info(f"[parse_spray] After distance filter (>=150 or NaN): {len(df)}/{before}")

    log.info(f"[parse_spray] Final kept: {len(df)} outfield balls")
    return df


def get_player_spray_dataframe(player_id: str) -> pd.DataFrame:
    """
    Convenience wrapper: Load raw JSON → convert to cleaned DataFrame.
    """
    spray_data = load_spray_data(player_id)
    if spray_data is None:
        return pd.DataFrame()
    return parse_spray_to_dataframe(spray_data)


# -------------------------------------------------------
# Filtering and utilities
# -------------------------------------------------------

def filter_players_by_handedness(
    players: List[Dict],
    handedness: Optional[str] = None
) -> List[Dict]:
    """Filter players by batting handedness."""
    if handedness is None:
        return players
    return [
        p for p in players
        if p.get("player_batting_handedness") == handedness
    ]


def get_unique_players_with_spray_data() -> List[Dict]:
    """Return only players for whom spray JSON files exist."""
    players = load_players()
    spray_dir = DATA_DIR / "spray"

    if not spray_dir.exists():
        return []

    available_ids = {f.stem for f in spray_dir.glob("*.json")}

    return [
        p for p in players
        if p.get("player_id") in available_ids
    ]