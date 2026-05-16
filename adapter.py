# adapter.py — SLUGGER API Request Adapter
# -*- coding: utf-8 -*-
"""
Module for fetching real baseball data through the SLUGGER API.
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

BASE_URL = "https://1ywv9dczq5.execute-api.us-east-2.amazonaws.com/ALPBAPI"
API_KEY = os.getenv("API_KEY")
DATA_DIR = Path(__file__).parent.parent / "data" / "spray"

if not API_KEY:
    log.warning("API_KEY not found in .env file. API calls will fail.")

HEADERS = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

# -------------------------------------------------------
# Shared filter helpers
# These must stay in sync with data_loader.parse_spray_to_dataframe
# -------------------------------------------------------

EXCLUDE_HIT_TYPES = {"groundball", "bunt"}

# Minimum number of qualifying outfield balls a player must have
# to appear in the dropdown menu AND to generate a spray chart.
MIN_QUALIFYING_BALLS = 15


def _normalize_pitch_call(value) -> str:
    """Collapse "InPlay", "in_play", "In Play" → "inplay"."""
    if not value:
        return ""
    return str(value).strip().lower().replace("_", "").replace(" ", "")


def _is_batted_ball(p: Dict) -> bool:
    """True if the record has at least one Trackman coordinate field populated."""
    return any([
        p.get("hit_trajectory_xc2") is not None,
        p.get("hit_trajectory_xc1") is not None,
        p.get("hit_trajectory_xc0") is not None,
        p.get("position_at_110_x") is not None,
        p.get("direction") is not None,
        p.get("exit_speed") is not None,
    ])


def _is_outfield_ball(p: Dict) -> bool:
    """
    True if this record qualifies as an outfield ball-in-play.

    Mirrors the filter logic in parse_spray_to_dataframe (NaN-safe):
      - pitch_call normalises to "inplay"
      - auto_hit_type not in {groundball, bunt}  (pass if null)
      - angle > 10°  (pass if null)
      - distance >= 150 ft  (pass if null)
    """
    # pitch_call check
    if _normalize_pitch_call(p.get("pitch_call")) != "inplay":
        return False

    # hit type check
    auto_ht = str(p.get("auto_hit_type") or "").strip().lower()
    if auto_ht in EXCLUDE_HIT_TYPES:
        return False
    tagged_ht = str(p.get("tagged_hit_type") or "").strip().lower()
    if tagged_ht in EXCLUDE_HIT_TYPES:
        return False

    # angle check (NaN-safe)
    angle = p.get("angle")
    if angle is not None:
        try:
            if float(angle) <= 10:
                return False
        except (TypeError, ValueError):
            pass

    # distance check (NaN-safe)
    distance = p.get("distance")
    if distance is not None:
        try:
            if float(distance) < 150:
                return False
        except (TypeError, ValueError):
            pass

    return True


def _normalize_pitcher_hand(value) -> Optional[str]:
    if not value:
        return None
    v = str(value).upper().strip()
    if v.startswith("RIGHT") or v == "R":
        return "R"
    if v.startswith("LEFT") or v == "L":
        return "L"
    return None


# -------------------------------------------------------
# Local fallback loader
# -------------------------------------------------------

def _load_spray_from_local_file(player_id: str,
                                 pitcher_hand: Optional[str] = None) -> List[Dict]:
    """Load spray data from local JSON fallback file."""
    spray_file = DATA_DIR / f"{player_id}.json"

    if not spray_file.exists():
        log.debug(f"Local file does not exist: {spray_file}")
        return []

    try:
        with open(spray_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            pitches_data = data.get("data", [])
        elif isinstance(data, list):
            pitches_data = data
        else:
            log.warning(f"Unexpected JSON format: {spray_file}")
            return []

        if not pitches_data:
            return []

        log.info(f"Local fallback: loaded {len(pitches_data)} entries from {spray_file.name}")

        filtered = [p for p in pitches_data if _is_batted_ball(p)]

        if pitcher_hand:
            ph = pitcher_hand.replace("HP", "").upper()
            filtered = [
                p for p in filtered
                if _normalize_pitcher_hand(p.get("pitcher_throws")) == ph
            ]

        log.info(f"Local fallback: {len(filtered)} valid entries after filter")
        return filtered

    except Exception as e:
        log.error(f"Failed to load local file {spray_file}: {e}")
        return []


# -------------------------------------------------------
# API helpers
# -------------------------------------------------------

def _get_with_retry(url: str, params: dict, max_retries: int = 2) -> Optional[dict]:
    """GET with exponential-backoff retry on 502/network errors."""
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)

            if response.status_code == 502:
                if attempt < max_retries:
                    time.sleep((attempt + 1) * 2)
                    continue
                raise requests.exceptions.HTTPError(f"502 after {max_retries} retries")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep((attempt + 1) * 2)
                continue
            log.error(f"Request failed: {e}")
            return None

    return None


# -------------------------------------------------------
# Public API functions
# -------------------------------------------------------

def fetch_ballparks(ballpark_name: Optional[str] = None,
                    city: Optional[str] = None,
                    state: Optional[str] = None,
                    limit: int = 50,
                    page: int = 1,
                    order: str = "ASC") -> List[Dict]:
    """Fetch list of ballparks."""
    url = f"{BASE_URL}/ballparks"
    params = {"limit": limit, "page": page, "order": order}
    if ballpark_name:
        params["ballpark_name"] = ballpark_name
    if city:
        params["city"] = city
    if state:
        params["state"] = state

    data = _get_with_retry(url, params)
    if data and data.get("success"):
        return data.get("data", [])
    log.error(f"fetch_ballparks failed: {data}")
    return []


def fetch_games(ballpark_name: Optional[str] = None,
                team_name: Optional[str] = None,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                limit: int = 50,
                page: int = 1,
                order: str = "DESC") -> List[Dict]:
    """Fetch list of games."""
    url = f"{BASE_URL}/games"
    params = {"limit": limit, "page": page, "order": order}
    if ballpark_name:
        params["ballpark_name"] = ballpark_name
    if team_name:
        params["team_name"] = team_name
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    data = _get_with_retry(url, params)
    if data and data.get("success"):
        return data.get("data", [])
    log.error(f"fetch_games failed: {data}")
    return []


def fetch_player_spray(player_id: str,
                       pitcher_hand: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       limit: int = 5000) -> List[Dict]:
    """
    Fetch spray chart data for a specific batter, paginating automatically.

    Filters applied here match parse_spray_to_dataframe (NaN-safe):
      - batted ball check (at least one coordinate field)
      - pitcher hand filter if specified

    The full parse_spray_to_dataframe is still needed downstream to clean
    coordinates and apply outcome/angle/distance filtering on the DataFrame.
    """
    PAGE_SIZE = 1000
    url = f"{BASE_URL}/pitches"

    pitcher_hand_upper = None
    if pitcher_hand:
        pitcher_hand_upper = pitcher_hand.replace("HP", "").upper()

    all_pitches: List[Dict] = []
    page = 1

    while len(all_pitches) < limit:
        params = {
            "batter_id": player_id,
            "limit": PAGE_SIZE,
            "page": page,
            # Filter to in-play pitches only at the API level.
            # This dramatically reduces data transfer since we only care
            # about balls put in play for outfield positioning.
            "pitch_call": "InPlay",
        }
        if start_date:
            params["date_range_start"] = start_date
        if end_date:
            params["date_range_end"] = end_date

        log.info(
            f"fetch_player_spray: player={player_id} page={page} "
            f"collected={len(all_pitches)}"
        )

        data = _get_with_retry(url, params)

        if not data:
            log.warning(f"Page {page} failed — stopping with {len(all_pitches)} rows")
            break

        if not data.get("success"):
            log.error(f"API error on page {page}: {data.get('message')}")
            break

        page_rows = data.get("data", [])
        log.info(f"Page {page} returned {len(page_rows)} raw rows")

        if not page_rows:
            log.info(f"No more data after page {page - 1}")
            break

        # Keep only actual batted balls
        batted = [p for p in page_rows if _is_batted_ball(p)]

        # Filter by pitcher hand if requested
        if pitcher_hand_upper:
            batted = [
                p for p in batted
                if _normalize_pitcher_hand(p.get("pitcher_throws")) == pitcher_hand_upper
            ]

        all_pitches.extend(batted)
        log.info(
            f"Page {page}: {len(batted)} batted balls kept "
            f"(total so far: {len(all_pitches)})"
        )

        # If the page returned fewer than PAGE_SIZE, we've exhausted the data
        if len(page_rows) < PAGE_SIZE:
            log.info(f"Last page reached — total collected: {len(all_pitches)}")
            break

        page += 1

    if not all_pitches:
        log.warning(f"No data from API for {player_id} — trying local fallback")
        return _load_spray_from_local_file(player_id, pitcher_hand)

    log.info(f"fetch_player_spray: {player_id} → {len(all_pitches)} batted balls")
    return all_pitches[:limit]


def fetch_players(team_name: Optional[str] = None,
                  handedness: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  limit: int = 5000) -> List[Dict]:
    """Fetch player list, paginating until limit or exhaustion."""
    PAGE_SIZE = 500
    url = f"{BASE_URL}/players"

    baseParams = {}
    if team_name:
        baseParams["team_name"] = team_name

    if handedness:
        h = handedness.upper()
        if h in ("LEFT", "L"):
            baseParams["player_batting_handedness"] = "Left"
        elif h in ("RIGHT", "R"):
            baseParams["player_batting_handedness"] = "Right"
        elif h in ("SWITCH", "S"):
            baseParams["player_batting_handedness"] = "Switch"
        else:
            baseParams["player_batting_handedness"] = handedness

    if not API_KEY:
        log.error("API_KEY missing.")
        return []

    allPlayers: List[Dict] = []
    page = 1
    MAX_PAGE_ATTEMPTS = 3

    while len(allPlayers) < limit:
        params = dict(baseParams)
        params["limit"] = PAGE_SIZE
        params["page"] = page

        log.info(
            f"fetch_players: page={page} collected={len(allPlayers)}"
        )

        # Retry the whole page a few times if it fails — don't bail the
        # entire fetch on one transient error.
        data = None
        for attempt in range(MAX_PAGE_ATTEMPTS):
            data = _get_with_retry(url, params, max_retries=2)
            if data and data.get("success"):
                break
            log.warning(
                f"fetch_players page={page} attempt={attempt + 1} failed, retrying"
            )
            time.sleep(1.0 * (attempt + 1))

        if not data or not data.get("success"):
            log.error(
                f"fetch_players: giving up on page {page} after "
                f"{MAX_PAGE_ATTEMPTS} attempts — total so far: {len(allPlayers)}"
            )
            break

        pageRows = data.get("data", [])
        log.info(f"Page {page} returned {len(pageRows)} players")

        if not pageRows:
            log.info(f"No more data after page {page - 1}")
            break

        allPlayers.extend(pageRows)

        # Last page reached
        if len(pageRows) < PAGE_SIZE:
            log.info(f"Last page reached — total collected: {len(allPlayers)}")
            break

        page += 1

    return allPlayers[:limit]


def fetch_batted_balls(player_ids: Optional[List[str]] = None,
                       handedness: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       limit: int = 5000) -> List[Dict]:
    """Fetch bulk batted-ball data."""
    url = f"{BASE_URL}/atbats"
    params = {"limit": limit}

    if player_ids:
        params["player_ids"] = ",".join(player_ids)
    if handedness:
        params["handedness"] = handedness
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    data = _get_with_retry(url, params, max_retries=1)
    if data and data.get("success"):
        return data.get("data", [])

    log.error(f"fetch_batted_balls failed: {data}")
    return []


def probe_player_has_data(player_id: str) -> bool:
    """
    Check if a player has enough qualifying outfield balls in play
    against BOTH pitcher hands.

    A player only passes if they have >= MIN_QUALIFYING_BALLS against
    RHP AND >= MIN_QUALIFYING_BALLS against LHP. This matches how the
    compute endpoint loads data (per-pitcher-hand), so a player who
    passes the probe is guaranteed to have enough data for either
    selection in the UI.

    Fails closed: if the API errors out, returns False so the player
    is dropped from the dropdown rather than shown and then erroring
    on click.
    """
    url = f"{BASE_URL}/pitches"
    PROBE_LIMIT = 500

    for hand in ("Right", "Left"):
        params = {
            "batter_id": player_id,
            "limit": PROBE_LIMIT,
            "pitch_call": "InPlay",
            "pitcher_throws": hand,
        }

        data = _get_with_retry(url, params, max_retries=1)

        if not data or not data.get("success"):
            log.debug(f"probe: API failed for {player_id} vs {hand} — failing closed")
            return False

        records = data.get("data", [])
        qualifying_count = sum(1 for p in records if _is_outfield_ball(p))

        if qualifying_count < MIN_QUALIFYING_BALLS:
            log.debug(
                f"probe: {player_id} vs {hand} — "
                f"{qualifying_count}/{MIN_QUALIFYING_BALLS} (FAIL)"
            )
            return False

        log.debug(
            f"probe: {player_id} vs {hand} — {qualifying_count} (PASS)"
        )

    return True