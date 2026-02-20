# adapter.py — SLUGGER API Request Adapter
# -*- coding: utf-8 -*-
"""
Module for fetching real baseball data through the SLUGGER API.

"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

# Load .env file
load_dotenv()

log = logging.getLogger(__name__)

# API Base Configuration
BASE_URL = "https://1ywv9dczq5.execute-api.us-east-2.amazonaws.com/ALPBAPI"
API_KEY = os.getenv("API_KEY")

# Local fallback spray data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "spray"


def _load_spray_from_local_file(player_id: str, pitcher_hand: Optional[str] = None) -> List[Dict]:
    """
    Load spray data from a local JSON file (fallback if API fails).
    
    Args:
        player_id: Batter ID
        pitcher_hand: Pitcher throwing hand filter (R/L, optional)
    
    Returns:
        List[Dict]: Batted ball data, or empty list if no file or invalid.
    """
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
            log.debug(f"Local file contains no data: {spray_file}")
            return []
        
        log.info(f"Loaded {len(pitches_data)} entries from local file")
        
        filtered_pitches = [
            p for p in pitches_data
            if any([
                p.get("hit_trajectory_xc2") is not None,
                p.get("hit_trajectory_xc1") is not None,
                p.get("hit_trajectory_xc0") is not None,
                p.get("direction") is not None,
                p.get("exit_speed") is not None
            ])
        ]
        
        log.info(f"Valid batted-ball entries: {len(filtered_pitches)}/{len(pitches_data)}")
        
        if pitcher_hand and filtered_pitches:
            pitcher_hand_upper = pitcher_hand.replace("HP", "").upper()
            original_count = len(filtered_pitches)
            
            def normalize_pitcher_throws(value):
                if not value:
                    return None
                v = str(value).upper().strip()
                if v.startswith("RIGHT") or v == "R":
                    return "R"
                if v.startswith("LEFT") or v == "L":
                    return "L"
                return None
            
            filtered_pitches = [
                p for p in filtered_pitches
                if normalize_pitcher_throws(p.get("pitcher_throws")) == pitcher_hand_upper
            ]
            
            log.info(
                f"Pitcher Hand Filter ({pitcher_hand_upper}): "
                f"{len(filtered_pitches)}/{original_count}"
            )
        
        return filtered_pitches
        
    except Exception as e:
        log.error(f"Failed to load local file: {spray_file}, Error: {e}")
        return []


if not API_KEY:
    log.warning("API_KEY not found in .env file. API calls will fail.")

HEADERS = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}


def fetch_ballparks(ballpark_name: Optional[str] = None,
                    city: Optional[str] = None,
                    state: Optional[str] = None,
                    limit: int = 50,
                    page: int = 1,
                    order: str = "ASC") -> List[Dict]:
    """
    Fetch list of ballparks.
    """
    url = f"{BASE_URL}/ballparks"
    params = {"limit": limit, "page": page, "order": order}
    
    if ballpark_name:
        params["ballpark_name"] = ballpark_name
    if city:
        params["city"] = city
    if state:
        params["state"] = state
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return data.get("data", [])
        
        log.error(f"API error: {data.get('message')}")
        return []
    
    except Exception as e:
        log.error(f"Failed to fetch ballparks: {e}")
        return []


def fetch_games(ballpark_name: Optional[str] = None,
                team_name: Optional[str] = None,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                limit: int = 50,
                page: int = 1,
                order: str = "DESC") -> List[Dict]:
    """
    Fetch list of games.
    """
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
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return data.get("data", [])
        
        log.error(f"API error: {data.get('message')}")
        return []
    
    except Exception as e:
        log.error(f"Failed to fetch games: {e}")
        return []


def fetch_player_spray(player_id: str,
                       pitcher_hand: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       limit: int = 5000) -> List[Dict]:
    """
    Fetch spray chart data for a specific batter using the /pitches endpoint.
    Paginates automatically to collect all available data up to `limit` rows.
    Each page requests 1000 records (the API maximum per request).
    """
    import time

    PAGE_SIZE = 1000
    url = f"{BASE_URL}/pitches"
    max_retries = 2

    def normalize_pitcher(value):
        if not value:
            return None
        v = str(value).upper().strip()
        if v.startswith("RIGHT") or v == "R":
            return "R"
        if v.startswith("LEFT") or v == "L":
            return "L"
        return None

    def is_batted_ball(p):
        return any([
            p.get("hit_trajectory_xc2") is not None,
            p.get("hit_trajectory_xc1") is not None,
            p.get("hit_trajectory_xc0") is not None,
            p.get("direction") is not None,
            p.get("exit_speed") is not None
        ])

    pitcher_hand_upper = None
    if pitcher_hand:
        pitcher_hand_upper = pitcher_hand.replace("HP", "").upper()

    all_pitches = []
    page = 1

    while len(all_pitches) < limit:
        params = {
            "batter_id": player_id,
            "limit": PAGE_SIZE,
            "page": page
        }
        if start_date:
            params["date_range_start"] = start_date
        if end_date:
            params["date_range_end"] = end_date

        log.info(f"Fetching page {page} for player {player_id} (collected {len(all_pitches)} so far)")

        retry_count = 0
        data = None

        while retry_count <= max_retries:
            try:
                response = requests.get(url, headers=HEADERS, params=params, timeout=30)
                log.info(f"Page {page} — HTTP {response.status_code}")

                if response.status_code == 502:
                    retry_count += 1
                    if retry_count <= max_retries:
                        time.sleep(retry_count * 2)
                        continue
                    raise requests.exceptions.HTTPError("502 after retries")

                response.raise_for_status()
                data = response.json()
                break

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count <= max_retries:
                    time.sleep(retry_count * 2)
                    continue
                log.error(f"API request failed on page {page}: {str(e)}")
                break

        # If this page failed entirely, stop paginating and use what we have
        if not data:
            log.warning(f"Page {page} failed — stopping pagination with {len(all_pitches)} rows")
            break

        if not data.get("success"):
            log.error(f"API error on page {page}: {data.get('message')}")
            break

        page_pitches = data.get("data", [])
        log.info(f"Page {page} returned {len(page_pitches)} rows")

        # No more data — we've exhausted all pages
        if not page_pitches:
            log.info(f"No more data after page {page - 1} — pagination complete")
            break

        # Filter to batted balls only
        page_pitches = [p for p in page_pitches if is_batted_ball(p)]

        # Filter by pitcher hand if requested
        if pitcher_hand_upper:
            page_pitches = [
                p for p in page_pitches
                if normalize_pitcher(p.get("pitcher_throws")) == pitcher_hand_upper
            ]

        all_pitches.extend(page_pitches)

        # If the API returned fewer than a full page, we're done
        if len(data.get("data", [])) < PAGE_SIZE:
            log.info(f"Last page reached — total collected: {len(all_pitches)}")
            break

        page += 1

    if not all_pitches:
        log.warning(f"No spray data from API for {player_id} — trying local fallback")
        return _load_spray_from_local_file(player_id, pitcher_hand)

    log.info(f"Total pitches collected for {player_id}: {len(all_pitches)}")
    return all_pitches[:limit]


def fetch_players(team_name: Optional[str] = None,
                  handedness: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  limit: int = 5000) -> List[Dict]:
    """
    Fetch player list.
    """
    url = f"{BASE_URL}/players"
    params = {"limit": min(limit, 1000)}
    
    if team_name:
        params["team_name"] = team_name
    
    if handedness:
        h = handedness.upper()
        if h in ["LEFT", "L"]:
            params["player_batting_handedness"] = "Left"
        elif h in ["RIGHT", "R"]:
            params["player_batting_handedness"] = "Right"
        elif h in ["SWITCH", "S"]:
            params["player_batting_handedness"] = "Switch"
        else:
            params["player_batting_handedness"] = handedness
    
    if not API_KEY:
        log.error("API_KEY missing.")
        return []
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return data.get("data", [])
        
        log.error(f"API Error: {data.get('message')}")
        return []
    
    except Exception as e:
        log.error(f"API request failed: {e}")
        return []


def fetch_batted_balls(player_ids: Optional[List[str]] = None,
                       handedness: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       limit: int = 5000) -> List[Dict]:
    """
    Fetch bulk batted-ball data.
    """
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
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return data.get("data", [])
        
        log.error(f"API error: {data.get('message')}")
        return []
    
    except Exception as e:
        log.error(f"Failed to fetch batted balls: {e}")
        return []


def probe_player_has_data(player_id: str) -> bool:
    """
    Check if a player has at least 1 usable outfield ball in play.
    Applies the same core filters as parse_spray_to_dataframe.
    """
    EXCLUDE_HIT_TYPES = {"groundball", "bunt"}
    try:
        url = f"{BASE_URL}/pitches"
        params = {"batter_id": player_id, "limit": 50}
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if response.status_code != 200:
            return False
        data = response.json()
        if not data.get("success"):
            return False
        for p in data.get("data", []):
            if (p.get("pitch_call") or "").strip().lower() != "inplay":
                continue
            if (p.get("auto_hit_type") or "").strip().lower() in EXCLUDE_HIT_TYPES:
                continue
            if (p.get("angle") or 0) <= 10:
                continue
            if (p.get("distance") or 0) < 150:
                continue
            return True
        return False
    except Exception:
        return False