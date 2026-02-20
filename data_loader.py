# data_loader.py â€” JSON-file-based data loader (development/testing use only)
#
# Note:
#   This module is intended ONLY for development and testing.
#   For distribution and production use, replace this with adapter.py
#   (which fetches real data from the SLUGGER API).

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd

# Base data directory
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# -------------------------------------------------------
# Basic load functions
# -------------------------------------------------------

def load_teams() -> List[Dict]:
    """
    Load team list from local JSON.

    Returns:
        List[Dict]: List of team dictionaries
    """
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
    
    # Load all team player files
    for json_file in players_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                team_players = data.get("data", [])
                
                # Filter by team name
                if team_name:
                    team_players = [
                        p for p in team_players
                        if p.get("team_name") == team_name
                    ]
                
                # Filter by handedness
                if batting_handedness:
                    team_players = [
                        p for p in team_players
                        if p.get("player_batting_handedness") == batting_handedness
                    ]
                
                players.extend(team_players)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")
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
        print(f"Error loading spray data for {player_id}: {e}")
        return None


def load_games() -> List[Dict]:
    """
    Load game list from JSON.

    Returns:
        List[Dict]: List of game dictionaries
    """
    games_file = DATA_DIR / "games" / "games.json"
    if not games_file.exists():
        return []
    
    with open(games_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("data", [])


def load_ballparks() -> List[Dict]:
    """
    Load ballpark list from JSON.

    Returns:
        List[Dict]: List of ballpark dictionaries
    """
    ballparks_file = DATA_DIR / "ballparks" / "ballparks.json"
    if not ballparks_file.exists():
        return []
    
    with open(ballparks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("data", [])


# -------------------------------------------------------
# Spray data â†’ DataFrame transformation
# -------------------------------------------------------

def parse_spray_to_dataframe(spray_data: List[Dict]) -> pd.DataFrame:
    """
    Convert raw spray JSON records into a clean DataFrame.

    Extracts the best-available x,y,z coordinates using a priority order:
        1. position_at_110_* (ideal)
        2. hit_trajectory_xc2/yc2 (landing point)
        3. hit_trajectory_xc1/yc1
        4. direction + distance (fallback)
        5. hit_trajectory_xc0/yc0

    Also computes outcome classification, filters invalid/groundball hits,
    and restricts to outfield-relevant balls (distance >= 200 ft).

    Args:
        spray_data: List of raw JSON pitch/spray records

    Returns:
        DataFrame with columns:
            x, y, z, distance, hang_time, outcome,
            batter_id, pitcher_throws, date, angle, direction, etc.
    """
    if not spray_data:
        return pd.DataFrame()
    
    records = []

    for item in spray_data:
        x, y, z = None, None, None

        # Priority 1: position_at_110 (ideal coordinate for trajectory)
        if item.get("position_at_110_x") is not None:
            x = item.get("position_at_110_x")
            y = item.get("position_at_110_y")
            z = item.get("position_at_110_z")

        # Priority 2: hit_trajectory xc2/yc2 (landing)
        elif item.get("hit_trajectory_xc2") is not None:
            x = item.get("hit_trajectory_xc2")
            y = item.get("hit_trajectory_yc2")
            z = item.get("hit_trajectory_zc2")

        # Priority 3: hit_trajectory xc1/yc1 (middle)
        elif item.get("hit_trajectory_xc1") is not None:
            x = item.get("hit_trajectory_xc1")
            y = item.get("hit_trajectory_yc1")
            z = item.get("hit_trajectory_zc1")

        # Priority 4: direction + distance fallback
        elif item.get("direction") is not None and item.get("distance") is not None:
            import math
            direction = float(item.get("direction"))
            distance = float(item.get("distance"))
            if distance > 0:
                # direction >0 = RF, <0 = LF, 0 = CF
                rad = math.radians(direction)
                x = distance * math.sin(rad)
                y = distance * math.cos(rad)
                z = item.get("position_at_110_z") or item.get("hit_trajectory_zc1")

        # Priority 5: initial trajectory point
        elif item.get("hit_trajectory_xc0") is not None:
            x = item.get("hit_trajectory_xc0")
            y = item.get("hit_trajectory_yc0")
            z = item.get("hit_trajectory_zc0")

        # ---------------------------------------------------
        # Outcome classification
        # ---------------------------------------------------
        play_result = (item.get("play_result") or "").upper()
        outs_on_play = item.get("outs_on_play", 0)
        distance_val = item.get("distance")
        runs_scored = item.get("runs_scored", 0)

        # Check explicit hit outcomes FIRST (a play can have outs_on_play > 0
        # and still be a hit, e.g. a single where another runner is thrown out)
        if play_result in ["SINGLE", "1B"]:
            outcome = "SINGLE"
        elif play_result in ["DOUBLE", "2B"]:
            outcome = "DOUBLE"
        elif play_result in ["TRIPLE", "3B"]:
            outcome = "TRIPLE"
        elif play_result in ["HOMERUN", "HOME_RUN", "HR"]:
            outcome = "HOMERUN"
        elif play_result == "OUT" or outs_on_play > 0:
            outcome = "OUT"
        else:
            # Infer outcome when "Undefined" or missing
            if outs_on_play > 0:
                outcome = "OUT"
            elif distance_val is not None:
                if distance_val >= 400:
                    outcome = "HOMERUN"
                elif distance_val >= 300:
                    outcome = "DOUBLE"
                elif distance_val >= 200:
                    outcome = "SINGLE"
                else:
                    outcome = "OUT"
            else:
                outcome = "OUT"

        # Hang time â€” use JSON value; placeholder for future physics calculation
        hang_time = item.get("hang_time")

        record = {
            "x": x,
            "y": y,
            "z": z,
            "distance": item.get("distance"),
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
        }

        records.append(record)

    df = pd.DataFrame(records)

    # Remove rows without coordinates
    df = df.dropna(subset=["x", "y"])

    # ---------------------------------------------------
    # ---------------------------------------------------
    # Filtering: keep only real outfield batted balls
    # ---------------------------------------------------

    # Step 1: Remove pitches not put in play (fouls, strikeouts, balls, etc.)
    if "pitch_call" in df.columns:
        df = df[df["pitch_call"].str.strip().str.lower() == "inplay"]

    # Step 2: Remove bunts and ground balls by hit type
    EXCLUDE_HIT_TYPES = {"groundball", "bunt"}
    if "auto_hit_type" in df.columns:
        df = df[~df["auto_hit_type"].str.strip().str.lower().isin(EXCLUDE_HIT_TYPES)]
    if "tagged_hit_type" in df.columns:
        df = df[~df["tagged_hit_type"].str.strip().str.lower().isin(EXCLUDE_HIT_TYPES)]

    # Step 3: Keep only outfield-relevant outcomes
    VALID_OUTCOMES = ["OUT", "SINGLE", "DOUBLE"]
    if "outcome" in df.columns:
        df = df[df["outcome"].isin(VALID_OUTCOMES)]

    # Step 4: Remove ground balls by launch angle (belt-and-suspenders)
    if "angle" in df.columns:
        df = df[df["angle"] > 10]

    # Step 5: Keep only outfield-distance balls (>= 150 ft)
    # Lowered from 200 to 150 to retain more data from limited sample sizes
    if "distance" in df.columns:
        df = df[df["distance"] >= 150]
    else:
        if len(df) > 0:
            threshold = df["y"].quantile(0.5)
            df = df[df["y"] >= threshold]

    return df


def get_player_spray_dataframe(player_id: str) -> pd.DataFrame:
    """
    Convenience wrapper:
    Load raw JSON â†’ convert to cleaned DataFrame.

    Args:
        player_id: UUID of player

    Returns:
        Cleaned DataFrame of spray data.
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
    """
    Filter players by batting handedness.

    Args:
        players: List of player dictionaries
        handedness: "Right", "Left", "Switch", or None

    Returns:
        Filtered list of players
    """
    if handedness is None:
        return players

    return [
        p for p in players
        if p.get("player_batting_handedness") == handedness
    ]


def get_unique_players_with_spray_data() -> List[Dict]:
    """
    Return only players for whom spray JSON files exist.

    Returns:
        List[Dict]: Players who have matching spray JSON data in /data/spray/
    """
    players = load_players()
    spray_dir = DATA_DIR / "spray"

    if not spray_dir.exists():
        return []

    available_ids = {f.stem for f in spray_dir.glob("*.json")}

    players_with_data = [
        p for p in players
        if p.get("player_id") in available_ids
    ]

    return players_with_data