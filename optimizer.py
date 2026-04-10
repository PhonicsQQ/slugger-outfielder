# optimizer.py — Excel Macro Algorithm Implementation
#
# This module implements the same optimization logic used in the Excel macro.
# The basic brute-force optimizer remains inside app.py for backward compatibility.

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime, date

# -------------------------------------------------------
# Outfielder Movement Parameters (Excel values)
# -------------------------------------------------------
FIELDER_PARAMS = {
    "RF": {
        "ramp_up_v": 5.0,   # acceleration speed (units/s)
        "cruise_v": 8.0,    # cruising speed (units/s)
        "ramp_up_t": 2.0,   # acceleration duration (s)
        "ramp_up_d": 15.0   # distance covered during ramp-up (units)
    },
    "CF": {
        "ramp_up_v": 6.0,
        "cruise_v": 10.0,
        "ramp_up_t": 2.0,
        "ramp_up_d": 14.0
    },
    "LF": {
        "ramp_up_v": 5.0,
        "cruise_v": 8.0,
        "ramp_up_t": 2.0,
        "ramp_up_d": 15.0
    }
}

# -------------------------------------------------------
# PENALTY / REWARD values (Excel model)
# -------------------------------------------------------
PENALTY_REWARD = {
    "OUT": 0.3,         # reward
    "SINGLE": -0.87,    # penalty
    "DOUBLE": -1.217,   # penalty
    "TRIPLE": -1.5,     # penalty (estimated)
    "HOMERUN": -2.0     # penalty (estimated)
}

# -------------------------------------------------------
# Default Excel grid configuration
# -------------------------------------------------------
DEFAULT_GRID_PARAMS = {
    "RF": {
        "min_x": 180, "max_x": 200, "step_x": 10,
        "min_y": 80,  "max_y": 130, "step_y": 10   # was 110
    },
    "CF": {
        "min_x": 105, "max_x": 145, "step_x": 10,
        "min_y": 60,  "max_y": 110, "step_y": 10   # was 90
    },
    "LF": {
        "min_x": 40,  "max_x": 70,  "step_x": 10,
        "min_y": 80,  "max_y": 130, "step_y": 10   # was 110
    }
}


# -------------------------------------------------------
# Fielder Movement Model
# -------------------------------------------------------

def calculate_fielder_time(
    fielder_pos: Tuple[float, float],
    ball_landing_pos: Tuple[float, float],
    fielder_type: str = "CF"
) -> float:
    """
    Compute the time it takes an outfielder to reach the ball (Excel logic).

    Args:
        fielder_pos: (x, y) of the fielder
        ball_landing_pos: (x, y) of ball landing point
        fielder_type: one of "RF", "CF", "LF"

    Returns:
        float: arrival time in seconds

    Excel logic:
        - ramp-up time = MIN(ramp_up_t, distance / ramp_up_v)
        - cruise time  = MAX(0, distance - ramp_up_d) / cruise_v
        - total time   = ramp-up + cruise
    """
    params = FIELDER_PARAMS.get(fielder_type, FIELDER_PARAMS["CF"])

    distance = np.hypot(
        ball_landing_pos[0] - fielder_pos[0],
        ball_landing_pos[1] - fielder_pos[1]
    )

    ramp_up_v = params["ramp_up_v"]
    cruise_v = params["cruise_v"]
    ramp_up_t = params["ramp_up_t"]
    ramp_up_d = params["ramp_up_d"]

    ramp_up_time = min(ramp_up_t, distance / ramp_up_v)
    cruise_distance = max(0.0, distance - ramp_up_d)
    cruise_time = cruise_distance / cruise_v

    return ramp_up_time + cruise_time


def calculate_fielder_penalty_reward(
    fielder_time: float,
    hangtime: float,
    ball_y: float,
    fielder_y: float,
    single_buffer: float = 2.0
) -> float:
    """
    Compute PENALTY/REWARD for each outfielder (Excel logic).

    Args:
        fielder_time: fielder arrival time
        hangtime: ball hang time
        ball_y: ball landing y coordinate
        fielder_y: fielder y coordinate
        single_buffer: additional buffer time used for deciding singles

    Returns:
        float: reward/penalty value
    """
    if fielder_time <= hangtime:
        return PENALTY_REWARD["OUT"]
    elif hangtime < fielder_time <= hangtime + single_buffer and ball_y >= fielder_y:
        return PENALTY_REWARD["SINGLE"]
    else:
        return PENALTY_REWARD["DOUBLE"]


def calculate_penalty_reward(outcome: str, weight: float = 1.0) -> float:
    """
    Legacy penalty/reward scorer retained for compatibility.

    Args:
        outcome: ball outcome label
        weight: scaling factor

    Returns:
        weighted penalty/reward
    """
    penalty = PENALTY_REWARD.get(outcome.upper(), 0.0)
    return penalty * weight


# -------------------------------------------------------
# Date-based recency weighting
# -------------------------------------------------------

def calculate_date_weight(
    game_date: str,
    reference_date: Optional[date] = None,
    weight_thresholds: Optional[list] = None,
    weight_values: Optional[list] = None
) -> float:
    """
    Compute recency-based weight for a hit ball (Excel logic).

    Args:
        game_date: date string YYYY-MM-DD
        reference_date: comparison date (default: today)
        weight_thresholds: e.g., [0, 365, 730, 1095]
        weight_values: e.g., [1.0, 0.7, 0.5, 0.3]

    Returns:
        float: recency weight
    """
    if reference_date is None:
        reference_date = date.today()

    if weight_thresholds is None:
        weight_thresholds = [0, 365, 730, 1095]

    if weight_values is None:
        weight_values = [1.0, 0.7, 0.5, 0.3]

    try:
        if isinstance(game_date, str):
            game_date_obj = datetime.strptime(game_date, "%Y-%m-%d").date()
        elif isinstance(game_date, date):
            game_date_obj = game_date
        else:
            return 1.0
    except:
        return 1.0

    days_back = (reference_date - game_date_obj).days

    for i, threshold in enumerate(weight_thresholds):
        if days_back <= threshold:
            return weight_values[i]

    return weight_values[-1] if weight_values else 1.0


# -------------------------------------------------------
# Depth-based importance weighting
# -------------------------------------------------------
# Deeper balls carry more positional leverage — mispositioning
# on a 370-ft line drive is catastrophic (fly out vs triple),
# while mispositioning on a 180-ft blooper barely matters
# (single either way).  This function scales each ball's
# contribution to the optimizer objective by its distance
# from home plate.
#
# Default thresholds (in feet, matching Trackman "distance"):
#   SHALLOW  < 200 ft  → weight 0.5  (low leverage)
#   MEDIUM   200–300 ft → weight 1.0  (baseline)
#   DEEP     > 300 ft  → weight 1.5  (high leverage)
# -------------------------------------------------------

DEPTH_WEIGHT_CONFIG = {
    "shallow_cutoff": 200.0,   # feet — below this is a blooper
    "deep_cutoff":    260.0,   # feet — above this is a deep fly
    "shallow_weight":   0.5,
    "medium_weight":    1.0,
    "deep_weight":      4.0,
}


def calculate_depth_weight(
    distance: float,
    config: Optional[Dict] = None
) -> float:
    """
    Compute depth-based importance weight for a batted ball.

    Args:
        distance: ball distance from home plate in feet
        config: optional override for depth weight thresholds

    Returns:
        float: multiplier (< 1 for shallow, > 1 for deep)
    """
    if config is None:
        config = DEPTH_WEIGHT_CONFIG

    if distance <= 0 or np.isnan(distance):
        return config["medium_weight"]

    if distance < config["shallow_cutoff"]:
        return config["shallow_weight"]
    elif distance > config["deep_cutoff"]:
        return config["deep_weight"]
    else:
        return config["medium_weight"]


# -------------------------------------------------------
# Excel Macro Optimization Engine
# -------------------------------------------------------

def optimize_outfield_excel(
    df: pd.DataFrame,
    grid_params: Optional[Dict] = None,
    weights: Optional[list] = None,
    use_date_weight: bool = True
) -> Dict[str, Tuple[float, float]]:
    """
    Full Excel macro optimization:

    Args:
        df: DataFrame containing x, y, outcome, hang_time
        grid_params: grid ranges for LF/CF/RF search
        weights: recency weights
        use_date_weight: apply recency logic if True

    Returns:
        Dict with optimized LF/CF/RF coordinates

    Excel algorithm (summary):
        1. Sweep every combination of LF, CF, RF grid coordinates.
        2. For each ball:
            - Compute arrival time for LF, CF, RF.
            - Compute penalty/reward independently.
            - Take MAX across the three outfielders.
            - Apply recency weight.
        3. Choose the combination that MINIMIZES the total penalty (or equivalently maximizes reward).
    """
    if grid_params is None:
        grid_params = DEFAULT_GRID_PARAMS

    if weights is None:
        weights = [1.0, 0.7, 0.5, 0.3]

    ball_positions = list(zip(df["x"].values, df["y"].values))
    outcomes = df["outcome"].values
    hang_times = df["hang_time"].fillna(0).values

    # Extract ball distances for depth-based weighting
    if "distance" in df.columns:
        ball_distances = pd.to_numeric(df["distance"], errors="coerce").fillna(0).values
    else:
        ball_distances = np.zeros(len(df))

    # Build grid ranges
    rf_x_range = range(grid_params["RF"]["min_x"], grid_params["RF"]["max_x"] + 1, grid_params["RF"]["step_x"])
    rf_y_range = range(grid_params["RF"]["min_y"], grid_params["RF"]["max_y"] + 1, grid_params["RF"]["step_y"])

    cf_x_range = range(grid_params["CF"]["min_x"], grid_params["CF"]["max_x"] + 1, grid_params["CF"]["step_x"])
    cf_y_range = range(grid_params["CF"]["min_y"], grid_params["CF"]["max_y"] + 1, grid_params["CF"]["step_y"])

    lf_x_range = range(grid_params["LF"]["min_x"], grid_params["LF"]["max_x"] + 1, grid_params["LF"]["step_x"])
    lf_y_range = range(grid_params["LF"]["min_y"], grid_params["LF"]["max_y"] + 1, grid_params["LF"]["step_y"])

    best_total = float("inf")
    best_positions = {}
    iteration = 0

    # Excel macro-style 6 nested loops
    for rfx in rf_x_range:
        for rfy in rf_y_range:
            for cfx in cf_x_range:
                for cfy in cf_y_range:
                    for lfx in lf_x_range:
                        for lfy in lf_y_range:

                            total = 0.0

                            for i, (ball_pos, outcome, hang_time) in enumerate(
                                zip(ball_positions, outcomes, hang_times)
                            ):
                                ball_x, ball_y = ball_pos

                                # Compute arrival times
                                rf_time = calculate_fielder_time((rfx, rfy), ball_pos, "RF")
                                cf_time = calculate_fielder_time((cfx, cfy), ball_pos, "CF")
                                lf_time = calculate_fielder_time((lfx, lfy), ball_pos, "LF")

                                # Compute penalty for each fielder independently
                                if hang_time > 0:
                                    rf_score = calculate_fielder_penalty_reward(rf_time, hang_time, ball_y, rfy)
                                    cf_score = calculate_fielder_penalty_reward(cf_time, hang_time, ball_y, cfy)
                                    lf_score = calculate_fielder_penalty_reward(lf_time, hang_time, ball_y, lfy)

                                    best_penalty = max(rf_score, cf_score, lf_score)
                                else:
                                    best_penalty = PENALTY_REWARD.get(outcome.upper(), 0.0)

                                # Apply recency weight
                                if use_date_weight and "date" in df.columns:
                                    game_date = df.iloc[i]["date"]
                                    weight = calculate_date_weight(game_date)
                                else:
                                    weight_idx = min(i, len(weights) - 1)
                                    weight = weights[weight_idx]

                                # Apply depth-based importance weight
                                depth_weight = calculate_depth_weight(ball_distances[i])

                                total += weight * depth_weight * best_penalty

                            # Track best combination
                            if total < best_total:
                                best_total = total
                                best_positions = {
                                    "RF": (float(rfx), float(rfy)),
                                    "CF": (float(cfx), float(cfy)),
                                    "LF": (float(lfx), float(lfy))
                                }

                            iteration += 1

    return best_positions