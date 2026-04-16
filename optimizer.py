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
# Grid bounds define the search space for each outfielder in Statcast hc_x/hc_y
# units. Bounds are intentionally generous so the optimizer can recommend
# meaningful shifts (depth changes, pull-side adjustments, no-doubles alignments)
# without hitting the edge. Each fielder stays in their own lateral zone so the
# optimizer can't accidentally swap LF and RF.
#
# Ball coordinate reference (from Altuve sample, n=188):
#   hc_x range ≈ 5–234 (midfield ~125)
#   hc_y range ≈ 31–147
#
# Lateral zones (no overlap):
#   LF: x 30–100   CF: x 90–160   RF: x 150–220
# Depth range: y 40–150 for all three (shallow bloopers to warning-track)
# -------------------------------------------------------
# Default Excel grid configuration
# -------------------------------------------------------
# Grid bounds define the search space for each outfielder in Statcast hc_x/hc_y
# units. Bounds are intentionally generous so the optimizer can recommend
# meaningful shifts (depth changes, pull-side adjustments, no-doubles alignments)
# without hitting the edge. Each fielder stays in their own lateral zone so the
# optimizer can't accidentally swap LF and RF.
#
# Ball coordinate reference (from Altuve sample, n=188):
#   hc_x range ≈ 5–234 (midfield ~125)
#   hc_y range ≈ 31–147
#
# Lateral zones (no overlap):
#   LF: x 30–100   CF: x 90–160   RF: x 150–220
# Depth range: y 40–150 for all three (shallow bloopers to warning-track)
# Step size: 10 units on both axes. The inner loop is vectorized with numpy,
# so even at ~885K combinations the search finishes in a few seconds.
# -------------------------------------------------------
DEFAULT_GRID_PARAMS = {
    "RF": {
        "min_x": 150, "max_x": 220, "step_x": 10,
        "min_y": 40,  "max_y": 150, "step_y": 10
    },
    "CF": {
        "min_x": 90,  "max_x": 160, "step_x": 10,
        "min_y": 40,  "max_y": 150, "step_y": 10
    },
    "LF": {
        "min_x": 30,  "max_x": 100, "step_x": 10,
        "min_y": 40,  "max_y": 150, "step_y": 10
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
        3. Choose the combination that MAXIMIZES the total score
           (OUT = +0.3 reward, SINGLE = -0.87, DOUBLE = -1.217; higher is better).
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
    rf_x_range = list(range(grid_params["RF"]["min_x"], grid_params["RF"]["max_x"] + 1, grid_params["RF"]["step_x"]))
    rf_y_range = list(range(grid_params["RF"]["min_y"], grid_params["RF"]["max_y"] + 1, grid_params["RF"]["step_y"]))
    cf_x_range = list(range(grid_params["CF"]["min_x"], grid_params["CF"]["max_x"] + 1, grid_params["CF"]["step_x"]))
    cf_y_range = list(range(grid_params["CF"]["min_y"], grid_params["CF"]["max_y"] + 1, grid_params["CF"]["step_y"]))
    lf_x_range = list(range(grid_params["LF"]["min_x"], grid_params["LF"]["max_x"] + 1, grid_params["LF"]["step_x"]))
    lf_y_range = list(range(grid_params["LF"]["min_y"], grid_params["LF"]["max_y"] + 1, grid_params["LF"]["step_y"]))

    # -------------------------------------------------------
    # Vectorized implementation
    # -------------------------------------------------------
    # The original Excel-style implementation nested 7 loops deep (6 grid
    # dimensions × N balls). That's O(|RF|·|CF|·|LF|·N) in Python-level
    # function calls — totally infeasible for the widened grid.
    #
    # Instead we:
    #   1. Precompute per-ball arrays (x, y, hang_time, weight)
    #   2. For each candidate position of each fielder, precompute the
    #      penalty array for all balls in one vectorized pass
    #   3. For each (RF, CF, LF) combo, take element-wise max of the three
    #      precomputed penalty arrays and sum — all in numpy
    #
    # This drops runtime from ~14 min to a few seconds for the widened grid.
    # -------------------------------------------------------

    ball_x = np.asarray([p[0] for p in ball_positions], dtype=float)
    ball_y = np.asarray([p[1] for p in ball_positions], dtype=float)
    hang = np.asarray(hang_times, dtype=float)
    n_balls = len(ball_x)

    # Per-ball weights (recency × depth), computed once
    per_ball_weight = np.ones(n_balls, dtype=float)
    for i in range(n_balls):
        if use_date_weight and "date" in df.columns:
            w_rec = calculate_date_weight(df.iloc[i]["date"])
        else:
            w_rec = weights[min(i, len(weights) - 1)]
        w_depth = calculate_depth_weight(ball_distances[i])
        per_ball_weight[i] = w_rec * w_depth

    # Fallback penalty vector for balls with non-positive hang_time
    fallback = np.array(
        [PENALTY_REWARD.get(str(o).upper(), 0.0) for o in outcomes],
        dtype=float,
    )
    has_hang = hang > 0

    def penalties_for_position(fx: int, fy: int, ftype: str) -> np.ndarray:
        """
        Compute the per-ball penalty/reward array for a single fielder at
        (fx, fy). Vectorized over all balls.
        """
        params = FIELDER_PARAMS[ftype]
        dist = np.hypot(ball_x - fx, ball_y - fy)
        ramp_time = np.minimum(params["ramp_up_t"], dist / params["ramp_up_v"])
        cruise_dist = np.maximum(0.0, dist - params["ramp_up_d"])
        arrival = ramp_time + cruise_dist / params["cruise_v"]

        # Vectorized version of calculate_fielder_penalty_reward
        single_buffer = 2.0
        out_mask = arrival <= hang
        single_mask = (
            (~out_mask)
            & (arrival <= hang + single_buffer)
            & (ball_y >= fy)
        )
        pen = np.full(n_balls, PENALTY_REWARD["DOUBLE"], dtype=float)
        pen[out_mask] = PENALTY_REWARD["OUT"]
        pen[single_mask & ~out_mask] = PENALTY_REWARD["SINGLE"]

        # For balls with non-positive hang_time, fall back to outcome table
        pen = np.where(has_hang, pen, fallback)
        return pen

    # Precompute penalty arrays for every candidate position of every fielder
    rf_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for rfx in rf_x_range:
        for rfy in rf_y_range:
            rf_cache[(rfx, rfy)] = penalties_for_position(rfx, rfy, "RF")

    cf_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for cfx in cf_x_range:
        for cfy in cf_y_range:
            cf_cache[(cfx, cfy)] = penalties_for_position(cfx, cfy, "CF")

    lf_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for lfx in lf_x_range:
        for lfy in lf_y_range:
            lf_cache[(lfx, lfy)] = penalties_for_position(lfx, lfy, "LF")

    # MAXIMIZE: OUT rewards are positive (+0.3), SINGLE/DOUBLE penalties are
    # negative (-0.87 / -1.217). Higher total = better defensive alignment.
    best_total = float("-inf")
    best_positions: Dict[str, Tuple[float, float]] = {}

    for rf_key, rf_pen in rf_cache.items():
        for cf_key, cf_pen in cf_cache.items():
            # Pairwise max of RF and CF penalty vectors (reused across LF loop)
            rf_cf_max = np.maximum(rf_pen, cf_pen)
            for lf_key, lf_pen in lf_cache.items():
                best_pen = np.maximum(rf_cf_max, lf_pen)
                total = float(np.sum(per_ball_weight * best_pen))
                if total > best_total:
                    best_total = total
                    best_positions = {
                        "RF": (float(rf_key[0]), float(rf_key[1])),
                        "CF": (float(cf_key[0]), float(cf_key[1])),
                        "LF": (float(lf_key[0]), float(lf_key[1])),
                    }

    return best_positions