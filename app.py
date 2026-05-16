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
        fetch_players, MIN_QUALIFYING_BALLS,
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


def build_player_dict(players: list) -> Dict[str, Dict]:
    """
    Convert a list of player records into the batter dict format
    used by the frontend.  Deduplicates by (name, hand).
    """
    result = {}
    seen = set()
    for p in players:
        pid = p.get("player_id")
        if not pid:
            continue
        name = (p.get("player_name") or "").strip()
        if not is_valid_player_name(name):
            continue
        hand = normalize_hand(p.get("player_batting_handedness") or "")
        key_pair = (name, hand)
        if key_pair in seen:
            continue
        seen.add(key_pair)
        result[str(pid)] = {
            "label": f"{name} ({hand})",
            "batter_name": name,
            "batter_hand": hand,
            "player_id": pid,
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


def make_plot(
    df: pd.DataFrame,
    positions: Optional[Dict[str, Tuple[float, float]]],
    batter_label: str,
    pitcher_hand: str,
) -> str:
    """Draw a synthetic field and overlay spray data. Returns base64 PNG."""
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
    ax.set_title(f"{batter_label} vs {pitcher_hand}",
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
) -> str:
    """
    Render spray chart over a real ballpark photo.
    Falls back to make_plot() if image/config unavailable.
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
        return make_plot(df, positions, batter_label, pitcher_hand)

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
        return make_plot(df, positions, batter_label, pitcher_hand)

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
    # Lateral (X): depth-weighted centroid (same as before).
    # Depth (Y):   place fielder at the 35th percentile of ball
    #              pixel_y in the zone — meaning 65% of balls are
    #              shallower (higher pixel_y) and 35% are deeper.
    #              Lower percentile = closer to fence.
    #              Adjust DEPTH_POSITION_PERCENTILE to taste:
    #                20 = very aggressive (near fence)
    #                35 = moderately deep (default)
    #                50 = plain median
    DEPTH_POSITION_PERCENTILE = 30

    optimized_pixel = {}
    if balls_pixel:
        sorted_dots = sorted(balls_pixel, key=lambda d: d[0])
        third = max(1, len(sorted_dots) // 3)
        for name, dots in [("LF", sorted_dots[:third]),
                           ("CF", sorted_dots[third:2*third]),
                           ("RF", sorted_dots[2*third:])]:
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

    # Legend
    ax.legend(handles=[
        Patch(facecolor=OUTCOME_COLORS["OUT"], label="OUT"),
        Patch(facecolor=OUTCOME_COLORS["SINGLE"], label="SINGLE"),
        Patch(facecolor=OUTCOME_COLORS["DOUBLE"], label="DOUBLE"),
    ], loc="upper right", framealpha=0.9, fontsize=10)

    ax.axis("off")
    ax.set_title(f"{batter_label} vs {pitcher_hand}",
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
    # ── API adapter mode ──
    if USE_API_ADAPTER and not USE_JSON_LOADER:
        pitcher_letter = pitcher_hand_label.replace("HP", "").upper()
        spray_data = fetch_player_spray(
            player_id=batter_id, pitcher_hand=pitcher_letter,
            start_date=None, end_date=None, limit=1000)

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
        background_image_path=background_image_path)

    return img_b64, batter_label


# ═════════════════════════════════════════════════════════
#  BACKGROUND PLAYER PROBE CACHE
# ═════════════════════════════════════════════════════════

_players_with_data_cache: dict = {}
_cache_ready: bool = False


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

    threading.Thread(target=run, daemon=True).start()
    log.info("Background probe started for %d players", len(batch))


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
                batters = build_player_dict(_filter_by_qualifying_cache(players))
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
                batters = build_player_dict(_filter_by_qualifying_cache(players))
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
    batters = build_player_dict(confirmed)

    return jsonify({
        "ready": _cache_ready,
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

        # ── API adapter mode ──
        if USE_API_ADAPTER and not USE_JSON_LOADER:
            pitcher_letter = pitcher_hand.replace("HP", "").upper() if pitcher_hand else "R"
            spray_data = fetch_player_spray(
                player_id=batter_id, pitcher_hand=pitcher_letter,
                start_date=None, end_date=None, limit=1000)

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
            pitcher_hand=pitcher_hand, background_image_path=bg_path)

        return jsonify({
            "ok": True,
            "batter_id": batter_id,
            "batter_label": meta["label"],
            "batter_hand": meta["batter_hand"],
            "pitcher_hand": pitcher_hand,
            "positions": positions_drawn or {},
            "image_base64": img_b64,
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

        spray_data = fetch_player_spray(
            player_id=player_id, pitcher_hand=pitcher_hand,
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
            background_image_path=bg_path)

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

@app.route("/api/pdf/<batter_id>", methods=["GET"])
def api_pdf(batter_id: str):
    """
    Generate a portrait 8.5×11 PDF with vs RHP (top) and
    vs LHP (bottom) spray charts stacked vertically.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.utils import ImageReader
    except ImportError:
        return jsonify({
            "ok": False,
            "error": "reportlab not installed — run: pip install reportlab"
        }), 500

    try:
        from PIL import Image as PILImage

        client_name = request.args.get("batter_name")
        client_hand = request.args.get("batter_hand")

        rhp_img, rhp_label = load_spray_and_render(
            batter_id, "RHP", client_name, client_hand)
        lhp_img, lhp_label = load_spray_and_render(
            batter_id, "LHP", client_name, client_hand)

        if rhp_img is None and lhp_img is None:
            return jsonify({
                "ok": False,
                "error": f"No data. RHP: {rhp_label}. LHP: {lhp_label}"
            }), 404

        page_w, page_h = letter  # 612 × 792 pt
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf, pagesize=letter)

        player_name = rhp_label or lhp_label or "Unknown Player"
        margin = 24

        # Title
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(page_w / 2, page_h - 32,
                            f"SLUGGER Scouting Report \u2014 {player_name}")

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
            scale = min(chart_w / pil.width, chart_h / pil.height)
            dw, dh = pil.width * scale, pil.height * scale
            c.drawImage(
                ImageReader(io.BytesIO(img_bytes)),
                margin + (chart_w - dw) / 2,
                y_bottom + (chart_h - dh) / 2,
                width=dw, height=dh)

        top_y = header_bottom - chart_h
        draw_chart(rhp_img, top_y, "vs RHP")
        draw_chart(lhp_img, top_y - gap - chart_h, "vs LHP")

        c.save()
        buf.seek(0)

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