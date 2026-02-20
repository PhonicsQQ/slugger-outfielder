# app.py â€” Outfield Positioning Optimizer Demo (drawn field)
# -*- coding: utf-8 -*-

import io
import base64
import sys
import os
import re
import logging
from typing import Dict, Tuple, Optional
from dotenv import load_dotenv

# Load .env file for environment variables
load_dotenv()

# Fix UTF-8 encoding issues for Windows terminals
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace',
            line_buffering=True
        )
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding='utf-8',
            errors='replace',
            line_buffering=True
        )

from flask import Flask, request, jsonify, render_template_string, render_template, send_file
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# -------------------------------------------------------
# CONFIG: Hardcoded demo batters (used when real data unavailable)
# -------------------------------------------------------
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

LAST_CSV_PATH = "optimized_positions.csv"

# -------------------------------------------------------
# SYNTHETIC SPRAY GENERATION (used when API/JSON data unavailable)
# -------------------------------------------------------
def generate_spray(batter_id: str, pitcher_hand: str) -> pd.DataFrame:
    """
    Generate synthetic spray data based on batter handedness and pitcher handedness.
    Uses a mixture of clusters to create realistic spray patterns with gaps and variation.
    """
    if batter_id in BATTERS:
        meta = BATTERS[batter_id]
        bhand = meta["batter_hand"]
    else:
        bhand = "R"

    seed = abs(hash(batter_id + "_" + pitcher_hand)) % (2**32)
    rng = np.random.default_rng(seed)
    n = 150

    # Create multiple spray clusters for realistic scatter
    # Each cluster: (center_x, center_y, std_x, std_y, count)
    if bhand == "L" and pitcher_hand == "RHP":
        clusters = [
            (200, 320, 30, 30, 45),   # pull side RF gap
            (170, 290, 25, 35, 35),   # right-center
            (150, 340, 20, 25, 25),   # straightaway CF deep
            (120, 280, 30, 30, 25),   # left-center
            (90, 310, 25, 25, 20),    # opposite field LF
        ]
    elif bhand == "L" and pitcher_hand == "LHP":
        clusters = [
            (150, 310, 35, 30, 40),   # center
            (180, 290, 25, 35, 30),   # right-center
            (110, 300, 30, 30, 30),   # left-center
            (200, 330, 20, 20, 25),   # RF line
            (80, 320, 20, 25, 25),    # LF line
        ]
    elif bhand == "R" and pitcher_hand == "LHP":
        clusters = [
            (100, 320, 30, 30, 45),   # pull side LF gap
            (130, 290, 25, 35, 35),   # left-center
            (150, 340, 20, 25, 25),   # straightaway CF deep
            (180, 280, 30, 30, 25),   # right-center
            (210, 310, 25, 25, 20),   # opposite field RF
        ]
    else:  # R vs RHP
        clusters = [
            (150, 310, 35, 30, 35),   # center
            (120, 290, 25, 35, 30),   # left-center
            (190, 300, 30, 30, 30),   # right-center
            (100, 330, 20, 20, 25),   # LF line
            (210, 320, 20, 25, 30),   # RF line
        ]

    all_x = []
    all_y = []
    for cx, cy, sx, sy, count in clusters:
        all_x.append(rng.normal(cx, sx, count))
        all_y.append(rng.normal(cy, sy, count))

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)

    # Shuffle to mix clusters
    idx = rng.permutation(len(x))
    x = x[idx][:n]
    y = y[idx][:n]

    x = np.clip(x, 50, 250)
    y = np.clip(y, 230, 400)

    return pd.DataFrame({"x": x, "y": y})

# -------------------------------------------------------
# BASIC OPTIMIZATION (grid search for LF/CF/RF positions)
# -------------------------------------------------------
def optimize_outfield(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Perform brute-force search over predefined LF/CF/RF grids
    to minimize total distance from each batted ball to the nearest fielder.
    """
    lf_grid = [(x, y) for x in range(70, 120, 10) for y in range(260, 330, 10)]
    cf_grid = [(x, y) for x in range(120, 180, 10) for y in range(310, 380, 10)]
    rf_grid = [(x, y) for x in range(180, 230, 10) for y in range(260, 330, 10)]

    bx = df["x"].to_numpy()
    by = df["y"].to_numpy()

    best_score = float("inf")
    best = {}

    for lf in lf_grid:
        dlf = np.hypot(bx - lf[0], by - lf[1])
        for cf in cf_grid:
            dcf = np.hypot(bx - cf[0], by - cf[1])
            for rf in rf_grid:
                drf = np.hypot(bx - rf[0], by - rf[1])
                dist_min = np.minimum(np.minimum(dlf, dcf), drf)
                total_distance = dist_min.sum()
                if total_distance < best_score:
                    best_score = total_distance
                    best = {"LF": lf, "CF": cf, "RF": rf}

    return best


def assign_distance_based_outcomes(df: pd.DataFrame, positions: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Assign realistic outcomes based on distance from each ball to the nearest fielder.
    Balls close to fielders are outs, farther balls are singles, very far balls are doubles.
    """
    bx = df["x"].to_numpy()
    by = df["y"].to_numpy()

    # Calculate distance to nearest fielder
    distances = []
    for name, (fx, fy) in positions.items():
        d = np.hypot(bx - fx, by - fy)
        distances.append(d)
    min_dist = np.minimum.reduce(distances)

    # Assign outcomes based on distance thresholds
    # Most outfield balls are caught - outs should dominate
    p65 = np.percentile(min_dist, 65)
    p90 = np.percentile(min_dist, 90)

    outcomes = []
    for d in min_dist:
        if d <= p65:
            outcomes.append("OUT")
        elif d <= p90:
            outcomes.append("SINGLE")
        else:
            outcomes.append("DOUBLE")
    
    df = df.copy()
    df["outcome"] = outcomes
    return df


# -------------------------------------------------------
# PLOTTING FUNCTION (drawn field visualization)
# -------------------------------------------------------
from matplotlib.patches import Polygon, Rectangle, Circle, Arc

def make_plot(df: pd.DataFrame,
              positions: Dict[str, Tuple[float, float]],
              batter_label: str,
              pitcher_hand: str) -> str:
    """
    Draw a baseball field and overlay spray data and optimized fielder positions.
    """
    # Identify an outcome column if available
    outcome_col = None
    for c in df.columns:
        if c.lower() in ("result", "outcome", "event"):
            outcome_col = c
            break

    # If no outcomes provided, generate synthetic labels
    if outcome_col is None:
        rng = np.random.default_rng(0)
        labels = np.array(["1B", "2B", "3B", "OUT"])
        df["outcome"] = rng.choice(
            labels, size=len(df),
            p=[0.55, 0.25, 0.03, 0.17]
        )
        outcome_col = "outcome"

    # Color map for outcomes
    color_map = {
        "1B": "#42a5f5",
        "2B": "#ec42f5",
        "3B": "#ffa726",
        "OUT": "#bdbdbd"
    }
    spray_colors = df[outcome_col].map(
        lambda v: color_map.get(str(v).upper(), "white")
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#144d14")  # grass color

    home = (150, 200)
    left_line = (60, 250)
    right_line = (240, 250)

    # Draw outfield fence arc
    fence_radius = 180
    fence_arc = Arc(
        home,
        width=fence_radius * 2,
        height=fence_radius * 2,
        theta1=22, theta2=158,
        edgecolor="white",
        linewidth=2, zorder=1
    )
    ax.add_patch(fence_arc)

    # Outfield grass polygon
    fence_points = []
    for angle in np.linspace(22, 158, 30):
        rad = np.radians(angle)
        px = home[0] + fence_radius * np.cos(rad)
        py = home[1] + fence_radius * np.sin(rad)
        fence_points.append((px, py))

    outfield_poly = Polygon(
        [left_line] + fence_points + [right_line],
        closed=True,
        facecolor="#1c6b1c",
        edgecolor="none",
        zorder=0
    )
    ax.add_patch(outfield_poly)

    # Infield dirt arc
    dirt_radius = 95
    dirt_arc = Arc(
        home,
        width=dirt_radius * 2,
        height=dirt_radius * 2,
        theta1=22, theta2=158,
        edgecolor="#c49a6c",
        linewidth=25, zorder=2
    )
    ax.add_patch(dirt_arc)

    # Basepath shape
    basepath = Polygon(
        [
            (150, 200),
            (170, 220),
            (150, 240),
            (130, 220)
        ],
        closed=True,
        facecolor="#c49a6c",
        edgecolor="white",
        linewidth=2,
        zorder=3
    )
    ax.add_patch(basepath)

    # Baselines
    ax.plot([home[0], left_line[0]], [home[1], left_line[1]],
            color="white", linewidth=2, zorder=4)
    ax.plot([home[0], right_line[0]], [home[1], right_line[1]],
            color="white", linewidth=2, zorder=4)

    # Centerline
    ax.plot([150, 150], [250, 380],
            color="white", linestyle="--",
            linewidth=1.2, alpha=0.6, zorder=4)

    # Spray dots
    ax.scatter(
        df["x"], df["y"],
        c=spray_colors, s=30, alpha=0.8,
        edgecolor="none", zorder=5
    )

    # Draw optimized LF/CF/RF boxes
    box_w, box_h = 12, 12
    for name, (cx, cy) in positions.items():
        rect = Rectangle(
            (cx - box_w/2, cy - box_h/2),
            box_w, box_h,
            linewidth=2, edgecolor="red",
            facecolor="none", zorder=7
        )
        ax.add_patch(rect)

        ax.scatter(cx, cy, c="red", s=70, zorder=8)

        ax.text(
            cx, cy + box_h + 3,
            name, color="red",
            fontsize=10, ha="center",
            va="bottom", weight="bold",
            zorder=9
        )

    ax.set_xlim(40, 260)
    ax.set_ylim(200, 420)
    ax.axis("off")

    ax.set_title(
        f"{batter_label} vs {pitcher_hand}",
        color="white", fontsize=16, pad=12
    )

    # Export as base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def make_plot_with_image(
    df: pd.DataFrame,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    batter_label: str = "Test Player",
    pitcher_hand: str = "RHP",
    background_image_path: str = "img/background.avif"
) -> str:
    """
    Visualization using a real ballpark background image (Phase 3 workflow).

    Args:
        df: DataFrame containing spray coordinates and outcomes.
        positions: Optional optimized outfielder positions in logical coordinates.
        batter_label: Display label for hitter.
        pitcher_hand: String label ("RHP" or "LHP").
        background_image_path: Path to the field background image.

    Returns:
        Base64-encoded PNG containing the rendered spray and field layout.
    """
    from PIL import Image

    # -------------------------------------------------------
    # Determine outcome labels & colors
    # -------------------------------------------------------
    outcome_col = None
    for c in df.columns:
        if c.lower() in ("result", "outcome", "event"):
            outcome_col = c
            break

    if outcome_col is None:
        outcome_col = "outcome"
        if "outcome" not in df.columns:
            rng = np.random.default_rng(42)
            df["outcome"] = rng.choice(
                ["OUT", "SINGLE", "DOUBLE"],
                size=len(df), p=[0.55, 0.30, 0.15]
            )

    color_map = {
        "OUT": "#bdbdbd",
        "SINGLE": "#42a5f5",
        "DOUBLE": "#ec42f5",
        "TRIPLE": "#ffa726",
        "HOMERUN": "#ef5350",
        "1B": "#42a5f5",
        "2B": "#ec42f5",
        "3B": "#ffa726",
        "HR": "#ef5350",
    }

    spray_colors = df[outcome_col].map(
        lambda v: color_map.get(str(v).upper(), "#ffffff")
    )

    # -------------------------------------------------------
    # Load background image
    # -------------------------------------------------------
    try:
        img = Image.open(background_image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)
    except Exception as e:
        log.warning(f"Could not load background image: {e}. Falling back to drawn field.")
        return make_plot(df, positions, batter_label, pitcher_hand)

    # -------------------------------------------------------
    # Load outfield polygon + affine transforms
    # -------------------------------------------------------
    outfield_manager = None
    try:
        from outfield_region import OutfieldRegionManager
        from pathlib import Path

        config_path = "outfield_region_config.json"
        if Path(config_path).exists():
            outfield_manager = OutfieldRegionManager(config_path)
    except Exception as e:
        print(f"[OutfieldRegion] Failed to load region config (ignored): {e}")

    if not outfield_manager:
        log.warning("OutfieldRegionManager unavailable â€” coordinate transforms disabled.")
        return make_plot(df, positions, batter_label, pitcher_hand)

    # Image dimensions
    img_height, img_width = img_array.shape[:2]
    original_width = img_width
    original_height = img_height

    # -------------------------------------------------------
    # Render figure with pixel-based axes
    # -------------------------------------------------------
    target_dpi = 150
    figsize_width = original_width / target_dpi
    figsize_height = original_height / target_dpi

    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height), dpi=target_dpi)

    ax.imshow(img_array, origin="upper", zorder=0)
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Invert y-axis to match image coordinates

    # -------------------------------------------------------
    # MLB â†’ logical â†’ pixel coordinate transformation for spray points
    # -------------------------------------------------------
    balls_pixel = []

    if len(df) > 0:
        from mlb_to_logical_converter import mlb_to_logical_simple_scale

        mlb_x_values = df["x"].dropna().tolist()
        mlb_y_values = df["y"].dropna().tolist()

        if mlb_x_values and mlb_y_values:
            mlb_x_range = (min(mlb_x_values), max(mlb_x_values))
            mlb_y_range = (min(mlb_y_values), max(mlb_y_values))

            for idx, row in df.iterrows():
                mlb_x = row["x"]
                mlb_y = row["y"]

                if pd.isna(mlb_x) or pd.isna(mlb_y):
                    continue

                logical_x, logical_y = mlb_to_logical_simple_scale(
                    mlb_x, mlb_y, mlb_x_range, mlb_y_range
                )

                pixel_x, pixel_y = outfield_manager.logical_to_pixel(
                    (logical_x, logical_y)
                )

                # Clamp to valid pixel range
                pixel_x = max(0, min(img_width - 1, pixel_x))
                pixel_y = max(0, min(img_height - 1, pixel_y))

                # Only keep dots in the outfield grass area
                # Defined by: fence (top), infield dirt arc (bottom), foul lines (sides)
                outfield_top = 750      # fence line
                outfield_bottom = 900   # where outfield grass meets infield dirt
                if pixel_y > outfield_bottom or pixel_y < outfield_top:
                    continue

                # Check dot is between the foul lines (fan/wedge from home plate)
                # Home plate pixel: (1170, 1400)
                # Left foul pole: (220, 850), Right foul pole: (2140, 854)
                # Slightly wider than actual poles to allow dots near the lines
                home_x, home_y = 1170, 1400
                x_left = home_x + (pixel_y - home_y) * (220 - home_x) / (850 - home_y)
                x_right = home_x + (pixel_y - home_y) * (2140 - home_x) / (854 - home_y)
                if pixel_x < x_left or pixel_x > x_right:
                    continue

                color = spray_colors.iloc[idx] if idx < len(spray_colors) else "#ffffff"
                balls_pixel.append((pixel_x, pixel_y, color))

    # -------------------------------------------------------
    # Calculate optimized fielder positions from spray dot locations
    # Place each fielder at the centroid of spray dots in their zone
    # -------------------------------------------------------
    optimized_pixel = {}

    if balls_pixel:
        # Sort dots by pixel x to divide into LF/CF/RF thirds
        sorted_dots = sorted(balls_pixel, key=lambda d: d[0])
        n_dots = len(sorted_dots)
        third = max(1, n_dots // 3)

        lf_dots = sorted_dots[:third]
        cf_dots = sorted_dots[third:2*third]
        rf_dots = sorted_dots[2*third:]

        for name, dots in [("LF", lf_dots), ("CF", cf_dots), ("RF", rf_dots)]:
            if dots:
                avg_x = sum(d[0] for d in dots) / len(dots)
                avg_y = sum(d[1] for d in dots) / len(dots)
                optimized_pixel[name] = (avg_x, avg_y)

    # -------------------------------------------------------
    # Reassign outcomes based on pixel-space distance to nearest fielder
    # Outs cluster around fielders, hits land in gaps
    # -------------------------------------------------------
    if balls_pixel and optimized_pixel:
        fielder_positions = list(optimized_pixel.values())
        min_dists = []

        for (px, py, _) in balls_pixel:
            d = min(np.hypot(px - fx, py - fy) for fx, fy in fielder_positions)
            min_dists.append(d)

        min_dists = np.array(min_dists)
        p65 = np.percentile(min_dists, 65)
        p90 = np.percentile(min_dists, 90)

        outcome_colors = {
            "OUT": "#bdbdbd",
            "SINGLE": "#42a5f5",
            "DOUBLE": "#ec42f5",
        }

        new_balls = []
        for i, (px, py, _) in enumerate(balls_pixel):
            if min_dists[i] <= p65:
                outcome = "OUT"
            elif min_dists[i] <= p90:
                outcome = "SINGLE"
            else:
                outcome = "DOUBLE"
            new_balls.append((px, py, outcome_colors[outcome]))

        balls_pixel = new_balls

    # Draw spray dots (after outcome reassignment)
    for px, py, color in balls_pixel:
        ax.scatter(
            px, py, s=40, c=color, alpha=0.7,
            edgecolor="white", linewidth=0.5, zorder=5,
        )

    # -------------------------------------------------------
    # Draw optimized LF/CF/RF pixel markers
    # -------------------------------------------------------
    box_w, box_h = 8, 8
    for name, (px, py) in optimized_pixel.items():
        rect = Rectangle(
            (px - box_w / 2, py - box_h / 2),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="red",
            facecolor="yellow",
            alpha=0.7,
            zorder=7,
        )
        ax.add_patch(rect)

        ax.scatter(
            px,
            py,
            c="red",
            s=60,
            edgecolor="white",
            linewidth=0.8,
            zorder=8,
        )

    # -------------------------------------------------------
    # Legend (OUT, SINGLE, DOUBLE, TRIPLE)
    # -------------------------------------------------------
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_map["OUT"], label="OUT"),
        Patch(facecolor=color_map["SINGLE"], label="SINGLE"),
        Patch(facecolor=color_map["DOUBLE"], label="DOUBLE"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.9,
        fontsize=10,
    )

    # -------------------------------------------------------
    # Final cleanup
    # -------------------------------------------------------
    ax.axis("off")
    ax.set_title(
        f"{batter_label} vs {pitcher_hand}",
        color="black",
        fontsize=16,
        pad=12,
        weight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Export the figure
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=target_dpi,
        facecolor="white",
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -------------------------------------------------------
# DATA LOADER (development/testing mode using JSON files)
#--------------------------------------------------------
# Notes:
# - JSON loader is used during development when API is unavailable.
#
# Environment override:
#   USE_API_MODE=true  â†’ force API mode (ignore JSON loader)
#   USE_API_MODE=false â†’ use JSON loader if available (default)
# -------------------------------------------------------

USE_API_MODE_ENV = os.getenv("USE_API_MODE", "false")
USE_API_MODE = USE_API_MODE_ENV.lower() == "true"

log.info("=" * 60)
log.info("Data Loading Mode")
log.info(f"USE_API_MODE (from .env): '{USE_API_MODE_ENV}'")
log.info(f"Mode interpreted as: {'API mode' if USE_API_MODE else 'JSON mode (default)'}")

if USE_API_MODE:
    # Force API mode via environment variable
    USE_JSON_LOADER = False
    log.info("API mode forced by environment variable (JSON loader disabled)")
else:
    # Default behavior:
    # If data_loader.py exists â†’ JSON loader mode
    # If not, fall back to API adapter
    try:
        from data_loader import (
            load_players,
            get_player_spray_dataframe,
            get_unique_players_with_spray_data,
            filter_players_by_handedness,
            parse_spray_to_dataframe
        )
        USE_JSON_LOADER = True
        log.info("JSON loader mode enabled (data_loader.py detected)")
    except ImportError:
        USE_JSON_LOADER = False
        log.warning("data_loader.py not found, falling back to API adapter")

# -------------------------------------------------------
# SLUGGER API ADAPTER LOADING
# -------------------------------------------------------
# If JSON loader is not used, attempt to load API adapter
# -------------------------------------------------------
try:
    from adapter import (
        fetch_ballparks,
        fetch_games,
        fetch_player_spray,
        fetch_players
    )
    USE_API_ADAPTER = True
    log.info("API adapter loaded successfully")
except ImportError:
    USE_API_ADAPTER = False
    log.warning("adapter.py not found â€” API requests disabled")

# -------------------------------------------------------
# Determine final data mode
# -------------------------------------------------------
if USE_API_ADAPTER and not USE_JSON_LOADER:
    final_mode = "API Adapter (SLUGGER API calls)"
elif USE_JSON_LOADER:
    final_mode = "JSON Loader (local spray data)"
else:
    final_mode = "Synthetic Data Mode (no API/JSON available)"

log.info(f"Final Mode Selected: {final_mode}")
log.info("=" * 60)

# -------------------------------------------------------
# OPTIMIZER (Excel-based optimization algorithm)
# -------------------------------------------------------
try:
    from optimizer import optimize_outfield_excel
    USE_EXCEL_ALGORITHM = True
    log.info("Excel-based optimizer loaded")
except ImportError:
    USE_EXCEL_ALGORITHM = False
    log.warning("optimizer.py not found â€” falling back to basic brute-force optimizer")

# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------

@app.route("/")
def index():
    """
    Main page route â€” renders templates/index.html.
    Player lists are loaded dynamically depending on mode:
    - JSON loader mode (development)
    - API adapter mode (production/test mode)
    - Synthetic fallback if neither is available
    """
    # ---------------------------------------------------
    # JSON LOADER MODE (local spray files available)
    # ---------------------------------------------------
    if USE_JSON_LOADER:
        try:
            players_with_data = get_unique_players_with_spray_data()

            actual_batters = {}
            seen_names = set()  # Deduplicate by (name, handedness)

            for player in players_with_data:
                player_id = player.get("player_id")
                player_name = player.get("player_name", "Unknown")
                batting_hand = (player.get("player_batting_handedness") or "").upper()

                # Normalize handedness to a single letter
                if batting_hand in ["LEFT", "L"]:
                    hand_suffix = "L"
                elif batting_hand in ["RIGHT", "R"]:
                    hand_suffix = "R"
                else:
                    hand_suffix = "U"

                # Clean and validate name text
                name_clean = player_name.strip()
                if not name_clean or len(name_clean) < 2:
                    continue
                if name_clean.startswith(",") or name_clean == ",":
                    continue
                if not re.search(r"[a-zA-Z0-9]", name_clean):
                    continue

                # Deduplicate identical name-hand combinations
                key_pair = (name_clean, hand_suffix)
                if key_pair in seen_names:
                    continue
                seen_names.add(key_pair)

                key = f"{player_id}"
                actual_batters[key] = {
                    "label": f"{name_clean} ({hand_suffix})",
                    "batter_name": name_clean,
                    "batter_hand": hand_suffix,
                    "player_id": player_id,
                }

            if actual_batters:
                log.info(f"Loaded {len(actual_batters)} players from JSON loader")
                return render_template("index.html", batters=actual_batters)
            else:
                log.warning("No players found via JSON loader, falling back to defaults")
                return render_template("index.html", batters=BATTERS)

        except Exception:
            log.exception("Failed to load player list via JSON loader")
            return render_template("index.html", batters=BATTERS)

    # ---------------------------------------------------
    # API ADAPTER MODE  
    # ---------------------------------------------------
    elif USE_API_ADAPTER:
        try:
            log.info("API mode enabled â€” fetching players via SLUGGER API")
            players = fetch_players(limit=1000)
            log.info(f"Received {len(players)} players from API")

            actual_batters = {}
            skipped_count = 0
            duplicate_count = 0
            seen_names = set()

            for player in players:
                player_id = player.get("player_id")
                player_name = player.get("player_name") or "Unknown"
                batting_hand_raw = player.get("player_batting_handedness") or ""
                batting_hand = batting_hand_raw.upper()

                if not player_id:
                    skipped_count += 1
                    continue

                name_clean = player_name.strip()
                if not name_clean or len(name_clean) < 2:
                    skipped_count += 1
                    continue
                if name_clean.startswith(","):
                    skipped_count += 1
                    continue
                if not re.search(r"[a-zA-Z0-9]", name_clean):
                    skipped_count += 1
                    continue

                # Convert handedness to "L", "R", or "U"
                if batting_hand in ["LEFT", "L"]:
                    hand_suffix = "L"
                elif batting_hand in ["RIGHT", "R"]:
                    hand_suffix = "R"
                else:
                    hand_suffix = "U"

                key_pair = (name_clean, hand_suffix)
                if key_pair in seen_names:
                    duplicate_count += 1
                    continue
                seen_names.add(key_pair)

                key = f"{player_id}"
                actual_batters[key] = {
                    "label": f"{name_clean} ({hand_suffix})",
                    "batter_name": name_clean,
                    "batter_hand": hand_suffix,
                    "player_id": player_id,
                }

            if skipped_count > 0:
                log.warning(f"{skipped_count} players skipped due to invalid data")
            if duplicate_count > 0:
                log.info(f"{duplicate_count} duplicate players removed")

            if actual_batters:
                log.info(f"Loaded {len(actual_batters)} players from API")
                if not _cache_ready and not _players_with_data_cache:
                    _start_background_probe(players)
                return render_template("index.html", batters=actual_batters)
            else:
                log.warning("API returned players, but all were invalid or removed")
                return render_template("index.html", batters=BATTERS)

        except Exception:
            log.exception("Failed to load player list via API")
            return render_template("index.html", batters=BATTERS)

    # ---------------------------------------------------
    # SYNTHETIC FALLBACK MODE
    # ---------------------------------------------------
    else:
        return render_template("index.html", batters=BATTERS)


# -------------------------------------------------------
# Background probe cache
# -------------------------------------------------------
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_players_with_data_cache = {}
_cache_ready = False

def _probe_player_has_data(player_id: str) -> bool:
    try:
        from adapter import probe_player_has_data
        return probe_player_has_data(player_id)
    except Exception:
        return False

def _start_background_probe(players: list) -> None:
    players_to_probe = players[:500]

    def run():
        global _players_with_data_cache, _cache_ready
        def probe(player):
            pid = player.get("player_id")
            if not pid:
                return pid, False
            return pid, _probe_player_has_data(pid)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(probe, p): p for p in players_to_probe}
            for future in as_completed(futures):
                pid, result = future.result()
                if pid:
                    _players_with_data_cache[pid] = result

        _cache_ready = True
        found = sum(1 for v in _players_with_data_cache.values() if v)
        log.info(f"Background probe complete: {found}/{len(players_to_probe)} players have data")

    t = threading.Thread(target=run, daemon=True)
    t.start()


@app.route("/api/cache-status")
def api_cache_status():
    try:
        players = fetch_players(limit=1000)
    except Exception:
        return jsonify({"ready": False, "batters": {}, "probed": 0, "total": 0})

    batters = {}
    seen_names = set()
    for player in players:
        player_id = player.get("player_id")
        if not player_id:
            continue
        if not _players_with_data_cache.get(player_id, False):
            continue
        player_name = (player.get("player_name") or "").strip()
        if not player_name or len(player_name) < 2:
            continue
        batting_hand = (player.get("player_batting_handedness") or "").upper()
        if batting_hand in ["LEFT", "L"]:
            hand = "L"
        elif batting_hand in ["RIGHT", "R"]:
            hand = "R"
        else:
            hand = "U"
        key_pair = (player_name, hand)
        if key_pair in seen_names:
            continue
        seen_names.add(key_pair)
        batters[player_id] = {
            "label": f"{player_name} ({hand})",
            "batter_name": player_name,
            "batter_hand": hand,
        }

    return jsonify({
        "ready": _cache_ready,
        "batters": batters,
        "probed": len(_players_with_data_cache),
        "total": len(players)
    })

@app.route("/api/compute", methods=["POST"])
def api_compute():
    """
    Core API endpoint for:
        - Loading spray data (API, JSON loader, or synthetic fallback)
        - Running outfield optimization
        - Rendering spray chart image (drawn field or background image)
        - Returning optimized fielder coordinates

    This is the main engine used by the frontend.
    """
    try:
        payload = request.get_json(force=True)

        batter_id = payload.get("batter_id")
        pitcher_hand = payload.get("pitcher_hand", "RHP")
        background_image_path = payload.get("background_image_path", "img/background.png")

        # Optional extra metadata sent from frontend
        client_batter_name = payload.get("batter_name")
        client_batter_hand = payload.get("batter_hand")

        log.info(
            f"Client-provided batter info: "
            f"name='{client_batter_name}', hand='{client_batter_hand}'"
        )

        # ---------------------------------------------------
        # MODE 1: API ADAPTER MODE - Production
        # ---------------------------------------------------
        if USE_API_ADAPTER and not USE_JSON_LOADER:

            pitcher_hand_for_api = (
                pitcher_hand.replace("HP", "").upper() if pitcher_hand else "R"
            )  # e.g., "RHP" â†’ "R"

            # Step 1 â€” Pull spray data from the SLUGGER API
            spray_data = fetch_player_spray(
                player_id=batter_id,
                pitcher_hand=pitcher_hand_for_api,
                start_date=None,
                end_date=None,
                limit=1000,
            )

            if not spray_data:
                log.warning(f"No spray data found for player {batter_id}")
                return jsonify({
                    "ok": False,
                    "error": "No spray data available for this player."
                }), 404

            # Step 2 â€” Convert JSON â†’ DataFrame
            from data_loader import parse_spray_to_dataframe
            df = parse_spray_to_dataframe(spray_data)

            # ---------------------------------------------------
            # Parsing failure â†’ fallback to synthetic
            # ---------------------------------------------------
            if df.empty:
                log.warning(f"Failed to parse spray data for {batter_id}, using synthetic fallback")

                # Determine batter metadata
                if client_batter_name and client_batter_name.strip():
                    name = client_batter_name.strip()
                    bh = client_batter_hand if client_batter_hand else "R"
                elif batter_id not in BATTERS:
                    name = f"Player {batter_id[:8]}"
                    bh = "R"
                else:
                    meta = BATTERS[batter_id]
                    name = meta["batter_name"]
                    bh = meta["batter_hand"]

                meta = {
                    "label": f"{name} ({bh})",
                    "batter_name": name,
                    "batter_hand": bh
                }

                df_drawn = generate_spray("dickerson_R", pitcher_hand)
                df = df_drawn.copy()
                df["x"] = (df_drawn["x"] - 150) * 0.5
                df["y"] = (df_drawn["y"] - 200) * 2.0
                df["hang_time"] = df.get("hang_time", 3.0)

                positions_drawn = optimize_outfield(df_drawn)
                df_drawn = assign_distance_based_outcomes(df_drawn, positions_drawn)
                df["outcome"] = df_drawn["outcome"].values[:len(df)]

            # ---------------------------------------------------
            # Parsed successfully but insufficient rows
            # ---------------------------------------------------
            else:
                df_filtered = df.dropna(subset=["x", "y"])

                if len(df_filtered) < 5:
                    log.warning(
                        f"Only {len(df_filtered)} rows available; using synthetic fallback"
                    )

                    if client_batter_name and client_batter_name.strip():
                        name = client_batter_name.strip()
                        bh = client_batter_hand if client_batter_hand else "R"
                    elif batter_id not in BATTERS:
                        name = f"Player {batter_id[:8]}"
                        bh = "R"
                    else:
                        meta = BATTERS[batter_id]
                        name = meta["batter_name"]
                        bh = meta["batter_hand"]

                    meta = {
                        "label": f"{name} ({bh})",
                        "batter_name": name,
                        "batter_hand": bh
                    }

                    df_drawn = generate_spray("dickerson_R", pitcher_hand)
                    df = df_drawn.copy()
                    df["x"] = (df_drawn["x"] - 150) * 0.5
                    df["y"] = (df_drawn["y"] - 200) * 2.0
                    df["hang_time"] = df.get("hang_time", 3.0)

                    positions_drawn = optimize_outfield(df_drawn)
                    df_drawn = assign_distance_based_outcomes(df_drawn, positions_drawn)
                    df["outcome"] = df_drawn["outcome"].values[:len(df)]

                # ---------------------------------------------------
                # Valid real API data â€” normal processing
                # ---------------------------------------------------
                else:
                    df_filtered["hang_time"] = df_filtered["hang_time"].fillna(3.0)
                    df_filtered["outcome"] = df_filtered["outcome"].fillna("OUT")
                    df = df_filtered

                    # Determine batter metadata
                    if client_batter_name and client_batter_name.strip():
                        name = client_batter_name.strip()
                        bh = client_batter_hand if client_batter_hand else "R"
                    else:
                        try:
                            players = fetch_players(limit=5000)
                            match = next(
                                (p for p in players if p.get("player_id") == batter_id),
                                None
                            )
                            if match:
                                name = match.get("player_name") or f"Player {batter_id[:8]}"
                                raw_hand = match.get("player_batting_handedness") or "R"
                                if raw_hand.upper() in ["LEFT", "L"]:
                                    bh = "L"
                                elif raw_hand.upper() in ["RIGHT", "R"]:
                                    bh = "R"
                                else:
                                    bh = "U"
                            else:
                                name = f"Player {batter_id[:8]}"
                                bh = "R"
                        except Exception:
                            name = f"Player {batter_id[:8]}"
                            bh = "R"

                    meta = {
                        "label": f"{name} ({bh})",
                        "batter_name": name,
                        "batter_hand": bh,
                        "player_id": batter_id
                    }

                    positions_drawn = None  # handled later inside make_plot_with_image()

        # ---------------------------------------------------
        # MODE 2: JSON LOADER MODE
        # ---------------------------------------------------
        elif USE_JSON_LOADER:

            players_with_data = get_unique_players_with_spray_data()
            player_ids = {p.get("player_id") for p in players_with_data}

            if batter_id in player_ids:
                selected = next(
                    p for p in players_with_data if p.get("player_id") == batter_id
                )

                name = selected.get("player_name", "Unknown")
                raw_hand = (selected.get("player_batting_handedness") or "").upper()

                if raw_hand in ["LEFT", "L"]:
                    bh = "L"
                elif raw_hand in ["RIGHT", "R"]:
                    bh = "R"
                else:
                    bh = "U"

                meta = {
                    "label": f"{name} ({bh})",
                    "batter_name": name,
                    "batter_hand": bh,
                    "player_id": batter_id
                }

                df = get_player_spray_dataframe(batter_id)

                if df.empty or df.dropna(subset=["x", "y"]).shape[0] < 5:
                    df_drawn = generate_spray("dickerson_R", pitcher_hand)
                    df = df_drawn.copy()
                    df["x"] = (df_drawn["x"] - 150) * 0.5
                    df["y"] = (df_drawn["y"] - 200) * 2.0
                    df["hang_time"] = 3.0
                    positions_drawn = optimize_outfield(df_drawn)
                    df_drawn = assign_distance_based_outcomes(df_drawn, positions_drawn)
                    df["outcome"] = df_drawn["outcome"].values[:len(df)]
                else:
                    df = df.dropna(subset=["x", "y"])
                    df["hang_time"] = df["hang_time"].fillna(3.0)
                    df["outcome"] = df["outcome"].fillna("OUT")
                    positions_drawn = None

            else:
                # Legacy fallback if batter_id not found
                if batter_id not in BATTERS:
                    return jsonify({"ok": False, "error": "Unknown batter"}), 400

                meta = BATTERS[batter_id]

                df_drawn = generate_spray(batter_id, pitcher_hand)
                df = df_drawn.copy()
                df["x"] = (df_drawn["x"] - 150) * 0.5
                df["y"] = (df_drawn["y"] - 200) * 2.0
                df["hang_time"] = 3.0
                positions_drawn = optimize_outfield(df_drawn)
                df_drawn = assign_distance_based_outcomes(df_drawn, positions_drawn)
                df["outcome"] = df_drawn["outcome"].values[:len(df)]

        # ---------------------------------------------------
        # MODE 3: Neither JSON loader nor API adapter available
        # ---------------------------------------------------
        else:
            if batter_id not in BATTERS:
                return jsonify({"ok": False, "error": "Unknown batter"}), 400

            meta = BATTERS[batter_id]

            df_drawn = generate_spray(batter_id, pitcher_hand)
            df = df_drawn.copy()
            df["x"] = (df_drawn["x"] - 150) * 0.5
            df["y"] = (df_drawn["y"] - 200) * 2.0
            df["hang_time"] = 3.0
            positions_drawn = optimize_outfield(df_drawn)
            df_drawn = assign_distance_based_outcomes(df_drawn, positions_drawn)
            df["outcome"] = df_drawn["outcome"].values[:len(df)]

        # ---------------------------------------------------
        # Save optimized CSV 
        # ---------------------------------------------------
        if positions_drawn:
            pd.DataFrame.from_dict(
                positions_drawn, orient="index", columns=["X", "Y"]
            ).to_csv(LAST_CSV_PATH)

        # ---------------------------------------------------
        # Create visualization (drawn or background image)
        # Passing positions=None triggers internal optimization
        # ---------------------------------------------------
        img_b64 = make_plot_with_image(
            df,
            positions=None,
            batter_label=meta["label"],
            pitcher_hand=pitcher_hand,
            background_image_path=background_image_path
        )

        return jsonify({
            "ok": True,
            "batter_id": batter_id,
            "batter_label": meta["label"],
            "batter_hand": meta["batter_hand"],
            "pitcher_hand": pitcher_hand,
            "positions": positions_drawn if positions_drawn else {},
            "image_base64": img_b64,
            "download_url": "/download"
        })

    except Exception as e:
        log.exception("api_compute failed")
        return jsonify({"ok": False, "error": str(e)}), 500
# -------------------------------------------------------
# ROUTE: Download Last Optimization Result
# -------------------------------------------------------

@app.route("/download")
def download():
    """
    Serves the last generated CSV containing optimized positions.
    Used by the frontend for 'Download Positions' functionality.
    """
    if not pd.io.common.file_exists(LAST_CSV_PATH):
        return "Run an optimization first.", 404

    return send_file(LAST_CSV_PATH, as_attachment=True)


# -------------------------------------------------------
# SLUGGER API ENDPOINTS
# -------------------------------------------------------
# These endpoints expose API adapter functionality for use
# by frontend tools or external utilities.
# -------------------------------------------------------

@app.route("/api/ballparks", methods=["GET"])
def api_ballparks():
    """
    Retrieve ballpark information from SLUGGER API.

    Query parameters:
        ballpark_name
        city
        state
        limit (default 50)
        page (default 1)
        order ("ASC" or "DESC")

    Returns:
        JSON: { success, data, count }
    """

    if not USE_API_ADAPTER:
        return jsonify({
            "success": False,
            "error": "API adapter not available"
        }), 503

    try:
        ballpark_name = request.args.get("ballpark_name")
        city = request.args.get("city")
        state = request.args.get("state")

        limit = int(request.args.get("limit", 50))
        page = int(request.args.get("page", 1))
        order = request.args.get("order", "ASC")

        ballparks = fetch_ballparks(
            ballpark_name=ballpark_name,
            city=city,
            state=state,
            limit=limit,
            page=page,
            order=order
        )

        return jsonify({
            "success": True,
            "data": ballparks,
            "count": len(ballparks)
        })

    except Exception as e:
        log.exception("api_ballparks failed")
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/api/games", methods=["GET"])
def api_games():
    """
    Retrieve game information from SLUGGER API.

    Query parameters:
        ballpark_name
        team_name
        start_date (YYYY-MM-DD)
        end_date (YYYY-MM-DD)
        limit (default 50)
        page (default 1)
        order ("ASC" or "DESC")

    Returns:
        JSON: { success, data, count }
    """

    if not USE_API_ADAPTER:
        return jsonify({
            "success": False,
            "error": "API adapter not available"
        }), 503

    try:
        ballpark_name = request.args.get("ballpark_name")
        team_name = request.args.get("team_name")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        limit = int(request.args.get("limit", 50))
        page = int(request.args.get("page", 1))
        order = request.args.get("order", "DESC")

        games = fetch_games(
            ballpark_name=ballpark_name,
            team_name=team_name,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            page=page,
            order=order
        )

        return jsonify({
            "success": True,
            "data": games,
            "count": len(games)
        })

    except Exception as e:
        log.exception("api_games failed")
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/api/players/<player_id>/spray", methods=["GET"])
def api_player_spray(player_id: str):
    """
    Retrieve spray data for a specific player using SLUGGER API.

    Path parameter:
        player_id

    Query parameters:
        pitcher_hand: "R" or "L"
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        limit: max number of rows (default 5000)

    Returns:
        JSON: { success, data, count }
    """

    if not USE_API_ADAPTER:
        return jsonify({
            "success": False,
            "error": "API adapter not available"
        }), 503

    try:
        pitcher_hand = request.args.get("pitcher_hand")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        limit = int(request.args.get("limit", 5000))

        spray_data = fetch_player_spray(
            player_id=player_id,
            pitcher_hand=pitcher_hand,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

        return jsonify({
            "success": True,
            "data": spray_data,
            "count": len(spray_data)
        })

    except Exception as e:
        log.exception("api_player_spray failed")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/optimize/<player_id>", methods=["GET", "POST"])
def api_optimize_and_visualize(player_id: str):
    """
    Compute optimal outfield positioning for a given player and return:
        - A rendered spray chart (base64 PNG)
        - Optimized LF, CF, RF coordinates (pixel coordinates for display)
        - Logical coordinates (Excel grid space for analysis)
        - Count of valid data points used

    This endpoint is API-only
    """

    if not USE_API_ADAPTER:
        return jsonify({
            "success": False,
            "error": "API adapter not available"
        }), 503

    try:
        # -------------------------------------------------------
        # Parse request parameters (GET or POST)
        # -------------------------------------------------------
        if request.method == "POST":
            payload = request.get_json(force=True) or {}
            pitcher_hand = payload.get("pitcher_hand", "R")
            start_date = payload.get("start_date")
            end_date = payload.get("end_date")
            background_image_path = payload.get("background_image_path", "img/background.png")
        else:
            pitcher_hand = request.args.get("pitcher_hand", "R")
            start_date = request.args.get("start_date")
            end_date = request.args.get("end_date")
            background_image_path = request.args.get("background_image_path", "img/background.png")

        # -------------------------------------------------------
        # Step 1 â€” Fetch spray data from SLUGGER API
        # -------------------------------------------------------
        spray_data = fetch_player_spray(
            player_id=player_id,
            pitcher_hand=pitcher_hand,
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )

        if not spray_data:
            return jsonify({
                "success": False,
                "error": f"No spray data found for player {player_id}"
            }), 404

        # -------------------------------------------------------
        # Step 2 â€” Convert raw spray JSON â†’ DataFrame
        # -------------------------------------------------------
        from data_loader import parse_spray_to_dataframe
        df = parse_spray_to_dataframe(spray_data)

        if df.empty:
            return jsonify({
                "success": False,
                "error": "Failed to parse spray data"
            }), 400

        # Keep only rows with MLB coordinates AND hang-time
        df_filtered = df.dropna(subset=["x", "y", "hang_time"])

        if len(df_filtered) < 5:
            return jsonify({
                "success": False,
                "error": f"Insufficient valid spray data: {len(df_filtered)} rows (minimum 5 required)"
            }), 400

        # -------------------------------------------------------
        # Step 3 â€” Convert MLB coordinates â†’ Logical coordinates
        # -------------------------------------------------------
        from mlb_to_logical_converter import convert_dataframe_mlb_to_logical

        df_logical = convert_dataframe_mlb_to_logical(
            df_filtered,
            mlb_x_col="x",
            mlb_y_col="y"
        )

        # -------------------------------------------------------
        # Step 4 â€” Run optimization in logical coordinate space
        # -------------------------------------------------------
        from optimizer import optimize_outfield_excel
        positions_excel_grid = optimize_outfield_excel(df_logical)

        # Convert optimizer grid output â†’ logical coordinates
        from excel_grid_to_logical_converter import convert_optimizer_positions_to_logical

        positions_logical = convert_optimizer_positions_to_logical(positions_excel_grid)

        # -------------------------------------------------------
        # Step 5 â€” Render visualization (background or drawn field)
        # -------------------------------------------------------
        batter_label = f"Player {player_id[:8]}"

        img_b64 = make_plot_with_image(
            df_filtered,
            positions=positions_logical,
            batter_label=batter_label,
            pitcher_hand="RHP" if pitcher_hand.upper() == "R" else "LHP",
            background_image_path=background_image_path
        )

        # -------------------------------------------------------
        # Step 6 â€” Convert logical â†’ pixel coordinates for display
        # -------------------------------------------------------
        from outfield_region import OutfieldRegionManager

        outfield_manager = OutfieldRegionManager("outfield_region_config.json")

        positions_pixel = {}
        for name, (lx, ly) in positions_logical.items():
            px, py = outfield_manager.logical_to_pixel((lx, ly))
            positions_pixel[name] = (float(px), float(py))

        # -------------------------------------------------------
        # Step 7 â€” Return complete response
        # -------------------------------------------------------
        return jsonify({
            "success": True,
            "image_base64": img_b64,
            "positions": positions_pixel,  # pixel coordinates for display
            "positions_logical": {
                k: (float(v[0]), float(v[1]))
                for k, v in positions_logical.items()
            },
            "data_count": len(df_filtered),
            "batter_label": batter_label,
            "player_id": player_id,
            "pitcher_hand": pitcher_hand
        })

    except Exception as e:
        log.exception("api_optimize_and_visualize failed")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# -------------------------------------------------------
# APPLICATION ENTRYPOINT
# -------------------------------------------------------
# Run the Flask server if this file is executed directly.
# In production (Railway, Render, etc.), you typically use:
#   gunicorn app:app
# but this local entrypoint allows easy development testing.

if __name__ == "__main__":
    # Host 0.0.0.0 makes the server accessible externally
    # Port 8080 aligns with common platform defaults (Railway, Render)
    app.run(host="0.0.0.0", port=8080)