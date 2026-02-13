#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: MLB Coordinate â†’ Logical Coordinate Transformation Mapping

This module provides functions to map real MLB batted ball coordinates
into the logical coordinate space used by the Excel optimizer.
"""

import numpy as np
from typing import Tuple, Optional

# Logical coordinate bounds (from outfield_region_config.json)
LOGICAL_X_MIN = -89.50
LOGICAL_X_MAX = 52.66
LOGICAL_Y_MIN = -43.85
LOGICAL_Y_MAX = -14.18

# Excel grid parameter bounds (same structure used in optimizer.py)
EXCEL_GRID_PARAMS = {
    "RF": {"min_x": 180, "max_x": 200, "min_y": 80, "max_y": 110},
    "CF": {"min_x": 105, "max_x": 145, "min_y": 60, "max_y": 90},
    "LF": {"min_x": 40, "max_x": 70, "min_y": 80, "max_y": 110}
}

# Excel grid â€œcenter pointsâ€ (in logical coordinates)
EXCEL_CENTERS = {
    "LF": (55.0, 95.0),   # (min_x + max_x)/2, (min_y + max_y)/2
    "CF": (125.0, 75.0),
    "RF": (190.0, 95.0)
}

# Fixed logical coordinates for LF/CF/RF mapping targets
# These are the centroids of the LF/CF/RF regions within the outfield polygon
# (from outfield_region_config.json), NOT the label positions which sit outside the polygon.
IMAGE_LABELS_LOGICAL = {
    "LF": (-67.0, -29.0),
    "CF": (-15.0, -24.0),
    "RF": (33.0, -29.0)
}


# -------------------------------------------------------
# Linear Mapping
# -------------------------------------------------------

def mlb_to_logical_linear(
    mlb_x: float,
    mlb_y: float,
    mlb_x_range: Tuple[float, float],
    mlb_y_range: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Linearly map MLB coordinates into logical coordinate space.
    Simple fallback method if affine transform is unavailable.

    Args:
        mlb_x, mlb_y: MLB coordinates.
        mlb_x_range: (min_x, max_x) of MLB data.
        mlb_y_range: (min_y, max_y) of MLB data.

    Returns:
        (logical_x, logical_y)
    """
    mlb_x_min, mlb_x_max = mlb_x_range
    mlb_y_min, mlb_y_max = mlb_y_range

    # Linear interpolation for X
    if mlb_x_max - mlb_x_min > 0:
        logical_x = LOGICAL_X_MIN + (mlb_x - mlb_x_min) * (
            (LOGICAL_X_MAX - LOGICAL_X_MIN) / (mlb_x_max - mlb_x_min)
        )
    else:
        logical_x = (LOGICAL_X_MIN + LOGICAL_X_MAX) / 2

    # Linear interpolation for Y
    if mlb_y_max - mlb_y_min > 0:
        logical_y = LOGICAL_Y_MIN + (mlb_y - mlb_y_min) * (
            (LOGICAL_Y_MAX - LOGICAL_Y_MIN) / (mlb_y_max - mlb_y_min)
        )
    else:
        logical_y = (LOGICAL_Y_MIN + LOGICAL_Y_MAX) / 2

    return (logical_x, logical_y)


# -------------------------------------------------------
# Affine Transform Calculation
# -------------------------------------------------------

def mlb_to_logical_affine(mlb_points: list, logical_points: list) -> Optional[np.ndarray]:
    """
    Compute an affine transform matrix that maps MLB coordinates
    into logical coordinates, using at least 3 matched reference points.

    Args:
        mlb_points: List of MLB coordinate pairs [(x1, y1), (x2, y2), (x3, y3)]
        logical_points: List of corresponding logical points

    Returns:
        3x3 affine transform matrix, or None if computation fails.
    """
    if len(mlb_points) < 3 or len(logical_points) < 3:
        return None

    mlb_pts = np.array(mlb_points[:3], dtype=np.float64)
    logical_pts = np.array(logical_points[:3], dtype=np.float64)

    # Build linear system for affine parameters
    # x' = a*x + b*y + tx
    # y' = c*x + d*y + ty
    A = np.zeros((6, 6))
    b = np.zeros(6)

    for i in range(3):
        x_mlb, y_mlb = mlb_pts[i]
        x_log, y_log = logical_pts[i]

        # Equation for x':
        A[i * 2, 0] = x_mlb
        A[i * 2, 1] = y_mlb
        A[i * 2, 2] = 1
        b[i * 2] = x_log

        # Equation for y':
        A[i * 2 + 1, 3] = x_mlb
        A[i * 2 + 1, 4] = y_mlb
        A[i * 2 + 1, 5] = 1
        b[i * 2 + 1] = y_log

    try:
        params = np.linalg.solve(A, b)

        # Construct 3x3 affine matrix
        transform = np.array([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [0, 0, 1]
        ])
        return transform
    except np.linalg.LinAlgError:
        return None


def apply_mlb_to_logical_transform(
    mlb_x: float,
    mlb_y: float,
    transform: np.ndarray
) -> Tuple[float, float]:
    """
    Apply an affine transform matrix to a single MLB coordinate.

    Args:
        mlb_x, mlb_y: MLB input coordinates.
        transform: Affine transform matrix (3x3).

    Returns:
        (logical_x, logical_y)
    """
    mlb_point = np.array([mlb_x, mlb_y, 1])
    logical_point = transform @ mlb_point
    return (float(logical_point[0]), float(logical_point[1]))


# -------------------------------------------------------
# Mapper Class
# -------------------------------------------------------

class MLBToLogicalMapper:
    """Affine-only mapper that converts MLB coordinates into logical coordinates."""

    def __init__(self):
        """Initialize mapper with no transform loaded."""
        self.transform = None

    def fit_from_excel_grid(self, mlb_x_values: list, mlb_y_values: list):
        """
        Compute affine transform using three representative points from MLB data.

        Algorithm:
            1. Sort MLB coordinates by X.
            2. Divide data into LF / CF / RF thirds.
            3. Use each segment's median as its representative point.
            4. Map these points to fixed logical LF/CF/RF label coordinates.

        Args:
            mlb_x_values: List of MLB x-coordinates.
            mlb_y_values: List of MLB y-coordinates.
        """
        if len(mlb_x_values) < 3 or len(mlb_y_values) < 3:
            raise ValueError("At least 3 MLB coordinates are required.")

        mlb_coords = list(zip(mlb_x_values, mlb_y_values))
        mlb_coords.sort(key=lambda p: p[0])  # sort by X

        n = len(mlb_coords)
        lf_coords = mlb_coords[: n // 3]
        cf_coords = mlb_coords[n // 3 : 2 * n // 3]
        rf_coords = mlb_coords[2 * n // 3 :]

        def representative(coords):
            if not coords:
                return None
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            return (float(np.median(xs)), float(np.median(ys)))

        mlb_lf = representative(lf_coords)
        mlb_cf = representative(cf_coords)
        mlb_rf = representative(rf_coords)

        if mlb_lf is None or mlb_cf is None or mlb_rf is None:
            raise ValueError("Unable to partition MLB coordinates into LF/CF/RF regions.")

        mlb_points = [mlb_lf, mlb_cf, mlb_rf]

        logical_points = [
            IMAGE_LABELS_LOGICAL["LF"],
            IMAGE_LABELS_LOGICAL["CF"],
            IMAGE_LABELS_LOGICAL["RF"],
        ]

        self.transform = mlb_to_logical_affine(mlb_points, logical_points)

        if self.transform is None:
            raise ValueError("Failed to compute affine transform (points may be collinear).")

    def fit_from_reference_points(self, reference_points: list):
        """
        Compute affine transform explicitly from reference point pairs.

        Args:
            reference_points: List of tuples
                [(mlb_x1, mlb_y1, logical_x1, logical_y1), ...]
                Must contain at least 3 mappings.
        """
        if len(reference_points) < 3:
            raise ValueError("At least 3 reference points are required.")

        mlb_pts = [(p[0], p[1]) for p in reference_points]
        logical_pts = [(p[2], p[3]) for p in reference_points]

        self.transform = mlb_to_logical_affine(mlb_pts, logical_pts)

        if self.transform is None:
            raise ValueError("Failed to compute affine transform")

    def transform_point(self, mlb_x: float, mlb_y: float) -> Tuple[float, float]:
        """
        Transform a single MLB coordinate into logical space.

        Returns:
            (logical_x, logical_y)
        """
        if self.transform is None:
            raise ValueError("You must call fit_from_excel_grid() or fit_from_reference_points() first.")

        return apply_mlb_to_logical_transform(mlb_x, mlb_y, self.transform)