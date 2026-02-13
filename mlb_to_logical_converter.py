#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB Coordinate â†’ Logical Coordinate Converter

Logical coordinate system definition:
    X range: -89.50 ~ 52.66
    Y range: -43.85 ~ -14.18

Reference logical label points:
    LF = (-60, 20)
    CF = (-20, 25)
    RF = (20, 20)

This module provides a simple scaling-based method for converting MLB hit
coordinates into the logical coordinate system used by the optimizer.

Note:
    A more precise conversion is done via affine transforms in other modules.
    This version serves as a robust fallback that guarantees all outputs remain
    within logical coordinate bounds.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

# Logical coordinate bounds for visualization mapping
# X bounds match the outfield polygon width
# Y bounds cover the full visible outfield: from the fence (pixel ~750, logical ~32)
# down to the infield edge (pixel ~1222, logical ~-14)
LOGICAL_X_MIN = -100.00
LOGICAL_X_MAX = 63.00
LOGICAL_Y_MIN = 5.00
LOGICAL_Y_MAX = 50.00

# Logical reference points (fixed)
LOGICAL_LF = (-60.0, 20.0)
LOGICAL_CF = (-20.0, 25.0)
LOGICAL_RF = (20.0, 20.0)


def mlb_to_logical_simple_scale(
    mlb_x: float, mlb_y: float,
    mlb_x_range: Tuple[float, float],
    mlb_y_range: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Convert MLB coordinates into logical coordinates using simple linear scaling.

    This method is not perfectly accurate, but it guarantees that resulting
    logical coordinates fall within the valid logical coordinate bounds.

    Args:
        mlb_x, mlb_y: MLB coordinates
        mlb_x_range: Tuple (min_x, max_x) of MLB x-values
        mlb_y_range: Tuple (min_y, max_y) of MLB y-values

    Returns:
        (logical_x, logical_y) â€” clamped within logical coordinate limits.
    """
    mlb_x_min, mlb_x_max = mlb_x_range
    mlb_y_min, mlb_y_max = mlb_y_range

    # Linear interpolation for X
    if mlb_x_max - mlb_x_min > 0:
        logical_x = LOGICAL_X_MIN + ((mlb_x - mlb_x_min) * 
                    (LOGICAL_X_MAX - LOGICAL_X_MIN) / (mlb_x_max - mlb_x_min))
    else:
        logical_x = (LOGICAL_X_MIN + LOGICAL_X_MAX) / 2

    # Linear interpolation for Y
    if mlb_y_max - mlb_y_min > 0:
        logical_y = LOGICAL_Y_MIN + ((mlb_y - mlb_y_min) * 
                    (LOGICAL_Y_MAX - LOGICAL_Y_MIN) / (mlb_y_max - mlb_y_min))
    else:
        logical_y = (LOGICAL_Y_MIN + LOGICAL_Y_MAX) / 2

    # Clamp to valid logical coordinate bounds
    logical_x = max(LOGICAL_X_MIN, min(LOGICAL_X_MAX, logical_x))
    logical_y = max(LOGICAL_Y_MIN, min(LOGICAL_Y_MAX, logical_y))

    return (logical_x, logical_y)


def convert_dataframe_mlb_to_logical(
    df: pd.DataFrame,
    mlb_x_col: str = "x",
    mlb_y_col: str = "y"
):
    """
    Convert MLB coordinates in an entire DataFrame into logical coordinates.

    Args:
        df: Input DataFrame containing MLB coordinate columns
        mlb_x_col: Name of MLB x-coordinate column
        mlb_y_col: Name of MLB y-coordinate column

    Returns:
        DataFrame where the specified x/y columns are replaced with their
        logical coordinate transformations.
    """
    df_logical = df.copy()

    mlb_x_values = df[mlb_x_col].dropna().tolist()
    mlb_y_values = df[mlb_y_col].dropna().tolist()

    # If no coordinate data is available, return unchanged DataFrame
    if not mlb_x_values or not mlb_y_values:
        return df_logical

    # Compute overall MLB coordinate range
    mlb_x_range = (min(mlb_x_values), max(mlb_x_values))
    mlb_y_range = (min(mlb_y_values), max(mlb_y_values))

    # Convert each row
    logical_coords = []
    for _, row in df.iterrows():
        mlb_x = row[mlb_x_col]
        mlb_y = row[mlb_y_col]

        if pd.isna(mlb_x) or pd.isna(mlb_y):
            logical_coords.append((None, None))
        else:
            logical_coords.append(
                mlb_to_logical_simple_scale(
                    mlb_x, mlb_y,
                    mlb_x_range, mlb_y_range
                )
            )

    # Replace columns with logical values
    df_logical[mlb_x_col] = [x for x, _ in logical_coords]
    df_logical[mlb_y_col] = [y for _, y in logical_coords]

    return df_logical