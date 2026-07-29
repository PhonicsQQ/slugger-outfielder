"""Tests for coach-standard outfielder placement constraints (Commit 1).

The raw depth-weighted centroid can drop a fielder in a physically absurd
spot (e.g. CF pulled 30 ft off the line, two fielders on top of each other).
``compute_constrained_positions`` sanity-clamps each placement into a coach
alignment window (LF/CF/RF anchors, bounded angular deviation, depth bounds)
and enforces a minimum inter-fielder separation, then reports which fielders
were moved so the chart can footnote the adjustment.

Inputs are constructed with the exact forward mapping (``angle_dist_to_pixel``)
so the polar round-trip is loss-free and assertions can be made in angle/dist
space.
"""

import itertools

import numpy as np
import pytest

import app


# ── helpers ──────────────────────────────────────────────────────────────

def _win(name):
    """The intersected angle window [lo, hi] for a fielder (test oracle)."""
    anchor = app.OF_ANGLE_ANCHORS[name]
    dev = app.OF_MAX_ANGLE_DEVIATION
    cfg = app.SPRAY_PIXEL_CONFIG
    return (max(anchor - dev, cfg["dir_min"]), min(anchor + dev, cfg["dir_max"]))


def _angle_of(px_py):
    """Recover the angle of a back-converted pixel placement."""
    return app.pixel_to_angle_dist(px_py[0], px_py[1])[0]


def _dist_of(px_py):
    return app.pixel_to_angle_dist(px_py[0], px_py[1])[1]


def _raw_oracle(balls_pixel):
    """Verbatim copy of the pre-refactor inline centroid logic (lines 650-671)."""
    optimized_pixel = {}
    if balls_pixel:
        sorted_dots = sorted(balls_pixel, key=lambda d: d[0])
        third = max(1, len(sorted_dots) // 3)
        for name, dots in [("LF", sorted_dots[:third]),
                           ("CF", sorted_dots[third:2 * third]),
                           ("RF", sorted_dots[2 * third:])]:
            if dots:
                total_w = 0.0
                wx_sum = 0.0
                for (px, py, _, dist) in dots:
                    w = app._get_depth_weight(dist)
                    wx_sum += w * px
                    total_w += w
                avg_x = wx_sum / total_w
                ys = [py for (_, py, _, _) in dots]
                depth_y = float(np.percentile(ys, app.DEPTH_POSITION_PERCENTILE))
                optimized_pixel[name] = (avg_x, depth_y)
    return optimized_pixel


# ── 1. round-trip property ────────────────────────────────────────────────

def test_pixel_angle_roundtrip_is_lossless():
    for px in range(300, 2101, 200):
        for py in range(720, 931, 30):
            angle, dist = app.pixel_to_angle_dist(px, py)
            px2, py2 = app.angle_dist_to_pixel(angle, dist)
            assert px2 == pytest.approx(px, abs=1e-6)
            assert py2 == pytest.approx(py, abs=1e-6)


# ── 2. sane triple passes through untouched ────────────────────────────────

def test_sane_triple_passthrough_nothing_engaged():
    raw = {
        "LF": app.angle_dist_to_pixel(-27.0, 285.0),
        "CF": app.angle_dist_to_pixel(0.0, 300.0),
        "RF": app.angle_dist_to_pixel(27.0, 285.0),
    }
    clamped, report = app.compute_constrained_positions(raw)
    for name in ("LF", "CF", "RF"):
        assert report[name]["engaged"] is False
        assert clamped[name][0] == pytest.approx(raw[name][0], abs=1e-6)
        assert clamped[name][1] == pytest.approx(raw[name][1], abs=1e-6)


# ── 3. angular clamp (mapping edge, not raw anchor±dev) ─────────────────────

def test_angular_clamp_respects_mapping_edge():
    raw = {
        "LF": app.angle_dist_to_pixel(-45.0, 300.0),   # beyond both anchor & mapping
        "CF": app.angle_dist_to_pixel(-20.0, 300.0),   # beyond CF window
        "RF": app.angle_dist_to_pixel(27.0, 300.0),
    }
    clamped, _ = app.compute_constrained_positions(raw)
    # LF pinned to the mapping edge -38, NOT the raw anchor-13 = -40.
    assert _angle_of(clamped["LF"]) == pytest.approx(-38.0, abs=1e-6)
    assert _angle_of(clamped["CF"]) == pytest.approx(-13.0, abs=1e-6)


# ── 4. depth clamp ─────────────────────────────────────────────────────────

def test_depth_clamp_to_bounds():
    raw = {
        "LF": app.angle_dist_to_pixel(-27.0, 360.0),   # over LF hi 325
        "CF": app.angle_dist_to_pixel(0.0, 380.0),     # over CF hi 345
        "RF": app.angle_dist_to_pixel(27.0, 200.0),    # under RF lo 240
    }
    clamped, _ = app.compute_constrained_positions(raw)
    assert _dist_of(clamped["LF"]) == pytest.approx(325.0, abs=1e-6)
    assert _dist_of(clamped["CF"]) == pytest.approx(345.0, abs=1e-6)
    assert _dist_of(clamped["RF"]) == pytest.approx(240.0, abs=1e-6)


# ── 5. separation push-apart is symmetric ──────────────────────────────────

def test_separation_symmetric_about_midpoint():
    raw = {
        "LF": app.angle_dist_to_pixel(-14.0, 300.0),
        "CF": app.angle_dist_to_pixel(-13.0, 300.0),
    }
    clamped, _ = app.compute_constrained_positions(raw)
    a_lf = _angle_of(clamped["LF"])
    a_cf = _angle_of(clamped["CF"])
    assert a_cf - a_lf >= app.OF_MIN_SEPARATION_DEG - 1e-6
    lf_lo, lf_hi = _win("LF")
    cf_lo, cf_hi = _win("CF")
    assert lf_lo - 1e-6 <= a_lf <= lf_hi + 1e-6
    assert cf_lo - 1e-6 <= a_cf <= cf_hi + 1e-6
    # symmetric about the original midpoint (-13.5)
    assert (a_lf + a_cf) / 2 == pytest.approx(-13.5, abs=1e-6)


# ── 6. degenerate all-in-one-spot fans out to three ordered fielders ───────

def test_degenerate_single_spot_fans_out():
    spot = app.angle_dist_to_pixel(30.0, 380.0)
    raw = {"LF": spot, "CF": spot, "RF": spot}
    clamped, _ = app.compute_constrained_positions(raw)
    a = {n: _angle_of(clamped[n]) for n in ("LF", "CF", "RF")}
    # three distinct, strictly ordered placements
    assert a["LF"] < a["CF"] < a["RF"]
    assert a["CF"] - a["LF"] >= app.OF_MIN_SEPARATION_DEG - 1e-6
    assert a["RF"] - a["CF"] >= app.OF_MIN_SEPARATION_DEG - 1e-6
    for n in ("LF", "CF", "RF"):
        lo, hi = _win(n)
        assert lo - 1e-6 <= a[n] <= hi + 1e-6


# ── 7. missing fielders ────────────────────────────────────────────────────

def test_missing_fielders_handled():
    # A single fielder is fine — no separation pass, no crash.
    clamped, report = app.compute_constrained_positions(
        {"LF": app.angle_dist_to_pixel(-27.0, 300.0)})
    assert set(clamped) == {"LF"}
    assert "LF" in report
    # Empty input round-trips to empty outputs.
    assert app.compute_constrained_positions({}) == ({}, {})


# ── 8. disabled flag is an identity ────────────────────────────────────────

def test_disabled_flag_identity(monkeypatch):
    monkeypatch.setattr(app, "OF_CONSTRAINTS_ENABLED", False)
    raw = {
        "LF": app.angle_dist_to_pixel(-45.0, 380.0),   # would clamp if enabled
        "CF": app.angle_dist_to_pixel(30.0, 200.0),
        "RF": app.angle_dist_to_pixel(35.0, 390.0),
    }
    clamped, report = app.compute_constrained_positions(raw)
    assert clamped == raw
    assert report == {}


# ── 9. invariants over a dense angle/depth grid ────────────────────────────

def test_invariants_over_grid():
    angles = [-38 + 7.6 * k for k in range(11)]  # -38 .. 38
    for a_lf, a_cf, a_rf in itertools.product(angles, repeat=3):
        for depth in (160.0, 300.0, 390.0):
            raw = {
                "LF": app.angle_dist_to_pixel(a_lf, depth),
                "CF": app.angle_dist_to_pixel(a_cf, depth),
                "RF": app.angle_dist_to_pixel(a_rf, depth),
            }
            clamped, report = app.compute_constrained_positions(raw)
            assert set(clamped) == {"LF", "CF", "RF"}
            out = {n: app.pixel_to_angle_dist(*clamped[n]) for n in clamped}
            # in-window angle + in-bounds depth for every fielder
            for n in ("LF", "CF", "RF"):
                lo, hi = _win(n)
                assert lo - 1e-6 <= out[n][0] <= hi + 1e-6
                dlo, dhi = app.OF_DEPTH_BOUNDS[n]
                assert dlo - 1e-6 <= out[n][1] <= dhi + 1e-6
                assert np.isfinite(clamped[n][0]) and np.isfinite(clamped[n][1])
            # disjoint windows guarantee structural left→right ordering
            assert out["LF"][0] <= out["CF"][0] + 1e-6 <= out["RF"][0] + 2e-6


# ── 10. raw centroid extraction is behaviour-identical ─────────────────────

def test_compute_raw_positions_matches_old_inline_formula():
    rng = np.random.default_rng(7)
    balls_pixel = []
    for _ in range(18):
        px = float(rng.integers(300, 2100))
        py = float(rng.integers(720, 930))
        dist = float(rng.uniform(150, 400))
        balls_pixel.append((px, py, "#ffffff", dist))
    got = app.compute_raw_positions(balls_pixel)
    expected = _raw_oracle(balls_pixel)
    assert set(got) == set(expected)
    for name in expected:
        assert got[name][0] == pytest.approx(expected[name][0], abs=1e-9)
        assert got[name][1] == pytest.approx(expected[name][1], abs=1e-9)


# ── 11. integration: make_plot_with_image invokes the constraint pass ──────

def test_make_plot_with_image_invokes_constraints(monkeypatch):
    import pandas as pd

    n = 15
    directions = np.linspace(-38.0, 38.0, n)
    df = pd.DataFrame({
        "x": np.linspace(-100, 100, n),
        "y": np.linspace(50, 300, n),
        "direction": directions,
        "distance": np.full(n, 390.0),   # deep — forces extreme placement
        "outcome": ["OUT"] * n,
    })

    calls = {"n": 0}
    real = app.compute_constrained_positions

    def _spy(raw):
        calls["n"] += 1
        return real(raw)

    monkeypatch.setattr(app, "compute_constrained_positions", _spy)

    b64 = app.make_plot_with_image(df, batter_label="Extreme", pitcher_hand="RHP")
    assert isinstance(b64, str) and len(b64) > 0
    assert calls["n"] >= 1


# ── 12. config validation rejects overlapping windows ──────────────────────

def test_validate_rejects_overlapping_windows(monkeypatch):
    # dev=20 → LF window hi -7 overlaps CF window lo -20.
    monkeypatch.setattr(app, "OF_MAX_ANGLE_DEVIATION", 20.0)
    with pytest.raises(ValueError):
        app._validate_of_constraints()
