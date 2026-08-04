"""Tests for the switch-hitter batting-side annotation on the chart title.

The report already splits vs-RHP and vs-LHP on the same page, and a switch
hitter bats left against a RHP and right against a LHP — so the two charts
already are the two permutations the report needs. What was missing is the
disclosure, so the title names the side instead of the report emitting a second
page. The side is read from the spray rows (``batter_side``) rather than the
roster handedness flag, which goes stale when a hitter converts mid-season.

``_batting_side_note`` is exercised directly on tiny hand-made frames, so these
tests never touch matplotlib, the background image, or the data loader.
"""

import pandas as pd

import app


# ── helpers ────────────────────────────────────────────────────────────────

def _spray(sides, dates=None):
    """Minimal spray frame carrying only what the note reads."""
    data = {"batter_side": list(sides)}
    if dates is not None:
        data["date"] = list(dates)
    return pd.DataFrame(data)


# ── 1-2. a clean single-sided sample names that side ───────────────────────

def test_switch_hitter_all_left_reads_lh():
    df = _spray(["Left"] * 20)
    assert app._batting_side_note(df, "Difo, Wilmer (S)") == " — batting LH"


def test_switch_hitter_all_right_reads_rh():
    df = _spray(["Right"] * 20)
    assert app._batting_side_note(df, "Difo, Wilmer (S)") == " — batting RH"


# ── 3. a one-sided roster label already states the side ────────────────────

def test_one_sided_label_is_never_annotated():
    # Across 117 measured one-sided panels the roster label never disagreed
    # with the data, so annotating them would add noise and no information.
    df = _spray(["Left"] * 20)
    assert app._batting_side_note(df, "Doe, Jane (L)") == ""
    assert app._batting_side_note(df, "Doe, John (R)") == ""


# ── 4. frames without the column (synthetic / JSON loader) stay silent ─────

def test_missing_batter_side_column_returns_empty():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    assert app._batting_side_note(df, "Difo, Wilmer (S)") == ""
    assert app._batting_side_note(pd.DataFrame(), "Difo, Wilmer (S)") == ""


# ── 5. a lone straggler does not suppress the note ─────────────────────────

def test_lopsided_sample_still_names_the_dominant_side():
    # Difo's real shape: 86 Left / 1 Right = 0.988, well clear of the threshold.
    df = _spray(["Left"] * 86 + ["Right"])
    assert app._batting_side_note(df, "Difo, Wilmer (S)") == " — batting LH"


# ── 6. null / undefined sides are filtered, not counted ────────────────────

def test_null_and_undefined_sides_are_ignored():
    df = _spray(["Left"] * 20 + [None, "", "Undefined"])
    assert app._batting_side_note(df, "Difo, Wilmer (S)") == " — batting LH"

    empty = _spray([None] * 10)
    assert app._batting_side_note(empty, "Difo, Wilmer (S)") == ""


# ── 7. a genuinely two-sided sample discloses both, current side first ─────

def test_two_sided_sample_leads_with_most_recent_side():
    # Bates's real shape: he stopped switch-hitting on 2026-05-30, so the
    # left-side rows sort first in the frame but describe a retired swing.
    left_dates = pd.date_range("2026-04-24", "2026-05-29", periods=26)
    right_dates = pd.date_range("2026-06-01", "2026-08-01", periods=38)
    df = _spray(
        ["Left"] * 26 + ["Right"] * 38,
        list(left_dates) + list(right_dates),
    )
    assert (app._batting_side_note(df, "Bates, Austin (S)")
            == " — batting RH (38) / LH (26)")


# ── 8. the production date shape — plain strings, some of them missing ──────

def test_two_sided_sample_tolerates_missing_dates():
    # data_loader stores date straight from the feed with no coercion, so the
    # column is plain strings and a row the feed omitted comes through as None.
    # Comparing those together used to raise TypeError and 500 the chart.
    df = _spray(
        ["Left"] * 4 + ["Right"] * 6,
        [None, "2026-05-01", "2026-05-02", None,
         "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04",
         "2026-06-05", None],
    )
    assert (app._batting_side_note(df, "Bates, Austin (S)")
            == " — batting RH (6) / LH (4)")

    # A side the feed dated on no row at all sorts last rather than raising.
    undated = _spray(
        ["Left"] * 4 + ["Right"] * 6,
        [None] * 4 + ["2026-06-0%d" % i for i in range(1, 7)],
    )
    assert (app._batting_side_note(undated, "Bates, Austin (S)")
            == " — batting RH (6) / LH (4)")


# ── 9. the sample-completeness note never reaches the title or the label ────

def test_sample_note_lands_in_a_footnote_not_the_title(monkeypatch):
    """The note says whether the chart stands on the hitter's whole tracked
    sample, and it must stay out of ``batter_label``: _batting_side_note gates
    on the label ending in "(S)", so appending anything to it would silently
    kill the switch-hitter disclosure with no error and no failing test.
    """
    from matplotlib.axes import Axes

    titles, texts = [], []
    orig_title, orig_text = Axes.set_title, Axes.text

    def _title(self, label="", *a, **k):
        titles.append(label)
        return orig_title(self, label, *a, **k)

    def _text(self, x, y, s="", *a, **k):
        texts.append(s)
        return orig_text(self, x, y, s, *a, **k)

    monkeypatch.setattr(Axes, "set_title", _title)
    monkeypatch.setattr(Axes, "text", _text)

    df = pd.DataFrame({
        "x": [10.0] * 20, "y": [200.0] * 20,
        "direction": [-5.0] * 20, "distance": [300.0] * 20,
        "outcome": ["OUT"] * 20, "batter_side": ["Left"] * 20,
    })
    label = "De Aza, Alejandro (S)"
    note = "sample pooled from 2 roster records"

    app.make_plot_with_image(df, positions=None, batter_label=label,
                             pitcher_hand="RHP", sample_note=note)

    assert titles == [f"{label} vs RHP" + app._batting_side_note(df, label)]
    assert note in texts
    assert note not in titles[0]
