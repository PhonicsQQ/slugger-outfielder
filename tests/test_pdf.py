"""Tests for the PDF report path (Commit 2).

The per-player page draw is extracted into ``_draw_player_report_page`` so the
client-side team merge can reuse the exact same single-page layout, and an
``img=jpeg`` query option re-encodes the embedded charts to shrink team reports.

``load_spray_and_render`` is monkeypatched to a tiny in-memory PNG so these
tests never touch matplotlib, the background image, or the data loader.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

import app


# ── helpers ────────────────────────────────────────────────────────────────

def _png_b64(size=(20, 20), seed=None):
    if seed is None:
        im = Image.new("RGB", size, (200, 30, 30))
    else:
        arr = np.random.default_rng(seed).integers(
            0, 256, (size[1], size[0], 3), dtype=np.uint8)
        im = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _stub_render(b64):
    def _fake(batter_id, pitcher_hand, *args, **kwargs):
        return b64, f"Test Player ({pitcher_hand})"
    return _fake


def _xobject_filters(pdf_bytes):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    filters = []
    for page in reader.pages:
        res = page.get("/Resources")
        if not res or "/XObject" not in res:
            continue
        xobjs = res["/XObject"].get_object()
        for name in xobjs:
            obj = xobjs[name].get_object()
            filt = obj.get("/Filter")
            if filt is not None:
                filters.append(str(filt))
    return filters


@pytest.fixture
def client():
    return app.app.test_client()


# ── 1. single-player PDF: 200, one page, correct disposition ───────────────

def test_single_player_pdf_one_page(client, monkeypatch):
    monkeypatch.setattr(app, "load_spray_and_render", _stub_render(_png_b64()))
    resp = client.get("/api/pdf/somebatter")
    assert resp.status_code == 200
    assert resp.mimetype == "application/pdf"
    cd = resp.headers.get("Content-Disposition", "")
    assert "SLUGGER_" in cd and "_report.pdf" in cd

    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(resp.data))
    assert len(reader.pages) == 1


# ── 2. no data for either hand → 404 JSON contract preserved ───────────────

def test_no_data_both_hands_404(client, monkeypatch):
    def _none(batter_id, pitcher_hand, *a, **k):
        return None, f"no qualifying data ({pitcher_hand})"
    monkeypatch.setattr(app, "load_spray_and_render", _none)

    resp = client.get("/api/pdf/x")
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["ok"] is False
    assert "No data" in data["error"]


# ── 3. img=jpeg embeds DCTDecode; default embeds none ──────────────────────

def test_jpeg_option_uses_dctdecode(client, monkeypatch):
    monkeypatch.setattr(app, "load_spray_and_render", _stub_render(_png_b64()))

    jpeg = client.get("/api/pdf/x?img=jpeg")
    assert jpeg.status_code == 200
    assert any("DCTDecode" in f for f in _xobject_filters(jpeg.data))

    png = client.get("/api/pdf/x")
    assert png.status_code == 200
    assert not any("DCTDecode" in f for f in _xobject_filters(png.data))


# ── 4. jpeg report is smaller than the png report ──────────────────────────

def test_jpeg_report_smaller_than_png(client, monkeypatch):
    monkeypatch.setattr(app, "load_spray_and_render",
                        _stub_render(_png_b64(size=(200, 200), seed=1)))
    png = client.get("/api/pdf/x")
    jpeg = client.get("/api/pdf/x?img=jpeg")
    assert png.status_code == 200 and jpeg.status_code == 200
    assert len(jpeg.data) < len(png.data)


# ── 5. unknown format falls back to png ────────────────────────────────────

def test_unknown_format_falls_back_to_png(client, monkeypatch):
    monkeypatch.setattr(app, "load_spray_and_render", _stub_render(_png_b64()))
    resp = client.get("/api/pdf/x?img=bmp")
    assert resp.status_code == 200
    assert not any("DCTDecode" in f for f in _xobject_filters(resp.data))


# ── 6. _draw_player_report_page draws exactly one page ─────────────────────

def test_draw_player_report_page_single_page(monkeypatch):
    monkeypatch.setattr(app, "load_spray_and_render", _stub_render(_png_b64()))
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rl_canvas
    from pypdf import PdfReader

    page_w, page_h = letter
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=letter)
    drawn, rhp, lhp = app._draw_player_report_page(c, page_w, page_h, "x")
    assert drawn is True
    c.save()
    buf.seek(0)
    reader = PdfReader(io.BytesIO(buf.getvalue()))
    assert len(reader.pages) == 1


# ── 7. a pooled batter draws one page from every member's spray ────────────

def test_pooled_batter_page_resolves_the_whole_union(monkeypatch):
    """Two roster records proven to be one hitter mean two upstream fetches per
    pitcher hand — four for the page — but still exactly one page, and the
    disclosure travels into the PDF for free because the same PNG is embedded.
    """
    fetched = []

    def _fake_fetch(player_id, pitcher_hand=None, start_date=None,
                    end_date=None, limit=5000):
        fetched.append((player_id, pitcher_hand))
        return [{"date": "2025-05-0%d" % len(fetched), "exit_speed": 90.0}]

    rendered = []

    def _fake_plot(df, positions=None, batter_label="", pitcher_hand="RHP",
                   background_image_path=None, sample_note=""):
        rendered.append(sample_note)
        return _png_b64()

    monkeypatch.setattr(app, "fetch_player_spray", _fake_fetch)
    monkeypatch.setattr(app, "make_plot_with_image", _fake_plot)
    monkeypatch.setattr(app, "_union_ids", {"a": ["a", "b"], "b": ["a", "b"]})
    monkeypatch.setattr(
        app, "_sample_notes",
        {"a": "sample pooled from 2 roster records",
         "b": "sample pooled from 2 roster records"})
    monkeypatch.setattr(app, "USE_API_ADAPTER", True)
    monkeypatch.setattr(app, "USE_JSON_LOADER", False)
    monkeypatch.setattr(app, "MIN_QUALIFYING_BALLS", 1)
    monkeypatch.setattr(app, "resolve_batter_meta",
                        lambda *a, **k: {"label": "De Aza, Alejandro (S)"})

    # load_spray_and_render imports the parser from data_loader at call time.
    import data_loader
    monkeypatch.setattr(
        data_loader, "parse_spray_to_dataframe",
        lambda rows: __import__("pandas").DataFrame({
            "x": [1.0] * len(rows), "y": [2.0] * len(rows),
            "hang_time": [None] * len(rows), "outcome": [None] * len(rows),
        }))

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rl_canvas
    from pypdf import PdfReader

    page_w, page_h = letter
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=letter)
    drawn, _, _ = app._draw_player_report_page(c, page_w, page_h, "a")
    c.save()

    assert drawn is True
    assert [pid for pid, _ in fetched] == ["a", "b", "a", "b"]
    assert rendered == ["sample pooled from 2 roster records"] * 2
    reader = PdfReader(io.BytesIO(buf.getvalue()))
    assert len(reader.pages) == 1


# ── 8. pooling must not rename a user-visible artifact ─────────────────────

def test_pdf_download_name_unchanged_for_merged_entry(client, monkeypatch):
    # The filename derives from the label resolve_batter_meta rebuilds, which
    # never carries the team qualifier — so a pooled entry downloads under the
    # same name a coach saw before the merge shipped.
    monkeypatch.setattr(app, "load_spray_and_render", _stub_render(_png_b64()))
    resp = client.get("/api/pdf/somebatter")
    assert resp.status_code == 200
    assert "SLUGGER_Test_Player_report.pdf" in \
        resp.headers.get("Content-Disposition", "")
