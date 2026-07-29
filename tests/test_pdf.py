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
