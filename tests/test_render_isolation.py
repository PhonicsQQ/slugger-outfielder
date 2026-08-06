"""Concurrent renders must not hand back each other's chart.

``plt.savefig`` writes pyplot's *current* figure, which is process-global. The
worker runs ``--workers 1 --threads 4``, so two coaches computing at the same
moment could each get the other's spray chart — returned ``ok: true``, under the
right hitter's name, showing the wrong hitter's batted balls. For a widget whose
only output is where to stand, that is the worst shape a bug can take: entirely
plausible and silently wrong.

Bound saves (``fig.savefig``) are the fix; these tests pin it both ways —
functionally, by rendering concurrently and comparing against the serial output,
and at the source level, so a future edit cannot quietly reintroduce ``plt.savefig``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

import app


def _spray(seed: int, n: int = 25) -> pd.DataFrame:
    """A small, deterministic batted-ball sample."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.uniform(-150, 150, n),
        "y": rng.uniform(20, 380, n),
        "direction": rng.uniform(-45, 45, n),
        "distance": rng.uniform(150, 400, n),
        "outcome": rng.choice(["1B", "2B", "OUT"], n),
    })


def _digest(png_b64: str) -> str:
    return hashlib.sha256(base64.b64decode(png_b64)).hexdigest()


@pytest.fixture
def hitters() -> list[tuple[str, pd.DataFrame]]:
    return [(f"Hitter {i}", _spray(seed=i)) for i in range(4)]


def test_concurrent_renders_return_their_own_chart(hitters) -> None:
    """Rendered in parallel, each hitter's chart must match its serial render."""
    def render(item: tuple[str, pd.DataFrame]) -> str:
        label, df = item
        return app.make_plot(df.copy(), None, label, "RHP")

    serial = [_digest(render(h)) for h in hitters]

    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = [_digest(png) for png in pool.map(render, hitters)]

    assert concurrent == serial, (
        "a concurrent render returned a different chart than the serial one — "
        "pyplot's current figure is shared across threads"
    )


def test_every_chart_in_the_batch_is_distinct(hitters) -> None:
    """Four different hitters, four different charts — no swaps, no duplicates."""
    def render(item: tuple[str, pd.DataFrame]) -> str:
        label, df = item
        return app.make_plot(df.copy(), None, label, "RHP")

    with ThreadPoolExecutor(max_workers=4) as pool:
        digests = [_digest(png) for png in pool.map(render, hitters)]

    assert len(set(digests)) == len(hitters)


def test_no_unbound_pyplot_saves_remain() -> None:
    """`plt.savefig` saves whatever figure is current — never use it here."""
    source = pathlib.Path(app.__file__).read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if re.search(r"\bplt\.savefig\b", line) and not line.strip().startswith("#")
    ]

    assert offenders == [], f"unbound pyplot saves: {offenders}"
