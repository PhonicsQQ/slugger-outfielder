"""The index route actually renders.

templates/index.html is a Jinja template that also carries the React source, so
a JSX inline style — ``style={{ ... }}`` — is parsed by Jinja as a print
statement and takes the whole page down with a 500 while every API route keeps
answering 200. That shipped once; the rest of the suite never rendered the
template, so nothing caught it. Styling belongs in a className.
"""

import os
import re

import pytest

os.environ.setdefault("USE_API_MODE", "true")

import app as app_module  # noqa: E402


@pytest.fixture()
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_index_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"<div id=\"root\"></div>" in res.data or b"root" in res.data


def test_template_has_no_jsx_double_brace():
    """Catch the Jinja/JSX collision at its source, not just via a 500."""
    path = os.path.join(os.path.dirname(app_module.__file__), "templates", "index.html")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    offenders = [
        line for line in source.splitlines() if re.search(r"=\{\{", line)
    ]
    assert not offenders, (
        "JSX double-brace expression in a Jinja template — use a className:\n"
        + "\n".join(offenders)
    )
