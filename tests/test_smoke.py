"""Smoke test: the Flask app module imports cleanly and exposes its core objects.

This is the initial test that establishes the pytest gate in CI. It intentionally
does no network I/O — importing `app` runs in JSON-loader mode locally and in CI
(USE_API_MODE unset), so no API key is required.
"""

import app


def test_app_imports():
    """The Flask application object is constructed at import time."""
    assert app.app is not None


def test_build_player_dict_is_callable():
    """The batter-list builder is importable for the dedup tests to target."""
    assert callable(app.build_player_dict)
