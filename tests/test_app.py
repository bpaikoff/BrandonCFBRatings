from __future__ import annotations

from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "apps" / "streamlit_app.py"


@pytest.mark.slow
def test_dashboard_runs_in_demo_mode():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    assert any("Rankings" in t for t in at.tabs[0].label.split()) or at.tabs
    assert at.title[0].value.startswith("College Football Ratings")
