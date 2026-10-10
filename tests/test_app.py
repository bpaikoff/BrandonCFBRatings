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
    assert at.sidebar.radio[0].value.startswith("Demo")  # no key in tests, so the dashboard never reaches CFBD
    labels = [tab.label for tab in at.tabs]
    assert any("Rankings" in label for label in labels) and any("Projections" in label for label in labels)
    assert at.title[0].value.startswith("College Football Ratings")
    assert any("warm-started" in c.value for c in at.caption)
