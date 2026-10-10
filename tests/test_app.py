from __future__ import annotations

import re
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "apps" / "streamlit_app.py"
RANK_CHANGE = re.compile(r"^(▲ \d+|▼ \d+|–|new)$")


def _run(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, at.exception
    return at


def _rankings_table(at: AppTest):
    """The first dataframe on the page is the rankings table."""
    return at.dataframe[0].value


def _weeks_slider(at: AppTest):
    return next(s for s in at.sidebar.slider if s.label == "Demo: weeks played")


@pytest.mark.slow
def test_dashboard_runs_in_demo_mode():
    at = _run(AppTest.from_file(str(APP), default_timeout=180))
    assert at.sidebar.radio[0].value.startswith("Demo")  # no key in tests, so the dashboard never reaches CFBD
    labels = [tab.label for tab in at.tabs]
    assert any("Rankings" in label for label in labels) and any("Projections" in label for label in labels)
    assert at.title[0].value.startswith("College Football Ratings")
    assert any("warm-started" in c.value for c in at.caption)

    # Sidebar provenance: synthetic data, and the API-call counter reads zero in demo mode.
    sidebar_captions = [c.value for c in at.sidebar.caption]
    assert any("Synthetic data" in c for c in sidebar_captions)
    assert any("API calls this session: 0" in c for c in sidebar_captions)

    # Rankings: logo and rank-change columns sit before the team name; every change is a valid label.
    table = _rankings_table(at)
    cols = list(table.columns)
    assert cols.index("change") < cols.index("logo") < cols.index("team")
    assert all(RANK_CHANGE.match(v) for v in table["change"])
    assert (table["logo"] == "").all()  # synthetic teams carry colours but no logos: blank, never "None"
    assert "Colour by" in [r.label for r in at.radio]

    # Backtest "by season" tables use the same percent scaling as the headline table.
    by_season = [df.value for df in at.dataframe if "season" in df.value.columns and "accuracy" in df.value.columns]
    assert by_season and all((df["accuracy"] > 1).all() for df in by_season)


@pytest.mark.slow
def test_dashboard_preseason_and_completed_season_states():
    at = _run(AppTest.from_file(str(APP), default_timeout=180))

    # Preseason: no game played, so no snapshots and no earlier week to compare ranks with.
    _weeks_slider(at).set_value(0)
    _run(at)
    assert any("preseason" in c.value for c in at.caption)
    table = _rankings_table(at)
    assert set(table["change"]) == {"–"}

    # Every game played: nothing left to project and no next-game tile, but nothing raises either.
    _weeks_slider(at).set_value(13)
    _run(at)
    assert any("still to play" not in c.value for c in at.caption)
    assert any("Every scheduled game has been played" in i.value for i in at.info)
