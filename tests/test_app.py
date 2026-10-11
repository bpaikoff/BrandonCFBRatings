from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from cfbratings.data import contrast_color  # noqa: E402
from cfbratings.viz import THEMES, contrast_ratio  # noqa: E402
from cfbratings.viz.charts import MIN_MARK_CONTRAST  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "apps" / "streamlit_app.py"
RANK_CHANGE = re.compile(r"^(▲ \d+|▼ \d+|–|new)$")
CHIP_STYLE = re.compile(r'class="chip" style="background:(#[0-9a-f]{6});color:(#[0-9a-f]{6})"')


def _run(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, at.exception
    return at


def _rankings_table(at: AppTest):
    """The first dataframe on the page is the rankings table."""
    return at.dataframe[0].value


def _frame_with(at: AppTest, column: str):
    """The (first) dataframe on the page that has ``column``."""
    return next(df.value for df in at.dataframe if column in df.value.columns)


def _weeks_slider(at: AppTest):
    return next(s for s in at.sidebar.slider if s.label == "Demo: weeks played")


def _widget(elements, label: str):
    return next(w for w in elements if w.label == label)


def _chart_titles(at: AppTest) -> list[str]:
    return [
        json.loads(c.proto.spec).get("layout", {}).get("title", {}).get("text") or "" for c in at.get("plotly_chart")
    ]


def _no_none_text(df) -> bool:
    """No cell of the display frame reads "None" or "nan" (what st.dataframe prints for a gap)."""
    return not df.astype(str).isin({"None", "nan", "NaN", "<NA>"}).any().any()


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

    # The calibration tile spells its source out; the chart title names the sort column as a phrase.
    assert any("cross-validated" in m.value for m in at.markdown)
    assert "Top 25 by rating" in _chart_titles(at)

    # Matchup chips: the fill is checked against the page surface and the text colour against the fill.
    chips = [m for m in at.markdown if 'class="chips"' in m.value]
    assert len(chips) == 1
    styles = CHIP_STYLE.findall(chips[0].value)
    assert len(styles) == 2
    for bg, fg in styles:
        assert contrast_ratio(bg, THEMES["light"]["surface"]) >= MIN_MARK_CONTRAST
        assert fg == contrast_color(bg)

    # Backtest summary tables carry no Vegas / against-the-spread columns when no lines were loaded,
    # and the out-of-sample predictions are the curated view (the CSV keeps the raw frame).
    for df in by_season:
        assert not {"vegas_n", "vegas_mae", "ats_accuracy"} & set(df.columns)
    oos = _frame_with(at, "pick")
    assert list(oos.columns) == [
        "method", "season", "week", "home", "away", "neutral", "pred_margin", "p_home", "actual_margin", "result", "pick"
    ]  # fmt: skip
    assert set(oos["neutral"]) <= {"n", ""} and set(oos["result"]) <= {"W", "L", "T", ""}
    assert set(oos["pick"]) <= {"✓", "✗", ""} and (oos["p_home"] > 1).any()
    assert _no_none_text(oos)


@pytest.mark.slow
def test_dashboard_preseason_and_completed_season_states():
    at = _run(AppTest.from_file(str(APP), default_timeout=180))

    # Preseason: no game played, so no snapshots and no earlier week to compare ranks with.
    _weeks_slider(at).set_value(0)
    _run(at)
    assert any("preseason" in c.value for c in at.caption)
    table = _rankings_table(at)
    assert set(table["change"]) == {"–"}
    # Nobody has played, so SOS is the mean rating of the scheduled opponents (numeric and ranked
    # for every team, so header sorting stays numeric), momentum is 0 everywhere, and the scatter
    # is shown with a caption saying what it plots.
    assert table["sos"].dtype.kind == "f" and table["sos"].notna().all() and table["sos_rank"].notna().all()
    assert (table["momentum"] == 0).all()
    assert _no_none_text(table)
    assert any("scheduled opponents" in c.value for c in at.caption)
    tiles = next(m.value for m in at.markdown if "SOS rank" in m.value)
    assert "scheduled opponents" in tiles and "no game yet" in tiles

    # Every game played: nothing left to project and no next-game tile, but nothing raises either.
    _weeks_slider(at).set_value(13)
    _run(at)
    assert any("still to play" not in c.value for c in at.caption)
    assert any("Every scheduled game has been played" in i.value for i in at.info)
    # Projections: records are final, the next-game columns are gone and no cell prints "None".
    proj = _frame_with(at, "proj_record")
    assert set(proj["proj_record"]) == {"Final"}
    assert not [c for c in proj.columns if c.startswith("next_")] and "remaining" not in proj.columns
    assert _no_none_text(proj)
    # The rankings table is numeric again, with a value in every SOS / momentum cell.
    table = _rankings_table(at)
    assert table["sos"].dtype.kind == "f" and table["sos"].notna().all() and table["sos_rank"].notna().all()


@pytest.mark.slow
def test_dashboard_matchup_rankings_and_backtest_controls():
    at = _run(AppTest.from_file(str(APP), default_timeout=180))

    # Matchup: the same team on both sides is refused instead of predicting a team against itself.
    home = _widget(at.selectbox, "Home").value
    _widget(at.selectbox, "Away").set_value(home)
    # Rankings: the chart title uses the column's display phrase, not its key.
    _widget(at.selectbox, "Sort by").set_value("ppoints")
    # Backtest: removing every method explains itself instead of showing nothing.
    _widget(at.multiselect, "Methods").set_value([])
    _run(at)

    assert any("Pick two different teams" in w.value for w in at.warning)
    assert not [m for m in at.markdown if 'class="chips"' in m.value]
    assert not [df for df in at.dataframe if "home_margin" in df.value.columns]
    assert "Top 25 by PPoints" in _chart_titles(at)
    assert any("Pick at least one method" in i.value for i in at.info)
    assert not [df for df in at.dataframe if "brier" in df.value.columns]
