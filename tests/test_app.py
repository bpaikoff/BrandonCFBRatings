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
def test_public_dashboard_hides_refresh_and_bounds_seasons(offline, monkeypatch):
    """CFB_PUBLIC_DASHBOARD: no refresh button; seasons limited to the archive plus the current one."""
    import dataclasses

    year = int(offline.year)
    offline.archive_dir.mkdir(parents=True, exist_ok=True)
    for y in (year - 2, year - 1):  # the archive holds two finished seasons
        (offline.archive_dir / f"games_{y}_both.json.gz").write_bytes(b"")
    public = dataclasses.replace(offline, public_dashboard=True)
    monkeypatch.setattr("cfbratings.settings", public)
    monkeypatch.setattr("cfbratings.config.settings", public)

    at = _run(AppTest.from_file(str(APP), default_timeout=180))
    assert not [b for b in at.sidebar.button if b.label == "Refresh API cache"]
    season = _widget(at.sidebar.number_input, "Season")
    assert (season.proto.min, season.proto.max, season.value) == (year - 2, year, year)
    warmup = _widget(at.sidebar.number_input, "Warm-start from prior seasons")
    assert (warmup.proto.max, warmup.value) == (2, 1)  # cannot warm-start from below the archive
    assert [tab.label for tab in at.tabs]  # the rest of the page still renders


@pytest.mark.slow
def test_dashboard_runs_in_demo_mode():
    at = _run(AppTest.from_file(str(APP), default_timeout=180))
    assert at.sidebar.radio[0].value.startswith("Demo")  # no key in tests, so the dashboard never reaches CFBD
    labels = [tab.label for tab in at.tabs]
    assert any("Rankings" in label for label in labels) and any("Projections" in label for label in labels)
    assert any("Playoff" in label for label in labels)
    page = " ".join(m.value for m in at.markdown)
    assert "First round" in page and "National Championship" in page  # the bracket is drawn
    assert any("Projected bracket" in m.value for m in at.markdown)  # and the simulated one
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


def test_load_data_cache_expires_with_the_disk_cache():
    """The public site has no refresh button and constant load_data arguments, so the Streamlit
    cache must carry a ttl or the first fetch after a deploy is served forever (regression)."""
    import ast

    tree = ast.parse(APP.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_data")
    deco = next(d for d in fn.decorator_list if isinstance(d, ast.Call) and ast.unparse(d.func) == "st.cache_data")
    ttl = next((kw.value for kw in deco.keywords if kw.arg == "ttl"), None)
    assert ttl is not None, "load_data's st.cache_data needs a ttl"
    assert "settings.cache_ttl_hours" in ast.unparse(ttl)


@pytest.mark.slow
def test_playoff_tab_season_projection_and_playoff_only_views():
    """The Playoff tab's new sections render in every view: the season projection sub-tabs, a custom
    field, and each playoff-only view (chalk, exact odds, Monte Carlo, one simulated playoff)."""
    at = AppTest.from_file(str(APP), default_timeout=300)
    _run(at)
    _widget(at.sidebar.radio, "Data source").set_value("Demo (synthetic season)")
    _weeks_slider(at).set_value(9)
    _run(at)
    labels = [tab.label for tab in at.tabs]
    for label in ("Remaining games", "Projected records", "Title games", "Playoff odds", "Seeds & win totals"):
        assert label in labels
    field = _widget(at.radio, "Field")
    assert field.options == ["Current bracket", "Projected bracket", "Custom field"]
    show = _widget(at.radio, "Show")
    for view in ("Exact odds", "Monte Carlo", "One simulated playoff", "Chalk bracket"):
        _widget(at.radio, "Show").set_value(view)
        _run(at)
    show = _widget(at.radio, "Show")
    assert show.value == "Chalk bracket"
    # a hand-picked field of six: the bracket still plays out
    _widget(at.radio, "Field").set_value("Custom field")
    _run(at)
    picker = next(m for m in at.multiselect if m.label.startswith("Teams in seed order"))
    assert len(picker.value) == 12
    picker.set_value(picker.value[:6])
    _run(at)
    _widget(at.radio, "Field").set_value("Projected bracket")
    _run(at)
    # the game-projection table reacts to its filters
    week = _widget(at.selectbox, "Week")
    week.set_value(int(week.options[1].split()[-1]))  # options are the formatted labels, values the week numbers
    _run(at)
    assert _widget(at.selectbox, "Week").value != "All"
    # scores appear in the matchup tab too
    assert any("Projected score" in str(md.value) for md in at.markdown)
