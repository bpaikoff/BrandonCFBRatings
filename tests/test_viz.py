"""Chart builders: team-colour resolution, per-chart colour options, week filtering and theme contrast.

Every figure here is built from a hand-made frame; nothing touches the API or the archive.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cfbratings.viz import (
    CATEGORICAL,
    PALETTE,
    THEMES,
    conference_bar,
    conference_dots,
    contrast_ratio,
    rating_bar,
    resolve_team_color,
    series_colors,
    trajectory_chart,
    weekly_metric_chart,
    win_distribution_chart,
)
from cfbratings.viz.charts import (
    CATEGORICAL_DARK,
    CATEGORICAL_LIGHT,
    MIN_MARK_CONTRAST,
    NEAR_COLOR_DISTANCE,
    color_distance,
    parse_color,
)

CRIMSON = "#9e1b32"  # Alabama
NEAR_CRIMSON = "#a31d35"  # a shade off: indistinguishable as a 2px line
SCARLET = "#bb0000"  # Ohio State: a different red that still reads the same at line width
MAIZE = "#ffcb05"  # Michigan: fine on dark, vanishes on light
BLUE = "#2a78d6"
LIGHT_SURFACE = THEMES["light"]["surface"]
DARK_SURFACE = THEMES["dark"]["surface"]


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rank": [1, 2, 3, 4],
            "team": ["Alabama", "Georgia", "Michigan", "Boise State"],
            "conference": ["SEC", "SEC", "Big Ten", "Mountain West"],
            "rating": [0.9, 0.85, 0.8, 0.7],
            "record": ["10-0", "9-1", "9-1", "8-2"],
            "sos": [0.55, 0.52, 0.5, 0.4],
        }
    )


def _bars(fig) -> dict[str, object]:
    """Real bars keyed by their y label; legend-only placeholder traces carry ``x=[None]``."""
    return {tr.y[0]: tr for tr in fig.data if tr.type == "bar" and tr.x[0] is not None}


def _snapshots() -> dict[int, dict[str, float]]:
    return {
        1: {"A": 10.0, "B": 8.0, "C": 6.0, "FCS (pooled)": 1.0},
        2: {"A": 9.0, "B": 9.5, "C": 5.0, "FCS (pooled)": 1.0},
        3: {"A": 9.0, "B": 9.0, "C": 7.0, "FCS (pooled)": 1.0},
    }


def _by_week(*, with_n: bool = True) -> dict[str, pd.DataFrame]:
    elo = {"week": [1, 2, 3, 4, 5], "accuracy": [0.5, 0.7, 0.6, 0.8, 0.75], "brier": [0.3, 0.2, 0.25, 0.18, 0.2]}
    massey = {**elo, "accuracy": [0.55, 0.65, 0.62, 0.78, 0.7]}
    if with_n:
        elo["n"] = massey["n"] = [2, 10, 3, 12, 20]
    return {"Elo": pd.DataFrame(elo), "Massey": pd.DataFrame(massey)}


def _conferences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "conference": ["SEC", "Big Ten", "MAC"],
            "strength_index": [1.5, 1.4, 0.7],
            "teams": [16, 18, 12],
            "top_team": ["Alabama", "Ohio State", "Toledo"],
            "mean_rating": [0.6, 0.55, 0.2],
        }
    )


# --- colour arithmetic ------------------------------------------------------------------------------


def test_parse_color_forms():
    assert parse_color("#9E1B32") == (158, 27, 50)
    assert parse_color("9e1b32") == (158, 27, 50)
    assert parse_color("#fff") == (255, 255, 255)
    assert parse_color("#abcd") == (170, 187, 204)  # #rgba: alpha dropped
    assert parse_color("#9e1b32ff") == (158, 27, 50)  # #rrggbbaa: alpha dropped
    assert parse_color("rgb(158, 27, 50)") == (158, 27, 50)
    assert parse_color("rgba(158,27,50,0.5)") == (158, 27, 50)
    assert parse_color("  #FFF  ") == (255, 255, 255)


@pytest.mark.parametrize(
    "bad", [None, "", "#", "#12", "#12345", "#1234567", "#gggggg", "red", 123, 4.5, "rgb(", "rgb(300,0,0)", ["#fff"]]
)
def test_parse_color_malformed_is_none(bad):
    assert parse_color(bad) is None


def test_contrast_ratio_matches_wcag_reference_values():
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0)
    assert contrast_ratio("#808080", "#808080") == pytest.approx(1.0)
    assert contrast_ratio("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)  # the classic just-under-AA grey
    assert contrast_ratio("nope", "#ffffff") == 1.0


def test_color_distance_is_perceptual():
    assert color_distance(CRIMSON, NEAR_CRIMSON) < NEAR_COLOR_DISTANCE
    assert color_distance(CRIMSON, SCARLET) < NEAR_COLOR_DISTANCE  # two reds: same hue at line width
    assert color_distance(BLUE, "#eb6834") > NEAR_COLOR_DISTANCE  # blue vs orange
    assert color_distance("#bf5700", "#ff8200") > NEAR_COLOR_DISTANCE  # burnt orange vs Tennessee orange
    assert color_distance(CRIMSON, CRIMSON) == 0
    assert color_distance(CRIMSON, "nope") == float("inf")


# --- resolver ---------------------------------------------------------------------------------------


def test_resolver_keeps_a_legible_colour_and_normalises_it():
    assert resolve_team_color("#9E1B32", "#ffffff") == CRIMSON
    assert resolve_team_color("9e1b32", None) == CRIMSON
    assert resolve_team_color("#F00", None) == "#ff0000"


def test_resolver_light_colour_on_light_theme_falls_back_to_alt():
    assert resolve_team_color("#ffffff", CRIMSON, "light") == CRIMSON
    assert resolve_team_color(MAIZE, BLUE, theme="light") == BLUE
    assert resolve_team_color(LIGHT_SURFACE, CRIMSON, "light") == CRIMSON


def test_resolver_dark_colour_on_dark_theme_falls_back():
    assert resolve_team_color("#000000", "#1a1a19", "dark") == series_colors("dark")[0]  # both illegible
    assert resolve_team_color("#00274c", MAIZE, "dark") == MAIZE  # navy fails, maize is fine on dark
    assert resolve_team_color("#000000", None, "dark", fallback="#abcdef") == "#abcdef"
    assert resolve_team_color("#000000", None, "dark", fallback="#ABC") == "#aabbcc"


def test_resolver_surface_decides_which_way_a_colour_fails():
    assert resolve_team_color("#111111", BLUE, "light") == "#111111"
    assert resolve_team_color("#111111", BLUE, "dark") == BLUE
    assert resolve_team_color("#f5f5f5", BLUE, "dark") == "#f5f5f5"
    assert resolve_team_color("#f5f5f5", BLUE, "light") == BLUE
    assert contrast_ratio(resolve_team_color(MAIZE, None, "light"), LIGHT_SURFACE) >= MIN_MARK_CONTRAST
    assert resolve_team_color(MAIZE, None, "dark") == MAIZE


@pytest.mark.parametrize(
    "bad", [None, "", "#", "#12", "#12345", "#gggggg", "red", 123, 4.5, "rgb(", "rgb(300,0,0)", ["#ffffff"], object()]
)
def test_resolver_malformed_input_never_raises(bad):
    for theme in THEMES:
        out = resolve_team_color(bad, bad, theme, fallback=bad)
        assert out == series_colors(theme)[0]
    assert resolve_team_color(bad, CRIMSON, "dark") == CRIMSON
    assert resolve_team_color(bad, bad, "light", fallback=CRIMSON) == CRIMSON


def test_resolver_unknown_theme_behaves_as_light():
    assert resolve_team_color(MAIZE, None, "sepia") == series_colors("light")[0]


# --- rating_bar -------------------------------------------------------------------------------------


def test_rating_bar_default_colours_by_conference_with_legend():
    fig = rating_bar(_table(), top=4)
    assert fig.layout.showlegend is True
    legend = sorted(tr.name for tr in fig.data if tr.x[0] is None)
    assert legend == ["Big Ten", "Mountain West", "SEC"]
    bars = _bars(fig)
    assert len(bars) == 4
    assert bars["1. Alabama"].marker.color == bars["2. Georgia"].marker.color  # same conference, same hue
    assert bars["1. Alabama"].marker.color != bars["3. Michigan"].marker.color
    assert all(tr.marker.color in CATEGORICAL for tr in bars.values())


def test_rating_bar_team_colours_drop_the_legend_but_keep_the_conference_in_hover():
    colors = {"Alabama": "#9E1B32", "Georgia": "#ba0c2f", "Michigan": MAIZE}  # Boise State deliberately missing
    fig = rating_bar(_table(), top=4, team_colors=colors)
    assert fig.layout.showlegend is False
    assert all(tr.x[0] is not None for tr in fig.data)  # no legend placeholder traces
    bars = _bars(fig)
    assert len(bars) == 4
    assert bars["1. Alabama"].marker.color == CRIMSON
    assert bars["2. Georgia"].marker.color == "#ba0c2f"
    assert bars["3. Michigan"].marker.color == series_colors("light")[0]  # maize is illegible on light: guarded
    assert bars["4. Boise State"].marker.color == series_colors("light")[0]  # not in the mapping
    assert bars["1. Alabama"].customdata[0][2] == "SEC"
    assert bars["4. Boise State"].customdata[0][2] == "Mountain West"
    assert "%{customdata[2]}" in bars["1. Alabama"].hovertemplate
    assert fig.layout.margin.t < rating_bar(_table(), top=4).layout.margin.t  # no legend row reserved


def test_rating_bar_team_colours_respect_the_dark_surface():
    fig = rating_bar(_table(), top=4, team_colors={"Michigan": MAIZE, "Alabama": "#000000"}, theme="dark")
    bars = _bars(fig)
    assert bars["3. Michigan"].marker.color == MAIZE
    assert bars["1. Alabama"].marker.color == series_colors("dark")[0]


def test_rating_bar_existing_keyword_calls_still_work():
    fig = rating_bar(_table(), top=2, value_col="sos", value_label="SOS", title="Top 2 by SOS", theme="dark")
    assert len(_bars(fig)) == 2
    assert fig.layout.xaxis.title.text == "SOS"
    assert fig.layout.title.text == "Top 2 by SOS"


# --- trajectory_chart -------------------------------------------------------------------------------


def test_trajectory_default_uses_series_slots_and_solid_lines():
    fig = trajectory_chart(_snapshots(), ["A", "B", "C"])
    assert [tr.line.color for tr in fig.data] == list(CATEGORICAL[:3])
    assert all(tr.line.dash is None for tr in fig.data)
    assert [tr.name for tr in fig.data] == ["A", "B", "C"]
    assert list(fig.data[1].y) == [2, 1, 2]  # rank, FCS pool excluded; the week-3 tie breaks by name


def test_trajectory_team_colours_dash_the_second_of_a_look_alike_pair():
    fig = trajectory_chart(_snapshots(), ["A", "B", "C"], team_colors={"A": CRIMSON, "B": NEAR_CRIMSON, "C": BLUE})
    a, b, c = fig.data
    assert (a.line.color, b.line.color, c.line.color) == (CRIMSON, NEAR_CRIMSON, BLUE)
    assert a.line.dash is None  # first of the pair keeps the solid line
    assert b.line.dash == "dash"
    assert c.line.dash is None


def test_trajectory_three_way_clash_gets_three_distinct_dashes():
    fig = trajectory_chart(_snapshots(), ["A", "B", "C"], team_colors={"A": CRIMSON, "B": NEAR_CRIMSON, "C": SCARLET})
    assert [tr.line.dash for tr in fig.data] == [None, "dash", "dot"]


def test_trajectory_distinct_team_colours_stay_solid():
    fig = trajectory_chart(_snapshots(), ["A", "B"], team_colors={"A": BLUE, "B": "#eb6834"}, metric="rating")
    assert [tr.line.dash for tr in fig.data] == [None, None]
    assert list(fig.data[0].y) == [10.0, 9.0, 9.0]


def test_trajectory_missing_or_illegible_colour_keeps_its_slot():
    fig = trajectory_chart(_snapshots(), ["A", "B", "C"], team_colors={"A": "#ffffff", "C": CRIMSON}, theme="light")
    assert fig.data[0].line.color == series_colors("light")[0]  # white on light: guarded to the slot
    assert fig.data[1].line.color == series_colors("light")[1]  # absent from the mapping
    assert fig.data[2].line.color == CRIMSON


# --- weekly_metric_chart ----------------------------------------------------------------------------


def test_weekly_metric_drops_thin_weeks_and_shows_game_counts():
    fig = weekly_metric_chart(_by_week(), min_games=5)
    assert len(fig.data) == 2
    for tr in fig.data:
        assert list(tr.x) == [2, 4, 5]
        assert list(tr.customdata) == [10, 12, 20]
        assert "games" in tr.hovertemplate
    assert list(fig.data[0].y) == [0.7, 0.8, 0.75]
    assert fig.layout.xaxis.tickformat == "d"
    assert fig.layout.xaxis.dtick == 1


def test_weekly_metric_threshold_defaults_to_five_and_zero_disables_it():
    assert list(weekly_metric_chart(_by_week()).data[0].x) == [2, 4, 5]
    assert list(weekly_metric_chart(_by_week(), min_games=0).data[0].x) == [1, 2, 3, 4, 5]
    assert list(weekly_metric_chart(_by_week(), min_games=11).data[0].x) == [4, 5]


def test_weekly_metric_without_n_column_is_not_filtered():
    fig = weekly_metric_chart(_by_week(with_n=False), min_games=5, metric="brier")
    for tr in fig.data:
        assert list(tr.x) == [1, 2, 3, 4, 5]
        assert tr.customdata is None
        assert "games" not in tr.hovertemplate
    assert fig.layout.yaxis.title.text == "Brier"


def test_weekly_metric_colours_follow_method_order():
    fig = weekly_metric_chart(_by_week(), theme="dark")
    assert [tr.line.color for tr in fig.data] == list(CATEGORICAL_DARK[:2])
    assert [tr.name for tr in fig.data] == ["Elo", "Massey"]


# --- accent -----------------------------------------------------------------------------------------


def test_win_distribution_accent():
    dist = [0.05, 0.25, 0.4, 0.3]
    assert win_distribution_chart(dist, 6, team="X").data[0].marker.color == series_colors("light")[0]
    assert win_distribution_chart(dist, 6, team="X", accent=CRIMSON).data[0].marker.color == CRIMSON
    assert win_distribution_chart(dist, 6, accent="#FFFFFF").data[0].marker.color == series_colors("light")[0]
    assert win_distribution_chart(dist, 6, accent="#ffffff", theme="dark").data[0].marker.color == "#ffffff"
    assert win_distribution_chart(dist, 6, accent="not a colour").data[0].marker.color == series_colors("light")[0]
    assert list(win_distribution_chart(dist, 6).data[0].x) == ["6", "7", "8", "9"]


def test_conference_dots_accent():
    def dots(fig):
        return next(tr for tr in fig.data if tr.mode == "markers+text")

    plain = dots(conference_dots(_conferences()))
    assert plain.marker.color == series_colors("light")[0]
    assert plain.marker.line.color == THEMES["light"]["ring"]
    assert dots(conference_dots(_conferences(), accent=CRIMSON, theme="dark")).marker.color == CRIMSON
    assert dots(conference_dots(_conferences(), accent=MAIZE)).marker.color == series_colors("light")[0]
    assert dots(conference_bar(_conferences(), accent=CRIMSON)).marker.color == CRIMSON  # alias keeps working


# --- themes -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_theme_text_tokens_clear_wcag_contrast_on_their_surface(theme):
    t = THEMES[theme]
    assert contrast_ratio(t["ink"], t["surface"]) >= 4.5
    assert contrast_ratio(t["ink_secondary"], t["surface"]) >= 4.5
    assert contrast_ratio(t["muted"], t["surface"]) >= 3.0
    assert contrast_ratio(t["other"], t["surface"]) >= 3.0
    assert contrast_ratio(t["ink"], t["hover_bg"]) >= 4.5  # tooltip text on the tooltip box
    assert contrast_ratio(t["ring"], t["surface"]) == pytest.approx(1.0)  # the ring is a surface gap


def test_series_palettes_survive_a_page_theme_mismatch():
    # The page can render light while the browser reports dark (and vice versa): marks must stay
    # visible on either surface, and every slot must clear the mark floor on its own surface.
    for color in CATEGORICAL_DARK:
        assert contrast_ratio(color, LIGHT_SURFACE) >= MIN_MARK_CONTRAST
    for color in CATEGORICAL_LIGHT:
        assert contrast_ratio(color, DARK_SURFACE) >= MIN_MARK_CONTRAST
    for theme in THEMES:
        for color in series_colors(theme):
            assert contrast_ratio(color, THEMES[theme]["surface"]) >= MIN_MARK_CONTRAST
    assert contrast_ratio(THEMES["light"]["muted"], DARK_SURFACE) >= 3.0  # tick labels are theme-invariant
    assert contrast_ratio(THEMES["dark"]["muted"], LIGHT_SURFACE) >= 3.0


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_hoverlabel_is_themed(theme):
    for fig in (
        win_distribution_chart([0.1, 0.6, 0.3], 5, theme=theme),
        rating_bar(_table(), theme=theme),
        weekly_metric_chart(_by_week(), theme=theme),
    ):
        hl = fig.layout.hoverlabel
        assert hl.bgcolor == THEMES[theme]["hover_bg"]
        assert hl.font.color == THEMES[theme]["ink"]
        assert hl.bordercolor == THEMES[theme]["axis"]
        assert contrast_ratio(hl.font.color, hl.bgcolor) >= 4.5
        assert fig.layout.template.layout.paper_bgcolor is not None  # a plotly template is still applied


def test_palette_surface_stays_transparent_for_existing_callers():
    assert PALETTE["surface"] == "rgba(0,0,0,0)"
    assert PALETTE["ink"] == THEMES["light"]["ink"]
