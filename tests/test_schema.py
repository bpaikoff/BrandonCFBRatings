from __future__ import annotations

import colorsys
import re

import pytest

from cfbratings.data.schema import (
    FCS_POOL,
    Game,
    Record,
    Team,
    completed_games,
    contrast_color,
    games_through_week,
    max_completed_week,
    normalize_games,
    normalize_hex_color,
    pool_non_fbs,
)
from cfbratings.data.synthetic import synthetic_season, synthetic_seasons


def test_from_cfbd_camel_and_snake_case():
    camel = {
        "id": 1,
        "season": 2025,
        "week": 3,
        "seasonType": "regular",
        "homeTeam": "A",
        "awayTeam": "B",
        "homePoints": 28,
        "awayPoints": 14,
        "neutralSite": False,
        "completed": True,
        "homeConference": "X",
        "awayConference": "Y",
    }
    snake = {
        "id": 1,
        "season": 2025,
        "week": 3,
        "season_type": "regular",
        "home_team": "A",
        "away_team": "B",
        "home_points": 28,
        "away_points": 14,
        "neutral_site": False,
        "completed": True,
        "home_conference": "X",
        "away_conference": "Y",
    }
    assert Game.from_cfbd(camel) == Game.from_cfbd(snake)
    g = Game.from_cfbd(camel)
    assert g.margin == 14 and g.home_result == 1.0 and g.winner == "A" and g.loser == "B"
    assert not g.is_conference_game
    assert g.margin_for("B") == -14 and g.opponent("A") == "B"


def test_incomplete_game_has_no_score():
    g = Game(id=1, season=2025, week=1, home="A", away="B", completed=True, home_points=None, away_points=None)
    assert not g.has_score
    with pytest.raises(ValueError):
        _ = g.margin


def test_tie_result():
    g = Game(id=1, season=2025, week=1, home="A", away="B", home_points=21, away_points=21, completed=True)
    assert g.home_result == 0.5 and g.winner is None


def test_postseason_weeks_relabelled_after_regular_season():
    raw = [
        {
            "id": 1,
            "season": 2025,
            "week": 14,
            "seasonType": "regular",
            "homeTeam": "A",
            "awayTeam": "B",
            "completed": True,
            "homePoints": 1,
            "awayPoints": 0,
        },
        {
            "id": 2,
            "season": 2025,
            "week": 1,
            "seasonType": "postseason",
            "homeTeam": "A",
            "awayTeam": "B",
            "completed": True,
            "homePoints": 1,
            "awayPoints": 0,
        },
        {
            "id": 3,
            "season": 2025,
            "week": 2,
            "seasonType": "regular",
            "homeTeam": "B",
            "awayTeam": "A",
            "completed": False,
        },
    ]
    games = normalize_games(raw)
    assert [g.id for g in games] == [3, 1, 2]
    bowl = games[-1]
    assert bowl.week == 15 and bowl.raw_week == 1 and bowl.season_type == "postseason"
    assert max_completed_week(games) == 15
    assert len(completed_games(games)) == 2
    assert [g.id for g in games_through_week(games, 14)] == [1]
    assert [g.id for g in games_through_week(games, 15, inclusive=False)] == [1]


def test_round_trip_dict():
    g = Game(id=7, season=2025, week=2, home="A", away="B", home_points=3, away_points=0, completed=True)
    assert Game.from_dict(g.to_dict()) == g


def test_team_and_record():
    t = Team.from_cfbd({"school": "Ohio State", "conference": "Big Ten", "classification": "fbs"})
    assert t.school == "Ohio State" and t.conference == "Big Ten"
    r = Record(10, 2, 1)
    assert str(r) == "10-2-1" and r.games == 13 and abs(r.win_pct - 10.5 / 13) < 1e-12
    assert str(Record(3, 1)) == "3-1"


def test_pool_non_fbs_rewrites_opponents_and_keeps_names():
    games = [
        Game(id=1, season=2025, week=1, home="A", away="B", home_points=1, away_points=0, completed=True),
        Game(
            id=2,
            season=2025,
            week=1,
            home="A",
            away="Tiny U",
            home_points=50,
            away_points=0,
            completed=True,
            away_conference="Big Sky",
        ),
        Game(id=3, season=2025, week=2, home="Tiny U", away="B", home_points=0, away_points=3, completed=True),
        Game(id=4, season=2025, week=2, home="Tiny U", away="Smaller U", home_points=0, away_points=3, completed=True),
    ]
    pooled = pool_non_fbs(games, ["A", "B"])
    assert [g.id for g in pooled] == [1, 2, 3]
    assert pooled[1].away == FCS_POOL and pooled[1].away_name == "Tiny U" and pooled[1].away_conference == "FCS"
    assert pooled[1].display_name(FCS_POOL) == "Tiny U" and pooled[1].display_name("A") == "A"
    assert pooled[2].home == FCS_POOL and pooled[2].home_display == "Tiny U"
    assert pooled[0].home_name is None


# --- team colours and logos ------------------------------------------------------------------

LOGOS = [
    "https://cdn.collegefootballdata.com/logos/500/333.png",
    "https://cdn.collegefootballdata.com/logos-dark/500/333.png",
    "https://cdn.collegefootballdata.com/logos/256/333.png",
    "https://cdn.collegefootballdata.com/logos-dark/256/333.png",
]
HEX = re.compile(r"#[0-9a-f]{6}")


def test_team_colours_and_logos_parse_from_cfbd_rows():
    t = Team.from_cfbd({"school": "Alabama", "color": "#9E1B32", "alternateColor": "FFFFFF", "logos": LOGOS})
    assert (t.color, t.alt_color) == ("#9e1b32", "#ffffff")  # lower-case, always with the hash
    assert t.logo == LOGOS[0] and t.logo_dark == LOGOS[1]  # the 500 px light logo and its logos-dark twin
    light_only = Team.from_cfbd({"school": "B", "logos": [LOGOS[0]]})
    assert light_only.logo == LOGOS[0] and light_only.logo_dark is None
    bare = Team.from_cfbd({"school": "C", "color": None, "alternateColor": "not a colour", "logos": None})
    assert (bare.color, bare.alt_color, bare.logo, bare.logo_dark) == (None, None, None, None)
    assert Team.from_cfbd({"school": "D"}) == Team(school="D")  # rows without the keys parse as before
    assert Team(school=FCS_POOL, conference="FCS", classification="fcs").color is None  # the pooled pseudo-team


def test_normalize_hex_color_and_contrast():
    assert normalize_hex_color("#BB0000") == "#bb0000" and normalize_hex_color("bb0000") == "#bb0000"
    assert normalize_hex_color(" #FfF ") == "#ffffff"  # shorthand expands
    assert normalize_hex_color(None) is None and normalize_hex_color("") is None
    assert normalize_hex_color("#12345") is None and normalize_hex_color("red") is None
    assert contrast_color("#000000") == "#ffffff" and contrast_color("#ffffff") == "#000000"
    assert contrast_color("#9e1b32") == "#ffffff" and contrast_color("#ffd100") == "#000000"
    assert contrast_color(None) == "#000000" and contrast_color("nope", dark="#111") == "#111"


def test_team_dict_round_trip_and_legacy_dicts():
    raw = {"school": "Alabama", "conference": "SEC", "color": "#9e1b32", "alternateColor": "#ffffff", "logos": LOGOS}
    t = Team.from_cfbd(raw)
    d = t.to_dict()
    assert set(d) == {
        "school",
        "conference",
        "classification",
        "abbreviation",
        "color",
        "alt_color",
        "logo",
        "logo_dark",
    }
    assert Team.from_dict(d) == t and Team(**d) == t  # the dashboard rebuilds teams both ways
    assert Team.from_dict(raw) == t  # a raw CFBD row is accepted too
    legacy = {"school": "Alabama", "conference": "SEC", "classification": "fbs", "abbreviation": None}
    assert Team.from_dict(legacy) == Team(school="Alabama", conference="SEC", classification="fbs")  # older caches


def _hue(color: str) -> float:
    r, g, b = (int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[0]


def test_synthetic_teams_get_deterministic_brand_colours(season):
    kw = {"n_teams": 24, "n_conferences": 4, "weeks": 3, "postseason_games": 0}
    a, b = synthetic_season(seed=3, **kw), synthetic_season(seed=3, **kw)
    assert [(t.color, t.alt_color) for t in a.teams] == [(t.color, t.alt_color) for t in b.teams]
    for t in a.teams:
        assert HEX.fullmatch(t.color) and t.alt_color in {"#ffffff", "#000000"}
        assert t.alt_color == contrast_color(t.color) and t.logo is None and t.logo_dark is None
    assert len({t.color for t in a.teams}) == 24 and len({t.color for t in season.teams}) == len(season.teams)
    by_conf: dict[str, list[str]] = {}
    for t in a.teams:
        by_conf.setdefault(t.conference, []).append(t.color)
    assert len(by_conf) == 4 and all(len(colours) == 6 for colours in by_conf.values())
    for colours in by_conf.values():  # one hue family per conference, different shades per team
        hues = [_hue(c) for c in colours]
        assert all(min(abs(h - hues[0]), 1.0 - abs(h - hues[0])) < 0.03 for h in hues)
    first, second = synthetic_seasons([2024, 2025], n_teams=8, n_conferences=2, weeks=2, postseason_games=0)
    assert [t.color for t in second.teams] == [t.color for t in first.teams]  # a carried roster keeps its colours
