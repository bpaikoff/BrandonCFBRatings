from __future__ import annotations

import pytest

from cfbratings.data.schema import (
    FCS_POOL,
    Game,
    Record,
    Team,
    completed_games,
    games_through_week,
    max_completed_week,
    normalize_games,
    pool_non_fbs,
)


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
