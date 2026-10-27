from __future__ import annotations

import numpy as np
import pytest

from cfbratings.data import synthetic_season
from cfbratings.data.schema import Game
from cfbratings.models import make_model
from cfbratings.predict import MarginCalibrator
from cfbratings.scores import DEFAULT_MEAN_POINTS, MIN_LOSER_POINTS, ScoreModel, fit_score_model


def _game(i, home, away, hp, ap, *, neutral=False):
    return Game(
        id=i,
        season=2026,
        week=1,
        home=home,
        away=away,
        home_points=hp,
        away_points=ap,
        completed=True,
        neutral_site=neutral,
    )


def test_no_games_gives_a_neutral_model():
    m = fit_score_model(["A", "B"], [], hfa=3.0)
    assert m.n_games == 0 and m.mean_points == DEFAULT_MEAN_POINTS and m.half_hfa == 1.5
    assert m.total("A", "B") == 2 * DEFAULT_MEAN_POINTS
    ps = m.project("A", "B", 7.0)
    assert ps.home_points - ps.away_points == pytest.approx(7.0)
    assert ps.home_points + ps.away_points == pytest.approx(2 * DEFAULT_MEAN_POINTS)


def test_fit_recovers_offence_and_defence_shape():
    # A scores a lot and allows a lot; D is the opposite. Every team plays everyone twice (home/away).
    teams = ["A", "B", "C", "D"]
    pts = {"A": (38, 31), "B": (28, 28), "C": (28, 28), "D": (17, 24)}  # (scored, allowed) tendencies
    games = []
    i = 0
    for h in teams:
        for a in teams:
            if h == a:
                continue
            hp = (pts[h][0] + pts[a][1]) / 2 + 1.5
            ap = (pts[a][0] + pts[h][1]) / 2 - 1.5
            games.append(_game(i, h, a, round(hp), round(ap)))
            i += 1
    m = fit_score_model(teams, games, hfa=3.0, ridge=0.5)
    assert m.n_games == 12
    assert m.offense["A"] > m.offense["B"] > m.offense["D"]
    assert m.defense["A"] > m.defense["B"] > m.defense["D"]  # A leaks points, D does not
    assert m.total("A", "A") > m.total("B", "C") > m.total("D", "D")


def test_projected_score_splits_total_around_the_margin():
    m = ScoreModel(mean_points=28.0, offense={"A": 4.0, "B": -2.0}, defense={"A": 1.0, "B": -3.0})
    ps = m.project("A", "B", 10.0)
    assert ps.total == pytest.approx(2 * 28 + 4 - 3 - 2 + 1)
    assert ps.home_points - ps.away_points == pytest.approx(10.0)
    assert ps.home_points + ps.away_points == pytest.approx(ps.total)
    assert ps.favorite == "A" and ps.text == f"{round(ps.home_points)}-{round(ps.away_points)}"
    assert ps.text_by_favorite.startswith("A ")
    # the away favourite comes first in the favourite-first text
    ps2 = m.project("A", "B", -3.0)
    assert ps2.favorite == "B" and ps2.text_by_favorite.startswith("B ")


def test_rounding_never_ties_and_the_loser_has_a_floor():
    m = ScoreModel(mean_points=20.0)
    ps = m.project("A", "B", 0.4)  # 20.2 vs 19.8 would round to a tie
    h, a = ps.rounded()
    assert h == a + 1
    ps = m.project("A", "B", -0.4)
    h, a = ps.rounded()
    assert a == h + 1
    # a huge margin cannot push the loser below the floor
    ps = m.project("A", "B", 60.0)
    assert ps.away_points == pytest.approx(MIN_LOSER_POINTS)
    assert ps.home_points - ps.away_points == pytest.approx(60.0)


def test_home_field_moves_points_between_teams_not_the_total():
    m = ScoreModel(mean_points=28.0, half_hfa=1.5)
    assert m.expected_points("A", "B", home=True) - m.expected_points("A", "B", home=False) == pytest.approx(3.0)
    assert m.expected_points("A", "B", home=None) == pytest.approx(28.0)
    assert m.total("A", "B") == pytest.approx(56.0)


def test_sampled_scores_follow_the_projection():
    m = ScoreModel(mean_points=28.0, total_sigma=10.0)
    rng = np.random.default_rng(0)
    draws = [m.sample("A", "B", 7.0, 14.0, rng) for _ in range(4000)]
    margins = np.array([d.margin for d in draws])
    totals = np.array([d.total for d in draws])
    assert margins.mean() == pytest.approx(7.0, abs=0.8)
    assert margins.std() == pytest.approx(14.0, abs=1.0)
    assert totals.mean() == pytest.approx(56.0, abs=1.0)
    assert all(round(d.margin) != 0 for d in draws)  # no ties
    assert all(min(d.home_points, d.away_points) >= MIN_LOSER_POINTS - 1e-9 for d in draws)


def test_fit_on_a_synthetic_season_is_well_behaved():
    season = synthetic_season(season=2026, n_teams=40, n_conferences=5, weeks=12, played_through=10, seed=11)
    names = season.team_names
    model = make_model("massey")
    model.fit(names, season.games)
    cal = MarginCalibrator.fit_cv(model, names, season.games, fallback_hfa=2.5)
    m = fit_score_model(names, season.games, hfa=cal.intercept)
    played = [g for g in season.games if g.has_score]
    assert m.n_games == len(played)
    assert 15 < m.mean_points < 45
    assert m.total_sigma >= 6.0
    # offence and defence are centred by the ridge: the mean of each is close to zero
    assert abs(float(np.mean(list(m.offense.values())))) < 1.0
    assert abs(float(np.mean(list(m.defense.values())))) < 1.0
    # the projected total of an average matchup is about twice the mean
    a, b = names[0], names[1]
    assert abs(m.total(a, b) - 2 * m.mean_points) < 30
