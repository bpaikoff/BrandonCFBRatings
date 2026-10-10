from __future__ import annotations

import numpy as np
import pytest

from cfbratings.data import synthetic_season
from cfbratings.data.schema import Game
from cfbratings.models import make_model
from cfbratings.predict import MarginCalibrator
from cfbratings.projections import project_team, projection_table, remaining_games, win_distribution


def _game(i, home, away, hp=None, ap=None, week=1):
    return Game(
        id=i,
        season=2026,
        week=week,
        home=home,
        away=away,
        home_points=hp,
        away_points=ap,
        completed=hp is not None,
    )


def test_win_distribution_matches_binomial_and_edge_cases():
    assert np.allclose(win_distribution([]), [1.0])
    assert np.allclose(win_distribution([1.0, 1.0]), [0, 0, 1])
    assert np.allclose(win_distribution([0.0]), [1, 0])
    d = win_distribution([0.5] * 4)
    assert np.allclose(d, [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16])
    d = win_distribution([0.9, 0.2])
    assert np.allclose(d, [0.1 * 0.8, 0.9 * 0.8 + 0.1 * 0.2, 0.9 * 0.2])
    assert abs(win_distribution([0.3, 0.7, 0.55, 0.1]).sum() - 1) < 1e-12


def test_project_team_uses_calibrated_probabilities():
    teams = ["A", "B", "C"]
    games = [
        _game(1, "A", "B", 30, 10, week=1),
        _game(2, "C", "A", 7, 28, week=2),
        _game(3, "A", "B", week=3),  # unplayed, at home
        _game(4, "C", "A", week=4),  # unplayed, away
    ]
    result = make_model("massey").fit(teams, games)
    cal = MarginCalibrator(slope=1.0, intercept=2.5, sigma=14.0, n_games=2, source="prior")
    assert [g.id for g in remaining_games("A", games)] == [3, 4]
    proj = project_team("A", games, result, cal)
    assert proj.record.wins == 2 and proj.record.losses == 0 and len(proj.remaining) == 2
    p_home = cal.win_prob(result.get("A"), result.get("B"))
    p_away = 1 - cal.win_prob(result.get("C"), result.get("A"))
    assert proj.remaining[0][1] == pytest.approx(p_home) and proj.remaining[1][1] == pytest.approx(p_away)
    assert p_home > 0.5 and p_away > 0.5  # A has beaten both comfortably
    assert proj.expected_final_wins == pytest.approx(2 + p_home + p_away)
    assert proj.p_at_least(2) == 1.0 and proj.p_at_least(5) == 0.0
    assert proj.p_at_least(4) == pytest.approx(p_home * p_away) == pytest.approx(proj.p_win_out)
    assert proj.p_at_least(3) == pytest.approx(1 - (1 - p_home) * (1 - p_away))
    assert proj.p_exactly(3) == pytest.approx(proj.p_at_least(3) - proj.p_at_least(4))
    assert proj.next_game[0].id == 3


def test_projection_table_on_partial_season():
    season = synthetic_season(n_teams=32, n_conferences=4, weeks=12, played_through=8, seed=3)
    played = [g for g in season.games if g.has_score]
    unplayed = [g for g in season.games if not g.has_score]
    assert played and unplayed and max(g.week for g in played) == 8 and min(g.week for g in unplayed) == 9
    assert all(g.season_type == "regular" for g in season.games)  # no bowls scheduled mid-season
    model = make_model("elo")
    result = model.fit(season.team_names, season.games)
    cal = MarginCalibrator.fit_cv(model, season.team_names, season.games)
    df = projection_table(season.team_names, season.games, result, cal, conference_map=season.conference_map)
    assert len(df) == 32 and df["remaining"].sum() == 2 * len(unplayed)
    assert ((df["p_6_wins"] >= df["p_8_wins"]) & (df["p_8_wins"] >= df["p_10_wins"])).all()
    assert (df["proj_wins"] + df["proj_losses"] - df["wins"] - df["losses"] - df["remaining"]).abs().max() < 1e-9
    assert df["proj_wins"].is_monotonic_decreasing
    assert df["next_opponent"].notna().sum() == (df["remaining"] > 0).sum()
    # A fully played season projects exactly the final record.
    full = synthetic_season(n_teams=32, n_conferences=4, weeks=12, seed=3)
    res_full = model.fit(full.team_names, full.games)
    df_full = projection_table(full.team_names, full.games, res_full, cal)
    assert (df_full["remaining"] == 0).all() and (df_full["proj_wins"] == df_full["wins"]).all()
    assert (df_full["p_win_out"] == 1.0).all()
