from __future__ import annotations

import numpy as np
import pytest

from cfbratings.config import Settings
from cfbratings.data.schema import Game
from cfbratings.models import MODEL_NAMES, fit_ratings, make_model
from cfbratings.models.bradley_terry import BradleyTerryModel
from cfbratings.models.colley import ColleyModel, build_colley, solve_colley
from cfbratings.models.elo import EloModel, elo_expected, mov_multiplier
from cfbratings.models.hybrid import HybridModel
from cfbratings.models.massey import MasseyModel, build_massey, solve_massey


def _game(i, home, away, hp, ap, week=1, neutral=False, season=2025):
    return Game(
        id=i,
        season=season,
        week=week,
        home=home,
        away=away,
        home_points=hp,
        away_points=ap,
        neutral_site=neutral,
        completed=True,
    )


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_recover_latent_strength(name, season, rank_corr):
    res = fit_ratings(name, season.team_names, season.games)
    assert set(res.ratings) == set(season.team_names)
    truth = [season.true_strength[t] for t in season.team_names]
    est = [res.ratings[t] for t in season.team_names]
    assert rank_corr(truth, est) > 0.8, name
    assert all(np.isfinite(v) for v in est)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_handle_degenerate_inputs(name):
    assert fit_ratings(name, [], []).ratings == {}
    res = fit_ratings(name, ["A", "B", "C"], [])
    assert len(res.ratings) == 3 and len(set(res.ratings.values())) == 1  # all equal
    res = fit_ratings(name, ["A", "B", "C"], [_game(1, "A", "B", 28, 7)])  # C isolated
    assert res.ratings["A"] > res.ratings["B"]
    assert all(np.isfinite(v) for v in res.ratings.values())


def test_colley_matches_textbook_values():
    teams = ["A", "B"]
    C, b = build_colley(teams, [_game(1, "A", "B", 10, 0)])
    assert np.allclose(C, [[3, -1], [-1, 3]]) and np.allclose(b, [1.5, 0.5])
    r = solve_colley(C, b)
    assert np.allclose(r, [0.625, 0.375])
    assert np.allclose(sum(r), 1.0)
    assert ColleyModel().fit(teams, []).ratings == {"A": 0.5, "B": 0.5}


def test_massey_is_centred_and_respects_hfa_and_cap():
    teams = ["A", "B", "C"]
    games = [_game(1, "A", "B", 24, 14), _game(2, "B", "C", 21, 14), _game(3, "C", "A", 7, 30)]
    M, b = build_massey(teams, games, ridge_lambda=0.5, hfa=3.0)
    r = solve_massey(M, b)
    assert abs(r.mean()) < 1e-9
    assert r[0] > r[1] > r[2]
    neutral = [_game(1, "A", "B", 24, 14, neutral=True)]
    res_neutral = MasseyModel(hfa=3.0).fit(["A", "B"], neutral)
    res_home = MasseyModel(hfa=3.0).fit(["A", "B"], [_game(1, "A", "B", 24, 14)])
    assert res_neutral.ratings["A"] - res_neutral.ratings["B"] > res_home.ratings["A"] - res_home.ratings["B"]
    capped = MasseyModel(max_margin=20).fit(["A", "B"], [_game(1, "A", "B", 80, 0)])
    uncapped = MasseyModel(max_margin=200).fit(["A", "B"], [_game(1, "A", "B", 80, 0)])
    assert capped.ratings["A"] < uncapped.ratings["A"]


def test_massey_recency_weighting_prefers_recent_games():
    teams = ["A", "B"]
    games = [_game(1, "A", "B", 0, 30, week=1), _game(2, "A", "B", 30, 0, week=10)]
    flat = MasseyModel(hfa=0.0, half_life_weeks=0.0).fit(teams, games).ratings
    recent = MasseyModel(hfa=0.0, half_life_weeks=2.0).fit(teams, games).ratings
    assert abs(flat["A"] - flat["B"]) < 1e-9  # symmetric when unweighted
    assert recent["A"] > recent["B"] + 5


def test_elo_formulas():
    assert elo_expected(1500, 1500) == 0.5
    assert elo_expected(1500, 1500, hfa_elo=62.5) > 0.5
    assert mov_multiplier(0, 0) == 0.0
    assert mov_multiplier(21, 100) < mov_multiplier(21, -100)  # favourites gain less
    model = EloModel(k=25, hfa=2.5, elo_per_point=25)
    assert model.hfa_elo == 62.5
    # Symmetric: a home win by 10 moves ratings by the same amount as an away win by 10 at a neutral site.
    r1 = {"A": 1500.0, "B": 1500.0}
    d1 = model.update(r1, _game(1, "A", "B", 20, 10, neutral=True))
    r2 = {"A": 1500.0, "B": 1500.0}
    d2 = model.update(r2, _game(2, "A", "B", 10, 20, neutral=True))
    assert abs(d1 + d2) < 1e-12 and d1 > 0
    assert abs(sum(r1.values()) - 3000.0) < 1e-9  # zero-sum


def test_elo_regresses_between_seasons_not_weeks():
    model = EloModel(k=40, season_regress=0.5)
    games = [_game(i, "A", "B", 30, 0, week=i) for i in range(1, 6)]
    r_same = model.fit(["A", "B"], games).ratings
    games_next = [*games, _game(99, "A", "B", 0, 0, week=1, season=2026)]  # tie → no update, but new season
    r_next = model.fit(["A", "B"], games_next).ratings
    assert abs((r_next["A"] - 1500) - 0.5 * (r_same["A"] - 1500)) < 1e-9
    assert r_same["A"] - 1500 > 100  # no weekly decay eating the signal


def test_bradley_terry_finds_positive_home_advantage(season):
    res = BradleyTerryModel(ridge=1.0).fit(season.team_names, season.games)
    assert res.extras["home_adv_logit"] > 0
    assert abs(np.mean(list(res.ratings.values()))) < 1e-9


def test_hybrid_weights_normalised_and_validated(season):
    model = make_model("hybrid", weights={"massey": 3, "elo": 1})
    assert isinstance(model, HybridModel)
    assert pytest.approx(model.weights["massey"]) == 0.75
    res = model.fit(season.team_names, season.games)
    assert set(res.extras["components"]) == {"massey", "elo"}
    with pytest.raises(ValueError):
        make_model("hybrid", weights={"hybrid": 1})
    with pytest.raises(ValueError):
        make_model("hybrid", weights={"nope": 1})
    with pytest.raises(ValueError):
        HybridModel(components={}, weights={})


def test_make_model_accepts_settings_and_model_names():
    s = Settings(elo_k=33, massey_ridge_lambda=2.5, home_field_adv=3.1)
    assert make_model("elo", s).k == 33
    assert make_model("elo", s, k=10).k == 10
    assert make_model("elo", s, elo_k=11).k == 11
    assert make_model("massey", s).ridge_lambda == 2.5 and make_model("massey", s).hfa == 3.1
    assert make_model("Bradley-Terry").name == "bradley_terry"
    with pytest.raises(ValueError):
        make_model("glicko")


def test_rating_result_helpers(season):
    res = fit_ratings("massey", season.team_names, season.games)
    ranked = res.ranked()
    assert ranked[0][1] >= ranked[1][1]
    assert res.rank_map()[ranked[0][0]] == 1
    z = res.zscores()
    assert abs(np.mean(list(z.values()))) < 1e-9
    frame = res.to_frame()
    assert list(frame.columns) == ["rank", "team", "rating"] and len(frame) == len(season.team_names)
