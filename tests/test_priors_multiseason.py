from __future__ import annotations

import numpy as np
import pytest

from cfbratings.analytics import conference_strength, schedule_table, team_table
from cfbratings.backtest import BacktestConfig, compare_methods, run_backtest
from cfbratings.backtest.metrics import summarize_predictions
from cfbratings.config import Settings
from cfbratings.data import FCS_POOL, synthetic_seasons
from cfbratings.data.schema import Game, pool_non_fbs
from cfbratings.models import MODEL_NAMES, fit_ratings, make_model
from cfbratings.predict import MarginCalibrator
from cfbratings.priors import preseason_prior, prior_for_season, regress_toward, season_priors
from cfbratings.snapshots import weekly_snapshots


@pytest.fixture(scope="module")
def chain():
    return synthetic_seasons(
        [2024, 2025, 2026], n_teams=48, n_conferences=6, weeks=12, strength_sd=11.0, noise_sd=12.0, seed=5
    )


def _game(i, home, away, hp, ap, week=1, season=2025):
    return Game(id=i, season=season, week=week, home=home, away=away, home_points=hp, away_points=ap, completed=True)


def test_synthetic_chain_shares_roster_and_persists_strength(chain):
    a, b, c = chain
    assert a.team_names == b.team_names == c.team_names
    assert [s.season for s in chain] == [2024, 2025, 2026]
    x = np.array([a.true_strength[t] for t in a.team_names])
    y = np.array([b.true_strength[t] for t in a.team_names])
    assert 0.4 < np.corrcoef(x, y)[0, 1] < 0.95


def test_regress_toward():
    assert regress_toward({"A": 10.0, "B": -10.0}, 0.5) == {"A": 5.0, "B": -5.0}
    assert regress_toward({"A": 1600.0}, 0.25, center=1500.0) == {"A": 1575.0}
    assert regress_toward({}, 0.5) == {}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_prior_with_no_games_returns_prior_and_prior_shifts_fit(name, chain):
    prev, cur = chain[0], chain[1]
    model = make_model(name)
    prior = preseason_prior(model, cur.team_names, prev.games, regress=0.3)
    assert prior  # non-empty
    empty = model.fit(cur.team_names, [], prior=prior)
    if name == "hybrid":
        assert set(prior) == set(model.components)
        comp = next(iter(prior.values()))
        assert set(comp) == set(cur.team_names)
    else:
        assert set(prior) == set(cur.team_names)
        for t in cur.team_names:
            assert empty.ratings[t] == pytest.approx(prior[t], abs=1e-6)
    # After a few weeks the warm-started fit still leans toward the prior.
    early = [g for g in cur.games if g.week <= 2]
    cold = model.fit(cur.team_names, early).ratings
    warm = model.fit(cur.team_names, early, prior=prior).ratings
    truth = np.array([cur.true_strength[t] for t in cur.team_names])

    def corr(r):
        return np.corrcoef(truth, [r[t] for t in cur.team_names])[0, 1]

    assert corr(warm) > corr(cold) + 0.1


def test_season_priors_chain_and_prior_for_season(chain):
    games = [g for s in chain for g in s.games]
    teams = chain[-1].team_names
    priors = season_priors("massey", games, teams, regress=0.33)
    assert list(priors) == [2024, 2025, 2026]
    assert priors[2024] == {} and priors[2025] and priors[2026]
    direct = prior_for_season("massey", games, teams, 2026, settings=Settings(season_regress=0.33))
    assert direct == priors[2026]
    assert prior_for_season("massey", games, teams, 2024) == {}
    assert abs(np.mean(list(priors[2026].values()))) < 1e-9  # Massey priors stay centred


def test_warm_start_improves_early_season_predictions(chain):
    games = [g for s in chain for g in s.games]
    teams = chain[-1].team_names
    warm = run_backtest(teams, games, BacktestConfig(method="massey", start_week=1)).predictions
    assert sorted(warm["season"].unique()) == [2024, 2025, 2026]
    last = warm[warm["season"] == 2026]
    assert last["week"].min() == 1  # a prior lets evaluation start in week 1
    early = last[last["week"] <= 3]
    model = summarize_predictions(early)
    baseline = summarize_predictions(
        early[["prior_p_home", "prior_margin", "home_win", "actual_margin"]].rename(
            columns={"prior_p_home": "p_home", "prior_margin": "pred_margin"}
        )
    )
    assert model["log_loss"] < baseline["log_loss"]
    assert model["mae"] < baseline["mae"]
    cold = run_backtest(teams, chain[-1].games, BacktestConfig(method="massey", start_week=1)).predictions
    assert cold["week"].min() > 1  # a cold start waits for min_train_games before predicting
    assert len(cold) < len(last)


def test_compare_methods_reports_seasons(chain):
    games = [g for s in chain[1:] for g in s.games]
    table, results = compare_methods(chain[-1].team_names, games, ["elo"], start_week=2)
    assert table.iloc[0]["seasons"] == "2025-2026"
    by_season = results["elo"].by_season()
    assert list(by_season["season"]) == [2025, 2026]


def test_explicit_first_season_prior_and_calibrator_prior(chain):
    prev, cur = chain[0], chain[1]
    model = make_model("elo")
    prior = preseason_prior(model, cur.team_names, prev.games, regress=0.33)
    res = run_backtest(cur.team_names, cur.games, BacktestConfig(method="elo", start_week=1), prior=prior)
    assert res.predictions["week"].min() == 1
    cal = MarginCalibrator.fit_cv(model, cur.team_names, cur.games[:40], prior=prior)
    assert cal.sigma > 0 and np.isfinite(cal.slope)
    snaps = weekly_snapshots("elo", cur.team_names, [*prev.games, *cur.games], use_cache=False, prior=prior)
    assert min(snaps) == 1 and all(abs(v - 1500) < 400 for v in snaps[1].values())
    assert fit_ratings("elo", cur.team_names, [], prior=prior).ratings == dict(prior)


def test_pooled_fcs_team_is_rated_but_never_ranked():
    teams = ["A", "B", "C"]
    games = pool_non_fbs(
        [
            _game(1, "A", "B", 21, 20),
            _game(2, "B", "C", 24, 10),
            _game(3, "A", "Tiny U", 49, 0),
            _game(4, "C", "Other FCS", 10, 13),
            _game(5, "A", "C", 30, 3, week=2),
        ],
        teams,
    )
    rated = [*teams, FCS_POOL]
    for name in MODEL_NAMES:
        res = fit_ratings(name, rated, games)
        assert FCS_POOL in res.ratings
        assert [t for t, _ in res.ranked()] == sorted(teams, key=lambda t: -res.ratings[t])
        assert FCS_POOL not in res.rank_map()
    res = fit_ratings("massey", rated, games)
    assert res.ratings[FCS_POOL] < res.ratings["C"]  # lost to A by 49, beat C by 3
    cal = MarginCalibrator.fit(res, games)
    table = team_table(rated, games, res, cal, {"A": "X", "B": "X", "C": "Y", FCS_POOL: "FCS"})
    assert list(table["team"]) == [t for t, _ in res.ranked()]
    assert table.loc[table["team"] == "A", "wins"].item() == 3  # the FCS win counts in the record
    assert "FCS" not in conference_strength(res.ratings, {"A": "X", "B": "X", "C": "Y", FCS_POOL: "FCS"})
    sched = schedule_table("A", games, res, cal)
    assert "Tiny U (FCS)" in set(sched["opponent"])
