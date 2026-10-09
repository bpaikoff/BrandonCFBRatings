from __future__ import annotations

import numpy as np
import pytest

from cfbratings.analytics import (
    conference_strength,
    conference_table,
    momentum,
    records,
    schedule_table,
    strength_of_schedule,
    team_table,
)
from cfbratings.data.schema import Game
from cfbratings.models import make_model
from cfbratings.predict import DEFAULT_SIGMA, MarginCalibrator, normal_cdf


def _game(i, home, away, hp, ap, week=1, neutral=False):
    return Game(
        id=i,
        season=2025,
        week=week,
        home=home,
        away=away,
        home_points=hp,
        away_points=ap,
        neutral_site=neutral,
        completed=True,
    )


def test_normal_cdf():
    assert normal_cdf(0.0) == 0.5
    assert abs(normal_cdf(1.96) - 0.975) < 1e-3


def test_calibrator_falls_back_to_prior_with_few_games():
    model = make_model("massey")
    res = model.fit(["A", "B"], [_game(1, "A", "B", 20, 10)])
    cal = MarginCalibrator.fit(res, [_game(1, "A", "B", 20, 10)], fallback_hfa=2.5)
    assert cal.source == "prior" and cal.slope == 1.0 and cal.sigma == DEFAULT_SIGMA
    assert cal.win_prob(10, 10, neutral=True) == 0.5
    assert cal.win_prob(10, 10) > 0.5  # home edge
    assert cal.win_prob(20, 10) > cal.win_prob(10, 10) > cal.win_prob(0, 10)


@pytest.mark.parametrize("name", ["massey", "elo", "colley", "hybrid"])
def test_cross_validated_calibrator_recovers_planted_hfa_and_noise(name, season):
    model = make_model(name)
    cal = MarginCalibrator.fit_cv(model, season.team_names, season.games, fallback_hfa=2.5)
    assert cal.source == "cross_val"
    assert abs(cal.intercept - season.hfa) < 3.0
    assert season.noise_sd - 3 < cal.sigma < season.noise_sd + 8
    res = model.fit(season.team_names, season.games)
    pred = cal.predict_game(res, season.team_names[0], season.team_names[1])
    assert 0 < pred.p_home < 1 and pred.spread.endswith(f"-{abs(pred.margin):.1f}")


def test_in_sample_calibrator_inflates_sigma_for_small_samples(season):
    model = make_model("massey", ridge_lambda=0.01)
    early = [g for g in season.games if g.week <= 2]
    res = model.fit(season.team_names, early)
    cal = MarginCalibrator.fit(res, early)
    assert cal.sigma > 10  # not fooled by an interpolating fit


def test_records_and_sos_and_momentum(season):
    recs = records(season.team_names, season.games)
    total_wins = sum(r.wins for r in recs.values())
    total_losses = sum(r.losses for r in recs.values())
    assert total_wins == total_losses == len(season.games)
    res = make_model("massey").fit(season.team_names, season.games)
    sos = strength_of_schedule(season.team_names, season.games, res.ratings)
    assert set(sos) == set(season.team_names)
    cal = MarginCalibrator.fit_cv(make_model("massey"), season.team_names, season.games)
    mom = momentum(season.team_names, season.games, res, cal, last_n=3)
    assert set(mom) == set(season.team_names)
    assert abs(np.mean(list(mom.values()))) < 10


def test_records_count_ties():
    recs = records(["A", "B"], [_game(1, "A", "B", 7, 7)])
    assert recs["A"].ties == 1 and recs["A"].wins == 0 and str(recs["A"]) == "0-0-1"


def test_conference_strength_band_and_table(season):
    res = make_model("massey").fit(season.team_names, season.games)
    strength = conference_strength(res.ratings, season.conference_map)
    assert set(strength) == set(season.conference_map.values())
    assert min(strength.values()) == pytest.approx(0.6) and max(strength.values()) == pytest.approx(1.6)
    table = conference_table(res.ratings, season.conference_map, records(season.team_names, season.games))
    assert list(table["strength_index"]) == sorted(table["strength_index"], reverse=True)
    assert table["teams"].sum() == len(season.team_names)
    assert conference_strength({}, {}) == {}


def test_team_table_and_schedule_table(season):
    model = make_model("hybrid")
    res = model.fit(season.team_names, season.games)
    cal = MarginCalibrator.fit_cv(model, season.team_names, season.games)
    table = team_table(
        season.team_names, season.games, res, cal, season.conference_map, ppoints=dict.fromkeys(season.team_names, 1.0)
    )
    assert list(table["rank"]) == list(range(1, len(season.team_names) + 1))
    assert {"team", "conference", "record", "rating", "sos", "sos_rank", "momentum", "ppoints"} <= set(table.columns)
    top = table.iloc[0]["team"]
    sched = schedule_table(top, season.games, res, cal)
    assert len(sched) == records(season.team_names, season.games)[top].games
    assert set(sched["result"]) <= {"W", "L", "T"}
    assert ((sched["win_prob"] > 0) & (sched["win_prob"] < 1)).all()
