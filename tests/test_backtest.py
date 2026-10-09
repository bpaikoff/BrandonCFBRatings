from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfbratings.backtest import (
    BacktestConfig,
    accuracy,
    brier_score,
    calibration_table,
    compare_methods,
    grid_search,
    log_loss,
    mae,
    rmse,
    run_backtest,
    summarize_predictions,
)
from cfbratings.backtest.engine import PREDICTION_COLUMNS, rolling_calibrator
from cfbratings.predict import MarginCalibrator


def test_metrics_basics():
    assert brier_score([1, 0], [1, 0]) == 0.0
    assert brier_score([0.5], [1]) == 0.25
    assert accuracy([0.9, 0.2, 0.6], [1, 0, 0]) == pytest.approx(2 / 3)
    assert np.isnan(accuracy([0.5], [0.5]))
    assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(np.log(2))
    assert mae([3, -3], [0, 0]) == 3 and rmse([3, -3], [0, 0]) == 3
    table = calibration_table([0.05, 0.95, 0.95], [0, 1, 0], bins=10)
    assert table["n"].sum() == 3 and table.iloc[-1]["observed"] == 0.5
    assert summarize_predictions(pd.DataFrame())["n"] == 0


def test_backtest_beats_home_baseline_on_synthetic_data(season):
    res = run_backtest(season.team_names, season.games, BacktestConfig(method="massey", start_week=4))
    df = res.predictions
    assert list(df.columns) == PREDICTION_COLUMNS
    assert df["week"].min() >= 4 and len(df) > 100
    assert df["p_home"].between(0, 1).all()
    s = res.summary()
    assert s["log_loss"] < s["baseline_log_loss"]
    assert s["mae"] < s["baseline_mae"]
    assert s["accuracy"] > 0.6
    weekly = res.by_week()
    assert list(weekly["week"]) == sorted(df["week"].unique())
    cal = res.calibration(5)
    assert cal["n"].sum() == len(df)


def test_backtest_respects_week_bounds_and_postseason(season):
    cfg = BacktestConfig(method="elo", start_week=6, end_week=8)
    df = run_backtest(season.team_names, season.games, cfg).predictions
    assert df["week"].min() == 6 and df["week"].max() == 8
    last = max(g.week for g in season.games)
    with_bowls = run_backtest(
        season.team_names, season.games, BacktestConfig(method="elo", start_week=last)
    ).predictions
    without = run_backtest(
        season.team_names, season.games, BacktestConfig(method="elo", start_week=last, include_postseason=False)
    ).predictions
    assert len(with_bowls) > 0 and len(without) == 0
    assert cfg.label() == "elo"
    assert BacktestConfig(method="elo", params={"k": 30}).label() == "elo(k=30)"
    with pytest.raises(ValueError):
        run_backtest(season.team_names, season.games, BacktestConfig(method="nope"))


def test_compare_methods_and_vegas_hook(season):
    lines = {
        g.id: -float(season.true_strength[g.home] - season.true_strength[g.away] + (0 if g.neutral_site else 2.5))
        for g in season.games
    }
    table, results = compare_methods(season.team_names, season.games, ["massey", "colley"], start_week=5, lines=lines)
    assert list(table["method"])[:2] == ["massey", "colley"]
    assert "home_prior (baseline)" in set(table["method"]) and "vegas (closing line)" in set(table["method"])
    assert {"accuracy", "brier", "log_loss", "mae", "rmse", "ats_accuracy", "vegas_mae"} <= set(table.columns)
    oracle = table[table["method"] == "vegas (closing line)"].iloc[0]
    assert oracle["mae"] < table.iloc[0]["mae"]  # the oracle line knows the true strengths
    assert results["massey"].predictions["vegas_margin"].notna().all()


def test_grid_search_sorts_best_first(season):
    df = grid_search(season.team_names, season.games, "elo", {"k": [10, 25, 60]}, metric="log_loss", start_week=5)
    assert list(df["k"]) != [] and df["log_loss"].is_monotonic_increasing
    df2 = grid_search(season.team_names, season.games, "elo", {"k": [10, 60]}, metric="accuracy", start_week=5)
    assert df2["accuracy"].is_monotonic_decreasing
    assert grid_search(season.team_names, season.games, "elo", {"k": []}).empty


def test_rolling_calibrator_blends_toward_evidence():
    prior = MarginCalibrator(slope=1.0, intercept=2.5, sigma=16.0, n_games=50, source="cross_val")
    rng = np.random.default_rng(0)
    pairs = []
    for week in range(1, 9):
        for _ in range(10):
            d = rng.normal(0, 10)
            pairs.append((week, d, 1.0, 0.5 * d + 2.5 + rng.normal(0, 5)))  # true slope is 0.5
    assert rolling_calibrator(prior, pairs[:10], 2) is prior
    cal = rolling_calibrator(prior, pairs, 9)
    assert cal.source == "out_of_sample" and 0.4 < cal.slope < 0.8 and cal.sigma < 12
