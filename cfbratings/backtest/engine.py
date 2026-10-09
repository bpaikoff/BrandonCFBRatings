"""Walk-forward backtesting.

For each evaluation week ``w`` the model is fitted on every completed game *before*
``w`` (optionally preceded by warm-up games from earlier seasons), a
:class:`~cfbratings.predict.MarginCalibrator` is fitted on the same training games, and
the games *in* week ``w`` are predicted out-of-sample. Predictions are pooled across
weeks and scored with the metrics in :mod:`cfbratings.backtest.metrics`.

Calibration is honest at every step: the prior for week ``w`` is a k-fold
cross-validated fit on the training window, and once enough held-out predictions from
earlier weeks have accumulated the calibrator is refitted on those out-of-sample pairs
(recent weeks weighted more heavily). A one-parameter-per-team model would otherwise look
far more certain than it is early in the season.

Two reference points are always reported alongside the model:

* ``home_prior`` -- always pick the home team with the training-set home win rate and
  mean home margin (a model that cannot beat this is useless);
* ``vegas`` -- the closing spread, when ``lines`` are supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..config import settings as default_settings
from ..data.schema import Game, max_completed_week
from ..models import MODEL_NAMES, make_model
from ..predict import MarginCalibrator, normal_cdf
from .metrics import calibration_table, summarize_predictions

PREDICTION_COLUMNS = [
    "season",
    "week",
    "game_id",
    "home",
    "away",
    "neutral",
    "rating_home",
    "rating_away",
    "pred_margin",
    "p_home",
    "actual_margin",
    "home_win",
    "correct",
    "prior_margin",
    "prior_p_home",
    "vegas_margin",
]


@dataclass(frozen=True)
class BacktestConfig:
    method: str = "hybrid"
    params: dict[str, Any] = field(default_factory=dict)
    start_week: int = 4
    end_week: int | None = None
    min_train_games: int = 40
    include_postseason: bool = True
    calibration_decay: float = 0.85  # weekly weight decay for out-of-sample calibration pairs

    def label(self) -> str:
        if not self.params:
            return self.method
        inner = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.method}({inner})"


@dataclass
class BacktestResult:
    config: BacktestConfig
    predictions: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        df = self.predictions
        out: dict[str, Any] = {"method": self.config.method, **summarize_predictions(df)}
        if not df.empty:
            prior = df[["prior_p_home", "prior_margin", "home_win", "actual_margin"]].rename(
                columns={"prior_p_home": "p_home", "prior_margin": "pred_margin"}
            )
            out.update(summarize_predictions(prior, prefix="baseline_"))
            out["weeks"] = f"{int(df['week'].min())}-{int(df['week'].max())}"
        return out

    def by_week(self) -> pd.DataFrame:
        rows = []
        for week, sub in self.predictions.groupby("week", sort=True):
            rows.append({"week": int(week), **summarize_predictions(sub)})
        return pd.DataFrame(rows)

    def calibration(self, bins: int = 10) -> pd.DataFrame:
        return calibration_table(self.predictions["p_home"], self.predictions["home_win"], bins=bins)


def rolling_calibrator(
    prior: MarginCalibrator,
    pairs: Sequence[tuple[int, float, float, float]],
    week: int,
    *,
    decay: float = 0.85,
    min_pairs: int = 30,
) -> MarginCalibrator:
    """Refit the calibrator on held-out ``(week, diff, home_ind, margin)`` pairs.

    ``prior`` (normally the cross-validated fit) anchors the coefficients, so the two
    blend smoothly as evidence accrues.
    """
    if len(pairs) < min_pairs:
        return prior
    weeks = np.array([p[0] for p in pairs], dtype=float)
    weights = decay ** (week - weeks)
    return MarginCalibrator.from_arrays(
        [p[1] for p in pairs],
        [p[2] for p in pairs],
        [p[3] for p in pairs],
        prior_slope=prior.slope,
        prior_intercept=prior.intercept,
        prior_sigma=prior.sigma,
        prior_weight=10.0,
        weights=weights,
        source="out_of_sample",
    )


def _warm_start(method: str, model, teams: Sequence[str], warmup: Sequence[Game], train: Sequence[Game]):
    """Fit with warm-up games prepended; Elo regresses between seasons automatically."""
    if warmup:
        return model.fit(teams, [*warmup, *train])
    return model.fit(teams, train)


def run_backtest(
    teams: Sequence[str],
    games: Sequence[Game],
    config: BacktestConfig | None = None,
    *,
    settings: Settings | None = None,
    warmup_games: Sequence[Game] | None = None,
    lines: Mapping[int, float] | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    s = settings or default_settings
    if cfg.method not in MODEL_NAMES:
        raise ValueError(f"unknown method {cfg.method!r}")
    model = make_model(cfg.method, s, **cfg.params)
    team_set = set(teams)
    played = sorted(
        (g for g in games if g.has_score and g.home in team_set and g.away in team_set),
        key=lambda g: g.order_key,
    )
    if not cfg.include_postseason:
        played = [g for g in played if g.season_type != "postseason"]
    warmup = list(warmup_games or [])
    last_week = max_completed_week(played)
    end_week = min(cfg.end_week, last_week) if cfg.end_week else last_week

    rows: list[dict[str, Any]] = []
    oos: list[tuple[int, float, float, float]] = []  # (week, diff, home_ind, margin) held-out pairs
    for week in range(cfg.start_week, end_week + 1):
        test = [g for g in played if g.week == week]
        if not test:
            continue
        train = [g for g in played if g.week < week]
        if len(train) + len(warmup) < cfg.min_train_games:
            continue
        result = _warm_start(cfg.method, model, teams, warmup, train)
        cv_cal = MarginCalibrator.fit_cv(model, teams, [*warmup, *train], fallback_hfa=s.home_field_adv)
        cal = rolling_calibrator(cv_cal, oos, week, decay=cfg.calibration_decay)

        train_margins = np.array([g.margin for g in train if not g.neutral_site], dtype=float)
        prior_margin = float(train_margins.mean()) if train_margins.size else s.home_field_adv
        prior_sd = float(train_margins.std()) if train_margins.size > 1 else 16.0
        prior_p = normal_cdf(prior_margin / max(prior_sd, 7.0))

        for g in test:
            pred = cal.predict_game(result, g.home, g.away, g.neutral_site)
            oos.append((week, result.diff(g.home, g.away), 0.0 if g.neutral_site else 1.0, float(g.margin)))
            vegas = None
            if lines and g.id in lines:
                vegas = -float(lines[g.id])
            rows.append(
                {
                    "season": g.season,
                    "week": g.week,
                    "game_id": g.id,
                    "home": g.home,
                    "away": g.away,
                    "neutral": g.neutral_site,
                    "rating_home": result.get(g.home),
                    "rating_away": result.get(g.away),
                    "pred_margin": pred.margin,
                    "p_home": pred.p_home,
                    "actual_margin": g.margin,
                    "home_win": g.home_result,
                    "correct": (pred.p_home >= 0.5) == (g.margin > 0) if g.margin != 0 else np.nan,
                    "prior_margin": 0.0 if g.neutral_site else prior_margin,
                    "prior_p_home": 0.5 if g.neutral_site else prior_p,
                    "vegas_margin": vegas,
                }
            )
    df = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    return BacktestResult(cfg, df)


def compare_methods(
    teams: Sequence[str],
    games: Sequence[Game],
    methods: Sequence[str] = MODEL_NAMES,
    *,
    start_week: int = 4,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    settings: Settings | None = None,
    warmup_games: Sequence[Game] | None = None,
    lines: Mapping[int, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Backtest several methods on the same games; returns (summary table, results)."""
    results: dict[str, BacktestResult] = {}
    rows: list[dict[str, Any]] = []
    for m in methods:
        cfg = BacktestConfig(method=m, params=dict((params or {}).get(m, {})), start_week=start_week)
        res = run_backtest(teams, games, cfg, settings=settings, warmup_games=warmup_games, lines=lines)
        results[m] = res
        rows.append(res.summary())
    if results:
        first = next(iter(results.values()))
        base = first.summary()
        baseline = {
            "method": "home_prior (baseline)",
            "n": base.get("baseline_n"),
            "accuracy": base.get("baseline_accuracy"),
            "brier": base.get("baseline_brier"),
            "log_loss": base.get("baseline_log_loss"),
            "mae": base.get("baseline_mae"),
            "rmse": base.get("baseline_rmse"),
            "weeks": base.get("weeks"),
        }
        rows.append(baseline)
        if lines:
            df = first.predictions.dropna(subset=["vegas_margin"])
            if not df.empty:
                vegas = df.assign(
                    p_home=[normal_cdf(m / 16.0) for m in df["vegas_margin"]],
                    pred_margin=df["vegas_margin"],
                )
                rows.append({"method": "vegas (closing line)", **summarize_predictions(vegas)})
    cols = ["method", "n", "accuracy", "brier", "log_loss", "mae", "rmse", "weeks"]
    extra = [c for c in ("ats_accuracy", "vegas_mae") if any(c in r for r in rows)]
    table = pd.DataFrame(rows)
    table = table[[c for c in cols + extra if c in table.columns]]
    return table, results
