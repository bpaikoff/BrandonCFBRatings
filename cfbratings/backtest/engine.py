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

``games`` may span several seasons. Seasons are evaluated in order and each one is
warm-started with a regressed prior built from the seasons before it (see
:mod:`cfbratings.priors`), so September predictions benefit from last year. A prior for
the first season can be supplied explicitly (``prior=``).

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
from ..priors import WarmStart, preseason_prior
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
    min_train_games: int = 40  # waived for seasons that start from a prior
    include_postseason: bool = True
    calibration_decay: float = 0.85  # weekly weight decay for out-of-sample calibration pairs
    season_regress: float | None = None  # carry-over regression; None → settings.season_regress

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
            seasons = sorted(df["season"].unique())
            out["seasons"] = f"{seasons[0]}-{seasons[-1]}" if len(seasons) > 1 else str(seasons[0])
        return out

    def by_week(self) -> pd.DataFrame:
        """Metrics per week, pooled across seasons when more than one was evaluated."""
        rows = []
        for week, sub in self.predictions.groupby("week", sort=True):
            rows.append({"week": int(week), **summarize_predictions(sub)})
        return pd.DataFrame(rows)

    def by_season(self) -> pd.DataFrame:
        rows = []
        for season, sub in self.predictions.groupby("season", sort=True):
            rows.append({"season": int(season), **summarize_predictions(sub)})
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


def run_backtest(
    teams: Sequence[str],
    games: Sequence[Game],
    config: BacktestConfig | None = None,
    *,
    settings: Settings | None = None,
    prior: Any = None,
    lines: Mapping[int, float] | None = None,
) -> BacktestResult:
    """``prior`` seeds the first season: a rating prior or a :class:`WarmStart` bundle."""
    cfg = config or BacktestConfig()
    s = settings or default_settings
    if cfg.method not in MODEL_NAMES:
        raise ValueError(f"unknown method {cfg.method!r}")
    model = make_model(cfg.method, s, **cfg.params)
    regress = s.season_regress if cfg.season_regress is None else cfg.season_regress
    team_set = set(teams)
    played = sorted(
        (g for g in games if g.has_score and g.home in team_set and g.away in team_set),
        key=lambda g: g.order_key,
    )
    if not cfg.include_postseason:
        played = [g for g in played if g.season_type != "postseason"]
    seasons = sorted({g.season for g in played})
    first_season = seasons[0] if seasons else 0

    rows: list[dict[str, Any]] = []
    oos: list[tuple[int, float, float, float]] = []  # (absolute week, diff, home_ind, margin)
    current_prior: Any = prior.prior if isinstance(prior, WarmStart) else (prior or None)
    carried: MarginCalibrator | None = prior.calibrator if isinstance(prior, WarmStart) else None
    for season in seasons:
        season_games = [g for g in played if g.season == season]
        last_week = max_completed_week(season_games)
        end_week = min(cfg.end_week, last_week) if cfg.end_week else last_week
        offset = (season - first_season) * 52
        for week in range(cfg.start_week, end_week + 1):
            test = [g for g in season_games if g.week == week]
            if not test:
                continue
            train = [g for g in season_games if g.week < week]
            if len(train) < cfg.min_train_games and not current_prior:
                continue
            result = model.fit(teams, train, prior=current_prior)
            cv_cal = MarginCalibrator.fit_cv(
                model, teams, train, fallback_hfa=s.home_field_adv, prior=current_prior, fallback=carried
            )
            cal = rolling_calibrator(cv_cal, oos, offset + week, decay=cfg.calibration_decay)

            train_margins = np.array([g.margin for g in train if not g.neutral_site], dtype=float)
            prior_margin = float(train_margins.mean()) if train_margins.size else s.home_field_adv
            prior_sd = float(train_margins.std()) if train_margins.size > 1 else 16.0
            prior_p = normal_cdf(prior_margin / max(prior_sd, 7.0))

            for g in test:
                pred = cal.predict_game(result, g.home, g.away, g.neutral_site)
                oos.append(
                    (offset + week, result.diff(g.home, g.away), 0.0 if g.neutral_site else 1.0, float(g.margin))
                )
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
        # Carry this season forward into the next one: regressed ratings and the calibrator.
        carried = MarginCalibrator.fit_cv(
            model, teams, season_games, fallback_hfa=s.home_field_adv, prior=current_prior, fallback=carried
        )
        current_prior = preseason_prior(model, teams, season_games, regress=regress, previous_prior=current_prior)
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
    priors: Mapping[str, Any] | None = None,
    lines: Mapping[int, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Backtest several methods on the same games; returns (summary table, results).

    ``priors`` optionally maps method → prior for the first season in ``games``.
    """
    results: dict[str, BacktestResult] = {}
    rows: list[dict[str, Any]] = []
    for m in methods:
        cfg = BacktestConfig(method=m, params=dict((params or {}).get(m, {})), start_week=start_week)
        res = run_backtest(teams, games, cfg, settings=settings, prior=(priors or {}).get(m), lines=lines)
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
            "seasons": base.get("seasons"),
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
    cols = ["method", "n", "accuracy", "brier", "log_loss", "mae", "rmse", "seasons", "weeks"]
    extra = [c for c in ("ats_accuracy", "vegas_mae") if any(c in r for r in rows)]
    table = pd.DataFrame(rows)
    table = table[[c for c in cols + extra if c in table.columns]]
    return table, results
