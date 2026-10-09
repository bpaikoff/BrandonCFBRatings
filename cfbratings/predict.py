"""Turn any model's ratings into point spreads and win probabilities.

Rating systems live on different scales (Colley ∈ [0,1], Massey in points, Elo around
1500). To compare them -- and to backtest them -- every model is run through the same
one-dimensional calibration::

    margin_home ≈ intercept * [not neutral] + slope * (r_home - r_away)
    P(home win)  = Φ(margin_home / sigma)

``Φ`` is the standard normal CDF; football margins are close to normal with a standard
deviation of roughly 15-17 points, so a probit link is the natural choice.

Two ways to fit the calibrator are provided:

* :meth:`MarginCalibrator.fit` -- least squares on the games the model was trained on.
  In-sample residuals of a model with one parameter per team are optimistic, so the
  residual variance is blended with a prior using a degrees-of-freedom correction.
* :meth:`MarginCalibrator.fit_cv` -- k-fold cross-validated pairs (each game predicted by
  a model that never saw it). Used for live rankings and as the backtest prior.
* :meth:`MarginCalibrator.from_arrays` with *out-of-sample* pairs (what the backtest
  engine does once a few weeks of held-out predictions exist).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .data.schema import Game
from .models.base import RatingModel, RatingResult

MIN_GAMES_FOR_FIT = 12
SIGMA_FLOOR = 7.0
DEFAULT_SIGMA = 16.0
PRIOR_WEIGHT = 20.0  # pseudo-observations backing the prior slope / intercept / sigma


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class Prediction:
    home: str
    away: str
    neutral: bool
    margin: float  # expected home margin, points
    p_home: float

    @property
    def favorite(self) -> str:
        return self.home if self.margin >= 0 else self.away

    @property
    def spread(self) -> str:
        """Market-style line from the favourite's perspective, e.g. ``'Ohio State -6.5'``."""
        return f"{self.favorite} -{abs(self.margin):.1f}"


@dataclass(frozen=True)
class MarginCalibrator:
    slope: float
    intercept: float  # home-field advantage in points
    sigma: float
    n_games: int = 0
    source: str = "prior"  # "prior" | "in_sample" | "cross_val" | "out_of_sample"

    # --- fitting -----------------------------------------------------------------
    @staticmethod
    def prior_for(result: RatingResult, fallback_hfa: float = 2.5) -> tuple[float, float]:
        """``(slope, intercept)`` prior implied by the model's scale."""
        if result.scale == "points":
            return 1.0, float(result.extras.get("hfa", fallback_hfa))
        if result.scale == "elo":
            return 1.0 / float(result.extras.get("elo_per_point", 25.0)), fallback_hfa
        values = np.array(list(result.ratings.values()), dtype=float)
        sd = float(values.std()) if values.size else 1.0
        # One standard deviation of rating ≈ 10 points is a reasonable neutral prior.
        return (10.0 / sd if sd > 1e-9 else 1.0), fallback_hfa

    @classmethod
    def from_arrays(
        cls,
        diffs: Sequence[float],
        home_ind: Sequence[float],
        margins: Sequence[float],
        *,
        prior_slope: float,
        prior_intercept: float,
        prior_sigma: float = DEFAULT_SIGMA,
        prior_weight: float = PRIOR_WEIGHT,
        weights: Sequence[float] | None = None,
        dof: int = 0,
        max_margin: float | None = 50.0,
        source: str = "out_of_sample",
    ) -> MarginCalibrator:
        """Bayesian-flavoured linear fit of ``margin ~ intercept*home + slope*diff``.

        ``prior_weight`` pseudo-observations pull the coefficients toward the prior and
        ``dof`` parameters are subtracted from the residual degrees of freedom (use the
        number of teams for an in-sample fit).
        """
        d = np.asarray(diffs, dtype=float)
        h = np.asarray(home_ind, dtype=float)
        y = np.asarray(margins, dtype=float)
        n = int(d.size)
        if n < MIN_GAMES_FOR_FIT:
            return cls(prior_slope, prior_intercept, prior_sigma, n, "prior")
        if max_margin is not None:
            y = np.clip(y, -max_margin, max_margin)
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        X = np.column_stack([h, d])
        XtW = X.T * w
        scale = np.array([max(float(np.mean(h * h)), 1e-6), max(float(np.var(d)), 1e-9)])
        lam = prior_weight * scale
        prior = np.array([prior_intercept, prior_slope])
        beta = np.linalg.solve(XtW @ X + np.diag(lam), XtW @ y + lam * prior)
        resid = y - X @ beta
        rss = float(np.sum(w * resid**2))
        eff_n = float(np.sum(w))
        var = (rss + prior_weight * prior_sigma**2) / (max(eff_n - dof, 1.0) + prior_weight)
        return cls(
            slope=float(beta[1]),
            intercept=float(beta[0]),
            sigma=max(math.sqrt(var), SIGMA_FLOOR),
            n_games=n,
            source=source,
        )

    @classmethod
    def fit(
        cls,
        result: RatingResult,
        games: Sequence[Game],
        *,
        fallback_hfa: float = 2.5,
        max_margin: float | None = 50.0,
    ) -> MarginCalibrator:
        """In-sample fit on the games the model was trained on."""
        usable = [g for g in games if g.has_score and g.home in result.ratings and g.away in result.ratings]
        prior_slope, prior_intercept = cls.prior_for(result, fallback_hfa)
        teams_seen = {g.home for g in usable} | {g.away for g in usable}
        return cls.from_arrays(
            [result.diff(g.home, g.away) for g in usable],
            [0.0 if g.neutral_site else 1.0 for g in usable],
            [float(g.margin) for g in usable],
            prior_slope=prior_slope,
            prior_intercept=prior_intercept,
            dof=len(teams_seen),
            max_margin=max_margin,
            source="in_sample",
        )

    @classmethod
    def fit_cv(
        cls,
        model: RatingModel,
        teams: Sequence[str],
        games: Sequence[Game],
        *,
        folds: int = 5,
        seed: int = 0,
        fallback_hfa: float = 2.5,
        max_margin: float | None = 50.0,
    ) -> MarginCalibrator:
        """Cross-validated fit: every game is predicted by a model that did not see it.

        This is the recommended calibrator for live rankings. It removes the optimism of
        in-sample residuals (and of an in-sample slope of 1.0 for Massey early in the
        season) at the cost of ``folds`` extra model fits, which take milliseconds.
        """
        team_set = set(teams)
        usable = [g for g in games if g.has_score and g.home in team_set and g.away in team_set]
        full = model.fit(teams, usable)
        if len(usable) < 2 * MIN_GAMES_FOR_FIT or folds < 2:
            return cls.fit(full, usable, fallback_hfa=fallback_hfa, max_margin=max_margin)
        prior_slope, prior_intercept = cls.prior_for(full, fallback_hfa)
        rng = np.random.default_rng(seed)
        fold_id = rng.permutation(len(usable)) % folds
        diffs: list[float] = []
        home: list[float] = []
        margins: list[float] = []
        for f in range(folds):
            train = [g for g, fid in zip(usable, fold_id, strict=True) if fid != f]
            held = [g for g, fid in zip(usable, fold_id, strict=True) if fid == f]
            res = model.fit(teams, train)
            for g in held:
                diffs.append(res.diff(g.home, g.away))
                home.append(0.0 if g.neutral_site else 1.0)
                margins.append(float(g.margin))
        return cls.from_arrays(
            diffs,
            home,
            margins,
            prior_slope=prior_slope,
            prior_intercept=prior_intercept,
            max_margin=max_margin,
            source="cross_val",
        )

    # --- prediction --------------------------------------------------------------
    def predict_margin(self, r_home: float, r_away: float, neutral: bool = False) -> float:
        return (0.0 if neutral else self.intercept) + self.slope * (r_home - r_away)

    def win_prob(self, r_home: float, r_away: float, neutral: bool = False) -> float:
        return normal_cdf(self.predict_margin(r_home, r_away, neutral) / self.sigma)

    def win_prob_from_margin(self, margin: float) -> float:
        return normal_cdf(margin / self.sigma)

    def predict_game(self, result: RatingResult, home: str, away: str, neutral: bool = False) -> Prediction:
        margin = self.predict_margin(result.get(home), result.get(away), neutral)
        return Prediction(
            home=home, away=away, neutral=neutral, margin=margin, p_home=self.win_prob_from_margin(margin)
        )
