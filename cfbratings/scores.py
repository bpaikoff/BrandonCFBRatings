"""Projected scores.

The rating models and :class:`~cfbratings.predict.MarginCalibrator` say *by how much* a team
should win. Turning that into a score needs one more number, the game's **total**, and that
comes from the games already in hand -- no extra API call. Every completed game contributes
two observations,

    home points = mu + O_home + D_away + h,
    away points = mu + O_away + D_home - h,

where ``mu`` is the league's mean points per team per game, ``O`` is a team's offence (points
scored above the mean, adjusted for the defences it faced), ``D`` its defence (points *allowed*
above the mean, so positive is leaky) and ``h`` half the home-field advantage in points. ``O``
and ``D`` are fitted by ridge least squares, which shrinks a team with few games toward
average. The projected total of a matchup is then ``2 mu + O_home + D_away + O_away + D_home``,
and a projected score splits that total around the calibrated margin::

    home = (total + margin) / 2,   away = (total - margin) / 2.

The spread is the better-calibrated of the two numbers (it is what the ratings are fitted to),
so the margin is always taken from the ratings and only the total from this model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .data.schema import Game

#: A plausible mean points per team per game when no game has been played yet.
DEFAULT_MEAN_POINTS = 27.5
#: Residual standard deviation of a game's total when there are too few games to measure it.
DEFAULT_TOTAL_SIGMA = 17.0
#: The loser of a projected game is never projected below this many points.
MIN_LOSER_POINTS = 3.0


@dataclass(frozen=True)
class ProjectedScore:
    home: str
    away: str
    neutral: bool
    margin: float  # expected home margin (from the ratings)
    total: float  # expected points in the game (from the score model)
    home_points: float
    away_points: float

    @property
    def favorite(self) -> str:
        return self.home if self.margin >= 0 else self.away

    def rounded(self) -> tuple[int, int]:
        """Integer scores whose order agrees with the favourite (no accidental tie)."""
        h, a = round(self.home_points), round(self.away_points)
        if h == a and self.margin != 0:
            if self.margin > 0:
                h += 1
            else:
                a += 1
        return int(h), int(a)

    @property
    def text(self) -> str:
        """``'31-24'`` in home-away order."""
        h, a = self.rounded()
        return f"{h}-{a}"

    @property
    def text_by_favorite(self) -> str:
        """``'Ohio State 31, Michigan 24'``: the favourite first."""
        h, a = self.rounded()
        if self.margin >= 0:
            return f"{self.home} {h}, {self.away} {a}"
        return f"{self.away} {a}, {self.home} {h}"


@dataclass(frozen=True)
class ScoreModel:
    """Offence/defence points model fitted on completed games (see the module docstring)."""

    mean_points: float = DEFAULT_MEAN_POINTS
    half_hfa: float = 1.25  # the home team's share of the home-field advantage, points
    offense: Mapping[str, float] = field(default_factory=dict)
    defense: Mapping[str, float] = field(default_factory=dict)
    total_sigma: float = DEFAULT_TOTAL_SIGMA
    n_games: int = 0

    def expected_points(self, team: str, opponent: str, *, home: bool | None) -> float:
        """Points ``team`` is expected to score against ``opponent`` (``home`` ``None`` = neutral)."""
        edge = 0.0 if home is None else (self.half_hfa if home else -self.half_hfa)
        return self.mean_points + self.offense.get(team, 0.0) + self.defense.get(opponent, 0.0) + edge

    def total(self, home: str, away: str) -> float:
        """Expected points in the game (home-field shifts points between the teams, not the total)."""
        return (
            2.0 * self.mean_points
            + self.offense.get(home, 0.0)
            + self.defense.get(away, 0.0)
            + self.offense.get(away, 0.0)
            + self.defense.get(home, 0.0)
        )

    def project(self, home: str, away: str, margin: float, *, neutral: bool = False) -> ProjectedScore:
        """Split the projected total around a margin that comes from the ratings."""
        total = max(self.total(home, away), abs(margin) + 2.0 * MIN_LOSER_POINTS)
        return ProjectedScore(
            home=home,
            away=away,
            neutral=neutral,
            margin=margin,
            total=total,
            home_points=(total + margin) / 2.0,
            away_points=(total - margin) / 2.0,
        )

    def sample(
        self, home: str, away: str, margin: float, margin_sigma: float, rng: np.random.Generator, *, neutral=False
    ) -> ProjectedScore:
        """One random score: the margin and the total are drawn around their projections."""
        m = float(rng.normal(margin, margin_sigma))
        if round(m) == 0:  # no ties in college football: the nearer side wins by one
            m = 1.0 if m >= 0 else -1.0
        t = max(float(rng.normal(self.total(home, away), self.total_sigma)), abs(m) + 2.0 * MIN_LOSER_POINTS)
        return ProjectedScore(home, away, neutral, m, t, (t + m) / 2.0, (t - m) / 2.0)


def fit_score_model(teams: Sequence[str], games: Sequence[Game], *, hfa: float = 2.5, ridge: float = 4.0) -> ScoreModel:
    """Fit offence and defence points-above-average for ``teams`` from the completed ``games``.

    ``hfa`` is the home-field advantage in points (the calibrator's intercept); half of it is
    credited to the home team's points and debited from the away team's. ``ridge`` is the
    penalty on each offence/defence term in points²·games: a team with few games is shrunk
    toward the league mean, a team with many is not.
    """
    index = {t: i for i, t in enumerate(teams)}
    played = [g for g in games if g.has_score and g.home in index and g.away in index]
    if not played:
        return ScoreModel(half_hfa=hfa / 2.0)
    n = len(teams)
    half = hfa / 2.0
    pts = np.array([[g.home_points, g.away_points] for g in played], dtype=float)
    mu = float(pts.mean())
    # Two rows per game: home points and away points, both centred on mu and the home edge.
    rows = 2 * len(played)
    X = np.zeros((rows, 2 * n))
    y = np.zeros(rows)
    for k, g in enumerate(played):
        h, a = index[g.home], index[g.away]
        edge = 0.0 if g.neutral_site else half
        X[2 * k, h] = 1.0  # home offence
        X[2 * k, n + a] = 1.0  # away defence
        y[2 * k] = pts[k, 0] - mu - edge
        X[2 * k + 1, a] = 1.0  # away offence
        X[2 * k + 1, n + h] = 1.0  # home defence
        y[2 * k + 1] = pts[k, 1] - mu + edge
    beta = np.linalg.solve(X.T @ X + ridge * np.eye(2 * n), X.T @ y)
    offense = {t: float(beta[i]) for t, i in index.items()}
    defense = {t: float(beta[n + i]) for t, i in index.items()}
    totals = pts.sum(axis=1)
    fitted = np.array(
        [
            2 * mu + beta[index[g.home]] + beta[n + index[g.away]] + beta[index[g.away]] + beta[n + index[g.home]]
            for g in played
        ]
    )
    resid = totals - fitted
    sigma = float(resid.std(ddof=1)) if len(played) >= 20 else DEFAULT_TOTAL_SIGMA
    return ScoreModel(
        mean_points=mu,
        half_hfa=half,
        offense=offense,
        defense=defense,
        total_sigma=max(sigma, 6.0),
        n_games=len(played),
    )


__all__ = [
    "DEFAULT_MEAN_POINTS",
    "DEFAULT_TOTAL_SIGMA",
    "MIN_LOSER_POINTS",
    "ProjectedScore",
    "ScoreModel",
    "fit_score_model",
]
