"""Bradley-Terry ratings with a home-advantage term, fitted by penalised maximum likelihood.

    P(home wins) = sigmoid(r_home - r_away + h * [not neutral])

The log-likelihood (ties count as half a win each way) is maximised with Newton's method
under an L2 penalty ``ridge`` on the team parameters (toward zero, or toward a supplied
``prior``), which keeps undefeated and winless teams finite early in the season. Ratings
are centred at zero; ``h`` is reported in ``extras["home_adv_logit"]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..data.schema import Game
from .base import Prior, RatingResult, index_teams, prior_vector, usable_games


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def fit_bradley_terry(
    teams: Sequence[str],
    games: Sequence[Game],
    *,
    ridge: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-8,
    prior: Prior | None = None,
) -> tuple[np.ndarray, float]:
    """Return ``(ratings, home_advantage)`` on the log-odds scale."""
    n = len(teams)
    if n == 0:
        return np.zeros(0), 0.0
    idx = index_teams(teams)
    usable = usable_games(games, idx)
    theta0 = np.append(prior_vector(teams, prior, 0.0), 0.0)
    theta0[:n] -= theta0[:n].mean()
    if not usable:
        return theta0[:n], 0.0

    m = len(usable)
    X = np.zeros((m, n + 1))
    y = np.zeros(m)
    for k, g in enumerate(usable):
        X[k, idx[g.home]] = 1.0
        X[k, idx[g.away]] = -1.0
        X[k, n] = 0.0 if g.neutral_site else 1.0
        y[k] = g.home_result

    theta = theta0.copy()
    penalty = np.full(n + 1, ridge)
    penalty[n] = 1e-3  # keep the home term nearly unpenalised but well-posed
    for _ in range(max_iter):
        p = _sigmoid(X @ theta)
        grad = X.T @ (y - p) - penalty * (theta - theta0)
        w = p * (1.0 - p)
        H = (X * w[:, None]).T @ X + np.diag(penalty)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        theta = theta + step
        if float(np.max(np.abs(step))) < tol:
            break
    r = theta[:n]
    return r - r.mean(), float(theta[n])


@dataclass
class BradleyTerryModel:
    ridge: float = 1.0
    name: str = "bradley_terry"

    def fit(self, teams: Sequence[str], games: Sequence[Game], prior: Prior | None = None) -> RatingResult:
        r, h = fit_bradley_terry(teams, games, ridge=self.ridge, prior=prior)
        return RatingResult(
            method=self.name,
            ratings={t: float(r[i]) for i, t in enumerate(teams)},
            scale="logit",
            params={"ridge": self.ridge},
            extras={"home_adv_logit": h},
        )
