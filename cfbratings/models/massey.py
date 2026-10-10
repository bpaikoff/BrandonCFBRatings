"""Massey least-squares ratings.

Each game contributes one equation ``r_home - r_away = margin - HFA`` (the HFA term is
dropped on neutral sites). The system is solved by weighted ridge regression::

    (Aᵀ W A + λ I) r = Aᵀ W y

Because every row of ``A`` sums to zero, the ridge solution is automatically centred
(mean rating 0) -- no ad-hoc constraint row is needed. With a ``prior`` the penalty becomes
``λ‖r - r0‖²`` so ratings shrink toward last season's (regressed) values instead of toward
zero. Weights ``W`` implement optional recency decay with a configurable half-life (weeks
are measured on an absolute timeline, so a game from last season is a year old), and
``max_margin`` caps blow-outs so that running up the score is not rewarded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..data.schema import Game
from .base import Prior, RatingResult, index_teams, prior_vector, usable_games


def build_massey(
    teams: Sequence[str],
    games: Sequence[Game],
    *,
    ridge_lambda: float = 0.01,
    hfa: float = 2.5,
    max_margin: float = 50.0,
    half_life_weeks: float = 0.0,
    prior: Prior | None = None,
    prior_lambda: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normal equations ``(M, b)`` for the ridge-regularised Massey system.

    ``prior_lambda`` is the penalty strength used when a ``prior`` is supplied (a prior
    carries real information, so it deserves more weight than shrinkage toward zero);
    it defaults to ``ridge_lambda``.
    """
    n = len(teams)
    if n == 0:
        return np.zeros((0, 0)), np.zeros(0)
    idx = index_teams(teams)
    usable = usable_games(games, idx)
    r0 = prior_vector(teams, prior, 0.0)
    r0 = r0 - r0.mean()
    lam = max(prior_lambda if (prior and prior_lambda is not None) else ridge_lambda, 1e-9)
    if not usable:
        return np.eye(n) * lam, lam * r0

    latest = max(absolute_week(g) for g in usable)
    A = np.zeros((len(usable), n))
    y = np.zeros(len(usable))
    w = np.ones(len(usable))
    for k, g in enumerate(usable):
        A[k, idx[g.home]] = 1.0
        A[k, idx[g.away]] = -1.0
        adj = g.margin - (0.0 if g.neutral_site else hfa)
        y[k] = float(np.clip(adj, -max_margin, max_margin))
        if half_life_weeks > 0:
            w[k] = 0.5 ** ((latest - absolute_week(g)) / half_life_weeks)
    AW = A * w[:, None]
    M = AW.T @ A + lam * np.eye(n)
    b = AW.T @ y + lam * r0
    return M, b


def absolute_week(g: Game) -> int:
    """Week index on a single timeline across seasons (52 weeks per season)."""
    return g.season * 52 + g.week


def solve_massey(M: np.ndarray, b: np.ndarray) -> np.ndarray:
    if b.size == 0:
        return np.zeros(0)
    try:
        r = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        r = np.linalg.lstsq(M, b, rcond=None)[0]
    return r - r.mean()


@dataclass
class MasseyModel:
    ridge_lambda: float = 0.01
    hfa: float = 2.5
    max_margin: float = 50.0
    half_life_weeks: float = 0.0
    prior_lambda: float = 3.0
    name: str = "massey"

    def fit(self, teams: Sequence[str], games: Sequence[Game], prior: Prior | None = None) -> RatingResult:
        M, b = build_massey(
            teams,
            games,
            ridge_lambda=self.ridge_lambda,
            hfa=self.hfa,
            max_margin=self.max_margin,
            half_life_weeks=self.half_life_weeks,
            prior=prior,
            prior_lambda=self.prior_lambda,
        )
        r = solve_massey(M, b)
        return RatingResult(
            method=self.name,
            ratings={t: float(r[i]) for i, t in enumerate(teams)},
            scale="points",
            params={
                "ridge_lambda": self.ridge_lambda,
                "hfa": self.hfa,
                "max_margin": self.max_margin,
                "half_life_weeks": self.half_life_weeks,
                "prior_lambda": self.prior_lambda,
            },
            extras={"hfa": self.hfa},
        )
