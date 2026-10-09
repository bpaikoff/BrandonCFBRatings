"""Massey least-squares ratings.

Each game contributes one equation ``r_home - r_away = margin - HFA`` (the HFA term is
dropped on neutral sites). The system is solved by weighted ridge regression::

    (Aᵀ W A + λ I) r = Aᵀ W y

Because every row of ``A`` sums to zero, the ridge solution is automatically centred
(mean rating 0) -- no ad-hoc constraint row is needed. Weights ``W`` implement optional
recency decay with a configurable half-life, and ``max_margin`` caps blow-outs so that
running up the score is not rewarded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..data.schema import Game
from .base import RatingResult, index_teams, usable_games


def build_massey(
    teams: Sequence[str],
    games: Sequence[Game],
    *,
    ridge_lambda: float = 0.01,
    hfa: float = 2.5,
    max_margin: float = 50.0,
    half_life_weeks: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normal equations ``(M, b)`` for the ridge-regularised Massey system."""
    n = len(teams)
    if n == 0:
        return np.zeros((0, 0)), np.zeros(0)
    idx = index_teams(teams)
    usable = usable_games(games, idx)
    if not usable:
        return np.eye(n) * max(ridge_lambda, 1e-9), np.zeros(n)

    latest_week = max(g.week for g in usable)
    A = np.zeros((len(usable), n))
    y = np.zeros(len(usable))
    w = np.ones(len(usable))
    for k, g in enumerate(usable):
        A[k, idx[g.home]] = 1.0
        A[k, idx[g.away]] = -1.0
        adj = g.margin - (0.0 if g.neutral_site else hfa)
        y[k] = float(np.clip(adj, -max_margin, max_margin))
        if half_life_weeks > 0:
            w[k] = 0.5 ** ((latest_week - g.week) / half_life_weeks)
    AW = A * w[:, None]
    M = AW.T @ A + ridge_lambda * np.eye(n)
    b = AW.T @ y
    return M, b


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
    name: str = "massey"

    def fit(self, teams: Sequence[str], games: Sequence[Game]) -> RatingResult:
        M, b = build_massey(
            teams,
            games,
            ridge_lambda=self.ridge_lambda,
            hfa=self.hfa,
            max_margin=self.max_margin,
            half_life_weeks=self.half_life_weeks,
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
            },
            extras={"hfa": self.hfa},
        )
