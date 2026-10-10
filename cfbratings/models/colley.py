"""Colley Matrix method.

Colley (2002) rates teams purely from wins and losses with a Laplace prior::

    (2 + n_i) r_i - sum_{j in opponents(i)} r_j = 1 + (w_i - l_i) / 2

The ``prior_strength`` parameter generalises the ``2`` (two pseudo-games against an
average team). With a ``prior`` the right-hand side becomes ``prior_strength * r0_i +
(w_i - l_i)/2``, i.e. the pseudo-games are played at the team's prior rating ``r0_i``
instead of 0.5 (useful for carrying last season forward). The system is symmetric
positive-definite; isolated teams simply keep their prior rating.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..data.schema import Game
from .base import Prior, RatingResult, index_teams, prior_vector, usable_games


def build_colley(
    teams: Sequence[str],
    games: Sequence[Game],
    prior_strength: float = 2.0,
    prior: Prior | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(C, b)`` such that ``C r = b`` gives Colley ratings."""
    n = len(teams)
    if n == 0:
        return np.zeros((0, 0)), np.zeros(0)
    idx = index_teams(teams)
    played = np.zeros(n)
    net = np.zeros(n)  # wins - losses (ties contribute 0)
    opp = np.zeros((n, n))
    for g in usable_games(games, idx):
        i, j = idx[g.home], idx[g.away]
        played[i] += 1
        played[j] += 1
        opp[i, j] += 1
        opp[j, i] += 1
        m = g.margin
        if m > 0:
            net[i] += 1
            net[j] -= 1
        elif m < 0:
            net[j] += 1
            net[i] -= 1
    C = np.diag(prior_strength + played) - opp
    b = prior_strength * prior_vector(teams, prior, 0.5) + 0.5 * net
    return C, b


def solve_colley(C: np.ndarray, b: np.ndarray) -> np.ndarray:
    if b.size == 0:
        return np.zeros(0)
    try:
        return np.linalg.solve(C, b)
    except np.linalg.LinAlgError:
        # Only reachable with prior_strength == 0 and an isolated team; fall back to lstsq.
        return np.linalg.lstsq(C, b, rcond=None)[0]


@dataclass
class ColleyModel:
    prior_strength: float = 2.0
    name: str = "colley"

    def fit(self, teams: Sequence[str], games: Sequence[Game], prior: Prior | None = None) -> RatingResult:
        C, b = build_colley(teams, games, self.prior_strength, prior=prior)
        r = solve_colley(C, b)
        return RatingResult(
            method=self.name,
            ratings={t: float(r[i]) for i, t in enumerate(teams)},
            scale="prob",
            params={"prior_strength": self.prior_strength},
        )
