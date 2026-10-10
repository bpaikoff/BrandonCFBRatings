"""Common types for rating models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..data.schema import Game, is_pool_team

Ratings = dict[str, float]
Prior = Mapping[str, float]


@dataclass
class RatingResult:
    """Ratings produced by a model plus enough metadata to interpret them.

    ``scale`` documents the units of ``ratings``:

    * ``"points"`` -- differences are expected point margins (Massey)
    * ``"elo"``    -- Elo points (divide by ``elo_per_point`` for a spread)
    * ``"prob"``   -- Colley's [0, 1] win-probability-like scale
    * ``"logit"``  -- Bradley-Terry log-odds
    * ``"z"``      -- standardised blend (Hybrid)
    """

    method: str
    ratings: Ratings
    scale: str = "points"
    params: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    # --- convenience -------------------------------------------------------------
    def __getitem__(self, team: str) -> float:
        return self.ratings[team]

    def get(self, team: str, default: float = 0.0) -> float:
        return self.ratings.get(team, default)

    def ranked(self) -> list[tuple[str, float]]:
        """Teams best-first. Pooled pseudo-teams (see ``FCS_POOL``) are rated but never ranked."""
        return sorted(((t, v) for t, v in self.ratings.items() if not is_pool_team(t)), key=lambda kv: (-kv[1], kv[0]))

    def rank_map(self) -> dict[str, int]:
        return {team: i + 1 for i, (team, _) in enumerate(self.ranked())}

    def zscores(self) -> Ratings:
        values = np.array(list(self.ratings.values()), dtype=float)
        if values.size == 0:
            return {}
        sd = float(values.std())
        sd = sd if sd > 1e-12 else 1.0
        mean = float(values.mean())
        return {t: (v - mean) / sd for t, v in self.ratings.items()}

    def diff(self, home: str, away: str) -> float:
        return self.get(home) - self.get(away)

    def to_frame(self) -> pd.DataFrame:
        rows = [{"rank": i + 1, "team": t, "rating": v} for i, (t, v) in enumerate(self.ranked())]
        return pd.DataFrame(rows, columns=["rank", "team", "rating"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "scale": self.scale,
            "params": self.params,
            "ratings": self.ratings,
        }


@runtime_checkable
class RatingModel(Protocol):
    name: str

    def fit(self, teams: Sequence[str], games: Sequence[Game], prior: Prior | None = None) -> RatingResult: ...


def index_teams(teams: Sequence[str]) -> dict[str, int]:
    return {t: i for i, t in enumerate(teams)}


def prior_vector(teams: Sequence[str], prior: Prior | None, default: float) -> np.ndarray:
    """Prior ratings aligned with ``teams``; unknown teams get ``default``."""
    if not prior:
        return np.full(len(teams), default, dtype=float)
    return np.array([float(prior.get(t, default)) for t in teams], dtype=float)


def usable_games(games: Sequence[Game], team_index: dict[str, int]) -> list[Game]:
    """Completed games where both participants are rated teams."""
    return [g for g in games if g.has_score and g.home in team_index and g.away in team_index]
