"""Elo ratings with a margin-of-victory multiplier (FiveThirtyEight style).

For each game, in chronological order::

    E_home = 1 / (1 + 10^((R_away - R_home - HFA_elo) / 400))
    mult   = ln(|margin| + 1) * 2.2 / (0.001 * elo_diff_winner + 2.2)
    R_home += K * mult * (S_home - E_home);  R_away -= the same

``elo_diff_winner`` is the winner's pre-game Elo (including home advantage) minus the
loser's, which damps blow-outs by heavy favourites. Home advantage is expressed in points
and converted with ``elo_per_point`` (25 Elo ≈ 1 point). Ratings are regressed toward the
mean only between seasons, never week to week.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..data.schema import Game
from .base import RatingResult, Ratings


def elo_expected(r_home: float, r_away: float, hfa_elo: float = 0.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_away - r_home - hfa_elo) / 400.0))


def mov_multiplier(margin: int, elo_diff_winner: float) -> float:
    if margin == 0:
        return 0.0
    return math.log(abs(margin) + 1.0) * 2.2 / (0.001 * elo_diff_winner + 2.2)


@dataclass
class EloModel:
    k: float = 25.0
    init: float = 1500.0
    hfa: float = 2.5  # points
    elo_per_point: float = 25.0
    season_regress: float = 0.33
    mov: bool = True
    name: str = "elo"

    @property
    def hfa_elo(self) -> float:
        return self.hfa * self.elo_per_point

    # --- incremental API ---------------------------------------------------------
    def update(self, ratings: dict[str, float], game: Game) -> float:
        """Apply one completed game in place; returns the home delta."""
        hfa = 0.0 if game.neutral_site else self.hfa_elo
        rh, ra = ratings[game.home], ratings[game.away]
        exp_home = elo_expected(rh, ra, hfa)
        s_home = game.home_result
        if self.mov:
            margin = game.margin
            diff_winner = (rh + hfa - ra) if margin > 0 else (ra - rh - hfa)
            mult = mov_multiplier(margin, diff_winner)
        else:
            mult = 1.0
        delta = self.k * mult * (s_home - exp_home)
        ratings[game.home] = rh + delta
        ratings[game.away] = ra - delta
        return delta

    def regress(self, ratings: dict[str, float]) -> None:
        for t in ratings:
            ratings[t] = ratings[t] + (self.init - ratings[t]) * self.season_regress

    def win_prob(self, r_home: float, r_away: float, neutral: bool = False) -> float:
        return elo_expected(r_home, r_away, 0.0 if neutral else self.hfa_elo)

    # --- batch API ---------------------------------------------------------------
    def fit(
        self,
        teams: Sequence[str],
        games: Sequence[Game],
        initial: Mapping[str, float] | None = None,
    ) -> RatingResult:
        ratings: Ratings = {t: float(initial.get(t, self.init)) if initial else self.init for t in teams}
        current_season: int | None = None
        for g in sorted((g for g in games if g.has_score), key=lambda g: g.order_key):
            if g.home not in ratings or g.away not in ratings:
                continue
            if current_season is None:
                current_season = g.season
            elif g.season != current_season:
                self.regress(ratings)
                current_season = g.season
            self.update(ratings, g)
        return RatingResult(
            method=self.name,
            ratings=ratings,
            scale="elo",
            params={
                "k": self.k,
                "init": self.init,
                "hfa": self.hfa,
                "elo_per_point": self.elo_per_point,
                "season_regress": self.season_regress,
                "mov": self.mov,
            },
            extras={"hfa_elo": self.hfa_elo, "elo_per_point": self.elo_per_point},
        )
