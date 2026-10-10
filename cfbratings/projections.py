"""Rest-of-season projections.

Every unplayed game on a team's schedule gets a win probability from the calibrated
ratings. The number of remaining wins is then a sum of independent Bernoulli trials -- a
Poisson-binomial distribution -- which is computed *exactly* by dynamic programming, so
"probability of reaching 10 wins" needs no Monte Carlo noise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analytics import records
from .data.schema import Game, Record, is_pool_team
from .models.base import RatingResult
from .predict import MarginCalibrator


def win_distribution(probs: Sequence[float]) -> np.ndarray:
    """``P(k wins)`` for ``k = 0..n`` given independent per-game win probabilities."""
    dist = np.array([1.0])
    for p in probs:
        p = float(min(max(p, 0.0), 1.0))
        new = np.zeros(dist.size + 1)
        new[:-1] += dist * (1.0 - p)
        new[1:] += dist * p
        dist = new
    return dist


@dataclass
class TeamProjection:
    team: str
    record: Record
    remaining: list[tuple[Game, float]] = field(default_factory=list)  # (game, P(team wins))
    distribution: np.ndarray = field(default_factory=lambda: np.array([1.0]))  # over remaining wins

    @property
    def expected_remaining_wins(self) -> float:
        return float(sum(p for _, p in self.remaining))

    @property
    def expected_final_wins(self) -> float:
        return self.record.wins + self.expected_remaining_wins

    @property
    def expected_final_losses(self) -> float:
        return self.record.losses + len(self.remaining) - self.expected_remaining_wins

    def p_at_least(self, total_wins: int) -> float:
        """Probability of finishing with at least ``total_wins`` wins."""
        need = total_wins - self.record.wins
        if need <= 0:
            return 1.0
        if need > len(self.remaining):
            return 0.0
        return float(self.distribution[need:].sum())

    def p_exactly(self, total_wins: int) -> float:
        k = total_wins - self.record.wins
        if k < 0 or k > len(self.remaining):
            return 0.0
        return float(self.distribution[k])

    @property
    def p_win_out(self) -> float:
        return float(self.distribution[-1])

    @property
    def next_game(self) -> tuple[Game, float] | None:
        return self.remaining[0] if self.remaining else None


def _record_for(team: str, games: Sequence[Game]) -> Record:
    wins = losses = ties = 0
    for g in games:
        if not g.has_score or team not in (g.home, g.away):
            continue
        m = g.margin_for(team)
        wins, losses, ties = wins + (m > 0), losses + (m < 0), ties + (m == 0)
    return Record(wins, losses, ties)


def remaining_games(team: str, games: Sequence[Game]) -> list[Game]:
    return sorted((g for g in games if not g.has_score and team in (g.home, g.away)), key=lambda g: g.order_key)


def project_team(
    team: str,
    games: Sequence[Game],
    result: RatingResult,
    calibrator: MarginCalibrator,
    record: Record | None = None,
) -> TeamProjection:
    rec = record if record is not None else _record_for(team, games)
    remaining: list[tuple[Game, float]] = []
    for g in remaining_games(team, games):
        p_home = calibrator.win_prob(result.get(g.home), result.get(g.away), g.neutral_site)
        remaining.append((g, p_home if g.home == team else 1.0 - p_home))
    return TeamProjection(
        team=team, record=rec, remaining=remaining, distribution=win_distribution([p for _, p in remaining])
    )


def projection_table(
    teams: Sequence[str],
    games: Sequence[Game],
    result: RatingResult,
    calibrator: MarginCalibrator,
    *,
    thresholds: Sequence[int] = (6, 8, 10, 12),
    conference_map: Mapping[str, str | None] | None = None,
) -> pd.DataFrame:
    """One row per (non-pooled) team: current record, remaining games and win-total odds."""
    recs = records(teams, games)
    rank = result.rank_map()
    rows = []
    for team in teams:
        if is_pool_team(team):
            continue
        proj = project_team(team, games, result, calibrator, recs[team])
        nxt = proj.next_game
        row = {
            "rank": rank.get(team),
            "team": team,
            "conference": (conference_map or {}).get(team) or "Independent",
            "record": str(proj.record),
            "wins": proj.record.wins,
            "losses": proj.record.losses,
            "remaining": len(proj.remaining),
            "proj_wins": proj.expected_final_wins,
            "proj_losses": proj.expected_final_losses,
            "proj_record": f"{proj.expected_final_wins:.1f}-{proj.expected_final_losses:.1f}",
            "p_win_out": proj.p_win_out,
            "next_opponent": (nxt[0].display_name(nxt[0].opponent(team)) if nxt else None),
            "next_site": (
                ("Neutral" if nxt[0].neutral_site else ("Home" if nxt[0].home == team else "Away")) if nxt else None
            ),
            "next_win_prob": nxt[1] if nxt else None,
        }
        for k in thresholds:
            row[f"p_{k}_wins"] = proj.p_at_least(k)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["proj_wins", "rank"], ascending=[False, True]).reset_index(drop=True)
