"""PPoints -- a résumé metric that rewards *who* you beat and *where*.

PPoints is deliberately not a predictive rating; it is a committee-style score that
answers "how impressive is this team's body of work?". For every completed game:

1. The opponent is placed in a tier by rank. The rank is a blend of the opponent's rank
   *entering that week* (from weekly rating snapshots) and its *current* rank, controlled
   by ``weight_current`` -- so beating a team that later collapsed is worth less, and
   beating a team that turned out to be great is worth more.
2. Tier points: top 10 → 8, top 25 → 5, top 40 → 3, top 60 → 2, else 1.
3. Road wins are multiplied by ``away_bonus`` (neutral sites count as road for both).
4. The value is scaled by the opponent's conference strength index (0.6 - 1.6); an
   independent opponent, which has no conference, is scaled by its own rating placed on the
   same scale (see :func:`cfbratings.analytics.independent_strength`).
5. A loss costs ``loss_penalty / opponent tier points``: losing to a top-10 team barely
   hurts, losing to a bottom-feeder hurts a lot. Non-conference games use a softer loss
   penalty and a bonus win factor to encourage ambitious scheduling.

Everything that depends only on the week (rank maps, conference strength) is computed once
per week rather than once per game. :class:`PPointsScorer` exposes the per-game arithmetic,
so a game that has not been played yet can be priced (:meth:`PPointsScorer.credit`): the
season simulation uses that to re-score every résumé on each simulated season.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .analytics import conference_strength, independent_strength
from .data.schema import Game, is_independent, is_pool_team
from .models.base import Ratings


@dataclass(frozen=True)
class PPointsConfig:
    weight_current: float = 0.5
    away_bonus: float = 1.20
    loss_penalty: float = 4.0
    nonconf_loss_penalty: float = 2.0
    nonconf_win_factor: float = 1.5
    tiers: tuple[tuple[int, float], ...] = field(default=((10, 8.0), (25, 5.0), (40, 3.0), (60, 2.0)))
    default_tier_points: float = 1.0

    def tier_points(self, rank: int | None) -> float:
        if rank is None:
            return self.default_tier_points
        for cutoff, pts in self.tiers:
            if rank <= cutoff:
                return pts
        return self.default_tier_points


@dataclass(frozen=True)
class GameCredit:
    """What one game is worth to each side before it is played.

    The winner gains its ``*_win`` value; the loser loses its ``*_loss`` value (both positive
    numbers, so a loss is applied by subtraction). The value of a win depends on the opponent's
    tier, the site and the opponent's conference; the cost of a loss shrinks with the opponent's
    tier, so losing to a top-10 team barely hurts.
    """

    home_win: float
    home_loss: float
    away_win: float
    away_loss: float

    def apply(self, scores: dict[str, float], home: str, away: str, home_won: bool) -> None:
        """Add the result to ``scores`` in place."""
        if home_won:
            scores[home] += self.home_win
            scores[away] -= self.away_loss
        else:
            scores[away] += self.away_win
            scores[home] -= self.home_loss


def _rank_map(ratings: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(((t, v) for t, v in ratings.items() if not is_pool_team(t)), key=lambda kv: (-kv[1], kv[0]))
    return {t: i + 1 for i, (t, _) in enumerate(ordered)}


class PPointsScorer:
    """PPoints game by game, with everything that depends only on the week computed once.

    ``weekly_ratings`` maps week → ratings *after* that week completed (see
    :mod:`cfbratings.snapshots`; week 0 is the preseason prior when one exists). The rank
    entering week ``w`` is taken from snapshot ``w - 1``; when unavailable -- which is always
    the case for a game still to be played -- the current ratings are used instead, never a
    snapshot that already contains the game itself.
    """

    def __init__(
        self,
        teams: Sequence[str],
        current_ratings: Mapping[str, float],
        conference_map: Mapping[str, str | None],
        weekly_ratings: Mapping[int, Ratings] | None = None,
        config: PPointsConfig | None = None,
    ) -> None:
        self.cfg = config or PPointsConfig()
        self.teams = list(teams)
        self._team_set = set(teams)
        self._conference_map = conference_map
        self._weekly = weekly_ratings or {}
        self._current_rank = _rank_map(current_ratings)
        self._current_conf = self._strength(current_ratings)
        self._week_cache: dict[int, tuple[dict[str, int], dict[str, float]]] = {}

    def _strength(self, ratings: Mapping[str, float]) -> dict[str, float]:
        # Independents are keyed by team, on the same scale as the conferences.
        out = conference_strength(ratings, self._conference_map)
        out.update(independent_strength(ratings, self._conference_map))
        return out

    def _week_context(self, week: int | None) -> tuple[dict[str, int], dict[str, float]]:
        if week is None:
            return self._current_rank, self._current_conf
        if week not in self._week_cache:
            snap = self._weekly.get(week - 1)  # entering-week snapshot (week 0 = preseason prior when warm-started)
            if snap:
                self._week_cache[week] = (_rank_map(snap), self._strength(snap))
            else:
                self._week_cache[week] = (self._current_rank, self._current_conf)
        return self._week_cache[week]

    def _opp_value(self, opponent: str, week_rank: Mapping[str, int]) -> float:
        wc = self.cfg.weight_current
        then = self.cfg.tier_points(week_rank.get(opponent))
        now = self.cfg.tier_points(self._current_rank.get(opponent))
        return (1.0 - wc) * then + wc * now

    def _conf_factor(self, conf: str | None, opponent: str, week_conf: Mapping[str, float]) -> float:
        key = opponent if is_independent(conf) else conf
        if is_pool_team(opponent) or key is None:
            return 1.0
        wc = self.cfg.weight_current
        return (1.0 - wc) * week_conf.get(key, 1.0) + wc * self._current_conf.get(key, 1.0)

    def credit_for(
        self,
        home: str,
        away: str,
        *,
        neutral: bool = False,
        week: int | None = None,
        home_conference: str | None = None,
        away_conference: str | None = None,
    ) -> GameCredit:
        """Price a game between ``home`` and ``away`` (``week`` ``None``: by the current ratings)."""
        cfg = self.cfg
        week_rank, week_conf = self._week_context(week)
        home_conf = home_conference or self._conference_map.get(home)
        away_conf = away_conference or self._conference_map.get(away)
        nonconf = bool(home_conf and away_conf and home_conf != away_conf)
        loss_penalty = cfg.nonconf_loss_penalty if nonconf else cfg.loss_penalty
        win_factor = cfg.nonconf_win_factor if nonconf else 1.0

        home_site = cfg.away_bonus if neutral else 1.0
        away_site = cfg.away_bonus

        opp_for_home = self._opp_value(away, week_rank)
        opp_for_away = self._opp_value(home, week_rank)
        value_for_home = opp_for_home * home_site * self._conf_factor(away_conf, away, week_conf)
        value_for_away = opp_for_away * away_site * self._conf_factor(home_conf, home, week_conf)
        return GameCredit(
            home_win=win_factor * value_for_home,
            home_loss=loss_penalty / max(opp_for_home, 1.0),
            away_win=win_factor * value_for_away,
            away_loss=loss_penalty / max(opp_for_away, 1.0),
        )

    def credit(self, game: Game) -> GameCredit:
        """Price a scheduled or completed game."""
        return self.credit_for(
            game.home,
            game.away,
            neutral=game.neutral_site,
            week=game.week,
            home_conference=game.home_conference,
            away_conference=game.away_conference,
        )

    def score(self, games: Sequence[Game]) -> dict[str, float]:
        """PPoints for every team from the completed games (ties count for nobody)."""
        scores = dict.fromkeys(self.teams, 0.0)
        for g in games:
            if not g.has_score or g.home not in self._team_set or g.away not in self._team_set:
                continue
            m = g.margin
            if m != 0:
                self.credit(g).apply(scores, g.home, g.away, m > 0)
        return scores


def ppoints(
    teams: Sequence[str],
    games: Sequence[Game],
    current_ratings: Mapping[str, float],
    conference_map: Mapping[str, str | None],
    weekly_ratings: Mapping[int, Ratings] | None = None,
    config: PPointsConfig | None = None,
) -> dict[str, float]:
    """Compute PPoints for every team (see :class:`PPointsScorer`)."""
    return PPointsScorer(teams, current_ratings, conference_map, weekly_ratings, config).score(games)


__all__ = ["GameCredit", "PPointsConfig", "PPointsScorer", "ppoints"]
