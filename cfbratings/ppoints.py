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
per week rather than once per game.
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


def _rank_map(ratings: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(((t, v) for t, v in ratings.items() if not is_pool_team(t)), key=lambda kv: (-kv[1], kv[0]))
    return {t: i + 1 for i, (t, _) in enumerate(ordered)}


def ppoints(
    teams: Sequence[str],
    games: Sequence[Game],
    current_ratings: Mapping[str, float],
    conference_map: Mapping[str, str | None],
    weekly_ratings: Mapping[int, Ratings] | None = None,
    config: PPointsConfig | None = None,
) -> dict[str, float]:
    """Compute PPoints for every team.

    ``weekly_ratings`` maps week → ratings *after* that week completed (see
    :mod:`cfbratings.snapshots`; week 0 is the preseason prior when one exists). The rank
    entering week ``w`` is taken from snapshot ``w - 1``; when unavailable the current
    ratings are used instead -- never a snapshot that already contains the game itself.
    """
    cfg = config or PPointsConfig()
    weekly = weekly_ratings or {}
    scores = dict.fromkeys(teams, 0.0)
    team_set = set(teams)

    current_rank = _rank_map(current_ratings)
    current_conf = conference_strength(current_ratings, conference_map)
    # Independents are keyed by team, on the same scale as the conferences (``_strength``).
    current_conf.update(independent_strength(current_ratings, conference_map))

    week_cache: dict[int, tuple[dict[str, int], dict[str, float]]] = {}

    def _strength(ratings: Mapping[str, float]) -> dict[str, float]:
        out = conference_strength(ratings, conference_map)
        out.update(independent_strength(ratings, conference_map))
        return out

    def week_context(week: int) -> tuple[dict[str, int], dict[str, float]]:
        if week not in week_cache:
            snap = weekly.get(week - 1)  # entering-week snapshot (week 0 = preseason prior when warm-started)
            if snap:
                week_cache[week] = (_rank_map(snap), _strength(snap))
            else:
                week_cache[week] = (current_rank, current_conf)
        return week_cache[week]

    wc = cfg.weight_current

    def opp_value(opponent: str, week_rank: Mapping[str, int]) -> float:
        then = cfg.tier_points(week_rank.get(opponent))
        now = cfg.tier_points(current_rank.get(opponent))
        return (1.0 - wc) * then + wc * now

    def conf_factor(conf: str | None, opponent: str, week_conf: Mapping[str, float]) -> float:
        key = opponent if is_independent(conf) else conf
        if is_pool_team(opponent) or key is None:
            return 1.0
        return (1.0 - wc) * week_conf.get(key, 1.0) + wc * current_conf.get(key, 1.0)

    for g in games:
        if not g.has_score or g.home not in team_set or g.away not in team_set:
            continue
        week_rank, week_conf = week_context(g.week)

        home_conf = g.home_conference or conference_map.get(g.home)
        away_conf = g.away_conference or conference_map.get(g.away)
        nonconf = bool(home_conf and away_conf and home_conf != away_conf)
        loss_penalty = cfg.nonconf_loss_penalty if nonconf else cfg.loss_penalty
        win_factor = cfg.nonconf_win_factor if nonconf else 1.0

        home_site = cfg.away_bonus if g.neutral_site else 1.0
        away_site = cfg.away_bonus

        opp_for_home = opp_value(g.away, week_rank)
        opp_for_away = opp_value(g.home, week_rank)
        value_for_home = opp_for_home * home_site * conf_factor(away_conf, g.away, week_conf)
        value_for_away = opp_for_away * away_site * conf_factor(home_conf, g.home, week_conf)

        m = g.margin
        if m > 0:
            scores[g.home] += win_factor * value_for_home
            scores[g.away] -= loss_penalty / max(opp_for_away, 1.0)
        elif m < 0:
            scores[g.away] += win_factor * value_for_away
            scores[g.home] -= loss_penalty / max(opp_for_home, 1.0)
    return scores
