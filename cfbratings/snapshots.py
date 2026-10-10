"""Weekly rating snapshots.

A snapshot for week ``w`` is the rating produced by fitting a model on every completed
game with ``week <= w``. Snapshots power the "rank at the time" component of PPoints and
the rating-trajectory charts. They are cached on disk keyed by season, method and a hash
of the model parameters, so changing a hyper-parameter never serves stale numbers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .config import Settings
from .config import settings as default_settings
from .data.cache import JsonCache
from .data.schema import Game, games_through_week, is_pool_team, max_completed_week
from .models import make_model
from .models.base import Ratings


def params_fingerprint(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:10]


def weekly_snapshots(
    method: str,
    teams: Sequence[str],
    games: Sequence[Game],
    *,
    season: int | None = None,
    settings: Settings | None = None,
    cache: JsonCache | None = None,
    use_cache: bool = True,
    prior: Any = None,
    **overrides: Any,
) -> dict[int, Ratings]:
    """Return ``{week: ratings}`` for every week of ``season`` with a completed game.

    ``games`` may span several seasons; only ``season`` (default: the latest) is
    snapshotted, warm-started with ``prior`` when one is given, in which case week 0 holds
    the preseason ratings implied by the prior alone.
    """
    s = settings or default_settings
    model = make_model(method, s, **overrides)
    season = season or max((g.season for g in games), default=0)
    games = [g for g in games if g.season == season]
    last_week = max_completed_week(games)
    if last_week == 0:
        return {}
    fingerprint = params_fingerprint(
        {"method": method, "teams": sorted(teams), "params": getattr(model, "__dict__", {}), "prior": prior}
    )
    key = f"snapshots_{season}_{method}_{fingerprint}"
    cache = cache or JsonCache(s.cache_dir, ttl_hours=None)

    cached: dict[int, Ratings] = {}
    if use_cache:
        raw = cache.get(key, ignore_ttl=True) or {}
        cached = {int(k): v for k, v in raw.items()}
    # A cached week is only reusable if no games from that week or earlier have changed;
    # we approximate that by storing the number of games used for each snapshot.
    out: dict[int, Ratings] = {}
    dirty = False
    if prior:
        # Week 0: the preseason ratings implied by the prior alone (no games yet).
        out[0] = model.fit(teams, [], prior=prior).ratings
    for week in range(1, last_week + 1):
        subset = games_through_week(games, week)
        if not subset:
            continue
        entry = cached.get(week)
        if entry and entry.get("_n_games") == len(subset):
            out[week] = {k: v for k, v in entry.items() if not k.startswith("_")}
            continue
        ratings = model.fit(teams, subset, prior=prior).ratings
        out[week] = ratings
        cached[week] = {**ratings, "_n_games": len(subset)}
        dirty = True
    if use_cache and dirty:
        cache.put(key, {str(k): v for k, v in cached.items()}, final=False)
    return out


def rank_map(ratings: Mapping[str, float]) -> dict[str, int]:
    """Rank by team (1 = best) for one snapshot, ties broken by name; pooled pseudo-teams are never ranked."""
    ordered = sorted(((t, v) for t, v in ratings.items() if not is_pool_team(t)), key=lambda kv: (-kv[1], kv[0]))
    return {t: i + 1 for i, (t, _) in enumerate(ordered)}


def trajectory(snapshots: dict[int, Ratings], team: str) -> list[tuple[int, int, float]]:
    """``(week, rank, rating)`` for one team across snapshots."""
    out = []
    for week in sorted(snapshots):
        ratings = snapshots[week]
        if team not in ratings:
            continue
        out.append((week, rank_map(ratings)[team], ratings[team]))
    return out


RANK_UNCHANGED = "–"  # noqa: RUF001  (an en dash: the dashboard's "no movement" glyph)
RANK_NEW = "new"


def previous_snapshot_week(snapshots: Mapping[int, Ratings], week: int | None = None) -> int | None:
    """The latest snapshot week strictly before ``week`` (default: before the latest one), or ``None``."""
    if not snapshots:
        return None
    if week is None:
        week = max(snapshots)
    earlier = [w for w in snapshots if w < week]
    return max(earlier) if earlier else None


def rank_changes(
    current: Mapping[str, int], snapshots: Mapping[int, Ratings], *, week: int | None = None
) -> dict[str, str]:
    """Rank movement of every team in ``current`` since the snapshot before ``week``.

    ``current`` is rank by team (``RatingResult.rank_map()``); ``week`` is the week those ranks
    belong to (default: the latest snapshot), so the comparison is against the snapshot just
    before it -- week 0, the preseason prior, counts. Labels: ``"▲ 3"`` climbed three places,
    ``"▼ 2"`` dropped two, an en dash (:data:`RANK_UNCHANGED`) unchanged, and ``"new"`` for a
    team the earlier snapshot did not rank. With no earlier snapshot (preseason, or the first
    week of a cold start) every team gets the en dash, so the column never raises.
    """
    before = previous_snapshot_week(snapshots, week)
    if before is None:
        return dict.fromkeys(current, RANK_UNCHANGED)
    previous = rank_map(snapshots[before])
    out: dict[str, str] = {}
    for team, rank in current.items():
        then = previous.get(team)
        if then is None:
            out[team] = RANK_NEW
        elif then > rank:
            out[team] = f"▲ {then - rank}"
        elif then < rank:
            out[team] = f"▼ {rank - then}"
        else:
            out[team] = RANK_UNCHANGED
    return out
