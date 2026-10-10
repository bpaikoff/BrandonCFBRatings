"""Weekly rating snapshots.

A snapshot for week ``w`` is the rating produced by fitting a model on every completed
game with ``week <= w``. Snapshots power the "rank at the time" component of PPoints and
the rating-trajectory charts. They are cached on disk keyed by season, method and a hash
of the model parameters, so changing a hyper-parameter never serves stale numbers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
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


def trajectory(snapshots: dict[int, Ratings], team: str) -> list[tuple[int, int, float]]:
    """``(week, rank, rating)`` for one team across snapshots."""
    out = []
    for week in sorted(snapshots):
        ratings = snapshots[week]
        if team not in ratings:
            continue
        ordered = sorted(((t, v) for t, v in ratings.items() if not is_pool_team(t)), key=lambda kv: (-kv[1], kv[0]))
        rank = next(i + 1 for i, (t, _) in enumerate(ordered) if t == team)
        out.append((week, rank, ratings[team]))
    return out
