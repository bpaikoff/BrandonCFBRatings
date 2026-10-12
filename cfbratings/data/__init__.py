"""Data layer: typed schema, on-disk cache, CFBD API client and synthetic seasons."""

from __future__ import annotations

from .cache import CacheInfo, JsonCache
from .cfbd import CFBDClient, CFBDError, CFBDQuotaError, load_season, load_seasons
from .schema import (
    FCS_POOL,
    Game,
    Record,
    Team,
    completed_games,
    contrast_color,
    games_through_week,
    is_independent,
    is_pool_team,
    max_completed_week,
    normalize_games,
    normalize_hex_color,
    normalize_teams,
    pool_non_fbs,
    team_names,
)
from .synthetic import SyntheticSeason, synthetic_season, synthetic_seasons

__all__ = [
    "FCS_POOL",
    "CFBDClient",
    "CFBDError",
    "CFBDQuotaError",
    "CacheInfo",
    "Game",
    "JsonCache",
    "Record",
    "SyntheticSeason",
    "Team",
    "completed_games",
    "contrast_color",
    "games_through_week",
    "is_independent",
    "is_pool_team",
    "load_season",
    "load_seasons",
    "max_completed_week",
    "normalize_games",
    "normalize_hex_color",
    "normalize_teams",
    "pool_non_fbs",
    "synthetic_season",
    "synthetic_seasons",
    "team_names",
]
