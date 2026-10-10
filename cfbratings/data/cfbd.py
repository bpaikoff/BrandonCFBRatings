"""Client for the CollegeFootballData.com API with transparent JSON caching.

Only the handful of endpoints the rating engine needs are wrapped. Responses are cached
to disk so repeated CLI / dashboard runs do not burn API quota; completed seasons are
marked final, never expire, and are additionally written gzipped to the committed
archive directory so that a fresh clone never re-downloads them. A full season costs
two calls (teams + games), three with betting lines.

Two things a UI should surface:

* ``CFBDClient.api_calls`` -- a class attribute counting the real HTTP requests made by
  every client in this process. The free tier has a small monthly budget, so display it
  next to any refresh control (the CLI prints it after every run).
* ``CFBDClient.freshness(year)`` -- when the season on screen was fetched and whether it
  would be served from the working cache or the committed archive, for a "data as of"
  caption. It never makes a request; ``None`` means the next load would.
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import requests

from ..config import Settings
from ..config import settings as default_settings
from .cache import CacheInfo, JsonCache
from .schema import FCS_POOL, Game, Team, normalize_games, normalize_teams, pool_non_fbs

log = logging.getLogger(__name__)

SEASON_TYPES = ("regular", "postseason", "both")


def _check_season_type(season_type: str) -> None:
    if season_type not in SEASON_TYPES:
        raise ValueError(f"season_type must be one of {SEASON_TYPES}, got {season_type!r}")


class CFBDError(RuntimeError):
    """Raised for configuration or transport problems talking to CFBD."""


class CFBDQuotaError(CFBDError):
    """CFBD answered HTTP 429: the call budget is spent (or calls are too fast). Never retried."""


class CFBDClient:
    def __init__(
        self,
        settings: Settings | None = None,
        cache: JsonCache | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.cache = cache or JsonCache(
            self.settings.cache_dir, self.settings.cache_ttl_hours, archive_dir=self.settings.archive_dir
        )
        self.session = session or requests.Session()
        self.calls: list[str] = []  # every real API request made through this client

    #: Real API requests made by *all* clients in this process (the free tier has a small
    #: monthly budget, so the CLI reports this after every run).
    api_calls: int = 0

    # --- transport ----------------------------------------------------------------
    def _get(self, path: str, params: Mapping[str, Any]) -> Any:
        if not self.settings.api_key:
            raise CFBDError(
                "CFBD_API_KEY is not set. Create a free key at https://collegefootballdata.com/key "
                "and put it in a .env file, or run with --demo to use synthetic data."
            )
        url = f"{self.settings.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.settings.api_key}", "Accept": "application/json"}
        CFBDClient.api_calls += 1
        self.calls.append(f"{path} {dict(params)}")
        log.info("CFBD API call #%d: GET %s %s", CFBDClient.api_calls, path, dict(params))
        resp = self.session.get(url, params=dict(params), headers=headers, timeout=self.settings.timeout)
        if resp.status_code == 401:
            raise CFBDError("CFBD rejected the API key (HTTP 401).")
        if resp.status_code == 429:
            raise CFBDQuotaError(
                "CFBD returned HTTP 429 (rate limit or monthly quota exhausted). Not retrying; "
                "cached and archived seasons remain available offline."
            )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _is_final(year: int) -> bool:
        """Seasons that ended before last February are frozen in the cache."""
        today = _dt.date.today()
        return year < today.year - 1 or (year == today.year - 1 and today.month >= 3)

    # --- endpoints ----------------------------------------------------------------
    def teams(self, year: int, *, force_refresh: bool = False) -> list[Team]:
        key = f"teams_{year}"
        raw = None if force_refresh else self.cache.get(key, ignore_ttl=True)
        if raw is None:
            raw = self._get("/teams/fbs", {"year": year})
            self.cache.put(key, raw, final=self._is_final(year))
        return normalize_teams(raw)

    def games(self, year: int, season_type: str = "both", *, force_refresh: bool = False) -> list[Game]:
        _check_season_type(season_type)
        key = f"games_{year}_{season_type}"
        raw = None if force_refresh else self.cache.get(key)
        if raw is None:
            raw = self._get("/games", {"year": year, "seasonType": season_type})
            self.cache.put(key, raw, final=self._is_final(year))
        return normalize_games(raw)

    def lines(self, year: int, season_type: str = "both", *, force_refresh: bool = False) -> dict[int, float]:
        """Consensus closing spread per game id (negative = home favoured, CFBD convention).

        The result is keyed by game id and expressed as the *home* spread so that
        ``-spread`` is the market's expected home margin. Missing lines are omitted.
        """
        key = f"lines_{year}_{season_type}"
        raw = None if force_refresh else self.cache.get(key)
        if raw is None:
            raw = self._get("/lines", {"year": year, "seasonType": season_type})
            self.cache.put(key, raw, final=self._is_final(year))
        return consensus_spreads(raw)

    # --- provenance ---------------------------------------------------------------
    def freshness(self, year: int, season_type: str = "both") -> CacheInfo | None:
        """Provenance of the cached ``/games`` payload for a season, without touching the API.

        ``tier`` says whether :meth:`games` would read the working cache or fall back to the
        committed archive; ``None`` means it would have to make a real request.
        """
        _check_season_type(season_type)
        return self.cache.info(f"games_{year}_{season_type}")


def spread_sign_consistent(spread: float, formatted: str | None, home: str | None, away: str | None) -> bool:
    """Cross-check a numeric ``spread`` against CFBD's ``formattedSpread`` (``"Ohio State -14.5"``).

    The API documents ``spread`` as *home-relative, negative favours home*, so the team named
    in the formatted string must be the home team when ``spread < 0`` and the away team when
    ``spread > 0``. Returns ``True`` when they agree or when the string cannot be parsed
    (pick'em, unknown team), so the check never discards a usable line.
    """
    if not formatted or not home or not away or spread == 0:
        return True
    name, _, number = str(formatted).strip().rpartition(" ")
    try:
        float(number)
    except ValueError:
        return True
    if name == home:
        return spread < 0
    if name == away:
        return spread > 0
    return True


def consensus_spreads(raw: list[Mapping[str, Any]]) -> dict[int, float]:
    """Home-relative closing spread per game id (CFBD convention: negative = home favoured).

    Prefers the ``consensus`` provider when present and the mean of all providers otherwise.
    Rows whose ``formattedSpread`` names the other side as favourite are counted and reported
    with a warning, as a guard against the sign convention silently changing.
    """
    out: dict[int, float] = {}
    mismatches = 0
    for row in raw:
        gid = row.get("id")
        home = row.get("homeTeam", row.get("home_team"))
        away = row.get("awayTeam", row.get("away_team"))
        lines = row.get("lines") or []
        spreads: list[float] = []
        preferred: float | None = None
        for line in lines:
            spread = line.get("spread")
            if spread is None:
                continue
            try:
                value = float(spread)
            except (TypeError, ValueError):
                continue
            if not spread_sign_consistent(value, line.get("formattedSpread"), home, away):
                mismatches += 1
            spreads.append(value)
            if preferred is None and str(line.get("provider", "")).lower() == "consensus":
                preferred = value
        if gid is None or not spreads:
            continue
        out[int(gid)] = preferred if preferred is not None else sum(spreads) / len(spreads)
    if mismatches:
        log.warning(
            "%d betting line(s) name a favourite that contradicts the home-relative sign of `spread`; "
            "check consensus_spreads() against the current CFBD payload",
            mismatches,
        )
    return out


def load_season(
    year: int,
    season_type: str = "both",
    *,
    force_refresh: bool = False,
    client: CFBDClient | None = None,
    fcs: str | None = None,
) -> tuple[list[Team], list[Game]]:
    """Fetch (teams, games) for a season.

    ``fcs`` controls games against non-FBS opponents: ``"pool"`` (default from settings)
    maps every such opponent onto one pseudo-team so the result still counts, ``"drop"``
    discards those games. When pooling, the pseudo-team is appended to ``teams``.
    """
    client = client or CFBDClient()
    mode = fcs or client.settings.fcs_handling
    teams = client.teams(year, force_refresh=force_refresh)
    games = client.games(year, season_type, force_refresh=force_refresh)
    names = {t.school for t in teams}
    if mode == "pool":
        games = pool_non_fbs(games, names)
        if any(FCS_POOL in (g.home, g.away) for g in games):
            teams = [*teams, Team(school=FCS_POOL, conference="FCS", classification="fcs")]
    elif mode == "drop":
        games = [g for g in games if g.home in names and g.away in names]
    else:
        raise ValueError(f"fcs must be 'pool' or 'drop', got {mode!r}")
    return teams, games


def load_seasons(
    years: Sequence[int],
    season_type: str = "both",
    *,
    force_refresh: bool = False,
    client: CFBDClient | None = None,
    fcs: str | None = None,
) -> tuple[list[Team], list[Game]]:
    """Several seasons at once; ``teams`` is the latest roster plus any earlier-only team."""
    client = client or CFBDClient()
    all_games: list[Game] = []
    latest: list[Team] = []
    seen: dict[str, Team] = {}
    for year in sorted(years):
        teams, games = load_season(year, season_type, force_refresh=force_refresh, client=client, fcs=fcs)
        all_games.extend(games)
        latest = teams
        for t in teams:
            seen.setdefault(t.school, t)
    latest_names = {t.school for t in latest}
    extra = [t for name, t in seen.items() if name not in latest_names]
    return [*latest, *extra], sorted(all_games, key=lambda g: g.order_key)
