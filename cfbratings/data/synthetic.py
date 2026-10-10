"""Synthetic seasons with known ground truth.

Used by the test-suite, the ``--demo`` CLI mode and the dashboard's demo mode so that the
whole pipeline runs without an API key. Each team gets a latent strength (in points);
game margins are ``strength difference + home-field advantage + Gaussian noise``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schema import Game, Team, normalize_games

_CONFERENCES = [
    "Atlantic Coast",
    "Big 12",
    "Big Ten",
    "SEC",
    "Mountain West",
    "American Athletic",
    "Sun Belt",
    "Mid-American",
]
# Distinct prefixes so that two conferences never generate the same school name.
_PREFIXES = ["Atlantic", "Prairie", "Lakes", "Delta", "Mountain", "Harbor", "Gulf", "Heartland"]


@dataclass
class SyntheticSeason:
    season: int
    teams: list[Team]
    games: list[Game]
    true_strength: dict[str, float]
    hfa: float
    noise_sd: float
    conference_map: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.conference_map = {t.school: t.conference or "Unknown" for t in self.teams}

    @property
    def team_names(self) -> list[str]:
        return [t.school for t in self.teams]


def synthetic_season(
    *,
    season: int = 2025,
    n_teams: int = 64,
    n_conferences: int = 8,
    weeks: int = 13,
    hfa: float = 2.5,
    noise_sd: float = 14.0,
    strength_sd: float = 9.0,
    conference_strength_sd: float = 4.0,
    neutral_rate: float = 0.05,
    bye_rate: float = 0.08,
    postseason_games: int = 8,
    seed: int = 7,
    carry_from: SyntheticSeason | None = None,
    carry_rho: float = 0.7,
    played_through: int | None = None,
) -> SyntheticSeason:
    """Generate a plausible FBS-like season.

    * Teams are split evenly across conferences; conferences have their own strength offset.
    * Weeks 1-3 are mostly non-conference, later weeks mostly in-conference.
    * ``postseason_games`` bowl games are added on neutral sites after the last week.
    * With ``carry_from`` the same teams return and each strength is ``carry_rho`` times
      last season's plus fresh noise, mimicking year-to-year persistence.
    * ``played_through`` leaves later weeks scheduled but unplayed (no bowls are scheduled
      while the regular season is still in progress), which is what a mid-season
      snapshot looks like.
    """
    rng = np.random.default_rng(seed)
    n_conferences = max(1, min(n_conferences, len(_CONFERENCES)))
    conf_names = _CONFERENCES[:n_conferences]
    conf_offset = rng.normal(0.0, conference_strength_sd, size=n_conferences)

    teams: list[Team] = []
    strength: dict[str, float] = {}
    if carry_from is not None:
        teams = list(carry_from.teams)
        innovation_sd = strength_sd * (1.0 - carry_rho**2) ** 0.5
        for t in teams:
            strength[t.school] = float(carry_rho * carry_from.true_strength[t.school] + rng.normal(0.0, innovation_sd))
    else:
        for i in range(n_teams):
            c = i % n_conferences
            name = f"{_PREFIXES[c]} {_team_label(i // n_conferences)}"
            if name in strength:
                raise ValueError(f"duplicate synthetic team name {name!r}")
            teams.append(Team(school=name, conference=conf_names[c], classification="fbs"))
            strength[name] = float(conf_offset[c] + rng.normal(0.0, strength_sd))

    by_conf: dict[str, list[str]] = {}
    for t in teams:
        by_conf.setdefault(t.conference or "", []).append(t.school)
    names = [t.school for t in teams]
    conf_of = {t.school: t.conference or "" for t in teams}

    games: list[dict] = []
    gid = season * 100000
    played: set[tuple[str, str]] = set()

    def add_game(week: int, home: str, away: str, neutral: bool, season_type: str = "regular") -> None:
        nonlocal gid
        gid += 1
        expected = strength[home] - strength[away] + (0.0 if neutral else hfa)
        margin = int(round(expected + rng.normal(0.0, noise_sd)))
        if margin == 0:
            margin = int(rng.choice([-1, 1]))
        base = int(rng.integers(10, 35))
        played = played_through is None or week <= played_through
        home_pts = base + max(margin, 0) if played else None
        away_pts = base + max(-margin, 0) if played else None
        games.append(
            {
                "id": gid,
                "season": season,
                "week": week,
                "seasonType": season_type,
                "homeTeam": home,
                "awayTeam": away,
                "homePoints": home_pts,
                "awayPoints": away_pts,
                "neutralSite": neutral,
                "completed": played,
                "homeConference": conf_of[home],
                "awayConference": conf_of[away],
                "startDate": f"{season}-{9 + (week - 1) // 4:02d}-{1 + 7 * ((week - 1) % 4):02d}",
            }
        )

    for week in range(1, weeks + 1):
        pool = [t for t in names if rng.random() > bye_rate]
        rng.shuffle(pool)
        in_conf_rate = 0.25 if week <= 3 else 0.85
        used: set[str] = set()
        for home in pool:
            if home in used:
                continue
            same = [t for t in by_conf[conf_of[home]] if t not in used and t != home and t in pool]
            other = [t for t in pool if t not in used and t != home and conf_of[t] != conf_of[home]]
            prefer_conf = rng.random() < in_conf_rate
            primary = same if (prefer_conf and same) else (other or same)
            fresh = [t for t in primary if (home, t) not in played and (t, home) not in played]
            candidates = fresh or primary  # a rematch beats an unplanned bye
            if not candidates:
                continue
            away = str(rng.choice(candidates))
            used.update({home, away})
            played.add((home, away))
            add_game(week, home, away, neutral=rng.random() < neutral_rate)

    if postseason_games and (played_through is None or played_through >= weeks):
        ordered = sorted(names, key=lambda t: strength[t], reverse=True)
        top = ordered[: 2 * postseason_games]
        rng.shuffle(top)
        for i in range(0, len(top) - 1, 2):
            add_game(1, top[i], top[i + 1], neutral=True, season_type="postseason")

    return SyntheticSeason(
        season=season,
        teams=teams,
        games=normalize_games(games),
        true_strength=strength,
        hfa=hfa,
        noise_sd=noise_sd,
    )


def synthetic_seasons(
    seasons: Sequence[int], *, seed: int = 7, played_through_last: int | None = None, **kwargs: Any
) -> list[SyntheticSeason]:
    """Consecutive seasons sharing a roster, with strengths that persist year to year.

    ``played_through_last`` leaves the final season partially played (a mid-season
    snapshot) while every earlier season is complete.
    """
    ordered = sorted(seasons)
    out: list[SyntheticSeason] = []
    prev: SyntheticSeason | None = None
    for i, season in enumerate(ordered):
        last = i == len(ordered) - 1
        prev = synthetic_season(
            season=season,
            seed=seed + i,
            carry_from=prev,
            played_through=played_through_last if last else None,
            **kwargs,
        )
        out.append(prev)
    return out


def _team_label(i: int) -> str:
    labels = [
        "State",
        "Tech",
        "A&M",
        "University",
        "Central",
        "Northern",
        "Western",
        "Eastern",
        "Valley",
        "Coastal",
        "Ridge",
        "Pacific",
        "Metro",
        "Polytechnic",
        "Institute",
        "Summit",
    ]
    return labels[i] if i < len(labels) else f"College {i + 1}"
