"""Typed representations of teams and games.

The CollegeFootballData (CFBD) API has shipped both ``camelCase`` and ``snake_case``
payloads over the years. Everything downstream works on the frozen dataclasses below so
that the rest of the package never has to care which flavour it was given.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

REGULAR = "regular"
POSTSEASON = "postseason"
FCS_POOL = "FCS (pooled)"

_HEX6 = re.compile(r"[0-9a-f]{6}")
_HEX3 = re.compile(r"[0-9a-f]{3}")


def is_pool_team(name: str | None) -> bool:
    return name == FCS_POOL


def _pick(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_hex_color(value: Any) -> str | None:
    """Coerce a CSS-style colour to lower-case ``#rrggbb``; ``None`` for anything unusable.

    CFBD ships ``#9e1b32``-style strings, but the leading ``#`` and the case are not worth
    trusting and ``#rgb`` shorthand is expanded. Blank or malformed values become ``None``
    so that an invalid colour never reaches a chart.
    """
    if value is None:
        return None
    s = str(value).strip().lower().removeprefix("#")
    if _HEX6.fullmatch(s):
        return f"#{s}"
    if _HEX3.fullmatch(s):
        return "#" + "".join(ch * 2 for ch in s)
    return None


def contrast_color(color: str | None, *, light: str = "#ffffff", dark: str = "#000000") -> str:
    """``light`` or ``dark``, whichever reads better on top of ``color`` (WCAG relative luminance).

    Unknown or invalid colours get ``dark``, the safe choice on the default light background.
    """
    hex_color = normalize_hex_color(color)
    if hex_color is None:
        return dark
    linear = []
    for i in (1, 3, 5):
        c = int(hex_color[i : i + 2], 16) / 255.0
        linear.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return light if luminance < 0.179 else dark  # the point where white and black contrast equally


@dataclass(frozen=True, slots=True)
class Team:
    school: str
    conference: str | None = None
    classification: str | None = None  # "fbs" | "fcs" | ...
    abbreviation: str | None = None
    color: str | None = None  # primary brand colour, lower-case "#rrggbb"
    alt_color: str | None = None  # secondary brand colour (CFBD ``alternateColor``)
    logo: str | None = None  # 500 px logo for light backgrounds (first entry of CFBD ``logos``)
    logo_dark: str | None = None  # the ``logos-dark`` variant for dark backgrounds, when there is one

    @classmethod
    def from_cfbd(cls, d: Mapping[str, Any]) -> Team:
        logos = [str(u) for u in (_pick(d, "logos", default=None) or []) if u]
        return cls(
            school=str(_pick(d, "school", "team", "name")),
            conference=_pick(d, "conference"),
            classification=_pick(d, "classification", "division"),
            abbreviation=_pick(d, "abbreviation"),
            color=normalize_hex_color(_pick(d, "color")),
            alt_color=normalize_hex_color(_pick(d, "alt_color", "alternateColor", "alternate_color")),
            logo=_pick(d, "logo", default=logos[0] if logos else None),
            logo_dark=_pick(d, "logo_dark", default=next((u for u in logos if "logos-dark" in u), None)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Team:
        """Inverse of :meth:`to_dict` (also accepts raw CFBD ``/teams`` rows)."""
        return cls.from_cfbd(d)


@dataclass(frozen=True, slots=True)
class Record:
    wins: int = 0
    losses: int = 0
    ties: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_pct(self) -> float:
        return (self.wins + 0.5 * self.ties) / self.games if self.games else 0.0

    def __str__(self) -> str:
        return f"{self.wins}-{self.losses}" + (f"-{self.ties}" if self.ties else "")


@dataclass(frozen=True, slots=True)
class Game:
    """One scheduled or completed game.

    ``week`` is the *effective* week used for ordering and snapshots: regular-season
    weeks keep their number and postseason weeks are relabelled to follow the final
    regular-season week (see :func:`normalize_games`). ``raw_week`` keeps the original.
    """

    id: int
    season: int
    week: int
    home: str
    away: str
    home_points: int | None = None
    away_points: int | None = None
    neutral_site: bool = False
    completed: bool = False
    season_type: str = REGULAR
    raw_week: int | None = None
    home_conference: str | None = None
    away_conference: str | None = None
    start_date: str | None = None
    home_name: str | None = None  # original school when ``home`` is a pooled pseudo-team
    away_name: str | None = None

    # --- derived -----------------------------------------------------------------
    @property
    def home_display(self) -> str:
        return self.home_name or self.home

    @property
    def away_display(self) -> str:
        return self.away_name or self.away

    def display_name(self, team: str) -> str:
        return self.home_display if team == self.home else self.away_display

    @property
    def has_score(self) -> bool:
        return self.completed and self.home_points is not None and self.away_points is not None

    @property
    def margin(self) -> int:
        """Home margin of victory (negative when the away team won)."""
        if not self.has_score:
            raise ValueError(f"game {self.id} has no final score")
        return int(self.home_points) - int(self.away_points)  # type: ignore[arg-type]

    @property
    def home_result(self) -> float:
        """1.0 home win, 0.0 away win, 0.5 tie."""
        m = self.margin
        return 1.0 if m > 0 else 0.0 if m < 0 else 0.5

    @property
    def winner(self) -> str | None:
        m = self.margin
        return self.home if m > 0 else self.away if m < 0 else None

    @property
    def loser(self) -> str | None:
        m = self.margin
        return self.away if m > 0 else self.home if m < 0 else None

    @property
    def is_conference_game(self) -> bool:
        return bool(self.home_conference) and self.home_conference == self.away_conference

    @property
    def order_key(self) -> tuple[int, int, str, int]:
        return (self.season, self.week, self.start_date or "", self.id)

    def opponent(self, team: str) -> str:
        if team == self.home:
            return self.away
        if team == self.away:
            return self.home
        raise ValueError(f"{team} did not play in game {self.id}")

    def margin_for(self, team: str) -> int:
        return self.margin if team == self.home else -self.margin

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Game:
        """Inverse of :meth:`to_dict` (also accepts raw CFBD payloads)."""
        if "home" in d and "away" in d:
            return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[arg-type]
        return cls.from_cfbd(d)

    @classmethod
    def from_cfbd(cls, d: Mapping[str, Any]) -> Game:
        home_points = _pick(d, "homePoints", "home_points")
        away_points = _pick(d, "awayPoints", "away_points")
        week = int(_pick(d, "week", default=0))
        return cls(
            id=int(_pick(d, "id", default=0)),
            season=int(_pick(d, "season", "year", default=0)),
            week=week,
            raw_week=week,
            season_type=str(_pick(d, "seasonType", "season_type", default=REGULAR)),
            home=str(_pick(d, "homeTeam", "home_team")),
            away=str(_pick(d, "awayTeam", "away_team")),
            home_points=None if home_points is None else int(home_points),
            away_points=None if away_points is None else int(away_points),
            neutral_site=bool(_pick(d, "neutralSite", "neutral_site", default=False)),
            completed=bool(_pick(d, "completed", default=False)),
            home_conference=_pick(d, "homeConference", "home_conference"),
            away_conference=_pick(d, "awayConference", "away_conference"),
            start_date=_pick(d, "startDate", "start_date"),
        )


# --- collections -------------------------------------------------------------------


def normalize_teams(raw: Iterable[Mapping[str, Any]]) -> list[Team]:
    return [Team.from_cfbd(t) for t in raw]


def normalize_games(raw: Iterable[Mapping[str, Any] | Game]) -> list[Game]:
    """Convert raw CFBD rows to :class:`Game` objects, sorted chronologically.

    Postseason weeks are renumbered to continue after the last regular-season week of
    the same season so that ``week`` is a monotone timeline within a season.
    """
    games = [g if isinstance(g, Game) else Game.from_cfbd(g) for g in raw]
    last_regular: dict[int, int] = {}
    for g in games:
        if g.season_type != POSTSEASON:
            last_regular[g.season] = max(last_regular.get(g.season, 0), g.week)
    out: list[Game] = []
    for g in games:
        if g.season_type == POSTSEASON:
            base = last_regular.get(g.season, 0)
            raw = g.raw_week if g.raw_week is not None else g.week
            g = Game(**{**g.to_dict(), "week": base + max(int(raw), 1)})
        out.append(g)
    out.sort(key=lambda g: g.order_key)
    return out


def completed_games(games: Iterable[Game]) -> list[Game]:
    return [g for g in games if g.has_score]


def games_through_week(games: Iterable[Game], week: int, *, inclusive: bool = True) -> list[Game]:
    """Completed games played up to ``week`` (inclusive by default)."""
    if inclusive:
        return [g for g in games if g.has_score and g.week <= week]
    return [g for g in games if g.has_score and g.week < week]


def max_completed_week(games: Iterable[Game]) -> int:
    return max((g.week for g in games if g.has_score), default=0)


def pool_non_fbs(games: Iterable[Game], fbs: Iterable[str], pool_name: str = FCS_POOL) -> list[Game]:
    """Rewrite games against non-FBS opponents so the opponent is one shared pseudo-team.

    Games where neither side is FBS are dropped. The original school name is kept in
    ``home_name`` / ``away_name`` for display.
    """
    fbs_set = set(fbs)
    out: list[Game] = []
    for g in games:
        home_fbs, away_fbs = g.home in fbs_set, g.away in fbs_set
        if home_fbs and away_fbs:
            out.append(g)
        elif home_fbs or away_fbs:
            d = g.to_dict()
            if not home_fbs:
                d.update(home=pool_name, home_name=g.home, home_conference="FCS")
            if not away_fbs:
                d.update(away=pool_name, away_name=g.away, away_conference="FCS")
            out.append(Game(**d))
    return out


def team_names(teams: Sequence[Team] | Sequence[str]) -> list[str]:
    return [t.school if isinstance(t, Team) else str(t) for t in teams]


def conference_map(teams: Sequence[Team]) -> dict[str, str | None]:
    return {t.school: t.conference for t in teams}
