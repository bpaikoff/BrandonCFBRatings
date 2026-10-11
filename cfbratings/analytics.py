"""Team-level analytics built on top of a fitted :class:`RatingResult`."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone, tzinfo
from statistics import median

import numpy as np
import pandas as pd

from .data.cache import CacheInfo
from .data.schema import Game, Record, is_pool_team
from .models.base import RatingResult, Ratings
from .predict import MarginCalibrator


def records(teams: Sequence[str], games: Sequence[Game]) -> dict[str, Record]:
    wins = dict.fromkeys(teams, 0)
    losses = dict.fromkeys(teams, 0)
    ties = dict.fromkeys(teams, 0)
    for g in games:
        if not g.has_score or g.home not in wins or g.away not in wins:
            continue
        m = g.margin
        if m > 0:
            wins[g.home] += 1
            losses[g.away] += 1
        elif m < 0:
            wins[g.away] += 1
            losses[g.home] += 1
        else:
            ties[g.home] += 1
            ties[g.away] += 1
    return {t: Record(wins[t], losses[t], ties[t]) for t in teams}


def strength_of_schedule(teams: Sequence[str], games: Sequence[Game], ratings: Mapping[str, float]) -> Ratings:
    """Mean opponent rating (same units as ``ratings``).

    Over completed games once a team has played; until then over its *scheduled* opponents, so
    the preseason table ranks schedules instead of showing zeros or gaps. Only a team with no
    games at all gets ``nan``.
    """
    played: dict[str, list[float]] = {t: [] for t in teams}
    scheduled: dict[str, list[float]] = {t: [] for t in teams}
    for g in games:
        if g.home not in played or g.away not in played:
            continue
        bucket = played if g.has_score else scheduled
        bucket[g.home].append(ratings.get(g.away, 0.0))
        bucket[g.away].append(ratings.get(g.home, 0.0))
    out: Ratings = {}
    for t in teams:
        v = played[t] or scheduled[t]
        out[t] = float(np.mean(v)) if v else math.nan
    return out


def momentum(
    teams: Sequence[str],
    games: Sequence[Game],
    result: RatingResult,
    calibrator: MarginCalibrator,
    *,
    last_n: int = 3,
) -> Ratings:
    """Average points-above-expectation over each team's last ``last_n`` games.

    "Expectation" is the calibrated predicted margin, so momentum is in points for every
    model. Positive values mean the team has recently beaten its own rating; a team with no
    completed game has no evidence either way and gets ``0.0`` (like PPoints), which keeps the
    column numeric and sortable in the preseason.
    """
    resid: dict[str, list[float]] = {t: [] for t in teams}
    for g in sorted((g for g in games if g.has_score), key=lambda g: g.order_key):
        if g.home not in resid or g.away not in resid:
            continue
        expected = calibrator.predict_margin(result.get(g.home), result.get(g.away), g.neutral_site)
        delta = g.margin - expected
        resid[g.home].append(delta)
        resid[g.away].append(-delta)
    return {t: float(np.mean(v[-last_n:])) if v else 0.0 for t, v in resid.items()}


def _band_scale(x: float, lo_src: float, hi_src: float, lo_dst: float, hi_dst: float) -> float:
    if hi_src - lo_src < 1e-12:
        return (lo_dst + hi_dst) / 2.0
    return lo_dst + (x - lo_src) / (hi_src - lo_src) * (hi_dst - lo_dst)


def conference_strength(
    ratings: Mapping[str, float],
    conference_map: Mapping[str, str | None],
    *,
    top_fraction: float = 0.35,
    mid_band: tuple[float, float] = (0.40, 0.60),
    elite_weight: float = 0.65,
    depth_weight: float = 0.35,
    band: tuple[float, float] = (0.6, 1.6),
) -> dict[str, float]:
    """Robust conference strength index scaled to ``band`` (default 0.6 - 1.6).

    The raw score blends the median of each conference's top ``top_fraction`` of teams
    (how good are its contenders) with the median of its middle band (how deep is it),
    which is far less sensitive to one outlier program than a plain mean.
    """
    by_conf: dict[str, list[float]] = {}
    for team, r in ratings.items():
        if is_pool_team(team):
            continue
        by_conf.setdefault(conference_map.get(team) or "Independent", []).append(float(r))
    raw: dict[str, float] = {}
    for conf, vals in by_conf.items():
        vals = sorted(vals)
        n = len(vals)
        cut = min(max(int(n * (1.0 - top_fraction)), 0), n - 1)
        elite = median(vals[cut:])
        lo = max(int(n * mid_band[0]), 0)
        hi = max(min(int(n * mid_band[1]), n), lo + 1)
        depth = median(vals[lo:hi])
        raw[conf] = elite_weight * elite + depth_weight * depth
    if not raw:
        return {}
    lo_src, hi_src = min(raw.values()), max(raw.values())
    return {c: _band_scale(v, lo_src, hi_src, band[0], band[1]) for c, v in raw.items()}


def conference_table(
    ratings: Mapping[str, float],
    conference_map: Mapping[str, str | None],
    recs: Mapping[str, Record] | None = None,
) -> pd.DataFrame:
    strength = conference_strength(ratings, conference_map)
    rows = []
    for conf, idx in strength.items():
        members = [
            t
            for t, c in conference_map.items()
            if (c or "Independent") == conf and t in ratings and not is_pool_team(t)
        ]
        vals = [ratings[t] for t in members]
        top = max(members, key=lambda t: ratings[t]) if members else ""
        row = {
            "conference": conf,
            "strength_index": idx,
            "teams": len(members),
            "mean_rating": float(np.mean(vals)) if vals else 0.0,
            "median_rating": float(np.median(vals)) if vals else 0.0,
            "top_team": top,
        }
        if recs is not None:
            row["wins"] = sum(recs[t].wins for t in members if t in recs)
            row["losses"] = sum(recs[t].losses for t in members if t in recs)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("strength_index", ascending=False).reset_index(drop=True)


def team_table(
    teams: Sequence[str],
    games: Sequence[Game],
    result: RatingResult,
    calibrator: MarginCalibrator,
    conference_map: Mapping[str, str | None] | None = None,
    *,
    ppoints: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """One row per team: rank, rating, record, SOS, momentum (and PPoints when given).

    ``sos`` comes from completed games, or from the scheduled opponents until a team has played,
    so the preseason table ranks schedules rather than zeros; ``sos_rank`` (nullable ``Int64``)
    is missing only for a team with no games at all. ``momentum`` is 0 until the first game.
    """
    recs = records(teams, games)
    sos = strength_of_schedule(teams, games, result.ratings)
    mom = momentum(teams, games, result, calibrator)
    sos_rank = {
        t: i + 1
        for i, (t, _) in enumerate(
            sorted(((t, v) for t, v in sos.items() if not is_pool_team(t) and not math.isnan(v)), key=lambda kv: -kv[1])
        )
    }
    rows = []
    for rank, (team, rating) in enumerate(result.ranked(), start=1):
        row = {
            "rank": rank,
            "team": team,
            "conference": (conference_map or {}).get(team) or "Independent",
            "record": str(recs[team]),
            "wins": recs[team].wins,
            "losses": recs[team].losses,
            "rating": rating,
            "sos": sos[team],
            "sos_rank": sos_rank.get(team),
            "momentum": mom[team],
        }
        if ppoints is not None:
            row["ppoints"] = float(ppoints.get(team, 0.0))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sos_rank"] = df["sos_rank"].astype("Int64")
    return df


def schedule_table(
    team: str,
    games: Sequence[Game],
    result: RatingResult,
    calibrator: MarginCalibrator,
) -> pd.DataFrame:
    """Game-by-game view for one team with expected vs actual margins."""
    rank = result.rank_map()
    rows = []
    for g in sorted((g for g in games if team in (g.home, g.away)), key=lambda g: g.order_key):
        opp = g.opponent(team)
        opp_label = g.display_name(opp) + (" (FCS)" if is_pool_team(opp) else "")
        is_home = g.home == team
        rh, ra = result.get(g.home), result.get(g.away)
        exp_home = calibrator.predict_margin(rh, ra, g.neutral_site)
        exp = exp_home if is_home else -exp_home
        p = calibrator.win_prob_from_margin(exp)
        row = {
            "week": g.week,
            "type": "Bowl" if g.season_type == "postseason" else "Regular",
            "opponent": opp_label,
            "opp_rank": rank.get(opp),
            "site": "Neutral" if g.neutral_site else ("Home" if is_home else "Away"),
            "expected_margin": exp,
            "win_prob": p,
        }
        if g.has_score:
            actual = g.margin_for(team)
            row.update(
                {
                    "result": "W" if actual > 0 else "L" if actual < 0 else "T",
                    "score": f"{g.home_points}-{g.away_points}" if is_home else f"{g.away_points}-{g.home_points}",
                    "actual_margin": actual,
                    "vs_expected": actual - exp,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --- labels for the dashboard ----------------------------------------------------------------
# Pure string helpers kept out of the Streamlit script so they can be unit-tested.


def age_label(seconds: float) -> str:
    """``"just now"``, ``"20 min ago"``, ``"3 h ago"`` or ``"2 d ago"`` for an age in seconds."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} min ago"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.0f} h ago"
    return f"{hours / 24:.0f} d ago"


def freshness_label(year: int, info: CacheInfo | None) -> str:
    """One line of data provenance for a season, from ``CFBDClient.freshness(year)``.

    ``"2024: archived · final"`` for the committed archive, ``"2025: cached 3 h ago"`` for the
    working cache (``" · final"`` appended once the season is finished) and
    ``"2026: not cached · next load calls the API"`` when nothing is on disk.
    """
    if info is None:
        return f"{year}: not cached · next load calls the API"
    if info.tier == "archive":
        return f"{year}: archived · final"
    label = f"{year}: cached {age_label(info.age_seconds)}"
    return label + " · final" if info.final else label


def _eastern() -> tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/New_York")
    except Exception:  # no tz database on this machine: standard time is the better of two guesses
        return timezone(timedelta(hours=-5))


def kickoff_label(start_date: str | None) -> str | None:
    """``"Sat 22 Nov"`` for an ISO-8601 kickoff such as CFBD's ``2025-11-22T20:30:00.000Z``.

    CFBD stamps are UTC, so a 7:30 pm Eastern kickoff reads as 00:30 the next day; an aware
    timestamp is shifted to US Eastern first so the label names the day fans know. A naive
    stamp is used as given. ``None`` when there is no date or it does not parse.
    """
    if not start_date:
        return None
    try:
        when = datetime.fromisoformat(str(start_date).replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is not None:
        when = when.astimezone(_eastern())
    return f"{when:%a} {when.day} {when:%b}"
