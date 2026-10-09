"""Team-level analytics built on top of a fitted :class:`RatingResult`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import median

import numpy as np
import pandas as pd

from .data.schema import Game, Record
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
    """Mean opponent rating over games played (same units as ``ratings``)."""
    opp: dict[str, list[float]] = {t: [] for t in teams}
    for g in games:
        if not g.has_score or g.home not in opp or g.away not in opp:
            continue
        opp[g.home].append(ratings.get(g.away, 0.0))
        opp[g.away].append(ratings.get(g.home, 0.0))
    return {t: float(np.mean(v)) if v else 0.0 for t, v in opp.items()}


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
    model. Positive values mean the team has recently beaten its own rating.
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
        members = [t for t, c in conference_map.items() if (c or "Independent") == conf and t in ratings]
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
    """One row per team: rank, rating, record, SOS, momentum (and PPoints when given)."""
    recs = records(teams, games)
    sos = strength_of_schedule(teams, games, result.ratings)
    mom = momentum(teams, games, result, calibrator)
    sos_rank = {t: i + 1 for i, (t, _) in enumerate(sorted(sos.items(), key=lambda kv: -kv[1]))}
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
            "sos_rank": sos_rank[team],
            "momentum": mom[team],
        }
        if ppoints is not None:
            row["ppoints"] = float(ppoints.get(team, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


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
        is_home = g.home == team
        rh, ra = result.get(g.home), result.get(g.away)
        exp_home = calibrator.predict_margin(rh, ra, g.neutral_site)
        exp = exp_home if is_home else -exp_home
        p = calibrator.win_prob_from_margin(exp)
        row = {
            "week": g.week,
            "type": "Bowl" if g.season_type == "postseason" else "Regular",
            "opponent": opp,
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
