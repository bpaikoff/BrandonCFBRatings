from statistics import mean

import numpy as np
from cfbratings.config import settings
from cfbratings.io import load_weekly_ratings
from statistics import median
from typing import Dict, List, Tuple

def records(team_list: List[str], games: List[dict]) -> Dict[str, Tuple[int, int]]:
    team_to_idx = {t: i for i, t in enumerate(team_list)}
    w = {t: 0 for t in team_list}
    l = {t: 0 for t in team_list}
    for g in games:
        if not g.get("completed", False):
            continue
        hp = g.get("homePoints"); ap = g.get("awayPoints")
        if hp is None or ap is None:
            continue
        home = g["homeTeam"]; away = g["awayTeam"]
        if home not in team_to_idx or away not in team_to_idx:
            continue
        if hp > ap:
            w[home] += 1; l[away] += 1
        elif ap > hp:
            w[away] += 1; l[home] += 1
        else:
            w[home] += 1; w[away] += 1
    return {t: (w[t], l[t]) for t in team_list}

def strength_of_schedule(team_list: List[str], games: List[dict], base_ratings: Dict[str, float]) -> Dict[str, float]:
    # Average opponent rating weighted by games played
    opps = {t: [] for t in team_list}
    for g in games:
        if not g.get("completed", False):
            continue
        home = g.get("homeTeam"); away = g.get("awayTeam")
        if home in opps and away in opps:
            opps[home].append(base_ratings.get(away, 0.0))
            opps[away].append(base_ratings.get(home, 0.0))
    return {t: float(np.mean(opps[t])) if opps[t] else 0.0 for t in team_list}

def momentum(team_list: List[str], games: List[dict], base_ratings: Dict[str, float]) -> Dict[str, float]:
        recent = {t: [] for t in team_list}
        team_set = set(team_list)

        for g in sorted(games, key=lambda x: (x.get("season", 0), x.get("week", 0))):
            if not g.get("completed", False):
                continue
            hp, ap = g.get("homePoints"), g.get("awayPoints")
            if hp is None or ap is None:
                continue
            home, away = g["homeTeam"], g["awayTeam"]

            # Skip if either team not in FBS list
            if home not in team_set or away not in team_set:
                continue

            eh = base_ratings.get(home, 0.0)
            ea = base_ratings.get(away, 0.0)
            expected_margin = eh - ea
            actual_margin = hp - ap
            delta_h = actual_margin - expected_margin
            delta_a = -delta_h

            recent[home].append(delta_h)
            recent[away].append(delta_a)

        def last3(xs):
            return float(np.mean(xs[-3:])) if xs else 0.0

        return {t: last3(recent[t]) for t in team_list}

def _quantiles(values: List[float], qs: List[float]) -> List[float]:
    if not values:
        return [0.0 for _ in qs]
    vals = sorted(values)
    n = len(vals)
    out = []
    for q in qs:
        idx = min(max(int(round(q * (n - 1))), 0), n - 1)
        out.append(vals[idx])
    return out

def _band_scale(x: float, lo_src: float, hi_src: float, lo_dst: float = 0.6, hi_dst: float = 1.6) -> float:
    if hi_src <= lo_src:
        return 1.0
    t = (x - lo_src) / (hi_src - lo_src)
    return lo_dst + t * (hi_dst - lo_dst)

def compute_conference_strength_robust(
    ratings: Dict[str, float],
    conference_map: Dict[str, str],
    top_percent: float = 0.35,        # elite slice
    mid_lo: float = 0.40,             # middle band (depth)
    mid_hi: float = 0.60,
    elite_weight: float = 0.65,       # emphasis on elite median
    depth_weight: float = 0.35,       # emphasis on middle median
) -> Dict[str, float]:
    # Group ratings by conference
    conf_ratings: Dict[str, List[float]] = {}
    for team, r in ratings.items():
        conf = conference_map.get(team, "Unknown")
        conf_ratings.setdefault(conf, []).append(r)

    # Compute robust per-conference score
    conf_score: Dict[str, float] = {}
    for conf, vals in conf_ratings.items():
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        if n == 0:
            conf_score[conf] = 1.0
            continue

        # Elite median: median of top_percent slice
        cut = max(int(n * (1 - top_percent)), 0)
        elite_slice = vals_sorted[cut:] if cut < n else vals_sorted[-1:]
        elite_med = median(elite_slice) if elite_slice else vals_sorted[-1]

        # Depth median: median of middle band
        lo_idx = max(int(n * mid_lo), 0)
        hi_idx = min(int(n * mid_hi), n)
        mid_slice = vals_sorted[lo_idx:hi_idx] if hi_idx > lo_idx else vals_sorted[lo_idx:lo_idx+1]
        depth_med = median(mid_slice) if mid_slice else median(vals_sorted)

        # Robust composite
        score = elite_weight * elite_med + depth_weight * depth_med
        conf_score[conf] = score

    # Scale to wider band for separation
    src_min = min(conf_score.values()) if conf_score else 0.0
    src_max = max(conf_score.values()) if conf_score else 1.0

    conf_strength: Dict[str, float] = {
        conf: _band_scale(val, src_min, src_max, lo_dst=0.6, hi_dst=1.6)
        for conf, val in conf_score.items()
    }
    return conf_strength

def ppoints(team_list, games, ratings, conference_map, method="hybrid", weight_current=0.5):
    scores = {t: 0.0 for t in team_list}

    # Current conference strength (end of season snapshot)
    current_conf_strength = compute_conference_strength_robust(ratings, conference_map)

    # Current rank map for tiering
    sorted_current = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)
    current_rank_map = {team: rank+1 for rank, (team, _) in enumerate(sorted_current)}

    def tier_points(rank: int) -> int:
        if rank <= 10: return 8
        elif rank <= 25: return 5
        elif rank <= 40: return 3
        elif rank <= 60: return 2
        else: return 1

    for g in games:
        loss_penalty_factor = 4.0  # tune this constant
        extra_win_factor = 1.0

        if not g.get("completed", False):
            continue
        hp, ap = g.get("homePoints"), g.get("awayPoints")
        if hp is None or ap is None:
            continue
        home, away = g["homeTeam"], g["awayTeam"]
        week = g.get("week", 0)

        # Weekly ratings snapshot for game-time rank
        week_ratings = load_weekly_ratings(settings.year, week, method) or ratings
        sorted_week = sorted(week_ratings.items(), key=lambda kv: kv[1], reverse=True)
        week_rank_map = {team: rank+1 for rank, (team, _) in enumerate(sorted_week)}

        # Tier points at game time vs current
        game_points_home = tier_points(week_rank_map.get(away, 1000))
        game_points_away = tier_points(week_rank_map.get(home, 1000))
        current_points_home = tier_points(current_rank_map.get(away, 1000))
        current_points_away = tier_points(current_rank_map.get(home, 1000))

        # Blend
        opp_points_home = (1 - weight_current) * game_points_home + weight_current * current_points_home
        opp_points_away = (1 - weight_current) * game_points_away + weight_current * current_points_away

        # Away multiplier
        base_home = opp_points_home * 1.00
        base_away = opp_points_away * 1.20

        # Conference strength blending
        week_conf_strength = compute_conference_strength_robust(week_ratings, conference_map)
        home_conf, away_conf = g.get("homeConference"), g.get("awayConference")
        home_strength = (1 - weight_current) * week_conf_strength.get(home_conf, 1.0) + weight_current * current_conf_strength.get(home_conf, 1.0)
        away_strength = (1 - weight_current) * week_conf_strength.get(away_conf, 1.0) + weight_current * current_conf_strength.get(away_conf, 1.0)

        base_home *= away_strength
        base_away *= home_strength

        # Non-conference multiplier
        if home_conf and away_conf and home_conf != away_conf:
            loss_penalty_factor = 2.0
            extra_win_factor = 1.5

        # Win multiplier
        if home not in scores or away not in scores:
            continue
        else:
            if hp > ap:
                scores[home] += extra_win_factor * base_home
                scores[away] -= loss_penalty_factor / max(opp_points_home, 1.0)
            elif ap > hp:
                scores[away] += extra_win_factor * base_away
                scores[home] -= loss_penalty_factor / max(opp_points_away, 1.0)


    return scores