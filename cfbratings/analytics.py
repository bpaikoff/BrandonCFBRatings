import numpy as np
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

## Adding **PPoints**

def ppoints(team_list: List[str], games: List[dict], ratings: Dict[str, float]) -> Dict[str, float]:
    team_set = set(team_list)
    scores = {t: 0.0 for t in team_list}

    # Helper: scale opponent rating to 1–5
    def scale_rating(r: float, all_ratings: List[float]) -> float:
        if not all_ratings:
            return 1.0
        min_r, max_r = min(all_ratings), max(all_ratings)
        if max_r == min_r:
            return 1.0
        return 1 + 4 * (r - min_r) / (max_r - min_r)

    all_ratings = list(ratings.values())

    for g in games:
        if not g.get("completed", False):
            continue
        hp, ap = g.get("homePoints"), g.get("awayPoints")
        if hp is None or ap is None:
            continue
        home, away = g["homeTeam"], g["awayTeam"]
        if home not in team_set or away not in team_set:
            continue  # skip non-FBS

        # Opponent rating scaled
        opp_rating_home = scale_rating(ratings.get(away, 0.0), all_ratings)
        opp_rating_away = scale_rating(ratings.get(home, 0.0), all_ratings)

        # Base points
        base_home = opp_rating_home
        base_away = opp_rating_away

        # Away multiplier
        base_home *= 1.0
        base_away *= 1.5

        # Win multiplier
        if hp > ap:
            scores[home] += base_home * 2
            scores[away] += base_away
        elif ap > hp:
            scores[home] += base_home
            scores[away] += base_away * 2
        else:
            scores[home] += base_home
            scores[away] += base_away

    return scores