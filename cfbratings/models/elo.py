import math
from typing import Dict, List

def run_elo(team_list: List[str], games: List[dict], init: float = 1500.0, k: float = 25.0,
            regress_to_mean: float = 0.20, hfa: float = 2.1) -> Dict[str, float]:
    # Elo on margins via expected win probability (logistic), with HFA added to home rating
    ratings = {t: init for t in team_list}
    team_to_idx = {t: i for i, t in enumerate(team_list)}
    mean = init

    # Optional: regress-to-mean at season checkpoints (e.g., weekly or postseason)
    def regress():
        for t in ratings:
            ratings[t] = ratings[t] * (1 - regress_to_mean) + mean * regress_to_mean

    # Process in chronological order if available
    games_sorted = sorted(games, key=lambda g: (g.get("season", 0), g.get("week", 0), g.get("id", 0)))
    week_marker = None

    for g in games_sorted:
        if not g.get("completed", False):
            continue
        hp = g.get("homePoints"); ap = g.get("awayPoints")
        if hp is None or ap is None:
            continue
        home = g["homeTeam"]; away = g["awayTeam"]
        if home not in ratings or away not in ratings:
            continue
        # Weekly regression
        w = g.get("week")
        if week_marker is None:
            week_marker = w
        elif w is not None and w != week_marker:
            regress()
            week_marker = w

        Rh = ratings[home] + hfa
        Ra = ratings[away]
        # Expected win probability (home)
        exp_h = 1.0 / (1.0 + 10 ** ((Ra - Rh) / 400.0))
        # Actual result
        if hp > ap:
            out_h = 1.0
        elif ap > hp:
            out_h = 0.0
        else:
            out_h = 0.5
        # Margin-based multiplier (soft cap)
        margin = abs(hp - ap)
        mult = math.log(max(margin, 1) + 1) * (2.2 / ((Rh - Ra) * 0.001 + 2.2))

        delta = k * mult * (out_h - exp_h)
        ratings[home] += delta
        ratings[away] -= delta

    return ratings