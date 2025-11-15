import numpy as np
from typing import Dict, List, Tuple

def build_colley(team_list: List[str], games: List[dict], prior_strength: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    n = len(team_list)
    team_to_idx = {t: i for i, t in enumerate(team_list)}
    wins = np.zeros(n, dtype=float)
    losses = np.zeros(n, dtype=float)
    games_played = np.zeros(n, dtype=float)
    opponent_matrix = np.zeros((n, n), dtype=float)

    for g in games:
        game = g if isinstance(g, dict) else g
        if isinstance(games, dict):  # if using cached payload {"data": [...]}
            game = g
        if not game.get("completed", False):
            continue
        hp = game.get("homePoints")
        ap = game.get("awayPoints")
        if hp is None or ap is None:
            continue
        home = game["homeTeam"]
        away = game["awayTeam"]
        if home not in team_to_idx or away not in team_to_idx:
            continue
        i = team_to_idx[home]; j = team_to_idx[away]

        games_played[i] += 1; games_played[j] += 1
        opponent_matrix[i, j] += 1; opponent_matrix[j, i] += 1

        if hp > ap:
            wins[i] += 1; losses[j] += 1
        elif ap > hp:
            wins[j] += 1; losses[i] += 1
        else:
            wins[i] += 0.5; wins[j] += 0.5
            losses[i] += 0.5; losses[j] += 0.5

    C = np.diag(prior_strength + games_played) - opponent_matrix
    b = (prior_strength / 2.0) + 0.5 * (wins - losses)
    return C, b

def solve_colley(C: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(C, b)