import numpy as np
from typing import List, Dict

def build_massey(team_list: List[str], games: List[dict], ridge_lambda: float = 0.01, hfa: float = 2.1):
    n = len(team_list)
    team_to_idx = {t: i for i, t in enumerate(team_list)}
    # Massey: A r = y, where A encodes matchups, y = margin (home - away adjusted)
    rows = []
    y = []
    for g in games:
        if not g.get("completed", False):
            continue
        hp = g.get("homePoints"); ap = g.get("awayPoints")
        if hp is None or ap is None:
            continue
        home = g["homeTeam"]; away = g["awayTeam"]
        if home not in team_to_idx or away not in team_to_idx:
            continue
        i = team_to_idx[home]; j = team_to_idx[away]
        margin = (hp - ap) - hfa  # subtract home-field advantage
        row = np.zeros(n); row[i] = 1.0; row[j] = -1.0
        rows.append(row); y.append(margin)
    if len(rows) == 0:
        return np.zeros((n, n)), np.zeros(n)

    A = np.vstack(rows)
    y = np.array(y)

    # Ridge regularization for stability
    M = A.T @ A + ridge_lambda * np.eye(n)
    b = A.T @ y

    # Anchor the mean rating to 0 (or any constant) for identifiability
    # Replace last equation with sum(r) = 0
    M[-1, :] = 1.0
    b[-1] = 0.0
    return M, b

def solve_massey(M: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(M, b)