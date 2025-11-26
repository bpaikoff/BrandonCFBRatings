import numpy as np
from typing import List, Dict
from .colley import build_colley, solve_colley
from .massey import build_massey, solve_massey

def hybrid_rating(team_list: List[str], games: List[dict],
                  colley_weight: float = 0.5, massey_weight: float = 0.5,
                  prior_strength: float = 2.0, ridge_lambda: float = 0.01, hfa: float = 2.1) -> Dict[str, float]:
    C, b = build_colley(team_list, games, prior_strength=prior_strength)
    colley_r = solve_colley(C, b)

    M, mb = build_massey(team_list, games, ridge_lambda=ridge_lambda, hfa=hfa)
    massey_r = solve_massey(M, mb)

    # Normalize scales (Colley is ~[0,1], Massey centered ~0; z-score both then blend)
    def zscore(x):
        m = np.mean(x)
        s = np.std(x)
        # Avoid division by zero if all ratings are identical
        if s < 1e-8:
            s = 1.0
        return (x - m) / s

    zc = zscore(colley_r)
    zm = zscore(massey_r)
    blend = colley_weight * zc + massey_weight * zm

    return {team_list[i]: float(blend[i]) for i in range(len(team_list))}