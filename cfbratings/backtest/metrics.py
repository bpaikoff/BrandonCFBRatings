"""Scoring rules for probabilistic and point-spread predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def accuracy(p_home, home_win) -> float:
    """Straight-up pick accuracy (ties in the outcome are excluded)."""
    p, y = _arr(p_home), _arr(home_win)
    mask = (y == 0.0) | (y == 1.0)
    if not mask.any():
        return float("nan")
    picks = (p[mask] >= 0.5).astype(float)
    return float(np.mean(picks == y[mask]))


def brier_score(p_home, home_win) -> float:
    p, y = _arr(p_home), _arr(home_win)
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


def log_loss(p_home, home_win) -> float:
    p, y = np.clip(_arr(p_home), EPS, 1 - EPS), _arr(home_win)
    if not p.size:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mae(pred_margin, actual_margin) -> float:
    a, b = _arr(pred_margin), _arr(actual_margin)
    return float(np.mean(np.abs(a - b))) if a.size else float("nan")


def rmse(pred_margin, actual_margin) -> float:
    a, b = _arr(pred_margin), _arr(actual_margin)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")


def calibration_table(p_home, home_win, bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data: predicted vs observed home-win rate per probability bin."""
    p, y = _arr(p_home), _arr(home_win)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "bin_mid": (edges[b] + edges[b + 1]) / 2,
                "n": int(mask.sum()),
                "predicted": float(p[mask].mean()) if mask.any() else float("nan"),
                "observed": float(y[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions(df: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    """Compute every headline metric from a predictions frame.

    Expects columns ``p_home``, ``home_win``, ``pred_margin`` and ``actual_margin``.
    """
    if df.empty:
        return {f"{prefix}n": 0}
    out = {
        f"{prefix}n": len(df),
        f"{prefix}accuracy": accuracy(df["p_home"], df["home_win"]),
        f"{prefix}brier": brier_score(df["p_home"], df["home_win"]),
        f"{prefix}log_loss": log_loss(df["p_home"], df["home_win"]),
        f"{prefix}mae": mae(df["pred_margin"], df["actual_margin"]),
        f"{prefix}rmse": rmse(df["pred_margin"], df["actual_margin"]),
    }
    if "vegas_margin" in df.columns and df["vegas_margin"].notna().any():
        sub = df.dropna(subset=["vegas_margin"])
        out[f"{prefix}vegas_n"] = len(sub)
        out[f"{prefix}vegas_mae"] = mae(sub["vegas_margin"], sub["actual_margin"])
        covered = np.sign(sub["actual_margin"] - sub["vegas_margin"])
        picked = np.sign(sub["pred_margin"] - sub["vegas_margin"])
        decided = covered != 0
        out[f"{prefix}ats_accuracy"] = (
            float(np.mean(covered[decided] == picked[decided])) if decided.any() else float("nan")
        )
    return out
