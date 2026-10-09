"""Grid search over model hyper-parameters using walk-forward backtests."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from ..config import Settings
from ..data.schema import Game
from .engine import BacktestConfig, run_backtest

LOWER_IS_BETTER = {"brier", "log_loss", "mae", "rmse"}


def grid_search(
    teams: Sequence[str],
    games: Sequence[Game],
    method: str,
    grid: Mapping[str, Sequence[Any]],
    *,
    metric: str = "log_loss",
    start_week: int = 4,
    settings: Settings | None = None,
    warmup_games: Sequence[Game] | None = None,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Evaluate every combination in ``grid``; rows sorted best-first by ``metric``."""
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    rows = []
    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo, strict=True))
        res = run_backtest(
            teams,
            games,
            BacktestConfig(method=method, params=params, start_week=start_week),
            settings=settings,
            warmup_games=warmup_games,
        )
        summary = res.summary()
        row = {**params, **{k: v for k, v in summary.items() if not k.startswith("baseline_")}}
        row.pop("method", None)
        rows.append(row)
        if progress:
            progress(i, len(combos), row)
    df = pd.DataFrame(rows)
    if df.empty or metric not in df.columns:
        return df
    return df.sort_values(metric, ascending=metric in LOWER_IS_BETTER).reset_index(drop=True)
