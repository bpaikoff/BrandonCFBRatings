"""Walk-forward backtesting and hyper-parameter tuning."""

from __future__ import annotations

from .engine import BacktestConfig, BacktestResult, compare_methods, run_backtest
from .metrics import accuracy, brier_score, calibration_table, log_loss, mae, rmse, summarize_predictions
from .tuning import grid_search

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "accuracy",
    "brier_score",
    "calibration_table",
    "compare_methods",
    "grid_search",
    "log_loss",
    "mae",
    "rmse",
    "run_backtest",
    "summarize_predictions",
]
