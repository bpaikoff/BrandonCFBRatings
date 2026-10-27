"""Plotly figures shared by the dashboard (and exportable from the CLI)."""

from __future__ import annotations

from .charts import (
    CATEGORICAL,
    PALETTE,
    THEMES,
    calibration_chart,
    category_colors,
    conference_bar,
    conference_dots,
    contrast_ratio,
    probability_heatmap,
    rating_bar,
    rating_sos_scatter,
    resolve_team_color,
    series_colors,
    trajectory_chart,
    weekly_metric_chart,
    win_distribution_chart,
)

__all__ = [
    "CATEGORICAL",
    "PALETTE",
    "THEMES",
    "calibration_chart",
    "category_colors",
    "conference_bar",
    "conference_dots",
    "contrast_ratio",
    "probability_heatmap",
    "rating_bar",
    "rating_sos_scatter",
    "resolve_team_color",
    "series_colors",
    "trajectory_chart",
    "weekly_metric_chart",
    "win_distribution_chart",
]
