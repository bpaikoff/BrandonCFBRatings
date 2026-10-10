"""cfbratings: college football rating systems, analytics, and backtesting.

Public surface:

* :mod:`cfbratings.data` -- typed ``Game``/``Team`` schema, cached CFBD client, synthetic seasons
* :mod:`cfbratings.models` -- Colley, Massey, Elo, Bradley-Terry and Hybrid rating models
* :mod:`cfbratings.predict` -- margin / win-probability calibration shared by every model
* :mod:`cfbratings.analytics` -- records, strength of schedule, momentum, conference strength
* :mod:`cfbratings.ppoints` -- the PPoints résumé metric
* :mod:`cfbratings.backtest` -- walk-forward evaluation and hyper-parameter tuning
"""

from __future__ import annotations

__version__ = "1.2.0"

from . import _blas  # noqa: F401  (must precede the first numpy import: pins BLAS to one thread)
from .config import Settings, settings
from .data.schema import Game, Team, completed_games, normalize_games, normalize_teams
from .models import MODEL_NAMES, fit_ratings, make_model
from .models.base import RatingResult
from .predict import MarginCalibrator

__all__ = [
    "MODEL_NAMES",
    "Game",
    "MarginCalibrator",
    "RatingResult",
    "Settings",
    "Team",
    "__version__",
    "completed_games",
    "fit_ratings",
    "make_model",
    "normalize_games",
    "normalize_teams",
    "settings",
]
