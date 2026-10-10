"""Preseason priors: carry last season's ratings into the new one.

College football seasons are short (12-13 games) and teams change less between seasons
than a cold start assumes, so every serious rating system seeds the new season with a
regressed version of the previous one. :func:`preseason_prior` does exactly that for any
model: fit the previous season (itself warm-started when a chain is available), then pull
each rating a fraction ``regress`` of the way back to the model's neutral value.

The resulting mapping is passed as ``prior=`` to ``model.fit``; see each model's
docstring for how the prior enters its equations. Hybrid priors are a dict of
component priors (one per blended model).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .config import settings as default_settings
from .data.schema import Game
from .models import make_model
from .models.base import Prior, RatingModel, RatingResult
from .models.elo import EloModel
from .models.hybrid import HybridModel
from .predict import MarginCalibrator


@dataclass(frozen=True)
class WarmStart:
    """What a new season inherits: a rating prior and last season's calibrator."""

    prior: Prior | dict[str, Prior] | None = None
    calibrator: MarginCalibrator | None = None
    seasons: tuple[int, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.prior)


def neutral_rating(model: RatingModel) -> float:
    """The rating an unknown team gets: Elo's ``init``, Colley's 0.5, otherwise 0."""
    if isinstance(model, EloModel):
        return model.init
    return 0.5 if getattr(model, "name", "") == "colley" else 0.0


def regress_toward(ratings: Mapping[str, float], fraction: float, center: float | None = None) -> dict[str, float]:
    """Move every rating ``fraction`` of the way toward ``center`` (default: the mean)."""
    if not ratings:
        return {}
    c = center if center is not None else sum(ratings.values()) / len(ratings)
    return {t: float(r + (c - r) * fraction) for t, r in ratings.items()}


def result_to_prior(model: RatingModel, result: RatingResult, regress: float) -> Prior | dict[str, Prior]:
    """Regressed end-of-season ratings in the units the model expects as a prior."""
    if isinstance(model, HybridModel):
        comps: dict[str, RatingResult] = result.extras.get("components", {})
        return {
            key: result_to_prior(model.components[key], comps[key], regress) for key in model.components if key in comps
        }
    return regress_toward(result.ratings, regress, neutral_rating(model))


def preseason_prior(
    model: RatingModel,
    teams: Sequence[str],
    previous_games: Sequence[Game],
    *,
    regress: float,
    previous_prior: Any = None,
) -> Prior | dict[str, Prior]:
    """Prior for the coming season from the previous season's games.

    ``teams`` should contain every team that appears in either season; teams without
    games keep the neutral rating. ``previous_prior`` warm-starts the previous season
    itself, which is how multi-season chains are built.
    """
    if not previous_games:
        return {}
    result = model.fit(teams, previous_games, prior=previous_prior)
    return result_to_prior(model, result, regress)


def season_priors(
    method: str,
    games: Sequence[Game],
    teams: Sequence[str],
    *,
    settings: Settings | None = None,
    regress: float | None = None,
    **overrides: Any,
) -> dict[int, Prior | dict[str, Prior]]:
    """Chain priors across every season in ``games``: ``{season: prior entering it}``.

    The earliest season gets an empty prior (cold start); each later season's prior is the
    regressed fit of the season before it, which was itself warm-started.
    """
    s = settings or default_settings
    model = make_model(method, s, **overrides)
    fraction = s.season_regress if regress is None else regress
    seasons = sorted({g.season for g in games if g.has_score})
    out: dict[int, Prior | dict[str, Prior]] = {}
    prior: Any = None
    for season in seasons:
        out[season] = prior or {}
        season_games = [g for g in games if g.season == season and g.has_score]
        prior = preseason_prior(model, teams, season_games, regress=fraction, previous_prior=prior)
    return out


def prior_for_season(
    method: str,
    games: Sequence[Game],
    teams: Sequence[str],
    season: int,
    *,
    settings: Settings | None = None,
    **overrides: Any,
) -> Prior | dict[str, Prior]:
    """Prior entering ``season`` built from all earlier seasons present in ``games``."""
    return warm_start(method, games, teams, season, settings=settings, **overrides).prior or {}


def warm_start(
    method: str,
    games: Sequence[Game],
    teams: Sequence[str],
    season: int,
    *,
    settings: Settings | None = None,
    regress: float | None = None,
    **overrides: Any,
) -> WarmStart:
    """Everything ``season`` inherits from the earlier seasons present in ``games``.

    Priors are chained season by season; the calibrator is the cross-validated fit of the
    season immediately before ``season`` (warm-started itself), which keeps week-1 spreads
    on a sensible scale for models whose ratings have no natural point units.
    """
    s = settings or default_settings
    earlier = sorted({g.season for g in games if g.has_score and g.season < season})
    if not earlier:
        return WarmStart()
    model = make_model(method, s, **overrides)
    fraction = s.season_regress if regress is None else regress
    prior: Any = None
    calibrator: MarginCalibrator | None = None
    for year in earlier:
        year_games = [g for g in games if g.season == year and g.has_score]
        calibrator = MarginCalibrator.fit_cv(
            model, teams, year_games, fallback_hfa=s.home_field_adv, prior=prior, fallback=calibrator
        )
        prior = preseason_prior(model, teams, year_games, regress=fraction, previous_prior=prior)
    return WarmStart(prior=prior or None, calibrator=calibrator, seasons=tuple(earlier))
