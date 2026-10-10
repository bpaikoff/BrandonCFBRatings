"""Model registry.

>>> from cfbratings.models import make_model, fit_ratings
>>> make_model("elo", k=30).name
'elo'
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..config import Settings
from ..config import settings as default_settings
from ..data.schema import Game
from .base import Prior, RatingModel, RatingResult, Ratings
from .bradley_terry import BradleyTerryModel
from .colley import ColleyModel
from .elo import EloModel
from .hybrid import HybridModel
from .massey import MasseyModel

MODEL_NAMES: tuple[str, ...] = ("colley", "massey", "elo", "bradley_terry", "hybrid")

MODEL_LABELS: dict[str, str] = {
    "colley": "Colley",
    "massey": "Massey",
    "elo": "Elo",
    "bradley_terry": "Bradley-Terry",
    "hybrid": "Hybrid",
}


def _pick(overrides: Mapping[str, Any], key: str, default: Any) -> Any:
    return overrides[key] if key in overrides and overrides[key] is not None else default


def make_model(name: str, settings: Settings | None = None, **overrides: Any) -> RatingModel:
    """Instantiate a model by name using ``settings`` for defaults.

    ``overrides`` may use either the model's own parameter names (``k``, ``ridge_lambda``)
    or the ``Settings`` names (``elo_k``, ``massey_ridge_lambda``).
    """
    s = settings or default_settings
    name = name.lower().replace("-", "_")
    hfa = _pick(overrides, "hfa", _pick(overrides, "home_field_adv", s.home_field_adv))
    if name == "colley":
        return ColleyModel(
            prior_strength=_pick(
                overrides, "prior_strength", _pick(overrides, "colley_prior_strength", s.colley_prior_strength)
            )
        )
    if name == "massey":
        return MasseyModel(
            ridge_lambda=_pick(
                overrides, "ridge_lambda", _pick(overrides, "massey_ridge_lambda", s.massey_ridge_lambda)
            ),
            hfa=hfa,
            max_margin=_pick(overrides, "max_margin", _pick(overrides, "massey_max_margin", s.massey_max_margin)),
            half_life_weeks=_pick(
                overrides, "half_life_weeks", _pick(overrides, "massey_half_life_weeks", s.massey_half_life_weeks)
            ),
            prior_lambda=_pick(
                overrides, "prior_lambda", _pick(overrides, "massey_prior_lambda", s.massey_prior_lambda)
            ),
        )
    if name == "elo":
        return EloModel(
            k=_pick(overrides, "k", _pick(overrides, "elo_k", s.elo_k)),
            init=_pick(overrides, "init", _pick(overrides, "elo_init", s.elo_init)),
            hfa=hfa,
            elo_per_point=_pick(overrides, "elo_per_point", s.elo_per_point),
            season_regress=_pick(overrides, "season_regress", _pick(overrides, "elo_season_regress", s.season_regress)),
            mov=_pick(overrides, "mov", _pick(overrides, "elo_mov_multiplier", s.elo_mov_multiplier)),
        )
    if name == "bradley_terry":
        return BradleyTerryModel(ridge=_pick(overrides, "ridge", _pick(overrides, "bt_ridge", s.bt_ridge)))
    if name == "hybrid":
        weights = dict(_pick(overrides, "weights", _pick(overrides, "hybrid_weights", s.hybrid_weight_map)))
        unknown = (set(weights) - set(MODEL_NAMES)) | ({"hybrid"} & set(weights))
        if unknown:
            raise ValueError(f"hybrid cannot blend {sorted(unknown)}")
        components = {key: make_model(key, s, **overrides) for key in weights}
        return HybridModel(components=components, weights=weights)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def fit_ratings(
    name: str,
    teams: Sequence[str],
    games: Sequence[Game],
    settings: Settings | None = None,
    prior: Prior | Mapping[str, Prior] | None = None,
    **overrides: Any,
) -> RatingResult:
    return make_model(name, settings, **overrides).fit(teams, games, prior=prior)


__all__ = [
    "MODEL_LABELS",
    "MODEL_NAMES",
    "BradleyTerryModel",
    "ColleyModel",
    "EloModel",
    "HybridModel",
    "MasseyModel",
    "Prior",
    "RatingModel",
    "RatingResult",
    "Ratings",
    "fit_ratings",
    "make_model",
]
