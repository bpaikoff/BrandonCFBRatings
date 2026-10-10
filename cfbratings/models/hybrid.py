"""Hybrid ensemble: a weighted blend of standardised component ratings.

Each component model is fitted independently, its ratings are converted to z-scores
(so that Colley's [0, 1] scale, Massey's points and Elo's 1500-centred scale are
commensurable) and the weighted sum is returned. Weights are normalised to sum to one.

A hybrid ``prior`` is a mapping ``{component_name: {team: rating}}`` -- one prior per
component in that component's own units (see :func:`cfbratings.priors.preseason_prior`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..data.schema import Game
from .base import Prior, RatingModel, RatingResult


@dataclass
class HybridModel:
    components: Mapping[str, RatingModel]
    weights: Mapping[str, float] = field(default_factory=dict)
    name: str = "hybrid"

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("HybridModel needs at least one component model")
        weights = {k: float(self.weights.get(k, 1.0)) for k in self.components}
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Hybrid weights must sum to a positive number")
        self.weights = {k: v / total for k, v in weights.items()}

    def fit(
        self,
        teams: Sequence[str],
        games: Sequence[Game],
        prior: Mapping[str, Prior] | Prior | None = None,
    ) -> RatingResult:
        blended = dict.fromkeys(teams, 0.0)
        component_results: dict[str, RatingResult] = {}
        for key, model in self.components.items():
            component_prior = prior.get(key) if isinstance(prior, Mapping) and key in prior else None
            res = model.fit(teams, games, prior=component_prior)
            component_results[key] = res
            z = res.zscores()
            w = self.weights[key]
            for t in teams:
                blended[t] += w * z.get(t, 0.0)
        return RatingResult(
            method=self.name,
            ratings=blended,
            scale="z",
            params={"weights": dict(self.weights)},
            extras={"components": component_results},
        )
