from __future__ import annotations

import numpy as np
import pytest

from cfbratings.config import Settings
from cfbratings.data import SyntheticSeason, synthetic_season


@pytest.fixture(scope="session")
def season() -> SyntheticSeason:
    """A deterministic 48-team season with known latent strengths."""
    return synthetic_season(
        season=2025, n_teams=48, n_conferences=6, weeks=12, strength_sd=11.0, noise_sd=12.0, seed=42
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(cache_dir=tmp_path / "cache", archive_dir=tmp_path / "archive", api_key="")


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(np.asarray(a)))
    rb = np.argsort(np.argsort(np.asarray(b)))
    return float(np.corrcoef(ra, rb)[0, 1])


@pytest.fixture(scope="session")
def rank_corr():
    return spearman
