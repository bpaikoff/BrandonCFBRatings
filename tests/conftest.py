from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import requests

from cfbratings.config import Settings
from cfbratings.data import SyntheticSeason, synthetic_season

# Every module that captured the process-wide ``settings`` object at import time. The suite
# replaces them all so that a developer's real ``CFBD_API_KEY`` (or ``.env``) can never turn a
# test into an API call, and nothing is written under the repository's ``data/`` tree.
_SETTINGS_HOLDERS = (
    "cfbratings.settings",
    "cfbratings.config.settings",
    "cfbratings.cli.default_settings",
    "cfbratings.backtest.engine.default_settings",
    "cfbratings.data.cfbd.default_settings",
    "cfbratings.models.default_settings",
    "cfbratings.priors.default_settings",
    "cfbratings.snapshots.default_settings",
)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path) -> Settings:
    """Run every test without an API key, with throw-away cache directories and no network."""
    from cfbratings.config import settings as live

    s = dataclasses.replace(live, api_key="", cache_dir=tmp_path / "cache", archive_dir=tmp_path / "archive")
    for target in _SETTINGS_HOLDERS:
        monkeypatch.setattr(target, s)

    def blocked(self, method, url, *args, **kwargs):
        raise AssertionError(f"the test-suite must not reach the network: {method} {url}")

    monkeypatch.setattr(requests.Session, "request", blocked)
    return s


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
