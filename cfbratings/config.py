"""Runtime configuration.

All knobs can be overridden through environment variables (or a ``.env`` file in the
working directory). Nothing here performs network access or touches the cache directory;
``Settings`` is a plain frozen dataclass that can be constructed explicitly in tests.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def default_season(today: _dt.date | None = None) -> int:
    """The season currently in progress: August onward belongs to the new season."""
    today = today or _dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


@dataclass(frozen=True)
class Settings:
    """Immutable configuration for data access and model defaults."""

    # --- data access -----------------------------------------------------------
    api_key: str = ""
    base_url: str = "https://api.collegefootballdata.com"
    cache_dir: Path = Path("data/cache")
    cache_ttl_hours: float = 6.0
    timeout: int = 20
    year: int = field(default_factory=default_season)
    season_type: str = "both"  # "regular" | "postseason" | "both"

    # --- shared ----------------------------------------------------------------
    home_field_adv: float = 2.5  # points; applied only to non-neutral games

    # --- Colley ----------------------------------------------------------------
    colley_prior_strength: float = 2.0

    # --- Massey ----------------------------------------------------------------
    massey_ridge_lambda: float = 1.0
    massey_max_margin: float = 50.0
    massey_half_life_weeks: float = 0.0  # 0 disables recency weighting
    massey_prior_lambda: float = 3.0  # ridge strength toward a preseason prior

    # --- Elo -------------------------------------------------------------------
    elo_k: float = 25.0
    elo_init: float = 1500.0
    elo_per_point: float = 25.0  # Elo points per point of spread (538 convention)
    elo_mov_multiplier: bool = True

    # --- season carry-over (all models) -----------------------------------------
    season_regress: float = 0.33  # fraction of last season's rating regressed to the mean
    fcs_handling: str = "pool"  # "pool" non-FBS opponents into one pseudo-team | "drop" those games

    # --- Bradley-Terry ---------------------------------------------------------
    bt_ridge: float = 1.0

    # --- Hybrid ----------------------------------------------------------------
    hybrid_weights: tuple[tuple[str, float], ...] = (
        ("colley", 1.0 / 3.0),
        ("massey", 1.0 / 3.0),
        ("elo", 1.0 / 3.0),
    )

    @classmethod
    def from_env(cls) -> Settings:
        base = cls()
        weights_env = os.getenv("CFB_HYBRID_WEIGHTS")
        weights = base.hybrid_weights
        if weights_env:
            parsed: list[tuple[str, float]] = []
            for part in weights_env.split(","):
                name, _, value = part.partition("=")
                if name.strip():
                    parsed.append((name.strip(), float(value or 0.0)))
            if parsed:
                weights = tuple(parsed)
        return cls(
            api_key=_env("CFBD_API_KEY", ""),
            base_url=_env("CFBD_BASE_URL", base.base_url),
            cache_dir=Path(_env("CFB_CACHE_DIR", str(base.cache_dir))),
            cache_ttl_hours=_env_float("CFB_CACHE_TTL_HOURS", base.cache_ttl_hours),
            timeout=int(_env("CFB_TIMEOUT", str(base.timeout))),
            year=int(_env("CFB_YEAR", str(base.year))),
            season_type=_env("CFB_SEASON_TYPE", base.season_type),
            home_field_adv=_env_float("CFB_HFA", base.home_field_adv),
            colley_prior_strength=_env_float("CFB_COLLEY_PRIOR", base.colley_prior_strength),
            massey_ridge_lambda=_env_float("CFB_MASSEY_LAMBDA", base.massey_ridge_lambda),
            massey_max_margin=_env_float("CFB_MASSEY_MAX_MARGIN", base.massey_max_margin),
            massey_half_life_weeks=_env_float("CFB_MASSEY_HALF_LIFE", base.massey_half_life_weeks),
            massey_prior_lambda=_env_float("CFB_MASSEY_PRIOR_LAMBDA", base.massey_prior_lambda),
            elo_k=_env_float("CFB_ELO_K", base.elo_k),
            elo_init=_env_float("CFB_ELO_INIT", base.elo_init),
            elo_per_point=_env_float("CFB_ELO_PER_POINT", base.elo_per_point),
            elo_mov_multiplier=_env("CFB_ELO_MOV", "1") not in {"0", "false", "False"},
            season_regress=_env_float("CFB_SEASON_REGRESS", _env_float("CFB_ELO_REGRESS", base.season_regress)),
            fcs_handling=_env("CFB_FCS", base.fcs_handling),
            bt_ridge=_env_float("CFB_BT_RIDGE", base.bt_ridge),
            hybrid_weights=weights,
        )

    def with_overrides(self, **overrides: Any) -> Settings:
        """Return a copy with the given fields replaced (unknown keys are ignored)."""
        valid = {k: v for k, v in overrides.items() if k in self.__dataclass_fields__}
        return replace(self, **valid)

    @property
    def hybrid_weight_map(self) -> dict[str, float]:
        return dict(self.hybrid_weights)


settings = Settings.from_env()
