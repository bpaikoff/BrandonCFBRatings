import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()  # reads .env and sets os.environ


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("CFBD_API_KEY", "")
    base_url: str = "https://api.collegefootballdata.com"
    cache_dir: str = os.getenv("CFB_CACHE_DIR", "data/cache")
    timeout: int = 20
    year: int = int(os.getenv("CFB_YEAR", "2025"))
    season_type: str = os.getenv("CFB_SEASON_TYPE", "both")  # "regular" | "postseason" | "both"
    # Rating knobs
    home_field_adv: float = float(os.getenv("CFB_HFA", "2.1"))  # points
    colley_prior_strength: float = float(os.getenv("CFB_COLLEY_PRIOR", "2.0"))
    massey_ridge_lambda: float = float(os.getenv("CFB_MASSEY_LAMBDA", "0.01"))
    elo_k: float = float(os.getenv("CFB_ELO_K", "25"))
    elo_regress_to_mean: float = float(os.getenv("CFB_ELO_REGRESS", "0.20"))
    elo_init: float = float(os.getenv("CFB_ELO_INIT", "1500"))

settings = Settings()