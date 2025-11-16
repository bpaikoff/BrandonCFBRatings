import json
import os
import time
from typing import Any, Dict, List
import requests
from .config import settings
from .models.hybrid import hybrid_rating


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _cache_path(kind: str, year: int, season_type: str = "both") -> str:
    _ensure_dir(settings.cache_dir)
    fname = f"{kind}_{year}{'' if kind=='teams' else f'_{season_type}'}.json"
    return os.path.join(settings.cache_dir, fname)

def _read_cache(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_cache(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def fetch_teams(year: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
    path = _cache_path("teams", year)
    if not force_refresh:
        cached = _read_cache(path)
        if cached:
            return cached
    if not settings.api_key:
        raise RuntimeError("CFBD_API_KEY not set")
    url = f"{settings.base_url}/teams/fbs?year={year}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {settings.api_key}"}, timeout=settings.timeout)
    resp.raise_for_status()
    data = resp.json()
    _write_cache(path, data)
    return data

def fetch_games(year: int, season_type: str = "both", force_refresh: bool = False) -> List[Dict[str, Any]]:
    path = _cache_path("games", year, season_type)
    if not force_refresh:
        cached = _read_cache(path)
        if cached:
            return cached
    if not settings.api_key:
        raise RuntimeError("CFBD_API_KEY not set")
    url = f"{settings.base_url}/games?year={year}&seasonType={season_type}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {settings.api_key}"}, timeout=settings.timeout)
    resp.raise_for_status()
    data = resp.json()
    # Optional: keep a timestamp for staleness checks
    payload = {"_cached_at": int(time.time()), "data": data}
    _write_cache(path, payload)
    return payload

def save_weekly_ratings(year: int, week: int, method: str, ratings: dict):
    path = os.path.join(settings.cache_dir, f"ratings_{year}_{method}_week{week}.json")
    os.makedirs(settings.cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2)

def load_weekly_ratings(year: int, week: int, method: str) -> dict | None:
    path = os.path.join(settings.cache_dir, f"ratings_{year}_{method}_week{week}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_snapshots(year: int, method: str = "hybrid") -> None:
    teams = fetch_teams(year)
    games_payload = fetch_games(year, season_type="both")
    games = games_payload["data"]

    team_list = [t["school"] for t in teams]
    max_week = max(g.get("week", 0) for g in games if g.get("completed", False))

    for week in range(1, max_week + 1):
        fname = os.path.join(settings.cache_dir, f"ratings_{year}_{method}_week{week}.json")
        if os.path.exists(fname):
            continue  # already cached

        week_games = [g for g in games if g.get("week", 0) <= week and g.get("completed", False)]
        ratings = hybrid_rating(
            team_list,
            week_games,
            colley_weight=0.5,
            massey_weight=0.5,
            prior_strength=settings.colley_prior_strength,
            ridge_lambda=settings.massey_ridge_lambda,
            hfa=settings.home_field_adv,
        )
        save_weekly_ratings(year, week, method, ratings)
        print(f"Cached weekly ratings → {fname}")