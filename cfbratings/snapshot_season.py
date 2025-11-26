from cfbratings.io import fetch_teams, fetch_games, save_weekly_ratings
from cfbratings.models.hybrid import hybrid_rating
from cfbratings.config import settings

def snapshot_season(year: int, method: str = "hybrid"):
    teams = fetch_teams(year)
    games_payload = fetch_games(year, season_type="both")
    games = games_payload["data"]

    team_list = [t["school"] for t in teams]
    completed_games = [g for g in games if g.get("completed", False)]
    max_week = max((g.get("week", 0) for g in completed_games), default=0)

    if max_week == 0:
        print(f"No completed games found for year {year}. Skipping snapshot.")
        return

    for week in range(1, max_week + 1):
        # Only include games up to this week
        week_games = [g for g in games if g.get("week", 0) <= week and g.get("completed", False)]

        # Use hybrid method
        ratings = hybrid_rating(
            team_list,
            week_games,
            colley_weight=0.5,
            massey_weight=0.5,
            prior_strength=settings.colley_prior_strength,
            ridge_lambda=settings.massey_ridge_lambda,
            hfa=settings.home_field_adv,
        )

        # Save snapshot
        fname = f"ratings_{year}_{method}_week{week}.json"
        save_weekly_ratings(year, week, method, ratings)
        print(f"Cached weekly ratings → {fname}")