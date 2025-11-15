import argparse
from cfbratings.config import settings
from cfbratings.io import fetch_teams, fetch_games
from cfbratings.models.colley import build_colley, solve_colley
from cfbratings.models.massey import build_massey, solve_massey
from cfbratings.models.elo import run_elo
from cfbratings.models.hybrid import hybrid_rating
from cfbratings.analytics import records, strength_of_schedule, momentum, ppoints

def main():
    parser = argparse.ArgumentParser(description="CFB Ratings CLI")
    parser.add_argument("--year", type=int, default=settings.year)
    parser.add_argument("--season_type", type=str, default=settings.season_type)
    parser.add_argument("--method", type=str, default="hybrid", choices=["colley", "massey", "elo", "hybrid"])
    parser.add_argument("--refresh", action="store_true", help="Force refresh cache")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    teams = fetch_teams(args.year, force_refresh=args.refresh)
    games_payload = fetch_games(args.year, season_type=args.season_type, force_refresh=args.refresh)
    games = games_payload["data"] if isinstance(games_payload, dict) and "data" in games_payload else games_payload

    team_list = [t["school"] for t in teams]

    if args.method == "colley":
        C, b = build_colley(team_list, games, prior_strength=settings.colley_prior_strength)
        r = solve_colley(C, b)
        ratings = {team_list[i]: float(r[i]) for i in range(len(team_list))}
    elif args.method == "massey":
        M, mb = build_massey(team_list, games, ridge_lambda=settings.massey_ridge_lambda, hfa=settings.home_field_adv)
        r = solve_massey(M, mb)
        ratings = {team_list[i]: float(r[i]) for i in range(len(team_list))}
    elif args.method == "elo":
        ratings = run_elo(team_list, games, init=settings.elo_init, k=settings.elo_k,
                          regress_to_mean=settings.elo_regress_to_mean, hfa=settings.home_field_adv)
    else:
        ratings = hybrid_rating(team_list, games,
                                colley_weight=0.5, massey_weight=0.5,
                                prior_strength=settings.colley_prior_strength,
                                ridge_lambda=settings.massey_ridge_lambda,
                                hfa=settings.home_field_adv)

    recs = records(team_list, games)
    sos = strength_of_schedule(team_list, games, ratings)
    mom = momentum(team_list, games, ratings)
    pp = ppoints(team_list, games, ratings)

    top_items = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:args.top]
    print(f"\n{args.year} FBS — {args.method.capitalize()} Ratings\n")
    print(f"{'Rank':<4} {'Team':<28} {'Rating':<8} {'Record':<8} {'SOS':<8} {'Momentum':<8} {'PPoints':<8}")
    print("-" * 84)
    for i, (team, val) in enumerate(top_items, start=1):
        rw, rl = recs[team]
        print(f"{i:<4} {team:<28} {val:>.4f}   {rw}-{rl:<5} {sos[team]:>.4f}   {mom[team]:>.3f}   {pp[team]:>.2f}")

if __name__ == "__main__":
    main()