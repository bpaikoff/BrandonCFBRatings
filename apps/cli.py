import argparse
from cfbratings.config import settings
from cfbratings.io import fetch_teams, fetch_games, ensure_snapshots
from cfbratings.models.colley import build_colley, solve_colley
from cfbratings.models.massey import build_massey, solve_massey
from cfbratings.models.elo import run_elo
from cfbratings.models.hybrid import hybrid_rating
from cfbratings.analytics import records, strength_of_schedule, momentum, ppoints, compute_conference_strength_robust


def main():
    parser = argparse.ArgumentParser(description="CFB Ratings CLI")
    parser.add_argument("--year", type=int, default=settings.year)
    parser.add_argument("--season_type", type=str, default=settings.season_type)
    parser.add_argument("--method", type=str, default="hybrid", choices=["colley", "massey", "elo", "hybrid"])
    parser.add_argument("--refresh", action="store_true", help="Force refresh cache")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--sort-by", type=str, default="rating",
                        choices=["rating", "sos", "momentum", "ppoints"],
                        help="Column to sort by (default: rating)")
    args = parser.parse_args()

    teams = fetch_teams(args.year, force_refresh=args.refresh)
    conference_map = {t["school"]: t.get("conference") for t in teams}
    games_payload = fetch_games(args.year, season_type=args.season_type, force_refresh=args.refresh)
    games = games_payload["data"] if isinstance(games_payload, dict) and "data" in games_payload else games_payload
    ensure_snapshots(args.year, method=args.method)

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
    pp = ppoints(team_list, games, ratings, conference_map)

    latest_week = max(g.get("week", 0) for g in games if g.get("completed", False))

    conference_map = {t["school"]: t.get("conference") for t in teams}
    conf_strength = compute_conference_strength_robust(ratings, conference_map)

    print(f"\nConference Strength ({args.year}, {args.method})")
    print(f"{'Conference':<20} {'Strength':<8}")
    print("-" * 30)
    for conf, val in sorted(conf_strength.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{conf:<20} {val:.2f}")
    print()

    # Build combined table
    rows = []
    for team, val in ratings.items():
        rw, rl = recs[team]
        rows.append({
            "team": team,
            "rating": val,
            "record": f"{rw}-{rl}",
            "sos": sos[team],
            "momentum": mom[team],
            "ppoints": pp[team],
        })

    # Sort by chosen column
    rows_sorted = sorted(rows, key=lambda r: r[args.sort_by], reverse=True)
    top_items = rows_sorted[:args.top]

    print(f"\n{args.year} FBS — {args.method.capitalize()} Ratings (sorted by {args.sort_by})\n")
    print(f"{'Rank':<4} {'Team':<28} {'Rating':<8} {'Record':<8} {'SOS':<8} {'Momentum':<8} {'PPoints':<8}")
    print("-" * 92)
    for i, row in enumerate(top_items, start=1):
        print(f"{i:<4} {row['team']:<28} {row['rating']:.4f}   {row['record']:<8} "
              f"{row['sos']:.4f}   {row['momentum']:.3f}   {row['ppoints']:.2f}")

if __name__ == "__main__":
    main()