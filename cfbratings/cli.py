"""Command-line interface.

cfbratings rank --year 2025 --method hybrid --top 25
cfbratings backtest --methods massey,elo,hybrid --start-week 4
cfbratings tune --method elo --grid "k=15,25,40;hfa=2,2.5,3"
cfbratings predict "Ohio State" "Michigan" --neutral
cfbratings --year 2025 --warmup-years 2 rank      # warm-started with 2023-24 priors
cfbratings --years 2022-2025 backtest             # multi-season walk-forward
cfbratings --demo rank                            # synthetic season, no API key needed
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .analytics import conference_table, schedule_table, team_table
from .backtest import compare_methods, grid_search
from .config import Settings
from .config import settings as default_settings
from .data import CFBDClient, CFBDError, Game, JsonCache, Team, load_seasons, synthetic_seasons
from .data.schema import conference_map as _conference_map
from .data.schema import is_pool_team
from .models import MODEL_LABELS, MODEL_NAMES, make_model
from .ppoints import ppoints as _ppoints
from .predict import MarginCalibrator
from .priors import WarmStart, warm_start
from .projections import projection_table
from .snapshots import trajectory, weekly_snapshots

log = logging.getLogger("cfbratings")

SORT_COLUMNS = ("rating", "sos", "momentum", "ppoints", "wins")


# --- helpers ---------------------------------------------------------------------------


def parse_params(items: Sequence[str] | None) -> dict[str, Any]:
    """``["k=30", "hfa=2.5", "mov=false"]`` → ``{"k": 30.0, "hfa": 2.5, "mov": False}``."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--param expects key=value, got {item!r}")
        out[key.strip()] = _coerce(value.strip())
    return out


def parse_grid(spec: str) -> dict[str, list[Any]]:
    """``"k=15,25,40;hfa=2,2.5"`` → ``{"k": [15, 25, 40], "hfa": [2.0, 2.5]}``."""
    grid: dict[str, list[Any]] = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        key, sep, values = part.partition("=")
        if not sep:
            raise SystemExit(f"--grid entries look like key=v1,v2,...; got {part!r}")
        grid[key.strip()] = [_coerce(v.strip()) for v in values.split(",") if v.strip()]
    if not grid:
        raise SystemExit("--grid is empty")
    return grid


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _settings(args: argparse.Namespace) -> Settings:
    s = default_settings
    overrides: dict[str, Any] = {}
    if args.cache_dir:
        overrides["cache_dir"] = Path(args.cache_dir)
    if getattr(args, "fcs", None):
        overrides["fcs_handling"] = args.fcs
    return s.with_overrides(**overrides) if overrides else s


def parse_years(spec: str) -> list[int]:
    """``"2022-2024"`` or ``"2021,2023"`` → sorted list of seasons."""
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if sep:
            a, b = int(lo), int(hi)
            years.update(range(min(a, b), max(a, b) + 1))
        else:
            years.add(int(part))
    if not years:
        raise SystemExit("--years needs at least one season, e.g. 2022-2024")
    return sorted(years)


def requested_years(args: argparse.Namespace) -> list[int]:
    if getattr(args, "years", None):
        return parse_years(args.years)
    warmup = max(int(getattr(args, "warmup_years", 0) or 0), 0)
    return list(range(args.year - warmup, args.year + 1))


def load_data(args: argparse.Namespace, s: Settings) -> tuple[list[Team], list[Game], str]:
    """Return ``(teams, games, source_label)`` from CFBD or the synthetic generator.

    ``games`` may cover several seasons (``--years`` or ``--warmup-years``); commands that
    describe one season use the latest and warm-start it from the earlier ones.
    """
    years = requested_years(args)
    span = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
    if args.demo:
        chain = synthetic_seasons(years, seed=args.seed, played_through_last=args.demo_weeks or None)
        games = [g for season in chain for g in season.games]
        return chain[-1].teams, games, f"synthetic season{'s' if len(years) > 1 else ''} {span} (seed {args.seed})"
    client = CFBDClient(s)
    teams, games = load_seasons(years, args.season_type, force_refresh=args.refresh, client=client)
    fcs = "FCS pooled" if s.fcs_handling == "pool" else "FCS dropped"
    return teams, games, f"CFBD {span} ({args.season_type}, {fcs})"


def season_view(
    s: Settings, teams: list[Team], games: list[Game], method: str, params: dict[str, Any]
) -> tuple[int, list[Game], WarmStart, str]:
    """The latest season's games plus what it inherits from earlier seasons."""
    names = [t.school for t in teams]
    season = max((g.season for g in games), default=0)
    ws = warm_start(method, games, names, season, settings=s, **params)
    note = (
        f"warm-started from {len(ws.seasons)} prior season{'s' if len(ws.seasons) > 1 else ''}" if ws else "cold start"
    )
    return season, [g for g in games if g.season == season], ws, note


def _print_df(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> None:
    if df.empty:
        print("(no rows)")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.max_rows", 500):
        print(df.to_string(index=False, float_format=floatfmt.format))


def _export(df: pd.DataFrame, csv: str | None, as_json: str | None) -> None:
    if csv:
        Path(csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        print(f"wrote {csv}")
    if as_json:
        Path(as_json).parent.mkdir(parents=True, exist_ok=True)
        Path(as_json).write_text(df.to_json(orient="records", indent=2))
        print(f"wrote {as_json}")


# --- commands --------------------------------------------------------------------------


def cmd_rank(args: argparse.Namespace) -> int:
    s = _settings(args)
    teams, games, source = load_data(args, s)
    names = [t.school for t in teams]
    conf_map = _conference_map(teams)
    params = parse_params(args.param)
    season, games, ws, note = season_view(s, teams, games, args.method, params)
    model = make_model(args.method, s, **params)
    result = model.fit(names, games, prior=ws.prior)
    cal = MarginCalibrator.fit_cv(
        model, names, games, fallback_hfa=s.home_field_adv, prior=ws.prior, fallback=ws.calibrator
    )

    pp = None
    if not args.no_ppoints:
        snaps = weekly_snapshots(
            args.method, names, games, settings=s, use_cache=not args.demo, prior=ws.prior, **params
        )
        pp = _ppoints(names, games, result.ratings, conf_map, snaps)

    table = team_table(names, games, result, cal, conf_map, ppoints=pp)
    if args.conference:
        table = table[table["conference"].str.lower() == args.conference.lower()]
    sort_col = args.sort_by if args.sort_by in table.columns else "rating"
    table = table.sort_values(sort_col, ascending=False)
    shown = table.head(args.top)

    label = MODEL_LABELS.get(args.method, args.method)
    print(f"\n{label} ratings — {season} — {source} — {note}")
    print(
        f"calibration: HFA {cal.intercept:+.2f} pts, slope {cal.slope:.3f}, "
        f"sigma {cal.sigma:.1f} pts ({cal.source}, n={cal.n_games})\n"
    )
    cols = ["rank", "team", "conference", "record", "rating", "sos", "sos_rank", "momentum"]
    if pp is not None:
        cols.append("ppoints")
    _print_df(shown[cols])
    _export(table, args.csv, args.json)  # exports keep every team, not just --top
    return 0


def cmd_conferences(args: argparse.Namespace) -> int:
    s = _settings(args)
    teams, games, source = load_data(args, s)
    names = [t.school for t in teams]
    conf_map = _conference_map(teams)
    params = parse_params(args.param)
    season, games, ws, note = season_view(s, teams, games, args.method, params)
    result = make_model(args.method, s, **params).fit(names, games, prior=ws.prior)
    from .analytics import records  # local import keeps the module namespace tidy

    df = conference_table(result.ratings, conf_map, records(names, games))
    df = df[df["conference"] != "FCS"]
    print(f"\nConference strength — {MODEL_LABELS.get(args.method, args.method)} — {season} — {source} — {note}\n")
    _print_df(df)
    _export(df, args.csv, args.json)
    return 0


def cmd_team(args: argparse.Namespace) -> int:
    s = _settings(args)
    teams, games, source = load_data(args, s)
    names = [t.school for t in teams]
    if args.team not in names or is_pool_team(args.team):
        close = [n for n in names if args.team.lower() in n.lower() and not is_pool_team(n)]
        hint = f" Did you mean: {', '.join(close[:5])}?" if close else ""
        raise SystemExit(f"unknown team {args.team!r}.{hint}")
    params = parse_params(args.param)
    season, games, ws, note = season_view(s, teams, games, args.method, params)
    model = make_model(args.method, s, **params)
    result = model.fit(names, games, prior=ws.prior)
    cal = MarginCalibrator.fit_cv(
        model, names, games, fallback_hfa=s.home_field_adv, prior=ws.prior, fallback=ws.calibrator
    )
    rank = result.rank_map()[args.team]
    print(f"\n{args.team} — #{rank} ({MODEL_LABELS.get(args.method, args.method)}, {season}, {source}, {note})\n")
    _print_df(schedule_table(args.team, games, result, cal))
    snaps = weekly_snapshots(args.method, names, games, settings=s, use_cache=not args.demo, prior=ws.prior, **params)
    traj = trajectory(snaps, args.team)
    if traj:
        print("\nWeekly rank: " + " → ".join(f"W{w}:#{r}" for w, r, _ in traj))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    s = _settings(args)
    teams, games, source = load_data(args, s)
    names = [t.school for t in teams]
    for team in (args.home, args.away):
        if team not in names or is_pool_team(team):
            raise SystemExit(f"unknown team {team!r}")
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else list(MODEL_NAMES)
    rows = []
    note = "cold start"
    for m in methods:
        _, season_games, ws, note = season_view(s, teams, games, m, {})
        model = make_model(m, s)
        result = model.fit(names, season_games, prior=ws.prior)
        cal = MarginCalibrator.fit_cv(
            model, names, season_games, fallback_hfa=s.home_field_adv, prior=ws.prior, fallback=ws.calibrator
        )
        pred = cal.predict_game(result, args.home, args.away, args.neutral)
        rows.append(
            {
                "method": MODEL_LABELS.get(m, m),
                "line": pred.spread,
                "home_margin": pred.margin,
                "p_home": pred.p_home,
                "p_away": 1 - pred.p_home,
            }
        )
    site = "neutral site" if args.neutral else f"at {args.home}"
    print(f"\n{args.away} @ {args.home} ({site}) — {source} — {note}\n")
    _print_df(pd.DataFrame(rows))
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    s = _settings(args)
    teams, games, source = load_data(args, s)
    names = [t.school for t in teams]
    conf_map = _conference_map(teams)
    params = parse_params(args.param)
    season, games, ws, note = season_view(s, teams, games, args.method, params)
    model = make_model(args.method, s, **params)
    result = model.fit(names, games, prior=ws.prior)
    cal = MarginCalibrator.fit_cv(
        model, names, games, fallback_hfa=s.home_field_adv, prior=ws.prior, fallback=ws.calibrator
    )
    df = projection_table(names, games, result, cal, conference_map=conf_map)
    n_remaining = sum(1 for g in games if not g.has_score)
    print(f"\nRest-of-season projections — {MODEL_LABELS.get(args.method, args.method)} — {season} — {source} — {note}")
    print(f"{n_remaining} unplayed games on the schedule\n")
    if args.conference:
        df = df[df["conference"].str.lower() == args.conference.lower()]
    cols = [
        "rank",
        "team",
        "record",
        "remaining",
        "proj_record",
        "p_6_wins",
        "p_10_wins",
        "p_win_out",
        "next_opponent",
        "next_win_prob",
    ]
    _print_df(df[cols].head(args.top))
    _export(df, args.csv, args.json)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    s = _settings(args)
    teams, games, source = load_data(args, s)
    names = [t.school for t in teams]
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else list(MODEL_NAMES)
    lines = None
    if args.lines and not args.demo:
        client = CFBDClient(s)
        lines = {}
        for year in requested_years(args):
            lines.update(client.lines(year, args.season_type, force_refresh=args.refresh))
    params = {m: parse_params(args.param) for m in methods}
    table, results = compare_methods(names, games, methods, start_week=args.start_week, params=params, lines=lines)
    seasons = sorted({g.season for g in games})
    warm = " (later seasons warm-started from the previous one)" if len(seasons) > 1 else ""
    print(f"\nWalk-forward backtest — {source} — evaluation from week {args.start_week}{warm}\n")
    _print_df(table, "{:.4f}")
    if len(seasons) > 1:
        for m, res in results.items():
            print(f"\n{MODEL_LABELS.get(m, m)} by season")
            _print_df(res.by_season(), "{:.4f}")
    if args.by_week:
        for m, res in results.items():
            print(f"\n{MODEL_LABELS.get(m, m)} by week")
            _print_df(res.by_week(), "{:.3f}")
    if args.calibration:
        for m, res in results.items():
            print(f"\n{MODEL_LABELS.get(m, m)} calibration")
            _print_df(res.calibration(), "{:.3f}")
    _export(table, args.csv, args.json)
    if args.predictions_csv and results:
        frames = [r.predictions.assign(method=m) for m, r in results.items()]
        pd.concat(frames).to_csv(args.predictions_csv, index=False)
        print(f"wrote {args.predictions_csv}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    s = _settings(args)
    teams, games, source = load_data(args, s)
    names = [t.school for t in teams]
    grid = parse_grid(args.grid)
    total = 1
    for v in grid.values():
        total *= len(v)
    print(f"\nGrid search — {MODEL_LABELS.get(args.method, args.method)} — {source} — {total} combinations\n")

    def progress(i: int, n: int, row: dict[str, Any]) -> None:
        if args.verbose:
            print(f"[{i}/{n}] {row}", file=sys.stderr)

    df = grid_search(names, games, args.method, grid, metric=args.metric, start_week=args.start_week, progress=progress)
    _print_df(df, "{:.4f}")
    if not df.empty:
        best = df.iloc[0]
        print(f"\nbest by {args.metric}: " + ", ".join(f"{k}={best[k]}" for k in grid))
    _export(df, args.csv, args.json)
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    s = _settings(args)
    teams, games, _ = load_data(args, s)
    names = [t.school for t in teams]
    params = parse_params(args.param)
    _, games, ws, _ = season_view(s, teams, games, args.method, params)
    snaps = weekly_snapshots(args.method, names, games, settings=s, use_cache=not args.demo, prior=ws.prior, **params)
    print(
        f"computed {len(snaps)} weekly snapshots for {args.method} (weeks {min(snaps, default=0)}-{max(snaps, default=0)})"
    )
    if args.team:
        for w, r, v in trajectory(snaps, args.team):
            print(f"week {w:>2}: #{r:<3} {v:+.3f}")
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    s = _settings(args)
    cache = JsonCache(s.cache_dir, archive_dir=s.archive_dir)
    if args.action == "clear":
        n = cache.clear()
        print(f"removed {n} cached files from {s.cache_dir} (archive in {s.archive_dir} kept)")
    else:
        files = sorted(cache.directory.glob("*.json")) if cache.directory.exists() else []
        print(f"{len(files)} cached files in {s.cache_dir}")
        for p in files:
            print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
        archived = sorted(s.archive_dir.glob("*.json.gz")) if s.archive_dir.exists() else []
        print(f"{len(archived)} archived (committed) payloads in {s.archive_dir}")
        for p in archived:
            print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


# --- parser ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfbratings", description="College football ratings, analytics and backtesting"
    )
    parser.add_argument("--version", action="version", version=f"cfbratings {__version__}")
    parser.add_argument("--year", type=int, default=default_settings.year, help="season (default: current)")
    parser.add_argument(
        "--season-type", default=default_settings.season_type, choices=["regular", "postseason", "both"]
    )
    parser.add_argument("--years", help="several seasons, e.g. 2022-2024 or 2021,2023 (overrides --year)")
    parser.add_argument(
        "--warmup-years", type=int, default=0, metavar="N", help="also load N earlier seasons to warm-start ratings"
    )
    parser.add_argument(
        "--fcs",
        choices=["pool", "drop"],
        help="games vs non-FBS opponents: pool into one pseudo-team (default) or drop",
    )
    parser.add_argument(
        "--demo", action="store_true", help="use a synthetic season instead of CFBD (no API key needed)"
    )
    parser.add_argument("--seed", type=int, default=7, help="seed for --demo data")
    parser.add_argument(
        "--demo-weeks", type=int, default=0, metavar="N", help="with --demo: only the first N weeks are played"
    )
    parser.add_argument("--refresh", action="store_true", help="bypass the JSON cache")
    parser.add_argument("--cache-dir", help="override the cache directory")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    def add_model_args(p: argparse.ArgumentParser, default: str = "hybrid") -> None:
        p.add_argument("--method", default=default, choices=MODEL_NAMES)
        p.add_argument(
            "--param", action="append", metavar="KEY=VALUE", help="model hyper-parameter override (repeatable)"
        )

    def add_export_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--csv", help="write the table to CSV")
        p.add_argument("--json", help="write the table to JSON")

    p = sub.add_parser("rank", help="print the ratings table")
    add_model_args(p)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--sort-by", default="rating", choices=SORT_COLUMNS)
    p.add_argument("--conference", help="only show one conference")
    p.add_argument("--no-ppoints", action="store_true", help="skip PPoints (faster)")
    add_export_args(p)
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("conferences", help="conference strength table")
    add_model_args(p)
    add_export_args(p)
    p.set_defaults(func=cmd_conferences)

    p = sub.add_parser("team", help="one team's schedule, expectations and weekly rank")
    p.add_argument("team")
    add_model_args(p)
    p.set_defaults(func=cmd_team)

    p = sub.add_parser("project", help="rest-of-season record projections (exact win-total distributions)")
    add_model_args(p)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--conference", help="only show one conference")
    add_export_args(p)
    p.set_defaults(func=cmd_project)

    p = sub.add_parser("predict", help="spread and win probability for a matchup")
    p.add_argument("home")
    p.add_argument("away")
    p.add_argument("--neutral", action="store_true")
    p.add_argument("--methods", help="comma-separated subset of methods (default: all)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("backtest", help="walk-forward evaluation of one or more methods")
    p.add_argument("--methods", help="comma-separated methods (default: all)")
    p.add_argument(
        "--param", action="append", metavar="KEY=VALUE", help="hyper-parameter override applied to every method"
    )
    p.add_argument("--start-week", type=int, default=4)
    p.add_argument("--by-week", action="store_true", help="also print per-week metrics")
    p.add_argument("--calibration", action="store_true", help="also print reliability tables")
    p.add_argument("--lines", action="store_true", help="fetch closing spreads from CFBD for a Vegas comparison")
    p.add_argument("--predictions-csv", help="dump every out-of-sample prediction")
    add_export_args(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("tune", help="grid-search hyper-parameters with walk-forward backtests")
    p.add_argument("--method", default="elo", choices=MODEL_NAMES)
    p.add_argument("--grid", required=True, help='e.g. "k=15,25,40;hfa=2,2.5,3"')
    p.add_argument("--metric", default="log_loss", choices=["log_loss", "brier", "accuracy", "mae", "rmse"])
    p.add_argument("--start-week", type=int, default=4)
    add_export_args(p)
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("snapshot", help="compute and cache weekly rating snapshots")
    add_model_args(p)
    p.add_argument("--team", help="print this team's weekly trajectory")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("cache", help="inspect or clear the JSON cache")
    p.add_argument("action", choices=["list", "clear"])
    p.set_defaults(func=cmd_cache)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        return int(args.func(args))
    except CFBDError as exc:  # missing key, 401, 429 quota: no traceback, just the reason
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:  # e.g. piped into `head`
        return 0
    finally:
        if CFBDClient.api_calls:  # the free tier has a small monthly budget: make every call visible
            print(f"[cfbd] {CFBDClient.api_calls} API call(s) made this run", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
