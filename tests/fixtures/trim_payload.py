"""Trim archived CFBD payloads into a small fixture (offline; reads data/archive only).

    python tests/fixtures/trim_payload.py --year 2024 --teams "Ohio State" "Notre Dame" "Oregon"

Writes ``tests/fixtures/games_<year>_sample.json`` with the selected teams' games and betting
lines from ``games_<year>_both`` / ``lines_<year>_both`` and the ``teams_<year>`` rows of every
FBS school involved, exactly as the API returned them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfbratings.data.cache import JsonCache

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--teams", nargs="+", required=True, help="schools whose games to keep")
    ap.add_argument("--archive-dir", default=str(REPO / "data" / "archive"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache = JsonCache(REPO / "data" / "cache", archive_dir=args.archive_dir)
    teams = cache.get(f"teams_{args.year}", ignore_ttl=True)
    games = cache.get(f"games_{args.year}_both", ignore_ttl=True)
    lines = cache.get(f"lines_{args.year}_both", ignore_ttl=True) or []
    if teams is None or games is None:
        raise SystemExit(f"season {args.year} is not archived; fetch it once with the CLI first")

    keep = set(args.teams)
    sample_games = [g for g in games if g.get("homeTeam") in keep or g.get("awayTeam") in keep]
    ids = {g["id"] for g in sample_games}
    # Every FBS school that appears in the kept games gets its roster row, so that the roster-based
    # FCS pooling in ``load_season`` behaves on the sample exactly as on the full payload.
    schools = {g[side] for g in sample_games for side in ("homeTeam", "awayTeam")}
    out = {
        "source": f"CollegeFootballData /teams/fbs, /games?seasonType=both, /lines?seasonType=both for {args.year}",
        "teams": [t for t in teams if t.get("school") in schools],
        "games": sample_games,
        "lines": [row for row in lines if row.get("id") in ids],
    }
    dest = Path(args.out) if args.out else HERE / f"games_{args.year}_sample.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}: {len(out['teams'])} teams, {len(out['games'])} games, {len(out['lines'])} line rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
