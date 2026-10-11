"""Parsers, pooling and lines against a real, trimmed 2024 payload (``tests/fixtures/games_2024_sample.json``).

The sample was cut from the archived season with ``tests/fixtures/trim_payload.py``: every game of
Ohio State, Notre Dame, Oregon and Texas (49 regular-season rows plus the two conference title
games and all nine of their College Football Playoff games), the matching betting lines, and the
``/teams/fbs`` row of each FBS school involved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cfbratings.config import Settings
from cfbratings.data import CFBDClient, load_season
from cfbratings.data.cfbd import consensus_spreads, spread_sign_consistent
from cfbratings.data.schema import FCS_POOL, Team, normalize_games, normalize_teams, pool_non_fbs

FIXTURE = Path(__file__).parent / "fixtures" / "games_2024_sample.json"
PARSED_KEYS = {
    "id", "season", "week", "seasonType", "startDate", "completed", "neutralSite", "homeTeam", "awayTeam",
    "homePoints", "awayPoints", "homeConference", "awayConference", "homeClassification", "awayClassification",
}  # fmt: skip

TITLE_GAME = 401677192  # Ohio State 34, Notre Dame 23 (Atlanta, 2025-01-21)
COTTON_BOWL = 401677191  # CFP semifinal, neutral: Ohio State 28, Texas 14 with Texas listed as "home"
FIRST_ROUND = 401677177  # CFP first round on campus: Tennessee 17 at Ohio State 42
THE_GAME = 401628566  # Michigan 13 at Ohio State 10; Ohio State was a ~20-point home favourite
IDAHO_AT_OREGON = 401628456  # the only FCS opponent among these teams' games


@pytest.fixture(scope="module")
def sample() -> dict:
    return json.loads(FIXTURE.read_text())


def test_sample_is_a_real_api_payload(sample):
    assert sample["source"].startswith("CollegeFootballData")
    assert (len(sample["teams"]), len(sample["games"]), len(sample["lines"])) == (46, 58, 58)
    for row in sample["games"]:
        assert set(row) >= PARSED_KEYS and row["season"] == 2024
    assert {t["school"] for t in sample["teams"]} >= {"Ohio State", "Notre Dame", "Oregon", "Texas"}
    assert all(t["classification"] == "fbs" for t in sample["teams"])


def test_postseason_rows_are_week_one_and_get_renumbered(sample):
    games = normalize_games(sample["games"])
    assert all(g.has_score for g in games)
    post = [g for g in games if g.season_type == "postseason"]
    assert len(post) == 9 and {g.raw_week for g in post} == {1}  # CFBD: every bowl and CFP round is postseason week 1
    last_regular = max(g.week for g in games if g.season_type == "regular")
    assert last_regular == 15 and all(g.week == 16 for g in post)  # one effective week after the title games
    assert games[-1].id == TITLE_GAME and games[-1].winner == "Ohio State"
    by_id = {g.id: g for g in games}
    assert by_id[FIRST_ROUND].winner == "Ohio State" and not by_id[FIRST_ROUND].neutral_site
    assert by_id[COTTON_BOWL].neutral_site and by_id[COTTON_BOWL].margin == -14  # the listed away team won


def test_roster_pooling_agrees_with_classification_fields(sample):
    fbs = {t.school for t in normalize_teams(sample["teams"])}
    for row in sample["games"]:
        assert (row["homeClassification"] == "fbs") == (row["homeTeam"] in fbs)
        assert (row["awayClassification"] == "fbs") == (row["awayTeam"] in fbs)
    pooled = pool_non_fbs(normalize_games(sample["games"]), fbs)
    assert len(pooled) == 58  # nothing dropped: no FCS-vs-FCS rows among these teams' games
    idaho = next(g for g in pooled if g.id == IDAHO_AT_OREGON)
    assert (idaho.home, idaho.away, idaho.away_name, idaho.away_conference) == ("Oregon", FCS_POOL, "Idaho", "FCS")
    assert sum(1 for g in pooled if FCS_POOL in (g.home, g.away)) == 1


def test_records_match_the_2024_season(sample):
    fbs = {t.school for t in normalize_teams(sample["teams"])}
    pooled = pool_non_fbs(normalize_games(sample["games"]), fbs)

    def record(team: str) -> tuple[int, int]:
        played = [g for g in pooled if team in (g.home, g.away)]
        wins = sum(1 for g in played if g.winner == team)
        return wins, len(played) - wins

    assert record("Ohio State") == (14, 2) and record("Notre Dame") == (14, 2)
    assert record("Oregon") == (13, 1) and record("Texas") == (13, 3)


def test_load_season_serves_the_sample_through_the_client(sample, tmp_path):
    class Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class Session:
        def get(self, url, params=None, headers=None, timeout=None):
            return Response(sample["teams"] if url.endswith("/teams/fbs") else sample["games"])

    s = Settings(api_key="k", cache_dir=tmp_path / "cache", archive_dir=tmp_path / "archive")
    client = CFBDClient(s, session=Session())
    teams, games = load_season(2024, "both", client=client)
    assert len(games) == 58 and len(teams) == 47 and FCS_POOL in {t.school for t in teams}
    assert client.cache.get_archived("games_2024_both") is not None  # a finished season lands in the archive tier
    assert client.freshness(2024).tier == "cache" and client.freshness(2024).final
    pool = next(t for t in teams if t.school == FCS_POOL)
    assert (pool.color, pool.alt_color, pool.logo, pool.logo_dark) == (None, None, None, None)  # no brand to show
    assert all(t.color and t.logo for t in teams if t.school != FCS_POOL)  # real schools keep theirs


def test_closing_lines_follow_the_home_relative_sign_convention(sample):
    spreads = consensus_spreads(sample["lines"])
    assert len(spreads) == 57 and IDAHO_AT_OREGON not in spreads  # a row with no book is a line-less game, not an error
    assert spreads[THE_GAME] == -20.0  # home favourite: negative (mean of -20.5, -20 and -19.5)
    assert spreads[COTTON_BOWL] == pytest.approx(5.8333, abs=1e-3)  # favourite listed as the away team: positive
    rows = [
        (x["spread"], x["formattedSpread"], r["homeTeam"], r["awayTeam"]) for r in sample["lines"] for x in r["lines"]
    ]
    assert rows and all(spread_sign_consistent(*row) for row in rows)
    assert {x["provider"] for r in sample["lines"] for x in r["lines"]} == {"ESPN Bet", "DraftKings", "Bovada"}


def test_team_rows_carry_brand_colours_and_logos(sample):
    hex_color = re.compile(r"#[0-9a-f]{6}")
    rows = {row["school"]: row for row in sample["teams"]}
    teams = normalize_teams(sample["teams"])
    for t in teams:
        assert hex_color.fullmatch(t.color) and hex_color.fullmatch(t.alt_color)
        assert t.logo == f"https://cdn.collegefootballdata.com/logos/500/{rows[t.school]['id']}.png"
        assert t.logo_dark == f"https://cdn.collegefootballdata.com/logos-dark/500/{rows[t.school]['id']}.png"
        assert Team.from_dict(t.to_dict()) == t
    osu = next(t for t in teams if t.school == "Ohio State")
    assert (osu.color, osu.alt_color) == ("#ba0c2f", "#a7b1b7")
    assert osu.logo.endswith("/logos/500/194.png") and osu.logo_dark.endswith("/logos-dark/500/194.png")
    assert len({t.color for t in teams}) > 1
