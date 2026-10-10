"""Parsers against the row shapes documented by the CFBD OpenAPI spec (``tests/fixtures/cfbd_v5_shapes.json``)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from cfbratings.cli import main
from cfbratings.config import Settings
from cfbratings.data import CFBDClient, CFBDQuotaError
from cfbratings.data.cfbd import SEASON_TYPES, consensus_spreads, spread_sign_consistent
from cfbratings.data.schema import FCS_POOL, Team, normalize_games, normalize_teams, pool_non_fbs

FIXTURE = Path(__file__).parent / "fixtures" / "cfbd_v5_shapes.json"

GAME_REQUIRED = {
    "id", "season", "week", "seasonType", "startDate", "startTimeTBD", "completed", "neutralSite", "conferenceGame",
    "attendance", "venueId", "venue", "homeId", "homeTeam", "homeConference", "homeClassification", "homePoints",
    "homeLineScores", "homePostgameWinProbability", "homePregameElo", "homePostgameElo", "awayId", "awayTeam",
    "awayConference", "awayClassification", "awayPoints", "awayLineScores", "awayPostgameWinProbability",
    "awayPregameElo", "awayPostgameElo", "excitementIndex", "highlights", "notes", "playoff",
}  # fmt: skip
LINE_REQUIRED = {
    "provider",
    "spread",
    "formattedSpread",
    "spreadOpen",
    "overUnder",
    "overUnderOpen",
    "homeMoneyline",
    "awayMoneyline",
}


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(FIXTURE.read_text())


def test_fixture_rows_carry_every_required_spec_key(payload):
    for row in payload["games"]:
        assert set(row) >= GAME_REQUIRED
    for row in payload["lines"]:
        for line in row["lines"]:
            assert set(line) >= LINE_REQUIRED
    assert set(SEASON_TYPES) <= set(payload["season_type_enum"])  # ``both`` is a documented value for /games and /lines


def test_games_parse_from_camel_case_rows(payload):
    games = normalize_games(payload["games"])
    by_id = {g.id: g for g in games}
    rivalry = by_id[401635000]
    assert (rivalry.home, rivalry.away, rivalry.margin) == ("Oregon", "Ohio State", 1)
    assert rivalry.completed and not rivalry.neutral_site and rivalry.is_conference_game
    assert rivalry.start_date.startswith("2024-10-12")
    final = by_id[401635002]
    assert final.season_type == "postseason" and final.raw_week == 1 and final.neutral_site
    assert final.week == 15  # postseason week 1 follows the last regular week (14) in the fixture
    assert final.winner == "Ohio State" and final.margin == -11
    unplayed = by_id[401635004]
    assert not unplayed.has_score and unplayed.completed is False
    assert [g.id for g in games][-1] == 401635002  # chronological: the title game sorts last


def test_fbs_pooling_matches_classification_fields(payload):
    teams = normalize_teams(payload["teams"])
    fbs = {t.school for t in teams}
    for row in payload["games"]:  # the roster-based test and the per-row classification must agree
        assert (row["homeClassification"] == "fbs") == (row["homeTeam"] in fbs)
        assert (row["awayClassification"] == "fbs") == (row["awayTeam"] in fbs)
    pooled = pool_non_fbs(normalize_games(payload["games"]), fbs)
    ids = {g.id: g for g in pooled}
    assert 401635003 not in ids  # FCS vs FCS: dropped
    vs_fcs = ids[401635001]
    assert vs_fcs.away == FCS_POOL and vs_fcs.away_name == "Idaho" and vs_fcs.away_conference == "FCS"
    assert ids[401635000].home_name is None  # FBS vs FBS untouched
    assert sum(1 for g in pooled if FCS_POOL in (g.home, g.away)) == 1


def test_team_rows_parse(payload):
    teams = normalize_teams(payload["teams"])
    assert teams[0] == Team(school="Ohio State", conference="Big Ten", classification="fbs", abbreviation="OSU")
    assert [t.school for t in teams] == ["Ohio State", "Notre Dame", "Oregon"]


def test_spread_sign_is_home_relative(payload):
    spreads = consensus_spreads(payload["lines"])
    assert spreads == {401635000: 3.25, 401635002: 8.5, 401635001: -31.5}
    # The spec documents ``spread`` as "home-relative; negative favors home", so the backtest's
    # expected home margin is -spread: Ohio State favoured by 3.25 at Oregon, Oregon by 31.5 vs Idaho.
    assert -spreads[401635000] < 0 and -spreads[401635001] > 0
    assert spreads[401635002] == 8.5  # the ``consensus`` provider wins over the mean when present
    for row in payload["lines"]:
        for line in row["lines"]:
            if line["spread"] is not None:
                assert spread_sign_consistent(line["spread"], line["formattedSpread"], row["homeTeam"], row["awayTeam"])


def test_spread_sign_check_parses_formatted_spread():
    assert spread_sign_consistent(-14.5, "Ohio State -14.5", "Ohio State", "Michigan")
    assert spread_sign_consistent(14.5, "Ohio State -14.5", "Michigan", "Ohio State")
    assert not spread_sign_consistent(14.5, "Ohio State -14.5", "Ohio State", "Michigan")
    assert spread_sign_consistent(-3.5, "Miami (OH) -3.5", "Miami (OH)", "Ohio")  # names with spaces / brackets
    assert spread_sign_consistent(0.0, "Ohio State 0", "Ohio State", "Michigan")  # pick'em: nothing to check
    assert spread_sign_consistent(-3.0, "EVEN", "A", "B") and spread_sign_consistent(-3.0, None, "A", "B")
    assert spread_sign_consistent(-3.0, "Somebody Else -3", "A", "B")  # unknown team: not our problem


def test_consensus_spreads_warns_on_sign_mismatch(caplog):
    raw = [
        {
            "id": 1,
            "homeTeam": "A",
            "awayTeam": "B",
            "lines": [{"provider": "X", "spread": 7, "formattedSpread": "A -7"}],
        }
    ]
    with caplog.at_level(logging.WARNING, logger="cfbratings.data.cfbd"):
        assert consensus_spreads(raw) == {1: 7.0}  # the value is still returned ...
    assert "contradicts" in caplog.text  # ... but loudly


class _Response:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def test_client_counts_calls_and_stops_on_429(tmp_path, monkeypatch):
    monkeypatch.setattr(CFBDClient, "api_calls", 0)
    s = Settings(api_key="k", cache_dir=tmp_path / "cache", archive_dir=tmp_path / "archive")
    client = CFBDClient(s, session=_Session([_Response(200, [{"school": "A"}]), _Response(429)]))
    assert [t.school for t in client.teams(2024)] == ["A"]
    assert CFBDClient.api_calls == 1 and client.calls == ["/teams/fbs {'year': 2024}"]
    with pytest.raises(CFBDQuotaError, match="429"):
        client.games(2024)
    assert CFBDClient.api_calls == 2 and client.cache.get("games_2024_both") is None  # nothing cached, no retry
    assert [t.school for t in client.teams(2024)] == ["A"] and CFBDClient.api_calls == 2  # served from cache


def test_cli_reports_call_count_and_cfbd_errors(tmp_path, monkeypatch, capsys):
    offline = Settings(api_key="", cache_dir=tmp_path / "cache", archive_dir=tmp_path / "archive")
    monkeypatch.setattr("cfbratings.cli.default_settings", offline)
    monkeypatch.setattr(CFBDClient, "api_calls", 0)
    assert main(["--year", "2024", "rank", "--no-ppoints"]) == 1  # no key: clean error, no traceback
    err = capsys.readouterr().err
    assert "error: CFBD_API_KEY is not set" in err and "[cfbd]" not in err
    monkeypatch.setattr(CFBDClient, "api_calls", 3)
    assert main(["--demo", "rank", "--top", "3", "--no-ppoints"]) == 0
    assert "[cfbd] 3 API call(s) made this run" in capsys.readouterr().err
