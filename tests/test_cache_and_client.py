from __future__ import annotations

import json

import pytest

from cfbratings.config import Settings
from cfbratings.data.cache import JsonCache
from cfbratings.data.cfbd import CFBDClient, CFBDError, consensus_spreads, load_season, load_seasons
from cfbratings.data.schema import FCS_POOL


def test_cache_round_trip_and_ttl(tmp_path, monkeypatch):
    cache = JsonCache(tmp_path, ttl_hours=1.0)
    assert cache.get("missing") is None
    cache.put("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    now = 1_000_000.0
    monkeypatch.setattr("cfbratings.data.cache.time.time", lambda: now)
    cache.put("k", {"a": 2})
    monkeypatch.setattr("cfbratings.data.cache.time.time", lambda: now + 7200)
    assert cache.get("k") is None  # stale
    assert cache.get("k", ignore_ttl=True) == {"a": 2}
    cache.put("final", [1, 2], final=True)
    monkeypatch.setattr("cfbratings.data.cache.time.time", lambda: now + 10**7)
    assert cache.get("final") == [1, 2]


def test_cache_ignores_corrupt_files(tmp_path):
    cache = JsonCache(tmp_path)
    cache.path("bad").parent.mkdir(parents=True, exist_ok=True)
    cache.path("bad").write_text("{not json")
    assert cache.get("bad") is None
    cache.path("legacy").write_text(json.dumps([1, 2, 3]))  # old unwrapped format
    assert cache.get("legacy") is None
    assert cache.clear() == 2


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    """Routes keyed by path, or by ``(path, year)`` when a payload depends on the season."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        params = dict(params or {})
        self.calls.append((url, params))
        assert headers["Authorization"].startswith("Bearer ")
        path = url.split("collegefootballdata.com")[-1]
        key = (path, params.get("year"))
        return _FakeResponse(self.routes[key] if key in self.routes else self.routes[path])


def test_client_requires_api_key(tmp_path):
    client = CFBDClient(Settings(api_key="", cache_dir=tmp_path))
    with pytest.raises(CFBDError):
        client.teams(2024)


def test_client_caches_and_filters(tmp_path):
    routes = {
        "/teams/fbs": [{"school": "A", "conference": "C1"}, {"school": "B", "conference": "C1"}],
        "/games": [
            {
                "id": 1,
                "season": 2024,
                "week": 1,
                "homeTeam": "A",
                "awayTeam": "B",
                "homePoints": 20,
                "awayPoints": 10,
                "completed": True,
            },
            {
                "id": 2,
                "season": 2024,
                "week": 1,
                "homeTeam": "A",
                "awayTeam": "FCS U",
                "homePoints": 50,
                "awayPoints": 0,
                "completed": True,
            },
        ],
        "/lines": [
            {"id": 1, "lines": [{"provider": "Bovada", "spread": -7.0}, {"provider": "consensus", "spread": -6.5}]}
        ],
    }
    session = _FakeSession(routes)
    client = CFBDClient(Settings(api_key="secret", cache_dir=tmp_path), session=session)
    teams, games = load_season(2024, "regular", client=client, fcs="drop")
    assert [t.school for t in teams] == ["A", "B"]
    assert [g.id for g in games] == [1]  # FCS game dropped
    teams, games = load_season(2024, "regular", client=client)  # default: pool
    assert [t.school for t in teams] == ["A", "B", FCS_POOL]
    assert [g.id for g in games] == [1, 2] and games[1].away == FCS_POOL and games[1].away_name == "FCS U"
    assert client.lines(2024, "regular") == {1: -6.5}
    n_calls = len(session.calls)
    load_season(2024, "regular", client=client)
    client.lines(2024, "regular")
    assert len(session.calls) == n_calls  # served from cache
    load_season(2024, "regular", client=client, force_refresh=True)
    assert len(session.calls) == n_calls + 2
    with pytest.raises(ValueError):
        load_season(2024, "regular", client=client, fcs="nope")


def test_load_seasons_merges_rosters(tmp_path):
    def game(gid, season, home, away):
        return {
            "id": gid,
            "season": season,
            "week": 1,
            "homeTeam": home,
            "awayTeam": away,
            "homePoints": 20,
            "awayPoints": 10,
            "completed": True,
        }

    routes = {
        ("/teams/fbs", 2023): [{"school": "A"}, {"school": "B"}, {"school": "C"}],  # C left FBS after 2023
        ("/teams/fbs", 2024): [{"school": "A"}, {"school": "B"}, {"school": "D"}],  # D joined in 2024
        ("/games", 2023): [game(1, 2023, "A", "C")],
        ("/games", 2024): [game(2, 2024, "D", "B")],
    }
    session = _FakeSession(routes)
    client = CFBDClient(Settings(api_key="secret", cache_dir=tmp_path), session=session)
    teams, games = load_seasons([2024, 2023], "regular", client=client)
    assert [t.school for t in teams] == ["A", "B", "D", "C"]  # latest roster first, earlier-only teams appended
    assert [(g.id, g.season) for g in games] == [(1, 2023), (2, 2024)]


def test_consensus_spreads_fallback_to_mean():
    raw = [
        {"id": 5, "lines": [{"provider": "X", "spread": -3}, {"provider": "Y", "spread": "-5"}]},
        {"id": 6, "lines": [{"provider": "X", "spread": None}]},
        {"id": None, "lines": [{"spread": 1}]},
    ]
    assert consensus_spreads(raw) == {5: -4.0}
