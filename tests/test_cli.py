from __future__ import annotations

import json

import pytest

from cfbratings.cli import main, parse_grid, parse_params


def run(capsys, *argv):
    code = main(["--demo", "--seed", "3", "--year", "2025", *argv])
    out = capsys.readouterr().out
    return code, out


def test_parse_helpers():
    assert parse_params(["k=30", "hfa=2.5", "mov=false", "name=x"]) == {"k": 30, "hfa": 2.5, "mov": False, "name": "x"}
    assert parse_grid("k=15,25;hfa=2,2.5") == {"k": [15, 25], "hfa": [2.0, 2.5]}
    with pytest.raises(SystemExit):
        parse_params(["bad"])
    with pytest.raises(SystemExit):
        parse_grid("")


def test_rank_and_exports(capsys, tmp_path):
    csv = tmp_path / "out" / "rank.csv"
    js = tmp_path / "rank.json"
    code, out = run(capsys, "rank", "--method", "massey", "--top", "5", "--csv", str(csv), "--json", str(js))
    assert code == 0 and "Massey ratings" in out and "ppoints" in out
    assert csv.exists() and len(json.loads(js.read_text())) == 64
    code, out = run(capsys, "rank", "--no-ppoints", "--sort-by", "sos", "--conference", "SEC", "--param", "k=30")
    assert code == 0 and "ppoints" not in out.split("\n")[3]


def test_other_commands(capsys):
    code, out = run(capsys, "conferences", "--method", "elo")
    assert code == 0 and "strength_index" in out
    code, out = run(capsys, "team", "Delta State", "--method", "colley")
    assert code == 0 and "Weekly rank" in out
    code, out = run(capsys, "predict", "Delta State", "Lakes State", "--neutral", "--methods", "massey,elo")
    assert code == 0 and "neutral site" in out and "Massey" in out
    code, out = run(capsys, "backtest", "--methods", "massey,elo", "--start-week", "6", "--by-week", "--calibration")
    assert code == 0 and "home_prior (baseline)" in out and "calibration" in out
    code, out = run(capsys, "tune", "--method", "elo", "--grid", "k=20,40", "--start-week", "8")
    assert code == 0 and "best by log_loss" in out
    code, out = run(capsys, "snapshot", "--method", "massey", "--team", "Delta State")
    assert code == 0 and "weekly snapshots" in out


def test_project_and_multi_season(capsys):
    code = main(
        ["--demo", "--seed", "3", "--year", "2025", "--demo-weeks", "9", "project", "--method", "massey", "--top", "5"]
    )
    out = capsys.readouterr().out
    assert code == 0 and "unplayed games" in out and "proj_record" in out and "0 unplayed" not in out
    code = main(["--demo", "--seed", "3", "--years", "2024-2025", "backtest", "--methods", "elo", "--start-week", "1"])
    out = capsys.readouterr().out
    assert code == 0 and "warm-started" in out and "by season" in out
    code = main(
        ["--demo", "--seed", "3", "--year", "2025", "--warmup-years", "1", "rank", "--top", "3", "--no-ppoints"]
    )
    out = capsys.readouterr().out
    assert code == 0 and "warm-started from 1 prior season" in out


def test_cache_commands(capsys, tmp_path):
    code = main(["--cache-dir", str(tmp_path), "cache", "list"])
    assert code == 0 and "0 cached files" in capsys.readouterr().out
    from cfbratings.data import JsonCache

    JsonCache(tmp_path).put("games_2024_both", [], final=True)
    JsonCache(tmp_path).put("games_2026_both", [])
    code = main(["--cache-dir", str(tmp_path), "cache", "list"])
    out = capsys.readouterr().out
    assert code == 0 and "2 cached files" in out
    assert "games_2024_both.json" in out and "never expires" in out
    assert "games_2026_both.json" in out and "fresh for another" in out
    code = main(["--cache-dir", str(tmp_path), "cache", "clear"])
    assert code == 0


def test_unknown_team_hint(capsys):
    with pytest.raises(SystemExit, match="Did you mean"):
        run(capsys, "team", "delta st")
