from __future__ import annotations

from cfbratings.data.schema import Game
from cfbratings.models import make_model
from cfbratings.ppoints import PPointsConfig, ppoints
from cfbratings.snapshots import trajectory, weekly_snapshots


def _game(i, home, away, hp, ap, week=1, neutral=False, hc=None, ac=None):
    return Game(
        id=i,
        season=2025,
        week=week,
        home=home,
        away=away,
        home_points=hp,
        away_points=ap,
        neutral_site=neutral,
        completed=True,
        home_conference=hc,
        away_conference=ac,
    )


def test_tier_points():
    cfg = PPointsConfig()
    assert [cfg.tier_points(r) for r in (1, 10, 11, 25, 26, 40, 41, 60, 61, None)] == [8, 8, 5, 5, 3, 3, 2, 2, 1, 1]


def test_ppoints_rewards_quality_road_and_nonconference_wins():
    teams = [f"T{i}" for i in range(1, 71)]
    ratings = {t: float(len(teams) - i) for i, t in enumerate(teams)}  # T1 best ... T70 worst
    conf = {t: ("C2" if t in {"T5", "T6"} else "C1") for t in teams}  # two elite teams form their own conference
    games = [
        _game(1, "T30", "T5", 21, 20, hc="C1", ac="C2"),  # T30 beats a top-10 team at home, non-conf
        _game(2, "T31", "T65", 21, 20, hc="C1", ac="C1"),  # T31 beats a bad team at home, in-conf
        _game(3, "T66", "T32", 20, 21, hc="C1", ac="C1"),  # T32 wins on the road vs a bad team, in-conf
        _game(4, "T33", "T6", 20, 21, hc="C1", ac="C2"),  # T33 loses at home to a top-10 team, non-conf
        _game(5, "T34", "T67", 20, 21, hc="C1", ac="C1"),  # T34 loses at home to a bad team, in-conf
    ]
    pp = ppoints(teams, games, ratings, conf)
    assert pp["T30"] > pp["T31"] > 0  # beating #5 beats beating #65
    assert pp["T32"] > pp["T31"]  # road bonus
    assert pp["T33"] < 0 and pp["T34"] < 0 and pp["T34"] < pp["T33"]  # bad losses hurt more
    assert pp["T5"] < 0 and pp["T65"] < 0  # losers are penalised
    pp_conf_only = ppoints(
        teams, games, ratings, conf, config=PPointsConfig(nonconf_win_factor=1.0, nonconf_loss_penalty=4.0)
    )
    assert pp_conf_only["T30"] < pp["T30"]  # non-conf bonus removed


def test_ppoints_uses_weekly_snapshot_for_game_time_rank():
    teams = [f"T{i}" for i in range(1, 71)]
    current = {t: float(len(teams) - i) for i, t in enumerate(teams)}
    then = dict(current)
    then["T50"] = 1000.0  # T50 was #1 entering week 2
    weekly = {1: then}
    games = [_game(1, "T40", "T50", 21, 20, week=2), _game(2, "T41", "T51", 21, 20, week=2)]
    pp = ppoints(
        teams,
        games,
        current,
        dict.fromkeys(teams, "C"),
        weekly_ratings=weekly,
        config=PPointsConfig(weight_current=0.0),
    )
    assert pp["T40"] > pp["T41"]
    pp_now = ppoints(
        teams,
        games,
        current,
        dict.fromkeys(teams, "C"),
        weekly_ratings=weekly,
        config=PPointsConfig(weight_current=1.0),
    )
    assert abs(pp_now["T40"] - pp_now["T41"]) < 1e-9


def test_weekly_snapshots_cache_and_trajectory(season, settings):
    snaps = weekly_snapshots("massey", season.team_names, season.games, settings=settings)
    weeks = sorted(snaps)
    assert weeks[0] == 1 and weeks[-1] == max(g.week for g in season.games)
    assert set(snaps[weeks[-1]]) == set(season.team_names)
    final = make_model("massey", settings).fit(season.team_names, season.games).ratings
    assert snaps[weeks[-1]] == final
    again = weekly_snapshots("massey", season.team_names, season.games, settings=settings)
    assert again == snaps  # served from cache (identical)
    assert list(settings.cache_dir.glob("snapshots_*.json"))
    different = weekly_snapshots("massey", season.team_names, season.games, settings=settings, ridge_lambda=5.0)
    assert different[weeks[-1]] != snaps[weeks[-1]]  # params are part of the cache key
    best = max(final, key=final.get)
    traj = trajectory(snaps, best)
    assert traj[-1][1] == 1 and [w for w, _, _ in traj] == weeks
    assert weekly_snapshots("elo", season.team_names, [], settings=settings) == {}
