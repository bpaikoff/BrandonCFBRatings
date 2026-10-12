from __future__ import annotations

import itertools

import numpy as np
import pytest

from cfbratings.analytics import conference_strength, conference_table, independent_strength, independents_table
from cfbratings.cfp import (
    FORMATS,
    CFPFormat,
    Seed,
    SeedingWeights,
    bracket_lines,
    bracket_odds,
    championship_games,
    committee_ranking,
    conference_champions,
    conference_standings,
    format_for,
    playoff_picture,
    select_field,
    simulate_playoff,
)
from cfbratings.data import synthetic_season
from cfbratings.data.schema import Game, Team, is_independent
from cfbratings.models import make_model
from cfbratings.ppoints import ppoints
from cfbratings.predict import MarginCalibrator


def _game(i, home, away, hp=None, ap=None, *, week=1, hc=None, ac=None, conf_game=None, notes=None, neutral=False):
    return Game(
        id=i,
        season=2026,
        week=week,
        home=home,
        away=away,
        home_points=hp,
        away_points=ap,
        completed=hp is not None,
        home_conference=hc,
        away_conference=ac,
        conference_game=conf_game,
        notes=notes,
        neutral_site=neutral,
    )


# --- independents -----------------------------------------------------------------------------


def test_independents_are_not_a_conference():
    assert is_independent(None) and is_independent("FBS Independents") and is_independent("Independent")
    assert not is_independent("SEC")
    nd_uconn = _game(1, "Notre Dame", "UConn", 35, 7, hc="FBS Independents", ac="FBS Independents")
    assert not nd_uconn.is_conference_game  # two independents never play a conference game
    army_navy = _game(2, "Navy", "Army", 17, 14, hc="American Athletic", ac="American Athletic", conf_game=True)
    assert not army_navy.is_conference_game  # flagged by CFBD, not counted by the conference
    flagged = _game(3, "A", "B", 10, 3, hc="X", ac="X", conf_game=False)
    assert not flagged.is_conference_game  # CFBD's flag wins over "same conference"
    derived = _game(4, "A", "B", 10, 3, hc="X", ac="X")
    assert derived.is_conference_game

    ratings = {"ND": 20.0, "UConn": -15.0, "A": 10.0, "B": 0.0, "C": -5.0, "D": 5.0, "E": -2.0, "F": -8.0}
    conf = {"ND": "FBS Independents", "UConn": None, "A": "X", "B": "X", "C": "X", "D": "Y", "E": "Y", "F": "Y"}
    strength = conference_strength(ratings, conf)
    assert set(strength) == {"X", "Y"}  # no "Independent" pseudo-conference
    ind = independent_strength(ratings, conf)
    assert set(ind) == {"ND", "UConn"}
    assert ind["ND"] == 1.6 and ind["UConn"] == 0.6  # clipped to the conference band
    assert "Independent" not in set(conference_table(ratings, conf)["conference"])
    table = independents_table(ratings, conf)
    assert list(table["team"]) == ["ND", "UConn"] and int(table.iloc[0]["rank"]) == 1


def test_ppoints_scales_independent_opponents_by_their_own_strength():
    teams = ["ND", "UConn", "A", "B", "C", "D"]
    conf = {"ND": "FBS Independents", "UConn": "FBS Independents", "A": "X", "B": "X", "C": "Y", "D": "Y"}
    ratings = {"ND": 20.0, "UConn": -20.0, "A": 8.0, "B": 2.0, "C": -2.0, "D": -8.0}
    games = [
        _game(1, "A", "ND", 21, 20, hc="X", ac="FBS Independents"),  # A beats a strong independent
        _game(2, "B", "UConn", 21, 20, hc="X", ac="FBS Independents"),  # B beats a weak one
        _game(3, "C", "D", 10, 7, hc="Y", ac="Y"),
    ]
    pp = ppoints(teams, games, ratings, conf)
    assert pp["A"] > pp["B"]  # same tier of result, but Notre Dame counts like a top-conference opponent


# --- standings and champions ---------------------------------------------------------------------


@pytest.fixture
def league():
    teams = [
        Team("A", "X"),
        Team("B", "X"),
        Team("C", "X"),
        Team("D", "X"),
        Team("E", "Y"),
        Team("F", "Y"),
        Team("ND", "FBS Independents"),
    ]
    games = [
        _game(1, "A", "B", 20, 10, hc="X", ac="X"),  # A beat B
        _game(2, "B", "C", 20, 10, hc="X", ac="X"),  # B beat C
        _game(3, "C", "A", 20, 10, hc="X", ac="X"),  # C beat A  -> A, B, C all 1-1
        _game(4, "D", "A", 10, 20, hc="X", ac="X"),  # A beat D  -> A 2-1
        _game(5, "D", "B", 10, 20, hc="X", ac="X"),  # B beat D  -> B 2-1, D 0-2
        _game(6, "C", "ND", 10, 20, hc="X", ac="FBS Independents"),  # non-conference loss for C
        _game(7, "E", "F", 30, 0, hc="Y", ac="Y"),
        _game(8, "ND", "E", 30, 0, hc="FBS Independents", ac="Y"),
    ]
    ratings = {"A": 10.0, "B": 12.0, "C": 5.0, "D": -5.0, "E": 8.0, "F": -10.0, "ND": 20.0}
    return teams, games, ratings


def test_standings_order_and_tiebreaks(league):
    teams, games, ratings = league
    standings = conference_standings(teams, games, ratings)
    assert set(standings) == {"X", "Y"}  # the independent has no standings
    x = [(s.team, s.conf_record) for s in standings["X"]]
    # A and B are both 2-1; A beat B head-to-head, so A leads despite B's higher rating.
    assert x == [("A", "2-1"), ("B", "2-1"), ("C", "1-1"), ("D", "0-2")]
    assert standings["X"][2].record == "1-2"  # C's overall record counts the non-conference loss
    assert [s.team for s in standings["Y"]] == ["E", "F"]


def test_title_game_detection_and_champions(league):
    teams, games, _ = league
    ccg = _game(9, "A", "B", 24, 21, week=15, hc="X", ac="X", conf_game=False, notes="X Championship", neutral=True)
    not_ccg = _game(10, "E", "F", 7, 3, week=15, hc="Y", ac="Y", notes="FCS Championship - First Round")
    assert ccg.is_conference_championship and not ccg.is_conference_game
    assert not not_ccg.is_conference_championship
    assert list(championship_games([*games, ccg, not_ccg])) == ["X"]

    result = make_model("massey").fit([t.school for t in teams], games)
    cal = MarginCalibrator(slope=1.0, intercept=2.5, sigma=14.0, n_games=5, source="prior")
    fmt = CFPFormat(power_conferences=("X",), neutral_championships=("X",))
    champs = conference_champions(teams, games, result, cal, fmt)
    assert not champs["X"].decided and {champs["X"].team, champs["X"].opponent} == {"A", "B"}
    assert champs["X"].p_champion >= 0.5
    played = conference_champions(teams, [*games, ccg], result, cal, fmt)
    assert played["X"].decided and played["X"].team == "A" and played["X"].opponent == "B"


# --- the committee and the field ---------------------------------------------------------------


def test_committee_ranking_weights():
    teams = ["A", "B", "C"]
    ratings = {"A": 10.0, "B": 9.0, "C": 0.0}
    # Same rating, more losses -> lower; a champion bump can lift B over A at equal losses.
    assert committee_ranking(teams, ratings, {"A": 2, "B": 0, "C": 0}, weights=SeedingWeights(loss=1.0)) == [
        "B",
        "A",
        "C",
    ]
    assert (
        committee_ranking(
            teams, ratings, {"A": 0, "B": 0, "C": 0}, weights=SeedingWeights(champion=5.0), champions={"B"}
        )[0]
        == "B"
    )
    assert committee_ranking(
        teams, ratings, {"A": 0, "B": 0, "C": 0}, weights=SeedingWeights(rating=1, loss=0, champion=0)
    ) == ["A", "B", "C"]
    w = SeedingWeights.parse("rating=2,loss=0.5")
    assert (w.rating, w.loss, w.resume, w.champion) == (2.0, 0.5, 0.5, 0.15)
    with pytest.raises(ValueError):
        SeedingWeights.parse("bogus=1")


def test_select_field_2026_rules():
    conf = {f"T{i}": "SEC" if i % 2 else "Big Ten" for i in range(1, 15)}
    conf.update({"P": "ACC", "Q": "Big 12", "G": "Mountain West", "H": "Sun Belt", "ND": "FBS Independents"})
    ranking = [f"T{i}" for i in range(1, 7)] + ["ND", "G"] + [f"T{i}" for i in range(7, 15)] + ["P", "Q", "H"]
    champions = {"SEC": "T1", "Big Ten": "T2", "ACC": "P", "Big 12": "Q", "Mountain West": "H", "Sun Belt": "H"}
    field = select_field(ranking, champions, conf, format_for(2026))
    seeds = {s.team: s for s in field}
    assert [s.team for s in field][:8] == ["T1", "T2", "T3", "T4", "T5", "T6", "ND", "G"]  # straight seeding
    assert seeds["P"].bid == "champion" and seeds["Q"].bid == "champion"  # power champions qualify from 20th
    assert seeds["G"].bid == "group"  # best Group-of-Six team, not a champion
    assert "H" not in seeds and seeds["ND"].bid == "at-large" and not seeds["ND"].champion
    assert [s.seed for s in field] == list(range(1, 13)) and field[-1].team == "Q"  # qualifiers seeded last
    assert sum(s.bid == "at-large" for s in field) == 7

    # 2024: five best champions, byes to the four best champions.
    field24 = select_field(ranking, champions, conf, FORMATS[2024])
    top = [s.team for s in field24[:4]]
    assert top == ["T1", "T2", "P", "Q"] and all(s.champion for s in field24[:4])
    assert "H" in {s.team for s in field24}  # the fifth champion
    assert FORMATS[2025].champion_bids == 5 and not FORMATS[2025].champion_byes
    assert format_for(2030) is FORMATS[2026] and format_for(2010) is FORMATS[2024]


def test_format_resolves_against_synthetic_conferences():
    fmt = format_for(2026).resolved({"Big Ten": 1.5, "Atlantic Coast": 1.2, "Big 12": 1.1, "SEC": 1.6, "Mid": 0.7})
    assert set(fmt.power_conferences) == {"Big Ten", "Big 12", "SEC", "Atlantic Coast"}
    assert "Mid" not in fmt.power_conferences


# --- the bracket ----------------------------------------------------------------------------------


def _field(n=12):
    return [Seed(i, f"S{i}", "C", False, "at-large", i) for i in range(1, n + 1)]


def test_bracket_odds_coin_flips_and_structure():
    odds = bracket_odds(_field(), lambda a, b, neutral: 0.5)
    t = odds.table.set_index("seed")
    assert np.allclose(t.loc[1:4, "p_title"], 1 / 8) and np.allclose(t.loc[5:12, "p_title"], 1 / 16)
    assert t["p_title"].sum() == pytest.approx(1.0)
    assert np.allclose(t.loc[1:4, "p_quarter"], 1.0) and np.allclose(t.loc[5:12, "p_quarter"], 0.5)
    assert t["p_semi"].sum() == pytest.approx(4.0) and t["p_final"].sum() == pytest.approx(2.0)
    rounds = [g.round for g in odds.games]
    assert rounds == ["first"] * 4 + ["quarter"] * 4 + ["semi"] * 2 + ["final"]
    first = {(g.home.seed, g.away.seed) for g in odds.games if g.round == "first"}
    assert first == {(5, 12), (6, 11), (7, 10), (8, 9)} and not any(g.neutral for g in odds.games if g.round == "first")
    quarter = {(g.home.seed, g.away.seed) for g in odds.games if g.round == "quarter"}
    assert quarter == {(1, 8), (2, 7), (3, 6), (4, 5)} and all(g.neutral for g in odds.games if g.round != "first")
    assert bracket_lines(odds)[0] == "First round:" and "(5) S5 vs (12) S12" in bracket_lines(odds)[1]


def test_bracket_odds_match_brute_force_with_home_advantage():
    """Exact recursion vs enumerating all 2^11 outcomes, with asymmetric (home) probabilities."""
    strength = {f"S{i}": 12.0 - i for i in range(1, 13)}
    cal = MarginCalibrator(slope=1.0, intercept=3.0, sigma=10.0, n_games=1, source="prior")

    def win_prob(a, b, neutral):
        return cal.win_prob(strength[a], strength[b], neutral)

    field = _field()
    odds = bracket_odds(field, win_prob).table.set_index("team")["p_title"]
    assert odds.sum() == pytest.approx(1.0)

    # brute force over every outcome of the 11 games
    def play(a, b, neutral, outcome):
        return a if outcome else b

    title = dict.fromkeys(strength, 0.0)
    pairs = [(5, 12), (8, 9), (6, 11), (7, 10)]
    for bits in itertools.product([0, 1], repeat=11):
        prob = 1.0
        k = 0
        w = {}
        for hi, lo in pairs:  # first round at the higher seed's home
            a, b = f"S{hi}", f"S{lo}"
            p = win_prob(a, b, False)
            prob *= p if bits[k] else 1 - p
            w[(hi, lo)] = play(a, b, False, bits[k])
            k += 1
        q = []
        for top, pair in ((1, (8, 9)), (4, (5, 12)), (3, (6, 11)), (2, (7, 10))):
            a, b = f"S{top}", w[pair]
            p = win_prob(a, b, True)
            prob *= p if bits[k] else 1 - p
            q.append(play(a, b, True, bits[k]))
            k += 1
        s = []
        for a, b in ((q[0], q[1]), (q[2], q[3])):
            p = win_prob(a, b, True)
            prob *= p if bits[k] else 1 - p
            s.append(play(a, b, True, bits[k]))
            k += 1
        p = win_prob(s[0], s[1], True)
        prob *= p if bits[k] else 1 - p
        title[play(s[0], s[1], True, bits[k])] += prob
    for team, p in title.items():
        assert odds[team] == pytest.approx(p, abs=1e-12)


def test_partial_field_gets_byes():
    odds = bracket_odds(_field(6), lambda a, b, neutral: 0.5)
    assert odds.table["p_title"].sum() == pytest.approx(1.0)


# --- end to end ---------------------------------------------------------------------------------


def test_playoff_picture_and_simulation_on_synthetic_season():
    season = synthetic_season(season=2026, n_teams=48, n_conferences=6, weeks=12, played_through=8, seed=3)
    names = season.team_names
    model = make_model("massey")
    result = model.fit(names, season.games)
    cal = MarginCalibrator.fit_cv(model, names, season.games, fallback_hfa=2.5)
    pic = playoff_picture(season.teams, season.games, result, cal, weights=SeedingWeights())
    assert len(pic.fmt.power_conferences) == 4 and len(pic.field) == 12
    assert [s.seed for s in pic.field] == list(range(1, 13))
    assert pic.odds.table["p_title"].sum() == pytest.approx(1.0)
    assert len(pic.champions) == 6 and not any(c.decided for c in pic.champions.values())  # mid-season
    assert sum(s.bid == "champion" for s in pic.field) == 4 and sum(s.bid == "group" for s in pic.field) == 1

    sim = simulate_playoff(season.teams, season.games, result, cal, n_sims=120, seed=7)
    t = sim.table
    assert len(t) == len(names) and t["p_title"].sum() == pytest.approx(1.0, abs=1e-9)
    assert t["p_playoff"].sum() == pytest.approx(12.0, abs=1e-9)
    assert t["p_bye"].sum() == pytest.approx(4.0, abs=1e-9)
    assert ((t["p_playoff"] >= t["p_bye"]) & (t["p_quarter"] >= t["p_semi"]) & (t["p_semi"] >= t["p_title"])).all()
    for _, grp in sim.champions.groupby("conference"):
        assert grp["p_champion"].sum() == pytest.approx(1.0)
    assert sim.champions["conference"].nunique() == 6
    assert len(sim.projected_field) == 12 and sim.projected_odds.table["p_title"].sum() == pytest.approx(1.0)
    # the best-rated team is a favourite: top-3 in title odds
    best = result.ranked()[0][0]
    assert best in set(t.head(3)["team"])
    # seeds and byes agree with the format
    assert t.loc[t["mean_seed"].notna(), "mean_seed"].between(1, 12).all()


def test_simulation_respects_played_title_game():
    teams = [Team(t, "X") for t in "ABCD"] + [Team(t, "Y") for t in "EFGH"]
    games = [
        _game(1, "A", "B", 30, 0, hc="X", ac="X"),
        _game(2, "C", "D", 30, 0, hc="X", ac="X"),
        _game(3, "A", "C", 30, 0, hc="X", ac="X"),
        _game(4, "B", "D", 30, 0, hc="X", ac="X"),
        _game(5, "E", "F", 30, 0, hc="Y", ac="Y"),
        _game(6, "G", "H", 30, 0, hc="Y", ac="Y"),
        _game(7, "E", "G", 3, 0, hc="Y", ac="Y"),
        _game(8, "F", "H", 30, 0, hc="Y", ac="Y"),
        # X's title game is played and C won it, upsetting the leader A
        _game(9, "A", "C", 0, 3, week=15, hc="X", ac="X", conf_game=False, notes="X Championship", neutral=True),
        # Y's title game is on the schedule but unplayed: E hosts G
        _game(10, "E", "G", week=15, hc="Y", ac="Y", conf_game=False, notes="Y Championship"),
    ]
    names = [t.school for t in teams]
    result = make_model("colley").fit(names, games)
    cal = MarginCalibrator(slope=1.0, intercept=2.5, sigma=14.0, n_games=5, source="prior")
    fmt = CFPFormat(field_size=4, byes=0, power_conferences=("X", "Y"), group_bids=0, neutral_championships=("X",))
    sim = simulate_playoff(teams, games, result, cal, fmt=fmt, n_sims=50, seed=0)
    champs = sim.champions.set_index(["conference", "team"])["p_champion"]
    assert champs[("X", "C")] == 1.0  # decided on the field
    assert set(sim.champions[sim.champions["conference"] == "Y"]["team"]) <= {"E", "G"}  # the scheduled matchup
    pic = playoff_picture(teams, games, result, cal, fmt=fmt)
    assert pic.champions["X"].decided and pic.champions["X"].team == "C"
    assert not pic.champions["Y"].decided and {pic.champions["Y"].team, pic.champions["Y"].opponent} == {"E", "G"}
