"""Conference standings and the College Football Playoff.

Everything the committee's Selection Sunday needs, built from the same ratings and
calibrated win probabilities as the rest of the package:

1. **Conference standings** from conference games only (CFBD's ``conferenceGame`` flag),
   ordered by conference record with head-to-head, overall record and rating as
   tie-breakers. Independents have no standings: they are not a conference.
2. **Champions**: the winner of a conference title game once it has been played (CFBD notes
   it as ``"<Conference> Championship"``); until then the top two of the standings meet in a
   projected title game, at a neutral site for the power conferences.
3. **A committee-style ranking** (:func:`committee_ranking`): a weighted blend of the power
   rating, the PPoints résumé score, a penalty per loss and a conference-champion bump. The
   default :class:`SeedingWeights` are a heuristic reading of how the committee has ranked
   since 2014 (losses first, then résumé, with a modest champion bump); they are parameters,
   not fitted values.
4. **The field and the bracket** per :class:`CFPFormat`: automatic bids, at-large bids,
   straight seeding, first round at the higher seed's home, every later round neutral.
5. **Odds**: exact advancement probabilities for a fixed bracket (:func:`bracket_odds`) and a
   Monte Carlo of the rest of the season (:func:`simulate_playoff`) that plays every unplayed
   game, settles the standings and title games, ranks, selects, seeds and then scores the
   bracket exactly, so title odds carry little simulation noise.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .analytics import conference_strength, records
from .data.schema import Game, Record, Team, is_independent, is_pool_team
from .models.base import RatingResult
from .predict import MarginCalibrator
from .scores import ProjectedScore, ScoreModel

POWER_FOUR: tuple[str, ...] = ("ACC", "Big 12", "Big Ten", "SEC")
GROUP_SIX: tuple[str, ...] = (
    "American Athletic",
    "Conference USA",
    "Mid-American",
    "Mountain West",
    "Pac-12",
    "Sun Belt",
)

ROUNDS: tuple[str, ...] = ("first", "quarter", "semi", "final")
ROUND_LABELS: dict[str, str] = {
    "first": "First round",
    "quarter": "Quarterfinals",
    "semi": "Semifinals",
    "final": "National Championship",
}


# --- the rules ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CFPFormat:
    """How the field is picked and seeded.

    * ``power_conferences``: their champions always qualify (2026: ACC, Big 12, Big Ten, SEC).
    * ``champion_bids``: this many further highest-ranked champions of other conferences
      qualify (2024 and 2025: the five best champions of any conference, so ``power`` is empty
      and ``champion_bids`` is 5).
    * ``group_bids``: this many highest-ranked teams of the non-power conferences qualify,
      champion or not (2026: one, the best of the Group of Six).
    * The rest of the ``field_size`` places are at-large, by ranking.
    * ``champion_byes``: the byes go to the highest-ranked champions (2024) instead of straight
      to seeds 1-4 by ranking (2025 onward).
    * ``neutral_championships``: conferences whose title game is at a neutral site; the others
      are hosted by the higher-placed team.
    """

    name: str = "2026"
    field_size: int = 12
    byes: int = 4
    power_conferences: tuple[str, ...] = POWER_FOUR
    champion_bids: int = 0
    group_bids: int = 1
    champion_byes: bool = False
    neutral_championships: tuple[str, ...] = POWER_FOUR

    @property
    def summary(self) -> str:
        bits = [f"{self.field_size} teams"]
        if self.power_conferences:
            bits.append(f"champions of {', '.join(self.power_conferences)} qualify")
        if self.champion_bids:
            bits.append(f"{self.champion_bids} highest-ranked champions qualify")
        if self.group_bids:
            bits.append(
                f"{self.group_bids} highest-ranked team{'s' if self.group_bids > 1 else ''} of the other conferences"
            )
        at_large = self.field_size - len(self.power_conferences) - self.champion_bids - self.group_bids
        bits.append(f"{at_large} at-large")
        bits.append(
            f"byes to the {self.byes} highest-ranked champions"
            if self.champion_byes
            else f"straight seeding, seeds 1-{self.byes} bye"
        )
        return "; ".join(bits)

    def resolved(self, conferences: Mapping[str, float]) -> CFPFormat:
        """The format against the conferences actually in the data.

        Power conferences missing from the data (a synthetic season, a renamed league) are
        replaced by the strongest other conferences by ``conferences`` (a strength index), so
        the demo plays the same rules with its own leagues.
        """
        present = [c for c in self.power_conferences if c in conferences]
        if len(present) < len(self.power_conferences):
            others = sorted((c for c in conferences if c not in present), key=lambda c: -conferences[c])
            present = present + others[: len(self.power_conferences) - len(present)]
        neutral = tuple(c for c in self.neutral_championships if c in conferences) or tuple(present)
        return replace(self, power_conferences=tuple(present), neutral_championships=neutral)


FORMATS: dict[int, CFPFormat] = {
    2024: CFPFormat(name="2024", power_conferences=(), champion_bids=5, group_bids=0, champion_byes=True),
    2025: CFPFormat(name="2025", power_conferences=(), champion_bids=5, group_bids=0),
    2026: CFPFormat(name="2026"),
}


def format_for(season: int) -> CFPFormat:
    """The rules in force for ``season`` (the latest known format for later seasons)."""
    years = sorted(FORMATS)
    if season in FORMATS:
        return FORMATS[season]
    return FORMATS[years[-1]] if season > years[-1] else FORMATS[years[0]]


# --- standings ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Standing:
    team: str
    conference: str
    conf_wins: int
    conf_losses: int
    wins: int
    losses: int
    rating: float

    @property
    def conf_pct(self) -> float:
        n = self.conf_wins + self.conf_losses
        return self.conf_wins / n if n else 0.0

    @property
    def win_pct(self) -> float:
        n = self.wins + self.losses
        return self.wins / n if n else 0.0

    @property
    def conf_record(self) -> str:
        return f"{self.conf_wins}-{self.conf_losses}"

    @property
    def record(self) -> str:
        return f"{self.wins}-{self.losses}"


def conference_members(teams: Sequence[Team]) -> dict[str, list[str]]:
    """Conference → member schools (independents and pooled pseudo-teams left out)."""
    out: dict[str, list[str]] = {}
    for t in teams:
        if is_pool_team(t.school) or is_independent(t.conference):
            continue
        out.setdefault(str(t.conference), []).append(t.school)
    return out


def _order_standings(
    members: Sequence[str],
    conf_rec: Mapping[str, tuple[int, int]],
    overall: Mapping[str, Record],
    ratings: Mapping[str, float],
    h2h: Mapping[tuple[str, str], int],
) -> list[str]:
    """Conference record, then head-to-head among the tied, then overall record, then rating."""

    def pct(w: int, losses: int) -> float:
        return w / (w + losses) if w + losses else 0.0

    groups: dict[float, list[str]] = {}
    for t in members:
        w, losses = conf_rec.get(t, (0, 0))
        groups.setdefault(round(pct(w, losses), 6), []).append(t)
    ordered: list[str] = []
    for key in sorted(groups, reverse=True):
        tied = groups[key]
        if len(tied) == 1:
            ordered.extend(tied)
            continue
        h2h_pct = {}
        for t in tied:
            w = sum(h2h.get((t, o), 0) for o in tied if o != t)
            losses = sum(h2h.get((o, t), 0) for o in tied if o != t)
            h2h_pct[t] = pct(w, losses) if w + losses else 0.5
        tied.sort(
            key=lambda t: (
                -h2h_pct[t],
                -(overall[t].win_pct if t in overall else 0.0),
                -ratings.get(t, 0.0),
                t,
            )
        )
        ordered.extend(tied)
    return ordered


def conference_standings(
    teams: Sequence[Team], games: Sequence[Game], ratings: Mapping[str, float]
) -> dict[str, list[Standing]]:
    """Current standings per conference from completed conference games.

    Title games are not conference games (the standings are settled before them), so they
    never move a team's conference record; they do count in the overall record.
    """
    members = conference_members(teams)
    names = [t.school for t in teams]
    overall = records(names, games)
    conf_rec: dict[str, list[int]] = {t: [0, 0] for conf in members.values() for t in conf}
    h2h: dict[tuple[str, str], int] = {}
    for g in games:
        if not g.has_score or not g.is_conference_game or g.home not in conf_rec or g.away not in conf_rec:
            continue
        w, loser = g.winner, g.loser
        if w is None or loser is None:
            continue
        conf_rec[w][0] += 1
        conf_rec[loser][1] += 1
        h2h[(w, loser)] = h2h.get((w, loser), 0) + 1
    out: dict[str, list[Standing]] = {}
    for conf, schools in members.items():
        order = _order_standings(schools, {t: tuple(conf_rec[t]) for t in schools}, overall, ratings, h2h)
        out[conf] = [
            Standing(
                team=t,
                conference=conf,
                conf_wins=conf_rec[t][0],
                conf_losses=conf_rec[t][1],
                wins=overall[t].wins,
                losses=overall[t].losses,
                rating=float(ratings.get(t, 0.0)),
            )
            for t in order
        ]
    return out


def championship_games(games: Sequence[Game]) -> dict[str, Game]:
    """Conference → its title game (played or scheduled) when the schedule has one."""
    out: dict[str, Game] = {}
    for g in games:
        if g.is_conference_championship and g.home_conference:
            out[str(g.home_conference)] = g
    return out


@dataclass(frozen=True)
class Champion:
    conference: str
    team: str
    decided: bool  # the title game has been played (or the conference has no title game left)
    opponent: str | None = None  # the other title-game participant
    p_champion: float | None = None  # P(team wins the title game) when it is still to be played


def conference_champions(
    teams: Sequence[Team],
    games: Sequence[Game],
    result: RatingResult,
    calibrator: MarginCalibrator,
    fmt: CFPFormat | None = None,
) -> dict[str, Champion]:
    """Who is the champion (title game played) or the projected champion (standings leader).

    A projected champion is the favourite of the title game the standings imply: the top two
    meet, at a neutral site for the conferences in ``fmt.neutral_championships`` and at the
    leader's home otherwise. ``p_champion`` is the leader's chance in that game.
    """
    fmt = fmt or format_for(max((g.season for g in games), default=2026))
    standings = conference_standings(teams, games, result.ratings)
    title = championship_games(games)
    out: dict[str, Champion] = {}
    for conf, table in standings.items():
        if not table:
            continue
        game = title.get(conf)
        if game is not None and game.has_score and game.winner:
            out[conf] = Champion(conf, game.winner, True, game.loser)
            continue
        if game is not None:  # scheduled with known participants
            a, b = game.home, game.away
            p_a = calibrator.win_prob(result.get(a), result.get(b), game.neutral_site)
        else:
            if len(table) == 1:
                out[conf] = Champion(conf, table[0].team, True)
                continue
            a, b = table[0].team, table[1].team
            p_a = calibrator.win_prob(result.get(a), result.get(b), conf in fmt.neutral_championships)
        fav, other, p = (a, b, p_a) if p_a >= 0.5 else (b, a, 1.0 - p_a)
        out[conf] = Champion(conf, fav, False, other, p)
    return out


def standings_table(standings: Mapping[str, Sequence[Standing]]) -> pd.DataFrame:
    rows = [
        {
            "conference": conf,
            "place": i + 1,
            "team": s.team,
            "conf_record": s.conf_record,
            "conf_wins": s.conf_wins,
            "conf_losses": s.conf_losses,
            "record": s.record,
            "wins": s.wins,
            "losses": s.losses,
            "rating": s.rating,
        }
        for conf, table in standings.items()
        for i, s in enumerate(table)
    ]
    return pd.DataFrame(rows)


# --- the committee --------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedingWeights:
    """How the committee proxy ranks: ``rating·z(rating) + resume·z(PPoints) - loss·losses + champion``.

    The defaults are a heuristic reading of the committee's record since 2014: a loss costs
    about a third of a rating standard deviation, the résumé counts half as much as the power
    rating, and a conference title is worth a small bump. They are knobs, not fitted values.
    """

    rating: float = 1.0
    resume: float = 0.5
    loss: float = 0.35
    champion: float = 0.15

    @classmethod
    def parse(cls, spec: str | None) -> SeedingWeights:
        """``"rating=1,resume=0.5,loss=0.35,champion=0.15"`` (any subset)."""
        if not spec:
            return cls()
        values: dict[str, float] = {}
        for part in spec.split(","):
            name, _, value = part.partition("=")
            name = name.strip()
            if name not in cls.__dataclass_fields__:
                raise ValueError(f"unknown seeding weight {name!r}; choose from {list(cls.__dataclass_fields__)}")
            values[name] = float(value)
        return cls(**values)


def _zscores(values: Mapping[str, float], teams: Sequence[str]) -> dict[str, float]:
    arr = np.array([float(values.get(t, 0.0)) for t in teams], dtype=float)
    sd = float(arr.std()) if arr.size else 0.0
    sd = sd if sd > 1e-12 else 1.0
    mean = float(arr.mean()) if arr.size else 0.0
    return {t: (float(values.get(t, 0.0)) - mean) / sd for t in teams}


def committee_scores(
    teams: Sequence[str],
    ratings: Mapping[str, float],
    losses: Mapping[str, int],
    *,
    ppoints: Mapping[str, float] | None = None,
    champions: Sequence[str] | set[str] = (),
    weights: SeedingWeights | None = None,
) -> dict[str, float]:
    w = weights or SeedingWeights()
    real = [t for t in teams if not is_pool_team(t)]
    z_rating = _zscores(ratings, real)
    z_resume = _zscores(ppoints, real) if ppoints else dict.fromkeys(real, 0.0)
    champs = set(champions)
    return {
        t: w.rating * z_rating[t]
        - w.loss * int(losses.get(t, 0))
        + w.resume * z_resume[t]
        + (w.champion if t in champs else 0.0)
        for t in real
    }


def committee_ranking(
    teams: Sequence[str],
    ratings: Mapping[str, float],
    losses: Mapping[str, int],
    *,
    ppoints: Mapping[str, float] | None = None,
    champions: Sequence[str] | set[str] = (),
    weights: SeedingWeights | None = None,
) -> list[str]:
    """Teams best-first by :func:`committee_scores`."""
    scores = committee_scores(teams, ratings, losses, ppoints=ppoints, champions=champions, weights=weights)
    return sorted(scores, key=lambda t: (-scores[t], t))


# --- the field ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Seed:
    seed: int
    team: str
    conference: str
    champion: bool
    bid: str  # "champion", "group" or "at-large"
    rank: int  # position in the committee ranking


def select_field(
    ranking: Sequence[str],
    champions: Mapping[str, str],
    conference_of: Mapping[str, str | None],
    fmt: CFPFormat | None = None,
) -> list[Seed]:
    """Pick and seed the field from a committee ranking and the conference champions.

    ``champions`` maps conference → champion. Automatic qualifiers take their places even when
    ranked outside the top ``field_size``; with straight seeding they are then seeded where
    their ranking puts them, last among the field.
    """
    fmt = fmt or format_for(2026)
    pos = {t: i + 1 for i, t in enumerate(ranking)}
    champ_of: dict[str, str] = {t: c for c, t in champions.items()}
    bids: dict[str, str] = {}

    def take(team: str, bid: str) -> None:
        if team not in bids and len(bids) < fmt.field_size:
            bids[team] = bid

    for conf in fmt.power_conferences:
        if conf in champions:
            take(champions[conf], "champion")
    others = sorted(
        (t for c, t in champions.items() if c not in fmt.power_conferences), key=lambda t: pos.get(t, 10**6)
    )
    for team in others[: fmt.champion_bids]:
        take(team, "champion")
    group_pool = [
        t
        for t in ranking
        if (c := conference_of.get(t)) and c not in fmt.power_conferences and not is_independent(c) and t not in bids
    ]
    for team in group_pool[: fmt.group_bids]:
        take(team, "group")
    for team in ranking:
        if len(bids) >= fmt.field_size:
            break
        take(team, "at-large")

    chosen = sorted(bids, key=lambda t: pos.get(t, 10**6))
    if fmt.champion_byes:
        champs = [t for t in chosen if t in champ_of][: fmt.byes]
        chosen = champs + [t for t in chosen if t not in champs]
    return [
        Seed(
            seed=i + 1,
            team=t,
            conference=str(conference_of.get(t) or "Independent"),
            champion=t in champ_of,
            bid=bids[t],
            rank=pos.get(t, len(ranking) + 1),
        )
        for i, t in enumerate(chosen)
    ]


# --- the bracket ----------------------------------------------------------------------------------

WinProb = Callable[[str, str, bool], float]  # P(first team beats second; neutral site?)

_Node = tuple  # ("seed", Seed) | ("game", round, left, right, neutral)


def _bracket_tree(field: Sequence[Seed]) -> _Node:
    """The 12-team tree: 5-12, 8-9, 6-11, 7-10 at home; then 1/8-9, 4/5-12, 3/6-11, 2/7-10 neutral."""
    by_seed = {s.seed: s for s in field}
    n = len(field)

    def leaf(seed: int) -> _Node | None:
        return ("seed", by_seed[seed]) if seed in by_seed else None

    def game(rnd: str, left: _Node | None, right: _Node | None, neutral: bool) -> _Node | None:
        if left is None:
            return right
        if right is None:
            return left
        return ("game", rnd, left, right, neutral)

    if n > 12:
        raise ValueError("the bracket holds at most 12 teams")
    first = {  # higher seed first (home in the first round)
        (5, 12): game("first", leaf(5), leaf(12), False),
        (8, 9): game("first", leaf(8), leaf(9), False),
        (6, 11): game("first", leaf(6), leaf(11), False),
        (7, 10): game("first", leaf(7), leaf(10), False),
    }
    q1 = game("quarter", leaf(1), first[(8, 9)], True)
    q2 = game("quarter", leaf(4), first[(5, 12)], True)
    q3 = game("quarter", leaf(3), first[(6, 11)], True)
    q4 = game("quarter", leaf(2), first[(7, 10)], True)
    s1 = game("semi", q1, q2, True)
    s2 = game("semi", q3, q4, True)
    final = game("final", s1, s2, True)
    if final is None:
        raise ValueError("an empty field has no bracket")
    return final


@dataclass
class BracketGame:
    round: str
    home: Seed
    away: Seed
    neutral: bool
    p_home: float  # P(home beats away) for the matchup that is set; the modal matchup otherwise
    score: ProjectedScore | None = None  # projected score of that matchup when a scorer is given


def _evaluate(
    node: _Node, win_prob: WinProb, games: list[BracketGame], reach: dict[str, dict[str, float]]
) -> dict[str, float]:
    """P(team wins this node) for every team under it; fills ``games`` and ``reach`` on the way."""
    if node[0] == "seed":
        return {node[1].team: 1.0}
    _, rnd, left, right, neutral = node
    pl = _evaluate(left, win_prob, games, reach)
    pr = _evaluate(right, win_prob, games, reach)
    for t, p in list(pl.items()) + list(pr.items()):
        reach[rnd][t] = p
    # The left side is the higher seed, which hosts a first-round game, so every matchup is
    # evaluated once from its point of view and the right side gets the complement.
    out: dict[str, float] = {}
    for a, pa in pl.items():
        out[a] = pa * sum(pb * win_prob(a, b, neutral) for b, pb in pr.items())
    for b, pb in pr.items():
        out[b] = pb * sum(pa * (1.0 - win_prob(a, b, neutral)) for a, pa in pl.items())
    # the matchup to show: the most likely pair (the only pair when both sides are fixed)
    a = max(pl, key=pl.get)
    b = max(pr, key=pr.get)
    home_seed, away_seed = _seed_of(left, a), _seed_of(right, b)
    if home_seed.seed > away_seed.seed:
        home_seed, away_seed = away_seed, home_seed
    games.append(BracketGame(rnd, home_seed, away_seed, neutral, win_prob(home_seed.team, away_seed.team, neutral)))
    return out


def _seed_of(node: _Node, team: str) -> Seed:
    if node[0] == "seed":
        return node[1]
    for child in (node[2], node[3]):
        try:
            return _seed_of(child, team)
        except KeyError:
            continue
    raise KeyError(team)


@dataclass
class BracketOdds:
    field: list[Seed]
    games: list[BracketGame]  # first round first; each later game shows its most likely matchup
    table: pd.DataFrame  # seed, team, conference, bid, p_quarter, p_semi, p_final, p_title

    def title_favourite(self) -> tuple[str, float]:
        row = self.table.sort_values("p_title", ascending=False).iloc[0]
        return str(row["team"]), float(row["p_title"])


def _bracket_probs(field: Sequence[Seed], win_prob: WinProb) -> tuple[list[BracketGame], dict[str, dict[str, float]]]:
    """``(games, probs)``: ``probs[round][team]`` is P(team reaches that round), plus ``probs["title"]``."""
    tree = _bracket_tree(field)
    games: list[BracketGame] = []
    reach: dict[str, dict[str, float]] = {r: {} for r in ROUNDS}
    title = _evaluate(tree, win_prob, games, reach)
    order = {r: i for i, r in enumerate(ROUNDS)}
    games.sort(key=lambda g: (order[g.round], g.home.seed))
    probs = {
        "quarter": {s.team: reach["quarter"].get(s.team, 1.0 if s.seed <= 4 else 0.0) for s in field},
        "semi": {s.team: reach["semi"].get(s.team, 0.0) for s in field},
        "final": {s.team: reach["final"].get(s.team, 0.0) for s in field},
        "title": {s.team: title.get(s.team, 0.0) for s in field},
    }
    return games, probs


def bracket_odds(field: Sequence[Seed], win_prob: WinProb, scorer: Scorer | None = None) -> BracketOdds:
    """Exact advancement probabilities for a seeded field.

    ``win_prob(a, b, neutral)`` is P(a beats b). Round-by-round probabilities are computed by
    the usual recursion over the bracket (every possible opponent weighted by its probability
    of being there), so the title odds are exact given the per-game probabilities. With a
    ``scorer`` (see :func:`calibrated_scorer`) every game shown carries a projected score.
    """
    if not field:
        return BracketOdds([], [], pd.DataFrame())
    games, probs = _bracket_probs(field, win_prob)
    if scorer is not None:
        for g in games:
            g.score = scorer(g.home.team, g.away.team, g.neutral)
    rows = [
        {
            "seed": s.seed,
            "team": s.team,
            "conference": s.conference,
            "bid": s.bid,
            "champion": s.champion,
            "p_quarter": probs["quarter"][s.team],
            "p_semi": probs["semi"][s.team],
            "p_final": probs["final"][s.team],
            "p_title": probs["title"][s.team],
        }
        for s in sorted(field, key=lambda s: s.seed)
    ]
    return BracketOdds(list(field), games, pd.DataFrame(rows))


def calibrated_win_prob(result: RatingResult, calibrator: MarginCalibrator) -> WinProb:
    """``win_prob(a, b, neutral)`` from the ratings: ``a`` hosts when the site is not neutral."""

    def win_prob(a: str, b: str, neutral: bool) -> float:
        return calibrator.win_prob(result.get(a), result.get(b), neutral)

    return win_prob


Scorer = Callable[[str, str, bool], ProjectedScore]  # projected score of (home, away, neutral?)


def calibrated_scorer(result: RatingResult, calibrator: MarginCalibrator, scores: ScoreModel) -> Scorer:
    """``scorer(home, away, neutral)``: the ratings' margin split around the score model's total."""

    def scorer(home: str, away: str, neutral: bool) -> ProjectedScore:
        margin = calibrator.predict_margin(result.get(home), result.get(away), neutral)
        return scores.project(home, away, margin, neutral=neutral)

    return scorer


# --- "if the season ended today" ---------------------------------------------------------------------


@dataclass
class PlayoffPicture:
    fmt: CFPFormat
    standings: dict[str, list[Standing]]
    champions: dict[str, Champion]
    ranking: list[str]
    field: list[Seed]
    odds: BracketOdds


def playoff_picture(
    teams: Sequence[Team],
    games: Sequence[Game],
    result: RatingResult,
    calibrator: MarginCalibrator,
    *,
    ppoints: Mapping[str, float] | None = None,
    weights: SeedingWeights | None = None,
    fmt: CFPFormat | None = None,
    scores: ScoreModel | None = None,
) -> PlayoffPicture:
    """The bracket if the season ended today: current standings leaders as champions, the
    committee proxy on current records, exact odds for that bracket (with projected scores
    when a :class:`~cfbratings.scores.ScoreModel` is given)."""
    names = [t.school for t in teams]
    conf_of = {t.school: t.conference for t in teams}
    strength = conference_strength(result.ratings, conf_of)
    fmt = (fmt or format_for(max((g.season for g in games), default=2026))).resolved(strength)
    champions = conference_champions(teams, games, result, calibrator, fmt)
    recs = records(names, games)
    losses = {t: r.losses for t, r in recs.items()}
    champ_teams = {c.team for c in champions.values()}
    ranking = committee_ranking(names, result.ratings, losses, ppoints=ppoints, champions=champ_teams, weights=weights)
    field = select_field(ranking, {c: ch.team for c, ch in champions.items()}, conf_of, fmt)
    scorer = calibrated_scorer(result, calibrator, scores) if scores is not None else None
    odds = bracket_odds(field, calibrated_win_prob(result, calibrator), scorer)
    return PlayoffPicture(fmt, conference_standings(teams, games, result.ratings), champions, ranking, field, odds)


# --- Monte Carlo ------------------------------------------------------------------------------------


@dataclass
class PlayoffSimulation:
    n_sims: int
    fmt: CFPFormat
    #: one row per team: record now, projected and most likely final record, p_undefeated,
    #: p_title_game, p_champion, p_playoff, p_bye, mean_seed, p_quarter … p_title, committee_score
    table: pd.DataFrame
    champions: pd.DataFrame  # conference, team, p_title_game, p_champion
    #: conference, home, away, neutral, p_matchup, p_home, favorite, proj_score, decided: every title
    #: game the simulations produced (a played one appears once, with its real score)
    title_games: pd.DataFrame
    seeds: pd.DataFrame  # team, conference, p_playoff, then P(seed = k) in columns "1" … "field_size"
    wins: pd.DataFrame  # team, conference, then P(final wins = k) in columns "0" … max games
    projected_field: list[Seed]  # the modal champions and the mean committee score, seeded
    projected_odds: BracketOdds


def simulate_playoff(
    teams: Sequence[Team],
    games: Sequence[Game],
    result: RatingResult,
    calibrator: MarginCalibrator,
    *,
    ppoints: Mapping[str, float] | None = None,
    weights: SeedingWeights | None = None,
    fmt: CFPFormat | None = None,
    n_sims: int = 2000,
    seed: int = 0,
    scores: ScoreModel | None = None,
) -> PlayoffSimulation:
    """Play out the rest of the season ``n_sims`` times and tally everything that follows.

    Each simulation samples every unplayed game from its calibrated win probability (ratings
    are held fixed), settles the conference standings and title games, ranks the teams with
    the committee proxy on the simulated records, selects and seeds the field, and then takes
    the *exact* bracket odds for that field, so the title probabilities average exact numbers
    rather than coin flips. Along the way it records every team's final record, title-game
    appearances, conference titles and seeds, and every title-game matchup that came up. The
    résumé term keeps its current value: PPoints is not re-scored inside a simulation. With a
    :class:`~cfbratings.scores.ScoreModel` the title games and the projected bracket carry
    projected scores.
    """
    rng = np.random.default_rng(seed)
    all_names = [t.school for t in teams]  # the pooled FCS team included: its games count in records
    names = [t for t in all_names if not is_pool_team(t)]
    index = {t: i for i, t in enumerate(all_names)}
    conf_of = {t.school: t.conference for t in teams}
    strength = conference_strength(result.ratings, conf_of)
    fmt = (fmt or format_for(max((g.season for g in games), default=2026))).resolved(strength)
    members = conference_members(teams)
    win_prob = calibrated_win_prob(result, calibrator)
    scorer = calibrated_scorer(result, calibrator, scores) if scores is not None else None

    # Fixed part: completed games.
    base = records(all_names, games)
    base_wins = np.array([base[t].wins for t in all_names], dtype=int)
    base_losses = np.array([base[t].losses for t in all_names], dtype=int)
    conf_rec: dict[str, list[int]] = {t: [0, 0] for conf in members.values() for t in conf}
    h2h_base: dict[tuple[str, str], int] = {}
    title_games = championship_games(games)
    decided: dict[str, str] = {}
    for g in games:
        if not g.has_score or g.winner is None or g.loser is None:
            continue
        if g.is_conference_game and g.winner in conf_rec and g.loser in conf_rec:
            conf_rec[g.winner][0] += 1
            conf_rec[g.loser][1] += 1
            h2h_base[(g.winner, g.loser)] = h2h_base.get((g.winner, g.loser), 0) + 1
    for conf, g in title_games.items():
        if g.has_score and g.winner:
            decided[conf] = g.winner

    # Random part: unplayed games between known teams, sampled all at once.
    pending = [g for g in games if not g.has_score and g.home in index and g.away in index]
    p_home = np.array([calibrator.win_prob(result.get(g.home), result.get(g.away), g.neutral_site) for g in pending])
    home_idx = np.array([index[g.home] for g in pending], dtype=int)
    away_idx = np.array([index[g.away] for g in pending], dtype=int)
    is_conf = np.array([g.is_conference_game for g in pending], dtype=bool)
    draws = rng.random((n_sims, len(pending))) < p_home if pending else np.zeros((n_sims, 0), dtype=bool)
    pending_title: dict[str, int] = {}  # conference → index of its scheduled, unplayed title game
    for i, g in enumerate(pending):
        if g.is_conference_championship and g.home_conference:
            pending_title[str(g.home_conference)] = i

    n = len(all_names)
    arange = np.arange(n)
    left = np.zeros(n, dtype=int)
    if pending:
        np.add.at(left, home_idx, 1)
        np.add.at(left, away_idx, 1)
    max_games = int((base_wins + base_losses + left).max(initial=0)) + 1  # +1: a projected title game
    stride = max_games + 1
    tally = {
        k: np.zeros(n)
        for k in ("p_champion", "p_playoff", "p_bye", "seed_sum", "p_quarter", "p_semi", "p_final", "p_title")
    }
    champ_counts: dict[str, Counter] = {conf: Counter() for conf in members}
    tg_pairs: dict[str, Counter] = {conf: Counter() for conf in members}  # (home, away, neutral) → count
    tg_count = np.zeros(n)
    wins_hist = np.zeros((n, stride))
    joint = np.zeros((n, stride * stride))  # (wins, losses) pairs, for the most likely record
    games_sum = np.zeros(n)
    undefeated = np.zeros(n)
    seed_hist = np.zeros((n, fmt.field_size + 1))
    score_sum = np.zeros(n)
    conf_game_ids = {
        conf: [i for i, g in enumerate(pending) if is_conf[i] and conf_of.get(g.home) == conf] for conf in members
    }

    for s in range(n_sims):
        row = draws[s]
        wins = base_wins.copy()
        losses = base_losses.copy()
        if pending:
            np.add.at(wins, np.where(row, home_idx, away_idx), 1)
            np.add.at(losses, np.where(row, away_idx, home_idx), 1)
        champions: dict[str, str] = dict(decided)
        matchups: dict[str, tuple[str, str, bool]] = {}  # this simulation's title games
        for conf, schools in members.items():
            if conf in champions:
                continue
            rec = {t: list(conf_rec[t]) for t in schools}
            h2h = dict(h2h_base)
            for i in conf_game_ids[conf]:
                w, loser = (pending[i].home, pending[i].away) if row[i] else (pending[i].away, pending[i].home)
                if w in rec and loser in rec:
                    rec[w][0] += 1
                    rec[loser][1] += 1
                    h2h[(w, loser)] = h2h.get((w, loser), 0) + 1
            if conf in pending_title:  # the title game is on the schedule with its participants
                g = pending[pending_title[conf]]
                champions[conf] = g.home if row[pending_title[conf]] else g.away
                matchups[conf] = (g.home, g.away, g.neutral_site)
                continue
            overall = {t: Record(int(wins[index[t]]), int(losses[index[t]])) for t in schools}  # breaks ties
            order = _order_standings(schools, {t: tuple(rec[t]) for t in schools}, overall, result.ratings, h2h)
            if len(order) == 1:
                champions[conf] = order[0]
                continue
            a, b = order[0], order[1]
            neutral = conf in fmt.neutral_championships
            # At a neutral site the pairing is the same whichever side led the standings, so it is
            # keyed with the better-rated team first; a hosted game keeps the leader as the host.
            matchups[conf] = (
                (
                    max(a, b, key=lambda t: (result.ratings.get(t, 0.0), t)),
                    min(a, b, key=lambda t: (result.ratings.get(t, 0.0), t)),
                    True,
                )
                if neutral
                else (a, b, False)
            )
            if rng.random() < win_prob(a, b, neutral):
                champions[conf] = a
                wins[index[a]] += 1
                losses[index[b]] += 1
            else:
                champions[conf] = b
                wins[index[b]] += 1
                losses[index[a]] += 1
        for conf, matchup in matchups.items():
            tg_pairs[conf][matchup] += 1
            tg_count[index[matchup[0]]] += 1
            tg_count[index[matchup[1]]] += 1
        for conf, team in champions.items():
            champ_counts[conf][team] += 1
            tally["p_champion"][index[team]] += 1
        np.add.at(wins_hist, (arange, wins), 1)
        np.add.at(joint, (arange, wins * stride + losses), 1)
        games_sum += wins + losses
        undefeated += losses == 0
        champ_set = set(champions.values())
        scores_now = committee_scores(
            names,
            result.ratings,
            {t: int(losses[index[t]]) for t in names},
            ppoints=ppoints,
            champions=champ_set,
            weights=weights,
        )
        for t, v in scores_now.items():
            score_sum[index[t]] += v
        ranking = sorted(scores_now, key=lambda t: (-scores_now[t], t))
        field = select_field(ranking, champions, conf_of, fmt)
        _, probs = _bracket_probs(field, win_prob)
        for sd in field:
            i = index[sd.team]
            tally["p_playoff"][i] += 1
            tally["seed_sum"][i] += sd.seed
            seed_hist[i, sd.seed] += 1
            if sd.seed <= fmt.byes:
                tally["p_bye"][i] += 1
            for k in ("quarter", "semi", "final", "title"):
                tally[f"p_{k}"][i] += probs[k][sd.team]
    for conf, g in title_games.items():  # a played title game: both sides were in it, every time
        if conf in decided:
            for t in (g.home, g.away):
                if t in index:
                    tg_count[index[t]] += n_sims

    k_wins = np.arange(stride)
    rows = []
    for t in names:
        i = index[t]
        made = tally["p_playoff"][i]
        proj_wins = float(wins_hist[i] @ k_wins) / n_sims
        proj_games = games_sum[i] / n_sims
        mode_w, mode_l = divmod(int(joint[i].argmax()), stride)
        rows.append(
            {
                "team": t,
                "conference": conf_of.get(t) or "Independent",
                "record": str(base[t]),
                "proj_wins": proj_wins,
                "proj_losses": proj_games - proj_wins,
                "proj_record": f"{proj_wins:.1f}-{proj_games - proj_wins:.1f}",
                "mode_record": f"{mode_w}-{mode_l}",
                "p_mode_record": joint[i].max() / n_sims,
                "p_undefeated": undefeated[i] / n_sims,
                "p_title_game": min(tg_count[i] / n_sims, 1.0),
                "p_champion": tally["p_champion"][i] / n_sims,
                "p_playoff": made / n_sims,
                "p_bye": tally["p_bye"][i] / n_sims,
                "mean_seed": (tally["seed_sum"][i] / made) if made else np.nan,
                "p_quarter": tally["p_quarter"][i] / n_sims,
                "p_semi": tally["p_semi"][i] / n_sims,
                "p_final": tally["p_final"][i] / n_sims,
                "p_title": tally["p_title"][i] / n_sims,
                "committee_score": score_sum[i] / n_sims,
            }
        )
    table = pd.DataFrame(rows).sort_values(["p_title", "p_playoff"], ascending=False).reset_index(drop=True)

    appear: dict[str, Counter] = {conf: Counter() for conf in members}
    for conf, pairs in tg_pairs.items():
        for (h, a, _), c in pairs.items():
            appear[conf][h] += c
            appear[conf][a] += c
    for conf in decided:
        g = title_games[conf]
        appear[conf][g.home] = appear[conf][g.away] = n_sims
    champ_rows = [
        {
            "conference": conf,
            "team": team,
            "p_title_game": min(appear[conf][team] / n_sims, 1.0),
            "p_champion": champ_counts[conf][team] / n_sims,
        }
        for conf in members
        for team in sorted(set(appear[conf]) | set(champ_counts[conf]), key=lambda t: (-champ_counts[conf][t], t))
    ]
    champions_df = pd.DataFrame(champ_rows, columns=["conference", "team", "p_title_game", "p_champion"])

    tg_cols = [
        "conference", "home", "away", "neutral", "p_matchup", "p_home", "favorite", "p_favorite",
        "proj_home", "proj_away", "proj_score", "decided",
    ]  # fmt: skip
    tg_rows = []
    for conf in members:
        if conf in decided:
            g = title_games[conf]
            tg_rows.append(
                {
                    "conference": conf,
                    "home": g.home,
                    "away": g.away,
                    "neutral": g.neutral_site,
                    "p_matchup": 1.0,
                    "p_home": 1.0 if g.winner == g.home else 0.0,
                    "favorite": g.winner,
                    "p_favorite": 1.0,
                    "proj_home": g.home_points,
                    "proj_away": g.away_points,
                    "proj_score": f"{g.home_points}-{g.away_points}",
                    "decided": True,
                }
            )
            continue
        total = sum(tg_pairs[conf].values())
        for (h, a, neutral), c in tg_pairs[conf].most_common():
            p_h = win_prob(h, a, neutral)
            r: dict = {
                "conference": conf,
                "home": h,
                "away": a,
                "neutral": neutral,
                "p_matchup": c / total,
                "p_home": p_h,
                "favorite": h if p_h >= 0.5 else a,
                "p_favorite": max(p_h, 1.0 - p_h),
                "proj_home": None,
                "proj_away": None,
                "proj_score": None,
                "decided": False,
            }
            if scorer is not None:
                ps = scorer(h, a, neutral)
                r["proj_home"], r["proj_away"] = ps.rounded()
                r["proj_score"] = ps.text
            tg_rows.append(r)
    title_games_df = pd.DataFrame(tg_rows, columns=tg_cols)

    ever = [t for t in names if tally["p_playoff"][index[t]] > 0]
    seeds_df = pd.DataFrame(
        [
            {
                "team": t,
                "conference": conf_of.get(t) or "Independent",
                "p_playoff": tally["p_playoff"][index[t]] / n_sims,
                **{str(k): seed_hist[index[t], k] / n_sims for k in range(1, fmt.field_size + 1)},
            }
            for t in ever
        ],
        columns=["team", "conference", "p_playoff", *(str(k) for k in range(1, fmt.field_size + 1))],
    ).sort_values("p_playoff", ascending=False, ignore_index=True)
    used = int(np.nonzero(wins_hist.sum(axis=0))[0].max(initial=0))
    wins_df = pd.DataFrame(
        [
            {
                "team": t,
                "conference": conf_of.get(t) or "Independent",
                **{str(k): wins_hist[index[t], k] / n_sims for k in range(used + 1)},
            }
            for t in names
        ],
        columns=["team", "conference", *(str(k) for k in range(used + 1))],
    )

    modal = {conf: counter.most_common(1)[0][0] for conf, counter in champ_counts.items() if counter}
    mean_ranking = sorted(names, key=lambda t: (-score_sum[index[t]], t))
    projected = select_field(mean_ranking, modal, conf_of, fmt)
    return PlayoffSimulation(
        n_sims,
        fmt,
        table,
        champions_df,
        title_games_df,
        seeds_df,
        wins_df,
        projected,
        bracket_odds(projected, win_prob, scorer),
    )


# --- the games still to play -----------------------------------------------------------------------

GAME_PROJECTION_COLUMNS: tuple[str, ...] = (
    "id", "season", "week", "date", "home", "away", "home_name", "away_name", "home_conference", "away_conference",
    "neutral", "conference_game", "title_game", "postseason", "p_home", "margin", "favorite", "p_favorite", "line",
    "proj_home", "proj_away", "proj_total", "proj_score",
)  # fmt: skip


def game_projections(
    games: Sequence[Game],
    result: RatingResult,
    calibrator: MarginCalibrator,
    *,
    scores: ScoreModel | None = None,
) -> pd.DataFrame:
    """One row per unplayed game between rated teams, in schedule order.

    ``p_home`` and ``margin`` come from the calibrated ratings; ``line`` is the market-style
    spread from the favourite's side; with a :class:`~cfbratings.scores.ScoreModel` the
    ``proj_*`` columns hold the projected score (``proj_score`` in home-away order).
    """
    rated = result.ratings
    rows = []
    pending = sorted(
        (g for g in games if not g.has_score and g.home in rated and g.away in rated), key=lambda g: g.order_key
    )
    for g in pending:
        margin = calibrator.predict_margin(result.get(g.home), result.get(g.away), g.neutral_site)
        p = calibrator.win_prob_from_margin(margin)
        fav = g.home if margin >= 0 else g.away
        row: dict = {
            "id": g.id,
            "season": g.season,
            "week": g.week,
            "date": g.start_date,
            "home": g.home,
            "away": g.away,
            "home_name": g.home_display,
            "away_name": g.away_display,
            "home_conference": g.home_conference,
            "away_conference": g.away_conference,
            "neutral": g.neutral_site,
            "conference_game": g.is_conference_game,
            "title_game": g.is_conference_championship,
            "postseason": g.season_type == "postseason",
            "p_home": p,
            "margin": margin,
            "favorite": g.display_name(fav),
            "p_favorite": max(p, 1.0 - p),
            "line": f"{g.display_name(fav)} -{abs(margin):.1f}",
            "proj_home": None,
            "proj_away": None,
            "proj_total": None,
            "proj_score": None,
        }
        if scores is not None:
            ps = scores.project(g.home, g.away, margin, neutral=g.neutral_site)
            row["proj_home"], row["proj_away"] = ps.rounded()
            row["proj_total"] = ps.total
            row["proj_score"] = ps.text
        rows.append(row)
    return pd.DataFrame(rows, columns=list(GAME_PROJECTION_COLUMNS))


# --- playing a bracket out ---------------------------------------------------------------------------


def custom_field(
    order: Sequence[str], conference_of: Mapping[str, str | None], champions: Sequence[str] | set[str] = ()
) -> list[Seed]:
    """A hand-picked field: ``order`` is the teams in seed order (at most twelve)."""
    if len(order) > 12:
        raise ValueError("the bracket holds at most 12 teams")
    if len(set(order)) != len(order):
        raise ValueError("a team cannot be seeded twice")
    champs = set(champions)
    return [
        Seed(
            seed=i + 1,
            team=t,
            conference=str(conference_of.get(t) or "Independent"),
            champion=t in champs,
            bid="custom",
            rank=i + 1,
        )
        for i, t in enumerate(order)
    ]


@dataclass
class PlayedGame:
    round: str
    home: Seed  # the higher seed (its home in the first round; a label at a neutral site)
    away: Seed
    neutral: bool
    p_home: float
    winner: str
    score: ProjectedScore | None = None  # projected (chalk) or sampled (simulation) score


@dataclass
class BracketRun:
    field: list[Seed]
    games: list[PlayedGame]  # first round first
    champion: Seed
    sampled: bool  # False: the favourite won every game ("chalk"); True: one random play-through


def play_bracket(
    field: Sequence[Seed],
    result: RatingResult,
    calibrator: MarginCalibrator,
    *,
    scores: ScoreModel | None = None,
    rng: np.random.Generator | None = None,
) -> BracketRun:
    """Play the bracket once.

    Without ``rng`` the favourite wins every game and each game carries its projected score
    (the chalk bracket). With ``rng`` every game is decided by a draw: a sampled score when a
    score model is given (its margin decides the winner), a Bernoulli draw otherwise.
    """
    if not field:
        raise ValueError("an empty field has no bracket")
    tree = _bracket_tree(field)
    games: list[PlayedGame] = []

    def play(node: _Node) -> Seed:
        if node[0] == "seed":
            return node[1]
        _, rnd, left, right, neutral = node
        a, b = play(left), play(right)
        home, away = (a, b) if a.seed < b.seed else (b, a)
        margin = calibrator.predict_margin(result.get(home.team), result.get(away.team), neutral)
        p = calibrator.win_prob_from_margin(margin)
        score: ProjectedScore | None = None
        if rng is None:
            home_wins = margin >= 0
            if scores is not None:
                score = scores.project(home.team, away.team, margin, neutral=neutral)
        elif scores is not None:
            score = scores.sample(home.team, away.team, margin, calibrator.sigma, rng, neutral=neutral)
            home_wins = score.margin > 0
        else:
            home_wins = bool(rng.random() < p)
        winner = home if home_wins else away
        games.append(PlayedGame(rnd, home, away, neutral, p, winner.team, score))
        return winner

    champion = play(tree)
    order = {r: i for i, r in enumerate(ROUNDS)}
    games.sort(key=lambda g: (order[g.round], g.home.seed))
    return BracketRun(list(field), games, champion, rng is not None)


@dataclass
class BracketSimulation:
    n_sims: int
    table: pd.DataFrame  # seed, team, conference, p_quarter, p_semi, p_final, p_title (frequencies)
    champions: Counter  # team → titles won

    def title_favourite(self) -> tuple[str, float]:
        team, c = self.champions.most_common(1)[0]
        return team, c / self.n_sims


def simulate_bracket(
    field: Sequence[Seed], win_prob: WinProb, *, n_sims: int = 5000, seed: int = 0
) -> BracketSimulation:
    """Monte Carlo of a fixed bracket: ``n_sims`` play-throughs, every game a draw.

    :func:`bracket_odds` gives these numbers exactly; the simulation is the same bracket seen
    game by game, and it converges to the exact odds as ``n_sims`` grows.
    """
    if not field:
        raise ValueError("an empty field has no bracket")
    tree = _bracket_tree(field)
    rng = np.random.default_rng(seed)
    cache: dict[tuple[str, str, bool], float] = {}
    reach: dict[str, Counter] = {r: Counter() for r in ROUNDS}
    titles: Counter = Counter()

    def wp(a: str, b: str, neutral: bool) -> float:
        key = (a, b, neutral)
        if key not in cache:
            cache[key] = win_prob(a, b, neutral)
        return cache[key]

    def play(node: _Node) -> str:
        if node[0] == "seed":
            return node[1].team
        _, rnd, left, right, neutral = node
        a, b = play(left), play(right)
        reach[rnd][a] += 1
        reach[rnd][b] += 1
        return a if rng.random() < wp(a, b, neutral) else b

    for _ in range(n_sims):
        titles[play(tree)] += 1
    rows = [
        {
            "seed": s.seed,
            "team": s.team,
            "conference": s.conference,
            "p_quarter": reach["quarter"][s.team] / n_sims,
            "p_semi": reach["semi"][s.team] / n_sims,
            "p_final": reach["final"][s.team] / n_sims,
            "p_title": titles[s.team] / n_sims,
        }
        for s in sorted(field, key=lambda s: s.seed)
    ]
    return BracketSimulation(n_sims, pd.DataFrame(rows), titles)


def bracket_lines(odds: BracketOdds) -> list[str]:
    """Plain-text bracket: one line per game, first round first."""
    lines = []
    for rnd in ROUNDS:
        games = [g for g in odds.games if g.round == rnd]
        if not games:
            continue
        lines.append(f"{ROUND_LABELS[rnd]}:")
        for g in games:
            site = "neutral" if g.neutral else f"at {g.home.team}"
            score = f" — proj. {g.score.text_by_favorite}" if g.score is not None else ""
            lines.append(
                f"  ({g.home.seed}) {g.home.team} vs ({g.away.seed}) {g.away.team} — {site} — "
                f"{g.home.team} {g.p_home:.0%}{score}"
            )
    return lines


def played_bracket_lines(run: BracketRun) -> list[str]:
    """Plain-text bracket of one played-out bracket (chalk or a single simulation)."""
    lines = []
    for rnd in ROUNDS:
        games = [g for g in run.games if g.round == rnd]
        if not games:
            continue
        lines.append(f"{ROUND_LABELS[rnd]}:")
        for g in games:
            site = "neutral" if g.neutral else f"at {g.home.team}"
            score = ""
            if g.score is not None:  # the winner's points first
                h, a = g.score.rounded()
                score = f" {h}-{a}" if g.winner == g.home.team else f" {a}-{h}"
            lines.append(
                f"  ({g.home.seed}) {g.home.team} vs ({g.away.seed}) {g.away.team} — {site} — "
                f"{g.winner} wins{score} ({g.p_home:.0%} home)"
            )
    lines.append(f"Champion: ({run.champion.seed}) {run.champion.team}")
    return lines


__all__ = [
    "FORMATS",
    "GROUP_SIX",
    "POWER_FOUR",
    "ROUNDS",
    "ROUND_LABELS",
    "BracketGame",
    "BracketOdds",
    "BracketRun",
    "BracketSimulation",
    "CFPFormat",
    "Champion",
    "PlayedGame",
    "PlayoffPicture",
    "PlayoffSimulation",
    "Scorer",
    "Seed",
    "SeedingWeights",
    "Standing",
    "bracket_lines",
    "bracket_odds",
    "calibrated_scorer",
    "calibrated_win_prob",
    "championship_games",
    "committee_ranking",
    "committee_scores",
    "conference_champions",
    "conference_members",
    "conference_standings",
    "custom_field",
    "format_for",
    "game_projections",
    "play_bracket",
    "played_bracket_lines",
    "playoff_picture",
    "select_field",
    "simulate_bracket",
    "simulate_playoff",
    "standings_table",
]
