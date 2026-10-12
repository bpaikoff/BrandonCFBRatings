"""College Football Ratings dashboard.

Run with ``streamlit run apps/streamlit_app.py``. Works without an API key in demo mode
(synthetic seasons with known ground truth); set ``CFBD_API_KEY`` to use real data.
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cfbratings import __version__, settings
from cfbratings.analytics import (
    conference_strength,
    conference_table,
    freshness_label,
    independents_table,
    kickoff_label,
    records,
    schedule_table,
    team_table,
)
from cfbratings.backtest import compare_methods
from cfbratings.cfp import (
    FORMATS,
    ROUND_LABELS,
    ROUNDS,
    BracketOdds,
    SeedingWeights,
    format_for,
    playoff_picture,
    simulate_playoff,
    standings_table,
)
from cfbratings.data import (
    CFBDClient,
    Game,
    Team,
    contrast_color,
    load_seasons,
    max_completed_week,
    synthetic_seasons,
)
from cfbratings.data.cfbd import CFBDError
from cfbratings.data.schema import conference_map as _conference_map
from cfbratings.data.schema import is_pool_team
from cfbratings.models import MODEL_LABELS, MODEL_NAMES, make_model
from cfbratings.ppoints import ppoints as _ppoints
from cfbratings.predict import MarginCalibrator
from cfbratings.priors import WarmStart, warm_start
from cfbratings.projections import TeamProjection, project_team, projection_table
from cfbratings.snapshots import RANK_UNCHANGED, rank_changes, weekly_snapshots
from cfbratings.viz import (
    THEMES,
    calibration_chart,
    conference_dots,
    contrast_ratio,
    rating_bar,
    rating_sos_scatter,
    resolve_team_color,
    trajectory_chart,
    weekly_metric_chart,
    win_distribution_chart,
)

st.set_page_config(page_title="CFB Ratings", page_icon="🏈", layout="wide")

METHODOLOGY = ROOT / "docs" / "METHODOLOGY.md"
DEMO_WEEKS = 13
SCALE_LABELS = {
    "points": "Rating (pts)",
    "elo": "Rating (Elo)",
    "prob": "Rating (0-1)",
    "logit": "Rating (log-odds)",
    "z": "Rating (z-score)",
}
# WCAG minimum for large text; a team colour below this on the page surface is not used for the name.
MIN_HEADING_CONTRAST = 3.0
NO_VALUE = RANK_UNCHANGED  # the same en dash on a stat tile that has nothing to show yet

# Every custom block (stat tiles, team header, matchup chips) is a wrapping flex row so that a
# 420 px phone gets stacked tiles instead of a horizontal scrollbar.
_CSS = """
<style>
.kpi-row{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 16px}
.kpi{flex:1 1 170px;min-width:150px;max-width:100%;padding:12px 14px;border-radius:10px;
     background:var(--secondary-background-color);color:var(--text-color)}
.kpi-label{font-size:.8rem;opacity:.7}
.kpi-value{font-size:1.65rem;font-weight:600;line-height:1.2;overflow-wrap:anywhere;
           display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.kpi-note{font-size:.8rem;opacity:.75;margin-top:4px}
.kpi-logo{width:36px;height:36px;object-fit:contain;flex:0 0 auto}
.team-strip{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin:2px 0 14px;color:var(--text-color)}
.team-logo{width:72px;height:72px;object-fit:contain;flex:0 0 auto}
.team-name{font-size:1.9rem;font-weight:700;line-height:1.15;overflow-wrap:anywhere}
.team-sub{font-size:.95rem;opacity:.75;margin-top:2px}
.chips{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:2px 0 14px;color:var(--text-color)}
.chip{display:inline-flex;align-items:center;gap:8px;padding:5px 14px 5px 6px;border-radius:999px;
      font-weight:600;max-width:100%;overflow-wrap:anywhere;border:1px solid rgba(128,128,128,.35)}
.chip img{width:32px;height:32px;padding:3px;box-sizing:border-box;border-radius:50%;object-fit:contain;
          flex:0 0 auto}
.chips-vs{opacity:.7;font-size:.9rem}
.bracket{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 14px;color:var(--text-color)}
.round{flex:1 1 190px;min-width:180px;max-width:100%}
.round-title{font-weight:600;font-size:.9rem;margin-bottom:4px}
.round-note{font-size:.8rem;opacity:.7;margin:2px 0 8px}
.game{border:1px solid rgba(128,128,128,.35);border-radius:8px;padding:6px 8px;margin:0 0 8px;
      background:var(--secondary-background-color)}
.game .t{display:flex;align-items:center;gap:6px;overflow-wrap:anywhere}
.game .seed{opacity:.6;font-size:.8rem;min-width:1.6em}
.game img{width:20px;height:20px;object-fit:contain;flex:0 0 auto}
.game .p{margin-left:auto;font-size:.85rem;opacity:.8;flex:0 0 auto}
.game .site{font-size:.75rem;opacity:.6;margin-top:2px}
</style>
"""


# --- small UI helpers ----------------------------------------------------------------------


def current_theme() -> str:
    """``"light"`` or ``"dark"``: the theme the page is actually rendered with.

    A ``[theme] base`` pinned in ``.streamlit/config.toml`` (or ``STREAMLIT_THEME_BASE``) is what
    the page is drawn with whatever the browser prefers, and it wins here. ``st.context.theme.type``
    is only consulted when no base is pinned, because it reports the browser's colour-scheme
    preference rather than the page: on a pinned-light page a dark-mode browser makes it say
    "dark", which is how the charts once came out with white titles on a white page. With
    neither (older Streamlit, no browser session, ``AppTest``) the answer is light. Viewers
    cannot switch themes behind the app's back either: ``toolbarMode = "minimal"`` in the config
    hides Streamlit's settings menu, so a pinned base is the only theme source.
    """
    base = st.get_option("theme.base")
    if base in ("light", "dark"):
        return base
    try:
        kind = st.context.theme.type
    except Exception:
        kind = None
    return "dark" if kind == "dark" else "light"


def chart(fig: go.Figure, theme: str) -> None:
    """Draw a figure on its own theme surface, so chart text stays legible if the page ever disagrees."""
    surface = THEMES[theme]["surface"]
    fig.update_layout(paper_bgcolor=surface, plot_bgcolor=surface)
    st.plotly_chart(fig, width="stretch")


def team_logo(t: Team | None, theme: str) -> str | None:
    if t is None:
        return None
    return (t.logo_dark or t.logo) if theme == "dark" else t.logo


def heading_color(t: Team | None, theme: str) -> str | None:
    """The team colour for its name, or ``None`` (inherit) when neither brand colour is legible."""
    if t is None:
        return None
    surface = THEMES[theme]["surface"]
    for c in (t.color, t.alt_color):
        if c and contrast_ratio(c, surface) >= MIN_HEADING_CONTRAST:
            return c
    return None


def _img(src: str | None, cls: str, style: str | None = None) -> str:
    if not src:
        return ""
    extra = f' style="{html.escape(style)}"' if style else ""
    return f'<img class="{cls}" src="{html.escape(src)}"{extra} alt="">'


def kpi_row(items: list[tuple[str, str, str | None]], extra_html: str = "") -> None:
    """Responsive stat tiles: wrap at narrow widths instead of truncating like ``st.metric``.

    ``extra_html`` is a pre-rendered tile (see :func:`next_game_tile`) appended to the row.
    """
    tiles = "".join(
        '<div class="kpi"><div class="kpi-label">{}</div><div class="kpi-value">{}</div>{}</div>'.format(
            html.escape(label),
            html.escape(value),
            f'<div class="kpi-note">{html.escape(note)}</div>' if note else "",
        )
        for label, value, note in items
    )
    st.markdown(f'<div class="kpi-row">{tiles}{extra_html}</div>', unsafe_allow_html=True)


def next_game_tile(team: str, game: Game, p_win: float, logo_map: Mapping[str, str | None]) -> str:
    """A stat tile for the next scheduled game: opponent (with logo), site, date and win probability."""
    opp = game.opponent(team)
    opp_name = game.display_name(opp)
    site = "Neutral" if game.neutral_site else ("Home" if game.home == team else "Away")
    when = "Bowl" if game.season_type == "postseason" else f"Week {game.week}"
    label = " · ".join(x for x in ("Next game", when, kickoff_label(game.start_date)) if x)
    value = "{}<span>{} {}{}</span>".format(
        _img(logo_map.get(opp_name), "kpi-logo"),
        "at" if site == "Away" else "vs",
        html.escape(opp_name + (" (FCS)" if is_pool_team(opp) else "")),
        " (neutral)" if site == "Neutral" else "",
    )
    note = f"win probability {p_win:.0%}"
    return (
        f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>'
    )


def team_header(t: Team | None, name: str, conference: str, theme: str) -> None:
    """Logo, team name in the team colour and conference, wrapping on narrow screens."""
    color = heading_color(t, theme)
    style = f' style="color:{html.escape(color)}"' if color else ""
    st.markdown(
        f'<div class="team-strip">{_img(team_logo(t, theme), "team-logo")}<div>'
        f'<div class="team-name"{style}>{html.escape(name)}</div>'
        f'<div class="team-sub">{html.escape(conference)}</div></div></div>',
        unsafe_allow_html=True,
    )


def team_chip(name: str, t: Team | None, theme: str, logo_map: Mapping[str, str | None]) -> str:
    """A pill in the team colour with its logo and name.

    The fill goes through :func:`resolve_team_color`, so a colour that would vanish against the
    page (Iowa's black on the dark page) is swapped for the alternate colour, and the text is
    black or white by :func:`contrast_color`. The logo is the page-theme variant, which is drawn
    for the page surface, not for the team colour (Indiana's crimson mark disappears on its own
    crimson pill), so it sits on a small disc in the page surface colour inside the pill.
    """
    bg = resolve_team_color(t.color if t else None, t.alt_color if t else None, theme)
    style = f"background:{html.escape(bg)};color:{contrast_color(bg)}"
    logo = _img(logo_map.get(name), "", style=f"background:{THEMES[theme]['surface']}")
    return f'<span class="chip" style="{style}">{logo}{html.escape(name)}</span>'


def matchup_header(home: str, away: str, neutral: bool, teams: Mapping[str, Team], theme: str, logo_map) -> None:
    st.markdown(
        '<div class="chips">{}<span class="chips-vs">vs</span>{}<span class="chips-vs">{}</span></div>'.format(
            team_chip(home, teams.get(home), theme, logo_map),
            team_chip(away, teams.get(away), theme, logo_map),
            "neutral site" if neutral else f"at {home}",
        ),
        unsafe_allow_html=True,
    )


def bracket_html(odds: BracketOdds, logo_map: Mapping[str, str | None], byes: int) -> str:
    """The bracket as four columns of game cards; later rounds show their most likely matchup."""

    def side(seed, p: float | None) -> str:
        logo = _img(logo_map.get(seed.team), "")
        prob = f'<span class="p">{p:.0%}</span>' if p is not None else ""
        return f'<div class="t"><span class="seed">{seed.seed}</span>{logo}<span>{html.escape(seed.team)}</span>{prob}</div>'

    cols = []
    for rnd in ROUNDS:
        games = [g for g in odds.games if g.round == rnd]
        if not games:
            continue
        cards = []
        for g in games:
            site = "neutral site" if g.neutral else f"at {html.escape(g.home.team)}"
            cards.append(
                f'<div class="game">{side(g.home, g.p_home)}{side(g.away, 1 - g.p_home)}'
                f'<div class="site">{site}</div></div>'
            )
        note = f'<div class="round-note">Seeds 1-{byes} have a bye</div>' if rnd == "first" and byes else ""
        if rnd != "first":
            note = '<div class="round-note">most likely matchup</div>'
        cols.append(
            f'<div class="round"><div class="round-title">{ROUND_LABELS[rnd]}</div>{note}{"".join(cards)}</div>'
        )
    return f'<div class="bracket">{"".join(cols)}</div>'


def table_height(n_rows: int, *, cap: int = 900) -> int:
    return int(min(cap, 35 * (n_rows + 1) + 3))


def percent_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Display copy with 0-1 probabilities scaled to 0-100 so column formats can say ``%.0f%%``."""
    return df.assign(**{c: df[c] * 100.0 for c in cols if c in df.columns})


def pct_bar(label: str, decimals: int = 0, **kw) -> st.column_config.ProgressColumn:
    return st.column_config.ProgressColumn(label, format=f"%.{decimals}f%%", min_value=0, max_value=100, **kw)


def pct_num(label: str, decimals: int = 0, **kw) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format=f"%.{decimals}f%%", **kw)


def logo_col(label: str = "") -> st.column_config.ImageColumn:
    return st.column_config.ImageColumn(label, width=44)


def fmt_blank(series: pd.Series, spec: str) -> pd.Series:
    """``spec.format(v)`` per value with a blank for a missing one.

    ``st.dataframe`` prints a missing value in a mixed column as the text "None", so columns that
    are empty for some rows (no next game, no rank for a pooled opponent) are pre-formatted as text.
    """
    return series.map(lambda v: "" if pd.isna(v) else spec.format(v))


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Without the columns that hold no value at all (a Vegas line that was never loaded)."""
    return df.dropna(axis="columns", how="all")


def text_when_gappy(df: pd.DataFrame, config: dict, specs: Mapping[str, str]) -> tuple[pd.DataFrame, dict]:
    """Number columns with a missing value become right-aligned text with a blank for the gap.

    ``specs`` maps a column to its ``str.format`` spec. A column with every value present keeps
    its numeric config (numeric sorting, the column's own number format); one with a gap would
    otherwise print "None" there, so it is pre-formatted and its config swapped for a
    ``TextColumn`` with the same label, width and help.
    """
    config = dict(config)
    for col, spec in specs.items():
        if col in df.columns and df[col].isna().any():
            df = df.assign(**{col: fmt_blank(df[col], spec)})
            c = config.get(col, {})
            config[col] = st.column_config.TextColumn(
                c.get("label"), width=c.get("width"), help=c.get("help"), alignment="right"
            )
    return df, config


SORT_LABELS = {
    "rating": "Rating",
    "ppoints": "PPoints",
    "sos": "Strength of schedule",
    "momentum": "Momentum",
    "wins": "Wins",
}
# The same columns as a phrase for a chart title: "Top 16 by PPoints", "... by strength of schedule".
SORT_PHRASES = {"rating": "rating", "ppoints": "PPoints", "sos": "strength of schedule", "momentum": "momentum"}
METRIC_LABELS = {"accuracy": "Accuracy", "brier": "Brier", "log_loss": "Log loss", "mae": "MAE (pts)"}
SEASON_TYPE_LABELS = {"both": "regular + postseason", "regular": "regular season", "postseason": "postseason"}
# How the calibrator was fitted, for the KPI tile; any other source reads as its own words.
CAL_SOURCE_LABELS = {"cross_val": "cross-validated", "in_sample": "in-sample", "out_of_sample": "out-of-sample"}
# Backtest metric columns, shared by the headline table and the per-season tables.
BT_PERCENT_COLS = ["accuracy", "ats_accuracy"]
BT_COLUMNS = {
    "method": "Method",
    "season": st.column_config.NumberColumn("Season", format="%d", width="small"),
    "n": st.column_config.NumberColumn("Games", width="small"),
    "accuracy": pct_num("Accuracy", decimals=1),
    "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
    "log_loss": st.column_config.NumberColumn("Log loss", format="%.4f"),
    "mae": st.column_config.NumberColumn("MAE (pts)", format="%.2f"),
    "rmse": st.column_config.NumberColumn("RMSE (pts)", format="%.2f"),
    "vegas_n": st.column_config.NumberColumn("Vegas games", width="small"),
    "vegas_mae": st.column_config.NumberColumn("Vegas MAE (pts)", format="%.2f"),
    "ats_accuracy": pct_num("ATS accuracy", decimals=1, help="Picks against the closing spread"),
    "seasons": "Seasons",
    "weeks": "Weeks",
}
# The out-of-sample predictions as a reader sees them; the CSV download keeps every raw column.
OOS_COLUMNS = {
    "method": "Method",
    "season": st.column_config.NumberColumn("Season", format="%d", width="small"),
    "week": st.column_config.NumberColumn("Week", width="small"),
    "home": st.column_config.TextColumn("Home", width="medium"),
    "away": st.column_config.TextColumn("Away", width="medium"),
    "neutral": st.column_config.TextColumn("Neutral", width="small", help="n = neutral site"),
    "pred_margin": st.column_config.NumberColumn("Predicted margin", format="%+.1f", help="Home margin"),
    "p_home": pct_num("Home win %"),
    "actual_margin": st.column_config.NumberColumn("Actual margin", format="%+d"),
    "result": st.column_config.TextColumn("Result", width="small", help="Home team's result"),
    "pick": st.column_config.TextColumn("Pick", width="small", help="Did the favourite win?"),
}


def backtest_table(df: pd.DataFrame) -> None:
    """A backtest summary table without the columns that have nothing in them.

    The Vegas and against-the-spread metrics only exist when closing lines were loaded, which the
    dashboard never does, so those columns (and their configs) are dropped when absent or empty.
    """
    df = drop_empty_columns(df)
    st.dataframe(
        percent_cols(df, BT_PERCENT_COLS),
        hide_index=True,
        width="stretch",
        height=table_height(len(df)),
        column_config={k: v for k, v in BT_COLUMNS.items() if k in df.columns},
    )


def oos_predictions_view(preds: pd.DataFrame) -> pd.DataFrame:
    """The curated out-of-sample predictions: one row per game and method, formatted for reading.

    Drops the prior/rating/Vegas working columns, writes the neutral flag as ``"n"`` or blank, the
    home result as W/L/T and the pick as a tick or cross, and leaves out any column that is empty
    for every row (``vegas_margin`` when no lines were loaded).
    """
    margin = preds["actual_margin"]
    view = preds.assign(
        neutral=preds["neutral"].map(lambda v: "n" if bool(v) else ""),
        result=margin.map(lambda m: "" if pd.isna(m) else "W" if m > 0 else "L" if m < 0 else "T"),
        pick=preds["correct"].map(lambda c: "" if pd.isna(c) else "✓" if c else "✗"),
    )
    view = view.reindex(columns=[c for c in OOS_COLUMNS if c in view.columns])
    return drop_empty_columns(view)


# --- cached computations ---------------------------------------------------------------------


@st.cache_data(show_spinner="Loading season…")
def load_data(
    demo: bool,
    year: int,
    warmup: int,
    season_type: str,
    fcs: str,
    refresh_token: int,
    force_refresh: bool,
    seed: int,
    demo_weeks: int,
):
    years = list(range(year - warmup, year + 1))
    span = f"{years[0]}-{years[-1]}" if warmup else str(year)
    if demo:
        chain = synthetic_seasons(years, seed=seed, played_through_last=demo_weeks if demo_weeks < DEMO_WEEKS else None)
        games = [g for season in chain for g in season.games]
        return (
            [t.to_dict() for t in chain[-1].teams],
            [g.to_dict() for g in games],
            f"Synthetic season {span} (seed {seed})",
        )
    teams, games = load_seasons(years, season_type, force_refresh=force_refresh, fcs=fcs)
    label = (
        f"CollegeFootballData · {span} · {SEASON_TYPE_LABELS.get(season_type, season_type)} · "
        f"FCS {'pooled' if fcs == 'pool' else 'dropped'}"
    )
    return [t.to_dict() for t in teams], [g.to_dict() for g in games], label


@st.cache_data(show_spinner="Building preseason prior…")
def warm_start_cached(method: str, params_json: str, data_key: str, _names: list[str], _games: list[Game], season: int):
    return warm_start(method, _games, _names, season, settings=settings, **json.loads(params_json))


@st.cache_data(show_spinner=False)
def fit_cached(method: str, params_json: str, data_key: str, _names: list[str], _games: list[Game], _ws: WarmStart):
    params = json.loads(params_json)
    model = make_model(method, settings, **params)
    result = model.fit(_names, _games, prior=_ws.prior)
    cal = MarginCalibrator.fit_cv(
        model, _names, _games, fallback_hfa=settings.home_field_adv, prior=_ws.prior, fallback=_ws.calibrator
    )
    return result, cal


@st.cache_data(show_spinner="Computing weekly snapshots…")
def snapshots_cached(
    method: str, params_json: str, data_key: str, _names: list[str], _games: list[Game], use_cache: bool, _ws: WarmStart
):
    return weekly_snapshots(
        method, _names, _games, settings=settings, use_cache=use_cache, prior=_ws.prior, **json.loads(params_json)
    )


@st.cache_data(show_spinner="Running walk-forward backtest…")
def backtest_cached(methods: tuple[str, ...], start_week: int, params_json: str, data_key: str, _names, _games):
    params = json.loads(params_json)
    table, results = compare_methods(
        _names, _games, list(methods), start_week=start_week, params=dict.fromkeys(methods, params)
    )
    detail = {m: (r.predictions, r.by_week(), r.calibration(), r.by_season()) for m, r in results.items()}
    return table, detail


@st.cache_data(show_spinner="Scoring résumés…")
def ppoints_cached(params_json: str, data_key: str, _names, _games, _ratings, _conf_map, _snaps):
    return _ppoints(_names, _games, _ratings, _conf_map, _snaps)


@st.cache_data(show_spinner="Building the playoff picture…")
def playoff_cached(params_json: str, data_key: str, weights: tuple, fmt_year: int, _teams, _games, _result, _cal, _pp):
    return playoff_picture(
        _teams, _games, _result, _cal, ppoints=_pp, weights=SeedingWeights(*weights), fmt=format_for(fmt_year)
    )


@st.cache_data(show_spinner="Simulating the rest of the season…")
def simulate_cached(
    params_json: str, data_key: str, weights: tuple, fmt_year: int, n_sims: int, _teams, _games, _result, _cal, _pp
):
    return simulate_playoff(
        _teams,
        _games,
        _result,
        _cal,
        ppoints=_pp,
        weights=SeedingWeights(*weights),
        fmt=format_for(fmt_year),
        n_sims=n_sims,
        seed=0,
    )


# --- sidebar -----------------------------------------------------------------------------------

st.markdown(_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.title("🏈 CFB Ratings")
    has_key = bool(settings.api_key)
    source = st.radio(
        "Data source",
        ["CollegeFootballData API", "Demo (synthetic season)"],
        index=0 if has_key else 1,
        help="Demo mode generates seasons with known team strengths so every feature works offline.",
    )
    demo = source.startswith("Demo")
    freshness_slot = st.empty()  # filled once the seasons are loaded, so a refresh reports itself
    if not has_key and not demo:
        st.warning("No `CFBD_API_KEY` found. Add one to `.env` or switch to demo mode.")
    # Public site: only seasons that cost no API call (the archive) plus the one in progress.
    public = settings.public_dashboard
    if public:
        archived = CFBDClient(settings).cache.archived_years()
        min_year, max_year = (archived[0] if archived else int(settings.year)), int(settings.year)
    else:
        min_year, max_year = 2000, 2035
    year = int(st.number_input("Season", min_value=min_year, max_value=max_year, value=int(settings.year), step=1))
    method = st.selectbox(
        "Rating method", MODEL_NAMES, index=MODEL_NAMES.index("hybrid"), format_func=lambda m: MODEL_LABELS[m]
    )

    with st.expander("Data options"):
        max_warmup = min(3, year - min_year) if public else 3  # never reach below the archive on the public site
        warmup = int(
            st.number_input(
                "Warm-start from prior seasons",
                min_value=0,
                max_value=max_warmup,
                value=min(1, max_warmup),
                step=1,
                help="Earlier seasons are loaded; their regressed final ratings seed this season and feed the backtest.",
            )
        )
        if demo:
            season_type, fcs = "both", settings.fcs_handling
            seed = int(st.number_input("Demo seed", min_value=0, max_value=9999, value=7, step=1))
            demo_weeks = int(
                st.slider(
                    "Demo: weeks played",
                    0,
                    DEMO_WEEKS,
                    10,
                    help="Leave later weeks unplayed to see a mid-season snapshot with projections; 0 is the preseason.",
                )
            )
        else:
            seed, demo_weeks = 7, DEMO_WEEKS
            season_type = st.selectbox("Season type", ["both", "regular", "postseason"], index=0)
            fcs = st.radio(
                "Games vs. FCS opponents",
                ["pool", "drop"],
                index=0 if settings.fcs_handling == "pool" else 1,
                horizontal=True,
                help="Pool: every FCS opponent becomes one shared pseudo-team so the result counts. Drop: ignore those games.",
            )

    with st.expander("Model parameters"):
        hfa = st.slider("Home-field advantage (pts)", 0.0, 5.0, float(settings.home_field_adv), 0.1)
        colley_prior = st.slider("Colley prior strength", 0.5, 6.0, float(settings.colley_prior_strength), 0.5)
        ridge_lambda = st.slider("Massey ridge lambda", 0.0, 5.0, float(settings.massey_ridge_lambda), 0.1)
        prior_lambda = st.slider("Massey prior strength", 0.5, 10.0, float(settings.massey_prior_lambda), 0.5)
        half_life = st.slider(
            "Massey recency half-life (weeks, 0 = off)", 0.0, 10.0, float(settings.massey_half_life_weeks), 0.5
        )
        elo_k = st.slider("Elo K-factor", 5.0, 80.0, float(settings.elo_k), 1.0)
        season_regress = st.slider("Season regression (all models)", 0.0, 1.0, float(settings.season_regress), 0.05)
        bt_ridge = st.slider("Bradley-Terry ridge", 0.1, 5.0, float(settings.bt_ridge), 0.1)
        st.markdown("**Hybrid weights**")
        w_colley = st.slider("Colley", 0.0, 1.0, float(settings.hybrid_weight_map.get("colley", 0.0)), 0.05)
        w_massey = st.slider("Massey", 0.0, 1.0, float(settings.hybrid_weight_map.get("massey", 0.0)), 0.05)
        w_elo = st.slider("Elo", 0.0, 1.0, float(settings.hybrid_weight_map.get("elo", 0.0)), 0.05)
        w_bt = st.slider("Bradley-Terry", 0.0, 1.0, float(settings.hybrid_weight_map.get("bradley_terry", 0.0)), 0.05)

    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = 0
    if public:  # no visitor can spend the key's monthly budget; the cache refreshes itself
        if not demo:
            st.caption(f"Data refreshes itself every {settings.cache_ttl_hours:g} h.")
    elif st.button("Refresh API cache", disabled=demo, help="Re-download this selection from CFBD once."):
        st.session_state.refresh_token += 1
        st.session_state.refresh_pending = True
    calls_slot = st.empty()  # "API calls this session", filled after the load so a refresh counts itself
    force_refresh = bool(st.session_state.pop("refresh_pending", False))
    st.caption(f"cfbratings v{__version__}")

weights = {
    k: v for k, v in {"colley": w_colley, "massey": w_massey, "elo": w_elo, "bradley_terry": w_bt}.items() if v > 0
}
params = {
    "hfa": hfa,
    "prior_strength": colley_prior,
    "ridge_lambda": ridge_lambda,
    "prior_lambda": prior_lambda,
    "half_life_weeks": half_life,
    "k": elo_k,
    "season_regress": season_regress,
    "ridge": bt_ridge,
    "weights": weights or {"massey": 1.0},
}
params_json = json.dumps(params, sort_keys=True)
theme = current_theme()

# --- data --------------------------------------------------------------------------------------

try:
    teams_raw, games_raw, source_label = load_data(
        demo, year, warmup, season_type, fcs, st.session_state.refresh_token, force_refresh, seed, demo_weeks
    )
except CFBDError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # network / HTTP problems
    st.error(f"Could not load data: {exc}")
    st.stop()

# Provenance, now that the load has happened (a refresh shows up as "cached just now").
if demo:
    freshness_slot.caption("Synthetic data · nothing is fetched")
else:
    _client = CFBDClient(settings)
    freshness_slot.caption(
        "  \n".join(freshness_label(y, _client.freshness(y, season_type)) for y in range(year - warmup, year + 1))
    )
calls_slot.caption(f"API calls this session: {0 if demo else CFBDClient.api_calls}")

teams = [Team.from_dict(t) for t in teams_raw]
all_games = [Game.from_dict(g) for g in games_raw]
names = [t.school for t in teams]
real_names = [n for n in names if not is_pool_team(n)]
conf_map = _conference_map(teams)
team_by_name = {t.school: t for t in teams}
logo_map = {t.school: team_logo(t, theme) for t in teams}
team_colors = {t.school: resolve_team_color(t.color, t.alt_color, theme) for t in teams}


def logos(names_col: pd.Series) -> pd.Series:
    """Logo URLs for a column of team names; a team without one (synthetic, pooled FCS) gets a blank cell.

    ``ImageColumn`` prints a missing value as the text "None", so the gaps are empty strings.
    """
    return names_col.map(logo_map).fillna("")


season = max((g.season for g in all_games), default=year)
games = [g for g in all_games if g.season == season]
last_week = max_completed_week(games)
n_played = sum(1 for g in games if g.has_score)
n_unplayed = sum(1 for g in games if not g.has_score)
n_prior_seasons = len({g.season for g in all_games if g.has_score and g.season < season})
_content = hashlib.md5(
    json.dumps(sorted((g.id, g.season, g.home_points, g.away_points, g.completed) for g in all_games)).encode()
).hexdigest()[:12]
data_key = f"{demo}-{year}-{warmup}-{season_type}-{fcs}-{seed}-{demo_weeks}-{_content}"

st.title("College Football Ratings")
bits = [source_label, MODEL_LABELS[method]]
bits.append(f"{n_played} games through week {last_week}" if n_played else "preseason (no games played yet)")
if n_unplayed:
    bits.append(f"{n_unplayed} still to play")
if n_prior_seasons > 0:
    bits.append(f"warm-started from {n_prior_seasons} prior season{'s' if n_prior_seasons > 1 else ''}")
st.caption(" · ".join(bits))

if n_played == 0 and n_prior_seasons <= 0:
    st.info("No completed games yet for this season. Pick another season, add prior seasons, or switch to demo mode.")
    st.stop()

ws = warm_start_cached(method, params_json, data_key, names, all_games, season)
result, cal = fit_cached(method, params_json, data_key, names, games, ws)
recs = records(names, games)
top_team, top_rating = result.ranked()[0]
rating_label = SCALE_LABELS.get(result.scale, "Rating")

kpi_row(
    [
        ("No. 1", top_team, str(recs[top_team])),
        ("Home-field advantage", f"{cal.intercept:+.1f} pts", "calibrated, cross-validated"),
        ("Spread uncertainty", f"±{cal.sigma:.1f} pts", "residual sigma of predicted margins"),
        ("Calibration games", f"{cal.n_games}", CAL_SOURCE_LABELS.get(cal.source, cal.source.replace("_", " "))),
    ]
)

tab_rank, tab_team, tab_proj, tab_conf, tab_cfp, tab_match, tab_bt, tab_doc = st.tabs(
    [
        "📊 Rankings",
        "🏟️ Team",
        "🔭 Projections",
        "🏆 Conferences",
        "🏈 Playoff",
        "🔮 Matchup",
        "🧪 Backtest",
        "📖 Methodology",
    ]
)

# --- rankings ------------------------------------------------------------------------------------

with tab_rank:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    top_n = int(c1.slider("Show top", 10, len(real_names), min(25, len(real_names)), 5))
    conf_filter = c2.selectbox("Conference", ["All", *sorted({conf_map[n] or "Independent" for n in real_names})])
    sort_by = c3.selectbox("Sort by", list(SORT_LABELS), format_func=SORT_LABELS.get)
    show_pp = c4.toggle("PPoints", value=True, help="Résumé metric: who you beat and where.")

    # Weekly snapshots feed PPoints (rank at game time) and the rank-change column; in the
    # preseason there are none and every change shows the "no movement" dash.
    snaps = snapshots_cached(method, params_json, data_key, names, games, not demo, ws)
    pp = _ppoints(names, games, result.ratings, conf_map, snaps) if show_pp or sort_by == "ppoints" else None
    changes = rank_changes(result.rank_map(), snaps, week=last_week)
    table = team_table(names, games, result, cal, conf_map, ppoints=pp)
    table = table.assign(logo=logos(table["team"]), change=table["team"].map(changes).fillna(RANK_UNCHANGED))
    if conf_filter != "All":
        table = table[table["conference"] == conf_filter]
    table = table.sort_values(sort_by, ascending=False).head(top_n)

    cols = ["rank", "change", "logo", "team", "conference", "record", "rating", "sos", "sos_rank", "momentum"] + (
        ["ppoints"] if pp is not None else []
    )
    # A team with no completed game (the whole league in the preseason, a bye-week team in week 1)
    # has no schedule strength or momentum: those cells are blank, never "None".
    shown, rank_config = text_when_gappy(
        table[cols],
        {
            "rank": st.column_config.NumberColumn("#", width="small"),
            "change": st.column_config.TextColumn("Δ", width="small", help="Rank change since the previous week"),
            "logo": logo_col(),
            "team": st.column_config.TextColumn("Team"),
            "conference": st.column_config.TextColumn("Conference"),
            "record": st.column_config.TextColumn("Record", width="small"),
            "rating": st.column_config.NumberColumn(rating_label, format="%.3f"),
            "sos": st.column_config.NumberColumn(
                "SOS",
                format="%.3f",
                help="Mean opponent rating, same units as the rating; scheduled opponents until the first game",
            ),
            "sos_rank": st.column_config.NumberColumn("SOS #", width="small"),
            "momentum": st.column_config.NumberColumn(
                "Momentum", format="%+.1f", help="Points above expectation, last 3 games"
            ),
            "ppoints": st.column_config.NumberColumn("PPoints", format="%.1f"),
        },
        {"sos": "{:.3f}", "sos_rank": "{}", "momentum": "{:+.1f}"},
    )
    st.dataframe(shown, hide_index=True, width="stretch", height=table_height(len(shown)), column_config=rank_config)
    st.download_button(
        "Download CSV",
        table.drop(columns=["logo"]).to_csv(index=False).encode(),
        f"ratings_{season}_{method}.csv",
        "text/csv",
    )

    value_col = sort_by if sort_by in {"rating", "ppoints", "sos", "momentum"} else "rating"
    color_by = st.radio("Colour by", ["Team", "Conference"], horizontal=True)
    chart(
        rating_bar(
            table,
            top=top_n,
            value_col=value_col,
            value_label=rating_label if value_col == "rating" else SORT_LABELS[value_col],
            title=f"Top {len(table)} by {SORT_PHRASES[value_col]}",
            theme=theme,
            team_colors=team_colors if color_by == "Team" else None,
        ),
        theme,
    )
    with st.expander("Rating vs. schedule strength", expanded=True):
        if n_played == 0:
            st.caption("Before the first game, schedule strength is the mean rating of the scheduled opponents.")
        chart(rating_sos_scatter(table, theme=theme), theme)

# --- team ----------------------------------------------------------------------------------------

with tab_team:
    full = team_table(names, games, result, cal, conf_map)
    team = st.selectbox("Team", real_names, index=real_names.index(top_team))
    row = full[full["team"] == team].iloc[0]
    proj: TeamProjection = project_team(team, games, result, cal, recs[team])
    team_header(team_by_name.get(team), team, conf_map.get(team) or "Independent", theme)
    has_games = recs[team].games > 0
    has_sos = pd.notna(row["sos"])  # false only for a team with nothing scheduled at all
    tiles = [
        ("Rank", f"#{int(row['rank'])}", f"of {len(real_names)}"),
        ("Record", row["record"], f"{len(proj.remaining)} game{'s' if len(proj.remaining) != 1 else ''} left"),
        (rating_label, f"{row['rating']:.3f}", None),
        (
            "SOS rank",
            f"#{int(row['sos_rank'])}" if has_sos else NO_VALUE,
            (f"SOS {row['sos']:.3f}" + ("" if has_games else " · scheduled opponents"))
            if has_sos
            else "nothing scheduled",
        ),
        ("Momentum", f"{row['momentum']:+.1f} pts", "last 3 games vs. expectation" if has_games else "no game yet"),
    ]
    if proj.remaining:
        tiles.append(
            (
                "Projected record",
                f"{proj.expected_final_wins:.1f}-{proj.expected_final_losses:.1f}",
                f"win out {proj.p_win_out:.0%}",
            )
        )
    nxt = proj.next_game
    kpi_row(tiles, extra_html=next_game_tile(team, nxt[0], nxt[1], logo_map) if nxt else "")

    compare = st.multiselect(
        "Compare with", [n for n in real_names if n != team], max_selections=7, placeholder="Add up to 7 teams"
    )
    snaps = snapshots_cached(method, params_json, data_key, names, games, not demo, ws)
    if snaps:
        metric_choice = st.radio("Trajectory", ["rank", "rating"], horizontal=True)
        chart(
            trajectory_chart(
                snaps,
                [team, *compare],
                metric=metric_choice,
                title="Week by week",
                theme=theme,
                team_colors=team_colors,
            ),
            theme,
        )
    else:
        st.caption("The week-by-week trajectory appears once the first week has been played.")

    SCHED_COLS = [
        "week", "type", "opp_logo", "opponent", "opp_rank", "site", "expected_margin", "win_prob",
        "result", "score", "actual_margin", "vs_expected",
    ]  # fmt: skip
    sched = schedule_table(team, games, result, cal).reindex(columns=[c for c in SCHED_COLS if c != "opp_logo"])

    # Unplayed games have no result yet and a pooled FCS opponent has no rank. Streamlit prints
    # missing numbers as "None", so these columns are pre-formatted as text with blanks instead.
    sched = sched.assign(
        opp_logo=logos(sched["opponent"]),  # the pooled "X (FCS)" label matches nothing: blank cell
        result=sched["result"].fillna(""),
        score=sched["score"].fillna(""),
        opp_rank=fmt_blank(sched["opp_rank"], "#{:.0f}"),
        actual_margin=fmt_blank(sched["actual_margin"], "{:+.0f}"),
        vs_expected=fmt_blank(sched["vs_expected"], "{:+.1f}"),
    )
    st.dataframe(
        percent_cols(sched, ["win_prob"])[SCHED_COLS],
        hide_index=True,
        width="stretch",
        height=table_height(len(sched)),
        column_config={
            "week": st.column_config.NumberColumn("Wk", width="small"),
            "type": st.column_config.TextColumn("Type", width="small"),
            "opp_logo": logo_col(),
            "opponent": st.column_config.TextColumn("Opponent"),
            "opp_rank": st.column_config.TextColumn("Opp #", width="small"),
            "site": st.column_config.TextColumn("Site", width="small"),
            "expected_margin": st.column_config.NumberColumn("Expected", format="%+.1f"),
            "win_prob": pct_bar("Win prob"),
            "result": st.column_config.TextColumn("W/L", width="small"),
            "score": st.column_config.TextColumn("Score", width="small"),
            "actual_margin": st.column_config.TextColumn("Actual", width="small"),
            "vs_expected": st.column_config.TextColumn("vs. exp.", width="small"),
        },
    )
    if proj.remaining:
        chart(
            win_distribution_chart(
                proj.distribution, proj.record.wins, team=team, theme=theme, accent=team_colors.get(team)
            ),
            theme,
        )

# --- projections -----------------------------------------------------------------------------------

with tab_proj:
    if n_unplayed == 0:
        st.info("Every scheduled game has been played, so projections equal the final records.")
    else:
        st.markdown(
            f"**{n_unplayed} games** remain on the schedule. Each is given a win probability from the calibrated "
            f"{MODEL_LABELS[method]} ratings; win-total odds are exact Poisson-binomial sums, not simulations."
        )
    proj_df = projection_table(names, games, result, cal, conference_map=conf_map)
    proj_view = proj_df  # the display copy; ``proj_df`` stays raw for the CSV download
    if not proj_df.empty:
        proj_df = proj_df.assign(logo=logos(proj_df["team"]), next_logo=logos(proj_df["next_opponent"]))
        # A team with nothing left to play has no next game: those cells are blank text rather
        # than the "None" that st.dataframe prints for a missing value, and its record is final.
        proj_view = proj_df.assign(
            proj_record=proj_df["proj_record"].where(proj_df["remaining"] > 0, "Final"),
            next_opponent=proj_df["next_opponent"].fillna(""),
            next_site=proj_df["next_site"].fillna(""),
            next_win_prob=fmt_blank(proj_df["next_win_prob"], "{:.0%}"),
        )
    p1, p2 = st.columns([1, 1])
    proj_sort = p1.selectbox(
        "Sort by",
        ["proj_wins", "p_10_wins", "p_win_out", "p_6_wins", "rank"],
        format_func=lambda c: {
            "proj_wins": "Projected wins",
            "p_10_wins": "P(10+ wins)",
            "p_win_out": "P(win out)",
            "p_6_wins": "P(bowl eligible)",
            "rank": "Rating rank",
        }[c],
    )
    proj_conf = p2.selectbox("Conference ", ["All", *sorted({conf_map[n] or "Independent" for n in real_names})])
    if proj_conf != "All":
        proj_df = proj_df[proj_df["conference"] == proj_conf]
        proj_view = proj_view[proj_view["conference"] == proj_conf]
    proj_df = proj_df.sort_values(proj_sort, ascending=proj_sort == "rank")
    proj_view = proj_view.sort_values(proj_sort, ascending=proj_sort == "rank")
    PROJ_COLS = [
        "rank", "logo", "team", "conference", "record", "remaining", "proj_record",
        "p_6_wins", "p_8_wins", "p_10_wins", "p_12_wins", "p_win_out",
        "next_logo", "next_opponent", "next_site", "next_win_prob",
    ]  # fmt: skip
    if n_unplayed == 0:  # season over: nothing is left and nobody has a next game
        PROJ_COLS = [c for c in PROJ_COLS if c != "remaining" and not c.startswith("next_")]
    st.dataframe(
        percent_cols(proj_view, ["p_6_wins", "p_8_wins", "p_10_wins", "p_12_wins", "p_win_out"])[PROJ_COLS],
        hide_index=True,
        width="stretch",
        height=table_height(len(proj_view)),
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "logo": logo_col(),
            "team": st.column_config.TextColumn("Team", width="medium"),
            "conference": st.column_config.TextColumn("Conference", width="medium"),
            "record": st.column_config.TextColumn("Record", width="small"),
            "remaining": st.column_config.NumberColumn("Left", width="small"),
            "proj_record": st.column_config.TextColumn("Projected", help="Expected final wins-losses"),
            "p_6_wins": pct_bar("6+ wins"),
            "p_8_wins": pct_bar("8+ wins"),
            "p_10_wins": pct_bar("10+ wins"),
            "p_12_wins": pct_bar("12+ wins"),
            "p_win_out": pct_num("Win out"),
            "next_logo": logo_col(),
            "next_opponent": st.column_config.TextColumn("Next opponent", width="medium"),
            "next_site": st.column_config.TextColumn("Site", width="small"),
            "next_win_prob": st.column_config.TextColumn("Next win %", width="small"),
        },
    )
    st.download_button(
        "Download projections CSV",
        proj_df.drop(columns=["logo", "next_logo"], errors="ignore").to_csv(index=False).encode(),
        f"projections_{season}_{method}.csv",
        "text/csv",
    )

# --- conferences -----------------------------------------------------------------------------------

with tab_conf:
    conf_df = conference_table(result.ratings, conf_map, recs)
    conf_df = conf_df[conf_df["conference"] != "FCS"]
    st.dataframe(
        conf_df,
        hide_index=True,
        width="stretch",
        height=table_height(len(conf_df)),
        column_config={
            "conference": "Conference",
            "strength_index": st.column_config.NumberColumn(
                "Index", format="%.2f", help="0.6 = weakest, 1.6 = strongest"
            ),
            "teams": st.column_config.NumberColumn("Teams", width="small"),
            "mean_rating": st.column_config.NumberColumn(f"Mean {rating_label.lower()}", format="%.3f"),
            "median_rating": st.column_config.NumberColumn("Median", format="%.3f"),
            "top_team": "Best team",
            "wins": st.column_config.NumberColumn("W", width="small"),
            "losses": st.column_config.NumberColumn("L", width="small"),
        },
    )
    chart(
        conference_dots(conf_df, title="Conference strength index (0.6 = weakest, 1.6 = strongest)", theme=theme),
        theme,
    )
    ind_df = independents_table(result.ratings, conf_map, recs)
    if not ind_df.empty:
        st.markdown(
            "**Independents** are not a conference: no conference schedule, standings, champion or automatic "
            "playoff bid. Each is listed on its own; its index is its own rating placed on the conference scale, "
            "which is how a résumé values a game against it."
        )
        ind_df = ind_df.assign(logo=logos(ind_df["team"]))
        st.dataframe(
            ind_df[["rank", "logo", "team", "record", "rating", "strength_index"]],
            hide_index=True,
            width="stretch",
            height=table_height(len(ind_df)),
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "logo": logo_col(),
                "team": "Team",
                "record": st.column_config.TextColumn("Record", width="small"),
                "rating": st.column_config.NumberColumn(rating_label, format="%.3f"),
                "strength_index": st.column_config.NumberColumn("Index", format="%.2f", help="On the 0.6-1.6 scale"),
            },
        )

# --- playoff ----------------------------------------------------------------------------------------

with tab_cfp:
    fmt_years = sorted(FORMATS)
    fmt_year = int(
        st.radio(
            "Rules",
            fmt_years,
            index=fmt_years.index(int(format_for(season).name)),
            horizontal=True,
            format_func=lambda y: f"{y} format",
            help="2026: ACC, Big 12, Big Ten and SEC champions plus the best Group-of-Six team, straight seeding. "
            "2025: the five best champions, straight seeding. 2024: the five best champions, byes to the four best champions.",
        )
    )
    cfp_fmt = format_for(fmt_year).resolved(conference_strength(result.ratings, conf_map))
    st.caption(
        f"{cfp_fmt.summary}. First-round games at the higher seed's home (home-field advantage {cal.intercept:+.1f} "
        "pts); every later round at a neutral site."
    )
    with st.expander("Seeding weights (the committee proxy)"):
        st.caption(
            "The committee is modelled as rating·z(rating) + résumé·z(PPoints) - loss·losses + champion bump. "
            "The defaults are a heuristic reading of its record since 2014, not fitted values."
        )
        w1, w2, w3, w4 = st.columns(4)
        w_rating = w1.slider("Power rating", 0.0, 2.0, 1.0, 0.05)
        w_resume = w2.slider("Résumé (PPoints)", 0.0, 2.0, 0.5, 0.05)
        w_loss = w3.slider("Per loss", 0.0, 1.0, 0.35, 0.05)
        w_champ = w4.slider("Champion bump", 0.0, 1.0, 0.15, 0.05)
    weights = (w_rating, w_resume, w_loss, w_champ)
    cfp_snaps = snapshots_cached(method, params_json, data_key, names, games, not demo, ws)
    cfp_pp = ppoints_cached(params_json, data_key, names, games, result.ratings, conf_map, cfp_snaps)
    pic = playoff_cached(params_json, data_key, weights, fmt_year, teams, games, result, cal, cfp_pp)

    st.subheader("Conference races")
    race_rows = []
    for conf in sorted(pic.champions, key=lambda c: (c not in cfp_fmt.power_conferences, c)):
        ch = pic.champions[conf]
        race_rows.append(
            {
                "conference": conf,
                "power": conf in cfp_fmt.power_conferences,
                "team": ch.team,
                "status": "Champion" if ch.decided else "Leads",
                "title_game": "" if ch.decided else f"vs {ch.opponent}",
                "p_champion": 1.0 if ch.decided else ch.p_champion,
            }
        )
    race_df = pd.DataFrame(race_rows)
    if not race_df.empty:
        race_df = race_df.assign(logo=logos(race_df["team"]))
        st.dataframe(
            percent_cols(race_df, ["p_champion"])[
                ["conference", "power", "logo", "team", "status", "title_game", "p_champion"]
            ],
            hide_index=True,
            width="stretch",
            height=table_height(len(race_df)),
            column_config={
                "conference": "Conference",
                "power": st.column_config.CheckboxColumn("Auto bid", help="Champion qualifies automatically"),
                "logo": logo_col(),
                "team": "Leader",
                "status": st.column_config.TextColumn("Status", width="small"),
                "title_game": st.column_config.TextColumn("Title game", help="The matchup the standings imply"),
                "p_champion": pct_bar("Wins conference"),
            },
        )
    with st.expander("Conference standings"):
        stand_df = standings_table(pic.standings)
        pick = st.selectbox("Conference  ", ["All", *sorted(pic.standings)])
        if pick != "All":
            stand_df = stand_df[stand_df["conference"] == pick]
        if not stand_df.empty:
            stand_df = stand_df.assign(logo=logos(stand_df["team"]))
            st.dataframe(
                stand_df[["conference", "place", "logo", "team", "conf_record", "record", "rating"]],
                hide_index=True,
                width="stretch",
                height=table_height(len(stand_df)),
                column_config={
                    "conference": "Conference",
                    "place": st.column_config.NumberColumn("#", width="small"),
                    "logo": logo_col(),
                    "team": "Team",
                    "conf_record": st.column_config.TextColumn("Conf", width="small"),
                    "record": st.column_config.TextColumn("Overall", width="small"),
                    "rating": st.column_config.NumberColumn(rating_label, format="%.3f"),
                },
            )
        st.caption("Conference record, then head-to-head among the tied, then overall record, then rating.")

    st.subheader("If the season ended today")
    fav, p_fav = pic.odds.title_favourite()
    kpi_row(
        [
            (
                "No. 1 seed",
                pic.field[0].team if pic.field else NO_VALUE,
                pic.field[0].conference if pic.field else None,
            ),
            ("Title favourite", fav, f"{p_fav:.0%} to win it all"),
            (
                "Last team in",
                pic.field[-1].team if pic.field else NO_VALUE,
                f"seed {pic.field[-1].seed}, {pic.field[-1].bid}" if pic.field else None,
            ),
            (
                "First team out",
                next((t for t in pic.ranking if t not in {s.team for s in pic.field}), NO_VALUE),
                "by the committee proxy",
            ),
        ]
    )
    st.markdown(bracket_html(pic.odds, logo_map, cfp_fmt.byes), unsafe_allow_html=True)
    odds_df = pic.odds.table.assign(logo=logos(pic.odds.table["team"]))
    st.dataframe(
        percent_cols(odds_df, ["p_quarter", "p_semi", "p_final", "p_title"])[
            ["seed", "logo", "team", "conference", "bid", "p_quarter", "p_semi", "p_final", "p_title"]
        ],
        hide_index=True,
        width="stretch",
        height=table_height(len(odds_df)),
        column_config={
            "seed": st.column_config.NumberColumn("Seed", width="small"),
            "logo": logo_col(),
            "team": "Team",
            "conference": "Conference",
            "bid": st.column_config.TextColumn("Bid", width="small"),
            "p_quarter": pct_bar("Quarterfinal"),
            "p_semi": pct_bar("Semifinal"),
            "p_final": pct_bar("Final"),
            "p_title": pct_bar("Title"),
        },
    )
    st.caption("Exact odds for this bracket from the calibrated win probabilities, every possible opponent weighted.")

    st.subheader("Rest of season, simulated")
    if n_unplayed == 0:
        st.info("Every scheduled game has been played: the bracket above is the final one.")
    else:
        n_sims = int(st.select_slider("Simulations", [250, 500, 1000, 2000], value=500))
        sim = simulate_cached(params_json, data_key, weights, fmt_year, n_sims, teams, games, result, cal, cfp_pp)
        sim_df = sim.table.assign(logo=logos(sim.table["team"]))
        sim_top = int(st.slider("Show top", 12, len(sim_df), min(25, len(sim_df)), 1, key="cfp_top"))
        shown = sim_df.head(sim_top)
        st.dataframe(
            percent_cols(shown, ["p_champion", "p_playoff", "p_bye", "p_title"])[
                ["logo", "team", "conference", "record", "p_champion", "p_playoff", "p_bye", "mean_seed", "p_title"]
            ],
            hide_index=True,
            width="stretch",
            height=table_height(len(shown)),
            column_config={
                "logo": logo_col(),
                "team": "Team",
                "conference": "Conference",
                "record": st.column_config.TextColumn("Record", width="small"),
                "p_champion": pct_bar("Wins conference"),
                "p_playoff": pct_bar("Makes playoff"),
                "p_bye": pct_bar("First-round bye"),
                "mean_seed": st.column_config.NumberColumn("Avg seed", format="%.1f", help="When it makes the field"),
                "p_title": pct_bar("Wins title"),
            },
        )
        st.caption(
            f"{sim.n_sims} simulations of the {n_unplayed} unplayed games: standings, title games, the committee "
            "proxy and the bracket are settled in each, and the title odds average exact bracket odds."
        )
        st.markdown(
            "**Projected bracket** (each conference's most likely champion, teams ranked by mean committee score)"
        )
        st.markdown(bracket_html(sim.projected_odds, logo_map, cfp_fmt.byes), unsafe_allow_html=True)
        st.download_button(
            "Download playoff projections CSV",
            sim.table.to_csv(index=False).encode(),
            f"playoff_{season}_{method}.csv",
            "text/csv",
        )

# --- matchup ----------------------------------------------------------------------------------------

with tab_match:
    ranked_names = [t for t, _ in result.ranked()]
    h1, h2, h3 = st.columns([2, 2, 1])
    home = h1.selectbox("Home", ranked_names, index=0)
    away = h2.selectbox("Away", ranked_names, index=1 if len(ranked_names) > 1 else 0)
    neutral = h3.toggle("Neutral site")
    if home == away:
        st.warning("Pick two different teams")
    else:
        matchup_header(home, away, neutral, team_by_name, theme, logo_map)
        rows = []
        for m in MODEL_NAMES:
            ws_m = warm_start_cached(m, params_json, data_key, names, all_games, season)
            res_m, cal_m = fit_cached(m, params_json, data_key, names, games, ws_m)
            p = cal_m.predict_game(res_m, home, away, neutral)
            rows.append({"method": MODEL_LABELS[m], "line": p.spread, "home_margin": p.margin, "p_home": p.p_home})
        df_m = pd.DataFrame(rows)
        consensus_margin = float(df_m["home_margin"].mean())
        consensus_p = float(df_m["p_home"].mean())
        fav = home if consensus_margin >= 0 else away
        kpi_row(
            [
                ("Consensus line", f"{fav} -{abs(consensus_margin):.1f}", "mean of all five models"),
                (f"{home} win probability", f"{consensus_p:.0%}", "home" if not neutral else "neutral site"),
                (f"{away} win probability", f"{1 - consensus_p:.0%}", "away" if not neutral else "neutral site"),
            ]
        )
        st.dataframe(
            percent_cols(df_m, ["p_home"]),
            hide_index=True,
            width="stretch",
            height=table_height(len(df_m)),
            column_config={
                "method": "Method",
                "line": "Line",
                "home_margin": st.column_config.NumberColumn("Home margin", format="%+.1f"),
                "p_home": pct_bar("Home win prob", decimals=1),
            },
        )

# --- backtest -----------------------------------------------------------------------------------------

with tab_bt:
    st.markdown(
        "Walk-forward evaluation: for each week the model is fitted on every earlier game and predicts that week's "
        "games out-of-sample. Lower Brier / log loss / MAE is better; the **home_prior** baseline always picks the home team."
        + (
            " Every season after the first is warm-started from the one before it, so evaluation can start in week 1."
            if n_prior_seasons > 0
            else ""
        )
    )
    b1, b2 = st.columns([3, 1])
    chosen = b1.multiselect("Methods", MODEL_NAMES, default=list(MODEL_NAMES), format_func=lambda m: MODEL_LABELS[m])
    max_week = max(2, max_completed_week(all_games))
    start_week = int(b2.slider("Start week", 1, max_week, 1 if n_prior_seasons > 0 else min(4, max_week)))
    if not chosen:
        st.info("Pick at least one method to run the backtest.")
    else:
        table_bt, detail = backtest_cached(tuple(chosen), start_week, params_json, data_key, names, all_games)
        table_bt = table_bt.assign(method=[MODEL_LABELS.get(m, m) for m in table_bt["method"]])
        backtest_table(table_bt)
        metric = st.selectbox("Weekly metric", list(METRIC_LABELS), index=0, format_func=METRIC_LABELS.get)
        wl, wr = st.columns(2)
        with wl:
            chart(
                weekly_metric_chart(
                    {MODEL_LABELS[m]: d[1] for m, d in detail.items()},
                    metric=metric,
                    title=f"{METRIC_LABELS[metric]} by week",
                    theme=theme,
                    min_games=5,
                ),
                theme,
            )
            st.caption("Weeks with fewer than five games are hidden so a near-empty week cannot swing a line.")
        with wr:
            chart(
                calibration_chart({MODEL_LABELS[m]: d[2] for m, d in detail.items()}, title="Reliability", theme=theme),
                theme,
            )
        if n_prior_seasons > 0:
            with st.expander("By season"):
                for m, d in detail.items():
                    st.markdown(f"**{MODEL_LABELS[m]}**")
                    backtest_table(d[3])
        with st.expander("Out-of-sample predictions"):
            all_preds = pd.concat([d[0].assign(method=m) for m, d in detail.items()], ignore_index=True)
            shown = oos_predictions_view(all_preds.assign(method=[MODEL_LABELS.get(m, m) for m in all_preds["method"]]))
            st.dataframe(
                percent_cols(shown, ["p_home"]),
                hide_index=True,
                width="stretch",
                column_config={k: v for k, v in OOS_COLUMNS.items() if k in shown.columns},
            )
            st.download_button(
                "Download predictions CSV", all_preds.to_csv(index=False).encode(), f"backtest_{season}.csv", "text/csv"
            )

# --- methodology ----------------------------------------------------------------------------------------

with tab_doc:
    if METHODOLOGY.exists():
        st.markdown(METHODOLOGY.read_text(encoding="utf-8"))
    else:
        st.info("See docs/METHODOLOGY.md in the repository.")
