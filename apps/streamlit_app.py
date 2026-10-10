"""College Football Ratings dashboard.

Run with ``streamlit run apps/streamlit_app.py``. Works without an API key in demo mode
(synthetic seasons with known ground truth); set ``CFBD_API_KEY`` to use real data.
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from cfbratings import __version__, settings
from cfbratings.analytics import conference_table, records, schedule_table, team_table
from cfbratings.backtest import compare_methods
from cfbratings.data import Game, Team, load_seasons, max_completed_week, synthetic_seasons
from cfbratings.data.cfbd import CFBDError
from cfbratings.data.schema import conference_map as _conference_map
from cfbratings.data.schema import is_pool_team
from cfbratings.models import MODEL_LABELS, MODEL_NAMES, make_model
from cfbratings.ppoints import ppoints as _ppoints
from cfbratings.predict import MarginCalibrator
from cfbratings.priors import WarmStart, warm_start
from cfbratings.projections import project_team, projection_table
from cfbratings.snapshots import weekly_snapshots
from cfbratings.viz import (
    calibration_chart,
    conference_dots,
    rating_bar,
    rating_sos_scatter,
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


# --- small UI helpers ----------------------------------------------------------------------


def current_theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit or no browser context
        return "light"


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Responsive stat tiles: wrap at narrow widths instead of truncating like st.metric."""
    tiles = "".join(
        '<div class="kpi"><div class="kpi-label">{}</div><div class="kpi-value">{}</div>{}</div>'.format(
            html.escape(label),
            html.escape(value),
            f'<div class="kpi-note">{html.escape(note)}</div>' if note else "",
        )
        for label, value, note in items
    )
    st.markdown(
        """
        <style>
        .kpi-row{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 16px}
        .kpi{flex:1 1 170px;min-width:150px;padding:12px 14px;border-radius:10px;
             background:var(--secondary-background-color);color:var(--text-color)}
        .kpi-label{font-size:.8rem;opacity:.7}
        .kpi-value{font-size:1.65rem;font-weight:600;line-height:1.2;overflow-wrap:anywhere}
        .kpi-note{font-size:.8rem;opacity:.75;margin-top:4px}
        </style>
        """
        + f'<div class="kpi-row">{tiles}</div>',
        unsafe_allow_html=True,
    )


def table_height(n_rows: int, *, cap: int = 900) -> int:
    return int(min(cap, 36 * (n_rows + 1) + 4))


def _team_dict(t: Team) -> dict:
    return {
        "school": t.school,
        "conference": t.conference,
        "classification": t.classification,
        "abbreviation": t.abbreviation,
    }


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
            [_team_dict(t) for t in chain[-1].teams],
            [g.to_dict() for g in games],
            f"Synthetic season {span} (seed {seed})",
        )
    teams, games = load_seasons(years, season_type, force_refresh=force_refresh, fcs=fcs)
    label = f"CollegeFootballData · {span} · {season_type} · FCS {'pooled' if fcs == 'pool' else 'dropped'}"
    return [_team_dict(t) for t in teams], [g.to_dict() for g in games], label


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


# --- sidebar -----------------------------------------------------------------------------------

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
    if not has_key and not demo:
        st.warning("No `CFBD_API_KEY` found. Add one to `.env` or switch to demo mode.")
    year = int(st.number_input("Season", min_value=2000, max_value=2035, value=int(settings.year), step=1))
    method = st.selectbox(
        "Rating method", MODEL_NAMES, index=MODEL_NAMES.index("hybrid"), format_func=lambda m: MODEL_LABELS[m]
    )

    with st.expander("Data options"):
        warmup = int(
            st.number_input(
                "Warm-start from prior seasons",
                min_value=0,
                max_value=3,
                value=1,
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
                    4,
                    DEMO_WEEKS,
                    10,
                    help="Leave later weeks unplayed to see a mid-season snapshot with projections.",
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
    if st.button("Refresh API cache", disabled=demo, help="Re-download this selection from CFBD once."):
        st.session_state.refresh_token += 1
        st.session_state.refresh_pending = True
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

teams = [Team(**t) for t in teams_raw]
all_games = [Game.from_dict(g) for g in games_raw]
names = [t.school for t in teams]
real_names = [n for n in names if not is_pool_team(n)]
conf_map = _conference_map(teams)
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
        ("Calibration games", f"{cal.n_games}", cal.source.replace("_", " ")),
    ]
)

tab_rank, tab_team, tab_proj, tab_conf, tab_match, tab_bt, tab_doc = st.tabs(
    ["📊 Rankings", "🏟️ Team", "🔭 Projections", "🏆 Conferences", "🔮 Matchup", "🧪 Backtest", "📖 Methodology"]
)

# --- rankings ------------------------------------------------------------------------------------

with tab_rank:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    top_n = int(c1.slider("Show top", 10, len(real_names), min(25, len(real_names)), 5))
    conf_filter = c2.selectbox("Conference", ["All", *sorted({conf_map[n] or "Independent" for n in real_names})])
    sort_by = c3.selectbox("Sort by", ["rating", "ppoints", "sos", "momentum", "wins"])
    show_pp = c4.toggle("PPoints", value=True, help="Résumé metric: who you beat and where.")

    pp = None
    if show_pp or sort_by == "ppoints":
        snaps = snapshots_cached(method, params_json, data_key, names, games, not demo, ws)
        pp = _ppoints(names, games, result.ratings, conf_map, snaps)
    table = team_table(names, games, result, cal, conf_map, ppoints=pp)
    if conf_filter != "All":
        table = table[table["conference"] == conf_filter]
    table = table.sort_values(sort_by, ascending=False).head(top_n)

    cols = ["rank", "team", "conference", "record", "rating", "sos", "sos_rank", "momentum"] + (
        ["ppoints"] if pp is not None else []
    )
    st.dataframe(
        table[cols],
        hide_index=True,
        width="stretch",
        height=table_height(len(table)),
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "team": st.column_config.TextColumn("Team"),
            "conference": st.column_config.TextColumn("Conference"),
            "record": st.column_config.TextColumn("Record", width="small"),
            "rating": st.column_config.NumberColumn(rating_label, format="%.3f"),
            "sos": st.column_config.NumberColumn(
                "SOS", format="%.3f", help="Mean opponent rating, same units as the rating"
            ),
            "sos_rank": st.column_config.NumberColumn("SOS #", width="small"),
            "momentum": st.column_config.NumberColumn(
                "Momentum", format="%+.1f", help="Points above expectation, last 3 games"
            ),
            "ppoints": st.column_config.NumberColumn("PPoints", format="%.1f"),
        },
    )
    st.download_button("Download CSV", table.to_csv(index=False).encode(), f"ratings_{season}_{method}.csv", "text/csv")

    value_col = sort_by if sort_by in {"rating", "ppoints", "sos", "momentum"} else "rating"
    st.plotly_chart(
        rating_bar(
            table,
            top=top_n,
            value_col=value_col,
            value_label=rating_label if value_col == "rating" else value_col.title(),
            title=f"Top {len(table)} by {value_col}",
            theme=theme,
        ),
        width="stretch",
    )
    with st.expander("Rating vs. schedule strength", expanded=True):
        st.plotly_chart(rating_sos_scatter(table, theme=theme), width="stretch")

# --- team ----------------------------------------------------------------------------------------

with tab_team:
    full = team_table(names, games, result, cal, conf_map)
    team = st.selectbox("Team", real_names, index=real_names.index(top_team))
    row = full[full["team"] == team].iloc[0]
    proj = project_team(team, games, result, cal, recs[team])
    tiles = [
        ("Rank", f"#{int(row['rank'])}", conf_map.get(team) or "Independent"),
        ("Record", row["record"], f"{len(proj.remaining)} game{'s' if len(proj.remaining) != 1 else ''} left"),
        (rating_label, f"{row['rating']:.3f}", None),
        ("SOS rank", f"#{int(row['sos_rank'])}", f"SOS {row['sos']:.3f}"),
        ("Momentum", f"{row['momentum']:+.1f} pts", "last 3 games vs. expectation"),
    ]
    if proj.remaining:
        tiles.append(
            (
                "Projected record",
                f"{proj.expected_final_wins:.1f}-{proj.expected_final_losses:.1f}",
                f"win out {proj.p_win_out:.0%}",
            )
        )
    kpi_row(tiles)

    compare = st.multiselect("Compare with", [n for n in real_names if n != team], max_selections=7)
    snaps = snapshots_cached(method, params_json, data_key, names, games, not demo, ws)
    metric_choice = st.radio("Trajectory", ["rank", "rating"], horizontal=True)
    st.plotly_chart(
        trajectory_chart(snaps, [team, *compare], metric=metric_choice, title="Week by week", theme=theme),
        width="stretch",
    )

    sched = schedule_table(team, games, result, cal)
    st.dataframe(
        sched,
        hide_index=True,
        width="stretch",
        height=table_height(len(sched)),
        column_config={
            "week": st.column_config.NumberColumn("Wk", width="small"),
            "type": st.column_config.TextColumn("Type", width="small"),
            "opponent": st.column_config.TextColumn("Opponent"),
            "opp_rank": st.column_config.NumberColumn("Opp #", width="small"),
            "site": st.column_config.TextColumn("Site", width="small"),
            "expected_margin": st.column_config.NumberColumn("Expected", format="%+.1f"),
            "win_prob": st.column_config.ProgressColumn("Win prob", format="percent", min_value=0, max_value=1),
            "result": st.column_config.TextColumn("W/L", width="small"),
            "score": st.column_config.TextColumn("Score", width="small"),
            "actual_margin": st.column_config.NumberColumn("Actual", format="%+d"),
            "vs_expected": st.column_config.NumberColumn("vs. exp.", format="%+.1f"),
        },
    )
    if proj.remaining:
        st.plotly_chart(
            win_distribution_chart(proj.distribution, proj.record.wins, team=team, theme=theme), width="stretch"
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
    proj_df = proj_df.sort_values(proj_sort, ascending=proj_sort == "rank")
    st.dataframe(
        proj_df[
            [
                "rank",
                "team",
                "conference",
                "record",
                "remaining",
                "proj_record",
                "p_6_wins",
                "p_8_wins",
                "p_10_wins",
                "p_12_wins",
                "p_win_out",
                "next_opponent",
                "next_site",
                "next_win_prob",
            ]
        ],
        hide_index=True,
        width="stretch",
        height=table_height(len(proj_df)),
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "team": "Team",
            "conference": "Conference",
            "record": st.column_config.TextColumn("Record", width="small"),
            "remaining": st.column_config.NumberColumn("Left", width="small"),
            "proj_record": st.column_config.TextColumn("Projected", help="Expected final wins-losses"),
            "p_6_wins": st.column_config.ProgressColumn("6+ wins", format="percent", min_value=0, max_value=1),
            "p_8_wins": st.column_config.ProgressColumn("8+ wins", format="percent", min_value=0, max_value=1),
            "p_10_wins": st.column_config.ProgressColumn("10+ wins", format="percent", min_value=0, max_value=1),
            "p_12_wins": st.column_config.ProgressColumn("12+ wins", format="percent", min_value=0, max_value=1),
            "p_win_out": st.column_config.NumberColumn("Win out", format="percent"),
            "next_opponent": "Next opponent",
            "next_site": st.column_config.TextColumn("Site", width="small"),
            "next_win_prob": st.column_config.NumberColumn("Next win %", format="percent"),
        },
    )
    st.download_button(
        "Download projections CSV",
        proj_df.to_csv(index=False).encode(),
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
    st.plotly_chart(
        conference_dots(conf_df, title="Conference strength index (0.6 = weakest, 1.6 = strongest)", theme=theme),
        width="stretch",
    )

# --- matchup ----------------------------------------------------------------------------------------

with tab_match:
    ranked_names = [t for t, _ in result.ranked()]
    h1, h2, h3 = st.columns([2, 2, 1])
    home = h1.selectbox("Home", ranked_names, index=0)
    away = h2.selectbox("Away", ranked_names, index=1 if len(ranked_names) > 1 else 0)
    neutral = h3.toggle("Neutral site")
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
        df_m,
        hide_index=True,
        width="stretch",
        height=table_height(len(df_m)),
        column_config={
            "method": "Method",
            "line": "Line",
            "home_margin": st.column_config.NumberColumn("Home margin", format="%+.1f"),
            "p_home": st.column_config.ProgressColumn("Home win prob", format="percent", min_value=0, max_value=1),
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
    if chosen:
        table_bt, detail = backtest_cached(tuple(chosen), start_week, params_json, data_key, names, all_games)
        table_bt = table_bt.assign(method=[MODEL_LABELS.get(m, m) for m in table_bt["method"]])
        st.dataframe(
            table_bt,
            hide_index=True,
            width="stretch",
            height=table_height(len(table_bt)),
            column_config={
                "method": "Method",
                "n": st.column_config.NumberColumn("Games", width="small"),
                "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
                "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
                "log_loss": st.column_config.NumberColumn("Log loss", format="%.4f"),
                "mae": st.column_config.NumberColumn("MAE (pts)", format="%.2f"),
                "rmse": st.column_config.NumberColumn("RMSE (pts)", format="%.2f"),
                "seasons": "Seasons",
                "weeks": "Weeks",
            },
        )
        metric = st.selectbox("Weekly metric", ["accuracy", "brier", "log_loss", "mae"], index=0)
        wl, wr = st.columns(2)
        with wl:
            st.plotly_chart(
                weekly_metric_chart(
                    {MODEL_LABELS[m]: d[1] for m, d in detail.items()},
                    metric=metric,
                    title=f"{metric} by week",
                    theme=theme,
                ),
                width="stretch",
            )
        with wr:
            st.plotly_chart(
                calibration_chart({MODEL_LABELS[m]: d[2] for m, d in detail.items()}, title="Reliability", theme=theme),
                width="stretch",
            )
        if n_prior_seasons > 0:
            with st.expander("By season"):
                for m, d in detail.items():
                    st.markdown(f"**{MODEL_LABELS[m]}**")
                    st.dataframe(d[3], hide_index=True, width="stretch", height=table_height(len(d[3])))
        with st.expander("Out-of-sample predictions"):
            all_preds = pd.concat([d[0].assign(method=m) for m, d in detail.items()], ignore_index=True)
            st.dataframe(all_preds, hide_index=True, width="stretch")
            st.download_button(
                "Download predictions CSV", all_preds.to_csv(index=False).encode(), f"backtest_{season}.csv", "text/csv"
            )

# --- methodology ----------------------------------------------------------------------------------------

with tab_doc:
    if METHODOLOGY.exists():
        st.markdown(METHODOLOGY.read_text(encoding="utf-8"))
    else:
        st.info("See docs/METHODOLOGY.md in the repository.")
