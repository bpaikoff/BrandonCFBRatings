"""College Football Ratings dashboard.

Run with ``streamlit run apps/streamlit_app.py``. Works without an API key in demo mode
(a synthetic season with known ground truth); set ``CFBD_API_KEY`` to use real data.
"""

from __future__ import annotations

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
from cfbratings.snapshots import weekly_snapshots
from cfbratings.viz import (
    calibration_chart,
    conference_bar,
    rating_bar,
    rating_sos_scatter,
    trajectory_chart,
    weekly_metric_chart,
)

st.set_page_config(page_title="CFB Ratings", page_icon="🏈", layout="wide")

METHODOLOGY = ROOT / "docs" / "METHODOLOGY.md"


# --- cached computations ---------------------------------------------------------------


@st.cache_data(show_spinner="Loading season…")
def load_data(demo: bool, year: int, warmup: int, season_type: str, fcs: str, refresh_token: int, seed: int):
    years = list(range(year - warmup, year + 1))
    span = f"{years[0]}-{years[-1]}" if warmup else str(year)
    if demo:
        chain = synthetic_seasons(years, seed=seed)
        games = [g for season in chain for g in season.games]
        return (
            [_team_dict(t) for t in chain[-1].teams],
            [g.to_dict() for g in games],
            f"Synthetic season {span} (seed {seed})",
        )
    teams, games = load_seasons(years, season_type, force_refresh=refresh_token > 0, fcs=fcs)
    season_type = f"{span} · {season_type} · FCS {'pooled' if fcs == 'pool' else 'dropped'}"
    return [_team_dict(t) for t in teams], [g.to_dict() for g in games], f"CollegeFootballData · {year} · {season_type}"


def _team_dict(t: Team) -> dict:
    return {
        "school": t.school,
        "conference": t.conference,
        "classification": t.classification,
        "abbreviation": t.abbreviation,
    }


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


# --- sidebar -----------------------------------------------------------------------------

with st.sidebar:
    st.title("🏈 CFB Ratings")
    has_key = bool(settings.api_key)
    source = st.radio(
        "Data source",
        ["CollegeFootballData API", "Demo (synthetic season)"],
        index=0 if has_key else 1,
        help="Demo mode generates a season with known team strengths so every feature works offline.",
    )
    demo = source.startswith("Demo")
    if not has_key and not demo:
        st.warning("No `CFBD_API_KEY` found. Add one to `.env` or switch to demo mode.")
    year = int(st.number_input("Season", min_value=2000, max_value=2035, value=int(settings.year), step=1))
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
    season_type = st.selectbox("Season type", ["both", "regular", "postseason"], index=0, disabled=demo)
    fcs = st.radio(
        "Games vs. FCS opponents",
        ["pool", "drop"],
        index=0 if settings.fcs_handling == "pool" else 1,
        horizontal=True,
        disabled=demo,
        help="Pool: every FCS opponent becomes one shared pseudo-team so the result counts. Drop: ignore those games.",
    )
    seed = int(st.number_input("Demo seed", min_value=0, max_value=9999, value=7, step=1)) if demo else 7
    method = st.selectbox(
        "Rating method", MODEL_NAMES, index=MODEL_NAMES.index("hybrid"), format_func=lambda m: MODEL_LABELS[m]
    )

    with st.expander("Model parameters"):
        hfa = st.slider("Home-field advantage (pts)", 0.0, 5.0, float(settings.home_field_adv), 0.1)
        colley_prior = st.slider("Colley prior strength", 0.0, 6.0, float(settings.colley_prior_strength), 0.5)
        ridge_lambda = st.slider("Massey ridge lambda", 0.0, 5.0, float(settings.massey_ridge_lambda), 0.1)
        half_life = st.slider(
            "Massey recency half-life (weeks, 0 = off)", 0.0, 10.0, float(settings.massey_half_life_weeks), 0.5
        )
        elo_k = st.slider("Elo K-factor", 5.0, 80.0, float(settings.elo_k), 1.0)
        elo_regress = st.slider("Season regression (all models)", 0.0, 1.0, float(settings.season_regress), 0.05)
        bt_ridge = st.slider("Bradley-Terry ridge", 0.1, 5.0, float(settings.bt_ridge), 0.1)
        st.markdown("**Hybrid weights**")
        w_colley = st.slider("Colley", 0.0, 1.0, float(settings.hybrid_weight_map.get("colley", 0.0)), 0.05)
        w_massey = st.slider("Massey", 0.0, 1.0, float(settings.hybrid_weight_map.get("massey", 0.0)), 0.05)
        w_elo = st.slider("Elo", 0.0, 1.0, float(settings.hybrid_weight_map.get("elo", 0.0)), 0.05)
        w_bt = st.slider("Bradley-Terry", 0.0, 1.0, float(settings.hybrid_weight_map.get("bradley_terry", 0.0)), 0.05)

    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = 0
    if st.button("Refresh API cache", disabled=demo):
        st.session_state.refresh_token += 1
    st.caption(f"cfbratings v{__version__}")

weights = {
    k: v for k, v in {"colley": w_colley, "massey": w_massey, "elo": w_elo, "bradley_terry": w_bt}.items() if v > 0
}
params = {
    "hfa": hfa,
    "prior_strength": colley_prior,
    "ridge_lambda": ridge_lambda,
    "half_life_weeks": half_life,
    "k": elo_k,
    "season_regress": elo_regress,
    "ridge": bt_ridge,
    "weights": weights or {"massey": 1.0},
}
params_json = json.dumps(params, sort_keys=True)

# --- data ------------------------------------------------------------------------------

try:
    teams_raw, games_raw, source_label = load_data(
        demo, year, warmup, season_type, fcs, st.session_state.refresh_token, seed
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
n_prior_seasons = len({g.season for g in all_games if g.has_score}) - 1
data_key = f"{demo}-{year}-{warmup}-{season_type}-{fcs}-{seed}-{len(all_games)}-{n_played}-{last_week}"

st.title("College Football Ratings")
warm_note = (
    f" · warm-started from {n_prior_seasons} prior season{'s' if n_prior_seasons > 1 else ''}"
    if n_prior_seasons > 0
    else ""
)
st.caption(f"{source_label} · {MODEL_LABELS[method]} · {n_played} games through week {last_week}{warm_note}")

if n_played == 0 and n_prior_seasons <= 0:
    st.info("No completed games yet for this season. Pick another season, add prior seasons, or switch to demo mode.")
    st.stop()

ws = warm_start_cached(method, params_json, data_key, names, all_games, season)
result, cal = fit_cached(method, params_json, data_key, names, games, ws)
recs = records(names, games)
top_team, top_rating = result.ranked()[0]

m1, m2, m3, m4 = st.columns([1.6, 1, 1, 1])
m1.metric("No. 1", top_team, str(recs[top_team]))
m2.metric(
    "Home-field advantage", f"{cal.intercept:+.1f} pts", help="Calibrated from this season's games (cross-validated)."
)
m3.metric(
    "Spread uncertainty (sigma)", f"{cal.sigma:.1f} pts", help="Residual standard deviation of predicted margins."
)
m4.metric("Calibration games", f"{cal.n_games}", cal.source.replace("_", " "))

tab_rank, tab_team, tab_conf, tab_match, tab_bt, tab_doc = st.tabs(
    ["📊 Rankings", "🏟️ Team", "🏆 Conferences", "🔮 Matchup", "🧪 Backtest", "📖 Methodology"]
)

# --- rankings -----------------------------------------------------------------------------

with tab_rank:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    top_n = int(c1.slider("Show top", 10, len(names), min(25, len(names)), 5))
    conf_filter = c2.selectbox("Conference", ["All", *sorted({c or "Independent" for c in conf_map.values()})])
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
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "team": st.column_config.TextColumn("Team"),
            "conference": st.column_config.TextColumn("Conference"),
            "record": st.column_config.TextColumn("Record", width="small"),
            "rating": st.column_config.NumberColumn("Rating", format="%.3f"),
            "sos": st.column_config.NumberColumn("SOS", format="%.3f", help="Mean opponent rating"),
            "sos_rank": st.column_config.NumberColumn("SOS #", width="small"),
            "momentum": st.column_config.NumberColumn(
                "Momentum", format="%+.1f", help="Points above expectation, last 3 games"
            ),
            "ppoints": st.column_config.NumberColumn("PPoints", format="%.1f"),
        },
    )
    st.download_button("Download CSV", table.to_csv(index=False).encode(), f"ratings_{year}_{method}.csv", "text/csv")

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            rating_bar(table, top=top_n, value_col=sort_by if sort_by != "wins" else "rating"), width="stretch"
        )
    with right:
        st.plotly_chart(rating_sos_scatter(table, title="Rating vs. schedule strength"), width="stretch")

# --- team ---------------------------------------------------------------------------------

with tab_team:
    full = team_table(names, games, result, cal, conf_map)
    team = st.selectbox("Team", real_names, index=real_names.index(top_team))
    row = full[full["team"] == team].iloc[0]
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Rank", f"#{int(row['rank'])}")
    t2.metric("Record", row["record"])
    t3.metric("Rating", f"{row['rating']:.3f}")
    t4.metric("SOS rank", f"#{int(row['sos_rank'])}")
    t5.metric("Momentum", f"{row['momentum']:+.1f} pts")

    compare = st.multiselect("Compare with", [n for n in real_names if n != team], max_selections=7)
    snaps = snapshots_cached(method, params_json, data_key, names, games, not demo, ws)
    metric_choice = st.radio("Trajectory", ["rank", "rating"], horizontal=True)
    st.plotly_chart(
        trajectory_chart(snaps, [team, *compare], metric=metric_choice, title="Week-by-week"), width="stretch"
    )

    sched = schedule_table(team, games, result, cal)
    st.dataframe(
        sched,
        hide_index=True,
        width="stretch",
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

# --- conferences ---------------------------------------------------------------------------

with tab_conf:
    conf_df = conference_table(result.ratings, conf_map, recs)
    conf_df = conf_df[conf_df["conference"] != "FCS"]
    cl, cr = st.columns([2, 3])
    with cl:
        st.dataframe(
            conf_df,
            hide_index=True,
            width="stretch",
            column_config={
                "conference": "Conference",
                "strength_index": st.column_config.NumberColumn("Index", format="%.2f"),
                "teams": st.column_config.NumberColumn("Teams", width="small"),
                "mean_rating": st.column_config.NumberColumn("Mean", format="%.3f"),
                "median_rating": st.column_config.NumberColumn("Median", format="%.3f"),
                "top_team": "Best team",
                "wins": st.column_config.NumberColumn("W", width="small"),
                "losses": st.column_config.NumberColumn("L", width="small"),
            },
        )
    with cr:
        st.plotly_chart(conference_bar(conf_df, title="Conference strength index"), width="stretch")

# --- matchup --------------------------------------------------------------------------------

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
    k1, k2, k3 = st.columns(3)
    fav = home if consensus_margin >= 0 else away
    k1.metric("Consensus line", f"{fav} -{abs(consensus_margin):.1f}")
    k2.metric(f"{home} win probability", f"{consensus_p:.0%}")
    k3.metric(f"{away} win probability", f"{1 - consensus_p:.0%}")
    st.dataframe(
        df_m,
        hide_index=True,
        width="stretch",
        column_config={
            "method": "Method",
            "line": "Line",
            "home_margin": st.column_config.NumberColumn("Home margin", format="%+.1f"),
            "p_home": st.column_config.ProgressColumn("Home win prob", format="percent", min_value=0, max_value=1),
        },
    )

# --- backtest ---------------------------------------------------------------------------------

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
                    {MODEL_LABELS[m]: d[1] for m, d in detail.items()}, metric=metric, title=f"{metric} by week"
                ),
                width="stretch",
            )
        with wr:
            st.plotly_chart(
                calibration_chart({MODEL_LABELS[m]: d[2] for m, d in detail.items()}, title="Reliability"),
                width="stretch",
            )
        if n_prior_seasons > 0:
            with st.expander("By season"):
                for m, d in detail.items():
                    st.markdown(f"**{MODEL_LABELS[m]}**")
                    st.dataframe(d[3], hide_index=True, width="stretch")
        with st.expander("Out-of-sample predictions"):
            all_preds = pd.concat([d[0].assign(method=m) for m, d in detail.items()], ignore_index=True)
            st.dataframe(all_preds, hide_index=True, width="stretch")
            st.download_button(
                "Download predictions CSV", all_preds.to_csv(index=False).encode(), f"backtest_{year}.csv", "text/csv"
            )

# --- methodology ------------------------------------------------------------------------------

with tab_doc:
    if METHODOLOGY.exists():
        st.markdown(METHODOLOGY.read_text(encoding="utf-8"))
    else:
        st.info("See docs/METHODOLOGY.md in the repository.")
