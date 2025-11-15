import streamlit as st
from cfbratings.config import settings
from cfbratings.io import fetch_teams, fetch_games
from cfbratings.models.colley import build_colley, solve_colley
from cfbratings.models.massey import build_massey, solve_massey
from cfbratings.models.elo import run_elo
from cfbratings.models.hybrid import hybrid_rating
from cfbratings.analytics import records, strength_of_schedule, momentum, ppoints

st.set_page_config(page_title="CFB Ratings Dashboard", layout="wide")

st.title("CFB Ratings — Real-Time with JSON Caching")
col1, col2, col3 = st.columns(3)
with col1:
    year = st.number_input("Year", min_value=2010, max_value=2030, value=settings.year, step=1)
with col2:
    season_type = st.selectbox("Season type", options=["regular", "postseason", "both"], index=["regular","postseason","both"].index(settings.season_type))
with col3:
    method = st.selectbox("Method", options=["colley", "massey", "elo", "hybrid"], index=["colley","massey","elo","hybrid"].index("hybrid"))

refresh = st.checkbox("Force refresh API (overwrite cache)", value=False)

teams = fetch_teams(year, force_refresh=refresh)
games_payload = fetch_games(year, season_type=season_type, force_refresh=refresh)
games = games_payload["data"] if isinstance(games_payload, dict) and "data" in games_payload else games_payload
team_list = [t["school"] for t in teams]

# Controls for method hyperparameters
with st.expander("Advanced settings"):
    hfa = st.slider("Home-field advantage (points)", 0.0, 5.0, settings.home_field_adv, 0.1)
    colley_prior = st.slider("Colley prior strength", 0.0, 5.0, settings.colley_prior_strength, 0.1)
    massey_lambda = st.slider("Massey ridge lambda", 0.0, 0.5, settings.massey_ridge_lambda, 0.01)
    elo_k = st.slider("Elo K-factor", 5.0, 60.0, settings.elo_k, 1.0)
    elo_reg = st.slider("Elo regress-to-mean", 0.0, 0.5, settings.elo_regress_to_mean, 0.01)
    elo_init = st.slider("Elo initial rating", 1200.0, 1800.0, settings.elo_init, 25.0)
    blend_colley = st.slider("Hybrid weight — Colley", 0.0, 1.0, 0.5, 0.05)
    blend_massey = 1.0 - blend_colley

# Compute ratings
if method == "colley":
    C, b = build_colley(team_list, games, prior_strength=colley_prior)
    r = solve_colley(C, b)
    ratings = {team_list[i]: float(r[i]) for i in range(len(team_list))}
elif method == "massey":
    M, mb = build_massey(team_list, games, ridge_lambda=massey_lambda, hfa=hfa)
    r = solve_massey(M, mb)
    ratings = {team_list[i]: float(r[i]) for i in range(len(team_list))}
elif method == "elo":
    ratings = run_elo(team_list, games, init=elo_init, k=elo_k, regress_to_mean=elo_reg, hfa=hfa)
else:
    ratings = hybrid_rating(team_list, games,
                            colley_weight=blend_colley, massey_weight=blend_massey,
                            prior_strength=colley_prior, ridge_lambda=massey_lambda, hfa=hfa)

recs = records(team_list, games)
sos = strength_of_schedule(team_list, games, ratings)
mom = momentum(team_list, games, ratings)
pp = ppoints(team_list, games, ratings)

# Table
st.subheader(f"Top 25 — {method.capitalize()} ({year}, {season_type})")
top_items = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:25]
st.dataframe({
    "Rank": [i+1 for i in range(len(top_items))],
    "Team": [t for t,_ in top_items],
    "Rating": [v for _,v in top_items],
    "Record": [f"{recs[t][0]}-{recs[t][1]}" for t,_ in top_items],
    "SOS": [sos[t] for t,_ in top_items],
    "Momentum": [mom[t] for t,_ in top_items],
    "PPoints": [pp[t] for t,_ in top_items]
})

# Chart
import plotly.express as px
df = {
    "Team": [t for t,_ in top_items],
    "Rating": [v for _,v in top_items],
    "Record": [f"{recs[t][0]}-{recs[t][1]}" for t,_ in top_items],
    "SOS": [sos[t] for t,_ in top_items],
    "Momentum": [mom[t] for t,_ in top_items]
}
fig = px.bar(x=df["Rating"], y=[f"{i+1}. {team} ({rec})" for i,(team,rec) in enumerate(zip(df["Team"], df["Record"]))],
             orientation="h", color=df["SOS"], color_continuous_scale="Blues",
             labels={"x":"Rating","y":"Team","color":"SOS"}, height=700)
st.plotly_chart(fig, use_container_width=True)

# Team details
st.subheader("Team detail")
team_sel = st.selectbox("Select team", options=sorted(team_list))
st.write({
    "Rating": ratings.get(team_sel, 0.0),
    "Record": f"{recs[team_sel][0]}-{recs[team_sel][1]}",
    "SOS": sos.get(team_sel, 0.0),
    "Momentum": mom.get(team_sel, 0.0)
})