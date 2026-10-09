# BrandonCFBRatings 🏈

[![CI](https://github.com/bpaikoff/BrandonCFBRatings/actions/workflows/ci.yml/badge.svg)](https://github.com/bpaikoff/BrandonCFBRatings/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A college football ratings engine with five rating systems, résumé analytics, a walk-forward
backtesting harness and an interactive dashboard. Every model is calibrated to point spreads and
win probabilities on the same scale, so you can ask the only question that matters: **which
system actually predicts games?**

| | |
|---|---|
| **Models** | Colley, Massey (ridge, recency-weighted), Elo (margin-of-victory), Bradley-Terry, Hybrid blend |
| **Analytics** | strength of schedule, momentum in points, conference strength index, PPoints résumé score, weekly rank trajectories |
| **Backtesting** | out-of-sample walk-forward evaluation: accuracy, Brier, log loss, MAE/RMSE, calibration, home-team baseline, Vegas closing-line comparison, grid-search tuning |
| **Interfaces** | `cfbratings` CLI with CSV/JSON export · Streamlit dashboard · importable Python package |
| **Data** | [CollegeFootballData.com](https://collegefootballdata.com) API with a TTL JSON cache, or a synthetic season generator for offline use |

## Quickstart

```bash
git clone https://github.com/bpaikoff/BrandonCFBRatings.git
cd BrandonCFBRatings
pip install -e ".[app]"          # add ,dev for tests + lint

# No API key? Everything runs on a synthetic season with known ground truth.
cfbratings --demo rank
cfbratings --demo backtest

# Real data: get a free key at https://collegefootballdata.com/key
cp .env.example .env             # then set CFBD_API_KEY=...
cfbratings rank --year 2025 --method hybrid --top 25
streamlit run apps/streamlit_app.py
```

## Command line

| Command | What it does |
|---|---|
| `cfbratings rank` | ratings table with record, SOS, momentum and PPoints (`--method`, `--top`, `--sort-by`, `--conference`, `--csv/--json`) |
| `cfbratings team "Ohio State"` | one team's schedule with expected vs. actual margins and its week-by-week rank |
| `cfbratings conferences` | conference strength index |
| `cfbratings predict "Ohio State" "Michigan" [--neutral]` | spread and win probability from every model |
| `cfbratings backtest` | walk-forward comparison of all methods (`--methods`, `--start-week`, `--by-week`, `--calibration`, `--lines` for a Vegas comparison, `--predictions-csv`) |
| `cfbratings tune --method elo --grid "k=15,25,40;hfa=2,2.5,3"` | grid search hyper-parameters against any backtest metric |
| `cfbratings snapshot` | compute and cache weekly rating snapshots |
| `cfbratings cache list\|clear` | inspect the JSON cache |

Global flags: `--year`, `--season-type {regular,postseason,both}`, `--demo`, `--refresh`, `--cache-dir`, `--param KEY=VALUE` (per-model overrides such as `--param k=30`).

Example (synthetic season, so the numbers are illustrative):

```
$ cfbratings --demo rank --top 5
Hybrid ratings — synthetic season 2026 (seed 7)
calibration: HFA +3.60 pts, slope 8.923, sigma 15.8 pts (cross_val, n=388)

 rank             team   conference record  rating    sos  sos_rank  momentum  ppoints
    1 Prairie Northern       Big 12    9-1   1.955  0.225        18     2.359   58.324
    2    Lakes Western      Big Ten   14-0   1.847 -0.265        49     5.432   80.193
    3   Lakes Northern      Big Ten   10-2   1.798  0.517         5     5.126   71.324
    4  Heartland State Mid-American   11-1   1.615 -0.050        34    -5.465   49.751
    5 Mountain Central Mountain West  11-1   1.500 -0.302        53    -3.158   43.992

$ cfbratings --demo backtest --start-week 4
               method   n  accuracy  brier  log_loss     mae    rmse weeks
               colley 301    0.7143 0.1909    0.5671 13.4276 16.9273  4-14
               massey 301    0.7243 0.1811    0.5458 12.9153 16.2227  4-14
                  elo 301    0.7276 0.1873    0.5630 13.3991 16.8007  4-14
        bradley_terry 301    0.7243 0.1913    0.5696 13.5428 17.0081  4-14
               hybrid 301    0.7575 0.1820    0.5478 12.9716 16.3349  4-14
home_prior (baseline) 301    0.5515 0.2476    0.6883 16.4269 20.2327  4-14
```

## Dashboard

`streamlit run apps/streamlit_app.py` opens a six-tab app that works in demo mode without a key:

* **Rankings** – sortable table, conference filter, ranked bar chart coloured by conference, rating-vs-SOS scatter, CSV download
* **Team** – schedule with expected margin and win probability per game, week-by-week rank trajectory, compare up to eight teams
* **Conferences** – robust strength index with aggregate records
* **Matchup** – pick any two teams for every model's spread and win probability plus a consensus line
* **Backtest** – run the walk-forward evaluation live: metrics table, per-week accuracy, reliability diagram, downloadable predictions
* **Methodology** – the formulas, rendered from `docs/METHODOLOGY.md`

Every model hyper-parameter (home-field advantage, Elo K, Massey ridge and recency half-life, hybrid weights, …) is a sidebar slider; results are cached so interaction is instant.

## How it works

Full derivations live in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). In brief:

| Model | Uses | Core idea |
|---|---|---|
| **Colley** | W/L | Laplace-prior linear system `(2+nᵢ)rᵢ − Σ r_opp = 1 + (wᵢ−lᵢ)/2`; immune to margin, blind to it too |
| **Massey** | margins | ridge least squares on `r_home − r_away = margin − HFA`; margins capped, optional recency weighting, automatically centred |
| **Elo** | sequence + margins | FiveThirtyEight-style update with `ln(margin+1)` multiplier damped by the winner's edge; HFA converted to Elo (25/pt); regression between seasons only |
| **Bradley-Terry** | W/L | penalised logistic MLE `P(home) = σ(r_h − r_a + η)` fitted by Newton's method |
| **Hybrid** | all | weighted blend of z-scored component ratings |

**Calibration.** Ratings are turned into spreads and probabilities with `margin = HFA + slope × (r_h − r_a)` and `P = Φ(margin/σ)`. The calibrator is fitted by *k-fold cross-validation* (each game predicted by a model that never saw it) with a degrees-of-freedom-corrected σ, which stops a one-parameter-per-team model from looking overconfident in September.

**PPoints** is a committee-style résumé score: wins earn tier points by the opponent's rank (blended between its rank *entering that week* and its rank *now*), multiplied by a road/neutral bonus and the opponent's conference strength; losses cost `4 ÷ tier`, so losing to #3 barely stings while losing to #90 costs four points. Non-conference games earn a bonus and a softer loss penalty.

## Backtesting

The engine refits the model each week on every earlier game, calibrates on cross-validated and previously held-out predictions, and scores that week's games out-of-sample. Two reference rows accompany every run: **home_prior** (always pick the home team) and, with `--lines`, the **Vegas closing spread**. A system that cannot beat the home-team baseline is noise; one that approaches the market is excellent.

```bash
cfbratings backtest --year 2024 --methods massey,elo,hybrid --start-week 4 --lines --by-week
cfbratings tune --year 2024 --method massey --grid "ridge_lambda=0.1,0.5,1,2,4;half_life_weeks=0,4,8" --metric brier
```

From Python:

```python
from cfbratings import fit_ratings, MarginCalibrator
from cfbratings.data import load_season
from cfbratings.backtest import compare_methods

teams, games = load_season(2024)
names = [t.school for t in teams]
table, results = compare_methods(names, games, ["massey", "elo", "hybrid"], start_week=4)
print(table)
```

## Configuration

Settings come from environment variables or a `.env` file (see [`.env.example`](.env.example)):

| Variable | Default | Meaning |
|---|---|---|
| `CFBD_API_KEY` | – | CollegeFootballData API key |
| `CFB_YEAR` / `CFB_SEASON_TYPE` | current season / `both` | default season |
| `CFB_CACHE_DIR` / `CFB_CACHE_TTL_HOURS` | `data/cache` / `6` | JSON cache (finished seasons never expire) |
| `CFB_HFA` | `2.5` | home-field advantage in points (never applied on neutral sites) |
| `CFB_COLLEY_PRIOR` | `2.0` | Colley pseudo-games |
| `CFB_MASSEY_LAMBDA` / `CFB_MASSEY_MAX_MARGIN` / `CFB_MASSEY_HALF_LIFE` | `1.0` / `50` / `0` | Massey ridge, margin cap, recency half-life (weeks) |
| `CFB_ELO_K` / `CFB_ELO_INIT` / `CFB_ELO_PER_POINT` / `CFB_ELO_REGRESS` / `CFB_ELO_MOV` | `25` / `1500` / `25` / `0.33` / `1` | Elo parameters |
| `CFB_BT_RIDGE` | `1.0` | Bradley-Terry L2 penalty |
| `CFB_HYBRID_WEIGHTS` | `colley=⅓,massey=⅓,elo=⅓` | hybrid blend |

## Project layout

```
cfbratings/
  config.py          settings from env / .env
  data/              Game & Team schema, JSON cache, CFBD client, synthetic seasons
  models/            colley, massey, elo, bradley_terry, hybrid + registry
  predict.py         MarginCalibrator: ratings → spreads & win probabilities
  analytics.py       records, SOS, momentum, conference strength, team/schedule tables
  ppoints.py         PPoints résumé metric
  snapshots.py       weekly rating snapshots (cached)
  backtest/          walk-forward engine, metrics, grid search
  viz/               Plotly charts (colour-blind-safe palette)
  cli.py             `cfbratings` command
apps/streamlit_app.py  dashboard
docs/METHODOLOGY.md    formulas
tests/                 pytest suite (models, calibration, analytics, backtest, CLI, dashboard)
```

## Development

```bash
make install     # pip install -e ".[dev]"
make lint        # ruff check + format check
make test        # pytest (57 tests, ~1 min incl. dashboard smoke test)
make test-fast   # skip the dashboard smoke test
make app         # streamlit run apps/streamlit_app.py
```

CI runs lint and the test-suite on Python 3.11–3.13 for every push and pull request. Models and the backtest engine are validated against synthetic seasons with known team strengths, so the tests assert that each system recovers the true ordering and beats the home-team baseline out of sample.

## Roadmap

* Multi-season warm starts in the CLI (the engine already accepts `warmup_games`)
* Pool FCS opponents into a single pseudo-team instead of dropping those games
* Preseason priors from returning production / recruiting
* Playoff-probability simulation from calibrated win probabilities

## Acknowledgements

Game data from [CollegeFootballData.com](https://collegefootballdata.com). Rating methods follow Colley (2002), Massey (1997), the FiveThirtyEight Elo formulation, and Bradley & Terry (1952).
