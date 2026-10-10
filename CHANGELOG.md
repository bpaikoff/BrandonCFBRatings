# Changelog

## 1.2.0 — 2026-10-10

### Features
- **Rest-of-season projections** (`cfbratings.projections`): exact Poisson-binomial win
  distributions for every team's remaining schedule — projected record, P(6+/8+/10+/12+
  wins), P(win out), next opponent and win probability. New `cfbratings project` command
  and a 🔭 Projections tab; the Team tab shows a projected record and win-total chart.
- Synthetic seasons can be partially played (`played_through`, `--demo-weeks`, dashboard
  "weeks played" slider) so projections are demonstrable offline.

### Dashboard UX
- Responsive stat tiles replace `st.metric` (no more truncated "Prairie W…" at laptop width).
- Sidebar reorganised: data source, season and rating method first; data options and
  model parameters in collapsed sections; demo-only controls hidden outside demo mode.
- Tables size to their rows (the full top-25 is visible without scrolling); the rating
  column names its units (points / Elo / z-score / 0-1 / log-odds).
- Trajectory chart keeps a sensible rank axis even for a team that never moves; single
  series charts drop the redundant legend; scatter labels no longer collide.
- Conference strength is a dot plot on its 0.6-1.6 band instead of bars from zero, and the
  tab no longer squeezes the table.
- Charts pick ink, grid and series colours for the active light/dark theme.

## 1.1.0 — 2026-10-10

### Formulas
- **Preseason priors / warm starts.** Every model accepts a `prior` built from the previous
  season's regressed ratings (`cfbratings.priors`): Colley plays its pseudo-games at the
  prior, Massey shrinks toward it with a dedicated `prior_lambda` (default 3.0), Elo starts
  from it, Bradley-Terry penalises distance from it, Hybrid takes one prior per component.
  A single `season_regress` setting (default 0.33, env `CFB_SEASON_REGRESS`) replaces the
  Elo-only knob. The previous season's calibrator is carried across the boundary so
  week-1 spreads keep a sensible scale.
- **FCS pooling.** Games against non-FBS opponents are no longer discarded: the opponent
  becomes one shared pseudo-team, `FCS (pooled)`, that is rated but never ranked
  (`CFB_FCS=pool|drop`, `--fcs`).
- Massey recency weighting now measures age on an absolute timeline across seasons.

### Backtesting
- The walk-forward engine handles multi-season inputs, chaining priors season to season,
  and reports a `seasons` column plus `BacktestResult.by_season()`.
- Multi-season CLI: `--years 2022-2024`, `--warmup-years N`; demo mode generates
  correlated synthetic seasons (`synthetic_seasons`).

### Presentation
- Dashboard sidebar gains "Warm-start from prior seasons" and "Games vs. FCS opponents";
  the Backtest tab evaluates across all loaded seasons and shows per-season metrics.
- CLI output states whether ratings are warm-started and from how many seasons.

### Data
- `load_seasons` fetches several seasons and merges rosters; `Game` keeps the original
  school name of a pooled opponent for display.

## 1.0.0 — 2026-10-09

A ground-up modernisation of the original Copilot-era prototype.

### Formulas
- **Elo**: home-field advantage is now converted from points to Elo (25 Elo per point, so
  2.5 pts → 62.5 Elo) instead of being added as 2.1 Elo points; the margin-of-victory
  multiplier uses the *winner's* rating edge as in the FiveThirtyEight formulation; regression
  to the mean happens between seasons (33 %) rather than 20 % every week.
- **Massey**: ridge regression replaces the overwritten sum-to-zero row; neutral-site games no
  longer receive home-field advantage; optional recency weighting; default ridge raised to 1.0
  after backtesting.
- **Colley**: unchanged mathematics, cleaned implementation, honours ties.
- **New Bradley-Terry model** (penalised logistic regression with a home-advantage term).
- **Hybrid** can blend any subset of models (default Colley + Massey + Elo).
- **Unified calibration** turns every model's ratings into point spreads and win
  probabilities, with cross-validated fitting and a degrees-of-freedom correction so early
  season probabilities are not overconfident.
- **Momentum** is now measured in points above expectation for every method.
- **PPoints** rewritten: per-week rank maps are computed once (not once per game), uses the
  season actually requested (the old code always read `settings.year` snapshots), uses the
  pre-game snapshot, treats neutral sites as road games, and exposes every constant via
  `PPointsConfig`.

### Backtesting (new)
- Walk-forward engine with rolling out-of-sample calibration, home-team baseline and an
  optional Vegas closing-line comparison (`cfbratings backtest --lines`).
- Metrics: accuracy, Brier, log loss, MAE, RMSE, ATS accuracy, reliability tables.
- `grid_search` / `cfbratings tune` for hyper-parameter sweeps.

### Data layer
- Typed `Game` / `Team` dataclasses that accept both camelCase and snake_case CFBD payloads.
- Postseason weeks are renumbered after the regular season so weeks form a timeline.
- JSON cache with TTL; finished seasons never expire; snapshots keyed by model parameters
  (the old snapshot cache silently served Hybrid ratings under every method name).
- Synthetic season generator with known ground truth (`--demo`) so everything runs offline.

### Presentation
- Subcommand CLI (`rank`, `team`, `conferences`, `predict`, `backtest`, `tune`, `snapshot`,
  `cache`) with CSV/JSON export.
- Streamlit dashboard rebuilt with tabs (Rankings, Team, Conferences, Matchup, Backtest,
  Methodology), cached computation, colour-blind-safe Plotly charts and a demo mode.
- README, methodology document, changelog.

### Engineering
- Single `pyproject.toml` (setup.py removed), console script, optional extras.
- pytest suite (schema, cache, client, models, calibration, analytics, PPoints, snapshots,
  backtest, CLI, dashboard smoke test), ruff lint/format, GitHub Actions CI, pre-commit.
