# Changelog

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
