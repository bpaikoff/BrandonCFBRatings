# Changelog

## Unreleased

### Real data: five seasons archived, backtested and tuned (17 API calls)
- `data/archive/` now ships the 2021–2025 CollegeFootballData seasons (teams, games and closing
  lines, gzipped), so every backtest, grid search and ranking below reproduces from a fresh clone
  with zero API calls. 17 calls were spent in total: 15 for the five seasons and 2 for a live look
  at the in-progress 2026 season.
- New [`docs/BACKTEST_RESULTS.md`](docs/BACKTEST_RESULTS.md): what the real payload looks like,
  cold single-season and warm multi-season walk-forward results for all five models against the
  home-team baseline and the closing line, by-season, by-week and calibration tables, and the tuning
  grids with their 2025 hold-out. Headline: warm-started Massey posts 0.511 log loss and 13.3 points
  of MAE over 4 457 games (2021–2025 from week 1) against 0.679 / 18.2 for the home-team baseline
  and 0.483 / 12.1 for the closing line; no model finds an edge against the spread (48.7–50.0 %).
- **Default hybrid weights changed** from ⅓/⅓/⅓ to `colley=0.2, massey=0.6, elo=0.2`. Log loss
  falls monotonically with Massey's share on the 2021–2024 tuning chain and on the 2025 hold-out
  alike; the equal blend trailed the new default by 0.007 log loss and 0.25 points of MAE on the
  hold-out (0.009 / 0.38 against the grid optimum at 0.8 Massey, which is no longer a blend).
  Massey (`prior_lambda=3`, no recency weighting), Elo (`k=25`, `hfa=2.5`) and
  `season_regress=0.33` are unchanged: their grid winners gain under 0.005 log loss on the
  hold-out. `.env.example`, the README and `docs/METHODOLOGY.md` state the new default.
- `tests/fixtures/games_2024_sample.json` is a real trimmed 2024 payload (Ohio State, Notre Dame,
  Oregon, Texas: 58 games, 58 line rows, 46 roster rows) and `tests/test_real_payload_sample.py`
  pins what the data looks like: every FBS bowl and CFP round is `week: 1` of
  `seasonType: postseason`, first-round playoff games are on campus and not neutral, the only FCS
  opponent pools, the four records come out right, and the spread sign holds across all three
  books (across the whole archive: 17 361 book rows, zero contradictions).
- `trim_payload.py` keeps the roster row of every FBS school involved in the sample, so
  roster-based pooling works on it; `data/archive/README.md` lists what is archived.

### Dashboard and data polish
- `Team` carries `color`, `alt_color`, `logo` and `logo_dark` from the CFBD roster rows (synthetic
  demo teams get deterministic colours), with `Team.to_dict()` / `Team.from_dict()`,
  `contrast_color()` and `normalize_hex_color()` helpers in `cfbratings.data`.
- Cache provenance: `JsonCache.info()` returns a `CacheInfo` (when a payload was fetched, whether it
  is final, and whether the working cache or the committed archive would serve it);
  `CFBDClient.freshness(year)` exposes it per season, and `cfbratings cache list` now prints the age
  of every entry and whether it can expire.
- Chart builders (`cfbratings.viz`): `rating_bar` and `trajectory_chart` take a team-colour map
  (`resolve_team_color()` picks the primary or alternate colour that is legible on the theme's
  surface; look-alike colours on one trajectory chart get dashed lines), `weekly_metric_chart`
  hides weeks with fewer than `min_games` (default 5) and shows the game count in the hover,
  `win_distribution_chart` and `conference_dots` take an `accent` colour, and tooltips are themed
  so they are no longer white boxes on dark. Contrast of every text token on both themes is tested.
- Dashboard: team logos in every table (rankings, schedule, projections, next opponent) and a rank
  change column (▲ / ▼ / – / new versus the previous snapshot week, the preseason prior included);
  the ranked bar chart colours by team (default) or conference; the Team tab has a header strip in
  the team's colours, a "Next game" tile with the kickoff day (US Eastern) and win probability,
  opponent logos, and team-coloured trajectory and win-total charts; Matchup shows both teams as
  coloured chips; the Backtest weekly chart hides weeks with fewer than five games and the by-season
  tables are formatted like the headline one; the sidebar reports where each loaded season came from
  (archived, cached N hours ago, or about to call the API) and the number of API calls made this
  session; the demo "weeks played" slider reaches 0 for a true preseason. `.streamlit/config.toml`
  pins the light theme (charts and page always agree; `STREAMLIT_THEME_BASE=dark` for dark mode,
  with dark logo variants), hides the Deploy toolbar and disables usage stats. Percentages are whole
  numbers everywhere, selectors show labels instead of column names, unplayed games show blank
  cells instead of `None`, and the blank strip under every table is gone. New pure helpers:
  `cfbratings.snapshots.rank_map / previous_snapshot_week / rank_changes` and
  `cfbratings.analytics.age_label / freshness_label / kickoff_label`, all unit-tested.
- A headless QA pass (light, dark, phone width, demo preseason / mid-season / finished season)
  drove eleven follow-up fixes: no `None` text anywhere (finished seasons show "Final" instead of
  next-opponent cells), Matchup chips put the logo on a neutral disc and resolve the fill against
  the page so dark-coloured teams survive dark mode, Home == Away is refused with a warning,
  preseason SOS and momentum are blank instead of ranked zeros, the ranked bar chart's hover and
  title follow the sort column, week axes run from the first to the last week with integer ticks,
  dark-mode team colours are tinted toward legibility instead of falling back to white, the
  rating-vs-SOS scatter labels the rows it is given (so filtered views keep labels), the
  out-of-sample predictions expander is a curated, formatted view, the Elo formulas in
  METHODOLOGY.md no longer clip at phone width, and the synthetic schedule never books the same
  opponent in consecutive weeks.
- ruff's `extend-exclude = ["data"]` matched `cfbratings/data` by basename, so that package was
  never linted; the pattern is now anchored to the repository's `data/` directory.

### Fixes
- **The test-suite made real API calls** whenever a `CFBD_API_KEY` was present: the dashboard smoke
  test picked the live source and fetched the current and previous seasons (four calls per run).
  `tests/conftest.py` now replaces every captured settings object with a key-less copy pointing at
  throw-away cache directories and blocks `requests.Session.request`; a test proves it.
- **BLAS pinned to one thread** (`cfbratings._blas`, the package's first import): OpenBLAS spun up
  its whole thread pool for every 135×135 solve, and two concurrent processes (two backtests, the
  dashboard next to a backtest, the test-suite) starved each other, a 1000× slowdown per solve. A
  four-season Massey walk-forward went from 447 s to 4 s. `make app` exports the same variables for
  the dashboard, whose process imports numpy before cfbratings.

### Real-data groundwork (no API quota spent)
- Schema and client checked against the CollegeFootballData OpenAPI spec (API 5.32.1):
  `/games` and `/lines` accept `seasonType=both`, every field the parsers read is camelCase
  as expected (`homeTeam`, `homePoints`, `neutralSite`, `completed`, `homeClassification`,
  …), `/games` has no default `classification` (the full multi-division payload is returned,
  which the FCS pooling relies on) and `spread` is documented as *home-relative, negative
  favours home*, which is the convention the Vegas comparison assumes.
- `CFBDClient` counts every real API request (`CFBDClient.api_calls`, `client.calls`) and the
  CLI prints `[cfbd] N API call(s) made this run` on stderr, so the free tier's monthly budget
  is visible. HTTP 429 raises `CFBDQuotaError` immediately (never retried, nothing cached);
  CFBD configuration/transport errors exit the CLI with a one-line message instead of a
  traceback.
- `consensus_spreads` cross-checks each `spread` against `formattedSpread` and warns when a
  payload contradicts the home-relative sign convention.
- `tests/fixtures/cfbd_v5_shapes.json` carries every required key of the spec's `Team`,
  `Game` and `BettingGame` schemas; new tests parse it, pool FCS opponents on it and verify
  the spread sign. `tests/fixtures/trim_payload.py` trims an archived season into a real
  fixture offline once one has been fetched.

## 1.2.0 — 2026-10-10

### Fixes (from a full-branch code review)
- `season_regress` passed as a model override (`--param`, dashboard slider, `compare_methods`
  params) now actually controls the prior carry-over; it was silently ignored.
- The dashboard runs in the preseason when a warm-up season is loaded (it previously
  refused with "no completed games") and shows preseason rankings and projections.
- PPoints no longer grades a week-1 opponent on the snapshot that already contains the
  game; warm-started snapshots now include week 0 (the preseason prior) for this.
- The pooled FCS pseudo-team no longer occupies a strength-of-schedule rank slot or a slot
  in the trajectory chart, and is excluded from z-score statistics.
- The Vegas reference row no longer reports a meaningless 0 % against-the-spread accuracy.
- Massey `prior_lambda` defaults to `ridge_lambda` as documented when constructed directly
  (settings still supply 3.0); the docs now say which knob matters on warm-started data.
- The rolling calibrator ignores stale cross-season pairs whose total weight is negligible.
- Dashboard caches key on game content (ids and scores), so a score correction after a
  refresh is picked up, and "Refresh API cache" is a one-shot rather than sticky.

### Features
- **Committed archive of finished seasons.** The cache gained a second tier: final payloads
  are written gzipped to `data/archive/` (tracked in git) and read as a fallback, so a fresh
  clone never re-downloads a season. `cfbratings cache list` shows both tiers;
  `cache clear` leaves the archive alone.
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
