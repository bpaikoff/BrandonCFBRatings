# Methodology

This document describes every formula the engine uses. Notation: $r_i$ is team $i$'s rating,
$m_g$ the home margin of game $g$ (home points minus away points), $h$ the home-field
advantage in points, and $\mathbb{1}_g$ an indicator that is 1 for games played at the home
team's stadium and 0 on neutral sites.

## 1. Rating models

### 1.1 Colley Matrix (wins and losses only)

Colley (2002) starts every team at a Laplace prior of $\tfrac12$ and solves the linear system

$$
(2 + n_i)\, r_i \;-\; \sum_{j \in \text{opp}(i)} r_j \;=\; 1 + \frac{w_i - l_i}{2},
$$

where $n_i$ is games played and $w_i - l_i$ the net wins (ties contribute 0). The
`prior_strength` parameter generalises the constant 2 (two pseudo-games against an average
team). The matrix is symmetric positive-definite, ratings average exactly $\tfrac12$, and
margin of victory is deliberately ignored, which makes Colley immune to running up the score
but blind to *how* a team wins.

### 1.2 Massey (least squares on margins)

Every game contributes one equation $r_{\text{home}} - r_{\text{away}} = m_g - h\,\mathbb{1}_g$.
The system is solved by weighted ridge regression

$$
\left(A^{\top} W A + \lambda I\right) r = A^{\top} W y,
$$

with $A$ the $\pm1$ design matrix, $y$ the HFA-adjusted margins clipped to
$\pm$`max_margin` (default 50) so blow-outs stop paying, and $W$ optional recency weights
$w_g = 2^{-(T - t_g)/\text{half\_life}}$. Because each row of $A$ sums to zero the ridge
solution is automatically centred at a mean of zero; no constraint row is needed. The
default $\lambda = 1$ was chosen by walk-forward backtests (the legacy value of 0.01 badly
overfits early-season data).

### 1.3 Elo with margin of victory

Games are processed chronologically. With $R$ the pre-game ratings and
$H = h \times \text{elo\_per\_point}$ the home advantage in Elo points (default $2.5 \times 25 = 62.5$):

$$
E_{\text{home}} = \frac{1}{1 + 10^{(R_{\text{away}} - R_{\text{home}} - H\,\mathbb{1}_g)/400}}
$$

$$
\text{mult} = \ln(|m_g| + 1)\,\frac{2.2}{0.001\,\Delta_{\text{winner}} + 2.2}, \qquad
\Delta R_{\text{home}} = K \cdot \text{mult} \cdot (S_{\text{home}} - E_{\text{home}}),
$$

where $S_{\text{home}} \in \{1, \tfrac12, 0\}$ and $\Delta_{\text{winner}}$ is the winner's
pre-game Elo (including home advantage) minus the loser's. The multiplier is the
FiveThirtyEight form: it rewards larger margins logarithmically and damps the gain of
heavy favourites, which prevents autocorrelation in rating changes. Ratings regress a
fraction `season_regress` (default 0.33) toward the initial rating **between seasons only**.
(The legacy implementation added 2.1 *Elo* points of HFA, i.e. essentially none, regressed
20 % toward the mean every week, and used the home-minus-away difference in the multiplier
regardless of who won.)

### 1.4 Bradley-Terry (penalised logistic model)

$$
P(\text{home wins}) = \sigma\!\left(r_{\text{home}} - r_{\text{away}} + \eta\,\mathbb{1}_g\right)
$$

is fitted by Newton's method on the penalised log-likelihood with an $L_2$ penalty `ridge`
on the team parameters (ties count half a win each way). $\eta$ is the home advantage in
log-odds and is reported alongside the ratings. Like Colley it uses only outcomes, but it
is a proper probability model and copes with undefeated teams thanks to the penalty.

### 1.5 Hybrid

Component ratings are standardised to z-scores (so points, probabilities and Elo units are
commensurable) and blended with user-supplied weights that are normalised to sum to one.
The default is an equal blend of Colley, Massey and Elo.

## 2. From ratings to spreads and probabilities

Every model is passed through the same calibration so that outputs are comparable:

$$
\hat m = \beta_0\,\mathbb{1}_g + \beta_1\,(r_{\text{home}} - r_{\text{away}}), \qquad
P(\text{home wins}) = \Phi\!\left(\hat m / \sigma\right).
$$

$\beta_0$ is the home-field advantage in points, $\beta_1$ converts rating units to points
and $\Phi$ is the standard normal CDF (football margins are close to normal with
$\sigma \approx 15$–$17$ points).

Fitting the calibrator on the same games the model was trained on is optimistic: a model
with one parameter per team nearly interpolates early in the season. Two safeguards are
therefore built in:

* **Cross-validated calibration** (`MarginCalibrator.fit_cv`, used for live rankings): each
  game is predicted by a model fitted on the other folds, and $\beta, \sigma$ are estimated
  from those held-out pairs.
* **Degrees-of-freedom correction** for the plain in-sample fit:
  $\sigma^2 = (\text{RSS} + n_0 \sigma_0^2) / (\max(n - p, 1) + n_0)$ with $p$ the number
  of teams and a prior of $\sigma_0 = 16$ backed by $n_0 = 20$ pseudo-games.

Both fits shrink $\beta$ toward a model-specific prior (slope 1 for Massey, $1/25$ for Elo)
with the same pseudo-observation weight, which keeps the first weeks of a season stable.

## 3. Analytics

* **Strength of schedule** – mean rating of opponents played (same units as the rating).
* **Momentum** – mean of $(\text{actual margin} - \hat m)$ over the last three games, in points,
  so it is comparable across methods.
* **Conference strength index** – per conference, $0.65 \times$ median of the top 35 % of
  teams plus $0.35 \times$ median of the 40–60 % band, rescaled across conferences to the
  band $[0.6, 1.6]$. Medians make the index robust to a single dominant program.

## 4. PPoints

PPoints is a résumé metric, not a predictor. For every game the opponent's rank is blended
between its rank **entering that week** (from weekly snapshots) and its **current** rank,
with weight `weight_current` (default 0.5). The rank maps to tier points

| Opponent rank | 1–10 | 11–25 | 26–40 | 41–60 | 61+ |
|---|---|---|---|---|---|
| Tier points | 8 | 5 | 3 | 2 | 1 |

A win earns `tier × site × conference`, where `site` is 1.2 for road and neutral-site games,
and `conference` is the opponent's conference strength index. A loss costs
`loss_penalty / tier` (default 4 ÷ tier), so losing to a top-10 team costs half a point while
losing to a bottom-feeder costs four. Non-conference games use a 1.5× win factor and a softer
loss penalty of 2 to reward ambitious scheduling.

## 5. Backtesting

The walk-forward engine evaluates genuinely out-of-sample predictions:

1. For each week $w \ge$ `start_week`, fit the model on all completed games with
   week $< w$ (optionally preceded by warm-up games from earlier seasons).
2. Build the calibrator from a cross-validated fit on that training window, then refit it
   on the held-out predictions already made for weeks $< w$ (recent weeks weighted by
   $0.85^{\Delta w}$) once at least 30 such pairs exist.
3. Predict every game in week $w$ and record margin, win probability and outcome.

Pooled predictions are scored with

| Metric | Definition | Better |
|---|---|---|
| Accuracy | share of straight-up picks that were right | higher |
| Brier | $\text{mean}\,(p - y)^2$ | lower |
| Log loss | $-\text{mean}\,[y \ln p + (1-y)\ln(1-p)]$ | lower |
| MAE / RMSE | error of the predicted margin in points | lower |
| ATS accuracy | share of games where the model picked the correct side of the closing spread (needs `--lines`) | higher |

Two reference rows accompany every comparison: **home_prior** (pick the home team with the
training-set home win rate and mean margin) and, when lines are available, the **closing
Vegas spread**. A rating system that cannot beat the home-team baseline is not informative;
one that approaches the market is excellent.

`grid_search` wraps the engine to tune hyper-parameters (Elo $K$, Massey $\lambda$, HFA,
hybrid weights, …) against any of the metrics above.
