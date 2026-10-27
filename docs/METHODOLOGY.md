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
\text{mult} = \ln(|m_g| + 1)\,\frac{2.2}{0.001\,\Delta_{\text{winner}} + 2.2},
$$

$$
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
The default blend is 0.6 Massey, 0.2 Colley and 0.2 Elo. It was an equal blend until the
real-data backtests ([`BACKTEST_RESULTS.md`](BACKTEST_RESULTS.md)): on 2021-2024 the log loss
falls monotonically as Massey's share rises, and the 2025 hold-out confirms it (0.482 for the
equal blend, 0.475 at 0.6 Massey, 0.473 at 0.8), so the blend now leans on the margin-based
view while keeping a real share of the win-loss and sequential views.

### 1.6 Preseason priors (warm starts)

A season is only 12-13 games, and a team's strength is strongly correlated with the
previous year's, so a cold start throws away real information. Every model therefore
accepts a **prior** $r^0$ built from the previous season:

$$
r^0_i = c + (1 - \rho)\,(r^{\text{prev}}_i - c), \qquad \rho = \texttt{season\_regress} = 0.33,
$$

where $c$ is the model's neutral rating (0.5 for Colley, 1500 for Elo, 0 otherwise). The
prior enters each model in its natural place:

| Model | Prior mechanism |
|---|---|
| Colley | the `prior_strength` pseudo-games are played at $r^0_i$ instead of 0.5: $b_i = \text{prior\_strength}\cdot r^0_i + (w_i - l_i)/2$ |
| Massey | ridge toward the prior, $\lambda_p \lVert r - r^0 \rVert^2$ with $\lambda_p = 3$ (`massey_prior_lambda`; a prior carries real information and gets more weight than shrinkage toward zero). With a prior present `ridge_lambda` plays no role, so tune $\lambda_p$ on warm-started data |
| Elo | ratings simply start at $r^0$ |
| Bradley-Terry | penalty $\tfrac{\text{ridge}}{2}\lVert r - r^0\rVert^2$ |
| Hybrid | one prior per component |

Priors chain across seasons (each season's prior is the regressed fit of a season that was
itself warm-started), and the previous season's calibrator is carried across the boundary
so that week-1 spreads are on a sensible scale even for models whose ratings have no
natural point units. For the synthetic benchmark ($\lambda_p=3$, $\rho=0.33$) the warm start
lowers log loss on identical weeks for every model and lets evaluation begin in week 1. The
real 2025 season behaves the same way: seeded from 2024 and graded from week 4, every model
gains 0.015 to 0.027 log loss and 0.3 to 0.5 points of MAE (see
[`BACKTEST_RESULTS.md`](BACKTEST_RESULTS.md)).

### 1.7 Games against FCS opponents

Dropping games against non-FBS opponents discards roughly one game per team, including the
most informative upsets. By default those opponents are pooled into a single pseudo-team,
``FCS (pooled)``, which is rated like any other team but never ranked and never counted
as a conference. Wins and losses against it count toward records, strength of schedule
and PPoints (a loss to the pool costs the full bottom-tier penalty). Set ``CFB_FCS=drop`` or
``--fcs drop`` to restore the old behaviour.

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
between its rank **entering that week** (from weekly snapshots; week 1 uses the preseason
prior when warm-started, otherwise the current rank) and its **current** rank,
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

The walk-forward engine evaluates genuinely out-of-sample predictions. Games may span
several seasons; seasons are processed in order and every season after the first is
warm-started from the one before it (section 1.6).

1. For each week $w \ge$ `start_week`, fit the model on the season's completed games with
   week $< w$, seeded with the season's prior. (Without a prior the first evaluation waits
   for `min_train_games`; with one it can start in week 1.)
2. Build the calibrator from a cross-validated fit on that training window (anchored on
   the previous season's calibrator when there is one), then refit it on the held-out
   predictions already made for earlier weeks (recent weeks weighted by $0.85^{\Delta w}$)
   once at least 30 such pairs exist.
3. Predict every game in week $w$ and record margin, win probability and outcome.
4. At the end of the season, regress the final ratings into the next season's prior.

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

## 6. Rest-of-season projections

Every unplayed game on a team's schedule gets a win probability $p_g$ from the calibrated
ratings (section 2). With $n$ games left, the number of additional wins $W$ is a sum of
independent Bernoulli trials -- a Poisson-binomial variable -- whose distribution is
computed exactly by the recursion

$$
P_k(w) = P_{k-1}(w)\,(1 - p_k) + P_{k-1}(w - 1)\,p_k, \qquad P_0(0) = 1,
$$

so quantities such as "probability of reaching 10 wins" or "probability of winning out"
are exact rather than Monte Carlo estimates. The projected record is
$(\text{wins} + \sum_g p_g,\ \text{losses} + n - \sum_g p_g)$. Games are treated as
independent given the ratings; correlation through shared uncertainty in the ratings
themselves is ignored, which makes the tails slightly too narrow late in the season.

## 7. Conferences, independents and the Playoff

**Conference strength** (section 3) is computed over real conferences only. An independent
(CFBD's *FBS Independents*: Notre Dame, UConn) is not a conference: it plays no conference
schedule, has no standings, no champion and no automatic playoff bid. In a résumé score an
independent opponent is placed on the conference strength scale by its *own* rating, scaled
exactly as a conference score would be and clipped to the 0.6–1.6 band, so beating Notre Dame
counts like beating a top-conference opponent and beating UConn like beating a weak one.

**Standings** use conference games only, taken from CFBD's `conferenceGame` flag (Army–Navy,
flagged but not counted by the American, is excluded). Teams are ordered by conference record,
then head-to-head among the tied, then overall record, then rating; a conference's own full
tie-breaker policy is not reproduced. A title game is the regular-season game CFBD notes as
"*Conference* Championship"; it is not a conference game, so the standings are settled before
it and its winner is the champion. Until it is played the top two of the standings meet in a
projected title game, at a neutral site for the power conferences and at the leader's home
otherwise, and the favourite is the projected champion.

**The committee proxy** ranks teams by

$$
S_i = w_r\, z(\text{rating}_i) + w_p\, z(\text{PPoints}_i) - w_\ell\, \text{losses}_i + w_c\,[i \text{ is a champion}],
$$

with $z$ the standard score over FBS teams. The default weights
$(w_r, w_p, w_\ell, w_c) = (1, 0.5, 0.35, 0.15)$ are a heuristic reading of how the committee has
ranked since 2014 (losses first, then résumé, then a modest bump for a title); they are
parameters exposed in the CLI and the dashboard, not fitted values.

**The field and the bracket** follow the season's rules. For 2026: twelve teams; the champions
of the ACC, Big 12, Big Ten and SEC qualify; the highest-ranked team of the other conferences
qualifies, champion or not; seven at-large places go by ranking; seeding is straight, so the
four highest-ranked teams receive the byes whatever their conference status. Seeds 5–12 meet
(5–12, 8–9, 6–11, 7–10) at the higher seed's home, with the home-field advantage of section 2;
the quarterfinals (1 vs 8/9, 4 vs 5/12, 3 vs 6/11, 2 vs 7/10), semifinals and final are neutral.
The 2024 (five best champions, byes to the four best champions) and 2025 (five best champions,
straight seeding) formats are kept for the archived seasons. A demo season maps the rules onto
its own strongest conferences.

**Odds.** For a fixed bracket the probability that a team reaches each round is computed
exactly: $P(i \text{ wins a game}) = P(i \text{ reaches it}) \sum_j P(j \text{ reaches it})\,p_{ij}$ over
every possible opponent $j$, with $p_{ij}$ the calibrated win probability for that site. The
rest-of-season projection is a Monte Carlo: each simulation samples every unplayed game from
its win probability (ratings held fixed, PPoints kept at its current value), settles the
standings and the title games, ranks with the committee proxy on the simulated records, selects
and seeds the field, and then takes the exact bracket odds for that field, so the title
probabilities average exact numbers rather than coin flips.

**Everything the simulation records.** Each simulated season also yields every team's final
record (so the projected record is a mean over simulations that includes a conference title
game when the team reaches one, and the *most likely* record is the modal wins–losses pair), the
probability of finishing undefeated, of reaching and of winning the conference title game, the
title-game pairings themselves (how often each pair met, at which site, with the favourite's
chance in that game), and the distribution of seeds. A neutral-site projected title game is the
same pairing whichever side led the standings; a hosted one keeps the leader as the host.

**The bracket on its own.** For a fixed field — the bracket if the season ended today, the
simulation's modal field, or a hand-picked one of up to twelve teams — the exact odds above are
complemented by three ways of playing it out: the *chalk* bracket, in which the favourite wins
every game and each game carries its projected score; a Monte Carlo of the bracket alone, in
which every game is a draw from its win probability (its frequencies converge to the exact odds
and are shown beside them); and a single play-through, in which each game's margin is drawn
from a normal distribution around the model's spread with the calibrated error $\sigma$ (section 2)
and its total around the projected total (section 8), so the sampled scores decide the winners.

## 8. Projected scores

A rating model says *by how much* a team should win; a score needs one more number, the game's
total. Every completed game of the season in hand contributes two observations,

$$
\text{pts}_{\text{home}} = \mu + O_{\text{home}} + D_{\text{away}} + h, \qquad
\text{pts}_{\text{away}} = \mu + O_{\text{away}} + D_{\text{home}} - h,
$$

where $\mu$ is the league's mean points per team per game, $O_i$ is team $i$'s offence (points
scored above the mean, adjusted for the defences it faced), $D_i$ its defence (points *allowed*
above the mean, so positive is leaky) and $h$ half the home-field advantage of section 2. $O$
and $D$ are fitted by ridge least squares with a penalty of 4 points²·games on each term, which
shrinks a team with few games toward average and leaves one with many alone; a season with no
completed games gets $\mu = 27.5$ and every team average. The projected total of a matchup is

$$
T = 2\mu + O_{\text{home}} + D_{\text{away}} + O_{\text{away}} + D_{\text{home}},
$$

and the projected score splits it around the calibrated margin $m$ of section 2:
$\text{home} = (T + m)/2$, $\text{away} = (T - m)/2$. The margin is the better-calibrated number
(it is what the ratings are fitted to), so it is always taken from the ratings and only the
total from this model; the total is never allowed below $|m| + 6$, so a projected loser scores
at least 3. Scores are shown rounded, nudged by a point when rounding would produce a tie. For a
sampled score (a single simulated playoff) the margin is drawn from $N(m, \sigma^2)$ with the
calibrator's $\sigma$ and the total from $N(T, \sigma_T^2)$ with $\sigma_T$ the residual standard
deviation of the fitted totals (17 points until twenty games have been played). No further API
call is involved: the model uses only the games already loaded for the ratings.
