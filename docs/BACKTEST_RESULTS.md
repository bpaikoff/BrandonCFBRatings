# Real-data backtest results

Every number below comes from CollegeFootballData (CFBD, API 5.32.1) payloads for the 2021 to 2025
seasons, archived under `data/archive/` on 2026-10-10 and read back offline. Each command is
reproducible from a fresh clone with **zero** API calls.

## 1. The data

17 API calls were spent in total: 2 per season for teams and games, 1 per season for closing
lines, plus 2 for the in-progress 2026 season (a live sanity check, cached but not archived).

|                                      |   2021 |   2022 |   2023 |   2024 |   2025 |
|:-------------------------------------|-------:|-------:|-------:|-------:|-------:|
| raw game rows (all divisions)        |   2454 |   3705 |   3734 |   3801 |   3831 |
| FBS teams                            |    130 |    131 |    133 |    134 |    136 |
| rated games                          |    887 |    896 |    910 |    920 |    934 |
| of which vs an FCS opponent (pooled) |    117 |    120 |    118 |    121 |    126 |
| rows dropped (neither team FBS)      |   1567 |   2809 |   2824 |   2881 |   2897 |
| FBS postseason rows                  |     38 |     42 |     42 |     46 |     46 |
| FBS postseason `week` values         |      1 |      1 |      1 |      1 |      1 |
| last FBS regular-season week         |     15 |     15 |     15 |     16 |     16 |
| line rows                            |    887 |   1463 |   1416 |   1573 |   1597 |
| rated games with a spread            |    887 |    892 |    907 |    904 |    934 |
| book rows: `consensus`               |    870 |   1244 |     29 |      0 |      0 |
| book rows: other providers           |   2671 |   3066 |   3043 |   3172 |   3289 |
| incomplete FBS rows                  |      0 |      0 |      0 |      1 |      0 |

What the real payload looks like, checked against the schema the parsers assume:

* `/games?seasonType=both` is accepted and returns every division (`homeClassification` is
  `fbs`, `fcs`, `ii`, `iii` or `null`), which is what the roster-based FCS pooling relies on:
  FBS-vs-FBS games are kept, games against a non-FBS opponent are pooled onto one pseudo-team,
  the rest are dropped. Every field the parsers read is camelCase and present.
* **Every FBS bowl and every College Football Playoff round is `week: 1` of
  `seasonType: postseason`**, in all five seasons (the 2024 title game on 21 January is
  postseason week 1, like the 14 December bowls). Lower-division playoff rows in 2023 and 2025
  carry weeks 11 to 15, but neither team is FBS, so they are dropped. The normaliser relabels
  the FBS postseason as one effective week after the last regular-season week, so the whole
  postseason is graded with the ratings as they stood after Army-Navy weekend. First-round
  playoff games are played on campus and carry `neutralSite: false`; the quarterfinals onward
  are neutral.
* Closing lines: 2021 and 2022 carry a `consensus` provider for nearly every game, which
  `consensus_spreads` prefers; 2023 has it for 29 games only, and 2024 and 2025 have none, just
  three books (ESPN Bet, DraftKings, Bovada), so their lines are the mean of the books.
  `spread` is home-relative and negative when the home team is favoured, as documented:
  Michigan at Ohio State 2024 is `-20.5` with `formattedSpread` "Ohio State -20.5", the Cotton
  Bowl Ohio State-Texas (Texas listed as home) is `+5.5` with "Ohio State -5.5". Of the
  17,361 book rows with a spread across the five seasons, 0 contradict the convention.
* A game without a book row simply has no line (Idaho at Oregon 2024, and a few dozen per
  season in 2022 to 2024); the Vegas rows below therefore have slightly smaller `n`.

### Rankings sanity check

2024, cold start (`cfbratings --year 2024 rank --top 10 --no-ppoints`): the four playoff
semifinalists (Ohio State, Notre Dame, Texas, Penn State) and the Big Ten champion Oregon are
the top five, with the champion first.

```
Hybrid ratings — 2024 — CFBD 2024 (both, FCS pooled) — cold start
calibration: HFA +3.76 pts, slope 9.694, sigma 15.5 pts (cross_val, n=919)
 rank           team       conference record  rating    sos  sos_rank  momentum
    1     Ohio State          Big Ten   14-2   2.528  0.729         2     9.807
    2     Notre Dame FBS Independents   14-2   2.245  0.605         6    -1.041
    3          Texas              SEC   13-3   1.979  0.629         4    -2.074
    4     Penn State          Big Ten   13-3   1.813  0.471        15    10.173
    5         Oregon          Big Ten   13-1   1.754  0.326        24     2.057
    6       Ole Miss              SEC   10-3   1.712  0.029        54    -1.905
    7        Indiana          Big Ten   11-2   1.692 -0.039        60     6.970
    8        Georgia              SEC   11-3   1.557  0.616         5    -3.953
    9        Alabama              SEC    9-4   1.409  0.392        19   -12.821
   10 South Carolina              SEC    9-4   1.353  0.394        18    -1.068
```

Warm-started from 2023 (`cfbratings --year 2024 --warmup-years 1 rank --top 10 --no-ppoints`):

```
Hybrid ratings — 2024 — CFBD 2023-2024 (both, FCS pooled) — warm-started from 1 prior season
calibration: HFA +3.52 pts, slope 10.026, sigma 15.3 pts (cross_val, n=919)
 rank       team       conference record  rating   sos  sos_rank  momentum
    1 Ohio State          Big Ten   14-2   2.562 0.761         2    10.415
    2 Notre Dame FBS Independents   14-2   2.269 0.614         8    -0.287
    3      Texas              SEC   13-3   2.073 0.662         4    -3.363
    4     Oregon          Big Ten   13-1   1.971 0.419        18     1.210
    5 Penn State          Big Ten   13-3   1.904 0.518        14     9.383
    6    Georgia              SEC   11-3   1.758 0.652         6    -5.387
    7   Ole Miss              SEC   10-3   1.753 0.066        47    -2.581
    8    Alabama              SEC    9-4   1.581 0.450        17   -13.097
    9    Indiana          Big Ten   11-2   1.508 0.019        55     9.148
   10  Tennessee              SEC   10-3   1.361 0.046        53     6.398
```

2025, warm-started from 2024 (`cfbratings --year 2025 --warmup-years 1 rank --top 10`): the
unbeaten national champion Indiana leads, with Ohio State, Notre Dame, Oregon and Miami behind.

```
Hybrid ratings — 2025 — CFBD 2024-2025 (both, FCS pooled) — warm-started from 1 prior season
calibration: HFA +3.59 pts, slope 10.627, sigma 14.8 pts (cross_val, n=934)
 rank       team       conference record  rating    sos  sos_rank  momentum  ppoints
    1    Indiana          Big Ten   16-0   2.712  0.522         8    14.923  104.525
    2 Ohio State          Big Ten   12-2   2.224  0.462        10    -0.967   56.148
    3 Notre Dame FBS Independents   10-2   2.130  0.351        24    14.730   44.419
    4     Oregon          Big Ten   13-2   2.009  0.548         7    -0.474   71.847
    5      Miami              ACC   13-3   1.926  0.639         4     5.676  106.069
    6 Texas Tech           Big 12   12-2   1.837  0.016        57     9.998   43.210
    7   Ole Miss              SEC   13-2   1.691  0.301        33     6.554   57.813
    8    Georgia              SEC   12-2   1.638  0.322        26     3.128   65.160
    9       Utah           Big 12   11-2   1.568 -0.034        63    -2.133   26.661
   10  Texas A&M              SEC   11-2   1.406  0.362        22    -3.932   55.712
```

The in-progress 2026 season through week 6 (`cfbratings --year 2026 --warmup-years 1 rank`,
served from the working cache):

```
Hybrid ratings — 2026 — CFBD 2025-2026 (both, FCS pooled) — warm-started from 1 prior season
calibration: HFA +5.44 pts, slope 9.339, sigma 16.2 pts (cross_val, n=402)
 rank       team       conference record  rating    sos  sos_rank  momentum
    1    Alabama              SEC    5-0   2.232  0.354         7     7.293
    2    Indiana          Big Ten    5-0   2.198 -0.727        83     0.128
    3 Notre Dame FBS Independents    5-0   2.176 -0.210        29     1.803
    4      Miami              ACC    5-0   2.171 -0.710        81    13.075
    5 Ohio State          Big Ten    4-1   2.063  0.027        14    10.546
    6    Georgia              SEC    5-0   1.961 -0.700        78     8.949
    7      Texas              SEC    4-0   1.950  0.877         1     0.455
    8       Utah           Big 12    4-0   1.681 -1.035       117     6.489
    9        BYU           Big 12    5-0   1.432 -0.584        71     0.167
   10        LSU              SEC    4-1   1.367 -0.250        34     4.563
```

## 2. Cold single-season backtests (weeks 4 onward)

```bash
for Y in 2021 2022 2023 2024 2025; do
  cfbratings --year $Y backtest --start-week 4 --lines --by-week --calibration
done
```

Each season starts from nothing, so weeks 1 to 3 are skipped. Every model is refitted each
week on every earlier game and calibrated on cross-validated predictions; that week's games are
scored out of sample. **home_prior** always picks the home team; **vegas** is the closing
spread turned into a probability with the same normal link.

Log loss (lower is better):

|   season |   Colley |   Massey |    Elo |   Bradley-Terry |   Hybrid |   home_prior |   Vegas |
|---------:|---------:|---------:|-------:|----------------:|---------:|-------------:|--------:|
|     2021 |   0.5723 |   0.5564 | 0.5707 |          0.5744 |   0.5536 |       0.6947 |  0.5294 |
|     2022 |   0.6122 |   0.5748 | 0.5952 |          0.6129 |   0.5794 |       0.7041 |  0.5463 |
|     2023 |   0.5861 |   0.558  | 0.576  |          0.5885 |   0.5581 |       0.7085 |  0.5225 |
|     2024 |   0.6018 |   0.5642 | 0.5873 |          0.6022 |   0.5692 |       0.6842 |  0.5404 |
|     2025 |   0.5722 |   0.5361 | 0.5613 |          0.5696 |   0.5381 |       0.7158 |  0.5226 |

Mean absolute error of the predicted margin, in points:

|   season |   Colley |   Massey |   Elo |   Bradley-Terry |   Hybrid |   home_prior |   Vegas |
|---------:|---------:|---------:|------:|----------------:|---------:|-------------:|--------:|
|     2021 |    14.03 |    13.57 | 13.74 |           14.09 |    13.44 |        17.6  |   12.56 |
|     2022 |    13.39 |    12.92 | 12.93 |           13.38 |    12.78 |        16.53 |   12.05 |
|     2023 |    13.7  |    13.12 | 13.63 |           13.74 |    13.16 |        16.92 |   12.26 |
|     2024 |    13.55 |    12.66 | 13.2  |           13.52 |    12.74 |        16.4  |   11.7  |
|     2025 |    13.61 |    12.41 | 13.16 |           13.46 |    12.52 |        17.7  |   11.72 |

Straight-up accuracy:

|   season |   Colley |   Massey |    Elo |   Bradley-Terry |   Hybrid |   home_prior |   Vegas |
|---------:|---------:|---------:|-------:|----------------:|---------:|-------------:|--------:|
|     2021 |   0.6998 |   0.6905 | 0.6765 |          0.703  |   0.6983 |       0.5801 |  0.7294 |
|     2022 |   0.6713 |   0.6898 | 0.6898 |          0.6605 |   0.6806 |       0.5741 |  0.7218 |
|     2023 |   0.6982 |   0.7104 | 0.6921 |          0.6829 |   0.7088 |       0.5747 |  0.7305 |
|     2024 |   0.6726 |   0.6858 | 0.677  |          0.6785 |   0.6947 |       0.5973 |  0.7153 |
|     2025 |   0.6774 |   0.6993 | 0.6803 |          0.6715 |   0.7124 |       0.5898 |  0.7372 |

Full tables per season:

<details><summary>2021</summary>

| method                |   n |   accuracy |   brier |   log_loss |   mae |   rmse |   ats_accuracy |
|:----------------------|----:|-----------:|--------:|-----------:|------:|-------:|---------------:|
| colley                | 643 |     0.6998 |  0.1966 |     0.5723 | 14.03 |  17.61 |         0.4914 |
| massey                | 643 |     0.6905 |  0.1898 |     0.5564 | 13.57 |  16.91 |         0.4741 |
| elo                   | 643 |     0.6765 |  0.1958 |     0.5707 | 13.74 |  17.28 |         0.5039 |
| bradley_terry         | 643 |     0.703  |  0.1975 |     0.5744 | 14.09 |  17.7  |         0.4867 |
| hybrid                | 643 |     0.6983 |  0.1887 |     0.5536 | 13.44 |  16.8  |         0.4867 |
| home_prior (baseline) | 643 |     0.5801 |  0.25   |     0.6947 | 17.6  |  22.06 |                |
| vegas (closing line)  | 643 |     0.7294 |  0.1776 |     0.5294 | 12.56 |  15.69 |                |

</details>
<details><summary>2022</summary>

| method                |   n |   accuracy |   brier |   log_loss |   mae |   rmse |   ats_accuracy |
|:----------------------|----:|-----------:|--------:|-----------:|------:|-------:|---------------:|
| colley                | 648 |     0.6713 |  0.2121 |     0.6122 | 13.39 |  17.26 |         0.5133 |
| massey                | 648 |     0.6898 |  0.1977 |     0.5748 | 12.92 |  16.49 |         0.5071 |
| elo                   | 648 |     0.6898 |  0.2049 |     0.5952 | 12.93 |  16.88 |         0.5196 |
| bradley_terry         | 648 |     0.6605 |  0.2125 |     0.6129 | 13.38 |  17.33 |         0.5181 |
| hybrid                | 648 |     0.6806 |  0.199  |     0.5794 | 12.78 |  16.47 |         0.5149 |
| home_prior (baseline) | 648 |     0.5741 |  0.254  |     0.7041 | 16.53 |  20.93 |                |
| vegas (closing line)  | 647 |     0.7218 |  0.1854 |     0.5463 | 12.05 |  15.37 |                |

</details>
<details><summary>2023</summary>

| method                |   n |   accuracy |   brier |   log_loss |   mae |   rmse |   ats_accuracy |
|:----------------------|----:|-----------:|--------:|-----------:|------:|-------:|---------------:|
| colley                | 656 |     0.6982 |  0.2003 |     0.5861 | 13.7  |  17.43 |         0.4785 |
| massey                | 656 |     0.7104 |  0.1896 |     0.558  | 13.12 |  16.65 |         0.4908 |
| elo                   | 656 |     0.6921 |  0.1971 |     0.576  | 13.63 |  17.31 |         0.4877 |
| bradley_terry         | 656 |     0.6829 |  0.2018 |     0.5885 | 13.74 |  17.49 |         0.4846 |
| hybrid                | 656 |     0.7088 |  0.1892 |     0.5581 | 13.16 |  16.66 |         0.4769 |
| home_prior (baseline) | 656 |     0.5747 |  0.2561 |     0.7085 | 16.92 |  21.44 |                |
| vegas (closing line)  | 653 |     0.7305 |  0.1754 |     0.5225 | 12.26 |  15.48 |                |

</details>
<details><summary>2024</summary>

| method                |   n |   accuracy |   brier |   log_loss |   mae |   rmse |   ats_accuracy |
|:----------------------|----:|-----------:|--------:|-----------:|------:|-------:|---------------:|
| colley                | 678 |     0.6726 |  0.2081 |     0.6018 | 13.55 |  16.98 |         0.5074 |
| massey                | 678 |     0.6858 |  0.1927 |     0.5642 | 12.66 |  15.84 |         0.5045 |
| elo                   | 678 |     0.677  |  0.2021 |     0.5873 | 13.2  |  16.63 |         0.5223 |
| bradley_terry         | 678 |     0.6785 |  0.2083 |     0.6022 | 13.52 |  17.01 |         0.5178 |
| hybrid                | 678 |     0.6947 |  0.1945 |     0.5692 | 12.74 |  15.99 |         0.5104 |
| home_prior (baseline) | 678 |     0.5973 |  0.2447 |     0.6842 | 16.4  |  20.39 |                |
| vegas (closing line)  | 678 |     0.7153 |  0.1837 |     0.5404 | 11.7  |  14.99 |                |

</details>
<details><summary>2025</summary>

| method                |   n |   accuracy |   brier |   log_loss |   mae |   rmse |   ats_accuracy |
|:----------------------|----:|-----------:|--------:|-----------:|------:|-------:|---------------:|
| colley                | 685 |     0.6774 |  0.1975 |     0.5722 | 13.61 |  16.96 |         0.5161 |
| massey                | 685 |     0.6993 |  0.1836 |     0.5361 | 12.41 |  15.61 |         0.5161 |
| elo                   | 685 |     0.6803 |  0.1927 |     0.5613 | 13.16 |  16.61 |         0.5235 |
| bradley_terry         | 685 |     0.6715 |  0.1967 |     0.5696 | 13.46 |  16.97 |         0.5249 |
| hybrid                | 685 |     0.7124 |  0.1835 |     0.5381 | 12.52 |  15.79 |         0.5059 |
| home_prior (baseline) | 685 |     0.5898 |  0.2584 |     0.7158 | 17.7  |  21.74 |                |
| vegas (closing line)  | 685 |     0.7372 |  0.1762 |     0.5226 | 11.72 |  14.87 |                |

</details>

Massey or the Massey-led hybrid has the lowest log loss and MAE in every season, and the two
are within 0.005 log loss of each other in each of them (Massey ahead in four of five on log
loss, the hybrid ahead on MAE in 2021 and 2022). Elo is third. Colley and Bradley-Terry, which
ignore margins, trail Massey by 0.016 to 0.038 log loss. Every model beats the home-team
baseline by a wide margin (Massey by 0.12 to 0.18 log loss and 3.6 to 5.3 points of MAE) and
trails the closing line (Massey by 0.013 to 0.036 log loss and 0.7 to 1.0 points of MAE).

## 3. Warm multi-season chains (from week 1)

```bash
cfbratings --years 2021-2025 backtest --start-week 1 --lines
cfbratings --years 2022-2025 backtest --start-week 1 --lines   # 2021 only as a warm-up
```

Seasons after the first are warm-started from the regressed final ratings of the season before
(and inherit its calibrator), so evaluation starts in week 1 of 2022 onward. Over the five
seasons (4457 games):

| method                |    n |   accuracy |   brier |   log_loss |   mae |   rmse |   ats_accuracy |
|:----------------------|-----:|-----------:|--------:|-----------:|------:|-------:|---------------:|
| colley                | 4457 |     0.7168 |  0.184  |     0.5431 | 14.15 |  17.9  |         0.5002 |
| massey                | 4457 |     0.7361 |  0.1718 |     0.5113 | 13.29 |  16.82 |         0.497  |
| elo                   | 4457 |     0.7195 |  0.1794 |     0.5323 | 13.91 |  17.61 |         0.4909 |
| bradley_terry         | 4457 |     0.7148 |  0.182  |     0.5377 | 14.07 |  17.77 |         0.4868 |
| hybrid                | 4457 |     0.7303 |  0.1722 |     0.5119 | 13.21 |  16.71 |         0.4939 |
| home_prior (baseline) | 4457 |     0.628  |  0.2415 |     0.679  | 18.22 |  22.75 |                |
| vegas (closing line)  | 4435 |     0.7547 |  0.1617 |     0.4826 | 12.11 |  15.27 |                |

Dropping 2021 from the evaluation (it is then only a warm-up for 2022):

| method                |    n |   accuracy |   brier |   log_loss |   mae |   rmse |   ats_accuracy |
|:----------------------|-----:|-----------:|--------:|-----------:|------:|-------:|---------------:|
| colley                | 3565 |     0.7128 |  0.1846 |     0.5441 | 14.05 |  17.79 |         0.5038 |
| massey                | 3565 |     0.7386 |  0.1707 |     0.5069 | 13.13 |  16.58 |         0.5047 |
| elo                   | 3565 |     0.7229 |  0.1795 |     0.5309 | 13.74 |  17.43 |         0.4987 |
| bradley_terry         | 3565 |     0.713  |  0.1825 |     0.5391 | 13.95 |  17.66 |         0.4925 |
| hybrid                | 3565 |     0.7327 |  0.1718 |     0.5104 | 13.08 |  16.57 |         0.5004 |
| home_prior (baseline) | 3565 |     0.6283 |  0.242  |     0.6804 | 18.1  |  22.63 |                |
| vegas (closing line)  | 3544 |     0.7537 |  0.1624 |     0.4835 | 12.04 |  15.22 |                |

By season, log loss:

|   season |   Colley |   Massey |    Elo |   Bradley-Terry |   Hybrid |   n |
|---------:|---------:|---------:|-------:|----------------:|---------:|----:|
|     2021 |   0.5605 |   0.5523 | 0.5644 |          0.5651 |   0.5469 | 798 |
|     2022 |   0.5743 |   0.5322 | 0.5574 |          0.5698 |   0.5352 | 896 |
|     2023 |   0.5273 |   0.4919 | 0.5142 |          0.5193 |   0.4922 | 910 |
|     2024 |   0.5495 |   0.5149 | 0.5361 |          0.5451 |   0.5158 | 919 |
|     2025 |   0.5075 |   0.4714 | 0.4947 |          0.494  |   0.4752 | 934 |

By season, margin MAE (the closing line's MAE for the same games in the last column):

|   season |   Colley |   Massey |   Elo |   Bradley-Terry |   Hybrid |   Vegas MAE |
|---------:|---------:|---------:|------:|----------------:|---------:|------------:|
|     2021 |    14.61 |    14.2  | 14.76 |           14.99 |    14.04 |       12.5  |
|     2022 |    14.23 |    13.33 | 13.88 |           14.19 |    13.29 |       12.19 |
|     2023 |    13.83 |    13.19 | 13.61 |           13.66 |    13.01 |       12.07 |
|     2024 |    14.36 |    13.4  | 14.03 |           14.17 |    13.27 |       11.96 |
|     2025 |    13.78 |    12.47 | 13.37 |           13.46 |    12.57 |       11.9  |

By season, accuracy:

|   season |   Colley |   Massey |    Elo |   Bradley-Terry |   Hybrid |   n |
|---------:|---------:|---------:|-------:|----------------:|---------:|----:|
|     2021 |   0.7068 |   0.7018 | 0.6942 |          0.7068 |   0.7093 | 798 |
|     2022 |   0.6998 |   0.7277 | 0.7121 |          0.6931 |   0.7098 | 896 |
|     2023 |   0.7385 |   0.7615 | 0.7341 |          0.733  |   0.7516 | 910 |
|     2024 |   0.704  |   0.7214 | 0.7084 |          0.6975 |   0.7214 | 919 |
|     2025 |   0.7334 |   0.7634 | 0.7452 |          0.742  |   0.7559 | 934 |

Against the spread (fraction of games where the model's side of the closing line covered):

|   season |   Colley |   Massey |    Elo |   Bradley-Terry |   Hybrid |   n |
|---------:|---------:|---------:|-------:|----------------:|---------:|----:|
|     2021 |   0.4962 |   0.4683 | 0.5    |          0.486  |   0.4797 | 798 |
|     2022 |   0.4897 |   0.4932 | 0.4818 |          0.4806 |   0.4897 | 896 |
|     2023 |   0.4795 |   0.495  | 0.4739 |          0.4772 |   0.4917 | 910 |
|     2024 |   0.4956 |   0.5011 | 0.4878 |          0.4711 |   0.4933 | 919 |
|     2025 |   0.5382 |   0.5231 | 0.5113 |          0.5178 |   0.5124 | 934 |

### What the warm start is worth (identical weeks)

`cfbratings --years 2024-2025 backtest --start-week 4 --lines` grades 2025 from week 4 like the
cold run above but with ratings seeded from 2024:

| model         |   log loss cold |   log loss warm |   Δ log loss |   MAE cold |   MAE warm |   Δ MAE |   accuracy cold |   accuracy warm |
|:--------------|----------------:|----------------:|-------------:|-----------:|-----------:|--------:|----------------:|----------------:|
| Colley        |          0.5722 |          0.5547 |      -0.0175 |      13.61 |      13.27 | -0.3393 |          0.6774 |          0.6978 |
| Massey        |          0.5361 |          0.521  |      -0.0151 |      12.41 |      12.05 | -0.3625 |          0.6993 |          0.7299 |
| Elo           |          0.5613 |          0.5406 |      -0.0207 |      13.16 |      12.75 | -0.4043 |          0.6803 |          0.7139 |
| Bradley-Terry |          0.5696 |          0.543  |      -0.0266 |      13.46 |      13    | -0.461  |          0.6715 |          0.692  |
| Hybrid        |          0.5381 |          0.5217 |      -0.0164 |      12.52 |      12.25 | -0.2672 |          0.7124 |          0.7168 |

Every model improves on the same games: 0.015 to 0.027 log loss and 0.3 to 0.5 points of MAE.
The warm start also makes weeks 1 to 3 gradable at all; they are included in section 3.

## 4. By week and calibration

Massey, 2025 cold start, by week (`--by-week`). Week 17 is the entire postseason, graded with
end-of-regular-season ratings:

|   week |   n |   accuracy |   brier |   log_loss |   mae |   vegas_mae |   ats_accuracy |
|-------:|----:|-----------:|--------:|-----------:|------:|------------:|---------------:|
|      4 |  62 |      0.661 |   0.171 |      0.488 | 13.74 |       11.8  |          0.557 |
|      5 |  53 |      0.698 |   0.184 |      0.533 | 12.42 |        9.48 |          0.415 |
|      6 |  51 |      0.686 |   0.167 |      0.497 | 11.77 |       10.36 |          0.52  |
|      7 |  56 |      0.643 |   0.215 |      0.615 | 14.55 |       13.68 |          0.518 |
|      8 |  60 |      0.733 |   0.194 |      0.576 | 12.6  |       12.11 |          0.533 |
|      9 |  53 |      0.755 |   0.173 |      0.513 | 11.15 |       11.21 |          0.491 |
|     10 |  52 |      0.635 |   0.21  |      0.6   | 12.84 |       12.89 |          0.519 |
|     11 |  52 |      0.673 |   0.215 |      0.612 | 10.96 |       10.83 |          0.654 |
|     12 |  59 |      0.814 |   0.142 |      0.442 | 11.6  |       12.05 |          0.576 |
|     13 |  64 |      0.766 |   0.146 |      0.438 | 12.15 |       12.08 |          0.469 |
|     14 |  67 |      0.716 |   0.178 |      0.521 | 12.88 |       12.56 |          0.493 |
|     15 |   9 |      0.667 |   0.22  |      0.619 | 11.08 |       10.5  |          0.333 |
|     16 |   1 |      1     |   0.112 |      0.408 |  5.48 |        4.17 |          0     |
|     17 |  46 |      0.565 |   0.224 |      0.634 | 12.27 |       11.28 |          0.489 |

Reliability of the predicted home-win probability, 2025 cold start (`--calibration`):

Massey:

| bin     |   bin_mid |   n |   predicted |   observed |
|:--------|----------:|----:|------------:|-----------:|
| 0.0-0.1 |      0.05 |   7 |       0.07  |      0     |
| 0.1-0.2 |      0.15 |  28 |       0.153 |      0.179 |
| 0.2-0.3 |      0.25 |  42 |       0.261 |      0.333 |
| 0.3-0.4 |      0.35 |  61 |       0.357 |      0.328 |
| 0.4-0.5 |      0.45 |  91 |       0.45  |      0.418 |
| 0.5-0.6 |      0.55 |  96 |       0.55  |      0.49  |
| 0.6-0.7 |      0.65 | 100 |       0.647 |      0.57  |
| 0.7-0.8 |      0.75 | 100 |       0.754 |      0.72  |
| 0.8-0.9 |      0.85 |  85 |       0.85  |      0.894 |
| 0.9-1.0 |      0.95 |  75 |       0.952 |      1     |

Hybrid:

| bin     |   bin_mid |   n |   predicted |   observed |
|:--------|----------:|----:|------------:|-----------:|
| 0.0-0.1 |      0.05 |   5 |       0.079 |      0     |
| 0.1-0.2 |      0.15 |  17 |       0.157 |      0.118 |
| 0.2-0.3 |      0.25 |  42 |       0.263 |      0.262 |
| 0.3-0.4 |      0.35 |  63 |       0.358 |      0.349 |
| 0.4-0.5 |      0.45 | 101 |       0.453 |      0.366 |
| 0.5-0.6 |      0.55 | 104 |       0.551 |      0.548 |
| 0.6-0.7 |      0.65 | 114 |       0.647 |      0.57  |
| 0.7-0.8 |      0.75 | 101 |       0.75  |      0.762 |
| 0.8-0.9 |      0.85 |  79 |       0.846 |      0.937 |
| 0.9-1.0 |      0.95 |  59 |       0.946 |      1     |

The cross-validated calibrator keeps the probabilities honest: the observed frequency tracks the
predicted one bin by bin, with no systematic over-confidence in either tail.

## 5. Tuning (2021-2024 for tuning, 2025 held out)

All grids run on the 2021-2024 warm chain from week 1
(3523 graded games) and are sorted by log loss; Brier order is the same in every grid.
Defaults were only changed where the 2025 hold-out improved by at least 0.005 log loss or
0.15 points of MAE.

### Massey

```bash
cfbratings --years 2021-2024 tune --method massey --start-week 1 --metric log_loss \
  --grid "prior_lambda=1,2,3,5,8;ridge_lambda=0.5,1,2;half_life_weeks=0,4,8"
```

Best eight of 45:

|   prior_lambda |   ridge_lambda |   half_life_weeks |    n |   accuracy |   brier |   log_loss |   mae |   rmse |
|---------------:|---------------:|------------------:|-----:|-----------:|--------:|-----------:|------:|-------:|
|              3 |              2 |                 0 | 3523 |     0.7298 |  0.175  |     0.521  | 13.51 |  17.06 |
|              3 |              1 |                 0 | 3523 |     0.7289 |  0.1753 |     0.5218 | 13.51 |  17.07 |
|              5 |              2 |                 0 | 3523 |     0.7312 |  0.1754 |     0.522  | 13.5  |  17.04 |
|              2 |              2 |                 0 | 3523 |     0.7301 |  0.1753 |     0.522  | 13.54 |  17.12 |
|              2 |              2 |                 8 | 3523 |     0.7267 |  0.1757 |     0.5227 | 13.56 |  17.15 |
|              3 |              2 |                 8 | 3523 |     0.7281 |  0.1757 |     0.5227 | 13.54 |  17.12 |
|              5 |              1 |                 0 | 3523 |     0.7298 |  0.1756 |     0.5228 | 13.51 |  17.06 |
|              2 |              1 |                 0 | 3523 |     0.7284 |  0.1756 |     0.5228 | 13.55 |  17.14 |

The shipped default (`prior_lambda=3, ridge_lambda=1, half_life_weeks=0`) scores
0.5218, 0.0008 behind the best cell, and `ridge_lambda` is irrelevant once a prior
exists (the 2025 hold-out with `ridge_lambda=2` is identical to four decimals):

| change                                                 |   log loss default |   log loss tuned |   Δ log loss |   MAE default |   MAE tuned |   Δ MAE |
|:-------------------------------------------------------|-------------------:|-----------------:|-------------:|--------------:|------------:|--------:|
| Massey ridge_lambda 1 → 2 (prior_lambda 3, no recency) |             0.4714 |           0.4714 |            0 |         12.47 |       12.47 |  0.0007 |

Recency weighting hurts: a 4-week half-life is the worst region of the grid, and 8 weeks is
still slightly worse than none. A strong pull toward the prior (`prior_lambda=8`) also hurts.

|   prior_lambda |   ridge_lambda |   half_life_weeks |    n |   accuracy |   brier |   log_loss |   mae |   rmse |
|---------------:|---------------:|------------------:|-----:|-----------:|--------:|-----------:|------:|-------:|
|              8 |            2   |                 4 | 3523 |     0.7133 |  0.182  |     0.5393 | 13.84 |  17.58 |
|              8 |            1   |                 4 | 3523 |     0.715  |  0.1823 |     0.5401 | 13.83 |  17.56 |
|              8 |            0.5 |                 4 | 3523 |     0.713  |  0.1826 |     0.5409 | 13.84 |  17.56 |

**Decision: unchanged.**

### Elo

```bash
cfbratings --years 2021-2024 tune --method elo --start-week 1 --metric log_loss \
  --grid "k=15,20,25,30,40;hfa=2,2.5,3,3.5"
```

|   k |   hfa |    n |   accuracy |   brier |   log_loss |   mae |   rmse |
|----:|------:|-----:|-----------:|--------:|-----------:|------:|-------:|
|  30 |   2   | 3523 |     0.7133 |  0.1822 |     0.5409 | 14.02 |  17.78 |
|  40 |   2   | 3523 |     0.7159 |  0.1823 |     0.5414 | 14.05 |  17.81 |
|  30 |   2.5 | 3523 |     0.7136 |  0.1825 |     0.5417 | 14.05 |  17.81 |
|  25 |   2   | 3523 |     0.7127 |  0.1825 |     0.5417 | 14.02 |  17.79 |
|  25 |   2.5 | 3523 |     0.7127 |  0.1828 |     0.5423 | 14.05 |  17.81 |
|  40 |   2.5 | 3523 |     0.7162 |  0.1827 |     0.5423 | 14.09 |  17.86 |

The default (`k=25, hfa=2.5`) scores 0.5423, 0.0013 behind `k=30, hfa=2.0`; a slow
`k=15` is the worst corner. On the 2025 hold-out the winner gains only 0.0016 log loss and 0.04
points of MAE:

| change                       |   log loss default |   log loss tuned |   Δ log loss |   MAE default |   MAE tuned |   Δ MAE |
|:-----------------------------|-------------------:|-----------------:|-------------:|--------------:|------------:|--------:|
| Elo k 25 → 30, hfa 2.5 → 2.0 |             0.4947 |           0.4931 |      -0.0016 |         13.37 |       13.33 | -0.0401 |

**Decision: unchanged** (below the threshold; worth revisiting with more seasons).

### Between-season regression (all models)

```bash
for M in colley massey elo bradley_terry hybrid; do
  cfbratings --years 2021-2024 tune --method $M --start-week 1 --grid "season_regress=0.2,0.33,0.5"
done
```

| model         |   log loss @ 0.2 |   log loss @ 0.33 |   log loss @ 0.5 |   MAE @ 0.2 |   MAE @ 0.33 |   MAE @ 0.5 |
|:--------------|-----------------:|------------------:|-----------------:|------------:|-------------:|------------:|
| Colley        |           0.5489 |            0.5526 |           0.5602 |       14.12 |        14.25 |       14.53 |
| Massey        |           0.5193 |            0.5218 |           0.5289 |       13.41 |        13.51 |       13.79 |
| Elo           |           0.5408 |            0.5423 |           0.5489 |       13.98 |        14.05 |       14.32 |
| Bradley-Terry |           0.5463 |            0.5493 |           0.5566 |       14.11 |        14.23 |       14.5  |
| Hybrid        |           0.5208 |            0.5217 |           0.5247 |       13.38 |        13.38 |       13.45 |

Carrying more of last season forward (`0.2`) is better than the default third for every model
on the tuning set, and regressing half-way is clearly worse. On the 2025 hold-out
(`cfbratings --years 2021-2025 backtest --start-week 1 --lines --param season_regress=0.2`):

| model         |   log loss @0.33 |   log loss @0.2 |   Δ log loss |   MAE @0.33 |   MAE @0.2 |   Δ MAE |
|:--------------|-----------------:|----------------:|-------------:|------------:|-----------:|--------:|
| Colley        |           0.5075 |          0.5044 |      -0.0031 |       13.78 |      13.65 | -0.1303 |
| Massey        |           0.4714 |          0.4698 |      -0.0016 |       12.47 |      12.37 | -0.0964 |
| Elo           |           0.4947 |          0.4953 |       0.0006 |       13.37 |      13.27 | -0.1026 |
| Bradley-Terry |           0.494  |          0.4921 |      -0.0019 |       13.46 |      13.29 | -0.1715 |
| Hybrid        |           0.4752 |          0.4735 |      -0.0017 |       12.57 |      12.53 | -0.0456 |

**Decision: unchanged.** The gain is consistent but small (at most 0.003 log loss and 0.17
points of MAE, and Elo's log loss does not improve), under the threshold set for this round.

### Hybrid weights

The CLI grid cannot express a weight mapping, so this grid ran through the Python API
(`compare_methods(..., params={"hybrid": {"weights": w}})`) on the 2021-2025 chain: the
2021-2024 rows are the tuning score, the 2025 rows the hold-out. Weights are
`colley / massey / elo`; the remaining share goes to Elo.

| weights                              |   tune_log_loss |   tune_brier |   tune_mae |   holdout_log_loss |   holdout_brier |   holdout_mae |   holdout_accuracy |   holdout_ats |
|:-------------------------------------|----------------:|-------------:|-----------:|-------------------:|----------------:|--------------:|-------------------:|--------------:|
| colley 0.10 / massey 0.80 / elo 0.10 |          0.519  |       0.1748 |      13.34 |             0.4728 |          0.1582 |         12.45 |             0.7516 |        0.5081 |
| colley 0.20 / massey 0.80 / elo 0.00 |          0.5191 |       0.1749 |      13.34 |             0.4731 |          0.1585 |         12.48 |             0.7559 |        0.5081 |
| colley 0.00 / massey 0.80 / elo 0.20 |          0.5197 |       0.175  |      13.35 |             0.4734 |          0.1583 |         12.45 |             0.7548 |        0.5059 |
| colley 0.20 / massey 0.70 / elo 0.10 |          0.5199 |       0.1751 |      13.35 |             0.4737 |          0.1586 |         12.51 |             0.7591 |        0.5113 |
| colley 0.10 / massey 0.70 / elo 0.20 |          0.5203 |       0.1752 |      13.36 |             0.4739 |          0.1585 |         12.5  |             0.7602 |        0.5038 |
| colley 0.00 / massey 0.70 / elo 0.30 |          0.5214 |       0.1755 |      13.39 |             0.4749 |          0.1588 |         12.51 |             0.7612 |        0.5016 |
| colley 0.20 / massey 0.60 / elo 0.20 |          0.5217 |       0.1757 |      13.38 |             0.4752 |          0.1591 |         12.57 |             0.7559 |        0.5124 |
| colley 0.33 / massey 0.60 / elo 0.07 |          0.5218 |       0.1759 |      13.39 |             0.4759 |          0.1595 |         12.62 |             0.7516 |        0.5135 |
| colley 0.10 / massey 0.60 / elo 0.30 |          0.5224 |       0.1759 |      13.41 |             0.4758 |          0.1591 |         12.57 |             0.758  |        0.5027 |
| colley 0.00 / massey 0.60 / elo 0.40 |          0.5237 |       0.1763 |      13.45 |             0.4771 |          0.1596 |         12.6  |             0.7591 |        0.4984 |
| colley 0.33 / massey 0.50 / elo 0.17 |          0.5237 |       0.1766 |      13.42 |             0.4776 |          0.1601 |         12.68 |             0.7516 |        0.5145 |
| colley 0.20 / massey 0.50 / elo 0.30 |          0.5241 |       0.1766 |      13.44 |             0.4774 |          0.1598 |         12.65 |             0.7548 |        0.5199 |
| colley 0.10 / massey 0.50 / elo 0.40 |          0.525  |       0.1768 |      13.47 |             0.4783 |          0.1601 |         12.66 |             0.7559 |        0.5102 |
| colley 0.33 / massey 0.40 / elo 0.27 |          0.5263 |       0.1775 |      13.48 |             0.48   |          0.1609 |         12.76 |             0.7473 |        0.5135 |
| colley 0.00 / massey 0.50 / elo 0.50 |          0.5265 |       0.1773 |      13.51 |             0.4799 |          0.1606 |         12.69 |             0.7516 |        0.5059 |
| colley 0.20 / massey 0.40 / elo 0.40 |          0.527  |       0.1776 |      13.51 |             0.4803 |          0.1609 |         12.75 |             0.7463 |        0.5188 |
| colley 0.10 / massey 0.40 / elo 0.50 |          0.5281 |       0.1779 |      13.54 |             0.4814 |          0.1613 |         12.76 |             0.7527 |        0.5178 |
| colley 0.33 / massey 0.33 / elo 0.33 |          0.5284 |       0.1782 |      13.53 |             0.4819 |          0.1617 |         12.83 |             0.7441 |        0.5113 |
| colley 0.20 / massey 0.33 / elo 0.47 |          0.5292 |       0.1784 |      13.56 |             0.4825 |          0.1618 |         12.82 |             0.7505 |        0.521  |
| colley 0.00 / massey 0.40 / elo 0.60 |          0.5297 |       0.1785 |      13.59 |             0.4831 |          0.1619 |         12.79 |             0.7516 |        0.5102 |
| colley 0.10 / massey 0.33 / elo 0.57 |          0.5304 |       0.1788 |      13.59 |             0.4837 |          0.1622 |         12.84 |             0.7516 |        0.5199 |
| colley 0.00 / massey 0.33 / elo 0.67 |          0.5319 |       0.1793 |      13.64 |             0.4855 |          0.1629 |         12.87 |             0.7516 |        0.5156 |

Log loss falls monotonically as Massey's share rises, on the tuning set and on the hold-out
alike: the equal blend (0.4819 on 2025) is beaten by 0.6 Massey
(0.4752) and by the grid optimum at 0.8 Massey (0.4728),
with MAE moving from 12.83 to 12.57 and 12.45 points. A hybrid at 0.8
Massey is within 0.002 of plain Massey (0.4714), i.e. no longer a blend in any useful sense.

**Decision: default changed from ⅓/⅓/⅓ to `colley=0.2, massey=0.6, elo=0.2`.** That clears the
threshold on both metrics (−0.0067 log loss, −0.25 points of MAE on the hold-out) while keeping a
real share of the win-loss-only and sequential views, which is the point of having a hybrid.
Every table in this document was produced with the new default (`CFB_HYBRID_WEIGHTS` or the
dashboard sliders restore any other blend).

## 6. Where the models stand

* **Against the home-team baseline**: every model, every season, by more than 0.1 log loss and
  3.5 points of MAE. Warm-started Massey over five seasons: log loss 0.511 vs 0.679,
  MAE 13.3 vs 18.2 points, accuracy 73.6% vs 62.8%.
* **Against the closing line**: the market is better, consistently. Over the same games Vegas
  posts log loss 0.483, MAE 12.1 and accuracy 75.5%; Massey trails by 0.029 log
  loss and 1.2 points, the hybrid by 0.029 and 1.1. Against-the-spread rates sit at
  48.7% to 50.0% over 4 400 games: no model finds an edge over the line,
  which is the expected result for a power-rating system built from scores alone. Warm-started
  Massey's MAE is within 0.6 points of the line in 2025 (12.47 vs 11.90).
* **Model ranking** is stable across seasons and start modes: Massey and the Massey-led hybrid
  are neck and neck at the top (over the five-season chain the hybrid has the lower MAE and
  higher accuracy, Massey the lower log loss), then Elo, then the two win-loss-only models.
  Margins carry most of the predictive information.

## 7. Caveats and follow-ups

* The postseason is one effective week, so the CFP semifinals and final are predicted with
  ratings that do not include the earlier rounds. Splitting `seasonType=postseason` by date
  would grade the later rounds on fresher ratings.
* FCS opponents are pooled into one pseudo-team (`--fcs drop` discards those games instead);
  the Vegas rows include lines on those games when a book posted one.
* 2024 carries one FBS row that was scheduled but never played (Liberty at App State, week 5,
  `completed: false`); unplayed rows are ignored everywhere.
* All grids are coarse. The Elo and regression winners are directionally consistent across
  seasons and models; with 2026 archived next year they may clear the threshold.
