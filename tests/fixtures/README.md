# Test fixtures

* `cfbd_v5_shapes.json` — row shapes copied from the CollegeFootballData OpenAPI spec (API 5.32.1):
  one dict per endpoint (`/teams/fbs`, `/games`, `/lines`) carrying **every required key** of
  the `Team`, `Game` and `BettingGame`/`GameLine` schemas, plus the `SeasonType` enum. The values
  are illustrative, not a recorded response. `tests/test_cfbd_payload_shapes.py` checks that the
  parsers consume these shapes, that FCS pooling behaves on them and that `spread` follows the
  documented *home-relative, negative favours home* convention.
* `games_2024_sample.json` — a **real** payload, trimmed offline from `data/archive/` with

  ```bash
  python tests/fixtures/trim_payload.py --year 2024 --teams "Ohio State" "Notre Dame" "Oregon" "Texas"
  ```

  It keeps those teams' 58 games exactly as `/games?seasonType=both` returned them (49
  regular-season rows, both conference title games and all nine of their College Football Playoff
  games), the 58 matching `/lines` rows and the `/teams/fbs` row of each of the 46 FBS schools
  involved, so roster-based FCS pooling works on the sample. No API call is made and no key
  material is written. `tests/test_real_payload_sample.py` pins down what the real data looks
  like: every bowl and playoff round is `week: 1` of `seasonType: postseason` (renumbered to one
  effective week after the last regular week), first-round playoff games are on campus and not
  neutral, the only FCS opponent pools, the four records come out as 14-2, 14-2, 13-1 and 13-3,
  and the three books (ESPN Bet, DraftKings, Bovada; there is no `consensus` provider) agree on
  the sign convention.
