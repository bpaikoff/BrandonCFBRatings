# Test fixtures

* `cfbd_v5_shapes.json` — row shapes copied from the CollegeFootballData OpenAPI spec (API 5.32.1):
  one dict per endpoint (`/teams/fbs`, `/games`, `/lines`) carrying **every required key** of
  the `Team`, `Game` and `BettingGame`/`GameLine` schemas, plus the `SeasonType` enum. The values
  are illustrative, not a recorded response. `tests/test_cfbd_payload_shapes.py` checks that the
  parsers consume these shapes, that FCS pooling behaves on them and that `spread` follows the
  documented *home-relative, negative favours home* convention.
* `games_2024_sample.json` (not yet present) — a real trimmed payload. Once `data/archive/` holds
  the 2024 season, generate it **offline** with

  ```bash
  python tests/fixtures/trim_payload.py --year 2024 --teams "Ohio State" "Notre Dame" "Oregon" "Texas"
  ```

  which keeps only those teams' games (a few dozen rows), the matching betting lines and the
  FBS team rows, and strips nothing else, so the fixture stays a faithful sample. No API call
  is made and no key material is written.
