# Archived CFBD payloads

Gzipped responses for *finished* seasons (`teams_YYYY`, `games_YYYY_both`, `lines_YYYY_both`), written
automatically by the client and read as a fallback when `data/cache/` is empty. Commit these so a fresh
clone never re-downloads a season. The current season is never archived (it is still changing).

| season | payloads | fetched |
|---|---|---|
| 2021 | teams, games, lines | 2026-10-10 |
| 2022 | teams, games, lines | 2026-10-10 |
| 2023 | teams, games, lines | 2026-10-10 |
| 2024 | teams, games, lines | 2026-10-10 |
| 2025 | teams, games, lines | 2026-10-10 |

Any command that names one of these seasons (`--year 2024`, `--years 2021-2025`, with or without
`--lines`) runs with **zero** API calls. The default season is the one in progress, so a command
*without* `--year`/`--years` fetches it (two calls, three with `--lines`) and refreshes it again
after `CFB_CACHE_TTL_HOURS`; it never lands here.
