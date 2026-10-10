# Archived CFBD payloads

Gzipped responses for *finished* seasons (`teams_YYYY`, `games_YYYY_both`, `lines_YYYY_both`), written
automatically by the client and read as a fallback when `data/cache/` is empty. Commit these so a fresh
clone never re-downloads a season. The current season is never archived (it is still changing).
