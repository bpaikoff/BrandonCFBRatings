"""A small JSON file cache with time-to-live semantics and a committed archive tier.

Two tiers:

* ``directory`` -- the working cache (gitignored). Entries carry a timestamp and expire
  after ``ttl_hours`` unless marked ``final``.
* ``archive_dir`` -- optional, gzipped, meant to be committed to the repository. Only
  ``final`` payloads (finished seasons) are written here, and a read that misses the
  working cache falls back to it. A fresh clone therefore never re-downloads a season
  that any earlier session already fetched.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from pathlib import Path
from typing import Any

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class JsonCache:
    def __init__(
        self, directory: str | Path, ttl_hours: float | None = None, archive_dir: str | Path | None = None
    ) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = None if ttl_hours is None else ttl_hours * 3600.0
        self.archive_dir = Path(archive_dir) if archive_dir else None

    def path(self, key: str) -> Path:
        return self.directory / f"{_SAFE.sub('_', key)}.json"

    def archive_path(self, key: str) -> Path | None:
        if self.archive_dir is None:
            return None
        return self.archive_dir / f"{_SAFE.sub('_', key)}.json.gz"

    # --- reads -----------------------------------------------------------------------
    def get(self, key: str, *, ignore_ttl: bool = False) -> Any | None:
        """Return the cached payload or ``None`` when missing / stale / unreadable."""
        p = self.path(key)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as fh:
                    wrapper = json.load(fh)
            except (OSError, json.JSONDecodeError):
                wrapper = None
            if isinstance(wrapper, dict) and "data" in wrapper:
                fresh = ignore_ttl or self.ttl_seconds is None or wrapper.get("_final", False)
                if not fresh:
                    age = time.time() - float(wrapper.get("_cached_at", 0))
                    fresh = age <= self.ttl_seconds
                if fresh:
                    return wrapper["data"]
        return self.get_archived(key)

    def get_archived(self, key: str) -> Any | None:
        ap = self.archive_path(key)
        if ap is None or not ap.exists():
            return None
        try:
            with gzip.open(ap, "rt", encoding="utf-8") as fh:
                wrapper = json.load(fh)
        except (OSError, EOFError, json.JSONDecodeError):
            return None
        return wrapper.get("data") if isinstance(wrapper, dict) else None

    # --- writes ----------------------------------------------------------------------
    def put(self, key: str, data: Any, *, final: bool = False) -> Path:
        """Store ``data``. ``final`` marks payloads that never go stale and are archived."""
        self.directory.mkdir(parents=True, exist_ok=True)
        p = self.path(key)
        wrapper = {"_cached_at": int(time.time()), "_final": final, "data": data}
        tmp = p.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(wrapper, fh, indent=1)
        tmp.replace(p)
        if final:
            self.archive(key, data)
        return p

    def archive(self, key: str, data: Any) -> Path | None:
        ap = self.archive_path(key)
        if ap is None:
            return None
        ap.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(ap, "wt", encoding="utf-8") as fh:
            json.dump({"_archived_at": int(time.time()), "data": data}, fh, separators=(",", ":"))
        return ap

    def delete(self, key: str) -> None:
        p = self.path(key)
        if p.exists():
            p.unlink()

    def clear(self) -> int:
        """Remove working-cache files (the archive is left untouched)."""
        n = 0
        if self.directory.exists():
            for p in self.directory.glob("*.json"):
                p.unlink()
                n += 1
        return n
