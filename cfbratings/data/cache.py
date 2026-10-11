"""A small JSON file cache with time-to-live semantics and a committed archive tier.

Two tiers:

* ``directory`` -- the working cache (gitignored). Entries carry a timestamp and expire
  after ``ttl_hours`` unless marked ``final``.
* ``archive_dir`` -- optional, gzipped, meant to be committed to the repository. Only
  ``final`` payloads (finished seasons) are written here, and a read that misses the
  working cache falls back to it. A fresh clone therefore never re-downloads a season
  that any earlier session already fetched.

:meth:`JsonCache.info` answers "which tier would :meth:`JsonCache.get` serve this key from,
and how old is it?" without handing back the payload, which is what a "data as of ..."
caption in a UI needs.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class CacheInfo:
    """Provenance of a cached payload: when it was fetched and which tier would serve it."""

    cached_at: int  # epoch seconds of the API call that produced the payload
    final: bool  # a finished season: never expires and is written to the archive too
    tier: str  # "cache" (working cache) or "archive" (gzipped, committed fallback)

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.cached_at)


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
        wrapper = self._read_working(key)
        if wrapper is not None and self._is_fresh(wrapper, ignore_ttl):
            return wrapper["data"]
        return self.get_archived(key)

    def get_archived(self, key: str) -> Any | None:
        wrapper = self._read_archive(key)
        return None if wrapper is None else wrapper["data"]

    def info(self, key: str, *, ignore_ttl: bool = False) -> CacheInfo | None:
        """Describe what :meth:`get` would serve for ``key``; ``None`` when it would miss.

        Reads exactly the files :meth:`get` reads and nothing more; the payload stays on disk.
        """
        wrapper = self._read_working(key)
        if wrapper is not None and self._is_fresh(wrapper, ignore_ttl):
            cached_at, final = int(wrapper.get("_cached_at", 0)), bool(wrapper.get("_final", False))
            return CacheInfo(cached_at=cached_at, final=final, tier="cache")
        archived = self._read_archive(key)
        if archived is None:
            return None
        return CacheInfo(cached_at=int(archived.get("_archived_at", 0)), final=True, tier="archive")

    def _is_fresh(self, wrapper: dict[str, Any], ignore_ttl: bool) -> bool:
        if ignore_ttl or self.ttl_seconds is None or wrapper.get("_final", False):
            return True
        return time.time() - float(wrapper.get("_cached_at", 0)) <= self.ttl_seconds

    def _read_working(self, key: str) -> dict[str, Any] | None:
        """The working-cache wrapper ``{_cached_at, _final, data}``, or ``None`` if missing / unreadable."""
        p = self.path(key)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as fh:
                wrapper = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        return wrapper if isinstance(wrapper, dict) and "data" in wrapper else None

    def _read_archive(self, key: str) -> dict[str, Any] | None:
        """The archive wrapper ``{_archived_at, data}``, or ``None`` if there is no usable archive."""
        ap = self.archive_path(key)
        if ap is None or not ap.exists():
            return None
        try:
            with gzip.open(ap, "rt", encoding="utf-8") as fh:
                wrapper = json.load(fh)
        except (OSError, EOFError, json.JSONDecodeError):
            return None
        return wrapper if isinstance(wrapper, dict) and "data" in wrapper else None

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
