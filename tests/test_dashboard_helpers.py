"""Pure helpers behind the dashboard: rank-change labels, data-freshness and kickoff labels."""

from __future__ import annotations

import time

from cfbratings.analytics import age_label, freshness_label, kickoff_label
from cfbratings.data import FCS_POOL, CacheInfo
from cfbratings.snapshots import previous_snapshot_week, rank_changes, rank_map

# --- rank changes ---------------------------------------------------------------------------


def test_rank_map_orders_best_first_and_skips_the_pool():
    ranks = rank_map({"A": 1.0, "B": 3.0, "C": 2.0, FCS_POOL: 99.0})
    assert ranks == {"B": 1, "C": 2, "A": 3}
    assert rank_map({"A": 1.0, "B": 1.0}) == {"A": 1, "B": 2}  # ties break by name
    assert rank_map({}) == {}


def test_previous_snapshot_week():
    snaps = {0: {"A": 1.0}, 1: {"A": 1.0}, 3: {"A": 1.0}}
    assert previous_snapshot_week(snaps) == 1  # before the latest (3)
    assert previous_snapshot_week(snaps, 3) == 1
    assert previous_snapshot_week(snaps, 1) == 0  # the preseason prior counts
    assert previous_snapshot_week(snaps, 0) is None
    assert previous_snapshot_week({}) is None


def test_rank_changes_arrows():
    last_week = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.5}  # A, B, C, D
    this_week = {"C": 3.0, "A": 2.0, "B": 1.0, "D": 0.5}  # C, A, B, D
    current = rank_map(this_week)
    changes = rank_changes(current, {1: last_week, 2: this_week})
    assert changes == {"C": "▲ 2", "A": "▼ 1", "B": "▼ 1", "D": "–"}


def test_rank_changes_explicit_week_and_new_team():
    snaps = {1: {"A": 2.0, "B": 1.0}, 2: {"A": 1.0, "B": 2.0, "C": 3.0}}
    changes = rank_changes({"C": 1, "B": 2, "A": 3}, snaps, week=2)
    assert changes == {"C": "new", "B": "–", "A": "▼ 2"}  # A was first, now third
    # Comparing week 1 against nothing earlier: every team is unchanged.
    assert rank_changes({"A": 1, "B": 2}, snaps, week=1) == {"A": "–", "B": "–"}


def test_rank_changes_without_earlier_week_never_raises():
    current = {"A": 1, "B": 2}
    assert rank_changes(current, {}) == {"A": "–", "B": "–"}  # preseason: no snapshots at all
    assert rank_changes(current, {1: {"A": 2.0, "B": 1.0}}) == {"A": "–", "B": "–"}  # only one week
    assert rank_changes({}, {1: {"A": 1.0}, 2: {"A": 1.0}}) == {}


# --- freshness --------------------------------------------------------------------------------


def test_age_label_units():
    assert age_label(5) == "just now"
    assert age_label(-5) == "just now"
    assert age_label(20 * 60) == "20 min ago"
    assert age_label(3 * 3600 + 120) == "3 h ago"
    assert age_label(47 * 3600) == "47 h ago"
    assert age_label(3 * 86400) == "3 d ago"


def test_freshness_label_tiers():
    now = int(time.time())
    assert freshness_label(2024, CacheInfo(cached_at=now - 10, final=True, tier="archive")) == "2024: archived · final"
    assert (
        freshness_label(2025, CacheInfo(cached_at=now - 3 * 3600, final=False, tier="cache")) == "2025: cached 3 h ago"
    )
    assert (
        freshness_label(2025, CacheInfo(cached_at=now - 90, final=True, tier="cache"))
        == "2025: cached 2 min ago · final"
    )
    assert freshness_label(2026, None) == "2026: not cached · next load calls the API"


# --- kickoff ----------------------------------------------------------------------------------


def test_kickoff_label_formats_and_shifts_to_eastern():
    assert kickoff_label("2025-11-22T20:30:00.000Z") == "Sat 22 Nov"  # 3:30 pm ET
    assert kickoff_label("2025-11-23T00:30:00.000Z") == "Sat 22 Nov"  # 7:30 pm ET prime time, still Saturday
    assert kickoff_label("2025-09-01") == "Mon 1 Sep"  # naive stamp used as given
    assert kickoff_label(None) is None
    assert kickoff_label("") is None
    assert kickoff_label("TBD") is None
