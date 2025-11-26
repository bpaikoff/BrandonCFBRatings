#!/usr/bin/env python3
"""
Simple tests to verify robustness improvements
"""
import numpy as np
from cfbratings.models.colley import build_colley, solve_colley
from cfbratings.models.massey import build_massey, solve_massey
from cfbratings.models.elo import run_elo
from cfbratings.models.hybrid import hybrid_rating

def test_empty_team_list():
    """Test that models handle empty team lists gracefully"""
    print("Testing empty team list...")

    # Empty team list
    teams = []
    games = []

    # Colley
    C, b = build_colley(teams, games)
    r = solve_colley(C, b)
    assert len(r) == 0, "Colley should return empty array for empty teams"
    print("  ✓ Colley handles empty teams")

    # Massey
    M, mb = build_massey(teams, games)
    r = solve_massey(M, mb)
    assert len(r) == 0, "Massey should return empty array for empty teams"
    print("  ✓ Massey handles empty teams")

    # Elo
    ratings = run_elo(teams, games)
    assert len(ratings) == 0, "Elo should return empty dict for empty teams"
    print("  ✓ Elo handles empty teams")

    # Hybrid
    ratings = hybrid_rating(teams, games)
    assert len(ratings) == 0, "Hybrid should return empty dict for empty teams"
    print("  ✓ Hybrid handles empty teams")

def test_no_completed_games():
    """Test that models handle teams with no games"""
    print("\nTesting teams with no completed games...")

    teams = ["Team A", "Team B", "Team C"]
    games = []

    # Colley - should return default ratings
    C, b = build_colley(teams, games)
    r = solve_colley(C, b)
    assert len(r) == 3, "Colley should return ratings for all teams"
    print(f"  ✓ Colley ratings with no games: {r}")

    # Massey - should return zeros
    M, mb = build_massey(teams, games)
    r = solve_massey(M, mb)
    assert len(r) == 3, "Massey should return ratings for all teams"
    assert np.allclose(r, 0), "Massey should return zeros for no games"
    print(f"  ✓ Massey ratings with no games: {r}")

    # Elo - should return initial ratings
    ratings = run_elo(teams, games, init=1500.0)
    assert len(ratings) == 3, "Elo should return ratings for all teams"
    assert all(r == 1500.0 for r in ratings.values()), "Elo should return init values"
    print(f"  ✓ Elo ratings with no games: {list(ratings.values())}")

def test_extreme_margins():
    """Test that Massey caps extreme margins"""
    print("\nTesting extreme margin capping...")

    teams = ["Team A", "Team B"]
    # Game with extreme margin (100-0)
    games = [{
        "completed": True,
        "homeTeam": "Team A",
        "awayTeam": "Team B",
        "homePoints": 100,
        "awayPoints": 0,
        "week": 1,
        "season": 2024
    }]

    # Build Massey with margin cap
    M, mb = build_massey(teams, games, max_margin=50.0)
    r = solve_massey(M, mb)
    print(f"  ✓ Massey with capped margin: {r}")

    # Build Massey without margin cap (very large cap)
    M2, mb2 = build_massey(teams, games, max_margin=1000.0)
    r2 = solve_massey(M2, mb2)
    print(f"  ✓ Massey without cap: {r2}")

    # The capped version should have smaller rating differences
    assert abs(r[0] - r[1]) < abs(r2[0] - r2[1]), "Capping should reduce rating differences"
    print("  ✓ Margin capping reduces extreme differences")

def test_identical_ratings():
    """Test that hybrid handles identical ratings (zero std dev)"""
    print("\nTesting identical ratings edge case...")

    teams = ["Team A", "Team B"]
    # Two teams with identical records (tie game)
    games = [{
        "completed": True,
        "homeTeam": "Team A",
        "awayTeam": "Team B",
        "homePoints": 21,
        "awayPoints": 21,
        "week": 1,
        "season": 2024
    }]

    ratings = hybrid_rating(teams, games)
    assert len(ratings) == 2, "Hybrid should return ratings for both teams"
    print(f"  ✓ Hybrid with tie game: {ratings}")

def test_singular_matrix_handling():
    """Test that singular matrix cases are handled"""
    print("\nTesting singular matrix handling...")

    # Create isolated components (Team C never plays)
    teams = ["Team A", "Team B", "Team C"]
    games = [{
        "completed": True,
        "homeTeam": "Team A",
        "awayTeam": "Team B",
        "homePoints": 28,
        "awayPoints": 21,
        "week": 1,
        "season": 2024
    }]

    # This could potentially create issues in some configurations
    C, b = build_colley(teams, games)
    r = solve_colley(C, b)
    assert len(r) == 3, "Should return ratings for all teams"
    print(f"  ✓ Colley with isolated team: {r}")

    M, mb = build_massey(teams, games)
    r = solve_massey(M, mb)
    assert len(r) == 3, "Should return ratings for all teams"
    print(f"  ✓ Massey with isolated team: {r}")

def test_max_week_empty_games():
    """Test that max() operations don't crash on empty completed games"""
    print("\nTesting max week with no completed games...")

    games = [
        {"completed": False, "week": 1},
        {"completed": False, "week": 2}
    ]

    # Simulate the pattern used in the codebase
    completed_games = [g for g in games if g.get("completed", False)]
    max_week = max((g.get("week", 0) for g in completed_games), default=0)

    assert max_week == 0, "Should return 0 for no completed games"
    print(f"  ✓ Max week with no completed games: {max_week}")

if __name__ == "__main__":
    print("=" * 60)
    print("Running Robustness Tests")
    print("=" * 60)

    try:
        test_empty_team_list()
        test_no_completed_games()
        test_extreme_margins()
        test_identical_ratings()
        test_singular_matrix_handling()
        test_max_week_empty_games()

        print("\n" + "=" * 60)
        print("✓ All robustness tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
