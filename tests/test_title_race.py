import pandas as pd
import pytest

from models import title_race as tr


# ---------------------------------------------------------------------------
# match_outcome_probabilities
# ---------------------------------------------------------------------------

def test_probabilities_always_sum_to_one():
    for gap in [-500, -100, 0, 100, 500, 1000]:
        p_h, p_d, p_a = tr.match_outcome_probabilities(1500, 1500 - gap, home_advantage=0)
        assert (p_h + p_d + p_a) == pytest.approx(1.0)


def test_probabilities_never_negative_at_extreme_gaps():
    p_h, p_d, p_a = tr.match_outcome_probabilities(2200, 1000, home_advantage=0)
    assert p_h >= 0 and p_d >= 0 and p_a >= 0
    p_h, p_d, p_a = tr.match_outcome_probabilities(1000, 2200, home_advantage=0)
    assert p_h >= 0 and p_d >= 0 and p_a >= 0


def test_equal_ratings_no_home_advantage_is_symmetric():
    p_h, p_d, p_a = tr.match_outcome_probabilities(1500, 1500, home_advantage=0)
    assert p_h == pytest.approx(p_a)
    assert p_d == pytest.approx(tr.DEFAULT_DRAW_RATE)


def test_home_advantage_favors_the_home_team_in_an_otherwise_even_match():
    p_h, p_d, p_a = tr.match_outcome_probabilities(1500, 1500, home_advantage=100)
    assert p_h > p_a


def test_big_favorite_has_high_win_probability_and_small_draw_probability():
    p_h, p_d, p_a = tr.match_outcome_probabilities(2000, 1200, home_advantage=0)
    assert p_h > 0.9
    assert p_d < 0.05


# ---------------------------------------------------------------------------
# simulate_title_race
# ---------------------------------------------------------------------------

def test_seeded_simulation_is_reproducible():
    points = {"A": 80, "B": 78}
    fixtures = [("A", "B"), ("B", "A")]
    ratings = {"A": 1700, "B": 1650}
    r1 = tr.simulate_title_race(points, fixtures, ratings, n_sims=500, seed=42)
    r2 = tr.simulate_title_race(points, fixtures, ratings, n_sims=500, seed=42)
    assert r1 == r2


def test_probabilities_sum_to_one():
    points = {"A": 70, "B": 68, "C": 65}
    fixtures = [("A", "B"), ("B", "C"), ("C", "A")]
    ratings = {"A": 1600, "B": 1550, "C": 1500}
    result = tr.simulate_title_race(points, fixtures, ratings, n_sims=1000, seed=1)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


def test_insurmountable_lead_with_no_fixtures_left_wins_with_certainty():
    points = {"A": 90, "B": 40}
    result = tr.simulate_title_race(points, fixtures_remaining=[], ratings={"A": 1500, "B": 1500},
                                     n_sims=200, seed=1)
    assert result["A"] == pytest.approx(1.0)
    assert result["B"] == pytest.approx(0.0)


def test_evenly_matched_decisive_head_to_head_is_roughly_even():
    """Two teams level on points, with the only remaining fixture being
    between them, and equal ratings -- title odds should land close to
    50/50 (statistical, not exact -- checked with a wide tolerance over
    enough simulations)."""
    points = {"A": 70, "B": 70}
    fixtures = [("A", "B")]
    ratings = {"A": 1600, "B": 1600}
    result = tr.simulate_title_race(points, fixtures, ratings, n_sims=5000,
                                     home_advantage=0, seed=7)
    assert 0.35 < result["A"] < 0.65
    assert 0.35 < result["B"] < 0.65


def test_unknown_team_falls_back_to_default_rating_without_crashing():
    points = {"A": 70, "B": 70}
    fixtures = [("A", "B")]
    result = tr.simulate_title_race(points, fixtures, ratings={}, n_sims=100, seed=1)
    assert sum(result.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# title_race_path
# ---------------------------------------------------------------------------

def _season_matches():
    return pd.DataFrame([
        {"date": "2024-08-10", "home_team": "A", "away_team": "B", "result": "H"},
        {"date": "2024-08-17", "home_team": "C", "away_team": "A", "result": "A"},
        {"date": "2024-08-24", "home_team": "B", "away_team": "C", "result": "D"},
    ])


def _empty_history():
    return pd.DataFrame(columns=["date", "team", "rating_after"])


def test_title_race_path_has_one_checkpoint_per_distinct_match_date():
    result = tr.title_race_path(_season_matches(), _empty_history(), n_sims=100, seed=1)
    assert sorted(result["date"].unique()) == ["2024-08-10", "2024-08-17", "2024-08-24"]


def test_title_race_path_probabilities_sum_to_one_at_every_checkpoint():
    result = tr.title_race_path(_season_matches(), _empty_history(), n_sims=200, seed=1)
    sums = result.groupby("date")["win_probability"].sum()
    # Each team's probability is individually rounded to 4dp for display,
    # so a few of them summed can be off by a couple of ten-thousandths.
    for total in sums:
        assert total == pytest.approx(1.0, abs=1e-3)


def test_title_race_path_final_checkpoint_is_certain_since_no_fixtures_remain():
    """At the last matchday, there's nothing left to simulate -- the
    'winner' of every simulated season is just whoever actually finished
    top, so the final checkpoint's probabilities collapse to 0/1."""
    result = tr.title_race_path(_season_matches(), _empty_history(), n_sims=200, seed=1)
    final = result[result["date"] == "2024-08-24"].set_index("team")["win_probability"]
    # M1: A(home) beats B -> A+3. M2: C(home) vs A(away), result A -> A wins again, A+3.
    # M3: B(home) vs C(away), draw -> B+1, C+1. Final: A=6, B=1, C=1 -- A tops the table.
    assert final["A"] == pytest.approx(1.0)
    assert final["B"] == pytest.approx(0.0)
    assert final["C"] == pytest.approx(0.0)
