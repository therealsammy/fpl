import pandas as pd
import pytest

from models import elo


def match(date, league, home, away, home_goals, away_goals, result):
    return {"date": date, "league": league, "home_team": home, "away_team": away,
            "home_goals": home_goals, "away_goals": away_goals, "result": result}


# ---------------------------------------------------------------------------
# load_all_matches
# ---------------------------------------------------------------------------

def test_load_all_matches_preserves_leading_zero_season_codes(tmp_path, monkeypatch):
    """Regression: season '0001' (2000/01) silently became the integer 1
    on CSV round-trip without an explicit string dtype -- verified against
    a real collected file before fixing this."""
    monkeypatch.setattr(elo, "MATCHES_ROOT", tmp_path)
    league_dir = tmp_path / "E0"
    league_dir.mkdir()
    pd.DataFrame([{
        "league": "E0", "season": "0001", "date": "2000-08-19",
        "home_team_raw": "Charlton", "away_team_raw": "Man City",
        "home_goals": 4, "away_goals": 0, "result": "H",
    }]).to_csv(league_dir / "0001.csv", index=False)

    result = elo.load_all_matches()
    assert result.iloc[0]["season"] == "0001"


# ---------------------------------------------------------------------------
# compute_elo_history
# ---------------------------------------------------------------------------

def test_new_teams_start_at_the_starting_rating():
    matches = pd.DataFrame([match("2024-01-01", "E0", "A", "B", 1, 1, "D")])
    history = elo.compute_elo_history(matches, starting_rating=1500)
    row_a = history[(history["team"] == "A")].iloc[0]
    assert row_a["rating_before"] == 1500


def test_home_win_increases_home_rating_and_decreases_away_rating():
    matches = pd.DataFrame([match("2024-01-01", "E0", "A", "B", 2, 0, "H")])
    history = elo.compute_elo_history(matches)
    a = history[history["team"] == "A"].iloc[0]
    b = history[history["team"] == "B"].iloc[0]
    assert a["rating_after"] > a["rating_before"]
    assert b["rating_after"] < b["rating_before"]
    assert a["result"] == "W"
    assert b["result"] == "L"


def test_draw_between_evenly_matched_teams_barely_moves_ratings():
    """Home advantage means a draw between EQUAL ratings still favors the
    away team slightly (they overperformed expectation) -- but the shift
    should be small, not a full win/loss-sized swing."""
    matches = pd.DataFrame([match("2024-01-01", "E0", "A", "B", 1, 1, "D")])
    history = elo.compute_elo_history(matches, k=32, home_advantage=100)
    a = history[history["team"] == "A"].iloc[0]
    assert abs(a["rating_after"] - a["rating_before"]) < 16  # well under a full K


def test_home_advantage_shifts_expected_outcome():
    """With home advantage applied, a draw should COST the home team
    rating points (they were expected to do better than a draw at home),
    while benefiting the away team."""
    matches = pd.DataFrame([match("2024-01-01", "E0", "A", "B", 1, 1, "D")])
    history = elo.compute_elo_history(matches, home_advantage=100)
    a = history[history["team"] == "A"].iloc[0]
    b = history[history["team"] == "B"].iloc[0]
    assert a["rating_after"] < a["rating_before"]
    assert b["rating_after"] > b["rating_before"]


def test_higher_k_produces_larger_rating_swings():
    matches_low = pd.DataFrame([match("2024-01-01", "E0", "A", "B", 2, 0, "H")])
    matches_high = matches_low.copy()
    low = elo.compute_elo_history(matches_low, k=10)
    high = elo.compute_elo_history(matches_high, k=50)
    delta_low = low[low["team"] == "A"].iloc[0]["rating_after"] - 1500
    delta_high = high[high["team"] == "A"].iloc[0]["rating_after"] - 1500
    assert delta_high > delta_low


def test_ratings_carry_forward_across_multiple_matches():
    matches = pd.DataFrame([
        match("2024-01-01", "E0", "A", "B", 2, 0, "H"),
        match("2024-01-08", "E0", "A", "C", 2, 0, "H"),
    ])
    history = elo.compute_elo_history(matches)
    first_game_after = history[(history["team"] == "A") & (history["opponent"] == "B")].iloc[0]["rating_after"]
    second_game_before = history[(history["team"] == "A") & (history["opponent"] == "C")].iloc[0]["rating_before"]
    assert first_game_after == pytest.approx(second_game_before)


def test_input_row_order_does_not_affect_result():
    """Ratings must be computed strictly chronologically regardless of
    what order the rows arrive in -- never using future information."""
    matches_forward = pd.DataFrame([
        match("2024-01-01", "E0", "A", "B", 2, 0, "H"),
        match("2024-01-08", "E0", "B", "A", 0, 3, "A"),
    ])
    matches_reversed = matches_forward.iloc[::-1].reset_index(drop=True)

    history_forward = elo.compute_elo_history(matches_forward)
    history_reversed = elo.compute_elo_history(matches_reversed)

    final_forward = elo.current_ratings(history_forward)
    final_reversed = elo.current_ratings(history_reversed)
    pd.testing.assert_frame_equal(
        final_forward.reset_index(drop=True), final_reversed.reset_index(drop=True))


def test_leagues_are_rated_independently():
    """A team named 'A' in two different leagues must not share a rating
    pool -- there's no real link between them (see module docstring)."""
    matches = pd.DataFrame([
        match("2024-01-01", "E0", "A", "B", 3, 0, "H"),   # A dominates in E0
        match("2024-01-01", "D1", "A", "C", 0, 3, "A"),   # A gets thrashed in D1
    ])
    history = elo.compute_elo_history(matches)
    ratings = elo.current_ratings(history)
    e0_a = ratings[(ratings["league"] == "E0") & (ratings["team"] == "A")].iloc[0]["rating"]
    d1_a = ratings[(ratings["league"] == "D1") & (ratings["team"] == "A")].iloc[0]["rating"]
    assert e0_a > 1500
    assert d1_a < 1500  # a heavy home loss should pull rating below the start


def test_empty_matches_returns_empty_correctly_shaped_frame():
    result = elo.compute_elo_history(pd.DataFrame(columns=["date", "league", "home_team",
                                                             "away_team", "result"]))
    assert result.empty
    assert list(result.columns) == ["date", "league", "team", "opponent", "home",
                                     "rating_before", "rating_after", "result"]


# ---------------------------------------------------------------------------
# current_ratings / rating_trajectory
# ---------------------------------------------------------------------------

def test_current_ratings_takes_the_latest_row_per_team():
    matches = pd.DataFrame([
        match("2024-01-01", "E0", "A", "B", 1, 0, "H"),
        match("2024-01-08", "E0", "A", "B", 0, 1, "A"),
    ])
    history = elo.compute_elo_history(matches)
    ratings = elo.current_ratings(history)
    a_rating = ratings[ratings["team"] == "A"].iloc[0]["rating"]
    last_row = history[(history["team"] == "A")].sort_values("date").iloc[-1]
    assert a_rating == pytest.approx(last_row["rating_after"])


def test_current_ratings_sorted_highest_first_within_league():
    matches = pd.DataFrame([match("2024-01-01", "E0", "A", "B", 5, 0, "H")])
    ratings = elo.current_ratings(elo.compute_elo_history(matches))
    assert ratings.iloc[0]["team"] == "A"  # the winner, ranked above the loser


def test_rating_trajectory_returns_one_teams_history_in_order():
    matches = pd.DataFrame([
        match("2024-01-08", "E0", "B", "A", 0, 1, "A"),
        match("2024-01-01", "E0", "A", "B", 1, 0, "H"),
    ])
    history = elo.compute_elo_history(matches)
    traj = elo.rating_trajectory(history, "E0", "A")
    assert list(traj["date"]) == ["2024-01-01", "2024-01-08"]
