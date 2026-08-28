import pandas as pd

from core import ledger_data as ld


def test_load_live_odds_predictions_delegates_to_the_pure_match_function(monkeypatch):
    """core/ledger_data.py should be a thin cache wrapper only -- the
    real logic (and its tests) live in models/match.py, which stays
    UI-independent so a plain script can call it without Streamlit."""
    sentinel = pd.DataFrame([{"home_team": "Arsenal", "away_team": "Chelsea",
                              "home_win": 0.5, "draw": 0.3, "away_win": 0.2}])
    monkeypatch.setattr(ld.match, "load_live_odds_predictions", lambda: sentinel)
    ld.load_live_odds_predictions.clear()

    result = ld.load_live_odds_predictions()
    pd.testing.assert_frame_equal(result, sentinel)


def test_load_live_scoreboard_delegates_to_score_archived_forecasts(monkeypatch):
    sentinel = pd.DataFrame([{"date": "2024-08-17", "league": "E0", "home_team": "A",
                              "away_team": "B", "result": "H", "source": "dixon_coles",
                              "home_win": 0.5, "draw": 0.3, "away_win": 0.2}])
    monkeypatch.setattr(ld, "load_matches", lambda: "matches-sentinel")
    monkeypatch.setattr(ld.scoreboard, "score_archived_forecasts",
                        lambda matches: sentinel if matches == "matches-sentinel" else None)
    ld.load_live_scoreboard.clear()

    result = ld.load_live_scoreboard()
    pd.testing.assert_frame_equal(result, sentinel)
