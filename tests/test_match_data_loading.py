import pandas as pd
import pytest

from models import match


def _fake_base_matches():
    return pd.DataFrame([
        {"league": "E0", "season": "2425", "date": "2024-08-16",
         "home_team": "Man United", "away_team": "Fulham",
         "home_team_id": "fpl-team-89", "away_team_id": "fpl-team-228",
         "home_goals": 1, "away_goals": 0, "result": "H"},
        {"league": "SP1", "season": "2425", "date": "2024-08-17",
         "home_team": "Real Madrid", "away_team": "Barcelona",
         "home_team_id": None, "away_team_id": None,   # non-PL clubs: not seeded in the id registry
         "home_goals": 2, "away_goals": 1, "result": "H"},
        {"league": "E1", "season": "2425", "date": "2024-08-17",
         "home_team": "Leeds", "away_team": "Portsmouth",
         "home_team_id": "fpl-team-999", "away_team_id": "fpl-team-998",
         "home_goals": 2, "away_goals": 1, "result": "H"},
    ])


def test_load_matches_with_xg_joins_on_resolved_team_ids_not_raw_names(monkeypatch):
    """football_data calls it 'Man United', Understat calls the same
    club 'Manchester United' -- those don't even clear the project's own
    fuzzy-match threshold (0.741 vs 0.85), so the join must go through
    the canonical team_id both collectors already resolved, not the raw
    (or even normalized) name strings."""
    monkeypatch.setattr(match.elo, "load_all_matches", lambda leagues=None: _fake_base_matches())
    xg = pd.DataFrame([{
        "league": "E0", "season": "2425", "date": "2024-08-16",
        "home_team_raw": "Manchester United", "away_team_raw": "Fulham",
        "home_team_id": "fpl-team-89", "away_team_id": "fpl-team-228",
        "home_xg": 2.04, "away_xg": 0.42,
    }])
    monkeypatch.setattr(match, "_load_xg", lambda leagues=None: xg)

    merged = match.load_matches_with_xg()
    row = merged[merged["league"] == "E0"].iloc[0]
    assert row["home_xg"] == pytest.approx(2.04)
    assert row["away_xg"] == pytest.approx(0.42)


def test_load_matches_with_xg_leaves_null_for_a_league_understat_does_not_cover(monkeypatch):
    """E1 (Championship) has no Understat coverage at all -- must stay
    null, not error or get matched to something wrong, even though both
    sides happen to have resolved team ids here."""
    monkeypatch.setattr(match.elo, "load_all_matches", lambda leagues=None: _fake_base_matches())
    monkeypatch.setattr(match, "_load_xg", lambda leagues=None: pd.DataFrame([{
        "league": "E0", "season": "2425", "date": "2024-08-16",
        "home_team_raw": "Manchester United", "away_team_raw": "Fulham",
        "home_team_id": "fpl-team-89", "away_team_id": "fpl-team-228",
        "home_xg": 2.04, "away_xg": 0.42,
    }]))

    merged = match.load_matches_with_xg()
    e1_row = merged[merged["league"] == "E1"].iloc[0]
    assert pd.isna(e1_row["home_xg"])
    assert pd.isna(e1_row["away_xg"])


def test_load_matches_with_xg_does_not_cross_match_unresolved_ids(monkeypatch):
    """Real Madrid vs Barcelona (both unresolved team ids, since the
    registry is FPL/PL-only) must NOT pick up some other unrelated
    unresolved match's xG just because both sides are null -- pandas
    treats NaN as equal to NaN in a merge key by default, which is
    exactly the trap this test guards against."""
    monkeypatch.setattr(match.elo, "load_all_matches", lambda leagues=None: _fake_base_matches())
    xg = pd.DataFrame([{
        "league": "SP1", "season": "2425", "date": "2024-08-17",
        "home_team_raw": "Sevilla", "away_team_raw": "Villarreal",
        "home_team_id": None, "away_team_id": None,   # a different, unrelated match, also unresolved
        "home_xg": 9.99, "away_xg": 9.99,
    }])
    monkeypatch.setattr(match, "_load_xg", lambda leagues=None: xg)

    merged = match.load_matches_with_xg()
    real_madrid_row = merged[merged["league"] == "SP1"].iloc[0]
    assert pd.isna(real_madrid_row["home_xg"])
    assert pd.isna(real_madrid_row["away_xg"])


def test_load_matches_with_xg_carries_through_the_extra_understat_fields(monkeypatch):
    monkeypatch.setattr(match.elo, "load_all_matches", lambda leagues=None: _fake_base_matches())
    xg = pd.DataFrame([{
        "league": "E0", "season": "2425", "date": "2024-08-16",
        "home_team_id": "fpl-team-89", "away_team_id": "fpl-team-228",
        "home_xg": 2.04, "away_xg": 0.42, "home_npxg": 1.8, "away_npxg": 0.42,
        "home_npxga": 0.42, "away_npxga": 1.8, "home_ppda": 8.5, "away_ppda": 12.1,
        "home_deep": 9, "away_deep": 2, "home_xpts": 2.3, "away_xpts": 0.4,
        "understat_forecast_home_win": 0.6, "understat_forecast_draw": 0.25,
        "understat_forecast_away_win": 0.15,
    }])
    monkeypatch.setattr(match, "_load_xg", lambda leagues=None: xg)

    merged = match.load_matches_with_xg()
    row = merged[merged["league"] == "E0"].iloc[0]
    assert row["home_npxg"] == pytest.approx(1.8)
    assert row["home_ppda"] == pytest.approx(8.5)
    assert row["home_deep"] == 9
    assert row["home_xpts"] == pytest.approx(2.3)
    assert row["understat_forecast_home_win"] == pytest.approx(0.6)


def test_load_matches_with_xg_returns_all_null_when_no_xg_data_exists_at_all(monkeypatch):
    monkeypatch.setattr(match.elo, "load_all_matches", lambda leagues=None: _fake_base_matches())
    monkeypatch.setattr(match, "_load_xg", lambda leagues=None: pd.DataFrame(
        columns=["league", "season", "date", "home_team_raw", "away_team_raw",
                "home_team_id", "away_team_id", "home_xg", "away_xg"]))

    merged = match.load_matches_with_xg()
    assert merged["home_xg"].isna().all()
    assert len(merged) == len(_fake_base_matches())
