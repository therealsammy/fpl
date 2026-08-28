import datetime as dt

import pandas as pd
import pytest
import requests

from collectors import understat as us
from core import ids


def _match(home, away, is_result=True, home_goals="1", away_goals="0",
          home_xg="1.5", away_xg="0.8", date="2024-08-16 19:00:00", forecast=True,
          home_id="1", away_id="2"):
    m = {
        "id": "1", "isResult": is_result,
        "h": {"id": home_id, "title": home, "short_title": home[:3].upper()},
        "a": {"id": away_id, "title": away, "short_title": away[:3].upper()},
        "goals": {"h": home_goals, "a": away_goals},
        "xG": {"h": home_xg, "a": away_xg},
        "datetime": date,
    }
    if forecast:
        m["forecast"] = {"w": "0.55", "d": "0.25", "l": "0.20"}
    return m


def _history_row(date, h_a, npxg=1.7, npxga=0.9, ppda_att=200, ppda_def=20, deep=8, xpts=1.8):
    return {"date": date, "h_a": h_a, "npxG": npxg, "npxGA": npxga,
           "ppda": {"att": ppda_att, "def": ppda_def}, "deep": deep, "xpts": xpts}


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, raise_exc=None):
        self.status_code = status_code
        self._json = json_data
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    def json(self):
        return self._json


# ---------------------------------------------------------------------------
# fetch_league_season
# ---------------------------------------------------------------------------

def test_fetch_league_season_returns_none_for_empty_coverage(monkeypatch):
    """Understat returns HTTP 200 with an empty dates list for a season
    outside its coverage (e.g. before 2014/15) -- verified live, not a 404."""
    fake = _FakeResponse(200, {"teams": {}, "players": [], "dates": []})
    monkeypatch.setattr(us.SESSION, "get", lambda *a, **k: fake)

    assert us.fetch_league_season("EPL", 2013) is None


def test_fetch_league_season_returns_data_when_present(monkeypatch):
    payload = {"teams": {}, "players": [], "dates": [_match("Arsenal", "Fulham")]}
    fake = _FakeResponse(200, payload)
    monkeypatch.setattr(us.SESSION, "get", lambda *a, **k: fake)

    result = us.fetch_league_season("EPL", 2024)
    assert result == payload


def test_fetch_league_season_retries_then_gives_up_on_network_error(monkeypatch):
    calls = {"n": 0}

    def flaky_get(*a, **k):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(us.SESSION, "get", flaky_get)
    monkeypatch.setattr(us.time, "sleep", lambda s: None)

    result = us.fetch_league_season("EPL", 2024)
    assert result is None
    assert calls["n"] == us.RETRIES


def test_fetch_league_season_treats_malformed_json_as_schema_drift(monkeypatch):
    fake = _FakeResponse(200, None)

    class BadJson(_FakeResponse):
        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(us.SESSION, "get", lambda *a, **k: BadJson())
    assert us.fetch_league_season("EPL", 2024) is None


# ---------------------------------------------------------------------------
# normalize_season
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_id_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")


def test_normalize_season_maps_a_played_match():
    data = {"dates": [_match("Arsenal", "Fulham", home_goals="2", away_goals="0",
                             home_xg="1.8", away_xg="0.6")]}
    df = us.normalize_season(data, "E0", 2024)

    row = df.iloc[0]
    assert row["league"] == "E0"
    assert row["season"] == "2425"
    assert row["date"] == "2024-08-16"
    assert row["home_team_raw"] == "Arsenal"
    assert row["away_team_raw"] == "Fulham"
    assert row["is_result"] == True
    assert row["home_goals"] == 2
    assert row["away_goals"] == 0
    assert row["home_xg"] == pytest.approx(1.8)
    assert row["away_xg"] == pytest.approx(0.6)
    assert row["understat_forecast_home_win"] == pytest.approx(0.55)


def test_normalize_season_keeps_nulls_for_an_unplayed_fixture():
    """The live-fixture-list bonus: an unplayed match has null goals/xG
    and no forecast key at all -- must not crash, must not fabricate 0s."""
    data = {"dates": [_match("Crystal Palace", "Man City", is_result=False,
                             home_goals=None, away_goals=None,
                             home_xg=None, away_xg=None, forecast=False)]}
    df = us.normalize_season(data, "E0", 2026)

    row = df.iloc[0]
    assert row["is_result"] == False
    assert pd.isna(row["home_goals"])
    assert pd.isna(row["away_goals"])
    assert pd.isna(row["home_xg"])
    assert pd.isna(row["away_xg"])
    assert pd.isna(row["understat_forecast_home_win"])


def test_normalize_season_joins_team_history_for_npxg_ppda_deep_xpts():
    date = "2024-08-16 19:00:00"
    data = {
        "dates": [_match("Arsenal", "Fulham", date=date, home_id="10", away_id="20")],
        "teams": {
            "10": {"id": "10", "title": "Arsenal",
                  "history": [_history_row(date, "h", npxg=2.1, npxga=0.5,
                                           ppda_att=300, ppda_def=15, deep=12, xpts=2.4)]},
            "20": {"id": "20", "title": "Fulham",
                  "history": [_history_row(date, "a", npxg=0.6, npxga=2.0,
                                           ppda_att=100, ppda_def=25, deep=3, xpts=0.5)]},
        },
    }
    df = us.normalize_season(data, "E0", 2024)
    row = df.iloc[0]

    assert row["home_npxg"] == pytest.approx(2.1)
    assert row["away_npxg"] == pytest.approx(0.6)
    assert row["home_npxga"] == pytest.approx(0.5)
    assert row["away_npxga"] == pytest.approx(2.0)
    assert row["home_ppda"] == pytest.approx(300 / 15)
    assert row["away_ppda"] == pytest.approx(100 / 25)
    assert row["home_deep"] == 12
    assert row["away_deep"] == 3
    assert row["home_xpts"] == pytest.approx(2.4)
    assert row["away_xpts"] == pytest.approx(0.5)


def test_normalize_season_history_fields_null_when_no_history_row_exists():
    """An unplayed fixture (or any match whose date doesn't line up with
    a team-history row) must leave these fields null, not crash on a
    missing lookup or fabricate a zero."""
    data = {"dates": [_match("Arsenal", "Fulham", is_result=False, home_goals=None,
                             away_goals=None, home_xg=None, away_xg=None, forecast=False)],
           "teams": {}}
    df = us.normalize_season(data, "E0", 2024)
    row = df.iloc[0]
    for col in ["home_npxg", "away_npxg", "home_npxga", "away_npxga",
               "home_ppda", "away_ppda", "home_deep", "away_deep", "home_xpts", "away_xpts"]:
        assert pd.isna(row[col])


def test_ppda_ratio_hand_computed():
    assert us._ppda_ratio({"att": 200, "def": 20}) == pytest.approx(10.0)


def test_ppda_ratio_none_when_no_defensive_actions_recorded():
    assert us._ppda_ratio({"att": 50, "def": 0}) is None


def test_ppda_ratio_none_when_missing_entirely():
    assert us._ppda_ratio(None) is None


# ---------------------------------------------------------------------------
# normalize_players
# ---------------------------------------------------------------------------

def _player(name="Mohamed Salah", team="Liverpool", xg="27.7", xa="15.9"):
    return {"id": "1", "player_name": name, "team_title": team, "time": "3392",
           "games": "38", "goals": "29", "xG": xg, "npg": "20", "npxG": "20.85",
           "assists": "18", "xA": xa, "xGChain": "48.5", "xGBuildup": "16.2"}


def test_normalize_players_maps_season_totals():
    data = {"players": [_player()]}
    df = us.normalize_players(data, "E0", 2024)
    row = df.iloc[0]

    assert row["league"] == "E0"
    assert row["season"] == "2425"
    assert row["player_name"] == "Mohamed Salah"
    assert row["team"] == "Liverpool"
    assert row["minutes"] == 3392
    assert row["goals"] == 29
    assert row["xg"] == pytest.approx(27.7)
    assert row["xa"] == pytest.approx(15.9)


def test_normalize_players_on_empty_players_list_returns_empty_frame():
    df = us.normalize_players({"players": []}, "E0", 2024)
    assert df.empty


def test_normalize_season_resolves_known_team_names():
    """Arsenal is seeded via FPL in the id registry fixture below -- a
    real resolution should succeed and return that canonical_id."""
    teams_df = pd.DataFrame([{
        "canonical_id": "fpl-team-1", "display_name": "Arsenal", "fpl_id": 1,
        "understat_id": pd.NA, "statsbomb_id": pd.NA, "football_data_name": pd.NA,
        "understat_name": pd.NA, "country": "England", "tier": 1,
        "confidence": 1.0, "method": "exact", "verified_at": pd.NA,
    }])
    ids._save(teams_df, ids.TEAMS_PATH, ids.TEAM_SCHEMA)

    data = {"dates": [_match("Arsenal", "Fulham")]}
    df = us.normalize_season(data, "E0", 2024)

    assert df.iloc[0]["home_team_id"] == "fpl-team-1"
    assert df.iloc[0]["away_team_id"] is None   # Fulham isn't seeded in this test registry


# ---------------------------------------------------------------------------
# collect_league_season caching
# ---------------------------------------------------------------------------

def test_collect_league_season_skips_a_cached_completed_season(tmp_path, monkeypatch):
    monkeypatch.setattr(us, "OUTPUT_ROOT", tmp_path)
    out_dir = tmp_path / "E0"
    out_dir.mkdir()
    (out_dir / "2324.csv").write_text("league,season\nE0,2324\n")

    called = {"n": 0}
    monkeypatch.setattr(us, "fetch_league_season", lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    status = us.collect_league_season("E0", "EPL", 2023, current_start_year=2026)
    assert status == "skipped_cached"
    assert called["n"] == 0


def test_collect_league_season_always_refetches_the_current_season(tmp_path, monkeypatch):
    monkeypatch.setattr(us, "OUTPUT_ROOT", tmp_path / "matches")
    monkeypatch.setattr(us, "PLAYERS_OUTPUT_ROOT", tmp_path / "players")
    out_dir = tmp_path / "matches" / "E0"
    out_dir.mkdir(parents=True)
    (out_dir / "2627.csv").write_text("league,season\nE0,2627\n")

    monkeypatch.setattr(us, "fetch_league_season",
                        lambda *a, **k: {"dates": [_match("Arsenal", "Fulham")], "players": []})

    status = us.collect_league_season("E0", "EPL", 2026, current_start_year=2026)
    assert status == "fetched"
    assert (tmp_path / "players" / "E0" / "2627.csv").exists()


def test_collect_league_season_reports_missing_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(us, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(us, "fetch_league_season", lambda *a, **k: None)

    status = us.collect_league_season("E0", "EPL", 2010, current_start_year=2026)
    assert status == "skipped_missing"


# ---------------------------------------------------------------------------
# season_start_years -- Understat's own earliest coverage, not football_data's
# ---------------------------------------------------------------------------

def test_season_start_years_begins_at_understat_earliest_season():
    years = us.season_start_years(today=dt.date(2026, 3, 1))
    assert years[0] == 2014
    assert years[-1] == 2025   # before-July cutoff -> still counts as the 2025/26 season
