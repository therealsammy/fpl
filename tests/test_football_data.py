import datetime as dt

import pandas as pd
import pytest
import requests

from collectors import football_data as fd
from core import ids


# ---------------------------------------------------------------------------
# fetch_season_csv content-type check (regression: a not-yet-published
# season returns HTTP 300 with an HTML page, not a 404 -- verified live)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code, content_type, text):
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.text = text

    def raise_for_status(self):
        pass  # 300 doesn't raise in real requests either -- that's the bug this guards


def test_fetch_season_csv_treats_non_csv_content_type_as_unavailable(monkeypatch):
    fake = _FakeResponse(300, "text/html; charset=iso-8859-1", "<html>Multiple Choices</html>")
    monkeypatch.setattr(fd.SESSION, "get", lambda *a, **k: fake)

    result = fd.fetch_season_csv("E0", 2026)
    assert result is None


def test_fetch_season_csv_accepts_real_csv_content_type(monkeypatch):
    fake = _FakeResponse(200, "text/csv", "Date,HomeTeam\n16/08/2024,Arsenal\n")
    monkeypatch.setattr(fd.SESSION, "get", lambda *a, **k: fake)

    result = fd.fetch_season_csv("E0", 2024)
    assert result == "Date,HomeTeam\n16/08/2024,Arsenal\n"


def test_fetch_season_csv_still_treats_404_as_unavailable(monkeypatch):
    fake = _FakeResponse(404, "text/html", "not found")
    monkeypatch.setattr(fd.SESSION, "get", lambda *a, **k: fake)

    assert fd.fetch_season_csv("ZZ", 1995) is None


# ---------------------------------------------------------------------------
# Season codes
# ---------------------------------------------------------------------------

def test_season_code_formats():
    assert fd.season_code(1993) == "9394"
    assert fd.season_code(1999) == "9900"
    assert fd.season_code(2024) == "2425"


def test_current_season_start_year_before_july_is_previous_calendar_year():
    assert fd.current_season_start_year(dt.date(2026, 3, 15)) == 2025


def test_current_season_start_year_july_onward_is_this_calendar_year():
    assert fd.current_season_start_year(dt.date(2026, 8, 24)) == 2026


def test_season_start_years_spans_the_full_range():
    years = fd.season_start_years(earliest=2020, today=dt.date(2023, 8, 1))
    assert years == [2020, 2021, 2022, 2023]


# ---------------------------------------------------------------------------
# Market consensus -- synthetic CSVs modeled on real eras (verified live
# against football-data.co.uk before writing this collector)
# ---------------------------------------------------------------------------

def test_compute_1x2_consensus_prefers_avg_columns_modern_era():
    df = pd.DataFrame({
        "AvgH": [1.9], "AvgD": [3.5], "AvgA": [4.2],
        "B365H": [1.85], "B365D": [3.4], "B365A": [4.5],  # should be ignored -- Avg wins
    })
    result = fd.compute_1x2_consensus(df)
    assert result.iloc[0]["home_price"] == 1.9


def test_compute_1x2_consensus_falls_back_to_bbav_era():
    df = pd.DataFrame({"BbAvH": [2.0], "BbAvD": [3.3], "BbAvA": [3.8]})
    result = fd.compute_1x2_consensus(df)
    assert result.iloc[0]["home_price"] == 2.0


def test_compute_1x2_consensus_falls_back_to_median_of_individual_bookmakers():
    """The ~2000/01 era: no consensus column at all, just a handful of
    individual bookmakers."""
    df = pd.DataFrame({
        "GBH": [2.0], "GBD": [3.2], "GBA": [3.5],
        "IWH": [2.1], "IWD": [3.3], "IWA": [3.4],
        "WHH": [1.9], "WHD": [3.1], "WHA": [3.6],
    })
    result = fd.compute_1x2_consensus(df)
    assert result.iloc[0]["home_price"] == pytest.approx(2.0)  # median of 1.9/2.0/2.1


def test_compute_1x2_consensus_none_when_no_odds_at_all():
    """The 1993/94 era: results only."""
    df = pd.DataFrame({"HomeTeam": ["Arsenal"], "AwayTeam": ["Coventry"]})
    assert fd.compute_1x2_consensus(df) is None


def test_compute_ou25_consensus_prefers_avg_then_bbav_then_individual():
    modern = pd.DataFrame({"Avg>2.5": [2.0], "Avg<2.5": [1.9]})
    assert fd.compute_ou25_consensus(modern).iloc[0]["over_price"] == 2.0

    bb_era = pd.DataFrame({"BbAv>2.5": [1.95], "BbAv<2.5": [1.95]})
    assert fd.compute_ou25_consensus(bb_era).iloc[0]["over_price"] == 1.95

    individual = pd.DataFrame({"B365>2.5": [2.1], "B365<2.5": [1.8]})
    assert fd.compute_ou25_consensus(individual).iloc[0]["over_price"] == 2.1


def test_compute_ou25_consensus_none_when_unavailable():
    df = pd.DataFrame({"AvgH": [1.9], "AvgD": [3.5], "AvgA": [4.2]})  # 1X2 only, no O/U
    assert fd.compute_ou25_consensus(df) is None


# ---------------------------------------------------------------------------
# Devig (thin wrapper -- confirms correct plumbing into collectors.fpl_odds)
# ---------------------------------------------------------------------------

def test_devig_1x2_probabilities_sum_to_one():
    consensus = pd.DataFrame({"home_price": [1.9], "draw_price": [3.5], "away_price": [4.2]})
    result = fd.devig_1x2(consensus).iloc[0]
    assert (result["prob_home_proportional"] + result["prob_draw_proportional"]
            + result["prob_away_proportional"]) == pytest.approx(1.0, abs=1e-6)
    assert (result["prob_home_shin"] + result["prob_draw_shin"]
            + result["prob_away_shin"]) == pytest.approx(1.0, abs=1e-6)


def test_devig_1x2_null_when_a_price_is_missing():
    consensus = pd.DataFrame({"home_price": [1.9], "draw_price": [None], "away_price": [4.2]})
    result = fd.devig_1x2(consensus).iloc[0]
    assert result["prob_home_proportional"] is None


def test_devig_ou25_probabilities_sum_to_one():
    consensus = pd.DataFrame({"over_price": [2.0], "under_price": [1.9]})
    result = fd.devig_ou25(consensus).iloc[0]
    assert (result["prob_over25_proportional"] + result["prob_under25_proportional"]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# normalize_season (end to end on synthetic era-shaped CSVs)
# ---------------------------------------------------------------------------

MODERN_CSV = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5
16/08/2024,Man United,Fulham,1,0,H,1.9,3.5,4.2,2.0,1.9
"""

NO_ODDS_CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,,,,
E0,14/08/93,Arsenal,Coventry,0,3,A,,,,
E0,14/08/93,Aston Villa,QPR,4,1,H,,,,
"""

# Modeled on the real 2003/04 Premier League file: header has 9 fields,
# most rows match, but one row has extra trailing empty commas (verified
# live -- pandas' C parser raises ParserError on this without a fix).
RAGGED_ROWS_CSV = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA
16/08/2003,Arsenal,Everton,2,1,H,1.4,3.8,8
03/04/2004,Tottenham,Chelsea,0,1,A,4,3.25,1.9,,,,,,,,
"""

SCHEMA_DRIFT_CSV = """Date,HomeTeam,FTHG,FTAG,FTR
16/08/2024,Man United,1,0,H
"""


def test_normalize_season_modern_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")

    result = fd.normalize_season(MODERN_CSV, "E0", 2024)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["league"] == "E0"
    assert row["season"] == "2425"
    assert row["date"] == "2024-08-16"
    assert row["home_team_raw"] == "Man United"
    assert row["home_goals"] == 1
    assert row["result"] == "H"
    assert row["prob_home_proportional"] is not None
    assert row["prob_over25_proportional"] is not None


def test_normalize_season_handles_no_odds_era(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")

    result = fd.normalize_season(NO_ODDS_CSV, "E0", 1993)

    assert len(result) == 2  # trailing all-empty columns dropped, real rows kept
    assert result.iloc[0]["result"] == "A"
    assert result.iloc[0]["prob_home_proportional"] is None  # honest gap, not fabricated
    assert result.iloc[0]["prob_over25_proportional"] is None


def test_normalize_season_raises_on_missing_required_column():
    with pytest.raises(KeyError, match="AwayTeam"):
        fd.normalize_season(SCHEMA_DRIFT_CSV, "E0", 2024)


def test_normalize_season_tolerates_ragged_rows_with_extra_trailing_commas(tmp_path, monkeypatch):
    """Regression: a real 2003/04 Premier League row has extra trailing
    empty fields beyond the header's width, which crashed pandas' C
    parser outright before _fix_ragged_rows existed."""
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")

    result = fd.normalize_season(RAGGED_ROWS_CSV, "E0", 2003)

    assert len(result) == 2
    assert result.iloc[1]["home_team_raw"] == "Tottenham"
    assert result.iloc[1]["result"] == "A"


def test_normalize_season_resolves_known_team_and_logs_unknown_team(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")
    ids.register_fpl_teams([{"id": 1, "name": "Man United"}])  # Fulham stays unregistered

    result = fd.normalize_season(MODERN_CSV, "E0", 2024)

    assert result.iloc[0]["home_team_id"] == "fpl-team-1"
    assert result.iloc[0]["away_team_id"] is None  # Fulham: not yet in the registry -- honest gap
    unresolved = pd.read_csv(tmp_path / "unresolved.csv")
    assert "Fulham" in unresolved["name"].values


# ---------------------------------------------------------------------------
# Orchestration: fetch-once-for-history, always-refresh-current
# ---------------------------------------------------------------------------

def test_collect_league_season_skips_a_cached_historical_season(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "OUTPUT_ROOT", tmp_path)
    out_path = tmp_path / "E0" / "2324.csv"
    out_path.parent.mkdir(parents=True)
    out_path.write_text("league,season\nE0,2324\n")

    calls = []
    monkeypatch.setattr(fd, "fetch_season_csv", lambda *a: calls.append(a) or "should not be used")

    status = fd.collect_league_season("E0", 2023, current_start_year=2024)

    assert status == "skipped_cached"
    assert calls == []  # never even tried to fetch


def test_collect_league_season_always_refetches_the_current_season(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(fd.ids, "PLAYERS_PATH", tmp_path / "players.csv")
    monkeypatch.setattr(fd.ids, "TEAMS_PATH", tmp_path / "teams.csv")
    monkeypatch.setattr(fd.ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")
    out_path = tmp_path / "E0" / "2425.csv"
    out_path.parent.mkdir(parents=True)
    out_path.write_text("stale,data\n1,2\n")

    monkeypatch.setattr(fd, "fetch_season_csv", lambda *a: MODERN_CSV)

    status = fd.collect_league_season("E0", 2024, current_start_year=2024)

    assert status == "fetched"
    refreshed = pd.read_csv(out_path)
    assert "home_team_raw" in refreshed.columns  # really overwritten, not left stale


def test_collect_league_season_missing_combo_returns_skipped_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(fd, "fetch_season_csv", lambda *a: None)  # simulates a 404

    status = fd.collect_league_season("XX", 1995, current_start_year=2024)
    assert status == "skipped_missing"
