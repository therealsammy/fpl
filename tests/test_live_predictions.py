import numpy as np
import pandas as pd
import pytest

from core import archive
from models import live_predictions as lp, match


def _league_history(rng, n_matches=200):
    """A synthetic but genuinely fittable league: 4 teams playing lots of
    goals-based matches, plus the npxg/ppda/deep_form columns
    compute_rolling_form needs for the GLM branch to have any coverage
    at all."""
    rows = []
    date = pd.Timestamp("2024-08-01")
    teams = ["A", "B", "C", "D"]
    for _ in range(n_matches):
        home, away = rng.choice(teams, size=2, replace=False)
        rows.append({
            "league": "E0", "season": "2425", "home_team": home, "away_team": away,
            "home_goals": rng.poisson(1.3), "away_goals": rng.poisson(1.1),
            "date": date.strftime("%Y-%m-%d"),
            "home_npxg": rng.gamma(4, 0.3), "away_npxg": rng.gamma(4, 0.3),
            "home_ppda": rng.uniform(8, 15), "away_ppda": rng.uniform(8, 15),
            "home_deep": rng.uniform(3, 10), "away_deep": rng.uniform(3, 10),
        })
        date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def _fixtures(rows):
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team"])


# ---------------------------------------------------------------------------
# predict_upcoming
# ---------------------------------------------------------------------------

def test_predict_upcoming_covers_every_source_and_outcome(monkeypatch):
    rng = np.random.default_rng(1)
    matches = _league_history(rng)
    fixtures = _fixtures([{"date": pd.Timestamp("2025-06-01"), "home_team": "A", "away_team": "B"}])
    monkeypatch.setattr(match, "upcoming_fixtures", lambda league, **kw: fixtures)
    odds = pd.DataFrame([{"home_team": "A", "away_team": "B",
                          "home_win": 0.5, "draw": 0.3, "away_win": 0.2}])

    result = lp.predict_upcoming("E0", matches, odds)

    assert set(result["source"]) == {"dixon_coles", "dixon_coles_glm", "closing_line"}
    assert set(result["metric"]) == {"home_win", "draw", "away_win"}
    for source in result["source"].unique():
        probs = result[result["source"] == source].set_index("metric")["value"]
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_predict_upcoming_market_row_matches_the_odds_input_exactly(monkeypatch):
    rng = np.random.default_rng(2)
    matches = _league_history(rng)
    fixtures = _fixtures([{"date": pd.Timestamp("2025-06-01"), "home_team": "A", "away_team": "B"}])
    monkeypatch.setattr(match, "upcoming_fixtures", lambda league, **kw: fixtures)
    odds = pd.DataFrame([{"home_team": "A", "away_team": "B",
                          "home_win": 0.5, "draw": 0.3, "away_win": 0.2}])

    result = lp.predict_upcoming("E0", matches, odds)
    market = result[result["source"] == "closing_line"].set_index("metric")["value"]
    assert market["home_win"] == pytest.approx(0.5)
    assert market["draw"] == pytest.approx(0.3)
    assert market["away_win"] == pytest.approx(0.2)


def test_predict_upcoming_omits_market_when_odds_do_not_cover_the_fixture(monkeypatch):
    rng = np.random.default_rng(3)
    matches = _league_history(rng)
    fixtures = _fixtures([{"date": pd.Timestamp("2025-06-01"), "home_team": "A", "away_team": "B"}])
    monkeypatch.setattr(match, "upcoming_fixtures", lambda league, **kw: fixtures)
    empty_odds = pd.DataFrame(columns=["home_team", "away_team", "home_win", "draw", "away_win"])

    result = lp.predict_upcoming("E0", matches, empty_odds)
    assert "closing_line" not in set(result["source"])
    assert "dixon_coles" in set(result["source"])


def test_predict_upcoming_empty_when_no_fixtures_scheduled(monkeypatch):
    rng = np.random.default_rng(4)
    matches = _league_history(rng)
    monkeypatch.setattr(match, "upcoming_fixtures", lambda league, **kw: _fixtures([]))
    result = lp.predict_upcoming("E0", matches, pd.DataFrame())
    assert result.empty


def test_predict_upcoming_empty_when_league_has_no_fittable_history(monkeypatch):
    fixtures = _fixtures([{"date": pd.Timestamp("2025-06-01"), "home_team": "A", "away_team": "B"}])
    monkeypatch.setattr(match, "upcoming_fixtures", lambda league, **kw: fixtures)
    empty_matches = pd.DataFrame(columns=["league", "home_team", "away_team", "home_goals",
                                          "away_goals", "date"])
    result = lp.predict_upcoming("E0", empty_matches, pd.DataFrame())
    assert result.empty


def test_fixture_entity_id_is_stable_across_a_few_days_reschedule():
    id_a = lp._fixture_entity_id("E0", "A", "B", pd.Timestamp("2025-08-30"))
    id_b = lp._fixture_entity_id("E0", "A", "B", pd.Timestamp("2025-09-02"))  # same season
    assert id_a == id_b == "E0|2526|A|B"


# ---------------------------------------------------------------------------
# archive_all_leagues -- the per-source-across-leagues write correctness
# ---------------------------------------------------------------------------

def test_archive_all_leagues_does_not_let_one_league_erase_anothers_rows(tmp_path, monkeypatch):
    """Real bug this guards against: write_forecast() overwrites its
    whole (as_of, source) file on every call. Writing per-league would
    have a second league silently erase the first league's rows for the
    same source and day -- rows from every league sharing a source must
    be combined before write_forecast is ever called."""
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)

    rng = np.random.default_rng(5)
    e0 = _league_history(rng, n_matches=100)
    e1 = e0.copy()
    e1["league"] = "E1"
    all_matches = pd.concat([e0, e1], ignore_index=True)
    monkeypatch.setattr(match, "load_matches_with_xg", lambda: all_matches)
    monkeypatch.setattr(match, "load_live_odds_predictions",
                        lambda: pd.DataFrame(columns=["home_team", "away_team", "home_win", "draw", "away_win"]))

    def fake_upcoming(league, **kw):
        return _fixtures([{"date": pd.Timestamp("2025-06-01"), "home_team": "A", "away_team": "B"}])
    monkeypatch.setattr(match, "upcoming_fixtures", fake_upcoming)

    written = lp.archive_all_leagues(as_of="2025-05-01")
    assert written["dixon_coles"] == 6   # 3 outcomes x 2 leagues

    on_disk = archive.read_forecasts(source="dixon_coles", as_of="2025-05-01")
    assert set(on_disk["entity_id"].str.split("|").str[0]) == {"E0", "E1"}


def test_archive_all_leagues_empty_when_no_matches_at_all(monkeypatch):
    monkeypatch.setattr(match, "load_matches_with_xg", lambda: pd.DataFrame())
    assert lp.archive_all_leagues() == {}
