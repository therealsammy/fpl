import numpy as np
import pandas as pd
import pytest

from core import archive
from validation import scoreboard as sb
from models import elo, match


def _row(home_win, draw, away_win, result):
    return {"home_win": home_win, "draw": draw, "away_win": away_win, "result": result}


# ---------------------------------------------------------------------------
# log_loss
# ---------------------------------------------------------------------------

def test_log_loss_hand_computed_single_row():
    scored = pd.DataFrame([_row(0.8, 0.15, 0.05, "H")])
    assert sb.log_loss(scored) == pytest.approx(-np.log(0.8))


def test_log_loss_zero_for_a_perfect_certain_forecaster():
    scored = pd.DataFrame([_row(1.0, 0.0, 0.0, "H"), _row(0.0, 0.0, 1.0, "A")])
    assert sb.log_loss(scored) == pytest.approx(0.0, abs=1e-9)


def test_log_loss_does_not_explode_to_infinity_on_a_zero_probability():
    """A model that confidently rules out what actually happened must
    be penalized hard, not crash the whole scoreboard with -inf/NaN."""
    scored = pd.DataFrame([_row(0.0, 0.0, 1.0, "H")])
    result = sb.log_loss(scored)
    assert np.isfinite(result)
    assert result > 10   # a harsh but finite penalty


def test_log_loss_on_empty_input_is_nan_not_a_crash():
    assert np.isnan(sb.log_loss(pd.DataFrame(columns=["home_win", "draw", "away_win", "result"])))


# ---------------------------------------------------------------------------
# ranked_probability_score -- hand-computed against the ordered scale
# ---------------------------------------------------------------------------

def test_rps_is_zero_for_a_perfect_certain_forecast():
    scored = pd.DataFrame([_row(1.0, 0.0, 0.0, "H")])
    assert sb.ranked_probability_score(scored) == pytest.approx(0.0, abs=1e-9)


def test_rps_is_worse_for_the_opposite_wrong_call_than_the_adjacent_one():
    """Predicting a draw when the away side won is a near-miss on the
    ordered win/draw/loss scale; predicting a home win for the same
    actual result is the furthest-possible miss -- RPS must reflect
    that, unlike log loss, which treats every wrong call on the actual
    outcome the same way regardless of the other two probabilities."""
    near_miss = pd.DataFrame([_row(0.0, 1.0, 0.0, "A")])
    far_miss = pd.DataFrame([_row(1.0, 0.0, 0.0, "A")])
    assert sb.ranked_probability_score(near_miss) == pytest.approx(0.5)
    assert sb.ranked_probability_score(far_miss) == pytest.approx(1.0)
    assert sb.ranked_probability_score(near_miss) < sb.ranked_probability_score(far_miss)


def test_rps_on_empty_input_is_nan_not_a_crash():
    assert np.isnan(sb.ranked_probability_score(pd.DataFrame(columns=["home_win", "draw", "away_win", "result"])))


# ---------------------------------------------------------------------------
# calibration_curve
# ---------------------------------------------------------------------------

def test_calibration_curve_tracks_a_well_calibrated_source():
    rng = np.random.default_rng(1)
    n = 2000
    p = rng.uniform(0.2, 0.8, n)
    actual_home = rng.binomial(1, p)
    rows = pd.DataFrame({
        "home_win": p, "draw": (1 - p) / 2, "away_win": (1 - p) / 2,
        "result": np.where(actual_home, "H", "A"),
    })
    curve = sb.calibration_curve(rows, outcome="home_win", n_bins=5)
    assert not curve.empty
    assert (curve["predicted"] - curve["actual"]).abs().max() < 0.1


def test_calibration_curve_drops_bins_with_too_few_matches():
    rows = pd.DataFrame([_row(0.95, 0.03, 0.02, "H")] * 3)   # a tiny, noisy bin
    curve = sb.calibration_curve(rows, outcome="home_win", n_bins=10)
    assert curve.empty


# ---------------------------------------------------------------------------
# home_advantage_baseline
# ---------------------------------------------------------------------------

def test_home_advantage_baseline_matches_training_frequencies():
    train = pd.DataFrame({"result": ["H", "H", "D", "A"]})
    baseline = sb.home_advantage_baseline(train)
    assert baseline["home_win"] == pytest.approx(0.5)
    assert baseline["draw"] == pytest.approx(0.25)
    assert baseline["away_win"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# elo_baseline_predictions
# ---------------------------------------------------------------------------

def test_elo_baseline_predictions_favor_the_team_with_more_history_of_winning():
    """elo_baseline_predictions now takes an already-computed Elo HISTORY
    (elo.compute_elo_history's output), not raw matches -- run_backtest
    computes this once per league instead of refitting it from scratch
    at every season boundary (a real ~330s-for-one-league performance
    bug, the same O(seasons x history) shape as an earlier, already-
    fixed bug in models/title_race.py)."""
    train = pd.DataFrame([
        {"date": "2024-01-01", "league": "E0", "home_team": "A", "away_team": "B",
         "home_goals": 3, "away_goals": 0, "result": "H"},
        {"date": "2024-01-08", "league": "E0", "home_team": "B", "away_team": "A",
         "home_goals": 0, "away_goals": 3, "result": "A"},
    ])
    history = elo.compute_elo_history(train)
    test = pd.DataFrame([{"home_team": "A", "away_team": "B"}])
    preds = sb.elo_baseline_predictions(history, test)
    assert preds.iloc[0]["home_win"] > preds.iloc[0]["away_win"]
    assert preds.iloc[0][["home_win", "draw", "away_win"]].sum() == pytest.approx(1.0)


def test_elo_baseline_predictions_on_empty_history_falls_back_to_starting_rating():
    empty_history = pd.DataFrame(columns=["date", "league", "team", "opponent", "home",
                                          "rating_before", "rating_after", "result"])
    test = pd.DataFrame([{"home_team": "A", "away_team": "B"}])
    preds = sb.elo_baseline_predictions(empty_history, test)
    assert preds.iloc[0][["home_win", "draw", "away_win"]].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# dixon_coles_predictions / market_predictions / _blend_rows
# ---------------------------------------------------------------------------

def _synthetic_train_test():
    rng = np.random.default_rng(5)
    rows = []
    date = pd.Timestamp("2020-08-01")
    for _ in range(100):
        for home, away, lh, la in [("Strong", "Weak", 2.2, 0.5), ("Weak", "Strong", 0.5, 2.0)]:
            rows.append({"home_team": home, "away_team": away,
                        "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la),
                        "date": date.strftime("%Y-%m-%d")})
            date += pd.Timedelta(days=1)
    train = pd.DataFrame(rows)
    test = pd.DataFrame([{"home_team": "Strong", "away_team": "Weak"}])
    return train, test


def test_dixon_coles_predictions_favors_the_fitted_stronger_team():
    train, test = _synthetic_train_test()
    preds = sb.dixon_coles_predictions(train, test, target="goals")
    assert preds.iloc[0]["home_win"] > preds.iloc[0]["away_win"]
    assert preds.iloc[0][["home_win", "draw", "away_win"]].sum() == pytest.approx(1.0)


def test_dixon_coles_predictions_on_target_with_no_data_returns_nan_not_a_crash():
    train, test = _synthetic_train_test()
    preds = sb.dixon_coles_predictions(train, test, target="xg")   # no xg columns in this synthetic data
    assert preds.iloc[0][["home_win", "draw", "away_win"]].isna().all()


def test_market_predictions_reads_shin_columns():
    test = pd.DataFrame([{"prob_home_shin": 0.5, "prob_draw_shin": 0.3, "prob_away_shin": 0.2}])
    preds = sb.market_predictions(test)
    assert preds.iloc[0]["home_win"] == pytest.approx(0.5)


def test_market_predictions_missing_columns_become_nan():
    test = pd.DataFrame([{"home_team": "A"}])
    preds = sb.market_predictions(test)
    assert preds.iloc[0][["home_win", "draw", "away_win"]].isna().all()


def test_blend_rows_blends_when_market_is_complete():
    model = pd.DataFrame([{"home_win": 0.6, "draw": 0.25, "away_win": 0.15}])
    market = pd.DataFrame([{"home_win": 0.4, "draw": 0.3, "away_win": 0.3}])
    blended = sb._blend_rows(model, market, market_weight=0.5)
    assert blended.iloc[0]["home_win"] == pytest.approx(0.5)


def test_blend_rows_passes_through_when_market_is_missing():
    model = pd.DataFrame([{"home_win": 0.6, "draw": 0.25, "away_win": 0.15}])
    market = pd.DataFrame([{"home_win": np.nan, "draw": np.nan, "away_win": np.nan}])
    blended = sb._blend_rows(model, market, market_weight=0.5)
    assert blended.iloc[0]["home_win"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# run_backtest / summarize -- end to end on synthetic multi-season data
# ---------------------------------------------------------------------------

def _synthetic_multiseason_matches():
    rng = np.random.default_rng(9)
    rows = []
    for season_i, season in enumerate(["2223", "2324", "2425"]):
        year = 2022 + season_i
        for round_n in range(20):
            date = pd.Timestamp(f"{year}-09-01") + pd.Timedelta(days=round_n * 7)
            for home, away, lh, la in [("Strong", "Weak", 2.2, 0.5), ("Weak", "Strong", 0.5, 2.0)]:
                hg, ag = rng.poisson(lh), rng.poisson(la)
                result = "H" if hg > ag else ("A" if ag > hg else "D")
                rows.append({
                    "league": "E0", "season": season, "date": date.strftime("%Y-%m-%d"),
                    "home_team": home, "away_team": away, "home_goals": hg, "away_goals": ag,
                    "result": result, "home_xg": np.nan, "away_xg": np.nan,
                    "prob_home_shin": np.nan, "prob_draw_shin": np.nan, "prob_away_shin": np.nan,
                })
    return pd.DataFrame(rows)


def test_run_backtest_only_predicts_seasons_after_the_minimum_training_window():
    matches = _synthetic_multiseason_matches()
    scored = sb.run_backtest(matches, min_train_seasons=2)
    # 3 seasons total (2223, 2324, 2425), min_train_seasons=2 -> only the
    # 3rd season (starting 2024-09-01) is ever predicted. Checked by date
    # threshold, not a calendar-year string prefix -- a season starting
    # in September genuinely runs into the following January, so
    # "season 2425" legitimately produces both 2024 and 2025 dates.
    assert scored["date"].min() >= "2024-09-01"


def test_run_backtest_dixon_coles_beats_the_naive_baseline_on_a_lopsided_synthetic_league():
    matches = _synthetic_multiseason_matches()
    scored = sb.run_backtest(matches, min_train_seasons=2)
    summary = sb.summarize(scored).set_index("source")
    assert summary.loc["dixon_coles", "log_loss"] < summary.loc["home_advantage_baseline", "log_loss"]


def test_summarize_excludes_sources_with_no_coverage_for_this_data():
    matches = _synthetic_multiseason_matches()   # no xG, npxG, or odds anywhere
    scored = sb.run_backtest(matches, min_train_seasons=2)
    summary = sb.summarize(scored)
    assert "dixon_coles_xg" not in summary["source"].values   # no xg column populated -> never added as a source
    assert "dixon_coles_npxg" not in summary["source"].values
    closing_line_row = summary[summary["source"] == "closing_line"]
    assert closing_line_row.iloc[0]["n"] == 0   # present as a source, but nothing it could score


def test_run_backtest_adds_npxg_source_when_coverage_exists():
    matches = _synthetic_multiseason_matches()
    matches["home_npxg"] = matches["home_goals"] * 0.9
    matches["away_npxg"] = matches["away_goals"] * 0.9
    scored = sb.run_backtest(matches, min_train_seasons=2)
    assert "dixon_coles_npxg" in scored["source"].values
    assert "dixon_coles_npxg_blended" in scored["source"].values


def test_run_backtest_adds_glm_source_when_covariate_coverage_exists():
    matches = _synthetic_multiseason_matches()
    rng = np.random.default_rng(4)
    matches["home_npxg"] = matches["home_goals"] * 0.9
    matches["away_npxg"] = matches["away_goals"] * 0.9
    matches["home_ppda"] = rng.uniform(5, 15, len(matches))
    matches["away_ppda"] = rng.uniform(5, 15, len(matches))
    matches["home_deep"] = rng.integers(0, 15, len(matches))
    matches["away_deep"] = rng.integers(0, 15, len(matches))

    scored = sb.run_backtest(matches, min_train_seasons=2)
    assert "dixon_coles_glm" in scored["source"].values
    assert "dixon_coles_glm_blended" in scored["source"].values
    glm_rows = scored[scored["source"] == "dixon_coles_glm"]
    assert glm_rows[["home_win", "draw", "away_win"]].notna().any().any()   # at least some rows actually scored


def test_run_backtest_omits_glm_source_without_covariate_coverage():
    matches = _synthetic_multiseason_matches()   # no ppda/deep/npxg anywhere
    scored = sb.run_backtest(matches, min_train_seasons=2)
    assert "dixon_coles_glm" not in scored["source"].values


def test_dixon_coles_predictions_target_columns_cover_goals_xg_and_npxg():
    assert sb.TARGET_COLUMNS["goals"] == ("home_goals", "away_goals")
    assert sb.TARGET_COLUMNS["xg"] == ("home_xg", "away_xg")
    assert sb.TARGET_COLUMNS["npxg"] == ("home_npxg", "away_npxg")


# ---------------------------------------------------------------------------
# score_archived_forecasts -- the LIVE (not backtest) tracking view
# ---------------------------------------------------------------------------

def _archive_forecast(as_of, entity_id, source, probs):
    df = pd.DataFrame([
        {"target_event": entity_id, "entity_id": entity_id, "entity_type": "fixture",
         "metric": outcome, "value": p}
        for outcome, p in probs.items()
    ])
    archive.write_forecast(source, as_of, df)


def test_score_archived_forecasts_joins_a_played_match_to_its_forecast(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    _archive_forecast("2024-08-10", "E0|2425|A|B", "dixon_coles",
                      {"home_win": 0.5, "draw": 0.3, "away_win": 0.2})
    matches = pd.DataFrame([{"league": "E0", "season": "2425", "home_team": "A", "away_team": "B",
                             "date": "2024-08-17", "result": "H"}])

    scored = sb.score_archived_forecasts(matches)
    assert len(scored) == 1
    row = scored.iloc[0]
    assert row["result"] == "H"
    assert row["source"] == "dixon_coles"
    assert row["home_win"] == pytest.approx(0.5)
    assert row["draw"] == pytest.approx(0.3)
    assert row["away_win"] == pytest.approx(0.2)


def test_score_archived_forecasts_excludes_a_fixture_not_yet_played(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    _archive_forecast("2024-08-10", "E0|2425|A|B", "dixon_coles",
                      {"home_win": 0.5, "draw": 0.3, "away_win": 0.2})
    no_results_yet = pd.DataFrame(columns=["league", "season", "home_team", "away_team", "date", "result"])

    scored = sb.score_archived_forecasts(no_results_yet)
    assert scored.empty


def test_score_archived_forecasts_keeps_sources_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    _archive_forecast("2024-08-10", "E0|2425|A|B", "dixon_coles",
                      {"home_win": 0.5, "draw": 0.3, "away_win": 0.2})
    _archive_forecast("2024-08-10", "E0|2425|A|B", "closing_line",
                      {"home_win": 0.6, "draw": 0.25, "away_win": 0.15})
    matches = pd.DataFrame([{"league": "E0", "season": "2425", "home_team": "A", "away_team": "B",
                             "date": "2024-08-17", "result": "H"}])

    scored = sb.score_archived_forecasts(matches)
    assert set(scored["source"]) == {"dixon_coles", "closing_line"}
    by_source = scored.set_index("source")["home_win"]
    assert by_source["dixon_coles"] == pytest.approx(0.5)
    assert by_source["closing_line"] == pytest.approx(0.6)


def test_score_archived_forecasts_empty_when_nothing_archived_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    scored = sb.score_archived_forecasts(pd.DataFrame())
    assert scored.empty
    assert list(scored.columns) == ["date", "league", "home_team", "away_team", "result", "source"] + sb.OUTCOME_COLS


def test_score_archived_forecasts_feeds_summarize_directly(tmp_path, monkeypatch):
    """The whole point: the live archive's scored shape must be usable
    by the exact same summarize()/log_loss() the historical backtest
    uses, with no separate code path."""
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    _archive_forecast("2024-08-10", "E0|2425|A|B", "dixon_coles",
                      {"home_win": 0.8, "draw": 0.15, "away_win": 0.05})
    matches = pd.DataFrame([{"league": "E0", "season": "2425", "home_team": "A", "away_team": "B",
                             "date": "2024-08-17", "result": "H"}])

    scored = sb.score_archived_forecasts(matches)
    summary = sb.summarize(scored)
    assert summary.iloc[0]["source"] == "dixon_coles"
    assert summary.iloc[0]["log_loss"] == pytest.approx(-np.log(0.8))
