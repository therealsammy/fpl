import math

import pandas as pd
import pytest

from collectors import fpl_odds as fo


# ---------------------------------------------------------------------------
# Devig
# ---------------------------------------------------------------------------

def test_proportional_devig_sums_to_one_and_recovers_fair_probabilities():
    true_probs = [0.5, 0.3, 0.2]
    margin = 1.06  # 6% overround, spread proportionally across outcomes
    odds = [1 / (p * margin) for p in true_probs]

    out = fo.proportional_devig(odds)

    assert sum(out) == pytest.approx(1.0)
    # Proportional margin injection is exactly what proportional devig undoes.
    for got, want in zip(out, true_probs):
        assert got == pytest.approx(want, abs=1e-9)


def test_shin_devig_sums_to_one():
    odds = [2.10, 3.40, 3.60]
    probs, z = fo.shin_devig(odds)
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert 0 <= z < 0.5


def test_shin_corrects_favorite_longshot_bias_versus_proportional():
    """Shin's well-documented property: it shifts probability mass toward
    the favorite and away from the longshot, relative to proportional."""
    odds = [2.10, 3.40, 3.60]  # home is the clear favorite, away the longshot
    prop = fo.proportional_devig(odds)
    shin, z = fo.shin_devig(odds)

    assert z > 0  # a real market has a real overround, so z should be nonzero
    assert shin[0] > prop[0]   # favorite gets MORE probability under Shin
    assert shin[2] < prop[2]   # longshot gets LESS probability under Shin


# ---------------------------------------------------------------------------
# Dixon-Coles fit
# ---------------------------------------------------------------------------

def test_fit_recovers_known_expected_goals_from_fair_market():
    true_h, true_a = 1.8, 1.1
    matrix = fo._score_matrix(true_h, true_a, fo.DIXON_COLES_RHO)
    p_h, p_d, p_a, p_o = fo.implied_from_score_matrix(matrix)

    fit = fo.fit_expected_goals(p_h, p_d, p_o)

    assert fit["lam_home"] == pytest.approx(true_h, abs=1e-2)
    assert fit["lam_away"] == pytest.approx(true_a, abs=1e-2)
    assert fit["residual"] < 1e-4


def test_fit_recovers_strong_favorite_case():
    true_h, true_a = 2.4, 0.7
    matrix = fo._score_matrix(true_h, true_a, fo.DIXON_COLES_RHO)
    p_h, p_d, p_a, p_o = fo.implied_from_score_matrix(matrix)

    fit = fo.fit_expected_goals(p_h, p_d, p_o)

    assert fit["lam_home"] == pytest.approx(true_h, abs=1e-2)
    assert fit["lam_away"] == pytest.approx(true_a, abs=1e-2)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _sample_event(home="Arsenal", away="Chelsea", event_id="ev1"):
    return {
        "id": event_id, "commence_time": "2026-01-10T15:00:00Z",
        "home_team": home, "away_team": away,
        "bookmakers": [
            {"title": "Bookie A", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": home, "price": 1.9}, {"name": "Draw", "price": 3.6},
                    {"name": away, "price": 4.2}]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": 2.5, "price": 1.95},
                    {"name": "Under", "point": 2.5, "price": 1.90},
                    {"name": "Over", "point": 3.5, "price": 3.5},  # wrong line -- must be dropped
                ]},
            ]},
            {"title": "Bookie B", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": home, "price": 1.85}, {"name": "Draw", "price": 3.7},
                    {"name": away, "price": 4.5}]},
            ]},
        ],
    }


def test_parse_odds_response_labels_outcomes_and_maps_teams():
    data = [_sample_event()]
    df = fo.parse_odds_response(data, "2026-01-06")

    assert set(df["Home team"]) == {"ARS"}
    assert set(df["Away team"]) == {"CHE"}
    assert "Home win" in df["Outcome"].values
    assert "Draw" in df["Outcome"].values
    assert "Away win" in df["Outcome"].values
    assert "Over 2.5" in df["Outcome"].values
    assert "Under 2.5" in df["Outcome"].values
    # The 3.5 line must never appear -- only the 2.5 line was requested.
    assert not df["Outcome"].str.contains("3.5").any()


def test_parse_odds_response_fails_loudly_on_unmapped_team():
    data = [_sample_event(home="Definitely Not A Real Club")]
    with pytest.raises(fo.UnmappedTeamError):
        fo.parse_odds_response(data, "2026-01-06")


def test_median_prices_takes_median_not_best():
    data = [_sample_event()]
    df = fo.parse_odds_response(data, "2026-01-06")
    med = fo.median_prices(df)

    home_win = med[(med["Outcome"] == "Home win")]["Median price"].iloc[0]
    # Two bookmakers quoted 1.9 and 1.85 for the home win -- median, not max.
    assert home_win == pytest.approx((1.9 + 1.85) / 2)


# ---------------------------------------------------------------------------
# Fixture projections (full pipeline, fair/zero-vig odds for exact recovery)
# ---------------------------------------------------------------------------

def _fair_median_df(true_h, true_a, fixture_id="ev1", home="ARS", away="CHE"):
    matrix = fo._score_matrix(true_h, true_a, fo.DIXON_COLES_RHO)
    p_h, p_d, p_a, p_o = fo.implied_from_score_matrix(matrix)
    rows = [
        {"Fixture ID": fixture_id, "Commence time": "2026-01-10T15:00:00Z",
         "Home team": home, "Away team": away, "Market": "h2h",
         "Outcome": "Home win", "Median price": 1 / p_h},
        {"Fixture ID": fixture_id, "Commence time": "2026-01-10T15:00:00Z",
         "Home team": home, "Away team": away, "Market": "h2h",
         "Outcome": "Draw", "Median price": 1 / p_d},
        {"Fixture ID": fixture_id, "Commence time": "2026-01-10T15:00:00Z",
         "Home team": home, "Away team": away, "Market": "h2h",
         "Outcome": "Away win", "Median price": 1 / p_a},
        {"Fixture ID": fixture_id, "Commence time": "2026-01-10T15:00:00Z",
         "Home team": home, "Away team": away, "Market": "totals",
         "Outcome": "Over 2.5", "Median price": 1 / p_o},
        {"Fixture ID": fixture_id, "Commence time": "2026-01-10T15:00:00Z",
         "Home team": home, "Away team": away, "Market": "totals",
         "Outcome": "Under 2.5", "Median price": 1 / (1 - p_o)},
    ]
    return pd.DataFrame(rows)


def test_build_fixture_projections_recovers_expected_goals_and_clean_sheet():
    true_h, true_a = 1.7, 1.0
    median_df = _fair_median_df(true_h, true_a)
    fixture_gws = {("ARS", "CHE"): 5}

    proj, digest = fo.build_fixture_projections(median_df, fixture_gws, "2026-01-06")

    assert len(proj) == 2
    home_row = proj[proj["Team"] == "ARS"].iloc[0]
    away_row = proj[proj["Team"] == "CHE"].iloc[0]

    assert home_row["Expected goals for"] == pytest.approx(true_h, abs=0.05)
    assert away_row["Expected goals for"] == pytest.approx(true_a, abs=0.05)
    assert home_row["GW"] == 5

    # Clean sheet probability must be exp(-opponent's expected goals), per the brief.
    assert home_row["Clean sheet probability"] == pytest.approx(
        math.exp(-away_row["Expected goals for"]), abs=1e-3)
    assert away_row["Clean sheet probability"] == pytest.approx(
        math.exp(-home_row["Expected goals for"]), abs=1e-3)

    assert len(digest) == 1


def test_build_fixture_projections_skips_incomplete_markets():
    median_df = pd.DataFrame([
        {"Fixture ID": "ev1", "Commence time": "2026-01-10T15:00:00Z",
         "Home team": "ARS", "Away team": "CHE", "Market": "h2h",
         "Outcome": "Home win", "Median price": 1.9},
        # Draw and Away win missing, and no totals market at all.
    ])
    proj, digest = fo.build_fixture_projections(median_df, {}, "2026-01-06")
    assert proj.empty
    assert digest == []


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_update_store_same_day_rerun_replaces(tmp_path):
    out = tmp_path / "fpl_odds.csv"
    rows = pd.DataFrame([{"Snapshot": "2026-01-06", "Fixture ID": "ev1",
                           "Home team": "ARS", "Away team": "CHE",
                           "Market": "h2h", "Outcome": "Home win", "Price": 1.9}])

    first = fo._update_store(rows, out)
    second = fo._update_store(rows, out)

    assert len(second) == len(first)
    assert second["Snapshot"].nunique() == 1
