import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from models import match


def _matches(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# time_weights
# ---------------------------------------------------------------------------

def test_time_weights_gives_a_recent_match_full_weight():
    dates = pd.Series(["2024-06-01"])
    w = match.time_weights(dates, as_of="2024-06-01", half_life_days=180)
    assert w[0] == pytest.approx(1.0)


def test_time_weights_halves_at_exactly_one_half_life():
    dates = pd.Series(["2024-01-01"])
    w = match.time_weights(dates, as_of="2024-06-30", half_life_days=180)  # 181 days
    assert w[0] == pytest.approx(0.5, rel=0.02)


def test_time_weights_older_matches_count_less():
    dates = pd.Series(["2020-01-01", "2024-01-01"])
    w = match.time_weights(dates, as_of="2024-06-01", half_life_days=180)
    assert w[0] < w[1]


# ---------------------------------------------------------------------------
# fit_team_strengths -- recovers known structure from synthetic data
# ---------------------------------------------------------------------------

def _synthetic_league(rng, n_teams=8, matches_per_pair=3, true_attack=None, true_defense=None,
                      home_adv=0.3):
    """Round-robin-ish synthetic data drawn from a KNOWN Dixon-Coles
    process, so the fit can be checked against ground truth instead of
    just 'it ran'."""
    teams = [f"T{i}" for i in range(n_teams)]
    true_attack = true_attack or {t: rng.normal(0, 0.4) for t in teams}
    true_defense = true_defense or {t: rng.normal(0, 0.4) for t in teams}

    rows = []
    date = pd.Timestamp("2024-08-01")
    for home in teams:
        for away in teams:
            if home == away:
                continue
            for _ in range(matches_per_pair):
                lh = np.exp(true_attack[home] + true_defense[away] + home_adv)
                la = np.exp(true_attack[away] + true_defense[home])
                rows.append({
                    "home_team": home, "away_team": away,
                    "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la),
                    "date": date.strftime("%Y-%m-%d"),
                })
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows), true_attack, true_defense, home_adv


def test_fit_team_strengths_recovers_relative_team_quality():
    """A team with a strong true attack/weak defense should end up
    fitted stronger (attack - defense) than a weak team, even though
    the exact numbers won't match ground truth 1:1 (identifiability
    only pins relative strength, see module docstring)."""
    rng = np.random.default_rng(42)
    strong_attack = {"Strong": 1.0, "Weak": -1.0}
    strong_defense = {"Strong": -0.5, "Weak": 0.5}
    teams = ["Strong", "Weak"]
    rows = []
    date = pd.Timestamp("2024-08-01")
    for _ in range(200):
        for home, away in [("Strong", "Weak"), ("Weak", "Strong")]:
            lh = np.exp(strong_attack[home] + strong_defense[away] + 0.2)
            la = np.exp(strong_attack[away] + strong_defense[home])
            rows.append({"home_team": home, "away_team": away,
                        "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la),
                        "date": date.strftime("%Y-%m-%d")})
            date += pd.Timedelta(days=1)
    matches = pd.DataFrame(rows)

    strengths = match.fit_team_strengths(matches, half_life_days=100_000)  # effectively no decay
    net_strong = strengths["teams"]["Strong"]["attack"] - strengths["teams"]["Strong"]["defense"]
    net_weak = strengths["teams"]["Weak"]["attack"] - strengths["teams"]["Weak"]["defense"]
    assert net_strong > net_weak


def test_neg_log_likelihood_grad_matches_scipys_own_finite_difference_estimate():
    """A wrong analytic gradient is worse than none -- L-BFGS-B trusts it
    completely and would converge to the wrong answer with no obvious
    sign anything was off. Checked against scipy's own numerical
    estimate before ever wiring this in (added for a real performance
    fix: L-BFGS-B without this fell back to finite-difference gradients,
    ~2 extra objective evaluations per parameter per iteration, and was
    the dominant cost in a walk-forward backtest over real data)."""
    rng = np.random.default_rng(21)
    n_teams = 6
    n_matches = 40
    home_idx = rng.integers(0, n_teams, n_matches)
    away_idx = (home_idx + rng.integers(1, n_teams, n_matches)) % n_teams   # never home == away
    home_target = rng.poisson(1.4, n_matches).astype(float)
    away_target = rng.poisson(1.1, n_matches).astype(float)
    weights = rng.uniform(0.3, 1.0, n_matches)

    x0 = rng.normal(0, 0.3, 2 * n_teams + 1)
    error = check_grad(match._neg_log_likelihood, match._neg_log_likelihood_grad, x0,
                       home_idx, away_idx, home_target, away_target, weights, n_teams)
    assert error < 1e-4


def test_fit_team_strengths_identifiability_penalty_centers_attack_near_zero():
    rng = np.random.default_rng(7)
    matches, *_ = _synthetic_league(rng, n_teams=6, matches_per_pair=2)
    strengths = match.fit_team_strengths(matches, half_life_days=100_000)
    mean_attack = np.mean([t["attack"] for t in strengths["teams"].values()])
    assert mean_attack == pytest.approx(0.0, abs=0.05)


def test_fit_team_strengths_drops_rows_with_no_target():
    matches = _matches([
        {"home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 1, "date": "2024-01-01"},
        {"home_team": "A", "away_team": "B", "home_goals": None, "away_goals": None, "date": "2024-01-08"},
    ])
    strengths = match.fit_team_strengths(matches)
    assert "A" in strengths["teams"] and "B" in strengths["teams"]


def test_fit_team_strengths_on_empty_input_returns_empty_result():
    empty = pd.DataFrame(columns=["home_team", "away_team", "home_goals", "away_goals", "date"])
    strengths = match.fit_team_strengths(empty)
    assert strengths["teams"] == {}


def test_fit_team_strengths_works_identically_on_fractional_xg_target():
    """The whole point of a pluggable target: fitting on non-integer xG
    values must not raise, and should still separate a clearly stronger
    team from a clearly weaker one."""
    rng = np.random.default_rng(3)
    rows = []
    date = pd.Timestamp("2024-08-01")
    for _ in range(150):
        for home, away, lh, la in [("Strong", "Weak", 2.1, 0.6), ("Weak", "Strong", 0.5, 1.9)]:
            rows.append({"home_team": home, "away_team": away,
                        "home_xg": round(rng.gamma(4, lh / 4), 2),
                        "away_xg": round(rng.gamma(4, la / 4), 2),
                        "date": date.strftime("%Y-%m-%d")})
            date += pd.Timedelta(days=1)
    matches = pd.DataFrame(rows)

    strengths = match.fit_team_strengths(matches, half_life_days=100_000,
                                         home_target_col="home_xg", away_target_col="away_xg")
    net_strong = strengths["teams"]["Strong"]["attack"] - strengths["teams"]["Strong"]["defense"]
    net_weak = strengths["teams"]["Weak"]["attack"] - strengths["teams"]["Weak"]["defense"]
    assert net_strong > net_weak


# ---------------------------------------------------------------------------
# tau / rho
# ---------------------------------------------------------------------------

def test_tau_is_one_outside_the_four_low_score_cells():
    assert match._tau(2, 2, 1.4, 1.1, rho=-0.1) == 1.0
    assert match._tau(3, 0, 1.4, 1.1, rho=-0.1) == 1.0


def test_tau_modifies_exactly_the_four_low_score_cells():
    rho = -0.1
    assert match._tau(0, 0, 1.4, 1.1, rho) != 1.0
    assert match._tau(0, 1, 1.4, 1.1, rho) != 1.0
    assert match._tau(1, 0, 1.4, 1.1, rho) != 1.0
    assert match._tau(1, 1, 1.4, 1.1, rho) != 1.0


def test_tau_array_matches_scalar_tau_cell_by_cell():
    """_tau_array is a vectorized rewrite of the scalar _tau, purely for
    fit_rho's performance (see its docstring) -- must produce identical
    numbers, not just plausible ones."""
    home_goals = np.array([0, 0, 1, 1, 2, 3, 0])
    away_goals = np.array([0, 1, 0, 1, 2, 0, 3])
    lambda_home = np.array([1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4])
    lambda_away = np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
    rho = -0.12

    vectorized = match._tau_array(home_goals, away_goals, lambda_home, lambda_away, rho)
    scalar = np.array([match._tau(int(h), int(a), lh, la, rho)
                       for h, a, lh, la in zip(home_goals, away_goals, lambda_home, lambda_away)])
    np.testing.assert_allclose(vectorized, scalar)


def test_fit_rho_stays_within_bounds():
    rng = np.random.default_rng(11)
    matches, *_ = _synthetic_league(rng, n_teams=6, matches_per_pair=4)
    strengths = match.fit_team_strengths(matches, half_life_days=100_000)
    rho = match.fit_rho(matches, strengths, half_life_days=100_000)
    assert match.DEFAULT_RHO_BOUNDS[0] <= rho <= match.DEFAULT_RHO_BOUNDS[1]


def test_fit_rho_on_empty_matches_returns_zero():
    empty = pd.DataFrame(columns=["home_team", "away_team", "home_goals", "away_goals", "date"])
    strengths = {"teams": {}, "home_advantage": 0.0, "as_of": "2024-01-01"}
    assert match.fit_rho(empty, strengths) == 0.0


# ---------------------------------------------------------------------------
# score_matrix / match_probabilities
# ---------------------------------------------------------------------------

def test_score_matrix_sums_to_one():
    grid = match.score_matrix(1.4, 1.1, rho=-0.1)
    assert grid.sum() == pytest.approx(1.0)


def test_score_matrix_sums_to_one_even_with_extreme_rho():
    grid = match.score_matrix(0.3, 0.2, rho=0.3)   # near the fit bound, could push tau negative
    assert grid.sum() == pytest.approx(1.0)
    assert (grid >= 0).all()


def test_match_probabilities_1x2_sums_to_one():
    grid = match.score_matrix(1.6, 1.0, rho=-0.1)
    probs = match.match_probabilities(grid)
    assert probs["home_win"] + probs["draw"] + probs["away_win"] == pytest.approx(1.0)


def test_match_probabilities_over_under_sums_to_one():
    grid = match.score_matrix(1.6, 1.0, rho=-0.1)
    probs = match.match_probabilities(grid)
    assert probs["over"] + probs["under"] == pytest.approx(1.0)


def test_match_probabilities_btts_sums_to_one():
    grid = match.score_matrix(1.6, 1.0, rho=-0.1)
    probs = match.match_probabilities(grid)
    assert probs["btts_yes"] + probs["btts_no"] == pytest.approx(1.0)


def test_match_probabilities_favors_the_stronger_attack():
    grid = match.score_matrix(2.5, 0.6, rho=-0.1)
    probs = match.match_probabilities(grid)
    assert probs["home_win"] > probs["away_win"]


def test_match_probabilities_clean_sheet_uses_the_opponents_score_row():
    # home (lambda=1.6) faces an away side that scores less (lambda=1.0)
    # -- clean_sheet_home = P(away scores 0) = Poisson(0; 1.0) ~= 0.368,
    # clean_sheet_away = P(home scores 0) = Poisson(0; 1.6) ~= 0.202.
    # The home team, facing the weaker-scoring side, keeps a clean sheet
    # MORE often, not less -- hand-traced after an initial wrong guess.
    grid = match.score_matrix(1.6, 1.0, rho=-0.1)
    probs = match.match_probabilities(grid)
    assert probs["clean_sheet_home"] > probs["clean_sheet_away"]


# ---------------------------------------------------------------------------
# team_rates / predict_match -- fallback for an unrated team
# ---------------------------------------------------------------------------

def test_team_rates_falls_back_to_average_for_an_unknown_team():
    strengths = {"teams": {"A": {"attack": 1.0, "defense": -1.0},
                           "B": {"attack": -1.0, "defense": 1.0}},
                "home_advantage": 0.2}
    lh, la = match.team_rates(strengths, "A", "Promoted FC")
    assert lh > 0 and la > 0   # doesn't crash, produces sane positive rates


def test_predict_match_end_to_end():
    strengths = {"teams": {"A": {"attack": 0.6, "defense": -0.3},
                           "B": {"attack": -0.2, "defense": 0.1}},
                "home_advantage": 0.25}
    probs = match.predict_match(strengths, rho=-0.1, home_team="A", away_team="B")
    assert probs["home_win"] + probs["draw"] + probs["away_win"] == pytest.approx(1.0)
    assert probs["lambda_home"] > 0


# ---------------------------------------------------------------------------
# blend_with_market
# ---------------------------------------------------------------------------

def test_blend_with_market_is_a_valid_distribution():
    model = {"home_win": 0.6, "draw": 0.25, "away_win": 0.15}
    market = {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
    blended = match.blend_with_market(model, market, market_weight=0.5)
    assert sum(blended.values()) == pytest.approx(1.0)
    assert blended["home_win"] == pytest.approx(0.55)


def test_blend_with_market_weight_zero_returns_the_model_untouched():
    model = {"home_win": 0.6, "draw": 0.25, "away_win": 0.15}
    market = {"home_win": 0.1, "draw": 0.1, "away_win": 0.8}
    blended = match.blend_with_market(model, market, market_weight=0.0)
    assert blended == pytest.approx(model)


def test_blend_with_market_weight_one_returns_the_market_untouched():
    model = {"home_win": 0.6, "draw": 0.25, "away_win": 0.15}
    market = {"home_win": 0.1, "draw": 0.1, "away_win": 0.8}
    blended = match.blend_with_market(model, market, market_weight=1.0)
    assert blended == pytest.approx(market)


def test_player_xg_shares_ranks_by_share_of_team_npxg():
    players = pd.DataFrame([
        {"league": "E0", "season": "2425", "player_name": "Star", "team": "A", "npxg": 15.0},
        {"league": "E0", "season": "2425", "player_name": "Squad player", "team": "A", "npxg": 5.0},
        {"league": "E0", "season": "2425", "player_name": "Other team's star", "team": "B", "npxg": 20.0},
    ])
    shares = match.player_xg_shares(players, "E0", "2425", "A")
    assert list(shares["player_name"]) == ["Star", "Squad player"]
    assert shares.iloc[0]["xg_share"] == pytest.approx(0.75)
    assert shares.iloc[1]["xg_share"] == pytest.approx(0.25)


def test_player_xg_shares_zero_total_does_not_divide_by_zero():
    players = pd.DataFrame([{"league": "E0", "season": "2425", "player_name": "Bench",
                             "team": "A", "npxg": 0.0}])
    shares = match.player_xg_shares(players, "E0", "2425", "A")
    assert shares.iloc[0]["xg_share"] == 0.0


def test_adjust_attack_for_missing_players_reduces_expected_goals():
    strengths = {"teams": {"A": {"attack": 0.5, "defense": -0.2}, "B": {"attack": 0.0, "defense": 0.0}},
                "home_advantage": 0.2}
    lambda_before, _ = match.team_rates(strengths, "A", "B")

    adjusted = match.adjust_attack_for_missing_players(strengths, "A", [0.3])
    lambda_after, _ = match.team_rates(adjusted, "A", "B")

    assert lambda_after < lambda_before
    assert lambda_after == pytest.approx(lambda_before * 0.7, rel=1e-6)


def test_adjust_attack_for_missing_players_does_not_mutate_the_original():
    strengths = {"teams": {"A": {"attack": 0.5, "defense": -0.2}}, "home_advantage": 0.2}
    match.adjust_attack_for_missing_players(strengths, "A", [0.3])
    assert strengths["teams"]["A"]["attack"] == pytest.approx(0.5)   # untouched


def test_adjust_attack_for_missing_players_caps_combined_share_below_total_wipeout():
    strengths = {"teams": {"A": {"attack": 0.5, "defense": -0.2}, "B": {"attack": 0.0, "defense": 0.0}},
                "home_advantage": 0.2}
    adjusted = match.adjust_attack_for_missing_players(strengths, "A", [0.6, 0.6])   # sums past 1.0
    lambda_after, _ = match.team_rates(adjusted, "A", "B")
    assert lambda_after > 0   # never zeroed out entirely


def test_adjust_attack_for_missing_players_unknown_team_is_a_no_op():
    strengths = {"teams": {"A": {"attack": 0.5, "defense": -0.2}}, "home_advantage": 0.2}
    adjusted = match.adjust_attack_for_missing_players(strengths, "Promoted FC", [0.3])
    assert adjusted["teams"]["A"]["attack"] == pytest.approx(0.5)


def test_blend_with_market_passes_through_keys_the_market_does_not_have():
    """BTTS has no collected market odds -- blending must not crash or
    drop the model's own BTTS estimate just because the market dict
    doesn't mention it."""
    model = {"home_win": 0.6, "draw": 0.25, "away_win": 0.15, "btts_yes": 0.55}
    market = {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
    blended = match.blend_with_market(model, market, market_weight=0.5)
    assert blended["btts_yes"] == pytest.approx(0.55)
