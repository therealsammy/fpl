import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from core import ids
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


def test_fit_team_strengths_ridge_reins_in_a_thin_data_teams_extreme_rating():
    """Real bug found live (2026-08-28): a club back in the Premier
    League after a 25-year gap has essentially one full-weight match
    once time-decay is applied. Without per-team ridge shrinkage,
    unconstrained MLE drove that team's attack to -6.7 (a realistic
    top-flight value is roughly -1..1) trying to perfectly explain a
    single 0-3 loss -- reproduced here with a deliberately extreme
    single result for an otherwise-average team."""
    rng = np.random.default_rng(50)
    rows = []
    date = pd.Timestamp("2020-08-01")
    # 18 other teams playing a normal, unremarkable season against each other
    for _ in range(150):
        home, away = rng.choice(["A", "B", "C", "D"], size=2, replace=False)
        rows.append({"home_team": home, "away_team": away,
                    "home_goals": rng.poisson(1.3), "away_goals": rng.poisson(1.1),
                    "date": date.strftime("%Y-%m-%d")})
        date += pd.Timedelta(days=1)
    # "Newcomer" plays exactly one match, loses badly
    rows.append({"home_team": "A", "away_team": "Newcomer",
                "home_goals": 5, "away_goals": 0, "date": date.strftime("%Y-%m-%d")})
    matches = pd.DataFrame(rows)

    with_ridge = match.fit_team_strengths(matches, half_life_days=100_000)
    without_ridge = match.fit_team_strengths(matches, half_life_days=100_000, ridge_lambda=0.0)

    assert abs(with_ridge["teams"]["Newcomer"]["attack"]) < 2.0
    assert abs(with_ridge["teams"]["Newcomer"]["attack"]) < abs(without_ridge["teams"]["Newcomer"]["attack"])


def test_fit_team_strengths_ridge_barely_touches_a_well_observed_team():
    """The other half of the same guarantee: a team with a full,
    unremarkable season of data shouldn't visibly shift just because
    ridge shrinkage exists -- it should matter for thin data, not
    everywhere."""
    rng = np.random.default_rng(3)
    rows = []
    date = pd.Timestamp("2020-08-01")
    for _ in range(300):
        home, away = rng.choice(["A", "B", "C", "D"], size=2, replace=False)
        rows.append({"home_team": home, "away_team": away,
                    "home_goals": rng.poisson(1.3), "away_goals": rng.poisson(1.1),
                    "date": date.strftime("%Y-%m-%d")})
        date += pd.Timedelta(days=1)
    matches = pd.DataFrame(rows)

    with_ridge = match.fit_team_strengths(matches, half_life_days=100_000)
    without_ridge = match.fit_team_strengths(matches, half_life_days=100_000, ridge_lambda=0.0)

    for team in ["A", "B", "C", "D"]:
        assert with_ridge["teams"][team]["attack"] == pytest.approx(
            without_ridge["teams"][team]["attack"], abs=0.1)


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


def _fake_xg_row(date, home, away, is_result):
    return {"league": "E0", "season": "2627", "date": date,
           "home_team_raw": home, "away_team_raw": away,
           "home_team_id": None, "away_team_id": None, "is_result": is_result,
           "home_goals": 1 if is_result else None, "away_goals": 0 if is_result else None,
           "home_xg": 1.2 if is_result else None, "away_xg": 0.5 if is_result else None}


def test_upcoming_fixtures_excludes_played_matches(monkeypatch):
    fake = pd.DataFrame([
        _fake_xg_row("2026-08-20", "A", "B", is_result=True),
        _fake_xg_row("2026-08-28", "C", "D", is_result=False),
    ])
    monkeypatch.setattr(match, "_load_xg", lambda leagues=None: fake)

    result = match.upcoming_fixtures("E0", today="2026-08-25")
    assert len(result) == 1
    assert result.iloc[0]["home_team"] == "C"


def test_upcoming_fixtures_windows_off_the_next_fixture_not_today(monkeypatch):
    """Anchoring to the soonest scheduled match, not to 'today', so a
    round that starts several days out isn't missed and the round after
    it doesn't bleed in -- verified against real data before picking
    this design (see the function's own docstring)."""
    fake = pd.DataFrame([
        _fake_xg_row("2026-08-28", "A", "B", is_result=False),   # round 1
        _fake_xg_row("2026-08-31", "C", "D", is_result=False),   # round 1, 3 days later
        _fake_xg_row("2026-09-05", "E", "F", is_result=False),   # round 2, well outside the window
    ])
    monkeypatch.setattr(match, "_load_xg", lambda leagues=None: fake)
    result = match.upcoming_fixtures("E0", round_window_days=4, today="2026-08-25")
    assert set(zip(result["home_team"], result["away_team"])) == {("A", "B"), ("C", "D")}


def test_upcoming_fixtures_empty_when_nothing_scheduled(monkeypatch):
    fake = pd.DataFrame([_fake_xg_row("2026-08-20", "A", "B", is_result=True)])
    monkeypatch.setattr(match, "_load_xg", lambda leagues=None: fake)
    result = match.upcoming_fixtures("E0", today="2026-08-25")
    assert result.empty


def test_upcoming_fixtures_translates_understat_names_to_football_data_spelling(tmp_path, monkeypatch):
    """The real bug this guards against: Understat calls it 'Manchester
    City', but fit_team_strengths/compute_rolling_form/a fitted
    strengths dict are all keyed by football-data.co.uk's naming (here,
    'Man City') via elo.load_all_matches() -- returning Understat's own
    spelling would silently never match anything downstream."""
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    teams_df = pd.DataFrame([{
        "canonical_id": "fpl-team-15", "display_name": "Man City", "fpl_id": 15,
        "understat_id": pd.NA, "statsbomb_id": pd.NA, "football_data_name": "Man City",
        "understat_name": "Manchester City", "country": "England", "tier": 1,
        "confidence": 1.0, "method": "exact", "verified_at": pd.NA,
    }])
    ids._save(teams_df, ids.TEAMS_PATH, ids.TEAM_SCHEMA)

    fake = pd.DataFrame([{
        "league": "E0", "season": "2627", "date": "2026-08-28",
        "home_team_raw": "Manchester City", "away_team_raw": "Some Other Club",
        "home_team_id": "fpl-team-15", "away_team_id": None, "is_result": False,
        "home_goals": None, "away_goals": None, "home_xg": None, "away_xg": None,
    }])
    monkeypatch.setattr(match, "_load_xg", lambda leagues=None: fake)

    result = match.upcoming_fixtures("E0", today="2026-08-25")
    assert result.iloc[0]["home_team"] == "Man City"           # translated
    assert result.iloc[0]["away_team"] == "Some Other Club"    # unresolved id -> falls back to raw name


def test_upcoming_fixtures_empty_when_no_data_at_all(monkeypatch):
    monkeypatch.setattr(match, "_load_xg", lambda leagues=None: pd.DataFrame())
    result = match.upcoming_fixtures("E0")
    assert result.empty


def test_compute_rolling_form_never_includes_the_matchs_own_value():
    """The core leakage guarantee: a match's own stat value must not
    appear in its own form column, only in LATER matches' form."""
    matches = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "home_ppda": 10.0, "away_ppda": 20.0, "date": "2024-08-01"},
        {"home_team": "B", "away_team": "A", "home_ppda": 30.0, "away_ppda": 40.0, "date": "2024-08-08"},
    ])
    result = match.compute_rolling_form(matches, ["ppda"], window=5)

    # A's first-ever match: no prior history -> NaN, not fabricated.
    assert pd.isna(result.iloc[0]["home_ppda_form"])
    # A's second match (as away in row 2): form = mean of A's OWN prior
    # values only (10.0, from row 1) -- never 40.0, its own value in this row.
    assert result.iloc[1]["away_ppda_form"] == pytest.approx(10.0)


def test_compute_rolling_form_averages_over_the_window():
    long_history = pd.DataFrame([
        {"home_team": "A", "away_team": "X", "home_ppda": v, "away_ppda": 0.0,
        "date": (pd.Timestamp("2024-08-01") + pd.Timedelta(days=7 * i)).strftime("%Y-%m-%d")}
        for i, v in enumerate([10.0, 20.0, 30.0, 40.0])
    ])
    result = match.compute_rolling_form(long_history, ["ppda"], window=2)
    # 4th match's form = mean of the 2 immediately prior (20, 30), not all 3.
    assert result.iloc[3]["home_ppda_form"] == pytest.approx(25.0)


def test_compute_rolling_form_tolerates_a_stat_with_no_underlying_columns():
    """A league Understat doesn't cover (or any caller without a given
    stat at all) must get all-NaN form columns, not a KeyError -- same
    'absent coverage is expected' convention as the rest of the xG
    pipeline."""
    matches = pd.DataFrame([{"home_team": "A", "away_team": "B", "date": "2024-08-01"}])
    result = match.compute_rolling_form(matches, ["ppda"], window=5)
    assert pd.isna(result.iloc[0]["home_ppda_form"])
    assert pd.isna(result.iloc[0]["away_ppda_form"])


def test_current_form_includes_the_most_recent_match_unlike_compute_rolling_form():
    matches = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "home_ppda": 10.0, "away_ppda": 5.0, "date": "2024-08-01"},
        {"home_team": "A", "away_team": "B", "home_ppda": 20.0, "away_ppda": 5.0, "date": "2024-08-08"},
    ])
    assert match.current_form(matches, "A", "ppda", window=5) == pytest.approx(15.0)


def test_current_form_returns_none_for_a_team_with_no_history():
    matches = pd.DataFrame([{"home_team": "A", "away_team": "B", "home_ppda": 10.0,
                             "away_ppda": 5.0, "date": "2024-08-01"}])
    assert match.current_form(matches, "Promoted FC", "ppda") is None


# ---------------------------------------------------------------------------
# GLM extension
# ---------------------------------------------------------------------------

def test_neg_log_likelihood_glm_grad_matches_scipys_finite_difference_estimate():
    """Same reasoning as the plain gradient's check: a wrong analytic
    gradient is worse than none, since L-BFGS-B trusts it completely."""
    rng = np.random.default_rng(31)
    n_teams, n_matches, n_covariates = 6, 40, 2
    home_idx = rng.integers(0, n_teams, n_matches)
    away_idx = (home_idx + rng.integers(1, n_teams, n_matches)) % n_teams
    home_target = rng.poisson(1.4, n_matches).astype(float)
    away_target = rng.poisson(1.1, n_matches).astype(float)
    home_features = rng.normal(0, 1, (n_matches, n_covariates))
    away_features = rng.normal(0, 1, (n_matches, n_covariates))
    weights = rng.uniform(0.3, 1.0, n_matches)

    x0 = rng.normal(0, 0.2, 2 * n_teams + 1 + n_covariates)
    error = check_grad(match._neg_log_likelihood_glm, match._neg_log_likelihood_glm_grad, x0,
                       home_idx, away_idx, home_target, away_target,
                       home_features, away_features, weights, n_teams, n_covariates)
    assert error < 1e-4


def test_fit_team_strengths_glm_recovers_a_genuine_positive_covariate_effect():
    """Synthetic data where a covariate has a KNOWN, real effect on
    scoring rate above and beyond team identity -- the fit should
    recover a clearly positive beta, not just 'not crash.'"""
    rng = np.random.default_rng(17)
    true_beta = 0.4
    rows = []
    date = pd.Timestamp("2024-08-01")
    for _ in range(300):
        form_value = rng.uniform(-1, 1)
        lh = np.exp(0.2 + form_value * true_beta)   # neutral team strength, only form varies
        rows.append({"home_team": "A", "away_team": "B", "home_goals": rng.poisson(lh),
                    "away_goals": rng.poisson(1.0), "date": date.strftime("%Y-%m-%d"),
                    "home_form": form_value, "away_form": 0.0})
        date += pd.Timedelta(days=1)
    matches = pd.DataFrame(rows)

    strengths = match.fit_team_strengths_glm(matches, covariates=["form"], half_life_days=100_000)
    assert strengths["coefficients"]["form"] == pytest.approx(true_beta, abs=0.1)


def test_fit_team_strengths_glm_drops_rows_missing_a_covariate():
    matches = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 1, "date": "2024-01-01",
        "home_ppda_form": 8.0, "away_ppda_form": 9.0},
        {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 0, "date": "2024-01-08",
        "home_ppda_form": None, "away_ppda_form": 9.0},   # no rolling form yet -- must be dropped
    ])
    strengths = match.fit_team_strengths_glm(matches, covariates=["ppda_form"])
    assert "A" in strengths["teams"]   # fit succeeded on the one usable row


def test_fit_team_strengths_glm_missing_covariate_columns_returns_empty_result():
    matches = pd.DataFrame([{"home_team": "A", "away_team": "B", "home_goals": 1,
                             "away_goals": 1, "date": "2024-01-01"}])
    strengths = match.fit_team_strengths_glm(matches, covariates=["ppda"])
    assert strengths["teams"] == {}
    assert strengths["coefficients"] == {"ppda": 0.0}


def test_team_rates_glm_applies_the_fitted_coefficient():
    strengths = {"teams": {"A": {"attack": 0.0, "defense": 0.0}, "B": {"attack": 0.0, "defense": 0.0}},
                "home_advantage": 0.0, "coefficients": {"ppda_form": 0.5}}
    lambda_with_form, _ = match.team_rates_glm(strengths, "A", "B",
                                               home_form={"ppda_form": 1.0}, away_form={})
    lambda_without_form, _ = match.team_rates_glm(strengths, "A", "B",
                                                  home_form={}, away_form={})
    assert lambda_with_form > lambda_without_form
    assert lambda_with_form == pytest.approx(lambda_without_form * np.exp(0.5))


def test_team_rates_glm_missing_covariate_in_form_dict_contributes_zero():
    strengths = {"teams": {"A": {"attack": 0.1, "defense": 0.0}}, "home_advantage": 0.0,
                "coefficients": {"deep_form": 0.3}}
    lambda_home, _ = match.team_rates_glm(strengths, "A", "A", home_form={}, away_form={})
    assert np.isfinite(lambda_home)


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


# ---------------------------------------------------------------------------
# _odds_code_to_football_data_name / load_live_odds_predictions
#
# Pure (uncached) by design -- see their docstrings -- so a plain script
# (models/live_predictions.py) can call them with no Streamlit runtime.
# Moved here from tests/test_ledger_data.py when the logic moved out of
# core/ledger_data.py's @st.cache_data wrappers.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_id_registry_for_odds_tests(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")


def _seed_arsenal_and_chelsea():
    teams_df = pd.DataFrame([
        {"canonical_id": "fpl-team-1", "display_name": "Arsenal", "fpl_id": 1,
         "understat_id": pd.NA, "statsbomb_id": pd.NA, "football_data_name": "Arsenal",
         "understat_name": pd.NA, "country": "England", "tier": 1,
         "confidence": 1.0, "method": "exact", "verified_at": pd.NA},
        {"canonical_id": "fpl-team-2", "display_name": "Chelsea", "fpl_id": 2,
         "understat_id": pd.NA, "statsbomb_id": pd.NA, "football_data_name": "Chelsea",
         "understat_name": pd.NA, "country": "England", "tier": 1,
         "confidence": 1.0, "method": "exact", "verified_at": pd.NA},
    ])
    ids._save(teams_df, ids.TEAMS_PATH, ids.TEAM_SCHEMA)


def test_odds_code_mapping_picks_football_datas_own_spelling_not_the_odds_apis():
    """MUN maps to both 'Man Utd' and 'Man United' in TEAM_NAME_MAP --
    naively keeping 'whichever comes last in the dict' gives the wrong
    one; resolving through the id registry must give football-data's
    own name instead."""
    teams_df = pd.DataFrame([{
        "canonical_id": "fpl-team-14", "display_name": "Man United", "fpl_id": 14,
        "understat_id": pd.NA, "statsbomb_id": pd.NA, "football_data_name": "Man United",
        "understat_name": pd.NA, "country": "England", "tier": 1,
        "confidence": 1.0, "method": "exact", "verified_at": pd.NA,
    }])
    ids._save(teams_df, ids.TEAMS_PATH, ids.TEAM_SCHEMA)

    mapping = match._odds_code_to_football_data_name()
    assert mapping["MUN"] == "Man United"


def test_odds_code_mapping_skips_a_code_with_no_football_data_name_on_record():
    ids._save(pd.DataFrame(columns=ids.TEAM_SCHEMA), ids.TEAMS_PATH, ids.TEAM_SCHEMA)
    mapping = match._odds_code_to_football_data_name()
    assert "ARS" not in mapping


def test_load_live_odds_predictions_uses_only_the_latest_snapshot(tmp_path, monkeypatch):
    _seed_arsenal_and_chelsea()
    path = tmp_path / "fixture_projections.csv"
    pd.DataFrame([
        {"Snapshot": "2026-08-20", "Team": "ARS", "Opponent": "CHE", "Home": True,
         "Win probability": 0.9, "Draw probability": 0.05, "Loss probability": 0.05},
        {"Snapshot": "2026-08-22", "Team": "ARS", "Opponent": "CHE", "Home": True,
         "Win probability": 0.5, "Draw probability": 0.3, "Loss probability": 0.2},
    ]).to_csv(path, index=False)

    result = match.load_live_odds_predictions(path=path)
    assert len(result) == 1
    assert result.iloc[0]["home_win"] == pytest.approx(0.5)


def test_load_live_odds_predictions_keeps_only_the_home_perspective_row(tmp_path, monkeypatch):
    _seed_arsenal_and_chelsea()
    path = tmp_path / "fixture_projections.csv"
    pd.DataFrame([
        {"Snapshot": "2026-08-22", "Team": "ARS", "Opponent": "CHE", "Home": True,
         "Win probability": 0.5, "Draw probability": 0.3, "Loss probability": 0.2},
        {"Snapshot": "2026-08-22", "Team": "CHE", "Opponent": "ARS", "Home": False,
         "Win probability": 0.2, "Draw probability": 0.3, "Loss probability": 0.5},
    ]).to_csv(path, index=False)

    result = match.load_live_odds_predictions(path=path)
    assert len(result) == 1
    assert result.iloc[0]["home_team"] == "Arsenal"
    assert result.iloc[0]["away_team"] == "Chelsea"


def test_load_live_odds_predictions_drops_a_fixture_with_an_unresolved_team(tmp_path, monkeypatch):
    _seed_arsenal_and_chelsea()
    path = tmp_path / "fixture_projections.csv"
    pd.DataFrame([
        {"Snapshot": "2026-08-22", "Team": "ARS", "Opponent": "XYZ", "Home": True,
         "Win probability": 0.5, "Draw probability": 0.3, "Loss probability": 0.2},
    ]).to_csv(path, index=False)

    result = match.load_live_odds_predictions(path=path)
    assert result.empty


def test_load_live_odds_predictions_empty_when_file_does_not_exist(tmp_path):
    result = match.load_live_odds_predictions(path=tmp_path / "nope.csv")
    assert result.empty
