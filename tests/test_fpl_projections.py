import numpy as np
import pandas as pd
import pytest

import fpl_projections as fp


def hist_row(snapshot, gw, finished, pid, player, team, pos, price=6.0,
             minutes=900, xg=3.0, xa=2.0, defcon_per90=1.0, bonus=3):
    return {"Snapshot": snapshot, "GW": gw, "GW finished": finished, "ID": pid,
            "Player": player, "Team": team, "Pos": pos, "Price": price,
            "Minutes": minutes, "xG": xg, "xA": xa, "DEFCON per 90": defcon_per90,
            "Bonus": bonus}


def minutes_row(snapshot, pid, start_prob=0.9, classification="Nailed",
                 min_per_app=85.0, appearances=10, starts_overall=9, games_overall=10):
    return {"Snapshot": snapshot, "ID": pid, "Start probability": start_prob,
            "Classification": classification, "Minutes per appearance": min_per_app,
            "Appearances": appearances, "Starts (overall)": starts_overall,
            "Games (overall)": games_overall}


def fixture_row(snapshot, gw, team, opponent, home, xg_for=1.8, xg_against=1.0, cs_prob=0.35):
    return {"Snapshot": snapshot, "GW": gw, "Team": team, "Opponent": opponent, "Home": home,
            "Expected goals for": xg_for, "Expected goals against": xg_against,
            "Clean sheet probability": cs_prob}


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------

def _six_snapshot_history():
    rows = []
    for i, gw in enumerate(range(1, 7)):
        rows.append(hist_row(f"2026-01-{6+i:02d}", gw, True, 1, "A", "ARS", "MID"))
    return pd.DataFrame(rows)


def test_gate_blocks_on_too_few_snapshots():
    history = pd.DataFrame([hist_row("2026-01-06", 1, True, 1, "A", "ARS", "MID")])
    problems = fp.check_readiness(history, None, None, None)
    assert any("snapshot(s) in fpl_history.csv" in p for p in problems)


def test_gate_blocks_on_missing_minutes_file():
    history = _six_snapshot_history()
    problems = fp.check_readiness(history, None, pd.DataFrame(), None)
    assert any("fpl_minutes.csv not found" in p for p in problems)


def test_gate_blocks_on_low_minutes_coverage():
    history = _six_snapshot_history()
    minutes_df = pd.DataFrame([minutes_row("2026-02-10", 1, start_prob=None)])
    problems = fp.check_readiness(history, minutes_df, pd.DataFrame(), None)
    assert any("covers only" in p for p in problems)


def test_gate_blocks_when_fixture_projections_missing_target_gw():
    history = _six_snapshot_history()
    minutes_df = pd.DataFrame([minutes_row("2026-02-10", 1)])
    fixture_proj = pd.DataFrame([fixture_row("2026-02-10", 3, "ARS", "CHE", True)])
    problems = fp.check_readiness(history, minutes_df, fixture_proj, target_gw=7)
    assert any("doesn't cover GW7" in p for p in problems)


def test_gate_open_when_everything_satisfied():
    history = _six_snapshot_history()
    minutes_df = pd.DataFrame([minutes_row("2026-02-10", 1)])
    fixture_proj = pd.DataFrame([fixture_row("2026-02-10", 7, "ARS", "CHE", True)])
    problems = fp.check_readiness(history, minutes_df, fixture_proj, target_gw=7)
    assert problems == []


def test_determine_target_gw_picks_soonest():
    fixture_proj = pd.DataFrame([
        fixture_row("2026-02-10", 8, "ARS", "CHE", True),
        fixture_row("2026-02-10", 7, "LIV", "MCI", False),
    ])
    assert fp.determine_target_gw(fixture_proj) == 7


def test_determine_target_gw_none_when_no_fixtures():
    assert fp.determine_target_gw(None) is None
    assert fp.determine_target_gw(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------

def test_team_trailing_shares_computes_ratio_and_nan_when_team_total_zero():
    latest = pd.DataFrame([
        hist_row("2026-02-10", 6, True, 1, "A", "ARS", "MID", xg=3.0, xa=1.0),
        hist_row("2026-02-10", 6, True, 2, "B", "ARS", "FWD", xg=6.0, xa=0.0),
        hist_row("2026-02-10", 6, True, 3, "C", "CHE", "FWD", xg=0.0, xa=0.0),
    ])
    out = fp.team_trailing_shares(latest).set_index("ID")

    assert out.loc[1, "xG share"] == pytest.approx(3.0 / 9.0)
    assert out.loc[2, "xG share"] == pytest.approx(6.0 / 9.0)
    assert pd.isna(out.loc[3, "xG share"])  # CHE's team total is zero -- NaN, not zero


def test_build_projection_inputs_drops_players_without_fixture_coverage():
    history = pd.DataFrame([
        hist_row("2026-02-10", 6, True, 1, "A", "ARS", "MID"),
        hist_row("2026-02-10", 6, True, 2, "B", "BUR", "MID"),  # no fixture this GW
    ])
    minutes_df = pd.DataFrame([minutes_row("2026-02-10", 1), minutes_row("2026-02-10", 2)])
    fixture_proj = pd.DataFrame([fixture_row("2026-02-10", 7, "ARS", "CHE", True)])

    df = fp.build_projection_inputs(history, minutes_df, fixture_proj, target_gw=7)
    assert list(df["ID"]) == [1]


def test_defcon_adjustment_enabled_reads_verdict():
    assert fp.defcon_adjustment_enabled(None) is False
    assert fp.defcon_adjustment_enabled(pd.DataFrame()) is False
    insufficient = pd.DataFrame([{"Snapshot": "2026-01-06", "Verdict": "Insufficient data (n=0, need >= 50)."}])
    assert fp.defcon_adjustment_enabled(insufficient) is False
    null_result = pd.DataFrame([{"Snapshot": "2026-01-06", "Verdict": "Null result -- 95% CI includes zero."}])
    assert fp.defcon_adjustment_enabled(null_result) is False
    real_effect = pd.DataFrame([{"Snapshot": "2026-01-06", "Verdict": "Real effect (moderate, r=0.35...)."}])
    assert fp.defcon_adjustment_enabled(real_effect) is True


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------

def _full_row(**overrides):
    base = {
        "ID": 1, "Player": "Nailer", "Team": "ARS", "Pos": "MID", "Price": 8.0,
        "Minutes": 900, "Start probability": 1.0, "Classification": "Nailed",
        "Minutes per appearance": 90.0, "Sub rate": 0.0,
        "Starts (overall)": 10, "Bonus": 30,
        "xG share": 0.3, "xA share": 0.2, "DEFCON per 90": 0.0,
        "Expected goals for": 2.0, "Expected goals against": 1.0,
        "Clean sheet probability": 0.3,
    }
    base.update(overrides)
    return pd.Series(base)


def test_components_sum_to_total_expected_points():
    rng = np.random.default_rng(42)
    result = project_player_wrapper(_full_row(), rng)
    component_sum = sum(result["components"].values())
    assert component_sum == pytest.approx(result["expected"], abs=1e-6)


def project_player_wrapper(row, rng):
    return fp.project_player(row, rng, n_sims=20000)


def test_certain_starter_gets_full_appearance_points_and_high_confidence():
    rng = np.random.default_rng(1)
    row = _full_row(**{"Minutes per appearance": 90.0, "Start probability": 1.0})
    result = fp.project_player(row, rng, n_sims=20000)
    # p_60plus ~1 with 90 min/appearance -> appearance component should be ~2.0
    assert result["components"]["Appearance pts"] == pytest.approx(2.0, abs=0.05)
    assert result["confidence"] == "High"


def test_never_starts_gets_only_sub_points():
    rng = np.random.default_rng(2)
    row = _full_row(**{"Start probability": 0.0, "Sub rate": 1.0})
    result = fp.project_player(row, rng, n_sims=20000)
    assert result["expected"] == pytest.approx(1.0, abs=0.05)
    assert result["components"]["Goals pts"] == 0.0


def test_low_confidence_flagged_for_thin_minutes_history():
    rng = np.random.default_rng(3)
    row = _full_row(**{"Minutes": 90, "Classification": "Insufficient data"})
    result = fp.project_player(row, rng, n_sims=1000)
    assert result["confidence"] == "Low"
    assert any("trailing minutes" in n for n in result["notes"])
    assert any("thin minutes history" in n for n in result["notes"])


def test_missing_start_probability_uses_neutral_placeholder_and_flags_low_confidence():
    rng = np.random.default_rng(4)
    row = _full_row(**{"Start probability": None})
    result = fp.project_player(row, rng, n_sims=1000)
    assert result["confidence"] == "Low"
    assert result["p_start"] == pytest.approx(0.5)
    assert any("no start probability" in n for n in result["notes"])


def test_goalkeeper_gets_save_points_outfield_does_not():
    rng = np.random.default_rng(5)
    gk_row = _full_row(**{"Pos": "GKP"})
    outfield_row = _full_row(**{"Pos": "MID"})
    gk_result = fp.project_player(gk_row, rng, n_sims=5000)
    of_result = fp.project_player(outfield_row, rng, n_sims=5000)
    assert gk_result["components"]["Save pts"] > 0
    assert of_result["components"]["Save pts"] == 0.0


def test_nan_xg_share_does_not_crash_the_simulation():
    """Regression: `x or 0.0` silently lets NaN through (NaN is truthy in
    Python), which used to crash numpy.random.poisson outright for any
    early-season player whose team hadn't scored yet -- exactly the
    situation team_trailing_shares() flags as NaN, not zero."""
    rng = np.random.default_rng(8)
    row = _full_row(**{"xG share": float("nan"), "xA share": float("nan"),
                        "DEFCON per 90": float("nan"), "Bonus": float("nan"),
                        "Minutes": float("nan"), "Price": float("nan")})
    result = fp.project_player(row, rng, n_sims=1000)
    assert result["components"]["Goals pts"] == 0.0
    assert result["components"]["Assists pts"] == 0.0
    assert not any(pd.isna(v) for v in result["components"].values())


def test_goalkeeper_has_no_defcon_points():
    rng = np.random.default_rng(6)
    row = _full_row(**{"Pos": "GKP", "DEFCON per 90": 5.0})
    result = fp.project_player(row, rng, n_sims=5000)
    assert result["components"]["DEFCON pts"] == 0.0


def test_higher_defcon_rate_yields_more_defcon_points():
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    low = fp.project_player(_full_row(Pos="DEF", **{"DEFCON per 90": 2.0}), rng1, n_sims=20000)
    high = fp.project_player(_full_row(Pos="DEF", **{"DEFCON per 90": 12.0}), rng2, n_sims=20000)
    assert high["components"]["DEFCON pts"] > low["components"]["DEFCON pts"]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_update_store_same_day_rerun_replaces(tmp_path):
    table = pd.DataFrame([{
        "Snapshot": "2026-02-10", "GW": 7, "ID": 1, "Player": "A", "Team": "ARS",
        "Pos": "MID", "Price": 8.0, "Start probability": 0.9,
        "Expected points": 5.2, "Confidence": "High", "Note": "",
    }])
    out = tmp_path / "projections.csv"

    first = fp.update_store(table, out)
    second = fp.update_store(table, out)

    assert len(second) == len(first)
    assert second["Snapshot"].nunique() == 1
