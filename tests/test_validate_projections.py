import pandas as pd
import pytest

import validate_projections as vp


def hist_row(snapshot, gw, finished, pid, pos, gw_pts=None, starts=0, exp_pts_next=None):
    return {"Snapshot": snapshot, "GW": gw, "GW finished": finished, "ID": pid,
            "Pos": pos, "GW pts": gw_pts, "Starts": starts, "Exp pts next": exp_pts_next}


def proj_row(gw, pid, expected_points):
    return {"GW": gw, "ID": pid, "Expected points": expected_points}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def test_actual_points_for_gw_none_when_not_finished():
    history = pd.DataFrame([hist_row("2026-01-13", 2, False, 1, "MID", gw_pts=8, starts=1)])
    actual, pos = vp.actual_points_for_gw(history, 2)
    assert actual is None and pos is None


def test_actual_points_for_gw_returns_points_and_position():
    history = pd.DataFrame([
        hist_row("2026-01-13", 2, True, 1, "MID", gw_pts=8, starts=1),
        hist_row("2026-01-13", 2, True, 2, "FWD", gw_pts=2, starts=1),
    ])
    actual, pos = vp.actual_points_for_gw(history, 2)
    assert actual[1] == 8
    assert pos[2] == "FWD"


def test_ep_next_baseline_requires_clean_immediate_predecessor():
    # GW1 snapshot exists, but GW2 is being validated with no GW1->GW2 gap: OK.
    history = pd.DataFrame([hist_row("2026-01-06", 1, True, 1, "MID", exp_pts_next=5.0)])
    baseline = vp.ep_next_baseline_for_gw(history, 2)
    assert baseline[1] == 5.0

    # GW1 snapshot exists but we're validating GW3 (GW2 missing) -- stale, refuse.
    baseline_gap = vp.ep_next_baseline_for_gw(history, 3)
    assert baseline_gap is None


def test_ep_next_baseline_none_with_no_prior_snapshots():
    history = pd.DataFrame([hist_row("2026-01-06", 1, True, 1, "MID", exp_pts_next=5.0)])
    assert vp.ep_next_baseline_for_gw(history, 1) is None


def test_starters_for_gw_uses_clean_single_gw_starts_delta():
    history = pd.DataFrame([
        hist_row("2026-01-06", 1, True, 1, "MID", starts=3),   # cumulative before GW2
        hist_row("2026-01-06", 1, True, 2, "DEF", starts=1),
        hist_row("2026-01-13", 2, True, 1, "MID", starts=4),   # started GW2 (+1)
        hist_row("2026-01-13", 2, True, 2, "DEF", starts=1),   # did NOT start GW2 (+0)
    ])
    starters = vp.starters_for_gw(history, 2)
    assert starters == {1}


def test_starters_for_gw_none_when_gap_in_settled_history():
    history = pd.DataFrame([
        hist_row("2026-01-06", 1, True, 1, "MID", starts=3),
        hist_row("2026-01-20", 3, True, 1, "MID", starts=5),   # GW2 missing entirely
    ])
    assert vp.starters_for_gw(history, 3) is None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_spearman_none_below_min_n_or_zero_variance():
    assert vp.spearman_rank_corr(pd.Series([1.0]), pd.Series([1.0])) is None
    assert vp.spearman_rank_corr(pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 2.0, 3.0])) is None


def test_spearman_perfect_rank_agreement():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([10.0, 20.0, 30.0, 40.0])
    assert vp.spearman_rank_corr(x, y) == pytest.approx(1.0)


def test_rmse_zero_for_identical_series():
    x = pd.Series([1.0, 2.0, 3.0])
    assert vp.rmse(x, x) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Full gameweek validation
# ---------------------------------------------------------------------------

def _build_scenario(n_players=15, gw=2, my_better=True):
    """GW1 -> GW2, n_players starters, with actual points, my projection and
    ep_next crafted so 'mine' either clearly outranks or underranks ep_next."""
    rows = []
    proj_rows = []
    for i in range(1, n_players + 1):
        pos = ["GKP", "DEF", "MID", "FWD"][i % 4]
        rows.append(hist_row("2026-01-06", 1, True, i, pos, starts=0, exp_pts_next=float(i % 5)))
        actual = float(i)  # actual points rank exactly with player index
        rows.append(hist_row("2026-01-13", 2, True, i, pos, gw_pts=actual, starts=1))
        my_pred = actual if my_better else float(n_players - i)  # matches or inverts rank
        proj_rows.append(proj_row(gw, i, my_pred))
    return pd.DataFrame(rows), pd.DataFrame(proj_rows)


def test_validate_gameweek_scores_mine_better_when_it_tracks_actual_rank():
    history, projections = _build_scenario(n_players=15, my_better=True)
    result = vp.validate_gameweek(history, projections, target_gw=2)
    assert result is not None
    o = result["overall"]
    assert o["n"] == 15
    assert o["mine_spearman"] > o["epnext_spearman"]


def test_validate_gameweek_none_when_projection_missing_for_gw():
    history, _ = _build_scenario(n_players=15)
    empty_proj = pd.DataFrame(columns=["GW", "ID", "Expected points"])
    assert vp.validate_gameweek(history, empty_proj, target_gw=2) is None


def test_validate_gameweek_none_when_gw_not_finished():
    history, projections = _build_scenario(n_players=15)
    history.loc[history["GW"] == 2, "GW finished"] = False
    assert vp.validate_gameweek(history, projections, target_gw=2) is None


def test_validate_gameweek_position_breakdown_insufficient_below_min_sample():
    history, projections = _build_scenario(n_players=15)
    result = vp.validate_gameweek(history, projections, target_gw=2)
    # 15 players split 4 ways -> each position has ~3-4, below MIN_POSITION_SAMPLE (5)
    for pos, stats in result["by_position"].items():
        assert stats["mine_spearman"] is None
        assert stats["n"] < vp.MIN_POSITION_SAMPLE


def test_validate_gameweek_below_min_gw_sample_yields_null_stats_not_none():
    history, projections = _build_scenario(n_players=5)  # below MIN_GW_SAMPLE (10)
    result = vp.validate_gameweek(history, projections, target_gw=2)
    assert result is not None  # still recorded -- it's a permanent fact about this GW
    assert result["overall"]["n"] == 5
    assert result["overall"]["mine_spearman"] is None


# ---------------------------------------------------------------------------
# Store idempotency (forever, not per-day)
# ---------------------------------------------------------------------------

def test_update_store_never_reprocesses_a_gameweek(tmp_path):
    out = tmp_path / "validate_report.csv"
    first_run = pd.DataFrame([
        {"Snapshot": "2026-01-13", "GW": 2, "Position": "All", "n": 15,
         "mine_spearman": 0.9, "mine_rmse": 1.0, "epnext_spearman": 0.5, "epnext_rmse": 2.0},
    ])
    vp.update_store(first_run, out)

    # A later run recomputes the SAME gameweek (e.g. re-run same day) --
    # the original row must survive unchanged, not be duplicated or overwritten.
    rerun = pd.DataFrame([
        {"Snapshot": "2026-01-14", "GW": 2, "Position": "All", "n": 15,
         "mine_spearman": 0.1, "mine_rmse": 9.0, "epnext_spearman": 0.5, "epnext_rmse": 2.0},
    ])
    combined = vp.update_store(rerun, out)

    assert len(combined) == 1
    assert combined.iloc[0]["Snapshot"] == "2026-01-13"  # first result kept, not replaced
    assert combined.iloc[0]["mine_spearman"] == 0.9


# ---------------------------------------------------------------------------
# Aggregate verdict
# ---------------------------------------------------------------------------

def test_verdict_insufficient_below_min_weeks():
    report = pd.DataFrame([
        {"GW": g, "Position": "All", "mine_spearman": 0.6, "epnext_spearman": 0.4}
        for g in range(1, 4)
    ])
    assert "Insufficient data" in vp.overall_verdict(report, min_weeks=6)


def test_verdict_beats_baseline_when_mine_wins_on_average():
    report = pd.DataFrame([
        {"GW": g, "Position": "All", "mine_spearman": 0.6, "epnext_spearman": 0.4}
        for g in range(1, 7)
    ])
    verdict = vp.overall_verdict(report, min_weeks=6)
    assert "Beats ep_next" in verdict
    assert "won 6/6" in verdict


def test_verdict_loses_to_baseline_when_mine_underperforms():
    report = pd.DataFrame([
        {"GW": g, "Position": "All", "mine_spearman": 0.2, "epnext_spearman": 0.5}
        for g in range(1, 7)
    ])
    verdict = vp.overall_verdict(report, min_weeks=6)
    assert "LOSES to ep_next" in verdict
    assert "Do not act on these projections" in verdict
