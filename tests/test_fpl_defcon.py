import math

import pandas as pd
import pytest

from models import defcon as fd


def row(snapshot, gw, finished, pid, player, team, pos, minutes, defcon):
    return {"Snapshot": snapshot, "GW": gw, "GW finished": finished,
            "ID": pid, "Player": player, "Team": team, "Pos": pos,
            "Minutes": minutes, "DEFCON": defcon}


def fixture(team, gw, fdr, home, opponent, double=False):
    return {"team": team, "gw": gw, "fdr": fdr, "home": home,
            "opponent": opponent, "double": double}


def lookup_from(fixtures):
    return {(f["team"], f["gw"]): {"fdr": f["fdr"], "home": f["home"],
                                    "opponent": f["opponent"], "double": f["double"]}
            for f in fixtures}


def test_goalkeepers_are_excluded():
    history = pd.DataFrame([
        row("2026-01-06", 1, True, 1, "Keeper", "ARS", "GKP", 90, 5),
        row("2026-01-13", 2, True, 1, "Keeper", "ARS", "GKP", 180, 10),
    ])
    lookup = lookup_from([fixture("ARS", 2, 3, True, "CHE")])
    dataset = fd.build_defcon_dataset(history, lookup)
    assert dataset.empty


def test_double_gameweek_is_excluded():
    history = pd.DataFrame([
        row("2026-01-06", 1, True, 1, "Def", "ARS", "DEF", 90, 2),
        row("2026-01-13", 2, True, 1, "Def", "ARS", "DEF", 270, 6),  # 2 games worth of minutes
    ])
    lookup = lookup_from([fixture("ARS", 2, 3, True, "CHE", double=True)])
    dataset = fd.build_defcon_dataset(history, lookup)
    assert dataset.empty


def test_blank_gameweek_is_excluded():
    history = pd.DataFrame([
        row("2026-01-06", 1, True, 1, "Def", "ARS", "DEF", 90, 2),
        row("2026-01-13", 2, True, 1, "Def", "ARS", "DEF", 180, 4),
    ])
    dataset = fd.build_defcon_dataset(history, fixtures_lookup={})  # no fixture data at all
    assert dataset.empty


def test_low_minutes_period_is_excluded():
    history = pd.DataFrame([
        row("2026-01-06", 1, True, 1, "Def", "ARS", "DEF", 90, 2),
        row("2026-01-13", 2, True, 1, "Def", "ARS", "DEF", 100, 3),  # only 10 min that period
    ])
    lookup = lookup_from([fixture("ARS", 2, 3, True, "CHE")])
    dataset = fd.build_defcon_dataset(history, lookup, min_period_minutes=30)
    assert dataset.empty


def test_multi_gw_gap_is_excluded():
    """A missed snapshot spanning 2 gameweeks can't be attributed to one
    opponent's territory, so it should be dropped, not guessed at."""
    history = pd.DataFrame([
        row("2026-01-06", 1, True, 1, "Def", "ARS", "DEF", 90, 2),
        row("2026-01-20", 3, True, 1, "Def", "ARS", "DEF", 270, 8),  # GW1 -> GW3, no GW2 snapshot
    ])
    lookup = lookup_from([fixture("ARS", 3, 3, True, "CHE")])
    dataset = fd.build_defcon_dataset(history, lookup)
    assert dataset.empty


def test_clean_period_produces_a_row_with_correct_per90():
    history = pd.DataFrame([
        row("2026-01-06", 1, True, 1, "Def", "ARS", "DEF", 90, 2),
        row("2026-01-13", 2, True, 1, "Def", "ARS", "DEF", 180, 5),  # +90 min, +3 DEFCON
    ])
    lookup = lookup_from([fixture("ARS", 2, 4, False, "CHE")])
    dataset = fd.build_defcon_dataset(history, lookup)

    assert len(dataset) == 1
    r = dataset.iloc[0]
    assert r["DEFCON per90 (period)"] == pytest.approx(3.0)
    assert r["Opponent FDR"] == 4
    assert bool(r["Home"]) is False


def test_pearson_ci_returns_none_below_minimum_n():
    result = fd.pearson_ci([1, 2, 3], [1, 2, 3])
    assert result["r"] is None
    assert result["n"] == 3


def test_pearson_ci_recovers_perfect_correlation_with_noise():
    x = list(range(20))
    y = [v * 2 + (0.01 if i % 2 == 0 else -0.01) for i, v in enumerate(x)]
    result = fd.pearson_ci(x, y)
    assert result["r"] > 0.99
    assert result["lo"] > 0.9  # tight CI, clearly excludes zero
    assert result["n"] == 20


def test_verdict_reports_insufficient_data_below_threshold():
    stats = {"r": 0.8, "lo": 0.5, "hi": 0.95, "n": 10}
    verdict = fd.verdict_for_correlation(stats, min_n=50)
    assert "Insufficient data" in verdict


def test_verdict_reports_null_result_when_ci_crosses_zero():
    stats = {"r": 0.05, "lo": -0.1, "hi": 0.2, "n": 100}
    verdict = fd.verdict_for_correlation(stats, min_n=50)
    assert "Null result" in verdict
    assert "NOT applying" in verdict


def test_verdict_reports_real_effect_when_ci_excludes_zero_and_meaningful():
    stats = {"r": 0.35, "lo": 0.15, "hi": 0.5, "n": 100}
    verdict = fd.verdict_for_correlation(stats, min_n=50)
    assert "Real effect" in verdict
    assert "moderate" in verdict


def test_home_away_split_computes_difference_with_ci():
    dataset = pd.DataFrame({
        "DEFCON per90 (period)": [5.0, 5.2, 4.8, 5.1] + [2.0, 2.1, 1.9, 2.2],
        "Home": [True, True, True, True, False, False, False, False],
    })
    home_stats, away_stats, diff = fd.home_away_split(dataset)

    assert home_stats["n"] == 4
    assert away_stats["n"] == 4
    assert diff is not None
    assert diff["mean_diff"] > 2.5  # home clearly higher in this synthetic case


def test_update_report_store_same_day_rerun_replaces(tmp_path):
    out = tmp_path / "defcon_report.csv"
    row1 = {"Snapshot": "2026-01-13", "GW": 2, "n": 0, "Verdict": "Insufficient data"}
    first = fd.update_report_store(row1, out)
    second = fd.update_report_store(row1, out)

    assert len(second) == len(first) == 1
