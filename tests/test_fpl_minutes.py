import pandas as pd
import pytest

import fpl_minutes as fm


def row(snapshot, gw, finished, pid, player, team, pos, minutes, starts, chance=None):
    return {
        "Snapshot": snapshot, "GW": gw, "GW finished": finished,
        "ID": pid, "Player": player, "Team": team, "Pos": pos,
        "Minutes": minutes, "Starts": starts, "Chance next GW": chance,
    }


DATES = ["2026-01-06", "2026-01-13", "2026-01-20", "2026-01-27", "2026-02-03"]


def build_five_gw_history():
    """
    Five finished gameweeks, four players with distinct playing patterns:
    nailed every week, genuine rotation (starts about half, subs the rest),
    fringe bench (one start, occasional late sub), and a pure benchwarmer
    (zero minutes throughout). Cumulative Starts/Minutes mirror how the FPL
    API actually reports them -- season-to-date totals, not per-GW deltas.
    """
    rows = []
    # Nailed: starts and plays 90 every week.
    cum_s, cum_m = 0, 0
    for gw, date in zip(range(1, 6), DATES):
        cum_s += 1
        cum_m += 90
        rows.append(row(date, gw, True, 1, "Nailer", "ARS", "MID", cum_m, cum_s, 100))

    # Rotation: start / sub / start / sub / start.
    starts_deltas = [1, 0, 1, 0, 1]
    minutes_deltas = [90, 20, 90, 10, 90]
    cum_s, cum_m = 0, 0
    for gw, date, ds, dm in zip(range(1, 6), DATES, starts_deltas, minutes_deltas):
        cum_s += ds
        cum_m += dm
        rows.append(row(date, gw, True, 2, "Rotator", "ARS", "MID", cum_m, cum_s, 60))

    # Bench: one start, two cameo subs, otherwise unused.
    starts_deltas = [0, 0, 0, 1, 0]
    minutes_deltas = [0, 15, 0, 30, 5]
    cum_s, cum_m = 0, 0
    for gw, date, ds, dm in zip(range(1, 6), DATES, starts_deltas, minutes_deltas):
        cum_s += ds
        cum_m += dm
        rows.append(row(date, gw, True, 3, "Fringe", "ARS", "DEF", cum_m, cum_s, 25))

    # Benchwarmer: never plays.
    for gw, date in zip(range(1, 6), DATES):
        rows.append(row(date, gw, True, 4, "Ghost", "ARS", "FWD", 0, 0, None))

    return pd.DataFrame(rows)


def test_single_snapshot_returns_insufficient_data():
    history = pd.DataFrame([
        row(DATES[0], 1, True, 1, "Nailer", "ARS", "MID", 90, 1, 100),
        row(DATES[0], 1, True, 2, "Rotator", "ARS", "MID", 90, 1, 60),
    ])

    table = fm.compute_minutes_table(history)

    assert (table["Classification"] == "Insufficient data").all()
    assert table["Start probability"].isna().all()
    assert "need at least 3" in table["Note"].iloc[0]


def test_unfinished_gameweek_does_not_count_as_settled():
    """Mirrors the real repo's current state: one snapshot, GW not finished."""
    history = pd.DataFrame([
        row(DATES[0], 1, False, 1, "Nailer", "ARS", "MID", 45, 0, 100),
    ])

    table = fm.compute_minutes_table(history)

    assert table["Classification"].iloc[0] == "Insufficient data"
    assert "only 0 finished-gameweek" in table["Note"].iloc[0]


def test_classification_matches_playing_pattern():
    history = build_five_gw_history()
    table = fm.compute_minutes_table(history).set_index("Player")

    assert table.loc["Nailer", "Classification"] == "Nailed"
    assert table.loc["Rotator", "Classification"] == "Rotation risk"
    assert table.loc["Fringe", "Classification"] == "Bench"
    assert table.loc["Ghost", "Classification"] == "Benchwarmer"

    # Nailed player should have a high start probability; benchwarmer near zero.
    assert table.loc["Nailer", "Start probability"] > 0.9
    assert table.loc["Ghost", "Start probability"] == 0.0

    # Components should be visible, not just the blended total.
    assert table.loc["Rotator", "Minutes per appearance"] == 60.0


def test_new_signing_gets_per_player_insufficient_data_not_global_average():
    """
    A player who only joined the data at GW4 has real minutes but too few
    snapshots of their own to trust -- even though the dataset as a whole
    has plenty of history. This must not fall back to a league average.
    """
    history = build_five_gw_history()
    newcomer = pd.DataFrame([
        row(DATES[3], 4, True, 5, "NewSigning", "CHE", "FWD", 90, 1, 100),
        row(DATES[4], 5, True, 5, "NewSigning", "CHE", "FWD", 180, 2, 100),
    ])
    history = pd.concat([history, newcomer], ignore_index=True)

    table = fm.compute_minutes_table(history).set_index("Player")

    assert table.loc["NewSigning", "Classification"] == "Insufficient data"
    assert pd.isna(table.loc["NewSigning", "Start probability"])
    assert "2 snapshot(s)" in table.loc["NewSigning", "Note"]
    # Established players are unaffected by the newcomer joining the dataset.
    assert table.loc["Nailer", "Classification"] == "Nailed"


def test_player_absent_from_latest_snapshot_is_dropped_not_crashed():
    """A player who left the league mid-season should simply not appear in
    the output -- not raise, and not carry stale data forward."""
    history = build_five_gw_history()
    departed = pd.DataFrame([
        row(DATES[0], 1, True, 6, "Departed", "BUR", "DEF", 90, 1, 100),
        row(DATES[1], 2, True, 6, "Departed", "BUR", "DEF", 180, 2, 100),
    ])
    history = pd.concat([history, departed], ignore_index=True)

    table = fm.compute_minutes_table(history)

    assert 6 not in table["ID"].values


def test_same_day_rerun_replaces_rather_than_duplicates(tmp_path):
    history = build_five_gw_history()
    table = fm.compute_minutes_table(history)
    out = tmp_path / "fpl_minutes.csv"

    first = fm.update_store(table, gw=5, path=out)
    second = fm.update_store(table, gw=5, path=out)

    assert len(second) == len(first)
    assert second["Snapshot"].nunique() == 1
    dupes = second.duplicated(subset=["Snapshot", "ID"])
    assert not dupes.any()
