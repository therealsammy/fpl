import pandas as pd
import pytest

from models import signals as fs


def row(snapshot, gw, pid, player, team, pens=None, corners=None, fk=None,
        minutes=90, form=5.0, rise_fall=None, likelihood=None, price=5.0):
    return {
        "Snapshot": snapshot, "GW": gw, "ID": pid, "Player": player, "Team": team,
        "Pens order": pens, "Corners/IFK order": corners, "Direct FK order": fk,
        "Minutes": minutes, "Form": form, "Price": price,
        "Rise/fall %": rise_fall, "Change likelihood": likelihood,
    }


def test_single_snapshot_returns_note_not_a_diff():
    history = pd.DataFrame([row("2026-01-06", 1, 1, "A", "ARS", pens=1)])
    changes, note = fs.detect_setpiece_changes(history)
    assert changes.empty
    assert "only 1 snapshot" in note


def test_detects_lost_gained_promoted_demoted():
    history = pd.DataFrame([
        # Player 1: loses penalty duty entirely.
        row("2026-01-06", 1, 1, "Loser", "ARS", pens=1),
        row("2026-01-13", 2, 1, "Loser", "ARS", pens=None),
        # Player 2: gains corner duty from nothing.
        row("2026-01-06", 1, 2, "Gainer", "ARS", corners=None),
        row("2026-01-13", 2, 2, "Gainer", "ARS", corners=1),
        # Player 3: promoted from 2nd choice to 1st choice penalty taker.
        row("2026-01-06", 1, 3, "Promo", "CHE", pens=2),
        row("2026-01-13", 2, 3, "Promo", "CHE", pens=1),
        # Player 4: demoted from 1st to 2nd choice free-kick taker.
        row("2026-01-06", 1, 4, "Demo", "CHE", fk=1),
        row("2026-01-13", 2, 4, "Demo", "CHE", fk=2),
        # Player 5: unchanged -- should not appear.
        row("2026-01-06", 1, 5, "Same", "MCI", pens=1),
        row("2026-01-13", 2, 5, "Same", "MCI", pens=1),
        # Baseline: establishes corners/FK columns had real prior data at
        # all, so the "no prior data" skip doesn't swallow Promo/Demo's cases.
        row("2026-01-06", 1, 6, "Filler", "MCI", pens=5, corners=5, fk=5),
        row("2026-01-13", 2, 6, "Filler", "MCI", pens=5, corners=5, fk=5),
    ])

    changes, note = fs.detect_setpiece_changes(history)
    assert note is None
    by_player = changes.set_index("Player")["Direction"].to_dict()

    assert by_player["Loser"] == "Lost duty"
    assert by_player["Gainer"] == "Gained duty"
    assert by_player["Promo"] == "Promoted (higher priority)"
    assert by_player["Demo"] == "Demoted (lower priority)"
    assert "Same" not in by_player


def test_new_player_compares_against_no_prior_data():
    """A player who only appears in the latest snapshot (new signing) should
    read as 'gained duty', not crash on a missing prior row."""
    history = pd.DataFrame([
        row("2026-01-06", 1, 1, "Existing", "ARS", pens=1, corners=5, fk=5),
        row("2026-01-13", 2, 1, "Existing", "ARS", pens=1, corners=5, fk=5),
        row("2026-01-13", 2, 2, "NewSigning", "CHE", pens=1),
    ])
    changes, note = fs.detect_setpiece_changes(history)
    assert note is None
    row_ = changes[changes["Player"] == "NewSigning"].iloc[0]
    assert row_["Direction"] == "Gained duty"


def test_column_with_no_prior_data_is_skipped_not_flagged_as_mass_change():
    """
    If a column was only just added to the store (e.g. SNAPSHOT_FIELDS was
    extended), the previous snapshot has it entirely NaN for everyone. That
    must not read as every single player 'gaining duty' -- it's a schema
    change, not a real signal. Other columns with real prior data should
    still diff normally.
    """
    history = pd.DataFrame([
        row("2026-01-06", 1, 1, "A", "ARS", pens=None, corners=1),
        row("2026-01-06", 1, 2, "B", "ARS", pens=None, corners=2),
        row("2026-01-13", 2, 1, "A", "ARS", pens=1, corners=1),
        row("2026-01-13", 2, 2, "B", "ARS", pens=2, corners=None),
    ])

    changes, note = fs.detect_setpiece_changes(history)

    assert "Pens order" in note
    assert (changes["Metric"] == "Pens order").sum() == 0
    # Corners/IFK order had real prior data, so B's genuine change still shows up.
    assert (changes["Metric"] == "Corners/IFK order").sum() == 1
    assert changes.iloc[0]["Direction"] == "Lost duty"


def test_fixture_adjusted_form_divides_by_opponent_fdr():
    latest = pd.DataFrame([
        row("2026-01-13", 2, 1, "Hard", "ARS", minutes=90, form=6.0),
        row("2026-01-13", 2, 2, "Easy", "CHE", minutes=90, form=6.0),
        row("2026-01-13", 2, 3, "Unused", "MCI", minutes=0, form=0.0),
    ])
    fdr_by_team = {"ARS": 5.0, "CHE": 1.0}  # MCI missing -> blank gameweek
    opponent_by_team = {"ARS": "MCI (H)", "CHE": "BUR (A)"}

    out = fs.fixture_adjusted_form(latest, fdr_by_team, opponent_by_team).set_index("Player")

    assert out.loc["Hard", "Fixture-adjusted form"] == pytest.approx(1.2)
    assert out.loc["Easy", "Fixture-adjusted form"] == pytest.approx(6.0)
    # Easier fixture (lower FDR) yields a higher adjusted score for equal raw form.
    assert out.loc["Easy", "Fixture-adjusted form"] > out.loc["Hard", "Fixture-adjusted form"]
    # Zero-minute players are excluded entirely, not zeroed out.
    assert "Unused" not in out.index


def test_fixture_adjusted_form_handles_blank_gameweek():
    latest = pd.DataFrame([row("2026-01-13", 2, 1, "Blank", "BUR", minutes=90, form=4.0)])
    out = fs.fixture_adjusted_form(latest, fdr_by_team={}, opponent_by_team={})
    assert out.iloc[0]["Fixture-adjusted form"] is None
    assert out.iloc[0]["Note"] == "blank gameweek"


def test_price_momentum_watch_filters_and_labels_direction():
    latest = pd.DataFrame([
        row("2026-01-13", 2, 1, "Riser", "ARS", rise_fall=57.4, likelihood=3),
        row("2026-01-13", 2, 2, "Faller", "CHE", rise_fall=-59.6, likelihood=-3),
        row("2026-01-13", 2, 3, "Stable", "MCI", rise_fall=5.0, likelihood=0),
        row("2026-01-13", 2, 4, "NoData", "TOT", rise_fall=None, likelihood=None),
    ])

    out = fs.price_momentum_watch(latest, threshold=40).set_index("Player")

    assert set(out.index) == {"Riser", "Faller"}
    assert out.loc["Riser", "Direction"] == "Rise watch"
    assert out.loc["Faller", "Direction"] == "Fall watch"


def test_update_store_same_day_rerun_replaces(tmp_path):
    table = pd.DataFrame([{
        "Snapshot": "2026-01-13", "GW": 2, "Signal": "Set-piece change",
        "ID": 1, "Player": "A", "Team": "ARS", "Metric": "Pens order",
        "Old value": 1, "New value": None, "Note": "Lost duty",
    }])
    out = tmp_path / "signals.csv"

    first = fs.update_store(table, out)
    second = fs.update_store(table, out)

    assert len(second) == len(first)
    assert second["Snapshot"].nunique() == 1
