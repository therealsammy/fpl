import pandas as pd
import pytest

from collectors import fpl_rivals as fr


def picks(entries):
    """entries: list of (player_id, is_captain, is_vice) tuples."""
    return {"picks": [{"element": pid, "is_captain": cap, "is_vice_captain": vc,
                        "multiplier": 2 if cap else 1}
                       for pid, cap, vc in entries]}


def test_fetch_standings_paginates(monkeypatch):
    calls = []

    def fake_get(path, params=None):
        calls.append(params["page_standings"])
        if params["page_standings"] == 1:
            return {"league": {"name": "The League"},
                     "standings": {"has_next": True, "results": [{"entry": 1}, {"entry": 2}]}}
        return {"league": {"name": "The League"},
                 "standings": {"has_next": False, "results": [{"entry": 3}]}}

    monkeypatch.setattr(fr, "get", fake_get)
    name, results = fr.fetch_standings(999)

    assert name == "The League"
    assert [r["entry"] for r in results] == [1, 2, 3]
    assert calls == [1, 2]


def test_fetch_standings_stops_on_missing_league(monkeypatch):
    monkeypatch.setattr(fr, "get", lambda path, params=None: None)
    name, results = fr.fetch_standings(999)
    assert name is None
    assert results == []


def test_build_rivals_rows_skips_managers_without_picks():
    standings = [
        {"entry": 1, "player_name": "Me", "entry_name": "TeamA", "rank": 1, "total": 100},
        {"entry": 2, "player_name": "Rival", "entry_name": "TeamB", "rank": 2, "total": 90},
    ]
    picks_by_entry = {
        1: picks([(10, True, False), (11, False, True)]),
        2: None,  # deadline for this GW hasn't passed for this manager's view
    }

    rows, skipped = fr.build_rivals_rows(standings, picks_by_entry, gw=3, snapshot_date="2026-01-06")

    assert set(rows["Entry ID"]) == {1}
    assert len(rows) == 2
    assert skipped == [(2, "Rival")]


def _long_df():
    """3 managers (1=me, 2 & 3 rivals). Player 10 owned by all, captained by
    me and rival 2. Player 20 only I own. Player 30 only rivals own."""
    rows = [
        {"Entry ID": 1, "Manager": "Me", "Player ID": 10, "Captain": True},
        {"Entry ID": 1, "Manager": "Me", "Player ID": 20, "Captain": False},
        {"Entry ID": 2, "Manager": "Rival2", "Player ID": 10, "Captain": True},
        {"Entry ID": 2, "Manager": "Rival2", "Player ID": 30, "Captain": False},
        {"Entry ID": 3, "Manager": "Rival3", "Player ID": 10, "Captain": False},
        {"Entry ID": 3, "Manager": "Rival3", "Player ID": 30, "Captain": True},
    ]
    for r in rows:
        r.setdefault("Snapshot", "2026-01-06")
        r.setdefault("GW", 3)
    return pd.DataFrame(rows)


def test_effective_ownership_combines_owned_and_captained_pct():
    out = fr.compute_effective_ownership(_long_df()).set_index("Player ID")

    assert out.loc[10, "Owned %"] == pytest.approx(100.0)
    assert out.loc[10, "Captained %"] == pytest.approx(200 / 3, rel=1e-2)
    assert out.loc[20, "Owned %"] == pytest.approx(100 / 3, rel=1e-2)
    assert out.loc[20, "Captained by"] == 0


def test_differentials_unique_to_me_and_missing_for_me():
    unique_to_me, missing_for_me = fr.compute_differentials(_long_df(), my_entry_id=1)

    assert list(unique_to_me["Player ID"]) == [20]
    assert list(missing_for_me["Player ID"]) == [30]
    assert missing_for_me.iloc[0]["Owned by rivals"] == 2


def test_captain_divergence_when_matching_field():
    df = _long_df()
    # Everyone but me captains 30; I captain 10 -- should diverge.
    result = fr.compute_captain_divergence(df, my_entry_id=1)
    assert result["my_captain"] == 10
    assert result["field_top_captain"] == 10  # 10 has 2 captains (me + rival2) vs 30's 1
    assert result["diverges"] is False


def test_captain_divergence_when_diverging():
    rows = [
        {"Entry ID": 1, "Manager": "Me", "Player ID": 10, "Captain": True},
        {"Entry ID": 2, "Manager": "Rival2", "Player ID": 30, "Captain": True},
        {"Entry ID": 3, "Manager": "Rival3", "Player ID": 30, "Captain": True},
    ]
    df = pd.DataFrame(rows)
    result = fr.compute_captain_divergence(df, my_entry_id=1)

    assert result["my_captain"] == 10
    assert result["field_top_captain"] == 30
    assert result["field_top_count"] == 2
    assert result["diverges"] is True


def test_captain_divergence_when_my_picks_missing():
    rows = [{"Entry ID": 2, "Manager": "Rival2", "Player ID": 30, "Captain": True}]
    df = pd.DataFrame(rows)
    result = fr.compute_captain_divergence(df, my_entry_id=1)

    assert result["my_captain"] is None
    assert result["diverges"] is None


def test_update_store_same_day_rerun(tmp_path):
    rows = pd.DataFrame([
        {"Snapshot": "2026-01-06", "GW": 3, "Entry ID": 1, "Manager": "Me",
         "Team name": "TeamA", "Rank": 1, "Total points": 100,
         "Player ID": 10, "Captain": True, "Vice captain": False, "Multiplier": 2},
    ])
    out = tmp_path / "fpl_rivals.csv"

    first = fr.update_store(rows, out)
    second = fr.update_store(rows, out)

    assert len(second) == len(first)
    assert second["Snapshot"].nunique() == 1


def test_fetch_picks_for_entries_handles_404(monkeypatch):
    monkeypatch.setattr(fr, "REQUEST_DELAY", 0)

    def fake_get(path, params=None):
        if path.startswith("entry/1/"):
            return picks([(10, True, False)])
        return None  # entry 2: deadline not passed / 404

    monkeypatch.setattr(fr, "get", fake_get)
    result = fr.fetch_picks_for_entries([1, 2], gw=3)

    assert result[1] is not None
    assert result[2] is None
