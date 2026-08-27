import json

from collectors import fpl_tracker


def test_sync_squad_json_writes_sorted_ids(tmp_path, monkeypatch):
    path = tmp_path / "squad.json"
    monkeypatch.setattr(fpl_tracker, "SQUAD_PATH", path)

    fpl_tracker.sync_squad_json([388, 4, 226])

    assert json.loads(path.read_text()) == [4, 226, 388]


def test_sync_squad_json_overwrites_a_previous_squad_after_transfers(tmp_path, monkeypatch):
    """squad.json is current STATE, not an archive -- a new gameweek's
    picks must replace last gameweek's, not merge with them."""
    path = tmp_path / "squad.json"
    path.write_text(json.dumps([1, 2, 3]))
    monkeypatch.setattr(fpl_tracker, "SQUAD_PATH", path)

    fpl_tracker.sync_squad_json([4, 5, 6])

    assert json.loads(path.read_text()) == [4, 5, 6]


def test_sync_squad_json_skips_write_when_picks_are_empty(tmp_path, monkeypatch):
    """An empty picks list means the API had nothing yet (pre-deadline or
    a transient hiccup), not that the squad is actually empty -- a real
    previously-saved squad must survive that run untouched."""
    path = tmp_path / "squad.json"
    path.write_text(json.dumps([1, 2, 3]))
    monkeypatch.setattr(fpl_tracker, "SQUAD_PATH", path)

    fpl_tracker.sync_squad_json([])

    assert json.loads(path.read_text()) == [1, 2, 3]


def test_sync_squad_json_does_not_create_a_file_when_there_was_none_and_picks_are_empty(tmp_path, monkeypatch):
    path = tmp_path / "squad.json"
    monkeypatch.setattr(fpl_tracker, "SQUAD_PATH", path)

    fpl_tracker.sync_squad_json([])

    assert not path.exists()
