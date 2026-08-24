import pandas as pd
import pytest

from core import archive


def forecast_df(**overrides):
    base = {
        "target_event": [5, 5],
        "entity_id": [101, 102],
        "entity_type": ["player", "player"],
        "metric": ["ep_next", "ep_next"],
        "value": [4.5, 2.1],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_write_forecast_raises_on_missing_required_column(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    incomplete = forecast_df()
    del incomplete["metric"]

    with pytest.raises(KeyError, match="metric"):
        archive.write_forecast("fpl_ep_next", "2026-01-06", incomplete)


def test_write_forecast_writes_to_expected_path(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    path = archive.write_forecast("fpl_ep_next", "2026-01-06", forecast_df())

    assert path == tmp_path / "2026-01-06" / "fpl_ep_next.parquet"
    assert path.exists()


def test_write_forecast_stamps_as_of_and_source_overriding_caller_values(tmp_path, monkeypatch):
    """as_of/source must come from the function's own arguments, never from
    the caller's frame -- otherwise the file path and its contents could
    silently disagree about which day or source a forecast belongs to."""
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    df = forecast_df()
    df["as_of"] = "WRONG-DATE"
    df["source"] = "wrong_source"

    path = archive.write_forecast("fpl_ep_next", "2026-01-06", df)
    result = pd.read_parquet(path)

    assert (result["as_of"] == "2026-01-06").all()
    assert (result["source"] == "fpl_ep_next").all()
    assert list(result.columns) == archive.SCHEMA_COLUMNS


def test_write_forecast_same_day_rerun_overwrites_not_appends(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    archive.write_forecast("fpl_ep_next", "2026-01-06", forecast_df())
    archive.write_forecast("fpl_ep_next", "2026-01-06",
                            forecast_df(value=[9.9, 8.8]))  # values moved intraday

    result = pd.read_parquet(tmp_path / "2026-01-06" / "fpl_ep_next.parquet")
    assert len(result) == 2  # not duplicated to 4 rows
    assert sorted(result["value"]) == [8.8, 9.9]  # latest write wins


def test_write_forecast_never_touches_a_different_days_file(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    archive.write_forecast("fpl_ep_next", "2026-01-06", forecast_df())
    archive.write_forecast("fpl_ep_next", "2026-01-07", forecast_df(value=[1.0, 2.0]))

    day1 = pd.read_parquet(tmp_path / "2026-01-06" / "fpl_ep_next.parquet")
    day2 = pd.read_parquet(tmp_path / "2026-01-07" / "fpl_ep_next.parquet")
    assert sorted(day1["value"]) == [2.1, 4.5]
    assert sorted(day2["value"]) == [1.0, 2.0]


def test_read_forecasts_returns_empty_correctly_shaped_frame_when_nothing_archived(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    result = archive.read_forecasts()
    assert result.empty
    assert list(result.columns) == archive.SCHEMA_COLUMNS


def test_read_forecasts_filters_by_source_and_as_of(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE_ROOT", tmp_path)
    archive.write_forecast("fpl_ep_next", "2026-01-06", forecast_df())
    archive.write_forecast("fpl_ep_next", "2026-01-07", forecast_df())
    archive.write_forecast("closing_odds", "2026-01-06", forecast_df(metric=["1x2", "1x2"]))

    all_rows = archive.read_forecasts()
    assert len(all_rows) == 6

    only_ep_next = archive.read_forecasts(source="fpl_ep_next")
    assert len(only_ep_next) == 4
    assert (only_ep_next["source"] == "fpl_ep_next").all()

    only_day1 = archive.read_forecasts(as_of="2026-01-06")
    assert len(only_day1) == 4
    assert (only_day1["as_of"] == "2026-01-06").all()

    exact = archive.read_forecasts(source="fpl_ep_next", as_of="2026-01-07")
    assert len(exact) == 2
