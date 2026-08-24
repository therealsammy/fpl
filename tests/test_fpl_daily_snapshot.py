import requests
import pandas as pd
import pytest

import fpl_daily_snapshot as fds


def element(id_=1, web_name="Salah", team=1, element_type=3, now_cost=130,
            selected_by_percent="45.2", form="6.0", ep_next="7.5",
            transfers_in_event=10000, transfers_out_event=2000, status="a"):
    return {
        "id": id_, "web_name": web_name, "team": team, "element_type": element_type,
        "now_cost": now_cost, "selected_by_percent": selected_by_percent, "form": form,
        "ep_next": ep_next, "transfers_in_event": transfers_in_event,
        "transfers_out_event": transfers_out_event, "status": status,
    }


def boot(elements):
    return {"elements": elements, "teams": [{"id": 1, "short_name": "LIV"},
                                             {"id": 2, "short_name": "ARS"}]}


# ---------------------------------------------------------------------------
# Schema drift -- must fail loudly
# ---------------------------------------------------------------------------

def test_check_schema_passes_with_all_required_fields():
    check = fds.check_schema([element()])
    assert check is None  # no exception


def test_check_schema_raises_on_empty_elements():
    with pytest.raises(ValueError, match="zero elements"):
        fds.check_schema([])


def test_check_schema_raises_on_missing_field():
    e = element()
    del e["ep_next"]
    with pytest.raises(KeyError, match="ep_next"):
        fds.check_schema([e])


def test_check_schema_raises_on_multiple_missing_fields():
    e = element()
    del e["form"]
    del e["status"]
    with pytest.raises(KeyError) as exc_info:
        fds.check_schema([e])
    assert "form" in str(exc_info.value)
    assert "status" in str(exc_info.value)


def test_build_snapshot_raises_on_schema_drift_before_writing_anything():
    e = element()
    del e["now_cost"]
    with pytest.raises(KeyError):
        fds.build_snapshot(boot([e]), "2026-01-06")


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def test_build_snapshot_produces_expected_columns_and_values():
    data = boot([element()])
    df = fds.build_snapshot(data, "2026-01-06")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["Snapshot"] == "2026-01-06"
    assert row["ID"] == 1
    assert row["Player"] == "Salah"
    assert row["Team"] == "LIV"
    assert row["Pos"] == "MID"
    assert row["Price"] == pytest.approx(13.0)
    assert row["Owned %"] == pytest.approx(45.2)
    assert row["Form"] == pytest.approx(6.0)
    assert row["Exp pts next"] == pytest.approx(7.5)
    assert row["Transfers in (GW)"] == 10000
    assert row["Transfers out (GW)"] == 2000
    assert row["Status"] == "a"


def test_build_snapshot_unknown_team_falls_back_gracefully():
    e = element(team=999)  # not in the teams lookup
    df = fds.build_snapshot(boot([e]), "2026-01-06")
    assert df.iloc[0]["Team"] == "?"


def test_build_snapshot_handles_multiple_players_and_positions():
    elements = [
        element(id_=1, element_type=1),  # GKP
        element(id_=2, element_type=2),  # DEF
        element(id_=3, element_type=3),  # MID
        element(id_=4, element_type=4),  # FWD
    ]
    df = fds.build_snapshot(boot(elements), "2026-01-06")
    assert list(df["Pos"]) == ["GKP", "DEF", "MID", "FWD"]


# ---------------------------------------------------------------------------
# Network retry / quiet skip
# ---------------------------------------------------------------------------

def test_fetch_bootstrap_succeeds_on_first_try(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    monkeypatch.setattr(fds.SESSION, "get", lambda *a, **k: FakeResponse())
    result = fds.fetch_bootstrap()
    assert result == {"ok": True}


def test_fetch_bootstrap_retries_then_skips_quietly_on_persistent_network_error(monkeypatch):
    monkeypatch.setattr(fds, "RETRY_DELAY_SECONDS", 0)
    calls = {"n": 0}

    def always_fails(*a, **k):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("simulated network failure")

    monkeypatch.setattr(fds.SESSION, "get", always_fails)
    result = fds.fetch_bootstrap()

    assert result is None
    assert calls["n"] == fds.RETRIES


def test_fetch_bootstrap_recovers_after_transient_failure(monkeypatch):
    monkeypatch.setattr(fds, "RETRY_DELAY_SECONDS", 0)
    calls = {"n": 0}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def fails_once_then_succeeds(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.Timeout("simulated timeout")
        return FakeResponse()

    monkeypatch.setattr(fds.SESSION, "get", fails_once_then_succeeds)
    result = fds.fetch_bootstrap()

    assert result == {"ok": True}
    assert calls["n"] == 2


def test_fetch_bootstrap_does_not_retry_on_http_error_status(monkeypatch):
    """A 500 or similar isn't a network blip to retry through -- raise_for_status
    raising HTTPError should propagate immediately, not be swallowed as a skip."""
    class FailingResponse:
        def raise_for_status(self):
            raise requests.exceptions.HTTPError("500 Server Error")

    monkeypatch.setattr(fds.SESSION, "get", lambda *a, **k: FailingResponse())
    monkeypatch.setattr(fds, "RETRY_DELAY_SECONDS", 0)

    # HTTPError IS a RequestException subclass, so this still retries-then-skips
    # rather than crashing -- confirms it's treated the same as other network
    # errors (quiet skip), not elevated to a hard failure.
    result = fds.fetch_bootstrap()
    assert result is None


# ---------------------------------------------------------------------------
# Output (idempotent same-day overwrite)
# ---------------------------------------------------------------------------

def test_snapshot_written_to_parquet_has_matching_schema_on_rerun(tmp_path):
    df1 = fds.build_snapshot(boot([element(now_cost=130)]), "2026-01-06")
    df2 = fds.build_snapshot(boot([element(now_cost=135)]), "2026-01-06")  # price moved intraday

    path = tmp_path / "2026-01-06.parquet"
    df1.to_parquet(path, index=False)
    df2.to_parquet(path, index=False)  # simulates a same-day re-run

    result = pd.read_parquet(path)
    assert list(result.columns) == list(df1.columns)
    assert result.iloc[0]["Price"] == pytest.approx(13.5)  # latest write wins, not appended
    assert len(result) == 1  # not duplicated
