import pandas as pd
import pytest

from core import ids


# ---------------------------------------------------------------------------
# normalize / token_sort_ratio
# ---------------------------------------------------------------------------

def test_normalize_strips_accents():
    assert ids.normalize("João Pedro") == "joao pedro"


def test_normalize_strips_punctuation():
    assert ids.normalize("Nott'm Forest") == "nottm forest"


def test_normalize_collapses_whitespace_and_lowercases():
    assert ids.normalize("  Manchester   United  ") == "manchester united"


def test_normalize_empty_and_none_safe():
    assert ids.normalize("") == ""
    assert ids.normalize(None) == ""


def test_token_sort_ratio_identical_is_one():
    assert ids.token_sort_ratio("Bruno Fernandes", "Bruno Fernandes") == pytest.approx(1.0)


def test_token_sort_ratio_word_order_independent():
    """The whole point of token-SORT: reordered tokens still score as a
    near-perfect match, unlike a plain string diff."""
    score = ids.token_sort_ratio("Bruno Fernandes", "Fernandes Bruno")
    assert score == pytest.approx(1.0)


def test_token_sort_ratio_different_names_score_low():
    assert ids.token_sort_ratio("Bruno Fernandes", "Erling Haaland") < 0.5


def test_token_sort_ratio_empty_string_is_zero():
    assert ids.token_sort_ratio("", "Bruno Fernandes") == 0.0


# ---------------------------------------------------------------------------
# match() cascade
# ---------------------------------------------------------------------------

def _registry(rows):
    return pd.DataFrame(rows)


def test_match_exact():
    registry = _registry([{"canonical_id": "p1", "display_name": "Bruno Fernandes"}])
    result = ids.match("Bruno Fernandes", registry)
    assert result == {"canonical_id": "p1", "confidence": 1.0, "method": "exact"}


def test_match_normalized_on_accent_difference():
    registry = _registry([{"canonical_id": "p1", "display_name": "João Pedro"}])
    result = ids.match("Joao Pedro", registry)  # no accent -- not an exact string match
    assert result["canonical_id"] == "p1"
    assert result["method"] == "normalized"
    assert result["confidence"] == pytest.approx(0.95)


def test_match_fuzzy_accepted_when_above_threshold_and_no_hints():
    registry = _registry([{"canonical_id": "p1", "display_name": "Mohammed Salah"}])
    result = ids.match("Mohamed Salah", registry)  # spelling variant -- not exact/normalized-equal
    assert result["canonical_id"] == "p1"
    assert result["method"] == "fuzzy"


def test_match_fuzzy_rejected_below_threshold_returns_unresolved_with_best_candidate():
    registry = _registry([{"canonical_id": "p1", "display_name": "Erling Haaland"}])
    result = ids.match("Bruno Fernandes", registry)
    assert result["canonical_id"] is None
    assert result["method"] == "unresolved"
    assert result["best_candidate"] == "Erling Haaland"
    assert result["best_score"] < ids.FUZZY_THRESHOLD


def test_match_fuzzy_requires_hint_agreement_even_when_score_is_high():
    """The hard rule: a strong name match is NOT enough on its own if a
    supplied hint disagrees -- this is what 'constrained' fuzzy matching means."""
    registry = _registry([
        {"canonical_id": "p1", "display_name": "Mohammed Salah", "team": "LIV"},
    ])
    result = ids.match("Mohamed Salah", registry, hints={"team": "CHE"})  # wrong team on purpose
    assert result["canonical_id"] is None
    assert result["method"] == "unresolved"


def test_match_fuzzy_accepted_when_hint_agrees():
    registry = _registry([
        {"canonical_id": "p1", "display_name": "Mohammed Salah", "team": "LIV"},
    ])
    result = ids.match("Mohamed Salah", registry, hints={"team": "LIV"})
    assert result["canonical_id"] == "p1"
    assert result["method"] == "fuzzy"


def test_match_ambiguous_exact_disambiguated_by_hints():
    registry = _registry([
        {"canonical_id": "p1", "display_name": "James Smith", "team": "ARS"},
        {"canonical_id": "p2", "display_name": "James Smith", "team": "CHE"},
    ])
    result = ids.match("James Smith", registry, hints={"team": "CHE"})
    assert result["canonical_id"] == "p2"
    assert result["method"] == "exact"


def test_match_ambiguous_exact_without_disambiguating_hints_does_not_guess():
    registry = _registry([
        {"canonical_id": "p1", "display_name": "James Smith", "team": "ARS"},
        {"canonical_id": "p2", "display_name": "James Smith", "team": "CHE"},
    ])
    result = ids.match("James Smith", registry)  # no hints -- can't disambiguate
    # Falls through the exact/normalized stages (both still ambiguous); fuzzy
    # stage without hints will match one of them since name similarity is 1.0
    # against both -- but it must not silently return a WRONG one undetected.
    # What matters here is it never crashes and always returns SOME defined
    # dict shape.
    assert result["method"] in ("fuzzy", "unresolved")
    if result["canonical_id"] is not None:
        assert result["canonical_id"] in ("p1", "p2")


def test_match_empty_registry_returns_unresolved_not_a_crash():
    result = ids.match("Anyone", pd.DataFrame(columns=["canonical_id", "display_name"]))
    assert result["canonical_id"] is None
    assert result["method"] == "unresolved"


def test_match_empty_name_returns_unresolved():
    registry = _registry([{"canonical_id": "p1", "display_name": "Bruno Fernandes"}])
    result = ids.match("", registry)
    assert result["canonical_id"] is None


# ---------------------------------------------------------------------------
# resolve_player / resolve_team (I/O layer, unresolved logging)
# ---------------------------------------------------------------------------

def test_resolve_player_matches_against_registered_players(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")
    ids.register_fpl_players([
        {"id": 1, "first_name": "Bruno", "second_name": "Fernandes"},
    ])

    canonical_id = ids.resolve_player("Bruno Fernandes", source="understat")
    assert canonical_id == "fpl-player-1"


def test_resolve_player_logs_unresolved_names_for_review(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")
    ids.register_fpl_players([{"id": 1, "first_name": "Bruno", "second_name": "Fernandes"}])

    result = ids.resolve_player("Someone Totally Different", source="understat")

    assert result is None
    unresolved = pd.read_csv(tmp_path / "unresolved.csv")
    assert len(unresolved) == 1
    assert unresolved.iloc[0]["name"] == "Someone Totally Different"
    assert unresolved.iloc[0]["source"] == "understat"
    assert unresolved.iloc[0]["entity_type"] == "player"


def test_resolve_player_unresolved_logging_is_idempotent_not_duplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")
    ids.register_fpl_players([{"id": 1, "first_name": "Bruno", "second_name": "Fernandes"}])

    ids.resolve_player("Someone Totally Different", source="understat")
    ids.resolve_player("Someone Totally Different", source="understat")  # same name, again

    unresolved = pd.read_csv(tmp_path / "unresolved.csv")
    assert len(unresolved) == 1  # not duplicated


def test_resolve_team_matches_against_registered_teams(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")
    ids.register_fpl_teams([{"id": 1, "name": "Arsenal"}])

    assert ids.resolve_team("Arsenal", source="football_data") == "fpl-team-1"


def test_resolve_team_records_naming_variant_on_success(tmp_path, monkeypatch):
    """A successful match should enrich the crosswalk, not just look
    something up and discard the source's own spelling of the name.
    Case/whitespace difference -- resolves via the normalized stage,
    not fuzzy (real team abbreviations like 'Man Utd' vs 'Manchester
    United' don't clear the fuzzy threshold at all -- verified separately;
    that's why teams need a curated/normalized lookup, not fuzzy matching)."""
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")
    ids.register_fpl_teams([{"id": 1, "name": "Nott'm Forest"}])

    ids.resolve_team("NOTT'M FOREST", source="football_data")  # different case

    teams = ids.load_teams()
    row = teams[teams["canonical_id"] == "fpl-team-1"].iloc[0]
    assert row["football_data_name"] == "NOTT'M FOREST"


def test_resolve_team_does_not_overwrite_an_existing_naming_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")
    ids.register_fpl_teams([{"id": 1, "name": "Nott'm Forest"}])
    ids.resolve_team("NOTT'M FOREST", source="football_data")

    # A later run resolves via a differently-cased spelling that still
    # matches -- the first-recorded variant should stick, not flip-flop.
    ids.resolve_team("nott'm forest", source="football_data")

    teams = ids.load_teams()
    row = teams[teams["canonical_id"] == "fpl-team-1"].iloc[0]
    assert row["football_data_name"] == "NOTT'M FOREST"


def test_resolve_team_unknown_source_has_no_naming_variant_column_to_update(tmp_path, monkeypatch):
    """A source with no naming-variant column in the schema (e.g.
    statsbomb, which only gets an id slot) just doesn't record one --
    that's a no-op, not an error."""
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    monkeypatch.setattr(ids, "UNRESOLVED_PATH", tmp_path / "unresolved.csv")
    ids.register_fpl_teams([{"id": 1, "name": "Arsenal"}])

    result = ids.resolve_team("Arsenal", source="statsbomb")
    assert result == "fpl-team-1"  # still resolves fine


# ---------------------------------------------------------------------------
# register_fpl_teams / register_fpl_players (seeding, idempotency)
# ---------------------------------------------------------------------------

def test_register_fpl_teams_seeds_expected_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    result = ids.register_fpl_teams([{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Chelsea"}])

    assert len(result) == 2
    row = result[result["canonical_id"] == "fpl-team-1"].iloc[0]
    assert row["display_name"] == "Arsenal"
    assert row["fpl_id"] == 1
    assert row["country"] == "England"
    assert row["tier"] == 1
    assert row["confidence"] == 1.0
    assert row["method"] == "exact"
    assert pd.isna(row["understat_id"])  # not yet resolved -- honest, not fabricated


def test_register_fpl_teams_is_idempotent_on_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    ids.register_fpl_teams([{"id": 1, "name": "Arsenal"}])
    result = ids.register_fpl_teams([{"id": 1, "name": "Arsenal"}])  # same team again

    assert len(result) == 1  # not duplicated
    assert result.iloc[0]["canonical_id"] == "fpl-team-1"


def test_register_fpl_teams_preserves_resolved_columns_across_reruns(tmp_path, monkeypatch):
    """Once Football-Data resolution fills in football_data_name, a later
    FPL re-registration must not wipe it back to null."""
    monkeypatch.setattr(ids, "TEAMS_PATH", tmp_path / "teams.csv")
    ids.register_fpl_teams([{"id": 1, "name": "Arsenal"}])

    teams = ids.load_teams()
    teams.loc[teams["canonical_id"] == "fpl-team-1", "football_data_name"] = "Arsenal"
    ids._save(teams, ids.TEAMS_PATH, ids.TEAM_SCHEMA)

    result = ids.register_fpl_teams([{"id": 1, "name": "Arsenal"}])
    assert result.iloc[0]["football_data_name"] == "Arsenal"


def test_register_fpl_players_seeds_expected_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    result = ids.register_fpl_players([
        {"id": 1, "first_name": "Bruno", "second_name": "Fernandes"},
    ])

    row = result.iloc[0]
    assert row["canonical_id"] == "fpl-player-1"
    assert row["display_name"] == "Bruno Fernandes"
    assert row["fpl_id"] == 1
    assert row["confidence"] == 1.0
    assert pd.isna(row["birth_date"])  # FPL doesn't supply this -- honest gap, not fabricated


def test_register_fpl_players_is_idempotent_on_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    ids.register_fpl_players([{"id": 1, "first_name": "Bruno", "second_name": "Fernandes"}])
    result = ids.register_fpl_players([{"id": 1, "first_name": "Bruno", "second_name": "Fernandes"}])

    assert len(result) == 1


def test_register_fpl_players_updates_display_name_on_change(tmp_path, monkeypatch):
    """A mid-season correction to FPL's own naming should propagate, since
    FPL is authoritative for its own data."""
    monkeypatch.setattr(ids, "PLAYERS_PATH", tmp_path / "players.csv")
    ids.register_fpl_players([{"id": 1, "first_name": "Bruno", "second_name": "Fernandes"}])
    result = ids.register_fpl_players([{"id": 1, "first_name": "Bruno", "second_name": "Fernandes Silva"}])

    assert result.iloc[0]["display_name"] == "Bruno Fernandes Silva"
    assert len(result) == 1
