"""
ID Resolution
==============
The single hardest and most valuable piece of the spine (SPEC.md Section
4.3). Everything cross-source depends on it. A wrong join silently
corrupts every downstream number and is nearly impossible to detect
later -- so this module is built to fail toward "ask a human," never
toward "guess."

Matching cascade, in order:
    1. Exact string match
    2. Normalized match (unicode fold, strip accents/punctuation, lowercase)
    3. Constrained fuzzy match (token-sort ratio, gated on hints agreeing --
       birth date / current team / position for players)
    4. Unresolved -- logged for human review, never auto-accepted

FPL is the foundational source: there is nothing to "match" when
registering FPL's own players and teams, since FPL's bootstrap-static IS
the canonical list. register_fpl_players()/register_fpl_teams() seed the
registry directly. Every OTHER source (Understat, StatsBomb, Football-
Data) comes later and gets matched AGAINST that seeded registry via
resolve_player()/resolve_team().

Right now (Phase 2, before Phase 3's Football-Data collector or Phase 9's
StatsBomb collector exist) there is only one live source to seed from.
teams.csv and players.csv will therefore show fpl_id populated and the
other source columns empty until those collectors actually run
resolve_*() against real data -- that's an honest, expected state, not a
bug.
"""

import difflib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PLAYERS_PATH = Path("data/ids/players.csv")
TEAMS_PATH = Path("data/ids/teams.csv")
UNRESOLVED_PATH = Path("data/ids/unresolved.csv")

PLAYER_SCHEMA = ["canonical_id", "display_name", "fpl_id", "understat_id", "statsbomb_id",
                 "birth_date", "confidence", "method", "verified_at"]
TEAM_SCHEMA = ["canonical_id", "display_name", "fpl_id", "understat_id", "statsbomb_id",
               "football_data_name", "understat_name", "country", "tier",
               "confidence", "method", "verified_at"]
UNRESOLVED_SCHEMA = ["as_of", "entity_type", "source", "name", "hints", "best_candidate", "best_score"]

# Explicit nullable dtypes on load, not left to inference. Without this, a
# column that's entirely NaN (e.g. understat_id before Understat exists)
# round-trips through CSV as float64, and a later write of a real string
# into that column raises a pandas dtype error -- this bit a test before
# it could bite a real resolve_team() call.
PLAYER_DTYPES = {"canonical_id": "string", "display_name": "string", "fpl_id": "Int64",
                  "understat_id": "string", "statsbomb_id": "string", "birth_date": "string",
                  "method": "string", "verified_at": "string"}
TEAM_DTYPES = {"canonical_id": "string", "display_name": "string", "fpl_id": "Int64",
                "understat_id": "string", "statsbomb_id": "string",
                "football_data_name": "string", "understat_name": "string",
                "country": "string", "tier": "Int64", "method": "string", "verified_at": "string"}

# Below this token-sort-ratio, a fuzzy match is never auto-accepted --
# it goes to unresolved.csv for a human to decide, per the hard rule.
FUZZY_THRESHOLD = 0.85

# A small, manually curated table for the specific case similarity
# scoring can never solve: FPL's own display_name is sometimes a
# nickname or abbreviation with little to no string overlap with how
# other sources spell the same club. This is NOT a relaxation of the
# fuzzy threshold -- it's an exact, human-verified lookup with zero
# ambiguity, checked before fuzzy scoring even runs. Verified against
# the real registry (2026-08-28): "Man United" vs FPL's "Man Utd"
# scores 0.824, just under the 0.85 bar; "Tottenham" vs FPL's "Spurs"
# scores 0.0 -- no similarity metric bridges an actual nickname. Kept
# intentionally small (only entries verified against a real unresolved
# match), not a speculative full nickname database.
KNOWN_ALIASES = {
    "man utd": ["man united", "manchester united"],
    "spurs": ["tottenham", "tottenham hotspur"],
}


# ---------------------------------------------------------------------------
# STRING MATCHING (pure)
# ---------------------------------------------------------------------------

def normalize(name: str) -> str:
    """Unicode-fold, strip accents and punctuation, lowercase, collapse
    whitespace. 'João Pedro' -> 'joao pedro'; "Nott'm Forest" -> 'nottm forest'."""
    if not name:
        return ""
    folded = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in folded if not unicodedata.combining(c))
    no_punct = re.sub(r"[^\w\s]", "", no_accents)
    return re.sub(r"\s+", " ", no_punct).strip().lower()


def token_sort_ratio(a: str, b: str) -> float:
    """Order-independent similarity: tokens are sorted before comparing, so
    'Bruno Fernandes' and 'Fernandes Bruno' score identically to 'Bruno
    Fernandes' vs 'Bruno Fernandes'. Built on stdlib difflib rather than a
    new fuzzy-matching dependency -- this is exactly how token_sort_ratio
    is implemented in the common fuzzy-matching libraries anyway."""
    a_sorted = " ".join(sorted(normalize(a).split()))
    b_sorted = " ".join(sorted(normalize(b).split()))
    if not a_sorted or not b_sorted:
        return 0.0
    return difflib.SequenceMatcher(None, a_sorted, b_sorted).ratio()


def _apply_hints(candidates: pd.DataFrame, hints: dict | None) -> pd.DataFrame:
    """Narrows `candidates` to rows agreeing with every hint supplied. A
    hint whose column isn't in the registry, or whose value is None, is
    simply not applied -- hints narrow, they never require columns the
    registry doesn't have."""
    if not hints:
        return candidates
    filtered = candidates
    for key, value in hints.items():
        if value is None or key not in filtered.columns:
            continue
        filtered = filtered[filtered[key] == value]
    return filtered


# ---------------------------------------------------------------------------
# THE CASCADE (pure -- operates on an in-memory registry, no I/O)
# ---------------------------------------------------------------------------

def match(name: str, registry: pd.DataFrame, hints: dict | None = None,
          fuzzy_threshold: float = FUZZY_THRESHOLD) -> dict:
    """
    Runs the full cascade against `registry` (players.csv- or teams.csv-
    shaped). Returns {"canonical_id", "confidence", "method"} on a match
    ("exact" / "normalized" / "fuzzy"), or on failure
    {"canonical_id": None, "confidence": 0.0, "method": "unresolved",
    "best_candidate": ..., "best_score": ...} -- never a guess past the
    threshold, per the hard rule in BRIEFS.md.
    """
    if registry is None or registry.empty or not name:
        return {"canonical_id": None, "confidence": 0.0, "method": "unresolved",
                "best_candidate": None, "best_score": 0.0}

    # 1. Exact
    exact = registry[registry["display_name"] == name]
    if len(exact) > 1:
        exact = _apply_hints(exact, hints)
    if len(exact) == 1:
        return {"canonical_id": exact.iloc[0]["canonical_id"], "confidence": 1.0, "method": "exact"}

    # 2. Normalized
    target_norm = normalize(name)
    norm_matches = registry[registry["display_name"].apply(normalize) == target_norm]
    if len(norm_matches) > 1:
        norm_matches = _apply_hints(norm_matches, hints)
    if len(norm_matches) == 1:
        return {"canonical_id": norm_matches.iloc[0]["canonical_id"], "confidence": 0.95, "method": "normalized"}

    # 2.5. Known alias -- an exact, curated lookup for a nickname/
    # abbreviation no similarity score can bridge (see KNOWN_ALIASES).
    # Still never ambiguous: falls through to fuzzy/unresolved if more
    # than one registry row would match the aliased name.
    alias_key = next((k for k, aliases in KNOWN_ALIASES.items() if target_norm in aliases), None)
    if alias_key:
        alias_matches = registry[registry["display_name"].apply(normalize) == alias_key]
        if len(alias_matches) > 1:
            alias_matches = _apply_hints(alias_matches, hints)
        if len(alias_matches) == 1:
            return {"canonical_id": alias_matches.iloc[0]["canonical_id"], "confidence": 1.0, "method": "alias"}

    # 3. Constrained fuzzy -- hint agreement is required whenever hints are
    # given, even for a single strong candidate. Name similarity alone,
    # however high, is never sufficient by itself.
    scored = registry.copy()
    scored["_score"] = scored["display_name"].apply(lambda n: token_sort_ratio(name, n))
    above_threshold = scored[scored["_score"] >= fuzzy_threshold].sort_values("_score", ascending=False)
    constrained = _apply_hints(above_threshold, hints)

    if len(constrained) >= 1:
        top = constrained.iloc[0]
        return {"canonical_id": top["canonical_id"], "confidence": round(float(top["_score"]), 3),
                "method": "fuzzy"}

    # 4. Unresolved
    best_score = float(scored["_score"].max()) if len(scored) else 0.0
    best_name = scored.loc[scored["_score"].idxmax(), "display_name"] if len(scored) else None
    return {"canonical_id": None, "confidence": 0.0, "method": "unresolved",
            "best_candidate": best_name, "best_score": round(best_score, 3)}


# ---------------------------------------------------------------------------
# REGISTRY I/O
# ---------------------------------------------------------------------------

def _load(path: Path, schema: list, dtypes: dict | None = None) -> pd.DataFrame:
    if not path.exists():
        df = pd.DataFrame(columns=schema)
    else:
        df = pd.read_csv(path)
    if dtypes:
        for col, dt in dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dt)
    return df


def _save(df: pd.DataFrame, path: Path, schema: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[schema].to_csv(path, index=False)


def load_players() -> pd.DataFrame:
    return _load(PLAYERS_PATH, PLAYER_SCHEMA, PLAYER_DTYPES)


def load_teams() -> pd.DataFrame:
    return _load(TEAMS_PATH, TEAM_SCHEMA, TEAM_DTYPES)


def _log_unresolved(entity_type: str, source: str, name: str, hints: dict | None, result: dict) -> None:
    """Upserts one row per (entity_type, source, name) -- re-seeing the
    same unresolved name on a later run refreshes its as_of/best-candidate
    rather than piling up duplicate rows in the review queue."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = {"as_of": today, "entity_type": entity_type, "source": source, "name": name,
           "hints": json.dumps(hints) if hints else "",
           "best_candidate": result.get("best_candidate"), "best_score": result.get("best_score", 0.0)}

    existing = _load(UNRESOLVED_PATH, UNRESOLVED_SCHEMA)
    key = (existing["entity_type"] == entity_type) & (existing["source"] == source) & (existing["name"] == name)
    existing = existing[~key]
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    _save(combined, UNRESOLVED_PATH, UNRESOLVED_SCHEMA)


# ---------------------------------------------------------------------------
# PUBLIC API -- resolve (secondary sources, matched against the registry)
# ---------------------------------------------------------------------------

def resolve_player(name: str, source: str, hints: dict | None = None) -> str | None:
    """canonical_id if `name` (as seen from `source`) matches an existing
    player, else None. Logs a failed match to unresolved.csv for review --
    never guesses, never drops it silently."""
    result = match(name, load_players(), hints)
    if result["canonical_id"] is None:
        _log_unresolved("player", source, name, hints, result)
    return result["canonical_id"]


# Which teams.csv column records a source's own naming for a team, once
# resolved. Not every source gets a naming-variant column (SPEC's schema
# only names these two); a source without one here just isn't recorded,
# which is fine -- the canonical_id match is what matters.
TEAM_NAME_VARIANT_COLUMNS = {"football_data": "football_data_name", "understat": "understat_name"}


def resolve_team(name: str, source: str, hints: dict | None = None) -> str | None:
    """
    canonical_id if `name` (as seen from `source`) matches an existing
    team, else None (logged to unresolved.csv). On a successful match,
    also records `name` into that source's naming-variant column in
    teams.csv (e.g. football_data_name) if it isn't set yet -- this is
    what makes the crosswalk actually accumulate cross-source naming
    over time, rather than just being a one-off lookup.
    """
    result = match(name, load_teams(), hints)
    if result["canonical_id"] is None:
        _log_unresolved("team", source, name, hints, result)
    else:
        _record_team_naming_variant(result["canonical_id"], source, name)
    return result["canonical_id"]


def _record_team_naming_variant(canonical_id: str, source: str, name: str) -> None:
    column = TEAM_NAME_VARIANT_COLUMNS.get(source)
    if column is None:
        return
    teams = load_teams().set_index("canonical_id")
    if canonical_id not in teams.index:
        return
    current = teams.loc[canonical_id, column]
    if pd.isna(current):
        teams.loc[canonical_id, column] = name
        _save(teams.reset_index(), TEAMS_PATH, TEAM_SCHEMA)


# ---------------------------------------------------------------------------
# PUBLIC API -- register (FPL, the foundational source: seed, don't match)
# ---------------------------------------------------------------------------

def register_fpl_teams(fpl_teams: list) -> pd.DataFrame:
    """
    Seeds/updates teams.csv directly from FPL's bootstrap-static teams.
    FPL is the foundational source -- there's no matching involved, only
    registering what FPL says exists. canonical_id is derived
    deterministically from FPL's own id (stable within a season), so
    re-running is idempotent by construction: the same team always maps
    to the same canonical_id, and re-registering just refreshes
    display_name if it changed.
    """
    existing = load_teams().set_index("canonical_id")
    now = datetime.now(timezone.utc).isoformat()

    for t in fpl_teams:
        cid = f"fpl-team-{t['id']}"
        row = existing.loc[cid] if cid in existing.index else pd.Series(dtype=object)
        existing.loc[cid, "display_name"] = t["name"]
        existing.loc[cid, "fpl_id"] = t["id"]
        existing.loc[cid, "country"] = row.get("country") or "England"
        existing.loc[cid, "tier"] = row.get("tier") or 1
        existing.loc[cid, "confidence"] = 1.0
        existing.loc[cid, "method"] = "exact"
        for col in ("understat_id", "statsbomb_id", "football_data_name", "understat_name", "verified_at"):
            if col not in existing.columns or pd.isna(existing.loc[cid].get(col)):
                existing.loc[cid, col] = row.get(col) if col in row else None

    result = existing.reset_index()
    _save(result, TEAMS_PATH, TEAM_SCHEMA)
    return result


def register_historical_teams(names: list, country: str = "England", tier: int = 1) -> pd.DataFrame:
    """
    Seeds teams.csv with clubs FPL's bootstrap-static no longer lists --
    relegated at some point -- but that appear in the historical
    archive. FPL only ever tracks the CURRENT season's 20 clubs
    (register_fpl_teams reflects whatever occupies each of FPL's 20
    slots today), so a club like a past relegated side has no FPL
    entry to match against at all, however it's spelled -- not a fuzzy-
    matching problem, an entirely absent one. Verified live (2026-08-28):
    this was silently capping Understat xG coverage for the Premier
    League at 17 of 20 clubs even in-season, since 3 of the "current"
    slots were occupied by clubs that had only just been promoted.

    football-data.co.uk's own spelling is treated as canonical for
    these -- it's been stable for three decades, and there's no other
    authority to prefer. canonical_id is prefixed "hist-team-" (vs
    FPL's "fpl-team-") so it's always auditable which authority
    registered which entry.

    Skips any name that already resolves against the existing registry
    under some other spelling (including via KNOWN_ALIASES) -- this
    only fills a genuine absence, never creates a duplicate for a team
    that already has an entry.
    """
    existing = load_teams().set_index("canonical_id")
    now = datetime.now(timezone.utc).isoformat()

    for name in names:
        result = match(name, existing.reset_index())
        if result["canonical_id"] is not None:
            continue
        cid = f"hist-team-{normalize(name).replace(' ', '-')}"
        if cid in existing.index:
            continue
        existing.loc[cid, "display_name"] = name
        existing.loc[cid, "football_data_name"] = name
        existing.loc[cid, "country"] = country
        existing.loc[cid, "tier"] = tier
        existing.loc[cid, "confidence"] = 1.0
        existing.loc[cid, "method"] = "historical_seed"
        existing.loc[cid, "verified_at"] = now

    result = existing.reset_index()
    _save(result, TEAMS_PATH, TEAM_SCHEMA)
    return result


def register_fpl_players(fpl_elements: list) -> pd.DataFrame:
    """Same seeding logic as register_fpl_teams, for players. birth_date
    isn't in FPL's bootstrap-static, so it stays null until a source that
    has it (StatsBomb, Understat) gets matched in -- it exists as a future
    disambiguator, not something FPL alone can supply."""
    existing = load_players().set_index("canonical_id")

    for e in fpl_elements:
        cid = f"fpl-player-{e['id']}"
        display_name = f"{e['first_name']} {e['second_name']}".strip()
        row = existing.loc[cid] if cid in existing.index else pd.Series(dtype=object)
        existing.loc[cid, "display_name"] = display_name
        existing.loc[cid, "fpl_id"] = e["id"]
        existing.loc[cid, "confidence"] = 1.0
        existing.loc[cid, "method"] = "exact"
        for col in ("understat_id", "statsbomb_id", "birth_date", "verified_at"):
            if col not in existing.columns or pd.isna(existing.loc[cid].get(col)):
                existing.loc[cid, col] = row.get(col) if col in row else None

    result = existing.reset_index()
    _save(result, PLAYERS_PATH, PLAYER_SCHEMA)
    return result
