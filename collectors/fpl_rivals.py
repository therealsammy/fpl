#!/usr/bin/env python3
"""
FPL Mini-League Rival Tracker
==============================
Global ownership is the wrong denominator when you're playing to win a
mini-league. This pulls the league's standings and every manager's current
squad, then measures ownership, captaincy and differentials against that
league only.

Run it after fpl_tracker.py, once a gameweek's transfer deadline has passed
(the picks endpoint 404s before that -- this handles it, it doesn't crash):

    python fpl_rivals.py

Config: set LEAGUE_ID and ENTRY_ID below, or export FPL_LEAGUE_ID /
FPL_ENTRY_ID. ENTRY_ID should match the one in fpl_tracker.py -- it is how
this script tells "you" apart from your rivals in the same standings list.

Reads nothing but the API (fpl_history.csv is used only to enrich player IDs
with names/teams in the printed digest -- optional, degrades gracefully if
absent). Writes fpl_rivals.csv, append-only, one row per (manager, squad
player) per gameweek, keyed on the same UTC snapshot-date convention as the
main store. Safe to re-run the same day.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

LEAGUE_ID = 0        # <-- your mini-league ID (from the standings page URL)
ENTRY_ID = 8592220   # <-- your manager ID, same as fpl_tracker.py


def _int_env(name, default):
    raw = (os.environ.get(name) or "").strip()
    return int(raw) if raw.isdigit() else default


LEAGUE_ID = _int_env("FPL_LEAGUE_ID", LEAGUE_ID)
ENTRY_ID = _int_env("FPL_ENTRY_ID", ENTRY_ID)

HISTORY = Path("fpl_history.csv")
OUTPUT = Path("fpl_rivals.csv")

# Picks is one request per manager per run (N+1 overall with standings).
# A small gap keeps this polite to a public, unauthenticated API.
REQUEST_DELAY = 0.3

BASE = "https://fantasy.premierleague.com/api"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
})


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------

def get(path, params=None):
    """GET a JSON endpoint, returning None on 404 rather than raising.

    A 404 here is expected and meaningful: the picks endpoint 404s for any
    gameweek whose deadline hasn't passed yet.
    """
    r = SESSION.get(f"{BASE}/{path}", params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def current_gw_from_bootstrap(boot):
    """Same logic as fpl_tracker.py: the last gameweek whose deadline has
    passed is the one whose picks are actually visible."""
    for ev in boot["events"]:
        if ev["is_current"]:
            return ev["id"], ev["finished"]
    nxt = next((e["id"] for e in boot["events"] if e["is_next"]), 1)
    return max(1, nxt - 1), None


def fetch_standings(league_id):
    """All entries in the classic league, handling pagination."""
    league_name = None
    results = []
    page = 1
    while True:
        data = get(f"leagues-classic/{league_id}/standings/",
                    params={"page_standings": page})
        if data is None:
            break
        league_name = league_name or data.get("league", {}).get("name")
        standings = data.get("standings", {})
        results.extend(standings.get("results", []))
        if not standings.get("has_next"):
            break
        page += 1
    return league_name, results


def fetch_picks_for_entries(entry_ids, gw):
    """One picks/ call per manager. None means not available yet (404) or
    an entry that has no picks for this event -- callers must handle both
    the same way: skip that manager for this run, don't crash the batch."""
    picks_by_entry = {}
    for i, eid in enumerate(entry_ids):
        picks_by_entry[eid] = get(f"entry/{eid}/event/{gw}/picks/")
        if i < len(entry_ids) - 1:
            time.sleep(REQUEST_DELAY)
    return picks_by_entry


# ---------------------------------------------------------------------------
# TRANSFORM (pure -- no I/O, so tests can hand these synthetic fixtures)
# ---------------------------------------------------------------------------

def build_rivals_rows(standings, picks_by_entry, gw, snapshot_date):
    """Long format: one row per manager per squad player. Returns the rows
    plus the list of managers skipped because their picks weren't available."""
    rows = []
    skipped = []
    for s in standings:
        eid = s["entry"]
        picks = picks_by_entry.get(eid)
        if not picks or not picks.get("picks"):
            skipped.append((eid, s.get("player_name")))
            continue
        for p in picks["picks"]:
            rows.append({
                "Snapshot": snapshot_date,
                "GW": gw,
                "Entry ID": eid,
                "Manager": s.get("player_name"),
                "Team name": s.get("entry_name"),
                "Rank": s.get("rank"),
                "Total points": s.get("total"),
                "Player ID": p["element"],
                "Captain": bool(p.get("is_captain")),
                "Vice captain": bool(p.get("is_vice_captain")),
                "Multiplier": p.get("multiplier"),
            })
    return pd.DataFrame(rows), skipped


def compute_effective_ownership(rivals_gw: pd.DataFrame) -> pd.DataFrame:
    """Owned% + captained%, as a fraction of THIS league, not the global game."""
    if rivals_gw.empty:
        return pd.DataFrame(columns=["Player ID", "Owned by", "Captained by",
                                      "Owned %", "Captained %", "Effective ownership %"])

    total_managers = rivals_gw["Entry ID"].nunique()
    owned = rivals_gw.groupby("Player ID")["Entry ID"].nunique().rename("Owned by")
    captained = (rivals_gw[rivals_gw["Captain"]]
                 .groupby("Player ID")["Entry ID"].nunique().rename("Captained by"))

    out = pd.concat([owned, captained], axis=1)
    out["Captained by"] = out["Captained by"].fillna(0).astype(int)
    out = out.reset_index()
    out["Owned %"] = (out["Owned by"] / total_managers * 100).round(1)
    out["Captained %"] = (out["Captained by"] / total_managers * 100).round(1)
    out["Effective ownership %"] = (out["Owned %"] + out["Captained %"]).round(1)
    return out.sort_values("Effective ownership %", ascending=False).reset_index(drop=True)


def compute_differentials(rivals_gw: pd.DataFrame, my_entry_id: int):
    """Players unique to me, and players rivals hold that I don't -- the two
    halves of "which players I own that nobody else does, and vice versa"."""
    my_squad = set(rivals_gw.loc[rivals_gw["Entry ID"] == my_entry_id, "Player ID"])
    rivals = rivals_gw[rivals_gw["Entry ID"] != my_entry_id]
    rival_owned = rivals.groupby("Player ID")["Entry ID"].nunique()

    unique_to_me = pd.DataFrame({
        "Player ID": sorted(p for p in my_squad if rival_owned.get(p, 0) == 0)
    })

    missing_for_me = (
        rival_owned[~rival_owned.index.isin(my_squad)]
        .rename("Owned by rivals")
        .reset_index()
        .sort_values("Owned by rivals", ascending=False)
        .reset_index(drop=True)
    )
    return unique_to_me, missing_for_me


def compute_captain_divergence(rivals_gw: pd.DataFrame, my_entry_id: int) -> dict:
    """Where my armband differs from the field's most popular captain."""
    mine = rivals_gw[(rivals_gw["Entry ID"] == my_entry_id) & (rivals_gw["Captain"])]
    my_captain = int(mine["Player ID"].iloc[0]) if not mine.empty else None

    field = rivals_gw[rivals_gw["Captain"]]
    counts = field.groupby("Player ID")["Entry ID"].nunique().sort_values(ascending=False)
    if counts.empty:
        return {"my_captain": my_captain, "field_top_captain": None,
                "field_top_count": 0, "field_total": 0, "field_top_pct": None,
                "diverges": None}

    field_top_captain = int(counts.index[0])
    field_top_count = int(counts.iloc[0])
    field_total = field["Entry ID"].nunique()
    return {
        "my_captain": my_captain,
        "field_top_captain": field_top_captain,
        "field_top_count": field_top_count,
        "field_total": field_total,
        "field_top_pct": round(field_top_count / field_total * 100, 1) if field_total else None,
        "diverges": None if my_captain is None else (my_captain != field_top_captain),
    }


# ---------------------------------------------------------------------------
# STORE (append-only, idempotent same-day)
# ---------------------------------------------------------------------------

def update_store(rows_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    today = rows_df["Snapshot"].iloc[0]

    if path.exists():
        prior = pd.read_csv(path)
        before = prior["Snapshot"].nunique()
        prior = prior[prior["Snapshot"] != today]
        if prior["Snapshot"].nunique() < before:
            print(f"fpl_rivals: replacing an earlier run from {today}")
        combined = pd.concat([prior, rows_df], ignore_index=True)
    else:
        combined = rows_df
        print("fpl_rivals: first snapshot, creating store")

    combined = combined.sort_values(
        ["Snapshot", "GW", "Entry ID", "Player ID"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


# ---------------------------------------------------------------------------
# ENRICHMENT (optional -- degrades gracefully without fpl_history.csv)
# ---------------------------------------------------------------------------

def _player_lookup() -> dict:
    if not HISTORY.exists():
        return {}
    hist = pd.read_csv(HISTORY, usecols=["Snapshot", "ID", "Player", "Team"])
    latest = hist[hist["Snapshot"] == hist["Snapshot"].max()]
    return latest.set_index("ID")[["Player", "Team"]].to_dict("index")


def _enrich(df: pd.DataFrame, lookup: dict, id_col="Player ID") -> pd.DataFrame:
    if df.empty or not lookup:
        return df
    df = df.copy()
    df["Player"] = df[id_col].map(lambda pid: lookup.get(pid, {}).get("Player", f"#{pid}"))
    df["Team"] = df[id_col].map(lambda pid: lookup.get(pid, {}).get("Team", ""))
    cols = ["Player", "Team"] + [c for c in df.columns if c not in ("Player", "Team")]
    return df[cols]


def _player_name(pid, lookup: dict):
    if pid is None:
        return None
    return lookup.get(pid, {}).get("Player", f"#{pid}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if LEAGUE_ID == 0:
        print("Set LEAGUE_ID at the top of this file, or export FPL_LEAGUE_ID.")
        sys.exit(1)

    print("Fetching bootstrap-static...")
    boot = get("bootstrap-static/")
    gw, _ = current_gw_from_bootstrap(boot)

    print(f"Fetching standings for league {LEAGUE_ID}...")
    league_name, standings = fetch_standings(LEAGUE_ID)
    if not standings:
        print(f"No standings found for league {LEAGUE_ID}. Check the ID.")
        sys.exit(1)
    print(f"League: {league_name} | {len(standings)} manager(s)")

    print(f"Fetching GW{gw} picks for {len(standings)} manager(s) "
          f"({REQUEST_DELAY}s apart)...")
    picks_by_entry = fetch_picks_for_entries([s["entry"] for s in standings], gw)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows_df, skipped = build_rivals_rows(standings, picks_by_entry, gw, today)

    if skipped:
        names = ", ".join(name or str(eid) for eid, name in skipped)
        print(f"Skipped {len(skipped)} manager(s) -- picks not available yet "
              f"(GW{gw} deadline may not have passed): {names}")

    if rows_df.empty:
        print("No picks available for anyone yet -- nothing to store this run.")
        return

    update_store(rows_df, OUTPUT)
    print(f"fpl_rivals: {len(rows_df)} rows this run -> {OUTPUT}")

    lookup = _player_lookup()
    eff = compute_effective_ownership(rows_df)
    unique_to_me, missing_for_me = compute_differentials(rows_df, ENTRY_ID)
    captain = compute_captain_divergence(rows_df, ENTRY_ID)

    print(f"\nTop effective ownership in {league_name}:")
    print(_enrich(eff.head(10), lookup).to_string(index=False))

    print(f"\nPlayers only you own ({len(unique_to_me)}):")
    print(_enrich(unique_to_me, lookup).to_string(index=False) if not unique_to_me.empty else "  none")

    print("\nPlayers rivals own that you don't (top 10 by how many rivals):")
    print(_enrich(missing_for_me.head(10), lookup).to_string(index=False)
          if not missing_for_me.empty else "  none")

    print("\nCaptain divergence:")
    if captain["my_captain"] is None:
        print("  Your picks weren't available this run -- can't compare captains.")
    else:
        my_name = _player_name(captain["my_captain"], lookup)
        field_name = _player_name(captain["field_top_captain"], lookup)
        verb = "matched" if not captain["diverges"] else "diverged from"
        print(f"  You captained {my_name}. The field's top pick was {field_name} "
              f"({captain['field_top_count']}/{captain['field_total']}, "
              f"{captain['field_top_pct']}%) -- you {verb} the field.")


if __name__ == "__main__":
    main()
