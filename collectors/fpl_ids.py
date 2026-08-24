#!/usr/bin/env python3
"""
FPL ID Registration
======================
Seeds data/ids/players.csv and data/ids/teams.csv from FPL's
bootstrap-static -- the foundational source for the ID-resolution
crosswalk (core/ids.py, SPEC.md Phase 2). Every other source
(Football-Data, StatsBomb, Understat) gets matched AGAINST this registry
once those collectors exist; this script doesn't do any matching itself,
since FPL's own list of teams and players IS the canonical list to seed
from -- there's nothing to disambiguate when the source is authoritative.

Run it any time FPL's roster might have changed (new signings, promoted
teams) -- most naturally after fpl_tracker.py's weekly run. As a module,
not a plain script, since it imports core.ids, a sibling package:

    python -m collectors.fpl_ids

Idempotent: canonical_id is derived deterministically from FPL's own id,
so re-running just refreshes display names for entities already
registered rather than creating duplicates.
"""

import sys

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core import ids

BASE = "https://fantasy.premierleague.com/api"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
})


def fetch_bootstrap():
    r = SESSION.get(f"{BASE}/bootstrap-static/", timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    boot = fetch_bootstrap()

    teams = ids.register_fpl_teams(boot["teams"])
    players = ids.register_fpl_players(boot["elements"])

    print(f"fpl_ids: {len(teams)} team(s) registered -> {ids.TEAMS_PATH}")
    print(f"fpl_ids: {len(players)} player(s) registered -> {ids.PLAYERS_PATH}")

    print("\nCross-source resolution status (honest, not a failure):")
    for col in ["understat_id", "statsbomb_id", "football_data_name", "understat_name"]:
        filled = teams[col].notna().sum()
        print(f"  teams.{col}: {filled}/{len(teams)} resolved")
    for col in ["understat_id", "statsbomb_id"]:
        filled = players[col].notna().sum()
        print(f"  players.{col}: {filled}/{len(players)} resolved")
    print("  (Expected to read 0 until the Football-Data/StatsBomb/Understat collectors "
          "exist and resolve_team()/resolve_player() have run against real data.)")


if __name__ == "__main__":
    main()
