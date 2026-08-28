#!/usr/bin/env python3
"""
Understat Collector
=====================
Shot-based Expected Goals (xG) per match, top five European men's
leagues, 2014/15-present (SPEC.md Phase 6/9 gap-filler; §3 rates
Understat "Medium risk" -- scraped, no licence, could break or be
blocked -- so this collector is isolated and fails quietly, per the
design rule in SPEC.md §3).

Endpoint verified live before writing this (2026-08-27), not assumed
from older tutorials -- most existing "Understat scraper" writeups
describe extracting a `JSON.parse('...')` blob embedded in the page's
raw HTML. That no longer works: understat.com now renders its tables
client-side and fetches data from a backend endpoint after page load.
The real endpoint, confirmed against a live response:

    GET https://understat.com/getLeagueData/{league}/{start_year}
    Headers: User-Agent, X-Requested-With: XMLHttpRequest (required --
             omitting it gets a 404, not the data)

Response is gzip-encoded JSON (requests decodes this transparently;
only matters if you're inspecting a raw response by hand). Body shape:
{"teams": {...}, "players": [...], "dates": [...]}.

`dates` -- one entry per match for the whole season, each side's
title, goals, and xG (both null for a match that hasn't been played
yet) -- is joined against `teams[team_id].history` (per-team, per-
match rows keyed by date, not by an explicit match id) to also pull
each side's own npxG (non-penalty xG -- a cleaner training signal than
raw xG, since penalties are high-variance and not about open-play
quality), npxGA (their non-penalty xG conceded), PPDA (passes allowed
per defensive action -- lower means more aggressive pressing, reported
here as the att/def ratio Understat itself uses), deep completions
(passes completed within ~20m of goal, a buildup-quality proxy), and
xPts (Understat's own expected-points-from-this-match figure). All are
null wherever a match has no history row yet (unplayed fixtures) --
same honest-gap convention as xG itself.

`players` -- season TOTALS per player per team (not match-by-match;
getting per-match player detail would mean a separate request per
match, out of scope here) -- normalize_players() writes these
separately, resolved to a team_id the same way match rows are. The
main use: a player's share of their team's season xG is enough to dock
that team's attack rating when the player's actually missing for a
specific fixture (an FPL status/chance_of_playing lookup, not built
here -- this just makes the data available for it).

Genuine bonus over collectors/football_data.py: the CURRENT season's
`dates` list includes every UNPLAYED fixture too (verified live: 370 of
380 EPL 2026/27 entries were future fixtures with goals/xG both null).
football-data.co.uk never carries forward fixtures at all -- this is
the first source in the project that does, for these five leagues.

Coverage gap, stated plainly rather than papered over: Understat only
covers the "big five" (EPL, La Liga, Bundesliga, Serie A, Ligue 1).
collectors/football_data.py also collects the English Championship
(E1) -- there is no Understat xG for that league, full stop.

Team names are Understat's own naming, resolved through
core.ids.resolve_team() exactly like football_data.py's teams --
same caveat applies: the ID registry is seeded from FPL only, so
teams outside the Premier League will legitimately fail to resolve
and get logged to data/ids/unresolved.csv. Expected, not a bug.

Run it as a module (imports core.ids and collectors.football_data,
both siblings, which only resolve correctly with the repo root on
sys.path):

    python -m collectors.understat
"""

import sys
import time

import requests
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

from core import ids
from collectors.football_data import season_code, current_season_start_year

BASE = "https://understat.com/getLeagueData"
OUTPUT_ROOT = Path("data/xg/understat")
PLAYERS_OUTPUT_ROOT = Path("data/xg/understat_players")

# Same six leagues collectors/football_data.py tracks, minus E1 (the
# Championship) -- Understat simply doesn't cover it. Slugs verified
# live against the real endpoint (both space and underscore forms
# work; underscore avoids any URL-encoding ambiguity).
LEAGUES = {
    "E0": "EPL",
    "SP1": "La_liga",
    "D1": "Bundesliga",
    "I1": "Serie_A",
    "F1": "Ligue_1",
}

EARLIEST_SEASON_START = 2014   # verified live: 2014 returns a full season, 2013 returns none
REQUEST_DELAY_SECONDS = 1.0    # be polite -- this endpoint has no public rate-limit policy
RETRIES = 3
RETRY_DELAY_SECONDS = 5

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",   # required -- omitting this gets a 404
})


def season_start_years(earliest: int = EARLIEST_SEASON_START, today=None) -> list:
    return list(range(earliest, current_season_start_year(today) + 1))


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------

def fetch_league_season(league_slug: str, start_year: int):
    """Parsed JSON dict, or None if this league/season doesn't exist yet
    (Understat returns HTTP 200 with an empty `dates` list for a season
    outside its coverage, not a 404) or after RETRIES network failures
    (a quiet skip, same convention as every other collector here)."""
    url = f"{BASE}/{league_slug}/{start_year}"
    for attempt in range(1, RETRIES + 1):
        try:
            r = SESSION.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            if not data.get("dates"):
                return None
            return data
        except requests.exceptions.RequestException as exc:
            if attempt == RETRIES:
                print(f"understat: network error fetching {league_slug} {start_year} "
                      f"after {RETRIES} attempts ({exc}) -- skipping.")
                return None
            time.sleep(RETRY_DELAY_SECONDS)
        except ValueError as exc:   # malformed/non-JSON response -- schema drift, not a network blip
            print(f"understat: unexpected response for {league_slug} {start_year} ({exc}) -- skipping.")
            return None
    return None


# ---------------------------------------------------------------------------
# NORMALIZE
# ---------------------------------------------------------------------------

def _nullable_float(value):
    return None if value is None else float(value)


def _nullable_int(value):
    return None if value is None else int(value)


def _build_history_lookup(data: dict) -> dict:
    """{(team_id, datetime_string): history_row} -- Understat's per-team
    history has no explicit match id, only a date, but a team plays at
    most one match on any given date, so (team_id, date) uniquely
    identifies one history row."""
    lookup = {}
    for team_id, team in data.get("teams", {}).items():
        for h in team.get("history", []):
            lookup[(team_id, h["date"])] = h
    return lookup


def _ppda_ratio(ppda: dict | None) -> float | None:
    """PPDA is reported as raw {att, def} counts -- att/def is the
    actual PPDA figure (passes the opponent was allowed per defensive
    action); a team with def=0 has taken no defensive actions in that
    match at all, which is degenerate, not a divide-by-zero to paper over."""
    if not ppda or not ppda.get("def"):
        return None
    return ppda["att"] / ppda["def"]


def normalize_season(data: dict, league_code: str, start_year: int) -> pd.DataFrame:
    """One row per match (played or not) for the whole season. Goals/xG
    (and every history-derived field below) are null for a fixture that
    hasn't been played yet -- that's the live-fixture-list bonus
    described in the module docstring, not missing data to be dropped."""
    history_lookup = _build_history_lookup(data)
    rows = []
    for m in data["dates"]:
        home_raw, away_raw = m["h"]["title"], m["a"]["title"]
        forecast = m.get("forecast") or {}
        home_hist = history_lookup.get((m["h"]["id"], m["datetime"])) or {}
        away_hist = history_lookup.get((m["a"]["id"], m["datetime"])) or {}
        rows.append({
            "league": league_code,
            "season": season_code(start_year),
            "date": m["datetime"][:10],
            "home_team_raw": home_raw,
            "away_team_raw": away_raw,
            "home_team_id": ids.resolve_team(home_raw, source="understat"),
            "away_team_id": ids.resolve_team(away_raw, source="understat"),
            "is_result": bool(m["isResult"]),
            "home_goals": _nullable_int(m["goals"]["h"]),
            "away_goals": _nullable_int(m["goals"]["a"]),
            "home_xg": _nullable_float(m["xG"]["h"]),
            "away_xg": _nullable_float(m["xG"]["a"]),
            "home_npxg": _nullable_float(home_hist.get("npxG")),
            "away_npxg": _nullable_float(away_hist.get("npxG")),
            "home_npxga": _nullable_float(home_hist.get("npxGA")),
            "away_npxga": _nullable_float(away_hist.get("npxGA")),
            "home_ppda": _ppda_ratio(home_hist.get("ppda")),
            "away_ppda": _ppda_ratio(away_hist.get("ppda")),
            "home_deep": _nullable_int(home_hist.get("deep")),
            "away_deep": _nullable_int(away_hist.get("deep")),
            "home_xpts": _nullable_float(home_hist.get("xpts")),
            "away_xpts": _nullable_float(away_hist.get("xpts")),
            # Understat's own pre-match model, free in the same payload --
            # not used by anything yet, kept as a future comparison point.
            "understat_forecast_home_win": _nullable_float(forecast.get("w")),
            "understat_forecast_draw": _nullable_float(forecast.get("d")),
            "understat_forecast_away_win": _nullable_float(forecast.get("l")),
        })
    return pd.DataFrame(rows)


def normalize_players(data: dict, league_code: str, start_year: int) -> pd.DataFrame:
    """One row per player per team for the season (a mid-season transfer
    produces two rows, one per team -- Understat's own shape, not
    something to merge here). team_id resolved the same way match rows
    are (core.ids, FPL/historical-seeded registry)."""
    rows = []
    for p in data.get("players", []):
        rows.append({
            "league": league_code,
            "season": season_code(start_year),
            "player_name": p["player_name"],
            "team": p["team_title"],
            "team_id": ids.resolve_team(p["team_title"], source="understat"),
            "minutes": _nullable_int(p.get("time")),
            "games": _nullable_int(p.get("games")),
            "goals": _nullable_int(p.get("goals")),
            "xg": _nullable_float(p.get("xG")),
            "npg": _nullable_int(p.get("npg")),
            "npxg": _nullable_float(p.get("npxG")),
            "assists": _nullable_int(p.get("assists")),
            "xa": _nullable_float(p.get("xA")),
            "xg_chain": _nullable_float(p.get("xGChain")),
            "xg_buildup": _nullable_float(p.get("xGBuildup")),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ORCHESTRATION (fetch-once-per-historical-season, always-refresh-current)
# ---------------------------------------------------------------------------

def collect_league_season(league_code: str, league_slug: str, start_year: int,
                          current_start_year: int) -> str:
    """Returns 'fetched', 'skipped_cached', or 'skipped_missing'. A
    completed season already on disk is never re-fetched; the current
    season always is, since its fixture list and results change daily.
    Writes both the match-level file and the player-season-totals file
    from the same fetched response -- one network call, two outputs."""
    out_path = OUTPUT_ROOT / league_code / f"{season_code(start_year)}.csv"
    players_out_path = PLAYERS_OUTPUT_ROOT / league_code / f"{season_code(start_year)}.csv"
    is_current = start_year == current_start_year

    if out_path.exists() and not is_current:
        return "skipped_cached"

    data = fetch_league_season(league_slug, start_year)
    if data is None:
        return "skipped_missing"

    normalized = normalize_season(data, league_code, start_year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(out_path, index=False)

    players = normalize_players(data, league_code, start_year)
    players_out_path.parent.mkdir(parents=True, exist_ok=True)
    players.to_csv(players_out_path, index=False)

    return "fetched"


def main():
    current_start_year = current_season_start_year()
    stats = {"fetched": 0, "skipped_cached": 0, "skipped_missing": 0}

    for league_code, league_slug in LEAGUES.items():
        print(f"understat: {league_slug} ({league_code})")
        for start_year in season_start_years():
            status = collect_league_season(league_code, league_slug, start_year, current_start_year)
            stats[status] += 1
            if status == "fetched":
                time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nunderstat: {stats['fetched']} fetched, {stats['skipped_cached']} already "
          f"cached, {stats['skipped_missing']} unavailable (league/season doesn't exist)")


if __name__ == "__main__":
    main()
