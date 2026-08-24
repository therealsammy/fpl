#!/usr/bin/env python3
"""
Football-Data.co.uk Collector
================================
~22 leagues, results and odds, 1993-present. The cheapest large win in
the project: static per-season CSVs, no scraping, decades of history
including closing odds -- and a completed season's data doesn't change,
so it's fetched once and never re-fetched (only the current, still-in-
progress season is refreshed on every run).

Column schema drifts hard across eras -- verified against real files
(1993/94, 2000/01, 2010/11, 2015/16, 2024/25) before writing this, not
assumed from memory:
  - 1993/94: results only. No odds columns exist at all.
  - ~2000/01: a handful of individual bookmakers' 1X2 odds (GB, IW, LB,
    SB, WH). No over/under market, no consensus/average column.
  - ~2010-2016: a "Bb" panel -- BbAvH/BbAvD/BbAvA for 1X2 and
    BbAv>2.5/BbAv<2.5 for over/under, but no individual-bookmaker
    over/under columns to fall back on.
  - modern (~2019+): AvgH/AvgD/AvgA and Avg>2.5/Avg<2.5 directly, plus
    both opening and closing prices per bookmaker.
This collector tries the best available column set per season and
accepts that older files simply won't have every market -- that's a
real, expected gap (null probabilities), not something to paper over or
crash on. The only columns treated as required in every era are the
match result itself (date, teams, goals, result).

Team names are football-data.co.uk's own naming, resolved through
core.ids.resolve_team(). The ID registry is currently seeded from FPL
only (Phase 2) -- teams outside the Premier League will legitimately
fail to resolve right now and get logged to data/ids/unresolved.csv.
That's expected, not a bug: the raw name is stored either way, and
resolution improves as the registry gets enriched from other sources.

Devigging reuses collectors.fpl_odds's proportional_devig/shin_devig --
the same validated math built for FPL's own odds collector, not
reimplemented here.

Run it as a module (imports core.ids and collectors.fpl_odds, both
siblings, which only resolve correctly with the repo root on sys.path):

    python -m collectors.football_data
"""

import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core import ids
from collectors.fpl_odds import proportional_devig, shin_devig

BASE = "https://www.football-data.co.uk/mmz4281"
OUTPUT_ROOT = Path("data/odds/football_data")

# Acceptance bar (BRIEFS.md Phase 3): top two English tiers + the big
# five European leagues. football-data.co.uk covers ~22 leagues in
# total -- more can be added here later without changing anything else.
LEAGUES = {
    "E0": "English Premier League",
    "E1": "English Championship",
    "SP1": "Spanish La Liga",
    "D1": "German Bundesliga",
    "I1": "Italian Serie A",
    "F1": "French Ligue 1",
}

EARLIEST_SEASON_START = 1993
REQUEST_DELAY_SECONDS = 1.0   # dozens of requests per run -- be polite
RETRIES = 3
RETRY_DELAY_SECONDS = 5

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Empirically verified individual-bookmaker column prefixes, spanning
# files actually fetched from 2000/01 through 2024/25 -- see module
# docstring. Consensus columns (Avg*/BbAv*) are always preferred over
# these when present.
H2H_BOOKMAKERS = ["B365", "BW", "GB", "IW", "LB", "SB", "WH", "SJ", "VC", "BS", "PS", "BF", "1XB"]
OU25_BOOKMAKERS = ["B365", "P", "BFE"]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
})


# ---------------------------------------------------------------------------
# SEASON CODES
# ---------------------------------------------------------------------------

def season_code(start_year: int) -> str:
    """1993 -> '9394', 1999 -> '9900', 2024 -> '2425'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start_year(today=None) -> int:
    """The Premier League season starts in August; treat July onward as
    already being the new season for collection purposes."""
    today = today or datetime.now(timezone.utc).date()
    return today.year if today.month >= 7 else today.year - 1


def season_start_years(earliest: int = EARLIEST_SEASON_START, today=None) -> list:
    return list(range(earliest, current_season_start_year(today) + 1))


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------

def fetch_season_csv(league_code: str, start_year: int):
    """
    Raw CSV text, or None if this league/season combo doesn't exist, or
    after RETRIES network failures (a quiet skip, same convention as
    every other collector in this project).

    "Doesn't exist" isn't always a 404: football-data.co.uk returns HTTP
    300 "Multiple Choices" with an HTML suggestions page for a not-yet-
    published season -- verified live against the (at the time) unpublished
    current season's file. `requests.raise_for_status()` doesn't treat 3xx
    as an error, so this checks Content-Type explicitly rather than
    trusting the status code alone -- otherwise that HTML page gets handed
    to the CSV parser as if it were data.
    """
    url = f"{BASE}/{season_code(start_year)}/{league_code}.csv"
    for attempt in range(1, RETRIES + 1):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            if "csv" not in r.headers.get("Content-Type", "").lower():
                return None
            return r.text
        except requests.exceptions.RequestException as exc:
            if attempt == RETRIES:
                print(f"football_data: network error fetching {league_code} "
                      f"{season_code(start_year)} after {RETRIES} attempts ({exc}) -- skipping.")
                return None
            time.sleep(RETRY_DELAY_SECONDS)
    return None


# ---------------------------------------------------------------------------
# MARKET CONSENSUS (pure -- best available column set per era)
# ---------------------------------------------------------------------------

def compute_1x2_consensus(df: pd.DataFrame):
    """DataFrame with home_price/draw_price/away_price, or None if this
    file has no usable 1X2 odds at all (e.g. 1993/94)."""
    for h, d, a in [("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA")]:
        if h in df.columns:
            return pd.DataFrame({"home_price": df[h], "draw_price": df[d], "away_price": df[a]})

    h_cols = [f"{p}H" for p in H2H_BOOKMAKERS if f"{p}H" in df.columns]
    d_cols = [f"{p}D" for p in H2H_BOOKMAKERS if f"{p}D" in df.columns]
    a_cols = [f"{p}A" for p in H2H_BOOKMAKERS if f"{p}A" in df.columns]
    if not h_cols:
        return None
    return pd.DataFrame({
        "home_price": df[h_cols].median(axis=1),
        "draw_price": df[d_cols].median(axis=1),
        "away_price": df[a_cols].median(axis=1),
    })


def compute_ou25_consensus(df: pd.DataFrame):
    """DataFrame with over_price/under_price, or None if unavailable."""
    for o, u in [("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5")]:
        if o in df.columns:
            return pd.DataFrame({"over_price": df[o], "under_price": df[u]})

    o_cols = [f"{p}>2.5" for p in OU25_BOOKMAKERS if f"{p}>2.5" in df.columns]
    u_cols = [f"{p}<2.5" for p in OU25_BOOKMAKERS if f"{p}<2.5" in df.columns]
    if not o_cols:
        return None
    return pd.DataFrame({
        "over_price": df[o_cols].median(axis=1),
        "under_price": df[u_cols].median(axis=1),
    })


# ---------------------------------------------------------------------------
# DEVIG (reuses collectors.fpl_odds's validated proportional/Shin math)
# ---------------------------------------------------------------------------

def devig_1x2(consensus: pd.DataFrame) -> pd.DataFrame:
    def _row(r):
        prices = [r["home_price"], r["draw_price"], r["away_price"]]
        if any(pd.isna(p) or p <= 1 for p in prices):
            return pd.Series({"prob_home_proportional": None, "prob_draw_proportional": None,
                              "prob_away_proportional": None, "prob_home_shin": None,
                              "prob_draw_shin": None, "prob_away_shin": None, "shin_z": None})
        prop = proportional_devig(prices)
        shin, z = shin_devig(prices)
        return pd.Series({
            "prob_home_proportional": prop[0], "prob_draw_proportional": prop[1],
            "prob_away_proportional": prop[2], "prob_home_shin": shin[0],
            "prob_draw_shin": shin[1], "prob_away_shin": shin[2], "shin_z": z,
        })
    return consensus.apply(_row, axis=1)


def devig_ou25(consensus: pd.DataFrame) -> pd.DataFrame:
    def _row(r):
        prices = [r["over_price"], r["under_price"]]
        if any(pd.isna(p) or p <= 1 for p in prices):
            return pd.Series({"prob_over25_proportional": None, "prob_under25_proportional": None,
                              "prob_over25_shin": None, "prob_under25_shin": None})
        prop = proportional_devig(prices)
        shin, _z = shin_devig(prices)
        return pd.Series({
            "prob_over25_proportional": prop[0], "prob_under25_proportional": prop[1],
            "prob_over25_shin": shin[0], "prob_under25_shin": shin[1],
        })
    return consensus.apply(_row, axis=1)


def _fix_ragged_rows(raw_csv: str) -> str:
    """
    Some older files have individual rows with MORE comma-separated
    fields than the header -- verified in the real 2003/04 Premier League
    file: most rows have 57 fields, but some have 62 or 72, all the
    excess being empty trailing commas. pandas' C parser raises on
    inconsistent row widths rather than tolerating this, so each
    over-long row is truncated to the header's width before parsing --
    the real data is always in the first N fields; anything past that is
    padding, not additional columns. Short rows are left alone; pandas
    pads those with NaN natively.
    """
    lines = raw_csv.splitlines()
    if not lines:
        return raw_csv
    expected = len(lines[0].split(","))
    fixed = [lines[0]]
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) > expected:
            fields = fields[:expected]
        fixed.append(",".join(fields))
    return "\n".join(fixed)


# ---------------------------------------------------------------------------
# NORMALIZE (pure -- given raw CSV text, no I/O beyond parsing it)
# ---------------------------------------------------------------------------

def normalize_season(raw_csv: str, league_code: str, start_year: int) -> pd.DataFrame:
    """
    Parses one season's raw CSV text into the common schema. Raises if
    even the always-present result columns are missing -- that's schema
    drift, not an expected era gap. Odds columns are always best-effort:
    absent means null probabilities for that season, not a failure.
    """
    df = pd.read_csv(io.StringIO(_fix_ragged_rows(raw_csv)))
    df = df.dropna(axis=1, how="all")   # trailing empty columns in older files
    df = df.dropna(subset=["Date"])     # trailing blank rows some files have

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"football_data: {league_code} {season_code(start_year)} is missing required "
            f"column(s) {missing} -- schema drift, not an expected era gap.")

    out = pd.DataFrame({
        "league": league_code,
        "season": season_code(start_year),
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.strftime("%Y-%m-%d"),
        "home_team_raw": df["HomeTeam"],
        "away_team_raw": df["AwayTeam"],
        "home_goals": df["FTHG"],
        "away_goals": df["FTAG"],
        "result": df["FTR"],
    })
    out["home_team_id"] = out["home_team_raw"].apply(lambda n: ids.resolve_team(n, source="football_data"))
    out["away_team_id"] = out["away_team_raw"].apply(lambda n: ids.resolve_team(n, source="football_data"))

    h2h = compute_1x2_consensus(df)
    if h2h is not None:
        out = pd.concat([out.reset_index(drop=True), devig_1x2(h2h).reset_index(drop=True)], axis=1)
    else:
        for c in ["prob_home_proportional", "prob_draw_proportional", "prob_away_proportional",
                  "prob_home_shin", "prob_draw_shin", "prob_away_shin", "shin_z"]:
            out[c] = None

    ou = compute_ou25_consensus(df)
    if ou is not None:
        out = pd.concat([out.reset_index(drop=True), devig_ou25(ou).reset_index(drop=True)], axis=1)
    else:
        for c in ["prob_over25_proportional", "prob_under25_proportional",
                  "prob_over25_shin", "prob_under25_shin"]:
            out[c] = None

    return out


# ---------------------------------------------------------------------------
# ORCHESTRATION (fetch-once-per-historical-season, always-refresh-current)
# ---------------------------------------------------------------------------

def collect_league_season(league_code: str, start_year: int, current_start_year: int) -> str:
    """Returns 'fetched', 'skipped_cached', or 'skipped_missing'. A
    completed season already on disk is never re-fetched; the current
    season always is."""
    out_path = OUTPUT_ROOT / league_code / f"{season_code(start_year)}.csv"
    is_current = start_year == current_start_year

    if out_path.exists() and not is_current:
        return "skipped_cached"

    raw = fetch_season_csv(league_code, start_year)
    if raw is None:
        return "skipped_missing"

    normalized = normalize_season(raw, league_code, start_year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(out_path, index=False)
    return "fetched"


def main():
    current_start_year = current_season_start_year()
    stats = {"fetched": 0, "skipped_cached": 0, "skipped_missing": 0}

    for league_code, league_name in LEAGUES.items():
        print(f"football_data: {league_name} ({league_code})")
        for start_year in season_start_years():
            status = collect_league_season(league_code, start_year, current_start_year)
            stats[status] += 1
            if status == "fetched":
                time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nfootball_data: {stats['fetched']} fetched, {stats['skipped_cached']} already "
          f"cached, {stats['skipped_missing']} unavailable (league/season doesn't exist)")


if __name__ == "__main__":
    main()
