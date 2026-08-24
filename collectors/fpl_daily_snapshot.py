#!/usr/bin/env python3
"""
FPL Daily Snapshot
====================
Phase 0 of the platform build (see SPEC.md, BRIEFS.md): a light, fast,
no-auth daily capture of price/ownership/form/ep_next/transfers/status for
every player. This is deliberately NOT the full fpl_tracker.py run -- no
fixtures, no manager picks, no Excel workbook. Just the handful of fields
that move every day, which the Tuesday-only cadence was silently losing
forever.

Why daily, not weekly: prices and ownership move every day. A snapshot
taken once a week can never be un-lost -- there is no way to backfill
Wednesday's ownership number after the fact once Tuesday's snapshot has
already overwritten what "current" means. Every day at the old cadence
was unrecoverable data. This script exists to stop that bleeding; it does
not replace fpl_tracker.py's heavier Tuesday run.

Run it once a day:

    python fpl_daily_snapshot.py

No API key, no manager ID needed -- bootstrap-static/ is public and global.
Writes data/fpl/snapshots/YYYY-MM-DD.parquet, one row per player. Safe to
re-run the same day: overwrites that day's file with a fresh fetch rather
than appending or duplicating.

Per the project's global rules: fails loudly (raises) if the API response
is missing an expected field -- that's schema drift, and writing a row
with silently-missing columns would corrupt an archive that can never be
reconstructed later. Fails quietly (retries, then skips with one clear
line and exit 0) on network errors -- a transient blip shouldn't page
anyone or fail the CI run.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE = "https://fantasy.premierleague.com/api"
OUTPUT_DIR = Path("data/fpl/snapshots")

RETRIES = 3
RETRY_DELAY_SECONDS = 5

# What this snapshot exists to capture, per BRIEFS.md Phase 0. Present in a
# "normal" bootstrap-static/ element -- if the API ever drops or renames
# one, that's schema drift and the run should fail loudly, not silently
# write a row with a missing column.
REQUIRED_ELEMENT_FIELDS = [
    "id", "web_name", "team", "element_type", "now_cost",
    "selected_by_percent", "form", "ep_next",
    "transfers_in_event", "transfers_out_event", "status",
]

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
})


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------

def fetch_bootstrap():
    """Retries on network errors; returns None once retries are exhausted
    (a quiet skip -- already logged). Raises immediately on anything that
    isn't a "try again later" case (bad status, malformed JSON)."""
    for attempt in range(1, RETRIES + 1):
        try:
            r = SESSION.get(f"{BASE}/bootstrap-static/", timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as exc:
            if attempt == RETRIES:
                print(f"fpl_daily_snapshot: network error after {RETRIES} attempts "
                      f"({exc}) -- skipping today's snapshot.")
                return None
            print(f"fpl_daily_snapshot: attempt {attempt}/{RETRIES} failed ({exc}), retrying...")
            time.sleep(RETRY_DELAY_SECONDS)
    return None


# ---------------------------------------------------------------------------
# TRANSFORM (pure -- no I/O, so tests can hand these synthetic fixtures)
# ---------------------------------------------------------------------------

def check_schema(elements: list) -> None:
    """Fail loudly, immediately, if the API has dropped or renamed a field
    this snapshot depends on."""
    if not elements:
        raise ValueError("bootstrap-static/ returned zero elements -- API response looks broken.")
    missing = [f for f in REQUIRED_ELEMENT_FIELDS if f not in elements[0]]
    if missing:
        raise KeyError(
            f"Schema drift: bootstrap-static/ elements no longer have {missing}. "
            f"The FPL API changed shape -- fix REQUIRED_ELEMENT_FIELDS and the "
            f"row-building logic in fpl_daily_snapshot.py before trusting this snapshot.")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_snapshot(boot: dict, snapshot_date: str) -> pd.DataFrame:
    elements = boot["elements"]
    check_schema(elements)
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}

    rows = []
    for e in elements:
        rows.append({
            "Snapshot": snapshot_date,
            "ID": e["id"],
            "Player": e["web_name"],
            "Team": teams.get(e["team"], "?"),
            "Pos": POSITIONS.get(e["element_type"], "?"),
            "Price": e["now_cost"] / 10,
            "Owned %": _num(e["selected_by_percent"]),
            "Form": _num(e["form"]),
            "Exp pts next": _num(e["ep_next"]),
            "Transfers in (GW)": e["transfers_in_event"],
            "Transfers out (GW)": e["transfers_out_event"],
            "Status": e["status"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    boot = fetch_bootstrap()
    if boot is None:
        return  # quiet skip -- already logged, exit 0

    snapshot = build_snapshot(boot, today)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{today}.parquet"
    snapshot.to_parquet(out_path, index=False)

    print(f"fpl_daily_snapshot: {len(snapshot)} players -> {out_path}")


if __name__ == "__main__":
    main()
