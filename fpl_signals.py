#!/usr/bin/env python3
"""
FPL Signals
============
Three cheap wins from data the tracker already collects:

1. Set-piece changes -- diff consecutive snapshots on penalty/corner/free-kick
   order. Losing a duty is a sell signal that arrives before the points drop;
   gaining one is a buy signal before the ownership catches up.
2. Fixture-adjusted form -- raw Form treats 6 points against a relegation
   side the same as 6 against a title contender. Divide by next-opponent FDR.
3. Price momentum -- surface players close to a rise or fall. Tiebreaker
   only, never a buy/sell reason on its own.

Run it after fpl_tracker.py:

    python fpl_signals.py

Reads fpl_history.csv (read-only -- never modifies it) and fetches the
live fixtures/ and bootstrap-static/ endpoints for next-gameweek FDR, since
that isn't something the history store needs to retain historically.

Writes signals.csv, append-only, keyed on the same UTC snapshot-date
convention as the main store. Safe to re-run the same day.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

# Windows consoles default to a codepage that can't render every player
# name (e.g. accented or non-Latin characters). GitHub Actions (Linux) is
# already UTF-8; this just makes local runs on Windows just as tolerant.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

HISTORY = Path("fpl_history.csv")
OUTPUT = Path("signals.csv")

ORDER_COLUMNS = ["Pens order", "Corners/IFK order", "Direct FK order"]

# |Rise/fall %| at or above this is "close enough to flag". A tiebreaker
# threshold, not a prediction -- see the reminder printed with the digest.
PRICE_CHANGE_THRESHOLD = 40

BASE = "https://fantasy.premierleague.com/api"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
})


# ---------------------------------------------------------------------------
# FETCH (live -- next-GW FDR isn't retained in fpl_history.csv, and doesn't
# need to be; it's only ever meaningful as "next fixture", not historically)
# ---------------------------------------------------------------------------

def get(path, params=None):
    r = SESSION.get(f"{BASE}/{path}", params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def fetch_next_gw_fdr():
    """Per-team average FDR and opponent label for the next gameweek.

    A team with no fixture that gameweek (a blank) simply has no entry --
    callers must treat that as "no signal", not zero difficulty.
    """
    boot = get("bootstrap-static/")
    fixtures = get("fixtures/")
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}

    next_gw = next((e["id"] for e in boot["events"] if e["is_next"]), None)
    if next_gw is None:
        next_gw = next((e["id"] for e in boot["events"]
                         if e["is_current"] and not e["finished"]), None)
    if next_gw is None:
        return {}, {}, None

    fdrs, opponents = {}, {}
    for f in fixtures:
        if f.get("event") != next_gw:
            continue
        h, a = teams.get(f["team_h"]), teams.get(f["team_a"])
        if h:
            fdrs.setdefault(h, []).append(f["team_h_difficulty"])
            opponents.setdefault(h, []).append(f"{a} (H)")
        if a:
            fdrs.setdefault(a, []).append(f["team_a_difficulty"])
            opponents.setdefault(a, []).append(f"{h} (A)")

    avg_fdr = {team: round(sum(v) / len(v), 2) for team, v in fdrs.items()}
    opponent_label = {team: " + ".join(v) for team, v in opponents.items()}
    return avg_fdr, opponent_label, next_gw


# ---------------------------------------------------------------------------
# TRANSFORM (pure -- no I/O, so tests can hand these synthetic fixtures)
# ---------------------------------------------------------------------------

def detect_setpiece_changes(history: pd.DataFrame):
    """
    Diff the two most recent snapshots on set-piece order. Returns
    (changes_df, note).

    `note` covers two distinct cold-start cases: fewer than two snapshots
    exist at all, or a column has no prior data to diff against (e.g. it
    was only just added to SNAPSHOT_FIELDS -- every player would otherwise
    read as "gained duty" purely because tracking started, not because
    anything changed). The second case skips only the affected column(s);
    the rest still diff normally.
    """
    snaps = sorted(history["Snapshot"].unique())
    if len(snaps) < 2:
        empty = pd.DataFrame(columns=["ID", "Player", "Team", "Metric",
                                       "Old value", "New value", "Direction"])
        return empty, f"only {len(snaps)} snapshot(s) recorded; need at least 2 to diff."

    prev_snap, curr_snap = snaps[-2], snaps[-1]
    prev = history[history["Snapshot"] == prev_snap].set_index("ID")
    curr = history[history["Snapshot"] == curr_snap].set_index("ID")
    prev = prev.reindex(curr.index)  # new-to-data players compare against NaN, not dropped

    rows = []
    skipped_columns = [col for col in ORDER_COLUMNS
                        if col not in prev.columns or prev[col].notna().sum() == 0]

    for col in ORDER_COLUMNS:
        if col in skipped_columns:
            continue
        for pid in curr.index:
            old, new = prev.loc[pid, col], curr.loc[pid, col]
            old_na, new_na = pd.isna(old), pd.isna(new)
            if old_na and new_na:
                continue
            if not old_na and not new_na and float(old) == float(new):
                continue

            if old_na:
                direction = "Gained duty"
            elif new_na:
                direction = "Lost duty"
            elif new < old:
                direction = "Promoted (higher priority)"
            else:
                direction = "Demoted (lower priority)"

            rows.append({
                "ID": pid, "Player": curr.loc[pid, "Player"], "Team": curr.loc[pid, "Team"],
                "Metric": col,
                "Old value": None if old_na else old, "New value": None if new_na else new,
                "Direction": direction,
            })

    note = (f"no prior data for {', '.join(skipped_columns)} yet -- column(s) newly "
            f"tracked or previous snapshot predates them; skipped this run."
            if skipped_columns else None)
    return pd.DataFrame(rows), note


def fixture_adjusted_form(latest: pd.DataFrame, fdr_by_team: dict,
                           opponent_by_team: dict) -> pd.DataFrame:
    """
    Form / next-opponent FDR. Restricted to players with minutes on the
    board -- the API pins Form at 0 for anyone unused, so including them
    would just be noise, not signal.
    """
    played = latest[latest["Minutes"] > 0].copy()
    rows = []
    for _, r in played.iterrows():
        fdr = fdr_by_team.get(r["Team"])
        form = r["Form"]
        if fdr is None:
            rows.append({"ID": r["ID"], "Player": r["Player"], "Team": r["Team"],
                         "Form": form, "Next fixture FDR": None, "Opponent": None,
                         "Fixture-adjusted form": None, "Note": "blank gameweek"})
            continue
        adjusted = round(form / fdr, 3) if pd.notna(form) else None
        rows.append({"ID": r["ID"], "Player": r["Player"], "Team": r["Team"],
                     "Form": form, "Next fixture FDR": fdr,
                     "Opponent": opponent_by_team.get(r["Team"]),
                     "Fixture-adjusted form": adjusted, "Note": ""})
    return pd.DataFrame(rows)


def price_momentum_watch(latest: pd.DataFrame, threshold: float = PRICE_CHANGE_THRESHOLD) -> pd.DataFrame:
    """Players close to a price change. A tiebreaker, never surfaced as a
    standalone buy/sell reason -- callers should keep saying so."""
    df = latest.dropna(subset=["Rise/fall %"]).copy()
    df = df[df["Rise/fall %"].abs() >= threshold]
    if df.empty:
        return pd.DataFrame(columns=["ID", "Player", "Team", "Price", "Rise/fall %",
                                      "Change likelihood", "Direction"])
    df["Direction"] = df["Rise/fall %"].apply(lambda v: "Rise watch" if v > 0 else "Fall watch")
    df = df.sort_values("Rise/fall %", key=lambda s: s.abs(), ascending=False)
    return df[["ID", "Player", "Team", "Price", "Rise/fall %",
               "Change likelihood", "Direction"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# STORE (append-only, idempotent same-day)
# ---------------------------------------------------------------------------

def build_signals_table(setpiece_df, form_df, price_df, snapshot, gw) -> pd.DataFrame:
    """Flatten the three signal types into one long table for storage."""
    rows = []
    for _, r in setpiece_df.iterrows():
        rows.append({"Snapshot": snapshot, "GW": gw, "Signal": "Set-piece change",
                     "ID": r["ID"], "Player": r["Player"], "Team": r["Team"],
                     "Metric": r["Metric"], "Old value": r["Old value"],
                     "New value": r["New value"], "Note": r["Direction"]})
    for _, r in form_df.iterrows():
        rows.append({"Snapshot": snapshot, "GW": gw, "Signal": "Fixture-adjusted form",
                     "ID": r["ID"], "Player": r["Player"], "Team": r["Team"],
                     "Metric": "Form / next-opponent FDR", "Old value": r["Form"],
                     "New value": r["Fixture-adjusted form"],
                     "Note": r["Note"] or f"vs {r['Opponent']} (FDR {r['Next fixture FDR']})"})
    for _, r in price_df.iterrows():
        rows.append({"Snapshot": snapshot, "GW": gw, "Signal": "Price momentum",
                     "ID": r["ID"], "Player": r["Player"], "Team": r["Team"],
                     "Metric": "Rise/fall %", "Old value": None, "New value": r["Rise/fall %"],
                     "Note": f"{r['Direction']} (likelihood {r['Change likelihood']})"})

    cols = ["Snapshot", "GW", "Signal", "ID", "Player", "Team",
            "Metric", "Old value", "New value", "Note"]
    return pd.DataFrame(rows, columns=cols)


def update_store(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    today = table["Snapshot"].iloc[0] if not table.empty else datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if path.exists():
        prior = pd.read_csv(path)
        before = prior["Snapshot"].nunique()
        prior = prior[prior["Snapshot"] != today]
        if prior["Snapshot"].nunique() < before:
            print(f"fpl_signals: replacing an earlier run from {today}")
        combined = pd.concat([prior, table], ignore_index=True)
    else:
        combined = table
        print("fpl_signals: first snapshot, creating store")

    combined.to_csv(path, index=False)
    return combined


# ---------------------------------------------------------------------------
# DIGEST
# ---------------------------------------------------------------------------

def print_digest(setpiece_df, setpiece_note, form_df, price_df, prev_snap, curr_snap, next_gw):
    print(f"\n=== Set-piece changes ({prev_snap} -> {curr_snap}) ===")
    if setpiece_note:
        print(f"  Note: {setpiece_note}")
    if setpiece_df.empty:
        if not setpiece_note:
            print("  No changes detected.")
    else:
        for _, r in setpiece_df.iterrows():
            print(f"  {r['Player']} ({r['Team']}) -- {r['Metric']}: "
                  f"{r['Old value']} -> {r['New value']}  [{r['Direction']}]")

    print(f"\n=== Fixture-adjusted form, next GW{next_gw} (top 15) ===")
    if form_df.empty:
        print("  No data (no minutes played yet, or fixture fetch unavailable).")
    else:
        ranked = form_df.dropna(subset=["Fixture-adjusted form"])
        top = ranked.sort_values("Fixture-adjusted form", ascending=False).head(15)
        for _, r in top.iterrows():
            print(f"  {r['Player']} ({r['Team']}) -- Form {r['Form']} vs {r['Opponent']} "
                  f"(FDR {r['Next fixture FDR']}) -> {r['Fixture-adjusted form']}")

    print(f"\n=== Price momentum watch (|Rise/fall %| >= {PRICE_CHANGE_THRESHOLD}) ===")
    if price_df.empty:
        print("  No players close to a price change.")
    else:
        for _, r in price_df.iterrows():
            print(f"  {r['Player']} ({r['Team']}, £{r['Price']}m) -- {r['Direction']} "
                  f"{r['Rise/fall %']}% (likelihood {r['Change likelihood']})")
    print("\n  Reminder: price momentum is a tiebreaker, never a buy/sell reason on its own.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if not HISTORY.exists():
        print(f"{HISTORY} not found. Run fpl_tracker.py first.")
        sys.exit(1)

    history = pd.read_csv(HISTORY)
    setpiece_df, setpiece_note = detect_setpiece_changes(history)

    print("Fetching fixtures for fixture-adjusted form...")
    fdr_by_team, opponent_by_team, next_gw = fetch_next_gw_fdr()

    snaps = sorted(history["Snapshot"].unique())
    prev_snap = snaps[-2] if len(snaps) >= 2 else None
    curr_snap = snaps[-1]
    latest = history[history["Snapshot"] == curr_snap]
    curr_gw = int(latest["GW"].max())

    form_df = fixture_adjusted_form(latest, fdr_by_team, opponent_by_team)
    price_df = price_momentum_watch(latest)

    print_digest(setpiece_df, setpiece_note, form_df, price_df, prev_snap, curr_snap, next_gw)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    table = build_signals_table(setpiece_df, form_df, price_df, today, curr_gw)
    update_store(table, OUTPUT)
    print(f"\nfpl_signals: {len(table)} signal row(s) this run -> {OUTPUT}")


if __name__ == "__main__":
    main()
