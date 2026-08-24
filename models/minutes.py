#!/usr/bin/env python3
"""
FPL Minutes Model
==================
Turns the snapshot history into a start probability per player: how likely
are they to start the next gameweek, and are they nailed, a rotation risk,
a bench option, or a benchwarmer.

Everything downstream (rivals, signals, projections) multiplies by this, so
it has to be honest about when it doesn't know rather than guessing. With
fewer than MIN_SETTLED_SNAPSHOTS finished-gameweek snapshots in the store --
globally, or for an individual player who joined the data later -- this
returns None and says why. It does not fall back to a league average.

Run it after fpl_tracker.py, from the same directory:

    python fpl_minutes.py

Reads fpl_history.csv. Writes fpl_minutes.csv, append-only, keyed on the
same UTC snapshot date as the main store. Safe to re-run the same day
(replaces that day's rows rather than duplicating them).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

HISTORY = Path("fpl_history.csv")
OUTPUT = Path("fpl_minutes.csv")

# Below this many finished-gameweek snapshots -- globally, or for one player
# -- there is no signal. Two data points is a line, not a rate.
MIN_SETTLED_SNAPSHOTS = 3

# "Last 4-6 GWs" per the brief. Override with FPL_MINUTES_WINDOW if needed.
RECENT_WINDOW = 5

# Blend weights for start_probability. Recent form matters more than the
# season-long rate; the API's own next-GW estimate is a light tiebreaker.
WEIGHT_RECENT = 0.5
WEIGHT_OVERALL = 0.3
WEIGHT_API_CHANCE = 0.2

# Classification thresholds. A "start" without minutes to back it up (e.g. a
# player subbed off in the 10th minute every week) shouldn't read as nailed.
NAILED_RATE = 0.75
NAILED_MIN_PER_APP = 60
ROTATION_RATE = 0.40


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------

def _settled(history: pd.DataFrame) -> pd.DataFrame:
    """Only finished gameweeks carry a reliable cumulative Starts/Minutes
    reading -- a snapshot taken mid-gameweek is a partial count."""
    return history[history["GW finished"] == True]  # noqa: E712


def _player_periods(rows: pd.DataFrame) -> list[dict]:
    """
    Turn a player's chronological settled-snapshot rows into per-period
    deltas (starts gained, minutes gained, GWs spanned) between consecutive
    snapshots we actually recorded.

    The first snapshot we have for a player is only a valid delta-from-zero
    if it IS the season's GW1 -- otherwise its cumulative total already
    bakes in gameweeks we never observed individually, and treating it as
    "since zero" would overcount. In that case we simply drop it and start
    counting from the next snapshot onward.
    """
    rows = rows.sort_values("Snapshot")
    periods = []
    prev = None
    for _, row in rows.iterrows():
        if prev is None:
            if row["GW"] == 1:
                games = 1
                periods.append({
                    "games": games,
                    "starts": max(int(row["Starts"]), 0),
                    "minutes": max(int(row["Minutes"]), 0),
                })
            prev = row
            continue

        games = int(row["GW"]) - int(prev["GW"])
        if games <= 0:
            # Same GW re-recorded (same-day re-run) or out-of-order data.
            prev = row
            continue

        periods.append({
            "games": games,
            "starts": max(int(row["Starts"]) - int(prev["Starts"]), 0),
            "minutes": max(int(row["Minutes"]) - int(prev["Minutes"]), 0),
            "gw_end": int(row["GW"]),
        })
        prev = row

    return periods


def _rate(periods: list[dict]) -> tuple[float | None, int, int]:
    games = sum(p["games"] for p in periods)
    starts = sum(p["starts"] for p in periods)
    rate = (starts / games) if games > 0 else None
    return rate, starts, games


def _appearances_and_minutes(periods: list[dict]) -> tuple[int, int]:
    """
    Appearances = starts + inferred sub appearances (minutes gained with no
    start in that period). This undercounts multiple sub-appearances inside
    a single multi-GW period (only possible after a missed snapshot), which
    is an acceptable approximation given what the store actually captures.
    """
    appearances = 0
    minutes = 0
    for p in periods:
        appearances += p["starts"]
        if p["starts"] == 0 and p["minutes"] > 0:
            appearances += 1
        minutes += p["minutes"]
    return appearances, minutes


def _classify(rate: float | None, minutes_per_app: float | None,
              appearances: int) -> str:
    if rate is None:
        return "Insufficient data"
    if rate >= NAILED_RATE and (minutes_per_app is None or minutes_per_app >= NAILED_MIN_PER_APP):
        return "Nailed"
    if rate >= ROTATION_RATE:
        return "Rotation risk"
    if appearances > 0:
        return "Bench"
    return "Benchwarmer"


def _start_probability(rate_recent, rate_overall, chance_next_gw) -> float | None:
    parts = []
    if rate_recent is not None:
        parts.append((rate_recent, WEIGHT_RECENT))
    if rate_overall is not None:
        parts.append((rate_overall, WEIGHT_OVERALL))
    if chance_next_gw is not None and not pd.isna(chance_next_gw):
        parts.append((chance_next_gw / 100, WEIGHT_API_CHANCE))

    if not parts:
        return None
    total_weight = sum(w for _, w in parts)
    return round(sum(v * w for v, w in parts) / total_weight, 3)


def compute_minutes_table(history: pd.DataFrame,
                           recent_window: int = RECENT_WINDOW,
                           min_settled: int = MIN_SETTLED_SNAPSHOTS) -> pd.DataFrame:
    """
    Pure transform: history dataframe in, one row per player out. No file
    I/O, so tests can hand it synthetic fixtures directly.
    """
    settled = _settled(history)
    settled_gws = sorted(settled["GW"].unique())

    # Latest snapshot overall (settled or not) gives us the current player
    # roster, team/position labels and the API's live "Chance next GW".
    latest_snap = history["Snapshot"].max()
    latest = history[history["Snapshot"] == latest_snap].set_index("ID")

    global_note = None
    if len(settled_gws) < min_settled:
        global_note = (f"only {len(settled_gws)} finished-gameweek snapshot(s) "
                        f"recorded; need at least {min_settled}.")

    recent_cutoff = settled_gws[-recent_window] if len(settled_gws) >= recent_window else (
        settled_gws[0] if settled_gws else None)

    rows = []
    for pid, prow in latest.iterrows():
        result = {
            "ID": pid,
            "Player": prow["Player"],
            "Team": prow["Team"],
            "Pos": prow["Pos"],
            "Chance next GW": prow.get("Chance next GW"),
        }

        if global_note is not None:
            rows.append({**result,
                         "Settled snapshots (player)": 0,
                         "Starts (recent)": None, "Games (recent)": None,
                         "Start rate (recent)": None,
                         "Starts (overall)": None, "Games (overall)": None,
                         "Start rate (overall)": None,
                         "Appearances": None, "Minutes per appearance": None,
                         "Classification": "Insufficient data",
                         "Start probability": None,
                         "Note": global_note})
            continue

        player_rows = settled[settled["ID"] == pid]
        if len(player_rows) == 0:
            rows.append({**result,
                         "Settled snapshots (player)": 0,
                         "Starts (recent)": None, "Games (recent)": None,
                         "Start rate (recent)": None,
                         "Starts (overall)": None, "Games (overall)": None,
                         "Start rate (overall)": None,
                         "Appearances": None, "Minutes per appearance": None,
                         "Classification": "Insufficient data",
                         "Start probability": None,
                         "Note": "no finished-gameweek data for this player yet "
                                 "(new signing, or not yet appeared in a snapshot)."})
            continue

        if len(player_rows) < min_settled:
            rows.append({**result,
                         "Settled snapshots (player)": len(player_rows),
                         "Starts (recent)": None, "Games (recent)": None,
                         "Start rate (recent)": None,
                         "Starts (overall)": None, "Games (overall)": None,
                         "Start rate (overall)": None,
                         "Appearances": None, "Minutes per appearance": None,
                         "Classification": "Insufficient data",
                         "Start probability": None,
                         "Note": f"only {len(player_rows)} snapshot(s) for this "
                                 f"player; need at least {min_settled}."})
            continue

        periods = _player_periods(player_rows)
        rate_overall, starts_overall, games_overall = _rate(periods)

        recent_periods = ([p for p in periods if p.get("gw_end", 0) > recent_cutoff]
                           if recent_cutoff is not None else periods)
        rate_recent, starts_recent, games_recent = _rate(recent_periods)

        appearances, minutes_total = _appearances_and_minutes(periods)
        minutes_per_app = round(minutes_total / appearances, 1) if appearances else None

        rate_for_class = rate_recent if rate_recent is not None else rate_overall
        classification = _classify(rate_for_class, minutes_per_app, appearances)
        start_probability = _start_probability(
            rate_recent, rate_overall, result["Chance next GW"])

        note = "" if games_overall > 0 else "no completed period yet for this player."

        rows.append({**result,
                     "Settled snapshots (player)": len(player_rows),
                     "Starts (recent)": starts_recent, "Games (recent)": games_recent,
                     "Start rate (recent)": round(rate_recent, 3) if rate_recent is not None else None,
                     "Starts (overall)": starts_overall, "Games (overall)": games_overall,
                     "Start rate (overall)": round(rate_overall, 3) if rate_overall is not None else None,
                     "Appearances": appearances, "Minutes per appearance": minutes_per_app,
                     "Classification": classification,
                     "Start probability": start_probability,
                     "Note": note})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STORE (append-only, idempotent same-day)
# ---------------------------------------------------------------------------

def update_store(table: pd.DataFrame, gw: int, path: Path) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    snap = table.copy()
    snap.insert(0, "Snapshot", today)
    snap.insert(1, "GW", gw)

    if path.exists():
        prior = pd.read_csv(path)
        before = prior["Snapshot"].nunique()
        prior = prior[prior["Snapshot"] != today]
        if prior["Snapshot"].nunique() < before:
            print(f"fpl_minutes: replacing an earlier run from {today}")
        combined = pd.concat([prior, snap], ignore_index=True)
    else:
        combined = snap
        print("fpl_minutes: first snapshot, creating store")

    combined = combined.sort_values(["Snapshot", "ID"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if not HISTORY.exists():
        print(f"{HISTORY} not found. Run fpl_tracker.py first.")
        sys.exit(1)

    history = pd.read_csv(HISTORY)
    table = compute_minutes_table(history)

    current_gw = int(history["GW"].max())
    combined = update_store(table, current_gw, OUTPUT)

    n_insufficient = (table["Classification"] == "Insufficient data").sum()
    n_ready = len(table) - n_insufficient
    print(f"fpl_minutes: {len(table)} players | {n_ready} with a start "
          f"probability | {n_insufficient} insufficient data -> {OUTPUT}")

    if n_ready == 0:
        settled = _settled(history)
        print(f"  Waiting on data: {len(settled['GW'].unique())} finished-gameweek "
              f"snapshot(s) so far, need at least {MIN_SETTLED_SNAPSHOTS}.")


if __name__ == "__main__":
    main()
