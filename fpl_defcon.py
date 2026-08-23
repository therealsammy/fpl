#!/usr/bin/env python3
"""
FPL DEFCON vs Territory -- a hypothesis test, not a feature
==============================================================
Hypothesis: defensive contributions scale with time spent without the ball,
so a player in a low-block side facing a possession-heavy opponent earns
more DEFCON than the same player facing another low block.

This is a TEST. It reports a correlation with a confidence interval and a
sample size, and it says plainly when the sample is too small or the effect
is too weak to act on. It does not build a projection on top of a null
result -- that decision belongs to whoever reads the report, not to this
script. A confident-looking number from 40 rows is worse than an honest
"not enough data yet".

Territory proxy: opponent FDR and home/away, from the fixtures/ endpoint.
Note: a possession-share stat (e.g. from FBref) would be a much more direct
measure of "time spent without the ball" than FDR -- FDR blends attack and
defence strength and isn't built for this. This is the proxy available
without scraping a paywalled source, per the brief's constraints.

Run it after fpl_tracker.py:

    python fpl_defcon.py

Reads fpl_history.csv (read-only) and fetches bootstrap-static/ + fixtures/
live for the historical opponent-per-gameweek lookup (fpl_history.csv
doesn't store fixtures, and doesn't need to -- the fixtures/ endpoint
already retains the whole season, past and future).

Writes:
  defcon_report.csv   append-only, one row per run, idempotent same-day --
                       the tracked verdict over time as more data arrives.
  defcon_dataset.csv  disposable, fully overwritten each run -- the
                       per-period observations behind the correlation, for
                       inspection. Fully re-derivable from fpl_history.csv
                       plus the live API, so it isn't append-only.
"""

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

HISTORY = Path("fpl_history.csv")
REPORT = Path("defcon_report.csv")
DATASET = Path("defcon_dataset.csv")

# A period with fewer minutes than this yields an unreliable per-90 rate --
# e.g. 1 DEFCON in 9 minutes implies a per-90 rate of 10, which tells you
# nothing about the player's true rate.
MIN_PERIOD_MINUTES = 30

# The brief's own example of "too small to trust" is 40 rows. Set the floor
# just above that.
MIN_SAMPLE_SIZE = 50

Z_95 = 1.959964  # two-tailed 95% critical value, standard normal

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
    r = SESSION.get(f"{BASE}/{path}", params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def fetch_fixtures_lookup() -> dict:
    """
    (team, gw) -> {fdr, home, opponent, double}. Covers the whole season,
    past and future, in one call -- the fixtures/ endpoint doesn't forget
    finished fixtures the way point-in-time player stats do.

    `double` marks a gameweek where the team played twice: DEFCON earned
    that gameweek can't be cleanly attributed to one opponent's territory,
    so callers should skip those rather than guess.
    """
    boot = get("bootstrap-static/")
    fixtures = get("fixtures/")
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}

    counts = {}
    for f in fixtures:
        gw = f.get("event")
        if gw is None:
            continue
        for team_id in (f["team_h"], f["team_a"]):
            team = teams.get(team_id)
            if team:
                counts[(team, gw)] = counts.get((team, gw), 0) + 1

    lookup = {}
    for f in fixtures:
        gw = f.get("event")
        if gw is None:
            continue
        h, a = teams.get(f["team_h"]), teams.get(f["team_a"])
        if h:
            lookup[(h, gw)] = {"fdr": f["team_h_difficulty"], "home": True,
                                "opponent": a, "double": counts.get((h, gw), 0) > 1}
        if a:
            lookup[(a, gw)] = {"fdr": f["team_a_difficulty"], "home": False,
                                "opponent": h, "double": counts.get((a, gw), 0) > 1}
    return lookup


# ---------------------------------------------------------------------------
# TRANSFORM (pure -- no I/O, so tests can hand these synthetic fixtures)
# ---------------------------------------------------------------------------

def _settled(history: pd.DataFrame) -> pd.DataFrame:
    return history[history["GW finished"] == True]  # noqa: E712


def _periods(rows: pd.DataFrame) -> list[dict]:
    """Same delta logic as fpl_minutes.py's _player_periods, tracking
    DEFCON and Minutes instead of Starts and Minutes. Kept standalone here
    (each script in this project is self-contained) rather than imported."""
    rows = rows.sort_values("Snapshot")
    periods = []
    prev = None
    for _, row in rows.iterrows():
        if prev is None:
            if row["GW"] == 1:
                periods.append({"gw_end": 1, "games": 1,
                                 "defcon": max(int(row["DEFCON"]), 0),
                                 "minutes": max(int(row["Minutes"]), 0),
                                 "team": row["Team"]})
            prev = row
            continue

        games = int(row["GW"]) - int(prev["GW"])
        if games <= 0:
            prev = row
            continue

        periods.append({
            "gw_end": int(row["GW"]), "games": games,
            "defcon": max(int(row["DEFCON"]) - int(prev["DEFCON"]), 0),
            "minutes": max(int(row["Minutes"]) - int(prev["Minutes"]), 0),
            "team": row["Team"],
        })
        prev = row

    return periods


def build_defcon_dataset(history: pd.DataFrame, fixtures_lookup: dict,
                          min_period_minutes: int = MIN_PERIOD_MINUTES) -> pd.DataFrame:
    """
    One row per player per clean single-fixture gameweek: the DEFCON-per-90
    rate earned in that gameweek, joined to that gameweek's opponent FDR
    and home/away. Excludes goalkeepers (DEFCON doesn't apply to them),
    double gameweeks (can't attribute to one opponent), multi-GW gaps
    (a missed snapshot spanning >1 gameweek, same reason), and periods
    below the minutes floor.
    """
    settled = _settled(history)
    rows = []
    for pid, player_rows in settled.groupby("ID"):
        pos = player_rows["Pos"].iloc[-1]
        if pos == "GKP":
            continue
        player_name = player_rows["Player"].iloc[-1]

        for period in _periods(player_rows):
            if period["games"] != 1:
                continue
            if period["minutes"] < min_period_minutes:
                continue
            fixture = fixtures_lookup.get((period["team"], period["gw_end"]))
            if fixture is None or fixture["double"]:
                continue

            per90 = period["defcon"] / (period["minutes"] / 90)
            rows.append({
                "ID": pid, "Player": player_name, "Team": period["team"],
                "GW": period["gw_end"], "Pos": pos,
                "DEFCON per90 (period)": round(per90, 3),
                "Opponent FDR": fixture["fdr"], "Home": fixture["home"],
                "Opponent": fixture["opponent"], "Minutes": period["minutes"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STATS (no scipy -- Fisher z gives a legitimate 95% CI on r without it)
# ---------------------------------------------------------------------------

def pearson_ci(x, y, confidence: float = 0.95) -> dict:
    x, y = pd.Series(x).astype(float), pd.Series(y).astype(float)
    n = len(x)
    if n < 5 or x.std() == 0 or y.std() == 0:
        return {"r": None, "lo": None, "hi": None, "n": n}

    r = float(x.corr(y))
    r_clamped = max(min(r, 0.999999), -0.999999)  # atanh blows up at +-1
    z = math.atanh(r_clamped)
    se = 1 / math.sqrt(n - 3)
    lo, hi = math.tanh(z - Z_95 * se), math.tanh(z + Z_95 * se)
    return {"r": round(r, 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": n}


def spearman_ci(x, y, confidence: float = 0.95) -> dict:
    """Spearman rho = Pearson r of the ranks. The Fisher-z CI is an
    approximation here (it's exact for Pearson, not Spearman), adequate
    for a directional read rather than a precise interval."""
    return pearson_ci(pd.Series(x).rank(), pd.Series(y).rank(), confidence)


def mean_ci(x, confidence: float = 0.95) -> dict:
    x = pd.Series(x).astype(float)
    n = len(x)
    if n < 2:
        return {"mean": round(float(x.mean()), 3) if n else None, "lo": None, "hi": None, "n": n}
    mean = float(x.mean())
    se = float(x.std(ddof=1)) / math.sqrt(n)
    return {"mean": round(mean, 3), "lo": round(mean - Z_95 * se, 3),
            "hi": round(mean + Z_95 * se, 3), "n": n}


def home_away_split(dataset: pd.DataFrame):
    home = dataset.loc[dataset["Home"], "DEFCON per90 (period)"]
    away = dataset.loc[~dataset["Home"], "DEFCON per90 (period)"]
    home_stats, away_stats = mean_ci(home), mean_ci(away)

    diff = None
    if home_stats["n"] >= 2 and away_stats["n"] >= 2:
        se_diff = math.sqrt(home.std(ddof=1) ** 2 / home_stats["n"]
                             + away.std(ddof=1) ** 2 / away_stats["n"])
        mean_diff = float(home.mean() - away.mean())
        diff = {"mean_diff": round(mean_diff, 3),
                "lo": round(mean_diff - Z_95 * se_diff, 3),
                "hi": round(mean_diff + Z_95 * se_diff, 3)}
    return home_stats, away_stats, diff


def classify_effect(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    return "strong"


def verdict_for_correlation(stats: dict, min_n: int = MIN_SAMPLE_SIZE) -> str:
    if stats["n"] < min_n or stats["r"] is None:
        return f"Insufficient data (n={stats['n']}, need >= {min_n})."

    ci_excludes_zero = stats["lo"] > 0 or stats["hi"] < 0
    label = classify_effect(stats["r"])
    if ci_excludes_zero and label != "negligible":
        return (f"Real effect ({label}, r={stats['r']}, "
                f"95% CI [{stats['lo']}, {stats['hi']}], n={stats['n']}).")
    return (f"Null result -- 95% CI [{stats['lo']}, {stats['hi']}] includes zero or the "
            f"effect is negligible (r={stats['r']}, n={stats['n']}). Recommend NOT applying "
            f"a fixture adjustment to DEFCON.")


# ---------------------------------------------------------------------------
# STORE
# ---------------------------------------------------------------------------

def update_report_store(row: dict, path: Path) -> pd.DataFrame:
    today = row["Snapshot"]
    new_row = pd.DataFrame([row])

    if path.exists():
        prior = pd.read_csv(path)
        before = len(prior)
        prior = prior[prior["Snapshot"] != today]
        if len(prior) < before:
            print(f"fpl_defcon: replacing an earlier report from {today}")
        combined = pd.concat([prior, new_row], ignore_index=True)
    else:
        combined = new_row
        print("fpl_defcon: first report, creating store")

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
    print("Fetching fixtures (whole season, for historical opponent lookup)...")
    fixtures_lookup = fetch_fixtures_lookup()

    dataset = build_defcon_dataset(history, fixtures_lookup)
    dataset.to_csv(DATASET, index=False)

    print(f"\n=== DEFCON vs opponent territory: hypothesis test ===")
    print(f"Usable observations: {len(dataset)} "
          f"(clean single-fixture gameweeks, >= {MIN_PERIOD_MINUTES} min played, DEF/MID/FWD only)")

    if dataset.empty:
        print("\nNo usable observations yet. This needs finished gameweeks with clean "
              "single fixtures behind them -- expected to read zero until then.")
        pearson = {"r": None, "lo": None, "hi": None, "n": 0}
        spearman = {"r": None, "lo": None, "hi": None, "n": 0}
        home_stats, away_stats, diff = {"mean": None, "lo": None, "hi": None, "n": 0}, \
                                        {"mean": None, "lo": None, "hi": None, "n": 0}, None
    else:
        pearson = pearson_ci(dataset["Opponent FDR"], dataset["DEFCON per90 (period)"])
        spearman = spearman_ci(dataset["Opponent FDR"], dataset["DEFCON per90 (period)"])
        home_stats, away_stats, diff = home_away_split(dataset)

        print(f"\nPearson r  (DEFCON/90 vs opponent FDR): {pearson['r']}  "
              f"95% CI [{pearson['lo']}, {pearson['hi']}]  n={pearson['n']}")
        print(f"Spearman rho (rank-based, robustness check): {spearman['r']}  "
              f"95% CI [{spearman['lo']}, {spearman['hi']}]  n={spearman['n']}")

        print(f"\nHome DEFCON/90:  mean={home_stats['mean']}  "
              f"95% CI [{home_stats['lo']}, {home_stats['hi']}]  n={home_stats['n']}")
        print(f"Away DEFCON/90:  mean={away_stats['mean']}  "
              f"95% CI [{away_stats['lo']}, {away_stats['hi']}]  n={away_stats['n']}")
        if diff:
            print(f"Home - Away diff: {diff['mean_diff']}  "
                  f"95% CI [{diff['lo']}, {diff['hi']}]")

    verdict = verdict_for_correlation(pearson)
    print(f"\nVerdict: {verdict}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    current_gw = int(history["GW"].max())
    update_report_store({
        "Snapshot": today, "GW": current_gw, "n": pearson["n"],
        "Pearson r": pearson["r"], "Pearson lo": pearson["lo"], "Pearson hi": pearson["hi"],
        "Spearman rho": spearman["r"], "Spearman lo": spearman["lo"], "Spearman hi": spearman["hi"],
        "Home mean": home_stats["mean"], "Home n": home_stats["n"],
        "Away mean": away_stats["mean"], "Away n": away_stats["n"],
        "Home-Away diff": diff["mean_diff"] if diff else None,
        "Verdict": verdict,
    }, REPORT)
    print(f"\nfpl_defcon: report -> {REPORT}, dataset -> {DATASET}")


if __name__ == "__main__":
    main()
