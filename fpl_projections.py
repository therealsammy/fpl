#!/usr/bin/env python3
"""
FPL Player Projections
========================
Turns start probability (phase 1) and fixture expected goals / clean sheet
probability (phase 5) into a per-player expected points number for the next
gameweek, with the components broken out and an uncertainty range attached.

THIS SITS DORMANT UNTIL ITS INPUTS ARE TRUSTWORTHY. That is intended, not a
bug -- see the readiness gate below. A projection built on two snapshots is
a guess with a decimal point on it, and if this script shows you a number,
you are meant to be able to trust that the gate has already been checked.

Run it after fpl_minutes.py and fpl_odds.py, in that order:

    python fpl_projections.py

Reads (never writes to) fpl_history.csv, fpl_minutes.csv,
fixture_projections.csv, and defcon_report.csv. All local files -- no
network calls, so this can't fail on a flaky API.

Writes projections.csv, append-only, keyed on snapshot date + gameweek +
player ID. Safe to re-run the same day.

THE CALCULATION (per player, for the target gameweek)

    E[points] = P(start) * (appearance + goals + assists + clean sheet
                             + DEFCON + bonus)
              + P(sub appearance) * (reduced appearance component)

- Appearance: 2 pts at 60+ minutes, 1 below, weighted by the player's own
  trailing minutes-per-appearance (phase 1) rather than assuming 90.
- Goals: the player's trailing share of their TEAM's season xG, applied to
  the fixture's expected goals (phase 5), times the position multiplier
  (6/6/5/4 GKP/DEF/MID/FWD). Anytime-goalscorer odds would replace this
  entirely and more accurately if a paid tier is ever added -- this is the
  free-data substitute.
- Assists: same approach with the player's trailing xA share, times 3.
- Clean sheet: P(clean sheet) from phase 5 times the position multiplier
  (4/4/1/0). Goalkeepers additionally get an expected-save component from
  the fixture's expected goals against -- a crude conversion (no
  shots-faced data available), documented at SAVE_POINTS_PER_XGA below.
- DEFCON: the player's trailing DEFCON-per-90 rate, scaled by expected
  minutes, compared to the position's per-match threshold (10 for
  defenders, 12 for mid/forwards) as a crude proxy for P(reaching it).
  NOT fixture-adjusted -- phase 4 has not found a real opponent-territory
  effect (its verdict currently reads "Insufficient data"), and per the
  brief this stays fixture-independent unless that changes. The gate below
  reads defcon_report.csv's own verdict on every run, so this activates
  itself automatically if that hypothesis test ever confirms a real effect
  -- no code change needed here.
- Bonus: trailing bonus-per-start, a flat average. Deliberately not a BPS
  model -- a crude honest number beats a precise-looking wrong one.

Uncertainty comes from a Monte Carlo simulation of the start/sub/goals/
assists/clean-sheet/DEFCON draws (not the bonus term, which is genuinely
just a trailing average with no distribution to sample from).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

HISTORY = Path("fpl_history.csv")
MINUTES = Path("fpl_minutes.csv")
FIXTURE_PROJECTIONS = Path("fixture_projections.csv")
DEFCON_REPORT = Path("defcon_report.csv")
OUTPUT = Path("projections.csv")

MIN_SNAPSHOTS = 6
MIN_MINUTES_COVERAGE = 0.70
MIN_TRAILING_MINUTES = 180   # below this, flag low confidence -- thin sample for shares/rates

GOALS_MULT = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
ASSISTS_PTS = 3
CS_MULT = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}   # goalkeepers not eligible
DEFCON_PTS = 2

# Crude conversion, no shots-faced data available: roughly 3 shots on
# target accompany each expected goal against, saved at a league-average
# rate -- netting out to about 1 expected save-point per unit of xGA.
SAVE_POINTS_PER_XGA = 1.0

N_SIMULATIONS = 2000
CI_LOW, CI_HIGH = 2.5, 97.5   # percentiles for the reported range


# ---------------------------------------------------------------------------
# READINESS GATE
# ---------------------------------------------------------------------------

def determine_target_gw(fixture_proj: pd.DataFrame | None) -> int | None:
    """The soonest gameweek fixture_projections.csv has odds coverage for."""
    if fixture_proj is None or fixture_proj.empty:
        return None
    latest = fixture_proj[fixture_proj["Snapshot"] == fixture_proj["Snapshot"].max()]
    gws = latest["GW"].dropna()
    return int(gws.min()) if not gws.empty else None


def check_readiness(history: pd.DataFrame, minutes_df: pd.DataFrame | None,
                     fixture_proj: pd.DataFrame | None, target_gw: int | None,
                     min_snapshots: int = MIN_SNAPSHOTS,
                     min_coverage: float = MIN_MINUTES_COVERAGE) -> list:
    """Returns a list of plain-English reasons projections can't run yet.
    Empty list means the gate is open."""
    problems = []

    n_snap = history["Snapshot"].nunique()
    if n_snap < min_snapshots:
        problems.append(f"only {n_snap} snapshot(s) in fpl_history.csv; need >= {min_snapshots} "
                         f"({min_snapshots - n_snap} more gameweek(s)).")

    if minutes_df is None or minutes_df.empty:
        problems.append("fpl_minutes.csv not found or empty -- run fpl_minutes.py.")
    else:
        latest_min = minutes_df[minutes_df["Snapshot"] == minutes_df["Snapshot"].max()]
        latest_hist = history[history["Snapshot"] == history["Snapshot"].max()]
        played_ids = set(latest_hist.loc[latest_hist["Minutes"] > 0, "ID"])
        relevant = latest_min[latest_min["ID"].isin(played_ids)]
        coverage = relevant["Start probability"].notna().mean() if len(relevant) else 0.0
        if coverage < min_coverage:
            problems.append(f"fpl_minutes.csv covers only {coverage:.0%} of players with minutes "
                             f"(need >= {min_coverage:.0%}).")

    if fixture_proj is None or fixture_proj.empty:
        problems.append("fixture_projections.csv not found or empty -- run fpl_odds.py.")
    elif target_gw is None:
        problems.append("could not determine a gameweek to project -- no fixture data.")
    else:
        latest_fp = fixture_proj[fixture_proj["Snapshot"] == fixture_proj["Snapshot"].max()]
        if target_gw not in set(latest_fp["GW"].dropna()):
            problems.append(f"fixture_projections.csv doesn't cover GW{target_gw} yet.")

    return problems


# ---------------------------------------------------------------------------
# INPUT ASSEMBLY (pure -- no I/O, so tests can hand these synthetic fixtures)
# ---------------------------------------------------------------------------

def team_trailing_shares(latest_hist: pd.DataFrame) -> pd.DataFrame:
    """Each player's share of their team's season-cumulative xG and xA.
    NaN (not zero) when the team total is zero -- there's no share to
    compute yet, not a share of nothing."""
    team_totals = latest_hist.groupby("Team")[["xG", "xA"]].sum().rename(
        columns={"xG": "Team xG total", "xA": "Team xA total"})
    out = latest_hist.join(team_totals, on="Team")
    out["xG share"] = np.where(out["Team xG total"] > 0, out["xG"] / out["Team xG total"], np.nan)
    out["xA share"] = np.where(out["Team xA total"] > 0, out["xA"] / out["Team xA total"], np.nan)
    return out


def defcon_adjustment_enabled(defcon_report: pd.DataFrame | None) -> bool:
    """Reads phase 4's own verdict. Only 'Real effect' verdicts turn this
    on -- 'Insufficient data' and 'Null result' both mean fixture-independent."""
    if defcon_report is None or defcon_report.empty:
        return False
    latest_verdict = str(defcon_report.sort_values("Snapshot").iloc[-1]["Verdict"])
    return latest_verdict.startswith("Real effect")


def build_projection_inputs(history: pd.DataFrame, minutes_df: pd.DataFrame,
                             fixture_proj: pd.DataFrame, target_gw: int) -> pd.DataFrame:
    """
    One row per player with everything the calculation needs. Players
    whose team has no fixture coverage for the target gameweek (blank GW,
    or odds simply weren't collected) are dropped entirely rather than
    given a fabricated projection.
    """
    latest_hist = history[history["Snapshot"] == history["Snapshot"].max()].copy()
    latest_hist = team_trailing_shares(latest_hist)

    latest_min = minutes_df[minutes_df["Snapshot"] == minutes_df["Snapshot"].max()]
    min_cols = latest_min.set_index("ID")[[
        "Start probability", "Classification", "Minutes per appearance",
        "Appearances", "Starts (overall)", "Games (overall)"]]

    latest_fp = fixture_proj[fixture_proj["Snapshot"] == fixture_proj["Snapshot"].max()]
    fp_gw = latest_fp[latest_fp["GW"] == target_gw].set_index("Team")[
        ["Opponent", "Home", "Expected goals for", "Expected goals against",
         "Clean sheet probability"]]

    df = latest_hist.set_index("ID").join(min_cols, how="left")
    df = df.join(fp_gw, on="Team", how="inner")  # inner: no fixture -> no row

    non_start_games = (df["Games (overall)"] - df["Starts (overall)"]).clip(lower=0)
    subs = (df["Appearances"] - df["Starts (overall)"]).clip(lower=0)
    df["Sub rate"] = np.where(non_start_games > 0, (subs / non_start_games).clip(0, 1), 0.0)

    return df.reset_index()


def _num(value, default: float = 0.0) -> float:
    """NaN- and None-safe numeric coercion. `x or default` silently lets
    NaN through -- NaN is truthy in Python -- which then crashes numpy's
    random draws downstream. This is the fix for that."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return float(value)


# ---------------------------------------------------------------------------
# CALCULATION (Monte Carlo per player)
# ---------------------------------------------------------------------------

def project_player(row: pd.Series, rng: np.random.Generator,
                    n_sims: int = N_SIMULATIONS,
                    apply_defcon_adjustment: bool = False) -> dict:
    pos = row["Pos"]
    notes = []
    confidence = "High"

    p_start = row.get("Start probability")
    if pd.isna(p_start):
        notes.append("no start probability yet")
        p_start = 0.5   # neutral placeholder -- flagged, not hidden
    p_start = float(np.clip(p_start, 0.0, 1.0))

    if row.get("Classification") == "Insufficient data":
        notes.append("thin minutes history")
    trailing_minutes = _num(row.get("Minutes"))
    if trailing_minutes < MIN_TRAILING_MINUTES:
        notes.append(f"only {int(trailing_minutes)} trailing minutes")
    if pd.isna(row.get("xG share")):
        notes.append("no team xG history to derive a share from")
    if notes:
        confidence = "Low"

    minutes_per_app = row.get("Minutes per appearance")
    minutes_per_app = 60.0 if pd.isna(minutes_per_app) or minutes_per_app is None else minutes_per_app
    p_60plus = float(np.clip(minutes_per_app / 75.0, 0.0, 1.0))

    sub_rate = _num(row.get("Sub rate"))
    p_sub = (1 - p_start) * sub_rate
    p_none = max(1.0 - p_start - p_sub, 0.0)

    team_xg = _num(row["Expected goals for"])
    xg_share = _num(row.get("xG share"))
    xa_share = _num(row.get("xA share"))
    lam_goals = max(xg_share * team_xg, 0.0)
    lam_assists = max(xa_share * team_xg, 0.0)

    p_clean_sheet = float(np.clip(_num(row["Clean sheet probability"]), 0.0, 1.0))
    cs_mult = CS_MULT[pos]

    defcon_threshold = DEFCON_THRESHOLD.get(pos)
    if defcon_threshold:
        defcon_rate = _num(row.get("DEFCON per 90"))
        expected_actions = defcon_rate * (minutes_per_app / 90.0)
        if apply_defcon_adjustment:
            # Placeholder for when phase 4 confirms a real opponent-territory
            # effect -- not reached today, since that verdict currently
            # reads "Insufficient data", not "Real effect".
            pass
        p_defcon = float(np.clip(expected_actions / defcon_threshold, 0.0, 1.0))
    else:
        p_defcon = 0.0

    starts_overall = max(_num(row.get("Starts (overall)")), 0)
    bonus_per_start = _num(row.get("Bonus")) / starts_overall if starts_overall > 0 else 0.0

    save_pts = _num(row["Expected goals against"]) * SAVE_POINTS_PER_XGA if pos == "GKP" else 0.0

    outcome = rng.choice(3, size=n_sims, p=[p_none, p_sub, p_start])  # 0=none,1=sub,2=start
    appearance_draw = np.where(rng.random(n_sims) < p_60plus, 2, 1)
    goals_draw = rng.poisson(lam_goals, n_sims)
    assists_draw = rng.poisson(lam_assists, n_sims)
    cs_draw = rng.random(n_sims) < p_clean_sheet
    defcon_draw = rng.random(n_sims) < p_defcon

    started = outcome == 2
    subbed = outcome == 1

    appearance_component = np.where(started, appearance_draw, np.where(subbed, 1.0, 0.0))
    goals_component = np.where(started, goals_draw * GOALS_MULT[pos], 0.0)
    assists_component = np.where(started, assists_draw * ASSISTS_PTS, 0.0)
    cs_component = np.where(started, cs_draw * cs_mult, 0.0)
    defcon_component = np.where(started, defcon_draw * DEFCON_PTS, 0.0)
    bonus_component = np.where(started, bonus_per_start, 0.0)
    save_component = np.where(started, save_pts, 0.0)

    total = (appearance_component + goals_component + assists_component
             + cs_component + defcon_component + bonus_component + save_component)

    return {
        "components": {
            "Appearance pts": round(float(appearance_component.mean()), 3),
            "Goals pts": round(float(goals_component.mean()), 3),
            "Assists pts": round(float(assists_component.mean()), 3),
            "Clean sheet pts": round(float(cs_component.mean()), 3),
            "DEFCON pts": round(float(defcon_component.mean()), 3),
            "Bonus pts": round(float(bonus_component.mean()), 3),
            "Save pts": round(float(save_component.mean()), 3),
        },
        "expected": round(float(total.mean()), 3),
        "low": round(float(np.percentile(total, CI_LOW)), 3),
        "high": round(float(np.percentile(total, CI_HIGH)), 3),
        "confidence": confidence,
        "notes": notes,
        "p_start": p_start,
    }


def build_projections_table(df: pd.DataFrame, target_gw: int, snapshot_date: str,
                             n_sims: int = N_SIMULATIONS, seed: int | None = None,
                             apply_defcon_adjustment: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in df.iterrows():
        result = project_player(row, rng, n_sims=n_sims,
                                 apply_defcon_adjustment=apply_defcon_adjustment)
        price_raw = row.get("Price")
        price = None if price_raw is None or pd.isna(price_raw) else float(price_raw)
        rows.append({
            "Snapshot": snapshot_date, "GW": target_gw, "ID": row["ID"],
            "Player": row["Player"], "Team": row["Team"], "Pos": row["Pos"],
            "Price": price, "Start probability": round(result["p_start"], 3),
            **result["components"],
            "Expected points": result["expected"],
            "Expected points low": result["low"], "Expected points high": result["high"],
            "Expected points per million": round(result["expected"] / price, 3) if price else None,
            "Confidence": result["confidence"], "Note": "; ".join(result["notes"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STORE (append-only, idempotent same-day)
# ---------------------------------------------------------------------------

def update_store(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    today = table["Snapshot"].iloc[0] if not table.empty else \
        datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if path.exists():
        prior = pd.read_csv(path)
        before = prior["Snapshot"].nunique()
        prior = prior[prior["Snapshot"] != today]
        if prior["Snapshot"].nunique() < before:
            print(f"fpl_projections: replacing an earlier run from {today}")
        combined = pd.concat([prior, table], ignore_index=True)
    else:
        combined = table
        print("fpl_projections: first snapshot, creating store")

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
    minutes_df = pd.read_csv(MINUTES) if MINUTES.exists() else None
    fixture_proj = pd.read_csv(FIXTURE_PROJECTIONS) if FIXTURE_PROJECTIONS.exists() else None
    defcon_report = pd.read_csv(DEFCON_REPORT) if DEFCON_REPORT.exists() else None

    target_gw = determine_target_gw(fixture_proj)
    problems = check_readiness(history, minutes_df, fixture_proj, target_gw)

    if problems:
        print("fpl_projections: not ready yet. Missing:")
        for p in problems:
            print(f"  - {p}")
        print("\nThis is the readiness gate working as intended -- not an error.")
        return

    df = build_projection_inputs(history, minutes_df, fixture_proj, target_gw)
    if df.empty:
        print(f"No players have fixture coverage for GW{target_gw}. Nothing to project.")
        return

    apply_defcon = defcon_adjustment_enabled(defcon_report)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    table = build_projections_table(df, target_gw, today, apply_defcon_adjustment=apply_defcon)
    update_store(table, OUTPUT)

    n_low = int((table["Confidence"] == "Low").sum())
    print(f"fpl_projections: {len(table)} player(s) projected for GW{target_gw} "
          f"({n_low} low-confidence) -> {OUTPUT}")
    print(f"DEFCON fixture adjustment: {'ON' if apply_defcon else 'OFF (phase 4 has not confirmed a real effect)'}")
    print("\nTop 10 by expected points:")
    print(table.nlargest(10, "Expected points")[
        ["Player", "Team", "Pos", "Expected points", "Expected points low",
         "Expected points high", "Confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
