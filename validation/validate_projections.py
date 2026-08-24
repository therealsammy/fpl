#!/usr/bin/env python3
"""
Projection Validator
======================
The part that decides whether fpl_projections.py is worth trusting. FPL
already publishes its own next-gameweek estimate (`Exp pts next`) in every
snapshot -- that is the benchmark. Once a gameweek finishes, this scores
both my projections and FPL's own against what actually happened.

If my projections don't beat ep_next, this says so plainly and keeps
saying so. A model that loses to the free baseline is worse than no model,
because it gets trusted and acted on. This script does not get tuned to
win -- it reports honestly and lets you decide whether to keep it.

Run it after a gameweek finishes (any time after fpl_tracker.py has
recorded that GW as finished):

    python validate_projections.py

Reads (never writes to) fpl_history.csv and projections.csv. No network
calls. Writes validate_report.csv, append-only, one row per (gameweek,
position) -- but unlike every other store in this project, this one is
idempotent FOREVER per gameweek, not per day: a finished gameweek's actual
results don't change, so once GW7 is scored it is never rescored, even
across many later runs. Catches up on any backlog of newly-finished,
not-yet-validated gameweeks in one run.

THE COMPARISON, per validated gameweek, restricted to players who actually
started that gameweek (a clean single-gameweek Starts delta, same
convention as fpl_minutes.py and fpl_defcon.py):
- my projection for that GW (projections.csv, "Expected points")
- FPL's ep_next for that GW (the snapshot from the gameweek immediately
  before, "Exp pts next" -- that's what the API was predicting right
  before this gameweek kicked off)
- actual points scored (fpl_history.csv, "GW pts", from the finished
  snapshot for that GW)

Scored by Spearman rank correlation and RMSE, both overall and broken out
by position, with sample size reported at every level. The AGGREGATE
verdict (not the weekly numbers, which are always shown once computable)
is gated behind ~6 validated gameweeks per the brief -- a verdict from
four data points is not a verdict.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

HISTORY = Path("fpl_history.csv")
PROJECTIONS = Path("projections.csv")
REPORT = Path("validate_report.csv")

MIN_GW_SAMPLE = 10        # below this many starters, a single gameweek's numbers are noise
MIN_POSITION_SAMPLE = 5   # same idea, per position
MIN_VALIDATED_WEEKS = 6   # the brief's own floor before trusting the aggregate verdict

POSITIONS = ["GKP", "DEF", "MID", "FWD"]


# ---------------------------------------------------------------------------
# STATS (no scipy -- rank-transform + Pearson gives Spearman; RMSE is arithmetic)
# ---------------------------------------------------------------------------

def spearman_rank_corr(x: pd.Series, y: pd.Series):
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return round(float(x.rank().corr(y.rank())), 4)


def rmse(x: pd.Series, y: pd.Series):
    if len(x) == 0:
        return None
    return round(float(((x - y) ** 2).mean() ** 0.5), 4)


# ---------------------------------------------------------------------------
# DATA EXTRACTION (pure -- no I/O, so tests can hand these synthetic fixtures)
# ---------------------------------------------------------------------------

def actual_points_for_gw(history: pd.DataFrame, target_gw: int):
    """(points Series, position Series) indexed by ID, or (None, None) if
    this gameweek hasn't finished yet."""
    settled = history[(history["GW"] == target_gw) & (history["GW finished"] == True)]  # noqa: E712
    if settled.empty:
        return None, None
    latest_snap = settled["Snapshot"].max()
    row = settled[settled["Snapshot"] == latest_snap].set_index("ID")
    return row["GW pts"], row["Pos"]


def ep_next_baseline_for_gw(history: pd.DataFrame, target_gw: int):
    """
    FPL's own prediction for `target_gw`, as it stood right before that
    gameweek started. Only trusted if the source snapshot is the CLEAN
    immediate predecessor (GW == target_gw - 1) -- a stale, older
    prediction isn't a fair comparison, so a gap here means "not
    validatable yet", not "use what's available".
    """
    candidates = history[history["GW"] < target_gw]
    if candidates.empty:
        return None
    best_gw = candidates["GW"].max()
    if best_gw != target_gw - 1:
        return None
    subset = candidates[candidates["GW"] == best_gw]
    latest_snap = subset["Snapshot"].max()
    row = subset[subset["Snapshot"] == latest_snap].set_index("ID")
    return row["Exp pts next"]


def starters_for_gw(history: pd.DataFrame, target_gw: int):
    """IDs of players with a clean, unambiguous start in `target_gw` --
    a single-gameweek Starts delta of at least 1 between the two
    immediately consecutive settled snapshots. None if that clean
    comparison isn't available (e.g. a missed snapshot spans >1 gameweek)."""
    settled = history[history["GW finished"] == True]  # noqa: E712
    curr = settled[settled["GW"] == target_gw]
    prev_candidates = settled[settled["GW"] < target_gw]
    if curr.empty or prev_candidates.empty:
        return None

    best_prev_gw = prev_candidates["GW"].max()
    if best_prev_gw != target_gw - 1:
        return None

    curr_snap = curr[curr["Snapshot"] == curr["Snapshot"].max()].set_index("ID")
    prev_snap = prev_candidates[prev_candidates["GW"] == best_prev_gw]
    prev_snap = prev_snap[prev_snap["Snapshot"] == prev_snap["Snapshot"].max()].set_index("ID")

    common = curr_snap.index.intersection(prev_snap.index)
    delta_starts = curr_snap.loc[common, "Starts"] - prev_snap.loc[common, "Starts"]
    return set(delta_starts[delta_starts >= 1].index)


# ---------------------------------------------------------------------------
# PER-GAMEWEEK VALIDATION
# ---------------------------------------------------------------------------

def validate_gameweek(history: pd.DataFrame, projections: pd.DataFrame, target_gw: int):
    """
    Returns None if this gameweek can't be validated yet (missing a
    prerequisite that a LATER run might supply -- so it should be retried,
    not permanently recorded). Returns a result dict otherwise, which IS
    permanent once a gameweek has finished (its actual results won't change).
    """
    actual, pos = actual_points_for_gw(history, target_gw)
    if actual is None:
        return None

    my_rows = projections[projections["GW"] == target_gw]
    if my_rows.empty:
        return None
    my_proj = my_rows.set_index("ID")["Expected points"]

    baseline = ep_next_baseline_for_gw(history, target_gw)
    if baseline is None:
        return None

    started_ids = starters_for_gw(history, target_gw)
    if started_ids is None:
        return None

    common = sorted(started_ids & set(my_proj.index) & set(baseline.index) & set(actual.index))
    m, b, a, p = (s.reindex(common) for s in (my_proj, baseline, actual, pos))

    overall = {
        "n": len(common),
        "mine_spearman": spearman_rank_corr(m, a) if len(common) >= MIN_GW_SAMPLE else None,
        "mine_rmse": rmse(m, a) if len(common) >= MIN_GW_SAMPLE else None,
        "epnext_spearman": spearman_rank_corr(b, a) if len(common) >= MIN_GW_SAMPLE else None,
        "epnext_rmse": rmse(b, a) if len(common) >= MIN_GW_SAMPLE else None,
    }

    by_position = {}
    for position in POSITIONS:
        mask = p == position
        n = int(mask.sum())
        if n < MIN_POSITION_SAMPLE:
            by_position[position] = {"n": n, "mine_spearman": None, "mine_rmse": None,
                                      "epnext_spearman": None, "epnext_rmse": None}
        else:
            by_position[position] = {
                "n": n,
                "mine_spearman": spearman_rank_corr(m[mask], a[mask]),
                "mine_rmse": rmse(m[mask], a[mask]),
                "epnext_spearman": spearman_rank_corr(b[mask], a[mask]),
                "epnext_rmse": rmse(b[mask], a[mask]),
            }

    return {"GW": target_gw, "overall": overall, "by_position": by_position}


def flatten_result(result: dict, run_date: str) -> list:
    rows = [{"Snapshot": run_date, "GW": result["GW"], "Position": "All", **result["overall"]}]
    for position, stats in result["by_position"].items():
        rows.append({"Snapshot": run_date, "GW": result["GW"], "Position": position, **stats})
    return rows


# ---------------------------------------------------------------------------
# STORE (idempotent forever per (GW, Position) -- not per-day)
# ---------------------------------------------------------------------------

def update_store(new_rows: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists():
        prior = pd.read_csv(path)
        combined = pd.concat([prior, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["GW", "Position"], keep="first")
    else:
        combined = new_rows
    combined = combined.sort_values(["GW", "Position"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


# ---------------------------------------------------------------------------
# AGGREGATE VERDICT
# ---------------------------------------------------------------------------

def overall_verdict(report: pd.DataFrame, min_weeks: int = MIN_VALIDATED_WEEKS) -> str:
    all_rows = report[(report["Position"] == "All") & report["mine_spearman"].notna()]
    n_weeks = all_rows["GW"].nunique()
    if n_weeks < min_weeks:
        return f"Insufficient data ({n_weeks} validated gameweek(s), need >= {min_weeks})."

    mine_mean = all_rows["mine_spearman"].mean()
    epnext_mean = all_rows["epnext_spearman"].mean()
    weeks_mine_beats = int((all_rows["mine_spearman"] > all_rows["epnext_spearman"]).sum())

    if mine_mean > epnext_mean:
        return (f"Beats ep_next on average across {n_weeks} gameweeks "
                f"(mean Spearman {mine_mean:.3f} vs {epnext_mean:.3f}; "
                f"won {weeks_mine_beats}/{n_weeks} individual weeks).")
    return (f"LOSES to ep_next on average across {n_weeks} gameweeks "
            f"(mean Spearman {mine_mean:.3f} vs {epnext_mean:.3f}; "
            f"won {weeks_mine_beats}/{n_weeks} individual weeks). Per the brief: a model "
            f"that loses to the free baseline is worse than no model. Do not act on these "
            f"projections until this changes.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if not HISTORY.exists():
        print(f"{HISTORY} not found. Run fpl_tracker.py first.")
        sys.exit(1)
    if not PROJECTIONS.exists():
        print(f"{PROJECTIONS} not found -- fpl_projections.py hasn't produced anything yet "
              f"(its readiness gate is probably still closed). Nothing to validate.")
        return

    history = pd.read_csv(HISTORY)
    projections = pd.read_csv(PROJECTIONS)

    already_validated = set()
    if REPORT.exists():
        already_validated = set(pd.read_csv(REPORT)["GW"].unique())

    settled_gws = sorted(history.loc[history["GW finished"] == True, "GW"].unique())  # noqa: E712
    candidates = [gw for gw in settled_gws if gw not in already_validated]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_rows = []
    for gw in candidates:
        result = validate_gameweek(history, projections, gw)
        if result is None:
            print(f"GW{gw}: not validatable yet (missing a projection, a clean ep_next "
                  f"baseline, or a clean single-gameweek starts comparison). Will retry.")
            continue
        new_rows.extend(flatten_result(result, today))
        print(f"GW{gw}: validated (n={result['overall']['n']} starters).")

    if new_rows:
        combined = update_store(pd.DataFrame(new_rows), REPORT)
    elif REPORT.exists():
        combined = pd.read_csv(REPORT)
    else:
        combined = pd.DataFrame()

    if combined.empty:
        print("No validated gameweeks yet.")
        return

    print("\n=== Weekly scores (Mine vs ep_next) ===")
    print(combined[combined["Position"] == "All"][
        ["GW", "n", "mine_spearman", "mine_rmse", "epnext_spearman", "epnext_rmse"]
    ].to_string(index=False))

    print(f"\nVerdict: {overall_verdict(combined)}")


if __name__ == "__main__":
    main()
