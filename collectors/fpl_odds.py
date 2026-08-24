#!/usr/bin/env python3
"""
FPL Odds Collector
====================
Bookmaker odds are point-in-time and cannot be backfilled -- a week without
a collector run is a week of data permanently lost. This pulls EPL 1X2 and
over/under 2.5 odds from The Odds API, stores the response untouched, then
derives expected goals and clean sheet probability from it.

Run it Friday afternoon, before Saturday deadlines:

    python fpl_odds.py

Config: ODDS_API_KEY must be set in the environment (see .env locally, or
the ODDS_API_KEY repo secret in CI). Validated for contents, not just
presence -- CI sets an unset secret to "", and int()/API calls on "" fail
in confusing ways rather than a clear message.

Writes, in order:
  odds_raw/YYYY-MM-DD.json   the UNMODIFIED API response. If the model
                             below changes later, re-derive from this
                             rather than re-collecting data that no longer
                             exists at that point in time.
  fpl_odds.csv               append-only, keyed on snapshot date + fixture
                             ID, idempotent on same-day re-run.
  fixture_projections.csv    append-only, keyed on snapshot date + GW +
                             team -- phase 6's validator needs past
                             projections to survive to compare against
                             actual results later, so this is a history,
                             not a disposable recompute.

This script does NOT rank players or feed transfer decisions. That's
phase 7, and only once phase 6 has validated this model against reality.

Math notes:
- Devig: proportional (implied prob / overround) is computed first because
  it's simple and obvious. Shin's method (Shin, 1992) is then used as the
  actual model input, because it corrects the well-documented
  favorite-longshot bias that proportional devigging doesn't. Both are
  printed side by side in the digest so you can see how much it matters.
- Expected goals: Dixon-Coles-adjusted Poisson, correlation parameter rho
  fixed at a literature-typical value (not fit per fixture -- 1X2 + O/U2.5
  gives 3 independent constraints for 2 unknowns (lambda_home, lambda_away),
  solved by damped Gauss-Newton least squares). No scipy dependency.
- Clean sheet probability = exp(-lambda_opponent), taken directly per the
  brief -- the Dixon-Coles low-score adjustment's effect on the marginal
  is negligible enough to ignore for this purpose.
"""

import json
import math
import os
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

SPORT = "soccer_epl"
REGIONS = "uk"
MARKETS = "h2h,totals"
ODDS_FORMAT = "decimal"
TOTALS_LINE = 2.5

RAW_DIR = Path("odds_raw")
ODDS_OUTPUT = Path("fpl_odds.csv")
PROJECTIONS_OUTPUT = Path("fixture_projections.csv")

DIXON_COLES_RHO = -0.1   # Dixon & Coles (1997)'s own fitted value for English league data
MAX_GOALS = 10           # Poisson tail beyond this is negligible for realistic lambdas

BASE = "https://fantasy.premierleague.com/api"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
})

# The Odds API's full team names -> FPL short codes. Promoted/relegated
# teams change every season -- if a fixture fails to map, THIS is what to
# update. Deliberately not fuzzy-matched: a quietly dropped fixture is
# worse than a crash, because it would go unnoticed.
TEAM_NAME_MAP = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "AFC Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton and Hove Albion": "BHA",
    "Brighton": "BHA",
    "Chelsea": "CHE",
    "Coventry City": "COV",
    "Coventry": "COV",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Hull City": "HUL",
    "Hull": "HUL",
    "Ipswich Town": "IPS",
    "Ipswich": "IPS",
    "Leeds United": "LEE",
    "Leeds": "LEE",
    "Liverpool": "LIV",
    "Manchester City": "MCI",
    "Man City": "MCI",
    "Manchester United": "MUN",
    "Man United": "MUN",
    "Man Utd": "MUN",
    "Newcastle United": "NEW",
    "Newcastle": "NEW",
    "Nottingham Forest": "NFO",
    "Nott'm Forest": "NFO",
    "Sunderland": "SUN",
    "Tottenham Hotspur": "TOT",
    "Tottenham": "TOT",
    "Spurs": "TOT",
}


class UnmappedTeamError(Exception):
    pass


def map_team(name: str) -> str:
    code = TEAM_NAME_MAP.get(name)
    if code is None:
        raise UnmappedTeamError(
            f"Unmapped team name from The Odds API: '{name}'. Add it to "
            f"TEAM_NAME_MAP in fpl_odds.py -- refusing to guess.")
    return code


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------

def _odds_api_key() -> str:
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        print("Set ODDS_API_KEY in the environment (.env locally, or the "
              "ODDS_API_KEY repo secret in CI).")
        sys.exit(1)
    return key


def fetch_odds(api_key: str):
    """Raises on any HTTP error -- there's nothing useful to derive without
    odds data, so this is a hard failure, not a gracefully-handled one."""
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = {"regions": REGIONS, "markets": MARKETS,
              "oddsFormat": ODDS_FORMAT, "apiKey": api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    remaining = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    print(f"The Odds API: {used} credits used this period, {remaining} remaining.")
    return r.json()


def get(path, params=None):
    r = SESSION.get(f"{BASE}/{path}", params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def fetch_fpl_fixture_gws() -> dict:
    """(home_code, away_code) -> gameweek, for matching Odds API fixtures
    to a specific FPL gameweek. Best-effort enrichment -- a fixture we
    can't match just gets no GW rather than failing the whole run."""
    boot = get("bootstrap-static/")
    fixtures = get("fixtures/")
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    lookup = {}
    for f in fixtures:
        gw = f.get("event")
        h, a = teams.get(f["team_h"]), teams.get(f["team_a"])
        if gw and h and a:
            lookup[(h, a)] = gw
    return lookup


# ---------------------------------------------------------------------------
# STORE RAW (before any parsing -- this is the irreplaceable point-in-time record)
# ---------------------------------------------------------------------------

def store_raw(data, snapshot_date: str) -> Path:
    RAW_DIR.mkdir(exist_ok=True)
    path = RAW_DIR / f"{snapshot_date}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# PARSE (pure -- no I/O, so tests can hand this synthetic API responses)
# ---------------------------------------------------------------------------

def parse_odds_response(data: list, snapshot_date: str) -> pd.DataFrame:
    """
    Flatten the API's nested event/bookmaker/market/outcome structure.
    Outcomes are relabeled to semantic names (Home win / Draw / Away win,
    Over 2.5 / Under 2.5) using each event's OWN home/away strings for the
    h2h match -- this needs no team mapping at all. Team mapping is only
    needed for the Home team / Away team columns themselves, and happens
    once per event, so a naming gap fails loudly exactly once per team,
    not once per bookmaker per market.
    """
    rows = []
    for event in data:
        home_raw, away_raw = event["home_team"], event["away_team"]
        home_code, away_code = map_team(home_raw), map_team(away_raw)
        commence = event.get("commence_time")

        for bk in event.get("bookmakers", []):
            for market in bk.get("markets", []):
                key = market.get("key")
                if key == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home_raw:
                            label = "Home win"
                        elif outcome["name"] == away_raw:
                            label = "Away win"
                        else:
                            label = "Draw"
                        rows.append({
                            "Snapshot": snapshot_date, "Fixture ID": event["id"],
                            "Commence time": commence, "Home team": home_code,
                            "Away team": away_code, "Bookmaker": bk.get("title"),
                            "Market": "h2h", "Outcome": label, "Price": outcome["price"],
                        })
                elif key == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("point") != TOTALS_LINE:
                            continue
                        label = f"{outcome['name']} {TOTALS_LINE}"
                        rows.append({
                            "Snapshot": snapshot_date, "Fixture ID": event["id"],
                            "Commence time": commence, "Home team": home_code,
                            "Away team": away_code, "Bookmaker": bk.get("title"),
                            "Market": "totals", "Outcome": label, "Price": outcome["price"],
                        })

    cols = ["Snapshot", "Fixture ID", "Commence time", "Home team", "Away team",
            "Bookmaker", "Market", "Outcome", "Price"]
    return pd.DataFrame(rows, columns=cols)


def median_prices(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Median price per fixture per outcome across bookmakers -- a
    consensus estimate, not the most generous single price."""
    if odds_df.empty:
        return pd.DataFrame(columns=["Fixture ID", "Commence time", "Home team",
                                      "Away team", "Market", "Outcome", "Median price"])
    grouped = (odds_df.groupby(["Fixture ID", "Commence time", "Home team",
                                 "Away team", "Market", "Outcome"])["Price"]
               .median().reset_index().rename(columns={"Price": "Median price"}))
    return grouped


# ---------------------------------------------------------------------------
# DEVIG
# ---------------------------------------------------------------------------

def proportional_devig(prices: list) -> list:
    implied = [1 / p for p in prices]
    total = sum(implied)
    return [p / total for p in implied]


def _shin_total(z: float, implied: list, overround: float) -> float:
    if z <= 1e-12:
        return sum(p / math.sqrt(overround) for p in implied)
    return sum((math.sqrt(z ** 2 + 4 * (1 - z) * p ** 2 / overround) - z) / (2 * (1 - z))
               for p in implied)


def shin_devig(prices: list, tol: float = 1e-10, max_iter: int = 100) -> tuple:
    """Shin (1992). Returns (probabilities summing to 1, estimated z).

    z=0 is the correct answer whenever the market already has ~zero
    overround (fair odds) -- checked explicitly, since the general bisection
    below uses a sign-product comparison that only distinguishes strictly
    positive from strictly negative, not "already zero".
    """
    implied = [1 / p for p in prices]
    overround = sum(implied)

    lo, hi = 0.0, 0.5
    f_lo = _shin_total(lo, implied, overround) - 1
    if abs(f_lo) < tol:
        z = 0.0
    else:
        f_hi = _shin_total(hi, implied, overround) - 1
        while f_hi > 0 and hi < 0.99:
            hi += 0.1
            f_hi = _shin_total(hi, implied, overround) - 1

        z = hi
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            f_mid = _shin_total(mid, implied, overround) - 1
            if abs(f_mid) < tol:
                z = mid
                break
            if f_lo * f_mid > 0:
                lo, f_lo = mid, f_mid
            else:
                hi, f_hi = mid, f_mid
            z = mid

    if z <= 1e-12:
        probs = [p / overround for p in implied]
    else:
        probs = [(math.sqrt(z ** 2 + 4 * (1 - z) * p ** 2 / overround) - z) / (2 * (1 - z))
                  for p in implied]
    total = sum(probs)
    return [p / total for p in probs], z


# ---------------------------------------------------------------------------
# DIXON-COLES POISSON FIT
# ---------------------------------------------------------------------------

def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _dc_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _score_matrix(lam_h: float, lam_a: float, rho: float, max_goals: int = MAX_GOALS):
    m = [[_poisson_pmf(h, lam_h) * _poisson_pmf(a, lam_a) * _dc_tau(h, a, lam_h, lam_a, rho)
          for a in range(max_goals + 1)] for h in range(max_goals + 1)]
    total = sum(sum(row) for row in m)
    return [[v / total for v in row] for row in m]


def implied_from_score_matrix(matrix: list) -> tuple:
    n = len(matrix)
    p_home = sum(matrix[h][a] for h in range(n) for a in range(n) if h > a)
    p_draw = sum(matrix[h][h] for h in range(n))
    p_away = sum(matrix[h][a] for h in range(n) for a in range(n) if a > h)
    p_over = sum(matrix[h][a] for h in range(n) for a in range(n) if h + a >= 3)
    return p_home, p_draw, p_away, p_over


def _residuals(lam_h, lam_a, target_home, target_draw, target_over, rho, max_goals):
    if lam_h <= 0.05 or lam_a <= 0.05 or lam_h > 6 or lam_a > 6:
        return [1e3, 1e3, 1e3]
    p_h, p_d, _, p_o = implied_from_score_matrix(_score_matrix(lam_h, lam_a, rho, max_goals))
    return [p_h - target_home, p_d - target_draw, p_o - target_over]


def fit_expected_goals(target_home: float, target_draw: float, target_over: float,
                        rho: float = DIXON_COLES_RHO, init: tuple = (1.3, 1.1),
                        max_iter: int = 60, tol: float = 1e-9,
                        max_goals: int = MAX_GOALS) -> dict:
    """
    Damped Gauss-Newton least squares for (lambda_home, lambda_away) against
    3 market-implied constraints (P(home), P(draw), P(over 2.5)) -- more
    constraints than the 2 unknowns, per the brief. No scipy: 2 parameters
    is small enough for a hand-rolled normal-equations solve per iteration.
    """
    lam_h, lam_a = init
    eps = 1e-4
    for _ in range(max_iter):
        r0 = _residuals(lam_h, lam_a, target_home, target_draw, target_over, rho, max_goals)
        r_h = _residuals(lam_h + eps, lam_a, target_home, target_draw, target_over, rho, max_goals)
        r_a = _residuals(lam_h, lam_a + eps, target_home, target_draw, target_over, rho, max_goals)
        J = [[(r_h[i] - r0[i]) / eps, (r_a[i] - r0[i]) / eps] for i in range(3)]

        JTJ = [[sum(J[k][i] * J[k][j] for k in range(3)) for j in range(2)] for i in range(2)]
        JTr = [sum(J[k][i] * r0[k] for k in range(3)) for i in range(2)]
        damping = 1e-3
        JTJ[0][0] += damping
        JTJ[1][1] += damping
        det = JTJ[0][0] * JTJ[1][1] - JTJ[0][1] * JTJ[1][0]
        if abs(det) < 1e-12:
            break

        delta_h = -(JTr[0] * JTJ[1][1] - JTr[1] * JTJ[0][1]) / det
        delta_a = -(JTJ[0][0] * JTr[1] - JTJ[1][0] * JTr[0]) / det
        lam_h = max(0.05, min(6.0, lam_h + delta_h))
        lam_a = max(0.05, min(6.0, lam_a + delta_a))
        if abs(delta_h) < tol and abs(delta_a) < tol:
            break

    resid = _residuals(lam_h, lam_a, target_home, target_draw, target_over, rho, max_goals)
    residual_norm = math.sqrt(sum(v ** 2 for v in resid))
    return {"lam_home": round(lam_h, 4), "lam_away": round(lam_a, 4),
            "residual": round(residual_norm, 6)}


# ---------------------------------------------------------------------------
# BUILD fixture_projections.csv (pure, given median prices + fixture->GW lookup)
# ---------------------------------------------------------------------------

def build_fixture_projections(median_df: pd.DataFrame, fixture_gws: dict,
                               snapshot_date: str, rho: float = DIXON_COLES_RHO) -> tuple:
    """Returns (projections_df, digest_rows) -- digest_rows carries the
    proportional-vs-Shin comparison per fixture for the printed report."""
    rows, digest = [], []

    for (fid, commence, home, away), grp in median_df.groupby(
            ["Fixture ID", "Commence time", "Home team", "Away team"]):
        h2h = grp[grp["Market"] == "h2h"].set_index("Outcome")["Median price"]
        totals = grp[grp["Market"] == "totals"].set_index("Outcome")["Median price"]

        needed_h2h = {"Home win", "Draw", "Away win"}
        needed_totals = {f"Over {TOTALS_LINE}", f"Under {TOTALS_LINE}"}
        if not needed_h2h.issubset(h2h.index) or not needed_totals.issubset(totals.index):
            continue  # incomplete market for this fixture -- skip, don't guess

        h2h_prices = [h2h["Home win"], h2h["Draw"], h2h["Away win"]]
        totals_prices = [totals[f"Over {TOTALS_LINE}"], totals[f"Under {TOTALS_LINE}"]]

        prop_h2h = proportional_devig(h2h_prices)
        shin_h2h, z = shin_devig(h2h_prices)
        shin_totals, _ = shin_devig(totals_prices)

        fit = fit_expected_goals(shin_h2h[0], shin_h2h[1], shin_totals[0], rho=rho)
        lam_h, lam_a = fit["lam_home"], fit["lam_away"]
        p_h, p_d, p_a, _ = implied_from_score_matrix(_score_matrix(lam_h, lam_a, rho))

        gw = fixture_gws.get((home, away))

        digest.append({
            "Fixture": f"{home} vs {away}", "Commence": commence,
            "P(Home) proportional": round(prop_h2h[0], 3), "P(Home) shin": round(shin_h2h[0], 3),
            "P(Draw) proportional": round(prop_h2h[1], 3), "P(Draw) shin": round(shin_h2h[1], 3),
            "P(Away) proportional": round(prop_h2h[2], 3), "P(Away) shin": round(shin_h2h[2], 3),
            "Shin z": round(z, 4), "lam_home": lam_h, "lam_away": lam_a,
            "Fit residual": fit["residual"],
        })

        rows.append({
            "Snapshot": snapshot_date, "GW": gw, "Fixture ID": fid, "Commence time": commence,
            "Team": home, "Opponent": away, "Home": True,
            "Expected goals for": lam_h, "Expected goals against": lam_a,
            "Clean sheet probability": round(math.exp(-lam_a), 4),
            "Win probability": round(p_h, 4), "Draw probability": round(p_d, 4),
            "Loss probability": round(p_a, 4), "Fit residual": fit["residual"],
        })
        rows.append({
            "Snapshot": snapshot_date, "GW": gw, "Fixture ID": fid, "Commence time": commence,
            "Team": away, "Opponent": home, "Home": False,
            "Expected goals for": lam_a, "Expected goals against": lam_h,
            "Clean sheet probability": round(math.exp(-lam_h), 4),
            "Win probability": round(p_a, 4), "Draw probability": round(p_d, 4),
            "Loss probability": round(p_h, 4), "Fit residual": fit["residual"],
        })

    return pd.DataFrame(rows), digest


# ---------------------------------------------------------------------------
# STORE (append-only, idempotent same-day)
# ---------------------------------------------------------------------------

def _update_store(new_rows: pd.DataFrame, path: Path, key_col: str = "Snapshot") -> pd.DataFrame:
    if new_rows.empty:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        today = new_rows[key_col].iloc[0]

    if path.exists():
        prior = pd.read_csv(path)
        before = prior[key_col].nunique()
        prior = prior[prior[key_col] != today]
        if prior[key_col].nunique() < before:
            print(f"{path.name}: replacing an earlier run from {today}")
        combined = pd.concat([prior, new_rows], ignore_index=True)
    else:
        combined = new_rows
        print(f"{path.name}: first snapshot, creating store")

    combined.to_csv(path, index=False)
    return combined


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    api_key = _odds_api_key()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("Fetching odds from The Odds API...")
    data = fetch_odds(api_key)
    raw_path = store_raw(data, today)
    print(f"Raw response stored -> {raw_path}")

    try:
        odds_df = parse_odds_response(data, today)
    except UnmappedTeamError as e:
        print(f"\n{e}")
        sys.exit(1)

    if odds_df.empty:
        print("No fixtures returned (off-season, or nothing in the odds window). Nothing to store.")
        return

    _update_store(odds_df, ODDS_OUTPUT, key_col="Snapshot")
    print(f"fpl_odds: {len(odds_df)} rows this run -> {ODDS_OUTPUT}")

    print("Fetching FPL fixtures for gameweek matching...")
    fixture_gws = fetch_fpl_fixture_gws()

    med = median_prices(odds_df)
    projections_df, digest = build_fixture_projections(med, fixture_gws, today)

    print(f"\n=== Devig comparison (proportional vs Shin) ===")
    for d in digest:
        print(f"  {d['Fixture']} ({d['Commence']})")
        print(f"    Home: {d['P(Home) proportional']} -> {d['P(Home) shin']}   "
              f"Draw: {d['P(Draw) proportional']} -> {d['P(Draw) shin']}   "
              f"Away: {d['P(Away) proportional']} -> {d['P(Away) shin']}   (z={d['Shin z']})")
        print(f"    Fitted: lam_home={d['lam_home']} lam_away={d['lam_away']} "
              f"(fit residual {d['Fit residual']})")

    if projections_df.empty:
        print("\nNo fixtures had complete h2h + totals markets across bookmakers -- "
              "nothing to project this run.")
        return

    _update_store(projections_df, PROJECTIONS_OUTPUT, key_col="Snapshot")
    print(f"\nfixture_projections: {len(projections_df)} team-rows this run -> {PROJECTIONS_OUTPUT}")

    unmatched = projections_df[projections_df["GW"].isna()]["Team"].nunique()
    if unmatched:
        print(f"Note: {unmatched} team(s) could not be matched to an FPL gameweek "
              f"(fixture list may not have updated yet).")


if __name__ == "__main__":
    main()
