"""
Elo Ratings
============
Standard Elo, computed from match results alone (SPEC.md Phase 4).

Rated PER LEAGUE, not across leagues. SPEC.md describes "cross-league
linking via European ties where fixtures exist" -- but collectors/
football_data.py only gathers domestic league fixtures (results.py from
football-data.co.uk carries no Champions League / Europa League data).
With no shared-opponent matches between, say, the Premier League and
Bundesliga, there is nothing real to link on. Rating them as one pool
anyway would produce a number that LOOKS like a cross-league comparison
but isn't backed by any actual match between those leagues' teams --
exactly the kind of confident-looking fabrication this project avoids
everywhere else. If a European-competition results source gets added
later, this is where the cross-league link would go.

Standard formula, no goal-margin scaling (a common refinement --
eloratings.net's multiplier for margin of victory, for example -- left
out to keep this the "standard Elo" SPEC.md actually asked for):

    expected_home = 1 / (1 + 10 ** ((rating_away - rating_home - home_advantage) / 400))
    rating_home' = rating_home + k * (actual_home - expected_home)

home_advantage is a ratings-point bonus applied only when computing the
expected score, not a permanent addition to the rating.
"""

from pathlib import Path

import pandas as pd

MATCHES_ROOT = Path("data/odds/football_data")

DEFAULT_K = 32
DEFAULT_HOME_ADVANTAGE = 100.0
STARTING_RATING = 1500.0


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_all_matches(leagues: list | None = None) -> pd.DataFrame:
    """Concatenates every collected football_data season file (or a
    filtered subset of leagues) into one chronological matches table."""
    if not MATCHES_ROOT.exists():
        return pd.DataFrame(columns=["date", "league", "season", "home_team", "away_team",
                                      "home_goals", "away_goals", "result"])

    league_dirs = ([MATCHES_ROOT / lg for lg in leagues] if leagues
                    else sorted(d for d in MATCHES_ROOT.iterdir() if d.is_dir()))

    # season must stay a string: codes like "0001" (2000/01) lose their
    # leading zero and become the wrong season entirely if read as a
    # number -- verified against a real collected file before fixing this.
    frames = [pd.read_csv(f, dtype={"season": str})
              for d in league_dirs if d.exists() for f in d.glob("*.csv")]
    if not frames:
        return pd.DataFrame(columns=["date", "league", "season", "home_team", "away_team",
                                      "home_goals", "away_goals", "result"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"home_team_raw": "home_team", "away_team_raw": "away_team"})
    combined = combined.dropna(subset=["result", "home_team", "away_team"])
    return combined.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# ELO (pure -- operates on an in-memory matches DataFrame)
# ---------------------------------------------------------------------------

def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo_history(matches: pd.DataFrame, k: float = DEFAULT_K,
                         home_advantage: float = DEFAULT_HOME_ADVANTAGE,
                         starting_rating: float = STARTING_RATING) -> pd.DataFrame:
    """
    One row per team per match: date, league, team, opponent, home
    (bool), rating_before, rating_after, result (from that team's own
    perspective: 'W'/'D'/'L'). Sorts by date within each league
    internally, so input order doesn't matter -- ratings are always
    computed strictly chronologically, never using future results.
    """
    if matches.empty:
        return pd.DataFrame(columns=["date", "league", "team", "opponent", "home",
                                      "rating_before", "rating_after", "result"])

    rows = []
    for league, league_matches in matches.groupby("league"):
        ratings = {}
        for _, m in league_matches.sort_values("date").iterrows():
            home, away = m["home_team"], m["away_team"]
            r_home = ratings.get(home, starting_rating)
            r_away = ratings.get(away, starting_rating)

            exp_home = _expected_score(r_home + home_advantage, r_away)
            exp_away = 1 - exp_home

            if m["result"] == "H":
                s_home, s_away, res_home, res_away = 1.0, 0.0, "W", "L"
            elif m["result"] == "A":
                s_home, s_away, res_home, res_away = 0.0, 1.0, "L", "W"
            else:
                s_home, s_away, res_home, res_away = 0.5, 0.5, "D", "D"

            new_r_home = r_home + k * (s_home - exp_home)
            new_r_away = r_away + k * (s_away - exp_away)

            rows.append({"date": m["date"], "league": league, "team": home, "opponent": away,
                        "home": True, "rating_before": r_home, "rating_after": new_r_home,
                        "result": res_home})
            rows.append({"date": m["date"], "league": league, "team": away, "opponent": home,
                        "home": False, "rating_before": r_away, "rating_after": new_r_away,
                        "result": res_away})

            ratings[home] = new_r_home
            ratings[away] = new_r_away

    return pd.DataFrame(rows)


def current_ratings(history: pd.DataFrame) -> pd.DataFrame:
    """Latest rating per (league, team), ranked highest first within
    each league."""
    if history.empty:
        return pd.DataFrame(columns=["league", "team", "rating", "date"])
    ordered = history.sort_values("date")
    latest = ordered.groupby(["league", "team"], as_index=False).tail(1)
    latest = latest[["league", "team", "rating_after", "date"]].rename(columns={"rating_after": "rating"})
    return latest.sort_values(["league", "rating"], ascending=[True, False]).reset_index(drop=True)


def rating_trajectory(history: pd.DataFrame, league: str, team: str) -> pd.DataFrame:
    """A single team's rating over time, for charting."""
    sub = history[(history["league"] == league) & (history["team"] == team)]
    return sub.sort_values("date")[["date", "rating_after", "opponent", "result"]]
