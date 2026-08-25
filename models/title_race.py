"""
Title Race Simulation
========================
Monte Carlo season simulation from Elo ratings (SPEC.md Phase 4). For the
current season: win probability per team from today. For a past season:
the probability path at every matchday, using Elo ratings and standings
as they genuinely stood on each date -- "what would this have said at
the time," not hindsight dressed up as foresight.

Elo's own formula only gives an expected SCORE (win=1/draw=0.5/loss=0,
blended into one number) -- there is no canonical way to split that into
three separate outcome probabilities. match_outcome_probabilities() uses
a simple, always-valid model instead of a fixed draw rate: draw
probability peaks at `draw_rate` for an even match and shrinks linearly
to zero as the rating gap widens. A fixed draw rate combined with a
lopsided expected score can go negative at extreme gaps; this can't, by
construction, and it matches the real-football pattern that heavy
favorites draw less often too.

Checkpoints in title_race_path() are the season's own distinct match
dates, not inferred "matchday" numbers -- football-data.co.uk doesn't
label rounds, so this uses what the data actually contains.
"""

import numpy as np
import pandas as pd

DEFAULT_HOME_ADVANTAGE = 100.0
DEFAULT_DRAW_RATE = 0.25
FALLBACK_RATING = 1500.0


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def match_outcome_probabilities(rating_home: float, rating_away: float,
                                 home_advantage: float = DEFAULT_HOME_ADVANTAGE,
                                 draw_rate: float = DEFAULT_DRAW_RATE) -> tuple:
    """(p_home_win, p_draw, p_away_win), always non-negative and summing to 1."""
    exp_home = _expected_score(rating_home + home_advantage, rating_away)
    p_draw = draw_rate * (1 - 2 * abs(exp_home - 0.5))
    p_home = exp_home - 0.5 * p_draw
    p_away = 1 - p_draw - p_home
    return p_home, p_draw, p_away


def simulate_title_race(current_points: dict, fixtures_remaining: list, ratings: dict,
                         n_sims: int = 2000, home_advantage: float = DEFAULT_HOME_ADVANTAGE,
                         draw_rate: float = DEFAULT_DRAW_RATE, seed=None) -> dict:
    """
    Monte Carlo simulation of the rest of a season from `current_points`
    forward. Ratings are frozen at their supplied value for the whole
    simulation (not updated match-by-match within one simulated season) --
    a reasonable simplification given how little a handful of remaining
    fixtures moves Elo, and it keeps each simulated season cheap enough
    to run thousands of times.

    Vectorized across all n_sims at once (a per-simulation, per-fixture
    Python loop calling rng.choice() individually was measured at 2+
    minutes for one real Premier League season's worth of checkpoints --
    millions of individual calls). Every fixture's outcome for every
    simulation is drawn in a single batched array operation instead.

    A tied final points total splits championship credit fractionally
    among the tied teams (1/n_tied) rather than picking one arbitrarily
    or modeling a playoff that doesn't exist in the data.

    Returns {team: win_probability} for every team in current_points,
    summing to 1.0 (a team with zero chance still appears at 0.0, not
    omitted -- so callers can always show the full table).
    """
    teams = list(current_points.keys())
    team_idx = {t: i for i, t in enumerate(teams)}
    base_points = np.array([current_points[t] for t in teams], dtype=float)

    if not fixtures_remaining:
        # Nothing left to simulate -- the table as it stands decides it.
        top = base_points.max()
        winners = (base_points == top).astype(float)
        credit = winners / winners.sum()
        return {t: round(float(c), 4) for t, c in zip(teams, credit)}

    rng = np.random.default_rng(seed)
    n_fixtures = len(fixtures_remaining)

    p_home = np.empty(n_fixtures)
    p_draw = np.empty(n_fixtures)
    home_idx = np.empty(n_fixtures, dtype=int)
    away_idx = np.empty(n_fixtures, dtype=int)
    for i, (home, away) in enumerate(fixtures_remaining):
        r_home = ratings.get(home, FALLBACK_RATING)
        r_away = ratings.get(away, FALLBACK_RATING)
        p_h, p_d, _p_a = match_outcome_probabilities(r_home, r_away, home_advantage, draw_rate)
        p_home[i], p_draw[i] = p_h, p_d
        home_idx[i] = team_idx[home]
        away_idx[i] = team_idx[away]

    # One draw per (simulation, fixture) cell, all at once.
    u = rng.random((n_sims, n_fixtures))
    home_win = u < p_home[None, :]
    draw = (~home_win) & (u < (p_home + p_draw)[None, :])
    home_pts = np.where(home_win, 3.0, np.where(draw, 1.0, 0.0))
    away_pts = np.where(~home_win & ~draw, 3.0, np.where(draw, 1.0, 0.0))

    totals = np.tile(base_points, (n_sims, 1))
    sim_rows = np.repeat(np.arange(n_sims), n_fixtures)
    np.add.at(totals, (sim_rows, np.tile(home_idx, n_sims)), home_pts.ravel())
    np.add.at(totals, (sim_rows, np.tile(away_idx, n_sims)), away_pts.ravel())

    top = totals.max(axis=1, keepdims=True)
    winners = totals == top
    credit_per_sim = winners / winners.sum(axis=1, keepdims=True)
    champion_credit = credit_per_sim.sum(axis=0) / n_sims

    return {t: round(float(champion_credit[team_idx[t]]), 4) for t in teams}


def title_race_path(season_matches: pd.DataFrame, full_history: pd.DataFrame,
                     n_sims: int = 500, home_advantage: float = DEFAULT_HOME_ADVANTAGE,
                     draw_rate: float = DEFAULT_DRAW_RATE, seed=None) -> pd.DataFrame:
    """
    For one league+season's matches: title-race win probability for
    every team at each distinct match date in that season, using
    standings and Elo ratings exactly as they stood on that date.

    A season has ~380 matches and tens of checkpoints; re-scanning the
    full match list and the full (multi-season, ~25k-row) Elo history
    from scratch at every checkpoint is O(checkpoints x history) and was
    measured at over two minutes for one real Premier League season.
    Both the standings tally and the ratings-as-of lookup instead advance
    a single sorted pointer once, since checkpoints are processed in
    ascending order -- each checkpoint only needs to apply what happened
    since the last one.

    Returns columns [date, team, win_probability].
    """
    season_matches = season_matches.sort_values("date").reset_index(drop=True)
    all_teams = sorted(set(season_matches["home_team"]) | set(season_matches["away_team"]))
    checkpoints = sorted(season_matches["date"].unique())

    s_dates = season_matches["date"].to_numpy()
    s_homes = season_matches["home_team"].to_numpy()
    s_aways = season_matches["away_team"].to_numpy()
    s_results = season_matches["result"].to_numpy()
    n_matches = len(season_matches)

    history_sorted = full_history.sort_values("date")
    h_dates = history_sorted["date"].to_numpy()
    h_teams = history_sorted["team"].to_numpy()
    h_ratings = history_sorted["rating_after"].to_numpy()
    n_history = len(history_sorted)

    points = {t: 0 for t in all_teams}
    ratings = {}
    match_ptr = 0
    hist_ptr = 0
    rows = []

    for checkpoint in checkpoints:
        while match_ptr < n_matches and s_dates[match_ptr] <= checkpoint:
            r, h, a = s_results[match_ptr], s_homes[match_ptr], s_aways[match_ptr]
            if r == "H":
                points[h] += 3
            elif r == "A":
                points[a] += 3
            else:
                points[h] += 1
                points[a] += 1
            match_ptr += 1

        while hist_ptr < n_history and h_dates[hist_ptr] <= checkpoint:
            ratings[h_teams[hist_ptr]] = h_ratings[hist_ptr]
            hist_ptr += 1

        remaining_fixtures = list(zip(s_homes[match_ptr:], s_aways[match_ptr:]))
        probs = simulate_title_race(dict(points), remaining_fixtures, dict(ratings), n_sims=n_sims,
                                     home_advantage=home_advantage, draw_rate=draw_rate, seed=seed)
        for team, p in probs.items():
            rows.append({"date": checkpoint, "team": team, "win_probability": p})

    return pd.DataFrame(rows)
