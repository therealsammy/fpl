"""
Live Prediction Archive
=========================
Predicts every upcoming fixture with each live source (plain team
strength, the style-adjusted GLM, and the betting market where
available) and archives the result via core.archive.write_forecast
BEFORE kickoff.

This is the only honest way to answer "how is the model actually doing
this season" later -- fpl's own live state (and this repo's own fitted
strengths) get overwritten every week, so without an archive taken
before each result is known, there is no way to reconstruct what the
model once said and check it against reality afterwards.

It is also what makes the GLM's recalibration real rather than
theoretical: every run below refits attack/defense/rho/the GLM's
coefficients from whatever's on disk at run time -- there is no cached
or pickled model that could go stale. New Understat/football-data/odds
data in (see .github/workflows/understat.yml, football_data.yml,
odds.yml) genuinely changes what gets predicted and archived out.

Run weekly via .github/workflows/live_predictions.yml, after odds.yml
so the betting line archived alongside the model is as fresh as
possible. validation.scoreboard.score_archived_forecasts() reads this
archive back once fixtures are played and turns it into a running
log-loss/RPS scoreboard for the current season -- kept as a genuinely
separate question from the historical walk-forward backtest in
run_backtest(), which answers "would this have worked historically",
not "is it working right now".
"""

from datetime import date as _date

import pandas as pd

from collectors.football_data import current_season_start_year, season_code
from core import archive
from models import match
from validation.scoreboard import GLM_COVARIATES

OUTCOMES = ["home_win", "draw", "away_win"]
GLM_STATS = [c.replace("_form", "") for c in GLM_COVARIATES]

# fixture_projections.csv (collectors/fpl_odds.py) only ever covers the
# Premier League -- FPL itself doesn't exist for any other competition.
MARKET_LEAGUE = "E0"


def _fixture_entity_id(league: str, home: str, away: str, fixture_date) -> str:
    """A stable key for one fixture, independent of the exact date --
    league + season + the ordered team pair is unique for a standard
    round-robin league, and stays correct even if a fixture is later
    postponed/rescheduled by a few days within the same season."""
    season = season_code(current_season_start_year(fixture_date))
    return f"{league}|{season}|{home}|{away}"


def predict_upcoming(league: str, matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Fresh predictions for every upcoming fixture in `league`, one row
    per (fixture, source, outcome) in core.archive's schema. Refits
    everything from `matches` every call -- no caching, deliberately
    (see module docstring). Empty if there's no upcoming fixture, or not
    enough history in this league to fit a model at all yet."""
    fixtures = match.upcoming_fixtures(league)
    if fixtures.empty:
        return pd.DataFrame(columns=archive.REQUIRED_COLUMNS + ["source"])

    league_matches = matches[matches["league"] == league].reset_index(drop=True)
    base_strengths = match.fit_team_strengths(league_matches)
    if not base_strengths["teams"]:
        return pd.DataFrame(columns=archive.REQUIRED_COLUMNS + ["source"])
    rho = match.fit_rho(league_matches, base_strengths)

    form_matches = match.compute_rolling_form(league_matches, GLM_STATS)
    glm_strengths = match.fit_team_strengths_glm(form_matches, GLM_COVARIATES)
    has_glm = bool(glm_strengths["teams"])

    rows = []
    for _, fx in fixtures.iterrows():
        home, away, fx_date = fx["home_team"], fx["away_team"], fx["date"]
        entity_id = _fixture_entity_id(league, home, away, fx_date)
        target_event = f"{league} {fx_date.strftime('%Y-%m-%d')} {home} vs {away}"

        sources = {}
        lh, la = match.team_rates(base_strengths, home, away)
        sources["dixon_coles"] = match.match_probabilities(match.score_matrix(lh, la, rho))

        if has_glm:
            home_form = {c: v for c, v in
                        ((c, match.current_form(form_matches, home, s)) for c, s in zip(GLM_COVARIATES, GLM_STATS))
                        if v is not None}
            away_form = {c: v for c, v in
                        ((c, match.current_form(form_matches, away, s)) for c, s in zip(GLM_COVARIATES, GLM_STATS))
                        if v is not None}
            lh_glm, la_glm = match.team_rates_glm(glm_strengths, home, away, home_form, away_form)
            sources["dixon_coles_glm"] = match.match_probabilities(match.score_matrix(lh_glm, la_glm, rho))

        market_row = odds[(odds["home_team"] == home) & (odds["away_team"] == away)]
        if not market_row.empty:
            r = market_row.iloc[0]
            sources["closing_line"] = {"home_win": r["home_win"], "draw": r["draw"], "away_win": r["away_win"]}

        for source, probs in sources.items():
            for outcome in OUTCOMES:
                rows.append({"target_event": target_event, "entity_id": entity_id,
                            "entity_type": "fixture", "metric": outcome,
                            "value": probs[outcome], "source": source})
    return pd.DataFrame(rows)


def archive_all_leagues(as_of: str | None = None) -> dict:
    """Predicts and archives every league's upcoming fixtures. Rows are
    grouped and written ONE write_forecast call per source across ALL
    leagues combined -- write_forecast overwrites its whole (as_of,
    source) file on every call, so writing per-league-per-source would
    have each later league silently erase the previous league's rows
    for that same source and day.

    Returns {source: n_rows_written}, for a quick log line -- not
    intended for programmatic use."""
    as_of = as_of or _date.today().isoformat()
    matches = match.load_matches_with_xg()
    if matches.empty:
        return {}

    market_odds = match.load_live_odds_predictions()
    empty_odds = pd.DataFrame(columns=["home_team", "away_team", "home_win", "draw", "away_win"])

    all_predictions = []
    for league in sorted(matches["league"].unique()):
        odds = market_odds if league == MARKET_LEAGUE else empty_odds
        predictions = predict_upcoming(league, matches, odds)
        if not predictions.empty:
            all_predictions.append(predictions)

    if not all_predictions:
        return {}

    combined = pd.concat(all_predictions, ignore_index=True)
    written = {}
    for source, chunk in combined.groupby("source"):
        archive.write_forecast(source, as_of, chunk.drop(columns=["source"]))
        written[source] = len(chunk)
    return written


if __name__ == "__main__":
    result = archive_all_leagues()
    if result:
        for source, n in sorted(result.items()):
            print(f"live_predictions: archived {n} rows for source={source!r}")
    else:
        print("live_predictions: nothing to archive (no upcoming fixtures found).")
