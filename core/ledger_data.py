"""
Ledger Page Data Loaders
=========================
Thin, Streamlit-cached wrappers around models.elo and models.title_race,
shared by every Ledger_*.py page so the full match load and Elo
computation (~1s over 30 years of data) happens once per cache window
instead of once per page. Kept out of models/ deliberately -- the model
layer stays UI-independent, this is purely the app's data-access glue.
"""

import streamlit as st
import pandas as pd

from collectors.football_data import LEAGUES
from models import elo, match, title_race
from validation import scoreboard

CACHE_TTL = 3600


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading match history…")
def load_matches() -> pd.DataFrame:
    return elo.load_all_matches()


@st.cache_data(ttl=CACHE_TTL, show_spinner="Computing Elo ratings…")
def load_history(k: float = elo.DEFAULT_K,
                  home_advantage: float = elo.DEFAULT_HOME_ADVANTAGE) -> pd.DataFrame:
    return elo.compute_elo_history(load_matches(), k=k, home_advantage=home_advantage)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_current_ratings(k: float = elo.DEFAULT_K,
                          home_advantage: float = elo.DEFAULT_HOME_ADVANTAGE) -> pd.DataFrame:
    return elo.current_ratings(load_history(k, home_advantage))


@st.cache_data(ttl=CACHE_TTL, show_spinner="Simulating…")
def simulate_path(league: str, season: str, n_sims: int, seed: int,
                   k: float = elo.DEFAULT_K,
                   home_advantage: float = elo.DEFAULT_HOME_ADVANTAGE) -> pd.DataFrame:
    """Cached title_race_path for one league+season. History is filtered
    to the same league before passing in -- title_race_path looks up
    ratings by team name alone with no league key, and a same-named team
    in two different leagues (unlikely across these six, but not
    impossible) would otherwise silently cross-contaminate ratings."""
    matches = load_matches()
    season_matches = matches[(matches["league"] == league) & (matches["season"] == season)]
    league_history = load_history(k, home_advantage)
    league_history = league_history[league_history["league"] == league]
    return title_race.title_race_path(season_matches, league_history,
                                       n_sims=n_sims, home_advantage=home_advantage, seed=seed)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_matches_with_xg() -> pd.DataFrame:
    return match.load_matches_with_xg()


@st.cache_data(ttl=CACHE_TTL, show_spinner="Backtesting the match model…")
def load_backtest(league: str, min_train_seasons: int = scoreboard.DEFAULT_MIN_TRAIN_SEASONS) -> pd.DataFrame:
    """Walk-forward backtest for one league, cached -- refitting Dixon-
    Coles for every season is real work (a few seconds per league), not
    something to redo on every widget interaction on the Scoreboard page."""
    matches = load_matches_with_xg()
    return scoreboard.run_backtest(matches, leagues=[league], min_train_seasons=min_train_seasons)


def league_label(code: str) -> str:
    return LEAGUES.get(code, code)


def _season_start_year(code: str) -> int:
    """'2425' -> 2024, '9394' -> 1993. The archive spans 1993-2099 in
    football-data.co.uk's own 2-digit-year codes, so the same 'yy' means
    different centuries depending on range -- e.g. '00' is 2000, but
    '99' is 1999, not 2099. There's over a century of runway in the 50
    cutoff below before this needs revisiting."""
    yy = int(code[:2])
    return (1900 if yy >= 50 else 2000) + yy


def season_label(code: str) -> str:
    """'2425' -> '2024/25', '9394' -> '1993/94'. Falls back to the raw
    code if it doesn't match the expected 4-digit format (shouldn't
    happen, but a label helper should never crash the page over it)."""
    if len(code) == 4 and code.isdigit():
        return f"{_season_start_year(code)}/{code[2:]}"
    return code


def complete_seasons(matches: pd.DataFrame, league: str) -> list:
    """Seasons for one league with (close to) a full match count -- i.e.
    every fixture in that season has actually been played. A season
    still in progress only has however many matches have happened so
    far in the archive; treating that partial list as 'all remaining
    fixtures' would make simulate_title_race think the season is over
    the moment the data run stops, and hand the current leader a false
    100%. football-data.co.uk never carries unplayed fixtures, so
    there's no way to simulate a genuinely live season from this
    archive alone -- excluding incomplete seasons here is the honest
    choice, not a missing feature.

    Sorted by actual start year, not the raw code string -- '9394'
    sorts after '2425' lexicographically, which would put 1993/94 at
    the end of a list that's supposed to run chronologically.

    Reference count is the MODE, not the max -- the Premier League ran
    22 teams (462 matches) through 1994/95 before dropping to 20 (380).
    Using the max would treat every one of the 29 modern 380-match
    seasons as "incomplete" because they're smaller than two 30-year-old
    outliers -- verified against the real archive before fixing this."""
    counts = matches[matches["league"] == league].groupby("season").size()
    if counts.empty:
        return []
    full = counts.mode().iloc[0]
    codes = counts[counts >= full * 0.95].index.tolist()
    return sorted(codes, key=_season_start_year)
