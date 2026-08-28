"""
Forecast Scoreboard
=====================
Grades every match-outcome forecast source against the same held-out
matches: Dixon-Coles trained on goals, Dixon-Coles trained on xG, both
blended with the closing line, the closing line itself, and two simple
baselines (home-advantage-only, Elo). All scored with the same two
metrics -- log loss and the Ranked Probability Score (RPS) -- so no
source gets a home-field advantage in how it's judged.

BRIEFS.md Phase 6's mandated presentation order (calibration first,
then beating baselines, then the market beating everything) is a page
concern, not a computation one -- it lives in Ledger_Scoreboard.py.
This module only produces honest numbers, in no particular order.

BACKTEST DESIGN: walk-forward at SEASON granularity, not per-matchday.
Refitting Dixon-Coles before every single match across a decade of
history would be far more "live-accurate" but is needless cost for
what this validates -- team strength doesn't move fast enough within a
season for a per-matchday refit to meaningfully change the verdict here.
For each league, the first `min_train_seasons` seasons are training-only
(no backtest rows); every season after that is predicted using a model
fit ONLY on strictly earlier matches (an expanding window), then the
next season becomes eligible for training too. No test-season match
ever contributes to its own prediction.
"""

import numpy as np
import pandas as pd

from core import archive
from models import elo, match, title_race

OUTCOME_COLS = ["away_win", "draw", "home_win"]   # a real order (from the home team's view), for RPS
RESULT_TO_COL = {"A": "away_win", "D": "draw", "H": "home_win"}

DEFAULT_MIN_TRAIN_SEASONS = 3
DEFAULT_HOME_ADVANTAGE_ELO = 100.0


# ---------------------------------------------------------------------------
# SCORING METRICS (pure -- operate on a scored DataFrame)
# ---------------------------------------------------------------------------

def log_loss(scored: pd.DataFrame) -> float:
    """Mean -log(probability assigned to what actually happened).
    Lower is better; a perfect, certain, always-right forecaster scores
    0. `scored` needs home_win/draw/away_win probability columns and a
    `result` column of 'H'/'D'/'A'."""
    if scored.empty:
        return float("nan")
    p_actual = scored.apply(lambda r: r[RESULT_TO_COL[r["result"]]], axis=1)
    p_actual = np.clip(p_actual.to_numpy(dtype=float), 1e-10, 1.0)   # a confident wrong call must cost a lot, not raise
    return float(-np.log(p_actual).mean())


def ranked_probability_score(scored: pd.DataFrame) -> float:
    """Mean RPS across `scored` -- like log loss, but credits a
    near-miss (predicting a draw when the away side actually won) more
    than a total miss (predicting a home win for the same result),
    since win/draw/loss is a genuinely ordered scale from either side's
    perspective. Lower is better; 0 is perfect."""
    if scored.empty:
        return float("nan")
    probs = scored[OUTCOME_COLS].to_numpy(dtype=float)
    actual = np.zeros_like(probs)
    for i, result in enumerate(scored["result"]):
        actual[i, OUTCOME_COLS.index(RESULT_TO_COL[result])] = 1.0

    cum_probs = np.cumsum(probs, axis=1)
    cum_actual = np.cumsum(actual, axis=1)
    # Standard RPS normalization by (n_categories - 1); the final
    # cumulative column is always (1, 1) for both and contributes 0.
    rps = ((cum_probs - cum_actual) ** 2).sum(axis=1) / (len(OUTCOME_COLS) - 1)
    return float(rps.mean())


def calibration_curve(scored: pd.DataFrame, outcome: str = "home_win", n_bins: int = 10) -> pd.DataFrame:
    """Reliability curve for one outcome column: bucket its predicted
    probability into `n_bins` equal-width bins, and compare the mean
    predicted probability in each bin against how often that outcome
    actually happened. A well-calibrated source tracks the diagonal --
    same idea and same shape as Ledger_Market_Efficiency.py's curve for
    the closing line, generalized here to any source."""
    if scored.empty:
        return pd.DataFrame(columns=["bin", "predicted", "actual", "n"])

    df = scored.copy()
    outcome_result = {v: k for k, v in RESULT_TO_COL.items()}[outcome]
    df["actual"] = (df["result"] == outcome_result).astype(int)
    df["bin"] = pd.cut(df[outcome], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)

    grouped = df.groupby("bin", observed=True).agg(
        predicted=(outcome, "mean"), actual=("actual", "mean"), n=("actual", "size")).reset_index()
    return grouped[grouped["n"] >= 5]


# ---------------------------------------------------------------------------
# BASELINES
# ---------------------------------------------------------------------------

def home_advantage_baseline(train_matches: pd.DataFrame) -> dict:
    """The dumbest baseline that still uses real information: overall
    home/draw/away frequency in the training window, applied as one
    constant prediction regardless of who's playing. Anything worth
    calling a model needs to clear this bar using team identity at all."""
    counts = train_matches["result"].value_counts(normalize=True)
    return {"home_win": float(counts.get("H", 0.0)),
           "draw": float(counts.get("D", 0.0)),
           "away_win": float(counts.get("A", 0.0))}


def elo_baseline_predictions(history_before_test: pd.DataFrame, test_matches: pd.DataFrame,
                             home_advantage: float = DEFAULT_HOME_ADVANTAGE_ELO) -> pd.DataFrame:
    """Elo-implied 1X2 for every match in `test_matches`, using ratings
    as they stood at the END of the training window (frozen for the
    whole test season -- a coarser approximation than Ledger_Title_Races'
    per-checkpoint ratings, acceptable here since this is a baseline,
    not the thing being sold as accurate).

    Takes an already-computed Elo HISTORY (elo.compute_elo_history's
    output, pre-filtered to matches before the test window), not raw
    matches to refit from scratch. run_backtest computes this once per
    league up front -- recomputing full Elo history at every one of
    ~30 season boundaries turned out to be the dominant cost in this
    module's first version (330+ seconds for one league), the exact
    same O(seasons x history) shape as an earlier, already-fixed bug in
    models/title_race.py's title_race_path()."""
    if history_before_test.empty:
        ratings = {}
    else:
        current = elo.current_ratings(history_before_test)
        ratings = dict(zip(current["team"], current["rating"]))

    rows = []
    for _, m in test_matches.iterrows():
        r_home = ratings.get(m["home_team"], elo.STARTING_RATING)
        r_away = ratings.get(m["away_team"], elo.STARTING_RATING)
        p_home, p_draw, p_away = title_race.match_outcome_probabilities(r_home, r_away, home_advantage)
        rows.append({"home_win": p_home, "draw": p_draw, "away_win": p_away})
    return pd.DataFrame(rows, index=test_matches.index)


# ---------------------------------------------------------------------------
# DIXON-COLES PREDICTIONS FOR A TEST SET
# ---------------------------------------------------------------------------

TARGET_COLUMNS = {
    "goals": ("home_goals", "away_goals"),
    "xg": ("home_xg", "away_xg"),
    "npxg": ("home_npxg", "away_npxg"),
}


def dixon_coles_predictions(train_matches: pd.DataFrame, test_matches: pd.DataFrame,
                            target: str = "goals", half_life_days: float = match.DEFAULT_HALF_LIFE_DAYS) -> pd.DataFrame:
    """Fits on `train_matches` (target='goals', 'xg', or 'npxg' -- see
    TARGET_COLUMNS), predicts every fixture in `test_matches`. Rows
    whose target is entirely missing in the training window (e.g. xG
    requested for a league/era Understat doesn't cover) come back as
    NaN probabilities, not a crash or a silent fallback to goals -- a
    caller asking for xG results should find out plainly when there
    wasn't any, not get goals unlabeled as xG."""
    home_col, away_col = TARGET_COLUMNS.get(target, TARGET_COLUMNS["goals"])

    strengths = match.fit_team_strengths(train_matches, half_life_days=half_life_days,
                                         home_target_col=home_col, away_target_col=away_col)
    if not strengths["teams"]:
        nan_row = {c: np.nan for c in OUTCOME_COLS}
        return pd.DataFrame([nan_row] * len(test_matches), index=test_matches.index)

    rho = match.fit_rho(train_matches, strengths, half_life_days=half_life_days)

    rows = []
    for _, m in test_matches.iterrows():
        probs = match.predict_match(strengths, rho, m["home_team"], m["away_team"])
        rows.append({"home_win": probs["home_win"], "draw": probs["draw"], "away_win": probs["away_win"]})
    return pd.DataFrame(rows, index=test_matches.index)


GLM_COVARIATES = ["npxg_form", "ppda_form", "deep_form"]


def dixon_coles_glm_predictions(train_matches: pd.DataFrame, test_matches: pd.DataFrame,
                                covariates: list = GLM_COVARIATES,
                                half_life_days: float = match.DEFAULT_HALF_LIFE_DAYS) -> pd.DataFrame:
    """The GLM extension (models.match.fit_team_strengths_glm), fit on
    goals with rolling-form covariates layered on top, predicting every
    fixture in `test_matches`. `train_matches`/`test_matches` must
    already carry home_{c}_form/away_{c}_form columns -- see
    models.match.compute_rolling_form(), called once per league in
    run_backtest() rather than per season (cheap, and there's no
    leakage risk in computing it over the whole league up front: each
    row's own form value already only reflects strictly earlier
    matches by construction).

    rho is still fit on the plain (non-GLM) attack/defense strengths --
    the low-score correlation correction has nothing to do with the
    covariate terms, same reasoning as fit_rho() itself."""
    strengths = match.fit_team_strengths_glm(train_matches, covariates, half_life_days=half_life_days)
    if not strengths["teams"]:
        nan_row = {c: np.nan for c in OUTCOME_COLS}
        return pd.DataFrame([nan_row] * len(test_matches), index=test_matches.index)

    base_strengths = match.fit_team_strengths(train_matches, half_life_days=half_life_days)
    rho = match.fit_rho(train_matches, base_strengths, half_life_days=half_life_days)

    rows = []
    for _, m in test_matches.iterrows():
        home_form = {c: m.get(f"home_{c}") for c in covariates}
        away_form = {c: m.get(f"away_{c}") for c in covariates}
        if any(pd.isna(v) for v in list(home_form.values()) + list(away_form.values())):
            rows.append({c: np.nan for c in OUTCOME_COLS})
            continue
        lambda_home, lambda_away = match.team_rates_glm(strengths, m["home_team"], m["away_team"],
                                                        home_form, away_form)
        grid = match.score_matrix(lambda_home, lambda_away, rho)
        probs = match.match_probabilities(grid)
        rows.append({"home_win": probs["home_win"], "draw": probs["draw"], "away_win": probs["away_win"]})
    return pd.DataFrame(rows, index=test_matches.index)


def _blend_rows(model_probs: pd.DataFrame, market_probs: pd.DataFrame, market_weight: float) -> pd.DataFrame:
    """Row-wise blend_with_market -- a row with no complete market
    quote (no odds collected, or a null model probability) passes
    through as the model's own, unblended value rather than blending
    against a partial/missing market read."""
    blended = []
    for idx in model_probs.index:
        model_row = model_probs.loc[idx].to_dict()
        market_row = market_probs.loc[idx]
        if market_row.notna().all() and pd.notna(list(model_row.values())).all():
            blended.append(match.blend_with_market(model_row, market_row.to_dict(), market_weight))
        else:
            blended.append(model_row)
    return pd.DataFrame(blended, index=model_probs.index)


def market_predictions(test_matches: pd.DataFrame) -> pd.DataFrame:
    """The closing line's own implied probabilities, already devigged
    (Shin) in Phase 3 -- NaN wherever a match has no odds on record."""
    return pd.DataFrame({
        "home_win": test_matches.get("prob_home_shin", np.nan),
        "draw": test_matches.get("prob_draw_shin", np.nan),
        "away_win": test_matches.get("prob_away_shin", np.nan),
    }, index=test_matches.index)


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def run_backtest(matches: pd.DataFrame, leagues=None,
                 min_train_seasons: int = DEFAULT_MIN_TRAIN_SEASONS,
                 half_life_days: float = match.DEFAULT_HALF_LIFE_DAYS,
                 market_weight: float = 0.5) -> pd.DataFrame:
    """
    Walk-forward backtest across every league in `matches` (output of
    models.match.load_matches_with_xg()). Returns one row per
    (match, source) with home_win/draw/away_win probabilities plus the
    match's actual `result` -- ready for log_loss()/ranked_probability_
    score()/calibration_curve(), each optionally filtered to one source
    first via `scored[scored["source"] == ...]`.

    Sources: home_advantage_baseline, elo_baseline, dixon_coles,
    dixon_coles_blended, dixon_coles_xg (only where xG coverage exists),
    dixon_coles_xg_blended, dixon_coles_npxg(_blended), dixon_coles_glm
    (only where GLM_COVARIATES coverage exists), closing_line.
    """
    matches = matches.dropna(subset=["result"]).sort_values("date").reset_index(drop=True)
    all_scored = []

    leagues = leagues or sorted(matches["league"].unique())
    for league in leagues:
        league_matches = matches[matches["league"] == league].reset_index(drop=True)
        seasons = sorted(league_matches["season"].unique(),
                         key=lambda s: (int(s[:2]) if int(s[:2]) < 50 else int(s[:2]) - 100))
        if len(seasons) <= min_train_seasons:
            continue

        # Computed ONCE for the whole league, not re-fit at every season
        # boundary -- see elo_baseline_predictions' docstring. Sliced by
        # date per test season below instead of recomputed.
        full_elo_history = elo.compute_elo_history(league_matches)

        # Same one-pass-per-league reasoning applies to rolling form:
        # each row's own form value only ever reflects strictly earlier
        # matches (compute_rolling_form's shift(1)), so computing it once
        # over the whole league carries no leakage risk across the
        # season-boundary slices taken below.
        glm_stats = [c.replace("_form", "") for c in GLM_COVARIATES]
        league_matches = match.compute_rolling_form(league_matches, glm_stats)

        for i in range(min_train_seasons, len(seasons)):
            test_season = seasons[i]
            train = league_matches[league_matches["season"].isin(seasons[:i])]
            test = league_matches[league_matches["season"] == test_season]
            if train.empty or test.empty:
                continue

            base_cols = test[["date", "league", "home_team", "away_team", "result"]].reset_index(drop=True)
            test = test.reset_index(drop=True)
            elo_history_before_test = full_elo_history[full_elo_history["date"] < test["date"].min()]

            sources = {
                "home_advantage_baseline": pd.DataFrame(
                    [home_advantage_baseline(train)] * len(test), index=test.index),
                "elo_baseline": elo_baseline_predictions(elo_history_before_test, test),
                "dixon_coles": dixon_coles_predictions(train, test, "goals", half_life_days),
                "closing_line": market_predictions(test),
            }
            sources["dixon_coles_blended"] = _blend_rows(
                sources["dixon_coles"], sources["closing_line"], market_weight)

            if train["home_xg"].notna().any():
                sources["dixon_coles_xg"] = dixon_coles_predictions(train, test, "xg", half_life_days)
                sources["dixon_coles_xg_blended"] = _blend_rows(
                    sources["dixon_coles_xg"], sources["closing_line"], market_weight)

            if "home_npxg" in train.columns and train["home_npxg"].notna().any():
                sources["dixon_coles_npxg"] = dixon_coles_predictions(train, test, "npxg", half_life_days)
                sources["dixon_coles_npxg_blended"] = _blend_rows(
                    sources["dixon_coles_npxg"], sources["closing_line"], market_weight)

            glm_cols = [f"home_{c}" for c in GLM_COVARIATES]
            if all(c in train.columns for c in glm_cols) and train[glm_cols].notna().all(axis=1).any():
                sources["dixon_coles_glm"] = dixon_coles_glm_predictions(train, test, GLM_COVARIATES, half_life_days)
                sources["dixon_coles_glm_blended"] = _blend_rows(
                    sources["dixon_coles_glm"], sources["closing_line"], market_weight)

            for source_name, probs in sources.items():
                chunk = base_cols.copy()
                chunk["source"] = source_name
                chunk[OUTCOME_COLS] = probs[OUTCOME_COLS].to_numpy()
                all_scored.append(chunk)

    if not all_scored:
        return pd.DataFrame(columns=["date", "league", "home_team", "away_team", "result", "source"] + OUTCOME_COLS)
    return pd.concat(all_scored, ignore_index=True)


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per source: log loss, RPS, and sample size -- dropping
    rows with any NaN probability first (a source with no coverage for
    a given match, e.g. xG before 2014/15, must not silently get
    credit or blame for matches it never actually predicted)."""
    rows = []
    for source, group in scored.groupby("source"):
        clean = group.dropna(subset=OUTCOME_COLS)
        rows.append({"source": source, "log_loss": log_loss(clean),
                     "rps": ranked_probability_score(clean), "n": len(clean)})
    return pd.DataFrame(rows).sort_values("log_loss")


def score_archived_forecasts(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Turns models.live_predictions.py's archive (data/forecasts/) into a
    "scored" DataFrame in the exact same shape run_backtest() produces
    (date, league, home_team, away_team, result, source, home_win, draw,
    away_win) -- so summarize()/log_loss()/ranked_probability_score()
    work identically on both, unmodified.

    Deliberately a SEPARATE question from run_backtest(): this only
    covers fixtures the live archive actually predicted in real time,
    scored only once their real result exists -- "is the model doing
    well right now," never "would it have worked historically." Only a
    fixture whose entity_id resolves to an ACTUAL played match (result
    known) is included; a forecast for a fixture not yet played is
    silently excluded, not scored as wrong or blank.

    entity_id is "{league}|{season}|{home_team}|{away_team}" (see
    live_predictions._fixture_entity_id) -- league+season+ordered team
    pair is unique for a standard round-robin league, and survives a
    fixture being postponed a few days within the same season, unlike
    joining on the exact archived date.
    """
    forecasts = archive.read_forecasts()
    forecasts = forecasts[forecasts["entity_type"] == "fixture"]
    if forecasts.empty:
        return pd.DataFrame(columns=["date", "league", "home_team", "away_team", "result", "source"] + OUTCOME_COLS)

    parts = forecasts["entity_id"].str.split("|", expand=True)
    forecasts = forecasts.assign(league=parts[0], season=parts[1], home_team=parts[2], away_team=parts[3])

    wide = forecasts.pivot_table(index=["league", "season", "home_team", "away_team", "source"],
                                  columns="metric", values="value", aggfunc="last").reset_index()

    matches = matches if matches is not None else elo.load_all_matches()
    results = matches.dropna(subset=["result"])[["league", "season", "home_team", "away_team", "date", "result"]]
    scored = wide.merge(results, on=["league", "season", "home_team", "away_team"], how="inner")
    if scored.empty:
        return pd.DataFrame(columns=["date", "league", "home_team", "away_team", "result", "source"] + OUTCOME_COLS)

    return scored[["date", "league", "home_team", "away_team", "result", "source"] + OUTCOME_COLS]
