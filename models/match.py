"""
Match Outcome Model -- Dixon-Coles
=====================================
Time-weighted Dixon-Coles (SPEC.md Phase 6): each team gets an attack
and a defense strength from historical scoring, a global home
advantage, and a low-score correlation correction (rho) fit against
real scorelines. Produces 1X2, over/under 2.5, both-teams-to-score, and
clean-sheet probabilities for any fixture.

WHY DIXON-COLES OVER PLAIN INDEPENDENT POISSON: modeling home and away
goals as independent Poisson variables systematically mispredicts low
scores -- real football has more 0-0 and 1-1 draws than independence
implies, because a game state itself changes how both teams play (a
team ahead 1-0 defends differently than 0-0). Dixon & Coles (1997)
correct exactly this with a single parameter, rho, applied only to the
four low-score cells (0-0, 1-0, 0-1, 1-1) via tau().

TRAINING TARGET IS PLUGGABLE (goals or xG): fit_team_strengths() takes
whatever's in `home_target`/`away_target` and treats it as a Poisson
rate to explain -- it doesn't need to be an integer goal count. Expected
Goals (xG) is a strictly better training signal than actual goals
(actual goals are noisy -- a team can dominate on xG and lose 0-1), so
the same function fits equally well on either. The log-likelihood uses
gammaln (the Gamma function) instead of a plain factorial specifically
so it stays well-defined for fractional xG values, not just integers --
it reduces to the exact standard Poisson log-likelihood when the target
happens to be an integer goal count.

rho is different: it corrects a pattern in ACTUAL final scorelines (real
football has extra 0-0s and 1-1s), so it is always fit against real
goals, never xG, regardless of what target trained the attack/defense
ratings. See fit_rho().

IDENTIFIABILITY: attack_i and defense_i are only determined up to a
shared additive constant (shifting every attack up by c and every
defense down by c leaves every predicted score rate unchanged) -- a
flat direction in the likelihood surface with no unique optimum. A
light penalty pins mean(attack) near zero so the optimizer converges to
one specific, comparable set of numbers instead of an arbitrary point
along that flat line; it does not change any predicted probability.

MARKET BLENDING (optional, separate from the standalone model): the
closing line is itself a strong predictor, and blend_with_market()
linearly combines this model's probabilities with the market's. Linear
blending of two valid probability distributions is always itself a
valid distribution (no renormalization needed), which is why it's used
here instead of a log-odds blend. This is kept as an explicit, separate
step -- never inside fit_team_strengths() -- so "the model" and "the
model blended with the market" can be scored as two distinct entries on
the Forecast Scoreboard (Phase 6) rather than one number that quietly
contains the other.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from models import elo

XG_ROOT = Path("data/xg/understat")

DEFAULT_HALF_LIFE_DAYS = 180.0
DEFAULT_MAX_GOALS = 10
DEFAULT_RHO_BOUNDS = (-0.3, 0.3)
FALLBACK_STRENGTH = 0.0   # a team with no fitted rating gets exactly average attack/defense


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

# Every per-match Understat field worth carrying into the merged frame,
# beyond home_xg/away_xg -- see collectors/understat.py's module
# docstring for what each one is and why it's collected.
XG_EXTRA_COLUMNS = [
    "home_npxg", "away_npxg", "home_npxga", "away_npxga",
    "home_ppda", "away_ppda", "home_deep", "away_deep",
    "home_xpts", "away_xpts",
    "understat_forecast_home_win", "understat_forecast_draw", "understat_forecast_away_win",
]


def _load_xg(leagues=None) -> pd.DataFrame:
    empty_cols = ["league", "season", "date", "home_team_raw", "away_team_raw",
                  "home_team_id", "away_team_id", "home_xg", "away_xg"] + XG_EXTRA_COLUMNS
    if not XG_ROOT.exists():
        return pd.DataFrame(columns=empty_cols)
    league_dirs = ([XG_ROOT / lg for lg in leagues] if leagues
                    else sorted(d for d in XG_ROOT.iterdir() if d.is_dir()))
    frames = [pd.read_csv(f, dtype={"season": str})
              for d in league_dirs if d.exists() for f in d.glob("*.csv")]
    if not frames:
        return pd.DataFrame(columns=empty_cols)
    return pd.concat(frames, ignore_index=True)


def load_matches_with_xg(leagues=None) -> pd.DataFrame:
    """
    Every collected match (models.elo.load_all_matches -- results and
    odds, all six leagues, 1993-present) left-joined with Understat's
    per-match numbers where they exist (five of the six leagues,
    2014/15-present): xG, npxG/npxGA, PPDA, deep completions, xPts, and
    Understat's own pre-match forecast (see XG_EXTRA_COLUMNS). All of
    these are null wherever Understat has no coverage -- the English
    Championship entirely, every match before 2014/15, and any club
    the id registry still can't resolve -- real, permanent gaps, not a
    loading bug.

    Joined on (league, season, date, home_team_id, away_team_id) -- NOT
    on raw team names, even normalized ones. Verified live before
    picking this: football_data.co.uk and Understat spell the same club
    differently often enough that name-matching quietly fails --
    "Man United" vs "Manchester United" scores 0.741 on the project's
    own fuzzy matcher, well under the 0.85 threshold that gates a real
    match. The two collectors already solved this correctly at
    collection time via core.ids.resolve_team(), so this reuses that
    resolution instead of redoing weaker name-matching here.

    Coverage as of the historical-team registry expansion (2026-08-28):
    every Premier League club that's been relegated at some point since
    2014/15 is now registered too (core.ids.register_historical_teams),
    not just FPL's current 20 -- previously this join only actually
    connected for whichever clubs happened to be in the top flight
    *this specific season*. The other four leagues (La Liga/Bundesliga/
    Serie A/Ligue 1) still don't resolve at all -- the registry has no
    non-English seed yet (StatsBomb, Phase 9). Rows with a null id on
    either side are excluded from the id-merge entirely rather than
    merged on (NaN, NaN) -- pandas treats NaN as equal to NaN in a join
    key, which would otherwise cross-match unrelated fixtures that both
    merely failed to resolve a team id.
    """
    base = elo.load_all_matches(leagues)
    xg = _load_xg(leagues)
    if xg.empty:
        base["home_xg"] = pd.NA
        base["away_xg"] = pd.NA
        for col in XG_EXTRA_COLUMNS:
            base[col] = pd.NA
        return base

    base = base.copy()
    resolvable = base["home_team_id"].notna() & base["away_team_id"].notna()
    xg_priced = xg.dropna(subset=["home_team_id", "away_team_id"])
    xg_cols = ["home_xg", "away_xg"] + XG_EXTRA_COLUMNS
    # Tolerant of an older/partial xg frame missing some of the newer
    # columns (a stale collected file, or a hand-built test fixture) --
    # reindex fills anything absent with NaN rather than a KeyError.
    xg_priced = xg_priced.reindex(
        columns=["league", "season", "date", "home_team_id", "away_team_id"] + xg_cols)

    merged_resolved = base[resolvable].merge(
        xg_priced, on=["league", "season", "date", "home_team_id", "away_team_id"], how="left")
    unresolved = base[~resolvable].copy()
    for col in xg_cols:
        unresolved[col] = pd.NA

    return pd.concat([merged_resolved, unresolved], ignore_index=True).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# TIME WEIGHTING
# ---------------------------------------------------------------------------

def time_weights(dates: pd.Series, as_of, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> np.ndarray:
    """Exponential decay by age in days -- a match half_life_days old
    counts half as much as one from today. Expressed as a half-life
    rather than a raw decay rate because a half-life in days is what
    someone tuning this can actually reason about."""
    as_of = pd.Timestamp(as_of)
    age_days = (as_of - pd.to_datetime(dates)).dt.days.clip(lower=0)
    xi = np.log(2) / half_life_days
    return np.exp(-xi * age_days.to_numpy())


# ---------------------------------------------------------------------------
# STAGE 1 -- ATTACK / DEFENSE / HOME ADVANTAGE (target-agnostic: goals or xG)
# ---------------------------------------------------------------------------

def _poisson_log_pmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """log Poisson pmf via gammaln instead of log(factorial(k)) --
    identical to the standard Poisson log-likelihood for integer k, but
    also well-defined for fractional k (xG), unlike scipy.stats.poisson
    which requires integer counts."""
    return k * np.log(lam) - lam - gammaln(k + 1)


def _unpack(params: np.ndarray, n_teams: int):
    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    home_advantage = params[2 * n_teams]
    return attack, defense, home_advantage


def _neg_log_likelihood(params: np.ndarray, home_idx, away_idx,
                        home_target, away_target, weights, n_teams: int,
                        mean_attack_penalty: float = 1000.0) -> float:
    attack, defense, home_advantage = _unpack(params, n_teams)

    lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_advantage)
    lambda_away = np.exp(attack[away_idx] + defense[home_idx])

    ll = weights * (_poisson_log_pmf(home_target, lambda_home)
                    + _poisson_log_pmf(away_target, lambda_away))

    # Identifiability penalty (see module docstring) -- pins the one
    # genuinely flat direction, doesn't touch any predicted probability.
    penalty = mean_attack_penalty * attack.mean() ** 2
    return -ll.sum() + penalty


def _neg_log_likelihood_grad(params: np.ndarray, home_idx, away_idx,
                             home_target, away_target, weights, n_teams: int,
                             mean_attack_penalty: float = 1000.0) -> np.ndarray:
    """Analytic gradient of _neg_log_likelihood. Poisson log-likelihood
    has a clean closed form here (d/d(log lambda) of the log-pmf is just
    target - lambda), so without this L-BFGS-B falls back to numerical
    (finite-difference) gradients -- ~2 extra full objective evaluations
    per parameter per iteration. That was the dominant cost in a walk-
    forward backtest (one league took over four minutes): with ~40
    parameters, finite-difference gradients alone were roughly 80x the
    cost of one plain objective evaluation, every single iteration.
    Verified against scipy's own finite-difference estimate in tests
    before trusting this -- a wrong analytic gradient would be worse
    than none, since L-BFGS-B trusts it completely."""
    attack, defense, home_advantage = _unpack(params, n_teams)
    lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_advantage)
    lambda_away = np.exp(attack[away_idx] + defense[home_idx])

    # d(log-likelihood)/d(log lambda) = target - lambda, for a Poisson
    # log-pmf -- and log(lambda) is exactly the linear combination of
    # parameters each one enters through, so this IS the per-match
    # contribution to every parameter's gradient it touches.
    resid_home = weights * (home_target - lambda_home)
    resid_away = weights * (away_target - lambda_away)

    grad_attack = np.zeros(n_teams)
    grad_defense = np.zeros(n_teams)
    np.add.at(grad_attack, home_idx, resid_home)    # attack[home] enters lambda_home
    np.add.at(grad_attack, away_idx, resid_away)    # attack[away] enters lambda_away
    np.add.at(grad_defense, away_idx, resid_home)   # defense[away] enters lambda_home
    np.add.at(grad_defense, home_idx, resid_away)   # defense[home] enters lambda_away
    grad_home_adv = resid_home.sum()

    grad_attack = -grad_attack + mean_attack_penalty * 2 * attack.mean() / n_teams
    grad_defense = -grad_defense
    return np.concatenate([grad_attack, grad_defense, [-grad_home_adv]])


def fit_team_strengths(matches: pd.DataFrame, as_of=None,
                       half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                       home_col: str = "home_team", away_col: str = "away_team",
                       home_target_col: str = "home_goals", away_target_col: str = "away_goals") -> dict:
    """
    Fits attack/defense/home_advantage from `matches`. `home_target_col`/
    `away_target_col` point at whatever column should be treated as the
    Poisson rate to explain -- pass the actual goals columns, or xG
    columns, interchangeably (see module docstring).

    Returns {"teams": {team: {"attack": a, "defense": d}}, "home_advantage": h,
    "as_of": as_of}. Silently drops rows with a null target (an unplayed
    fixture, or a match with no xG on record) -- there's nothing to fit
    on. The SAME empty result comes back if the target columns don't
    exist in `matches` at all (e.g. xG requested against a frame that
    was never joined with Understat data) -- a caller asking for xG
    should find out plainly that there was none, not get a KeyError.
    """
    if home_target_col not in matches.columns or away_target_col not in matches.columns:
        return {"teams": {}, "home_advantage": 0.0, "as_of": as_of}

    matches = matches.dropna(subset=[home_target_col, away_target_col]).reset_index(drop=True)
    if matches.empty:
        return {"teams": {}, "home_advantage": 0.0, "as_of": as_of}

    as_of = as_of or matches["date"].max()
    teams = sorted(set(matches[home_col]) | set(matches[away_col]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = matches[home_col].map(team_idx).to_numpy()
    away_idx = matches[away_col].map(team_idx).to_numpy()
    home_target = matches[home_target_col].to_numpy(dtype=float)
    away_target = matches[away_target_col].to_numpy(dtype=float)
    weights = time_weights(matches["date"], as_of, half_life_days)

    x0 = np.zeros(2 * n_teams + 1)
    result = minimize(_neg_log_likelihood, x0, jac=_neg_log_likelihood_grad,
                      args=(home_idx, away_idx, home_target, away_target, weights, n_teams),
                      method="L-BFGS-B")

    attack, defense, home_advantage = _unpack(result.x, n_teams)
    return {
        "teams": {t: {"attack": float(attack[i]), "defense": float(defense[i])} for t, i in team_idx.items()},
        "home_advantage": float(home_advantage),
        "as_of": as_of,
    }


# ---------------------------------------------------------------------------
# STAGE 2 -- RHO (always fit against real scorelines, never xG)
# ---------------------------------------------------------------------------

def _tau(x: int, y: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """The Dixon-Coles low-score correction. 1.0 (no correction) for
    every scoreline except the four where independent Poisson is known
    to be systematically wrong."""
    if x == 0 and y == 0:
        return 1 - lambda_home * lambda_away * rho
    if x == 0 and y == 1:
        return 1 + lambda_home * rho
    if x == 1 and y == 0:
        return 1 + lambda_away * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _tau_array(home_goals: np.ndarray, away_goals: np.ndarray, lambda_home: np.ndarray,
               lambda_away: np.ndarray, rho: float) -> np.ndarray:
    """Vectorized _tau -- same four-cell correction, applied via boolean
    masks across a whole match array instead of scalar-by-scalar. Used
    by _rho_neg_log_likelihood, which minimize_scalar calls many times
    per fit; a per-row Python loop here was the dominant cost in a walk-
    forward backtest (one full league took 330+ seconds before this)."""
    tau = np.ones_like(home_goals, dtype=float)
    mask00 = (home_goals == 0) & (away_goals == 0)
    mask01 = (home_goals == 0) & (away_goals == 1)
    mask10 = (home_goals == 1) & (away_goals == 0)
    mask11 = (home_goals == 1) & (away_goals == 1)
    tau[mask00] = 1 - lambda_home[mask00] * lambda_away[mask00] * rho
    tau[mask01] = 1 + lambda_home[mask01] * rho
    tau[mask10] = 1 + lambda_away[mask10] * rho
    tau[mask11] = 1 - rho
    return tau


def _rho_neg_log_likelihood(rho: float, lambda_home, lambda_away, home_goals, away_goals, weights) -> float:
    base = _poisson_log_pmf(home_goals, lambda_home) + _poisson_log_pmf(away_goals, lambda_away)
    tau = _tau_array(home_goals, away_goals, lambda_home, lambda_away, rho)
    tau = np.clip(tau, 1e-10, None)   # a pathological rho can drive tau <= 0 -- clip rather than log(negative)
    return -(weights * (base + np.log(tau))).sum()


def fit_rho(matches: pd.DataFrame, strengths: dict, half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
           home_col: str = "home_team", away_col: str = "away_team",
           home_goals_col: str = "home_goals", away_goals_col: str = "away_goals",
           bounds=DEFAULT_RHO_BOUNDS) -> float:
    """rho is fit against ACTUAL goals only (see module docstring),
    using the already-fitted lambda for each match -- it corrects a
    property of real scorelines, not of whatever trained the ratings."""
    matches = matches.dropna(subset=[home_goals_col, away_goals_col]).reset_index(drop=True)
    if matches.empty:
        return 0.0

    teams = strengths["teams"]
    home_adv = strengths["home_advantage"]
    avg = {"attack": np.mean([t["attack"] for t in teams.values()]) if teams else FALLBACK_STRENGTH,
          "defense": np.mean([t["defense"] for t in teams.values()]) if teams else FALLBACK_STRENGTH}

    def strength(team, key):
        return teams.get(team, avg)[key]

    lambda_home = np.array([
        np.exp(strength(h, "attack") + strength(a, "defense") + home_adv)
        for h, a in zip(matches[home_col], matches[away_col])
    ])
    lambda_away = np.array([
        np.exp(strength(a, "attack") + strength(h, "defense"))
        for h, a in zip(matches[home_col], matches[away_col])
    ])
    weights = time_weights(matches["date"], strengths.get("as_of") or matches["date"].max(), half_life_days)

    result = minimize_scalar(_rho_neg_log_likelihood, bounds=bounds, method="bounded",
                             args=(lambda_home, lambda_away,
                                   matches[home_goals_col].to_numpy(),
                                   matches[away_goals_col].to_numpy(), weights))
    return float(result.x)


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------

def team_rates(strengths: dict, home_team: str, away_team: str) -> tuple:
    """(lambda_home, lambda_away) for one fixture. A team missing from
    `strengths` (e.g. newly promoted, or simply unrated) falls back to
    exactly-average attack/defense -- a neutral assumption, not a guess
    dressed up as one."""
    teams = strengths["teams"]
    avg = {"attack": np.mean([t["attack"] for t in teams.values()]) if teams else FALLBACK_STRENGTH,
          "defense": np.mean([t["defense"] for t in teams.values()]) if teams else FALLBACK_STRENGTH}
    home = teams.get(home_team, avg)
    away = teams.get(away_team, avg)
    lambda_home = np.exp(home["attack"] + away["defense"] + strengths["home_advantage"])
    lambda_away = np.exp(away["attack"] + home["defense"])
    return float(lambda_home), float(lambda_away)


def score_matrix(lambda_home: float, lambda_away: float, rho: float = 0.0,
                 max_goals: int = DEFAULT_MAX_GOALS) -> np.ndarray:
    """P(home scores i, away scores j) for i, j in [0, max_goals], tau-
    corrected and renormalized to sum to exactly 1 -- the tau correction
    redistributes probability among the four low-score cells and can
    leave the raw grid summing to fractionally more or less than 1."""
    i = np.arange(max_goals + 1)
    home_pmf = np.exp(_poisson_log_pmf(i, lambda_home))
    away_pmf = np.exp(_poisson_log_pmf(i, lambda_away))
    grid = np.outer(home_pmf, away_pmf)

    for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        grid[x, y] *= _tau(x, y, lambda_home, lambda_away, rho)

    grid = np.clip(grid, 0, None)
    return grid / grid.sum()


def match_probabilities(grid: np.ndarray, over_under_line: float = 2.5) -> dict:
    """1X2, over/under, BTTS, and both clean-sheet probabilities from a
    score matrix. All the pairwise ones (home/draw/away, over/under,
    btts yes/no) sum to 1 by construction -- direct reads off the same
    grid, not independently estimated numbers that happen to agree."""
    n = grid.shape[0]
    i, j = np.indices(grid.shape)

    home_win = float(grid[i > j].sum())
    draw = float(grid[i == j].sum())
    away_win = float(grid[i < j].sum())

    total_goals = i + j
    over = float(grid[total_goals > over_under_line].sum())
    under = 1.0 - over

    btts_yes = float(grid[(i >= 1) & (j >= 1)].sum())
    btts_no = 1.0 - btts_yes

    clean_sheet_home = float(grid[j == 0].sum())    # away scored nothing
    clean_sheet_away = float(grid[i == 0].sum())    # home scored nothing

    return {
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "over": over, "under": under,
        "btts_yes": btts_yes, "btts_no": btts_no,
        "clean_sheet_home": clean_sheet_home, "clean_sheet_away": clean_sheet_away,
    }


def predict_match(strengths: dict, rho: float, home_team: str, away_team: str,
                  max_goals: int = DEFAULT_MAX_GOALS, over_under_line: float = 2.5) -> dict:
    """The end-to-end call most callers want: fixture in, every market
    out."""
    lambda_home, lambda_away = team_rates(strengths, home_team, away_team)
    grid = score_matrix(lambda_home, lambda_away, rho, max_goals)
    probs = match_probabilities(grid, over_under_line)
    probs["lambda_home"] = lambda_home
    probs["lambda_away"] = lambda_away
    return probs


# ---------------------------------------------------------------------------
# MARKET BLENDING (explicit, separate step -- see module docstring)
# ---------------------------------------------------------------------------

def blend_with_market(model_probs: dict, market_probs: dict, market_weight: float = 0.5) -> dict:
    """Linear blend of two valid probability distributions over the SAME
    keys (e.g. {"home_win", "draw", "away_win"}) -- always itself a valid
    distribution, no renormalization needed. Keys present in one dict but
    not the other (e.g. this model's btts_yes when the market has no
    BTTS odds collected) pass through unblended, at the model's own
    value -- there's nothing to blend them with."""
    blended = dict(model_probs)
    for key, market_p in market_probs.items():
        if key in model_probs:
            blended[key] = market_weight * market_p + (1 - market_weight) * model_probs[key]
    return blended


# ---------------------------------------------------------------------------
# MISSING-PLAYER ADJUSTMENT (prediction-time only -- see docstrings)
# ---------------------------------------------------------------------------

PLAYERS_ROOT = Path("data/xg/understat_players")


def _load_player_seasons(leagues=None) -> pd.DataFrame:
    """Understat's season-total-per-player files (collectors/understat.py's
    normalize_players output) -- season aggregates, not match-by-match
    (see that module's docstring for why)."""
    empty = pd.DataFrame(columns=["league", "season", "player_name", "team", "team_id",
                                  "minutes", "games", "goals", "xg", "npg", "npxg",
                                  "assists", "xa", "xg_chain", "xg_buildup"])
    if not PLAYERS_ROOT.exists():
        return empty
    league_dirs = ([PLAYERS_ROOT / lg for lg in leagues] if leagues
                    else sorted(d for d in PLAYERS_ROOT.iterdir() if d.is_dir()))
    frames = [pd.read_csv(f, dtype={"season": str})
              for d in league_dirs if d.exists() for f in d.glob("*.csv")]
    return pd.concat(frames, ignore_index=True) if frames else empty


def player_xg_shares(players: pd.DataFrame, league: str, season: str, team: str,
                     metric: str = "npxg") -> pd.DataFrame:
    """One team's players for one season, ranked by their share of the
    team's total (default: non-penalty) xG -- the basis for docking
    that team's attack rating when a specific player is missing for one
    fixture. npxG by default rather than raw xG for the same reason
    it's preferred as a fit target: penalties are a high-variance,
    low-skill component that inflates one player's apparent share for
    reasons that have nothing to do with open-play creation."""
    team_players = players[(players["league"] == league) & (players["season"] == season)
                           & (players["team"] == team)].copy()
    total = team_players[metric].sum()
    team_players["xg_share"] = (team_players[metric] / total) if total else 0.0
    return team_players[["player_name", metric, "xg_share"]].sort_values("xg_share", ascending=False)


def adjust_attack_for_missing_players(strengths: dict, team: str, missing_shares: list) -> dict:
    """
    A COPY of `strengths` with `team`'s attack rating reduced for ONE
    upcoming fixture a specific player (or players) will miss --
    `missing_shares` is that player's share of the team's season xG,
    from player_xg_shares(). This is deliberately a prediction-time
    adjustment, not something folded into fit_team_strengths: the
    fitted rating reflects the team's performance across a season with
    that player available most of the time, and this correction only
    applies to the one match they're actually out for -- it must never
    leak into the historical rating itself.

    Because lambda = exp(attack + ...), attack enters the goal rate
    multiplicatively -- docking it by log(1 - total_share) scales the
    team's expected goals down by roughly that fraction for this match
    (a player responsible for 30% of the team's xG being out cuts
    expected goals by about 30%, not the whole rating by 30 raw points).
    Capped at 95% combined share so a missing player can never predict
    a team incapable of scoring at all -- ten outfield players remain.
    """
    total_share = min(sum(missing_shares), 0.95)
    adjusted = {
        "teams": {t: dict(v) for t, v in strengths["teams"].items()},
        "home_advantage": strengths["home_advantage"],
        "as_of": strengths.get("as_of"),
    }
    if team in adjusted["teams"] and total_share > 0:
        adjusted["teams"][team]["attack"] += np.log(1 - total_share)
    return adjusted
