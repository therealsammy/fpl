"""
Ledger -- Match Predictions
=============================
A prediction is a plain Dixon-Coles rating (models/match.py) with one
extra ingredient: each team's recent form in pressing intensity, buildup
quality, and underlying chance quality (Understat's PPDA, deep
completions, and non-penalty xG) -- signals a bare goals-based model
has no way to use at all. Technically a Poisson GLM; nothing on this
page needs that name to make sense.

Two things this page keeps visually separate on purpose: what the
model predicts for real, current fixtures (section 1 -- the reason
this page exists) versus whether the model has actually earned any
trust (section 3, a walk-forward backtest never fit on the seasons it's
scored against). A page that only ever shows predictions with no
honest report card reads as more confident than it should.
"""

import altair as alt
import pandas as pd
import streamlit as st

from core import export, ledger_data as ld, theme
from models import match
from validation import scoreboard as sb

st.title("Match predictions")
st.caption("Predictions built on team strength, recent form, and playing style -- "
          "not just who's historically been good.")

COVARIATE_LABELS = {"npxg_form": "Chance quality", "ppda_form": "Pressing intensity",
                    "deep_form": "Buildup play"}
OUTCOME_COLORS = {"Home win": theme.CATEGORICAL_COLORS[0], "Draw": "#c9c9c9",
                  "Away win": theme.CATEGORICAL_COLORS[1]}
GLM_STATS = [c.replace("_form", "") for c in sb.GLM_COVARIATES]
SOURCE_LABELS = {"dixon_coles_glm": "This model (style-adjusted)",
                 "dixon_coles_glm_blended": "This model + betting market blended",
                 "dixon_coles": "Team strength only",
                 "dixon_coles_xg": "Trained on xG instead of goals", "dixon_coles_npxg": "Trained on non-penalty xG",
                 "elo_baseline": "Simple Elo ranking", "home_advantage_baseline": "Just assume home advantage",
                 "closing_line": "Betting market's own price"}

matches = ld.load_matches_with_xg()
if matches.empty:
    st.info("No match data yet. Run `python -m collectors.football_data` and "
            "`python -m collectors.understat`.")
    st.stop()

leagues = sorted(matches["league"].unique(), key=ld.league_label)
default_idx = leagues.index("E0") if "E0" in leagues else 0
league = st.selectbox("League", leagues, index=default_idx, format_func=ld.league_label)

glm = ld.load_glm(league)
strengths, rho, base_strengths = glm["strengths"], glm["rho"], glm["base_strengths"]

if not strengths["teams"]:
    st.info("Not enough data for this league yet to build style-adjusted predictions "
           "(this currently works best for the Premier League). Try English Premier League.")
    st.stop()

league_matches = ld.load_matches_with_form(league)


def _form_for(team: str) -> dict:
    """Current form for `team`, or an empty dict (the GLM then simply
    applies no style adjustment for them) if there's no history at all
    -- a newly-promoted side's very first match, most likely."""
    form = {c: match.current_form(league_matches, team, stat) for c, stat in zip(sb.GLM_COVARIATES, GLM_STATS)}
    return {c: v for c, v in form.items() if v is not None}


def _predict(home: str, away: str) -> dict:
    home_form, away_form = _form_for(home), _form_for(away)
    lambda_home, lambda_away = match.team_rates_glm(strengths, home, away, home_form, away_form)
    return match.match_probabilities(match.score_matrix(lambda_home, lambda_away, rho))


def _predict_plain(home: str, away: str) -> dict:
    lambda_home, lambda_away = match.team_rates(base_strengths, home, away)
    return match.match_probabilities(match.score_matrix(lambda_home, lambda_away, rho))


SOURCE_ORDER = {"Style-adjusted": 0, "Team strength only": 1, "Betting market": 2}


def _predict_all_sources(home: str, away: str, odds: pd.DataFrame) -> list:
    """One row per available source for this fixture -- betting market
    is simply omitted (not shown as a guess) when fixture_projections.csv
    has no live odds for it, e.g. fpl_odds.py hasn't been run recently
    or this league isn't the Premier League."""
    rows = [
        {"Source": "Style-adjusted", **_scaled(_predict(home, away))},
        {"Source": "Team strength only", **_scaled(_predict_plain(home, away))},
    ]
    market_row = odds[(odds["home_team"] == home) & (odds["away_team"] == away)]
    if not market_row.empty:
        r = market_row.iloc[0]
        rows.append({"Source": "Betting market",
                     "Home win": round(r["home_win"] * 100, 1), "Draw": round(r["draw"] * 100, 1),
                     "Away win": round(r["away_win"] * 100, 1)})
    return rows


def _scaled(probs: dict) -> dict:
    return {"Home win": round(probs["home_win"] * 100, 1), "Draw": round(probs["draw"] * 100, 1),
           "Away win": round(probs["away_win"] * 100, 1)}


OUTCOME_ORDER = {"Home win": 0, "Draw": 1, "Away win": 2}


def _stacked_bar(rows: pd.DataFrame, y_field: str, y_sort: list, height_per_row: int = 46):
    long = rows.melt(id_vars=[y_field], value_vars=["Home win", "Draw", "Away win"],
                     var_name="Outcome", value_name="Percent")
    long["Outcome order"] = long["Outcome"].map(OUTCOME_ORDER)
    return alt.Chart(long).mark_bar(height=20).encode(
        x=alt.X("Percent:Q", stack="zero", title=None, axis=alt.Axis(format=".0f")),
        y=alt.Y(f"{y_field}:N", sort=y_sort, title=None),
        color=alt.Color("Outcome:N", title=None,
                        scale=alt.Scale(domain=list(OUTCOME_COLORS), range=list(OUTCOME_COLORS.values()))),
        order=alt.Order("Outcome order:Q"),
        tooltip=[y_field, "Outcome", alt.Tooltip("Percent:Q", format=".0f")],
    ).properties(height=max(80, height_per_row * rows[y_field].nunique()))


def _fixture_card_chart(rows_df: pd.DataFrame):
    """ONE fixture's own small chart -- deliberately not a single big
    faceted chart across all fixtures. Altair's row-facet headers
    overlap into illegible stacked text once there are 10 of them with
    long team names (verified live -- that's what this replaced); a
    separate small chart per match, laid out via Streamlit's own grid,
    renders every label cleanly instead.

    Percentages are drawn ON the bars (not just in the tooltip) since
    the whole point of this page is being able to just look at it.
    Vega-Lite auto-stacks the BAR mark from a plain quantitative
    encoding, but a text mark sharing that same encoding is NOT
    auto-positioned at its segment's stacked midpoint -- it would place
    every label at x=(that segment's own value), which is only correct
    for the first (bottom) segment. The midpoint is computed explicitly
    here instead and given its own x channel for the text layer."""
    long = rows_df.melt(id_vars=["Source"], value_vars=["Home win", "Draw", "Away win"],
                        var_name="Outcome", value_name="Percent")
    long["Outcome order"] = long["Outcome"].map(OUTCOME_ORDER)
    long["Source order"] = long["Source"].map(SOURCE_ORDER)
    long = long.sort_values(["Source", "Outcome order"])
    long["cum_end"] = long.groupby("Source")["Percent"].cumsum()
    long["mid"] = long["cum_end"] - long["Percent"] / 2
    long["Label"] = long["Percent"].round().astype(int).astype(str) + "%"

    color_scale = alt.Scale(domain=list(OUTCOME_COLORS), range=list(OUTCOME_COLORS.values()))
    y_enc = alt.Y("Source:N", sort=alt.SortField("Source order"), title=None,
                  axis=alt.Axis(labelLimit=130, labelFontSize=11))

    bars = alt.Chart(long).mark_bar(height=18).encode(
        x=alt.X("Percent:Q", stack="zero", title=None, axis=None, scale=alt.Scale(domain=[0, 100])),
        y=y_enc,
        color=alt.Color("Outcome:N", title=None, legend=None, scale=color_scale),
        order=alt.Order("Outcome order:Q"),
        tooltip=["Source", "Outcome", alt.Tooltip("Percent:Q", format=".0f")],
    )
    # Only label a segment wide enough for the text to actually fit --
    # a 2% sliver with "2%" stamped on it is noise, not information.
    labels = alt.Chart(long).mark_text(color="white", fontSize=10, fontWeight="bold").encode(
        x=alt.X("mid:Q", scale=alt.Scale(domain=[0, 100])),
        y=y_enc,
        text=alt.Text("Label:N"),
        opacity=alt.condition("datum.Percent > 8", alt.value(1), alt.value(0)),
    )
    return (bars + labels).properties(height=26 * rows_df["Source"].nunique() + 10)


# ---------------------------------------------------------------------------
# 1. THIS WEEK'S FIXTURES -- the whole point of this page
# ---------------------------------------------------------------------------

st.header("This week's fixtures")
st.caption("Three ways of calling the same match: this model with style form, team strength "
          "alone, and the betting market's own price where available.")
fixtures = match.upcoming_fixtures(league)
odds = ld.load_live_odds_predictions()

if fixtures.empty:
    st.info("No scheduled fixtures found for this league right now.")
else:
    fixture_predictions = []   # one entry per fixture: {"home", "away", "date", "rows": DataFrame}
    table_rows = []
    for _, fx in fixtures.iterrows():
        home, away = fx["home_team"], fx["away_team"]
        date_str = fx["date"].strftime("%a %d %b")
        source_rows = _predict_all_sources(home, away, odds)
        fixture_predictions.append({"home": home, "away": away, "date": date_str,
                                    "rows": pd.DataFrame(source_rows)})
        for source_row in source_rows:
            table_rows.append({"Match": f"{home} vs {away}", "Date": date_str, **source_row})

    table_df = pd.DataFrame(table_rows)
    has_market = "Betting market" in table_df["Source"].values
    st.caption(f"{len(fixtures)} match(es), {fixtures['date'].min().strftime('%a %d %b')} to "
              f"{fixtures['date'].max().strftime('%a %d %b')}."
              + ("" if has_market else " No live betting odds on record for this window."))
    st.markdown(
        f"🟦 Home win &nbsp;&nbsp; ⬜ Draw &nbsp;&nbsp; 🟧 Away win",
        help="Every bar below uses this same color key.")

    # Two fixtures per row, each its own small chart -- a single chart
    # faceted across all 10 fixtures put its row labels (long team
    # names, stacked vertically) on top of each other, unreadable.
    for i in range(0, len(fixture_predictions), 2):
        pair = fixture_predictions[i:i + 2]
        cols = st.columns(len(pair))
        for col, fx_pred in zip(cols, pair):
            with col:
                st.markdown(f"**{fx_pred['home']} vs {fx_pred['away']}**  \n{fx_pred['date']}")
                st.altair_chart(_fixture_card_chart(fx_pred["rows"]), width="stretch")

    with st.expander("See it as a table"):
        st.dataframe(table_df, width="stretch", hide_index=True,
                    column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                  for c in ["Home win", "Draw", "Away win"]})

    png = export.bar_png(
        [f"{r['Match']} -- {r['Source']}" for r in table_rows], [r["Home win"] for r in table_rows],
        title=f"This week's fixtures -- {ld.league_label(league)}",
        subtitle="Home win probability shown per source; see the app for the full breakdown",
        source="football-data.co.uk + Understat + The Odds API", x_label="Home win %")
    st.download_button("Download PNG", png, "fixtures.png", "image/png", key="png_fixtures")

st.divider()

# ---------------------------------------------------------------------------
# 2. TRY ANY MATCHUP
# ---------------------------------------------------------------------------

st.header("Try any matchup")
teams = sorted(strengths["teams"].keys())
col_a, col_b = st.columns(2)
home_team = col_a.selectbox("Home team", teams, index=0)
away_team = col_b.selectbox("Away team", teams, index=min(1, len(teams) - 1))

probs = _predict(home_team, away_team)
matchup_row = pd.DataFrame([{
    "Match": f"{home_team} vs {away_team}",
    "Home win": round(probs["home_win"] * 100, 1),
    "Draw": round(probs["draw"] * 100, 1),
    "Away win": round(probs["away_win"] * 100, 1),
}])
st.altair_chart(_stacked_bar(matchup_row, "Match", matchup_row["Match"].tolist(), height_per_row=60),
                width="stretch")
m1, m2, m3 = st.columns(3)
m1.metric(f"{home_team} win", f"{matchup_row['Home win'].iloc[0]:.0f}%")
m2.metric("Draw", f"{matchup_row['Draw'].iloc[0]:.0f}%")
m3.metric(f"{away_team} win", f"{matchup_row['Away win'].iloc[0]:.0f}%")

st.divider()

# ---------------------------------------------------------------------------
# 3. HOW TRUSTWORTHY IS THIS?
# ---------------------------------------------------------------------------

st.header("How trustworthy is this?")
st.caption("Every method below is tested the same way: predict a season using only data from "
          "before it, then check how well those predictions held up against what actually "
          "happened. Shorter bars are better -- the method's predictions matched reality more "
          "closely, on average, across thousands of past matches.")

backtest = ld.load_backtest(league)
if "dixon_coles_glm" not in backtest["source"].unique():
    st.info("Not enough historical coverage for this league to check trustworthiness yet.")
else:
    summary = sb.summarize(backtest).set_index("source")
    rows = [s for s in ["dixon_coles_glm", "dixon_coles_glm_blended", "dixon_coles", "dixon_coles_xg",
                        "dixon_coles_npxg", "elo_baseline", "home_advantage_baseline", "closing_line"]
           if s in summary.index]
    show = summary.loc[rows].reset_index()
    show["Source"] = show["source"].map(SOURCE_LABELS)
    show = show.sort_values("log_loss")

    st.altair_chart(
        alt.Chart(show).mark_bar().encode(
            x=alt.X("log_loss:Q", title="Prediction error (lower is better)"),
            y=alt.Y("Source:N", sort=alt.SortField("log_loss", order="descending"), title=None),
            color=alt.condition("datum.source == 'dixon_coles_glm' || datum.source == 'dixon_coles_glm_blended'",
                                alt.value(theme.CATEGORICAL_COLORS[0]), alt.value(theme.SEQUENTIAL_BLUE[1])),
            tooltip=["Source", alt.Tooltip("log_loss:Q", format=".4f", title="Log loss"), "n"],
        ).properties(height=max(160, 40 * len(show))),
        width="stretch")

    by_source = show.set_index("source")["log_loss"]
    glm_rank = list(show["source"]).index("dixon_coles_glm") + 1
    beats_plain = bool(by_source["dixon_coles_glm"] < by_source["dixon_coles"]) \
        if "dixon_coles" in by_source else None
    if beats_plain is True:
        st.success(f"This model ranks #{glm_rank} of {len(show)} here -- style-adjusting genuinely "
                  "helps on this league's history.")
    elif beats_plain is False:
        st.warning(f"This model ranks #{glm_rank} of {len(show)} here -- adding style form does NOT "
                  "beat plain team-strength ratings on this league's history. Shown as-is, not softened.")

    # Whether anything here actually beats the market varies by league and
    # changes as more/better data comes in (it flipped on this exact page
    # once a team-name resolution bug was fixed) -- checked live rather
    # than asserted, so this caption can never go stale and wrong.
    beats_market = "closing_line" in by_source.index and by_source.drop("closing_line").min() < by_source["closing_line"]
    if beats_market:
        best_beats_market = by_source.drop("closing_line").idxmin()
        st.caption(f"**{SOURCE_LABELS.get(best_beats_market, best_beats_market)}** edges out the betting "
                  "market's own closing price here -- worth noting, but one backtest beating an "
                  "efficient market by a small margin isn't proof of a durable edge.")
    else:
        st.caption("Every method here loses to the betting market's own closing price, which is normal "
                  "and expected -- markets aggregate a lot of information no model here has access to. "
                  "That's not a flaw in the model, it's just an honest benchmark.")

st.divider()

# ---------------------------------------------------------------------------
# 4. THIS SEASON SO FAR -- live tracking, not a backtest
# ---------------------------------------------------------------------------

st.header("This season so far")
st.caption("Different question from the backtest above: every prediction counted here was "
          "archived by models/live_predictions.py BEFORE its match kicked off, then scored once "
          "the real result was known. This is how the model is actually doing right now, not how "
          "it would have done historically -- expect this to be noisy early in a season with only "
          "a few matches scored, and to settle down as more results come in.")

live_scored = ld.load_live_scoreboard()
league_live = live_scored[live_scored["league"] == league] if not live_scored.empty else live_scored

if league_live.empty:
    st.info("No archived predictions have been matched to a played result yet for this league. "
           "This fills in automatically once this week's live predictions have been archived "
           "and their fixtures played -- see .github/workflows/live_predictions.yml.")
else:
    live_summary = sb.summarize(league_live).set_index("source")
    live_rows = [s for s in ["dixon_coles_glm", "dixon_coles", "closing_line"] if s in live_summary.index]
    show_live = live_summary.loc[live_rows].reset_index()
    show_live["Source"] = show_live["source"].map(SOURCE_LABELS)
    show_live = show_live.sort_values("log_loss")

    n_matches = int(show_live["n"].max())
    st.caption(f"Based on {n_matches} scored match(es) so far this season.")

    st.altair_chart(
        alt.Chart(show_live).mark_bar().encode(
            x=alt.X("log_loss:Q", title="Prediction error so far (lower is better)"),
            y=alt.Y("Source:N", sort=alt.SortField("log_loss", order="descending"), title=None),
            color=alt.condition("datum.source == 'dixon_coles_glm'",
                                alt.value(theme.STATUS_COLORS.get("good", theme.CATEGORICAL_COLORS[0])),
                                alt.value(theme.SEQUENTIAL_BLUE[1])),
            tooltip=["Source", alt.Tooltip("log_loss:Q", format=".4f", title="Log loss"), "n"],
        ).properties(height=max(120, 40 * len(show_live))),
        width="stretch")

    if "dixon_coles_glm" in live_summary.index and "closing_line" in live_summary.index:
        glm_live = live_summary.loc["dixon_coles_glm", "log_loss"]
        market_live = live_summary.loc["closing_line", "log_loss"]
        if glm_live < market_live:
            st.caption("This model is currently edging out the live betting market -- with this few "
                      "matches, treat that as noise rather than a proven edge either way.")
        else:
            st.caption("The live betting market is currently ahead of this model, which is the normal, "
                      "expected state -- markets price in information this model doesn't have access to.")

# ---------------------------------------------------------------------------
# ADVANCED: what the model is actually using
# ---------------------------------------------------------------------------

with st.expander("Advanced: what the model is actually using"):
    st.markdown(
        "On top of each team's long-run attack and defense rating, this model adds three "
        "signals from a team's last 5 matches -- each one measured strictly before the match "
        "being predicted, never including it:\n\n"
        "- **Chance quality** -- non-penalty expected goals (a cleaner read on attacking threat "
        "than actual goals scored, which bounce around a lot).\n"
        "- **Pressing intensity** -- PPDA, the number of passes an opponent is allowed before a "
        "defensive action. Lower means a team has been pressing more aggressively.\n"
        "- **Buildup play** -- completed passes deep in the opponent's half.\n\n"
        "The chart below shows how much each one actually moves the prediction, fit on all "
        "available history for this league. A positive bar means more of that recently predicts "
        "*more* goals for a team, beyond what their season-long rating alone would say; negative "
        "means fewer."
    )
    coef = strengths["coefficients"]
    coef_df = pd.DataFrame([{"Signal": COVARIATE_LABELS.get(c, c), "Effect": b}
                            for c, b in coef.items()]).sort_values("Effect", key=abs, ascending=False)
    st.altair_chart(
        alt.Chart(coef_df).mark_bar().encode(
            x=alt.X("Effect:Q"),
            y=alt.Y("Signal:N", sort=alt.SortField("Effect", order="descending"), title=None),
            color=alt.condition("datum.Effect > 0", alt.value(theme.STATUS_COLORS["good"]),
                                alt.value(theme.STATUS_COLORS["critical"])),
            tooltip=["Signal", alt.Tooltip("Effect:Q", format=".3f")],
        ).properties(height=140),
        width="stretch")
    png = export.bar_png(coef_df["Signal"].tolist(), coef_df["Effect"].tolist(),
                         title=f"What moves the prediction -- {ld.league_label(league)}",
                         source="football-data.co.uk + Understat", x_label="Effect on expected goals")
    st.download_button("Download PNG", png, "glm_coefficients.png", "image/png", key="png_glm_coef")
