"""
Ledger -- Scoreboard
======================
The flagship page (BRIEFS.md Phase 6): grades the Dixon-Coles match
model, honestly, against simple baselines and the closing line, all on
the same walk-forward backtest (validation/scoreboard.py).

PRESENTATION ORDER IS MANDATORY, per BRIEFS.md: calibration first, then
the model beating simple baselines, THEN the market beating the model.
That ordering is what makes a loss to the closing line read as
"measured honestly against a hard benchmark" instead of "built a bad
model" -- reversing it, or leading with the market comparison, would
make the exact same true numbers look like failure instead of rigor.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import export, ledger_data as ld, theme
from validation import scoreboard as sb

st.title("Scoreboard")
st.caption("The match model, graded honestly against simple baselines and the closing line -- "
          "on the same walk-forward backtest, every source scored the same way.")

SOURCE_LABELS = {
    "home_advantage_baseline": "Home advantage only",
    "elo_baseline": "Elo",
    "dixon_coles": "Dixon-Coles (goals)",
    "dixon_coles_blended": "Dixon-Coles (goals) + market",
    "dixon_coles_xg": "Dixon-Coles (xG)",
    "dixon_coles_xg_blended": "Dixon-Coles (xG) + market",
    "dixon_coles_npxg": "Dixon-Coles (npxG)",
    "dixon_coles_npxg_blended": "Dixon-Coles (npxG) + market",
    "closing_line": "Closing line (market)",
}
BASELINE_SOURCES = ["home_advantage_baseline", "elo_baseline"]
MODEL_SOURCES = ["dixon_coles", "dixon_coles_xg", "dixon_coles_npxg"]
MARKET_SOURCES = ["closing_line", "dixon_coles_blended", "dixon_coles_xg_blended", "dixon_coles_npxg_blended"]

matches = ld.load_matches()
if matches.empty:
    st.info("No match data yet. Run `python -m collectors.football_data`.")
    st.stop()

leagues = sorted(matches["league"].unique(), key=ld.league_label)
default_idx = leagues.index("E0") if "E0" in leagues else 0
league = st.selectbox("League", leagues, index=default_idx, format_func=ld.league_label)

scored = ld.load_backtest(league)
if scored.empty:
    st.info("Not enough seasons of history for this league to backtest yet "
            f"(need more than {sb.DEFAULT_MIN_TRAIN_SEASONS} seasons on record).")
    st.stop()

summary = sb.summarize(scored).set_index("source")
available = [s for s in SOURCE_LABELS if s in summary.index and summary.loc[s, "n"] > 0]

st.caption(f"Backtest: {int(scored['date'].nunique())} matchdays, "
          f"{scored['date'].min()} to {scored['date'].max()}, walk-forward by season "
          f"(each season predicted using only strictly earlier data).")

# ---------------------------------------------------------------------------
# 1. CALIBRATION -- first, on its own, independent of beating anything
# ---------------------------------------------------------------------------

st.header("1. Calibration")
st.caption("When the model says 70%, does that outcome happen about 70% of the time? "
          "This is checked before any comparison to a baseline or the market -- a model "
          "has to pass this bar on its own terms first.")

calib_source = st.selectbox("Source", [s for s in MODEL_SOURCES if s in available],
                            format_func=lambda s: SOURCE_LABELS[s])
calib_outcome = st.radio("Outcome", ["home_win", "draw", "away_win"], horizontal=True,
                         format_func=lambda o: {"home_win": "Home win", "draw": "Draw", "away_win": "Away win"}[o])

source_scored = scored[scored["source"] == calib_source].dropna(subset=["home_win", "draw", "away_win"])
curve = sb.calibration_curve(source_scored, outcome=calib_outcome, n_bins=8)

if curve.empty:
    st.info("Not enough matches in this slice to bin meaningfully.")
else:
    diagonal = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    points = alt.Chart(curve).mark_circle(size=90, color=theme.DIVERGING_BLUE_RED[0]).encode(
        x=alt.X("predicted:Q", title="Predicted probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("actual:Q", title="Actual frequency", scale=alt.Scale(domain=[0, 1])),
        size=alt.Size("n:Q", title="Matches", legend=None),
        tooltip=[alt.Tooltip("predicted:Q", format=".2f"), alt.Tooltip("actual:Q", format=".2f"), "n"],
    )
    line_ref = alt.Chart(diagonal).mark_line(strokeDash=[5, 5], color="gray").encode(x="x:Q", y="y:Q")
    st.altair_chart((points + line_ref).properties(height=380).interactive(), width="stretch")
    png = export.scatter_png(curve, "predicted", "actual", size_col="n", diagonal=True,
                             title=f"Calibration -- {SOURCE_LABELS[calib_source]}",
                             subtitle=f"{ld.league_label(league)}, {calib_outcome.replace('_', ' ')}",
                             source="Backtest on football-data.co.uk", x_label="Predicted probability",
                             y_label="Actual frequency")
    st.download_button("Download PNG", png, "scoreboard_calibration.png", "image/png", key="png_calibration")

# ---------------------------------------------------------------------------
# 2. MODEL VS SIMPLE BASELINES
# ---------------------------------------------------------------------------

st.header("2. Beating simple baselines")
st.caption("Home-advantage-only and Elo use none or almost none of what Dixon-Coles uses. "
          "If the model can't clear these, nothing past this point matters.")

baseline_rows = summary.loc[[s for s in MODEL_SOURCES + BASELINE_SOURCES if s in available]].reset_index()
baseline_rows["label"] = baseline_rows["source"].map(SOURCE_LABELS)
baseline_rows = baseline_rows.sort_values("log_loss")

st.altair_chart(
    alt.Chart(baseline_rows).mark_bar().encode(
        x=alt.X("log_loss:Q", title="Log loss (lower is better)"),
        y=alt.Y("label:N", sort=alt.SortField("log_loss", order="descending"), title=None),
        color=alt.Color("label:N", legend=None,
                        scale=alt.Scale(domain=baseline_rows["label"].tolist(),
                                        range=theme.CATEGORICAL_COLORS[:len(baseline_rows)])),
        tooltip=["label", alt.Tooltip("log_loss:Q", format=".4f"), alt.Tooltip("rps:Q", format=".4f"), "n"],
    ).properties(height=max(120, 40 * len(baseline_rows))),
    width="stretch")
png = export.bar_png(baseline_rows["label"].tolist(), baseline_rows["log_loss"].tolist(),
                     title="Model vs. simple baselines", source="Backtest on football-data.co.uk",
                     x_label="Log loss (lower is better)")
st.download_button("Download PNG", png, "scoreboard_baselines.png", "image/png", key="png_baselines")

st.dataframe(
    baseline_rows[["label", "log_loss", "rps", "n"]].rename(
        columns={"label": "Source", "log_loss": "Log loss", "rps": "RPS", "n": "Matches"}),
    width="stretch", hide_index=True,
    column_config={"Log loss": st.column_config.NumberColumn(format="%.4f"),
                   "RPS": st.column_config.NumberColumn(format="%.4f")})

model_best = baseline_rows[baseline_rows["source"].isin(MODEL_SOURCES)].iloc[0] if \
    baseline_rows["source"].isin(MODEL_SOURCES).any() else None
baseline_best = baseline_rows[baseline_rows["source"].isin(BASELINE_SOURCES)].iloc[0] if \
    baseline_rows["source"].isin(BASELINE_SOURCES).any() else None
if model_best is not None and baseline_best is not None:
    if model_best["log_loss"] < baseline_best["log_loss"]:
        st.success(f"**{model_best['label']}** beats the best simple baseline "
                  f"(**{baseline_best['label']}**): {model_best['log_loss']:.4f} vs "
                  f"{baseline_best['log_loss']:.4f} log loss.")
    else:
        st.warning(f"**{model_best['label']}** does NOT beat the best simple baseline "
                  f"(**{baseline_best['label']}**) on this league/backtest -- "
                  f"{model_best['log_loss']:.4f} vs {baseline_best['log_loss']:.4f} log loss. "
                  "Reported as-is, not softened.")

# ---------------------------------------------------------------------------
# 3. THE MARKET -- shown last, deliberately
# ---------------------------------------------------------------------------

st.header("3. Against the closing line")
st.caption("Shown last, on purpose (see the module docstring): the closing line is one of the "
          "hardest benchmarks in forecasting, and a loss here is the expected result, not a "
          "verdict on the model -- but only once the first two sections have already shown it "
          "isn't a bad model.")

market_rows = summary.loc[[s for s in MODEL_SOURCES + MARKET_SOURCES if s in available]].reset_index()
market_rows["label"] = market_rows["source"].map(SOURCE_LABELS)
market_rows = market_rows.sort_values("log_loss")

if market_rows.empty or "closing_line" not in market_rows["source"].values or market_rows.loc[
        market_rows["source"] == "closing_line", "n"].iloc[0] == 0:
    st.info("No closing-odds coverage for this league/era in the backtest window -- "
           "nothing to compare against the market here.")
else:
    st.altair_chart(
        alt.Chart(market_rows).mark_bar().encode(
            x=alt.X("log_loss:Q", title="Log loss (lower is better)"),
            y=alt.Y("label:N", sort=alt.SortField("log_loss", order="descending"), title=None),
            color=alt.Color("label:N", legend=None,
                            scale=alt.Scale(domain=market_rows["label"].tolist(),
                                            range=theme.CATEGORICAL_COLORS[:len(market_rows)])),
            tooltip=["label", alt.Tooltip("log_loss:Q", format=".4f"), alt.Tooltip("rps:Q", format=".4f"), "n"],
        ).properties(height=max(120, 40 * len(market_rows))),
        width="stretch")
    png = export.bar_png(market_rows["label"].tolist(), market_rows["log_loss"].tolist(),
                         title="Model vs. the closing line", source="Backtest on football-data.co.uk",
                         x_label="Log loss (lower is better)")
    st.download_button("Download PNG", png, "scoreboard_market.png", "image/png", key="png_market")

    st.dataframe(
        market_rows[["label", "log_loss", "rps", "n"]].rename(
            columns={"label": "Source", "log_loss": "Log loss", "rps": "RPS", "n": "Matches"}),
        width="stretch", hide_index=True,
        column_config={"Log loss": st.column_config.NumberColumn(format="%.4f"),
                       "RPS": st.column_config.NumberColumn(format="%.4f")})

    market_row = market_rows[market_rows["source"] == "closing_line"].iloc[0]
    model_row = market_rows[market_rows["source"].isin(MODEL_SOURCES)].sort_values("log_loss").iloc[0] \
        if market_rows["source"].isin(MODEL_SOURCES).any() else None
    if model_row is not None:
        if model_row["log_loss"] < market_row["log_loss"]:
            st.success(f"**{model_row['label']}** beats the closing line on this league/backtest: "
                      f"{model_row['log_loss']:.4f} vs {market_row['log_loss']:.4f} log loss.")
        else:
            st.warning(f"The closing line beats **{model_row['label']}** on this league/backtest: "
                      f"{market_row['log_loss']:.4f} vs {model_row['log_loss']:.4f} log loss. "
                      "Expected against one of the most efficient markets there is -- reported "
                      "plainly, not softened.")

with st.expander("All sources, every metric"):
    full = summary.reset_index()
    full["label"] = full["source"].map(SOURCE_LABELS).fillna(full["source"])
    st.dataframe(
        full[["label", "log_loss", "rps", "n"]].rename(
            columns={"label": "Source", "log_loss": "Log loss", "rps": "RPS", "n": "Matches"}),
        width="stretch", hide_index=True,
        column_config={"Log loss": st.column_config.NumberColumn(format="%.4f"),
                       "RPS": st.column_config.NumberColumn(format="%.4f")})
