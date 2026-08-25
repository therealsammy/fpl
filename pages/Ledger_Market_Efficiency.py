"""
Ledger -- Market Efficiency
=============================
Calibration of the closing line (Shin-devigged) against what actually
happened: bucket every 1X2 outcome's implied probability into deciles,
and check whether outcomes priced around, say, 70% actually occurred
about 70% of the time. A well-functioning market should track the
diagonal closely -- that's the whole test, not a modeling exercise.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import ledger_data as ld

st.title("Market efficiency")
st.caption("Calibration of closing-line implied probabilities against actual outcome frequency.")

DIVERGING_BLUE_RED = ["#104281", "#86b6ef", "#f0efec", "#f0a68a", "#d03b3b"]

matches = ld.load_matches()
if matches.empty:
    st.info("No match data yet. Run `python -m collectors.football_data`.")
    st.stop()

priced = matches.dropna(subset=["prob_home_shin", "prob_draw_shin", "prob_away_shin"]).copy()
if priced.empty:
    st.info("No matches with closing odds on record yet.")
    st.stop()

# Long format: one row per (match, outcome), so all three 1X2 outcomes
# across every match contribute to the same calibration curve.
long = pd.concat([
    pd.DataFrame({"league": priced["league"], "date": priced["date"],
                 "predicted": priced["prob_home_shin"], "actual": (priced["result"] == "H").astype(int)}),
    pd.DataFrame({"league": priced["league"], "date": priced["date"],
                 "predicted": priced["prob_draw_shin"], "actual": (priced["result"] == "D").astype(int)}),
    pd.DataFrame({"league": priced["league"], "date": priced["date"],
                 "predicted": priced["prob_away_shin"], "actual": (priced["result"] == "A").astype(int)}),
], ignore_index=True)
long["season_start"] = pd.to_datetime(long["date"]).dt.year
long["era"] = (long["season_start"] // 5 * 5).astype(str) + "s"

leagues = sorted(long["league"].unique())
chosen_leagues = st.multiselect("League", leagues, format_func=ld.league_label)
view = long[long["league"].isin(chosen_leagues)] if chosen_leagues else long

eras = sorted(view["era"].unique())
chosen_eras = st.multiselect("Era (5-year bucket)", eras)
if chosen_eras:
    view = view[view["era"].isin(chosen_eras)]

n_bins = st.slider("Bins", 5, 20, 10)
view = view.copy()
view["bin"] = pd.cut(view["predicted"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)

calib = view.groupby("bin", observed=True).agg(
    predicted=("predicted", "mean"), actual=("actual", "mean"), n=("actual", "size")).reset_index()
calib = calib[calib["n"] >= 20]   # a bin with a handful of matches is noise, not signal

if calib.empty:
    st.info("Not enough priced matches in this selection to bin meaningfully. Widen the filters.")
    st.stop()

brier = ((view["predicted"] - view["actual"]) ** 2).mean()
st.metric("Brier score (lower is better; 0 = perfect, 0.33 = always guessing 1/3)", f"{brier:.4f}")

diagonal = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
points = alt.Chart(calib).mark_circle(size=90, color=DIVERGING_BLUE_RED[0]).encode(
    x=alt.X("predicted:Q", title="Implied probability (closing line)", scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("actual:Q", title="Actual frequency", scale=alt.Scale(domain=[0, 1])),
    size=alt.Size("n:Q", title="Matches", legend=None),
    tooltip=[alt.Tooltip("predicted:Q", format=".2f"), alt.Tooltip("actual:Q", format=".2f"), "n"],
)
line_ref = alt.Chart(diagonal).mark_line(strokeDash=[5, 5], color="gray").encode(x="x:Q", y="y:Q")
st.altair_chart((points + line_ref).properties(height=440).interactive(), width="stretch")

st.caption("Dashed line is perfect calibration. Points above it mean the market underpriced that "
          "outcome band (it happened more than the odds implied); below means it overpriced it.")

with st.expander("Bin detail"):
    show = calib.copy()
    show["bin"] = show["bin"].astype(str)   # Interval objects don't survive Arrow serialization
    show = show.rename(columns={"bin": "Probability bin", "predicted": "Mean implied",
                                "actual": "Actual frequency", "n": "Matches"})
    st.dataframe(show, width="stretch", hide_index=True,
                column_config={"Mean implied": st.column_config.NumberColumn(format="%.3f"),
                               "Actual frequency": st.column_config.NumberColumn(format="%.3f")})
