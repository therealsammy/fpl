"""
Ledger -- Upsets
=================
Results ranked by the implied probability of the team that actually
won -- lowest probability first. Uses the Shin-devigged closing odds
already collected in Phase 3 (collectors/football_data.py), since Shin
strips the bookmaker's margin more carefully than a flat proportional
split. Matches with no odds on record (older seasons, or leagues/eras
football-data.co.uk didn't quote) are excluded -- there's no implied
probability to rank them by.
"""

import altair as alt
import streamlit as st

from core import export, ledger_data as ld, theme

st.title("Upsets")
st.caption("Results ranked by how unlikely the closing odds said the winner was.")

STATUS_COLORS = theme.STATUS_COLORS
SOURCE = "football-data.co.uk, Shin-devigged closing odds"

matches = ld.load_matches()
if matches.empty:
    st.info("No match data yet. Run `python -m collectors.football_data`.")
    st.stop()

priced = matches.dropna(subset=["prob_home_shin", "prob_draw_shin", "prob_away_shin"]).copy()
if priced.empty:
    st.info("No matches with closing odds on record yet.")
    st.stop()

priced["winner_probability"] = priced.apply(
    lambda r: {"H": r["prob_home_shin"], "D": r["prob_draw_shin"], "A": r["prob_away_shin"]}[r["result"]],
    axis=1)
priced["winner"] = priced.apply(
    lambda r: r["home_team"] if r["result"] == "H"
    else (r["away_team"] if r["result"] == "A" else "Draw"), axis=1)
priced["score"] = priced["home_goals"].astype(int).astype(str) + "-" + priced["away_goals"].astype(int).astype(str)

leagues = sorted(priced["league"].unique())
chosen_leagues = st.multiselect("League", leagues, format_func=ld.league_label)
view = priced[priced["league"].isin(chosen_leagues)] if chosen_leagues else priced

years = sorted(view["date"].str[:4].unique())
if len(years) > 1:
    lo, hi = st.select_slider("Years", years, value=(years[0], years[-1]))
    view = view[view["date"].str[:4].between(lo, hi)]

n = st.slider("Show top N upsets", 10, 100, 25)
biggest = view.nsmallest(n, "winner_probability").copy()
biggest["Winner's implied probability %"] = (biggest["winner_probability"] * 100).round(1)
biggest["League"] = biggest["league"].map(ld.league_label)

st.altair_chart(
    alt.Chart(biggest.head(20)).mark_bar(color=STATUS_COLORS["critical"]).encode(
        x=alt.X("Winner's implied probability %:Q"),
        y=alt.Y("date:N", sort=alt.SortField("Winner's implied probability %", order="ascending"),
               title=None),
        tooltip=["date", "League", "home_team", "away_team", "score", "winner",
                "Winner's implied probability %"],
    ).properties(height=max(160, 24 * min(20, len(biggest)))),
    width="stretch")
top20 = biggest.head(20)
match_labels = (top20["home_team"] + " " + top20["score"] + " " + top20["away_team"]
               + "  (" + top20["date"] + ")")
png = export.bar_png(match_labels.tolist(), top20["Winner's implied probability %"].tolist(),
                     title="Biggest upsets", source=SOURCE,
                     x_label="Winner's implied probability %", color=STATUS_COLORS["critical"])
st.download_button("Download PNG", png, "upsets.png", "image/png", key="png_upsets")

table = biggest[["date", "League", "home_team", "away_team", "score", "winner",
                 "Winner's implied probability %"]].rename(columns={
    "date": "Date", "home_team": "Home", "away_team": "Away", "score": "Score", "winner": "Winner"})
st.dataframe(table, width="stretch", hide_index=True,
            column_config={"Winner's implied probability %": st.column_config.NumberColumn(format="%.1f")})
