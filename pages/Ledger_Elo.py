"""
Ledger -- Elo Ratings
======================
Standard Elo (models/elo.py), any of the six collected leagues, any era
back to 1993 where that league's history reaches. Rated strictly per
league -- see the module docstring in models/elo.py for why cross-league
ratings aren't shown here.
"""

import altair as alt
import streamlit as st

from core import ledger_data as ld

st.title("Elo ratings")
st.caption("Standard Elo from results alone. Each league is its own independent rating pool.")

CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                      "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]

matches = ld.load_matches()
if matches.empty:
    st.info("No match data yet. Run `python -m collectors.football_data`.")
    st.stop()

leagues = sorted(matches["league"].unique(), key=ld.league_label)
default_idx = leagues.index("E0") if "E0" in leagues else 0
league = st.selectbox("League", leagues, index=default_idx, format_func=ld.league_label)

history = ld.load_history()
league_history = history[history["league"] == league]
ratings = ld.load_current_ratings()
league_ratings = ratings[ratings["league"] == league].reset_index(drop=True)

tab_current, tab_trajectory = st.tabs(["Current ratings", "Rating trajectory"])

with tab_current:
    st.caption(f"As of {league_ratings['date'].max()} -- {len(league_ratings)} teams.")
    n = st.slider("Show top N", 5, min(40, len(league_ratings)), min(20, len(league_ratings)))
    top = league_ratings.head(n)

    st.altair_chart(
        alt.Chart(top).mark_bar().encode(
            x=alt.X("rating:Q", title="Elo rating", scale=alt.Scale(zero=False)),
            y=alt.Y("team:N", sort="-x", title=None),
            color=alt.Color("rating:Q", legend=None, scale=alt.Scale(range=SEQUENTIAL_BLUE)),
            tooltip=["team", alt.Tooltip("rating:Q", format=".1f"), "date"],
        ).properties(height=max(160, 24 * len(top))),
        width="stretch")
    st.dataframe(league_ratings.rename(columns={"team": "Team", "rating": "Rating", "date": "As of"}),
                 width="stretch", hide_index=True,
                 column_config={"Rating": st.column_config.NumberColumn(format="%.1f")})

with tab_trajectory:
    teams = sorted(league_history["team"].unique())
    default = league_ratings.head(5)["team"].tolist()
    picks = st.multiselect("Teams", teams, default=default, max_selections=8)
    if not picks:
        st.info("Pick at least one team.")
    else:
        traj = league_history[league_history["team"].isin(picks)].sort_values("date")
        st.altair_chart(
            alt.Chart(traj).mark_line(strokeWidth=2).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("rating_after:Q", title="Elo rating", scale=alt.Scale(zero=False)),
                color=alt.Color("team:N", title=None,
                                scale=alt.Scale(domain=picks, range=CATEGORICAL_COLORS[:len(picks)])),
                tooltip=["team", "date", alt.Tooltip("rating_after:Q", format=".1f"), "opponent", "result"],
            ).properties(height=420).interactive(),
            width="stretch")
