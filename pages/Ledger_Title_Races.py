"""
Ledger -- Title Races
======================
Monte Carlo season simulation (models/title_race.py): for a completed
season, the championship-win probability at every matchday, using
standings and Elo ratings exactly as they stood on that date.

Only complete seasons are offered -- football-data.co.uk only ever
carries matches that have already been played, so a season still in
progress has no known remaining fixtures to simulate from. Showing one
anyway would have simulate_title_race treat "no more matches in the
file yet" as "season over," handing the current leader a false 100%.
See core/ledger_data.py's complete_seasons() for the exact rule.
"""

import altair as alt
import streamlit as st

from core import ledger_data as ld

st.title("Title races")
st.caption("Monte Carlo simulation from Elo ratings, run at every matchday of a completed season.")

CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                      "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

matches = ld.load_matches()
if matches.empty:
    st.info("No match data yet. Run `python -m collectors.football_data`.")
    st.stop()

leagues = sorted(matches["league"].unique(), key=ld.league_label)
default_idx = leagues.index("E0") if "E0" in leagues else 0
league = st.selectbox("League", leagues, index=default_idx, format_func=ld.league_label)

seasons = ld.complete_seasons(matches, league)
if not seasons:
    st.info("No complete season found for this league yet.")
    st.stop()
season = st.selectbox("Season", seasons, index=len(seasons) - 1, format_func=ld.season_label)

col_a, col_b = st.columns(2)
n_sims = col_a.select_slider("Simulations per checkpoint", [200, 500, 1000, 2000], value=500)
seed = col_b.number_input("Random seed", value=42, step=1,
                          help="Fixed by default so the chart doesn't jitter on every rerun.")

path = ld.simulate_path(league, season, n_sims, int(seed))
if path.empty:
    st.info("Nothing to simulate -- this season has no matches on record.")
    st.stop()

final_checkpoint = path[path["date"] == path["date"].max()]
prob_sum = final_checkpoint["win_probability"].sum()
st.caption(f"{path['date'].nunique()} checkpoints · probabilities sum to "
          f"{prob_sum:.4f} at every checkpoint (allow ±0.001 for per-team rounding).")

champion = final_checkpoint.sort_values("win_probability", ascending=False).iloc[0]
st.metric(f"{ld.season_label(season)} {ld.league_label(league)} champion (simulated)",
         champion["team"], f"{champion['win_probability'] * 100:.1f}% at the final matchday")

top_teams = final_checkpoint.nlargest(8, "win_probability")["team"].tolist()
chart_data = path[path["team"].isin(top_teams)]

st.altair_chart(
    alt.Chart(chart_data).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("win_probability:Q", title="Title probability", axis=alt.Axis(format="%")),
        color=alt.Color("team:N", title=None,
                        scale=alt.Scale(domain=top_teams, range=CATEGORICAL_COLORS[:len(top_teams)])),
        tooltip=["team", "date", alt.Tooltip("win_probability:Q", format=".1%")],
    ).properties(height=440).interactive(),
    width="stretch")

with st.expander("Final probabilities, every team"):
    table = final_checkpoint[["team", "win_probability"]].sort_values(
        "win_probability", ascending=False).copy()
    table["Title probability %"] = (table["win_probability"] * 100).round(1)
    table = table.rename(columns={"team": "Team"})[["Team", "Title probability %"]]
    st.dataframe(table, width="stretch", hide_index=True,
                column_config={"Title probability %": st.column_config.NumberColumn(format="%.1f")})
