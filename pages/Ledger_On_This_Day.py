"""
Ledger -- On This Day
======================
Every match played on a given month/day since 1993, across all six
collected leagues. The year picked in the date widget is ignored --
only month and day matter, so this genuinely answers "what happened on
this day" across every season on record, not just one year of it.
"""

from datetime import date

import pandas as pd
import streamlit as st

from core import ledger_data as ld

st.title("On this day")
st.caption("Every collected match played on this month and day, any year since 1993.")

matches = ld.load_matches()
if matches.empty:
    st.info("No match data yet. Run `python -m collectors.football_data`.")
    st.stop()

picked = st.date_input("Date", value=date.today())
st.caption("Only the month and day are used -- the year is just a convenient picker.")

dates = pd.to_datetime(matches["date"])
on_day = matches[(dates.dt.month == picked.month) & (dates.dt.day == picked.day)].copy()

if on_day.empty:
    st.info(f"No matches on record for {picked.strftime('%B')} {picked.day}.")
    st.stop()

on_day["Year"] = pd.to_datetime(on_day["date"]).dt.year
on_day["League"] = on_day["league"].map(ld.league_label)
on_day["Score"] = on_day["home_goals"].astype(int).astype(str) + "-" + on_day["away_goals"].astype(int).astype(str)

st.caption(f"{len(on_day)} match(es) across {on_day['Year'].nunique()} year(s).")

table = on_day.sort_values(["Year", "League"], ascending=[False, True])[
    ["Year", "League", "home_team", "away_team", "Score", "result"]
].rename(columns={"home_team": "Home", "away_team": "Away", "result": "Result"})
st.dataframe(table, width="stretch", hide_index=True)
