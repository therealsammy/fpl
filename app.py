#!/usr/bin/env python3
"""
FPL Tracker -- Streamlit app
===========================
Reads the append-only snapshot store written by fpl_tracker.py and gives you
both a browsable view of every player and the analysis a static page cannot do.

    pip install -r requirements-app.txt
    streamlit run app.py

Data source, in order of preference:
  1. ./fpl_history.csv                    (local, alongside this file)
  2. GITHUB_CSV_URL below                 (raw.githubusercontent.com)
  3. File uploader in the sidebar         (manual fallback)

Unlike the static dashboard, this can call the FPL API directly -- the browser
is blocked by CORS, a Python process is not. See the "Live" page.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

import fpl_rivals as fr
import fpl_projections as fpr

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CSV_PATH = Path("fpl_history.csv")
SQUAD_PATH = Path("squad.json")
GITHUB_CSV_URL = "https://raw.githubusercontent.com/therealsammy/fpl/refs/heads/main/fpl_history.csv"
API = "https://fantasy.premierleague.com/api"

MINUTES_PATH = Path("fpl_minutes.csv")
RIVALS_PATH = Path("fpl_rivals.csv")
SIGNALS_PATH = Path("signals.csv")
DEFCON_REPORT_PATH = Path("defcon_report.csv")
PROJECTIONS_PATH = Path("fixture_projections.csv")     # phase 5: fixture-level xG/clean sheet
PLAYER_PROJECTIONS_PATH = Path("projections.csv")       # phase 7: per-player expected points

NUMERIC = [
    "GW", "Price", "Owned %", "Form", "Total pts", "GW pts", "PPG", "Minutes",
    "Starts", "Goals", "Assists", "xGI", "DEFCON", "DEFCON per 90",
    "Exp pts next", "Transfers in (GW)", "Transfers out (GW)", "Price change (GW)",
]
POS_ORDER = ["GKP", "DEF", "MID", "FWD"]

# Validated data-viz reference palette (see the dataviz skill). Categorical
# hues are used in this fixed order, never cycled or reassigned by filter.
CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                      "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]
DIVERGING_BLUE_RED = ["#104281", "#86b6ef", "#f0efec", "#f0a68a", "#d03b3b"]

# Status colors are reserved for state, never reused as a categorical hue.
STATUS_COLORS = {"good": "#0ca30c", "warning": "#fab219",
                 "serious": "#ec835a", "critical": "#d03b3b", "neutral": "#898781"}
STATUS_BADGE = {"good": "green", "warning": "orange",
                "serious": "orange", "critical": "red", "neutral": "gray"}

st.set_page_config(page_title="FPL Tracker", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def load_csv(text: str | None, path_mtime: float | None) -> pd.DataFrame:
    """Parse the snapshot store. Cached on file mtime so edits invalidate it."""
    if text is not None:
        from io import StringIO
        df = pd.read_csv(StringIO(text))
    else:
        df = pd.read_csv(CSV_PATH)
    return _clean(df)


@st.cache_data(ttl=900, show_spinner=False)
def load_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    return _clean(pd.read_csv(StringIO(r.text)))


@st.cache_data(ttl=900, show_spinner=False)
def _load_side_csv(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(Path(path_str))


def load_side(path: Path) -> pd.DataFrame | None:
    """Any of the phase 1-5 outputs. None if the file doesn't exist yet --
    every page below treats that as 'run the script', not an error."""
    if not path.exists():
        return None
    return _load_side_csv(str(path), path.stat().st_mtime)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ID"]).copy()
    df["ID"] = df["ID"].astype(int)
    df["Snapshot"] = df["Snapshot"].astype(str)

    # Derived columns the store doesn't carry
    df["G+A"] = df.get("Goals", 0).fillna(0) + df.get("Assists", 0).fillna(0)
    df["xGI gap"] = (df["G+A"] - df["xGI"]).round(2)          # +ve = overperforming
    df["Pts per £m"] = (df["Total pts"] / df["Price"]).round(1)
    df["Net transfers"] = df.get("Transfers in (GW)", 0).fillna(0) - \
                          df.get("Transfers out (GW)", 0).fillna(0)
    mins = df["Minutes"].replace(0, np.nan)
    df["xGI per 90"] = (df["xGI"] / mins * 90).round(3)
    df["Pts per 90"] = (df["Total pts"] / mins * 90).round(2)
    return df.sort_values(["Snapshot", "ID"]).reset_index(drop=True)


def get_data() -> pd.DataFrame | None:
    """Resolve a data source, preferring local file, then URL, then upload.
    Collapsed by default -- the common case (local file) needs no attention,
    so this shouldn't compete with page filters for sidebar space."""
    with st.sidebar.expander("Data source", expanded=False):
        src = st.radio(
            "Source", ["Local file", "GitHub URL", "Upload"],
            help="Local reads ./fpl_history.csv next to this app.",
            label_visibility="collapsed",
        )
        try:
            if src == "Local file":
                if not CSV_PATH.exists():
                    st.error(f"{CSV_PATH} not found.")
                    return None
                return load_csv(None, CSV_PATH.stat().st_mtime)

            if src == "GitHub URL":
                url = st.text_input("Raw CSV URL", value=GITHUB_CSV_URL,
                                    placeholder="https://raw.githubusercontent.com/...")
                if not url:
                    st.info("Paste a raw.githubusercontent.com link.")
                    return None
                return load_url(url)

            up = st.file_uploader("fpl_history.csv", type="csv")
            if up is None:
                return None
            return load_csv(up.getvalue().decode("utf-8"), None)

        except Exception as exc:
            st.error(f"Could not read data: {exc}")
            return None


def latest_with_deltas(df: pd.DataFrame, back: int = 1) -> pd.DataFrame:
    """Most recent snapshot, plus change vs `back` snapshots earlier."""
    snaps = sorted(df["Snapshot"].unique())
    cur = df[df["Snapshot"] == snaps[-1]].set_index("ID").copy()

    delta_cols = ["Price", "Owned %", "Form", "Total pts", "xGI", "Minutes"]
    if len(snaps) > back:
        prev = df[df["Snapshot"] == snaps[-1 - back]].set_index("ID")
        for c in delta_cols:
            if c in cur.columns:
                cur[f"Δ{c}"] = (cur[c] - prev[c].reindex(cur.index)).round(2)
    else:
        for c in delta_cols:
            cur[f"Δ{c}"] = np.nan

    return cur.reset_index()


def attach_projections(cur: pd.DataFrame) -> pd.DataFrame:
    """Left-joins the latest phase-7 projections onto the Players table.
    A no-op (silently) whenever the readiness gate hasn't opened yet --
    the columns just won't exist, which the Players page already handles."""
    proj = load_side(PLAYER_PROJECTIONS_PATH)
    if proj is None or proj.empty:
        return cur
    latest = proj[proj["Snapshot"] == proj["Snapshot"].max()]
    cols = latest.set_index("ID")[["Expected points", "Expected points low",
                                    "Expected points high", "Confidence"]]
    return cur.join(cols, on="ID")


def series_for(df: pd.DataFrame, pid: int) -> pd.DataFrame:
    return df[df["ID"] == pid].sort_values("Snapshot")


# ---------------------------------------------------------------------------
# SQUAD (persisted to squad.json, falls back to session)
# ---------------------------------------------------------------------------

def load_squad() -> set[int]:
    if "squad" in st.session_state:
        return st.session_state["squad"]
    ids: set[int] = set()
    if SQUAD_PATH.exists():
        try:
            ids = set(json.loads(SQUAD_PATH.read_text()))
        except Exception:
            ids = set()
    st.session_state["squad"] = ids
    return ids


def save_squad(ids: set[int]) -> str:
    st.session_state["squad"] = ids
    try:
        SQUAD_PATH.write_text(json.dumps(sorted(ids)))
        return f"Saved to {SQUAD_PATH}"
    except Exception:
        return "Kept for this session only (couldn't write squad.json)"


# ---------------------------------------------------------------------------
# CHART HELPERS
# ---------------------------------------------------------------------------

def line(data: pd.DataFrame, metric: str, color: str | None = None, height: int = 210):
    """Time series of one metric. `color` groups multiple players."""
    enc = dict(
        x=alt.X("Snapshot:O", title=None, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(zero=False)),
        tooltip=["Snapshot", "Player", alt.Tooltip(f"{metric}:Q", format=".2f")],
    )
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=None)
    return alt.Chart(data).mark_line(point=True, strokeWidth=2).encode(**enc).properties(height=height)


def scatter(data, x, y, tip, color="Pos", size=None, height=440):
    color_scale = (alt.Scale(domain=POS_ORDER, range=CATEGORICAL_COLORS[:4])
                   if color == "Pos" else alt.Undefined)
    enc = dict(
        x=alt.X(f"{x}:Q", scale=alt.Scale(zero=False)),
        y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False)),
        color=alt.Color(f"{color}:N", sort=POS_ORDER, title=None, scale=color_scale),
        tooltip=tip,
    )
    if size:
        enc["size"] = alt.Size(f"{size}:Q", title=size, legend=None)
    return alt.Chart(data).mark_circle(opacity=.72).encode(**enc).properties(height=height).interactive()


def col_config(df: pd.DataFrame) -> dict:
    """Sensible number formatting for st.dataframe."""
    cfg = {}
    money = {"Price", "ΔPrice", "Sell value"}
    two_dp = {"xGI", "xGI gap", "xGI per 90", "Pts per 90",
              "Expected points", "Expected points low", "Expected points high"}
    for c in df.columns:
        if c in money:
            cfg[c] = st.column_config.NumberColumn(c, format="%.1f")
        elif c in two_dp:
            cfg[c] = st.column_config.NumberColumn(c, format="%.2f")
        elif c in {"Owned %", "ΔOwned %", "Form", "ΔForm", "PPG",
                   "Exp pts next", "DEFCON per 90", "Pts per £m", "Rise/fall %"}:
            cfg[c] = st.column_config.NumberColumn(c, format="%.1f")
    return cfg


STATUS_MAPS = {
    "classification": {
        "Nailed": "good", "Rotation risk": "warning", "Bench": "serious",
        "Benchwarmer": "critical", "Insufficient data": "neutral",
    },
    "confidence": {"High": "good", "Low": "warning"},
    "setpiece_direction": {
        "Gained duty": "good", "Promoted (higher priority)": "good",
        "Demoted (lower priority)": "warning", "Lost duty": "critical",
    },
    "price_direction": {"Rise watch": "good", "Fall watch": "critical"},
}


FPL_GREEN = "#00ff87"


def style_own_row(df: pd.DataFrame, id_col: str, my_id, color: str = FPL_GREEN):
    """Highlights the viewer's own row -- e.g. their entry in a rivals
    standings table -- so they can find themselves at a glance."""
    def _paint(row):
        hit = row[id_col] == my_id
        return [f"background-color: {color}26; font-weight: 600" if hit else "" for _ in row]
    return df.style.apply(_paint, axis=1) if id_col in df.columns else df


def style_status(df: pd.DataFrame, column: str, mapping: dict):
    """Tints one column's text/background by a status mapping (good/warning/
    serious/critical/neutral) -- the reserved status palette, never reused
    as a categorical hue elsewhere in the app."""
    def _paint(val):
        status = mapping.get(val)
        if status is None:
            return ""
        hex_color = STATUS_COLORS[status]
        return f"background-color: {hex_color}22; color: {hex_color}; font-weight: 600"
    return df.style.map(_paint, subset=[column]) if column in df.columns else df


# ---------------------------------------------------------------------------
# FILTERS
# ---------------------------------------------------------------------------

def sidebar_filters(cur: pd.DataFrame, squad: set[int], key: str) -> pd.DataFrame:
    st.sidebar.markdown("### Filters")
    q = st.sidebar.text_input("Search player", key=f"q{key}")
    pos = st.sidebar.multiselect("Position", POS_ORDER, key=f"pos{key}")
    teams = st.sidebar.multiselect("Team", sorted(cur["Team"].unique()), key=f"team{key}")

    pmin, pmax = float(cur["Price"].min()), float(cur["Price"].max())
    price = st.sidebar.slider("Price £m", pmin, pmax, (pmin, pmax), 0.1, key=f"pr{key}")
    owned = st.sidebar.slider("Owned %", 0.0, 100.0, (0.0, 100.0), 0.5, key=f"ow{key}")

    mmax = int(cur["Minutes"].max()) if cur["Minutes"].notna().any() else 0
    mins = st.sidebar.slider("Minimum minutes", 0, max(mmax, 1), 0, 90, key=f"mn{key}")

    fit = st.sidebar.checkbox("Available only", key=f"fit{key}")
    mine = st.sidebar.checkbox(f"My squad only ({len(squad)})", key=f"mine{key}",
                               disabled=not squad)

    out = cur.copy()
    if q:
        out = out[out["Player"].str.contains(q, case=False, na=False)]
    if pos:
        out = out[out["Pos"].isin(pos)]
    if teams:
        out = out[out["Team"].isin(teams)]
    out = out[out["Price"].between(*price) & out["Owned %"].between(*owned)]
    out = out[out["Minutes"].fillna(0) >= mins]
    if fit:
        out = out[out["Status"] == "Available"]
    if mine and squad:
        out = out[out["ID"].isin(squad)]
    return out


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------

def page_players(df, cur, squad):
    st.header("Players")
    view = sidebar_filters(cur, squad, "pl")
    st.caption(f"{len(view)} of {len(cur)} players")

    if view.empty:
        st.info("Nothing matches. Widen a filter in the sidebar.")
        return

    default = ["Player", "Team", "Pos", "Price", "ΔPrice", "Owned %", "ΔOwned %",
               "Form", "Total pts", "xGI", "xGI gap", "DEFCON per 90", "Minutes",
               "Exp pts next", "Expected points", "Pts per £m", "Status"]
    available = [c for c in view.columns if c not in {"Snapshot", "Snapshot UTC", "GW finished"}]
    cols = st.multiselect("Columns", available,
                          default=[c for c in default if c in available])
    sort_by = st.selectbox("Sort by", cols or available,
                           index=(cols or available).index("Total pts")
                           if "Total pts" in (cols or available) else 0)
    asc = st.toggle("Ascending", value=False)

    shown = view[cols or available].sort_values(sort_by, ascending=asc)
    st.dataframe(shown, width="stretch", hide_index=True,
                 column_config=col_config(shown), height=560)
    st.download_button("Download this view as CSV",
                       shown.to_csv(index=False).encode(),
                       "fpl_view.csv", "text/csv")


def page_player(df, cur, squad):
    st.header("Player detail")
    names = cur.sort_values("Total pts", ascending=False)
    label = names["Player"] + "  ·  " + names["Team"] + " " + names["Pos"]
    pick = st.selectbox("Player", label.tolist())
    pid = int(names.iloc[label.tolist().index(pick)]["ID"])

    p = cur[cur["ID"] == pid].iloc[0]
    s = series_for(df, pid)

    top = st.columns(6)
    for col, (k, v, d) in zip(top, [
        ("Price", f"£{p['Price']:.1f}m", p.get("ΔPrice")),
        ("Owned", f"{p['Owned %']:.1f}%", p.get("ΔOwned %")),
        ("Points", int(p["Total pts"]), p.get("ΔTotal pts")),
        ("Form", f"{p['Form']:.1f}", p.get("ΔForm")),
        ("xGI", f"{p['xGI']:.2f}", p.get("ΔxGI")),
        ("Minutes", int(p["Minutes"]), p.get("ΔMinutes")),
    ]):
        col.metric(k, v, None if pd.isna(d) else round(float(d), 2))

    in_squad = pid in squad
    if st.button("Remove from my squad" if in_squad else "Add to my squad"):
        squad.discard(pid) if in_squad else squad.add(pid)
        st.toast(save_squad(squad))
        st.rerun()

    if p["Status"] != "Available":
        st.warning(f"{p['Status']}"
                   + (f" · {p['Chance next GW']:.0f}% chance next GW"
                      if pd.notna(p.get("Chance next GW")) else ""))

    if len(s) < 2:
        st.info("Charts need at least two snapshots. Run the tracker again next week.")
    else:
        metrics = st.multiselect(
            "Track", ["Price", "Owned %", "Form", "Total pts", "xGI", "Minutes",
                      "DEFCON per 90", "Exp pts next"],
            default=["Price", "Owned %", "Form", "xGI"])
        for pair in [metrics[i:i + 2] for i in range(0, len(metrics), 2)]:
            for col, m in zip(st.columns(len(pair)), pair):
                col.altair_chart(line(s, m), width="stretch")

    with st.expander("All recorded snapshots"):
        st.dataframe(s.drop(columns=["ID"]), width="stretch", hide_index=True)


def page_compare(df, cur, squad):
    st.header("Compare")
    st.caption("Overlay any players on the same axis.")

    names = cur.sort_values("Total pts", ascending=False)
    label = (names["Player"] + "  ·  " + names["Team"] + " " + names["Pos"]).tolist()
    picks = st.multiselect("Players (2–6)", label, max_selections=6)
    if len(picks) < 2:
        st.info("Pick at least two.")
        return

    ids = [int(names.iloc[label.index(p)]["ID"]) for p in picks]
    sub = df[df["ID"].isin(ids)]
    latest = cur[cur["ID"].isin(ids)]

    st.subheader("Side by side")
    show = ["Player", "Team", "Pos", "Price", "Owned %", "Total pts", "Form", "PPG",
            "Minutes", "Starts", "Goals", "Assists", "xGI", "xGI gap",
            "DEFCON per 90", "Pts per £m", "Exp pts next", "Status"]
    st.dataframe(latest[[c for c in show if c in latest.columns]],
                 width="stretch", hide_index=True, column_config=col_config(latest))

    if sub["Snapshot"].nunique() < 2:
        st.info("Trend overlays need two or more snapshots.")
        return

    st.subheader("Over time")
    metrics = st.multiselect("Metrics", ["Total pts", "Form", "Price", "Owned %",
                                         "xGI", "Minutes", "DEFCON per 90"],
                             default=["Total pts", "Form", "Owned %", "xGI"])
    for pair in [metrics[i:i + 2] for i in range(0, len(metrics), 2)]:
        for col, m in zip(st.columns(len(pair)), pair):
            col.altair_chart(line(sub, m, color="Player"), width="stretch")


def page_movers(df, cur, squad):
    st.header("Movers")
    if df["Snapshot"].nunique() < 2:
        st.info("Movers compares snapshots. Run the tracker again next week.")
        return

    snaps = sorted(df["Snapshot"].unique())
    back = st.slider("Compare against", 1, len(snaps) - 1, 1,
                     help="How many snapshots back to measure change from.")
    d = latest_with_deltas(df, back)
    st.caption(f"{snaps[-1 - back]} → {snaps[-1]}")

    floor = st.number_input("Ignore players owned below %", 0.0, 20.0, 0.5, 0.5)
    d = d[d["Owned %"] >= floor]
    n = st.slider("Rows per table", 5, 40, 12)

    cols = ["Player", "Team", "Pos", "Price", "ΔPrice", "Owned %", "ΔOwned %",
            "Form", "ΔForm", "Total pts", "Status"]
    cols = [c for c in cols if c in d.columns]

    for metric, label in [("ΔOwned %", "Ownership"), ("ΔPrice", "Price"),
                          ("ΔForm", "Form"), ("ΔTotal pts", "Points scored")]:
        if metric not in d.columns or d[metric].isna().all():
            continue
        st.subheader(label)
        a, b = st.columns(2)
        a.caption("Rising"); b.caption("Falling")
        a.dataframe(d.nlargest(n, metric)[cols], width="stretch", hide_index=True,
                    column_config=col_config(d))
        b.dataframe(d.nsmallest(n, metric)[cols], width="stretch", hide_index=True,
                    column_config=col_config(d))


def page_teams(df, cur, squad):
    st.header("Teams")
    g = cur.groupby("Team").agg(
        Players=("ID", "count"),
        **{"Total pts": ("Total pts", "sum"),
           "Avg price": ("Price", "mean"),
           "Owned % (sum)": ("Owned %", "sum"),
           "Best xGI": ("xGI", "max")}
    ).round(1).reset_index().sort_values("Total pts", ascending=False)

    top = cur.loc[cur.groupby("Team")["Total pts"].idxmax(),
                  ["Team", "Player", "Total pts", "Price"]]
    top.columns = ["Team", "Top scorer", "Their pts", "Their £m"]
    g = g.merge(top, on="Team")

    st.dataframe(g, width="stretch", hide_index=True, column_config=col_config(g))
    st.altair_chart(
        alt.Chart(g).mark_bar().encode(
            x=alt.X("Total pts:Q"),
            y=alt.Y("Team:N", sort="-x", title=None),
            color=alt.Color("Total pts:Q", legend=None, scale=alt.Scale(range=SEQUENTIAL_BLUE)),
            tooltip=["Team", "Total pts", "Players", "Top scorer"],
        ).properties(height=520), width="stretch")

    team = st.selectbox("Squad breakdown", sorted(cur["Team"].unique()))
    sq = cur[cur["Team"] == team].sort_values("Total pts", ascending=False)
    st.dataframe(sq[["Player", "Pos", "Price", "Owned %", "Total pts", "Form",
                     "Minutes", "xGI", "DEFCON per 90", "Status"]],
                 width="stretch", hide_index=True, column_config=col_config(sq))


def page_squad(df, cur, squad):
    st.header("My squad")

    names = cur.sort_values("Player")
    label = (names["Player"] + "  ·  " + names["Team"] + " " + names["Pos"]).tolist()
    preset = [l for l, i in zip(label, names["ID"]) if i in squad]
    picks = st.multiselect("Your 15", label, default=preset, max_selections=15)

    if st.button("Save squad"):
        ids = {int(names.iloc[label.index(p)]["ID"]) for p in picks}
        st.toast(save_squad(ids))
        st.rerun()

    if not squad:
        st.info("Pick your players above and hit Save.")
        return

    mine = cur[cur["ID"].isin(squad)]
    a, b, c, d = st.columns(4)
    a.metric("Squad value", f"£{mine['Price'].sum():.1f}m")
    b.metric("Total points", int(mine["Total pts"].sum()))
    c.metric("Avg ownership", f"{mine['Owned %'].mean():.1f}%")
    unfit = int((mine["Status"] != "Available").sum())
    d.metric("Not fully fit", unfit, delta=None if not unfit else "check news",
             delta_color="inverse")

    if unfit:
        st.warning("Flagged: " + ", ".join(
            f"{r['Player']} ({r['Status']})"
            for _, r in mine[mine["Status"] != "Available"].iterrows()))

    cols = ["Player", "Team", "Pos", "Price", "ΔPrice", "Owned %", "ΔOwned %",
            "Form", "ΔForm", "Total pts", "ΔTotal pts", "xGI", "xGI gap",
            "Minutes", "Exp pts next", "Status"]
    st.dataframe(mine[[c for c in cols if c in mine.columns]]
                 .sort_values("Pos", key=lambda s: s.map({p: i for i, p in enumerate(POS_ORDER)})),
                 width="stretch", hide_index=True, column_config=col_config(mine))

    if df["Snapshot"].nunique() >= 2:
        st.subheader("Squad trend")
        sub = df[df["ID"].isin(squad)]
        metric = st.selectbox("Metric", ["Total pts", "Form", "Price", "Owned %", "xGI"])
        st.altair_chart(line(sub, metric, color="Player", height=340),
                        width="stretch")


def page_rivals(df, cur, squad):
    st.header("Mini-league rivals")
    rivals = load_side(RIVALS_PATH)
    if rivals is None or rivals.empty:
        st.info("No data yet. Run `python fpl_rivals.py` (needs FPL_LEAGUE_ID set) "
                "once a gameweek's transfer deadline has passed.")
        return

    latest_snap = rivals["Snapshot"].max()
    gw_data = rivals[rivals["Snapshot"] == latest_snap]
    gw_data["Captain"] = gw_data["Captain"].astype(bool)
    latest_gw = int(gw_data["GW"].iloc[0])
    st.caption(f"GW{latest_gw} · {gw_data['Entry ID'].nunique()} manager(s) · snapshot {latest_snap}")

    standings = (gw_data[["Entry ID", "Manager", "Team name", "Rank", "Total points"]]
                 .drop_duplicates().sort_values("Rank"))
    st.subheader("Standings")
    st.caption("Your row is highlighted.")
    st.dataframe(style_own_row(standings, "Entry ID", fr.ENTRY_ID),
                 width="stretch", hide_index=True)

    names = cur.set_index("ID")[["Player", "Team"]]

    def enrich(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty:
            return d
        d = d.join(names, on="Player ID")
        cols = ["Player", "Team"] + [c for c in d.columns if c not in ("Player", "Team")]
        return d[cols]

    st.subheader("Effective ownership (this league)")
    st.caption("Owned % + Captained %, as a share of this league only -- not the global game.")
    eff = enrich(fr.compute_effective_ownership(gw_data)).head(15)
    if not eff.empty:
        st.altair_chart(
            alt.Chart(eff).mark_bar(color=CATEGORICAL_COLORS[0], size=14).encode(
                x=alt.X("Effective ownership %:Q"),
                y=alt.Y("Player:N", sort="-x", title=None),
                tooltip=["Player", "Team", "Owned %", "Captained %", "Effective ownership %"],
            ).properties(height=max(120, 24 * len(eff))),
            width="stretch")
    st.dataframe(eff, width="stretch", hide_index=True)

    my_entry_id = fr.ENTRY_ID
    unique_to_me, missing_for_me = fr.compute_differentials(gw_data, my_entry_id)
    a, b = st.columns(2)
    a.subheader(f"Only you own ({len(unique_to_me)})")
    a.dataframe(enrich(unique_to_me), width="stretch", hide_index=True)
    b.subheader("Rivals own, you don't (top 10)")
    b.dataframe(enrich(missing_for_me).head(10), width="stretch", hide_index=True)

    st.subheader("Captain divergence")
    captain = fr.compute_captain_divergence(gw_data, my_entry_id)
    if captain["my_captain"] is None:
        st.info("Your picks weren't available in this run -- can't compare captains.")
    else:
        my_name = names["Player"].get(captain["my_captain"], f"#{captain['my_captain']}")
        field_name = names["Player"].get(captain["field_top_captain"], f"#{captain['field_top_captain']}")
        msg = (f"You captained **{my_name}**. Field's top pick: **{field_name}** "
               f"({captain['field_top_count']}/{captain['field_total']}, {captain['field_top_pct']}%).")
        if captain["diverges"]:
            st.warning(msg)
        else:
            st.success(msg)


def page_signals(df, cur, squad):
    st.header("Signals")
    sig = load_side(SIGNALS_PATH)
    if sig is None or sig.empty:
        st.info("No data yet. Run `python fpl_signals.py` after the tracker.")
        return

    latest_snap = sig["Snapshot"].max()
    latest = sig[sig["Snapshot"] == latest_snap]
    st.caption(f"Snapshot {latest_snap} · GW{int(latest['GW'].iloc[0])}")

    st.subheader("Set-piece changes")
    setpiece = latest[latest["Signal"] == "Set-piece change"]
    if setpiece.empty:
        st.info("No changes detected this run.")
    else:
        view = setpiece[["Player", "Team", "Metric", "Old value", "New value", "Note"]]
        st.dataframe(style_status(view, "Note", STATUS_MAPS["setpiece_direction"]),
                     width="stretch", hide_index=True)

    st.subheader("Price momentum watch")
    st.caption("Tiebreaker only -- never a standalone buy/sell reason.")
    price = latest[latest["Signal"] == "Price momentum"].copy()
    if price.empty:
        st.info("No players close to a price change.")
    else:
        price = price.rename(columns={"New value": "Rise/fall %"})
        price["Direction"] = price["Note"].str.split(" (", n=1, regex=False).str[0]
        price = price.sort_values("Rise/fall %", key=lambda s: s.abs(), ascending=False)

        st.altair_chart(
            alt.Chart(price).mark_bar(size=14).encode(
                x=alt.X("Rise/fall %:Q", scale=alt.Scale(domain=[-100, 100])),
                y=alt.Y("Player:N", sort=alt.SortField("Rise/fall %", order="descending"), title=None),
                color=alt.Color("Direction:N", title=None,
                                scale=alt.Scale(domain=["Rise watch", "Fall watch"],
                                                range=[DIVERGING_BLUE_RED[0], DIVERGING_BLUE_RED[-1]])),
                tooltip=["Player", "Team", alt.Tooltip("Rise/fall %:Q", format=".1f"), "Note"],
            ).properties(height=max(120, 24 * len(price))),
            width="stretch")

        view = price[["Player", "Team", "Direction", "Rise/fall %", "Note"]]
        st.dataframe(style_status(view, "Direction", STATUS_MAPS["price_direction"]),
                     width="stretch", hide_index=True, column_config=col_config(view))

    st.subheader("Fixture-adjusted form")
    form = latest[latest["Signal"] == "Fixture-adjusted form"].copy()
    if form.empty:
        st.info("No data (no minutes played yet, or fixtures unavailable).")
    else:
        form = form.rename(columns={"Old value": "Form", "New value": "Fixture-adjusted form"})
        top_n = st.slider("Show top N", 5, 50, 15)
        st.dataframe(
            form.nlargest(top_n, "Fixture-adjusted form")[
                ["Player", "Team", "Form", "Fixture-adjusted form", "Note"]],
            width="stretch", hide_index=True)


def page_odds(df, cur, squad):
    st.header("Fixture projections")
    st.caption("Derived from bookmaker odds via a Dixon-Coles Poisson fit. Not wired into "
               "player rankings yet -- this gets validated against reality first.")
    proj = load_side(PROJECTIONS_PATH)
    if proj is None or proj.empty:
        st.info("No data yet. Run `python fpl_odds.py` (needs ODDS_API_KEY set).")
        return

    latest_snap = proj["Snapshot"].max()
    latest = proj[proj["Snapshot"] == latest_snap].copy()
    gws = sorted(g for g in latest["GW"].dropna().unique())
    gw = st.selectbox("Gameweek", gws) if gws else None
    view = latest[latest["GW"] == gw] if gw is not None else latest
    st.caption(f"Snapshot {latest_snap}")

    view["Clean sheet %"] = (view["Clean sheet probability"] * 100).round(1)
    view["Win %"] = (view["Win probability"] * 100).round(1)
    view["Draw %"] = (view["Draw probability"] * 100).round(1)
    view["Loss %"] = (view["Loss probability"] * 100).round(1)

    cols = ["Team", "Opponent", "Home", "Expected goals for", "Expected goals against",
            "Clean sheet %", "Win %", "Draw %", "Loss %"]
    st.dataframe(view[cols].sort_values("Expected goals for", ascending=False),
                 width="stretch", hide_index=True,
                 column_config={
                     "Expected goals for": st.column_config.NumberColumn(format="%.2f"),
                     "Expected goals against": st.column_config.NumberColumn(format="%.2f"),
                 })

    st.subheader("Best clean sheet bets")
    st.dataframe(view.nlargest(8, "Clean sheet %")[["Team", "Opponent", "Home", "Clean sheet %"]],
                 width="stretch", hide_index=True)


def page_projections(df, cur, squad):
    st.header("Player projections")
    st.caption("Expected points per player for the next gameweek. Dormant until the "
               "readiness gate opens -- a projection built on too little history is a "
               "guess with a decimal point on it, so this says exactly what it's waiting for.")

    minutes_df = load_side(MINUTES_PATH)
    fixture_proj = load_side(PROJECTIONS_PATH)
    target_gw = fpr.determine_target_gw(fixture_proj)
    problems = fpr.check_readiness(df, minutes_df, fixture_proj, target_gw)

    if problems:
        st.warning("Not ready yet -- missing:")
        for p in problems:
            st.markdown(f"- {p}")
        st.info("This is the readiness gate working as intended, not an error.")
        return

    proj = load_side(PLAYER_PROJECTIONS_PATH)
    if proj is None or proj.empty:
        st.info("The gate is open but `projections.csv` doesn't exist yet -- "
                "run `python fpl_projections.py`.")
        return

    latest_snap = proj["Snapshot"].max()
    latest = proj[proj["Snapshot"] == latest_snap]
    n_low = int((latest["Confidence"] == "Low").sum())
    st.caption(f"GW{int(latest['GW'].iloc[0])} · snapshot {latest_snap} · "
               f"{len(latest)} player(s) · {n_low} low-confidence")

    pos_filter = st.multiselect("Position", POS_ORDER)
    view = latest[latest["Pos"].isin(pos_filter)] if pos_filter else latest

    st.altair_chart(
        alt.Chart(view).mark_circle(opacity=0.72).encode(
            x=alt.X("Price:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("Expected points:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Confidence:N", title=None,
                            scale=alt.Scale(domain=["High", "Low"],
                                            range=[STATUS_COLORS["good"], STATUS_COLORS["warning"]])),
            tooltip=["Player", "Team", "Pos", "Price", "Expected points", "Confidence"],
        ).properties(height=380).interactive(),
        width="stretch")

    cols = ["Player", "Team", "Pos", "Price", "Start probability", "Expected points",
            "Expected points low", "Expected points high", "Expected points per million",
            "Confidence", "Note"]
    table = view[[c for c in cols if c in view.columns]].sort_values("Expected points", ascending=False)
    st.dataframe(style_status(table, "Confidence", STATUS_MAPS["confidence"]),
                 width="stretch", hide_index=True, height=500,
                 column_config=col_config(table))

    with st.expander("Component breakdown"):
        st.caption("Each term's contribution to the total -- so a surprising number "
                   "can be traced to which component caused it.")
        comp_cols = ["Player", "Team", "Appearance pts", "Goals pts", "Assists pts",
                     "Clean sheet pts", "DEFCON pts", "Bonus pts", "Save pts"]
        st.dataframe(view[[c for c in comp_cols if c in view.columns]].sort_values("Player"),
                     width="stretch", hide_index=True)


def page_models(df, cur, squad):
    st.header("Minutes model & DEFCON test")

    st.subheader("Start probability")
    minutes = load_side(MINUTES_PATH)
    if minutes is None or minutes.empty:
        st.info("No data yet. Run `python fpl_minutes.py` after the tracker.")
    else:
        latest = minutes[minutes["Snapshot"] == minutes["Snapshot"].max()]
        n_ready = int((latest["Classification"] != "Insufficient data").sum())
        if n_ready == 0:
            note = latest["Note"].iloc[0] if "Note" in latest.columns and not latest.empty else ""
            st.warning(f"Dormant: {note} Expected early in the season.")
        else:
            st.caption(f"{n_ready} of {len(latest)} players have a start probability this run.")
        counts = latest["Classification"].value_counts().reindex(
            ["Nailed", "Rotation risk", "Bench", "Benchwarmer", "Insufficient data"]).dropna().reset_index()
        counts.columns = ["Classification", "Players"]
        status_domain = list(STATUS_MAPS["classification"].keys())
        status_range = [STATUS_COLORS[STATUS_MAPS["classification"][c]] for c in status_domain]
        st.altair_chart(
            alt.Chart(counts).mark_bar().encode(
                x=alt.X("Players:Q"),
                y=alt.Y("Classification:N", sort=status_domain, title=None),
                color=alt.Color("Classification:N", legend=None,
                                scale=alt.Scale(domain=status_domain, range=status_range)),
                tooltip=["Classification", "Players"],
            ).properties(height=180),
            width="stretch")

        classes = sorted(latest["Classification"].unique())
        default = [c for c in ["Nailed", "Rotation risk"] if c in classes]
        chosen = st.multiselect("Classification", classes, default=default)
        view = latest[latest["Classification"].isin(chosen)] if chosen else latest
        cols = ["Player", "Team", "Pos", "Classification", "Start probability",
                "Start rate (recent)", "Start rate (overall)", "Minutes per appearance", "Note"]
        view = view[[c for c in cols if c in view.columns]].sort_values(
            "Start probability", ascending=False, na_position="last")
        st.dataframe(style_status(view, "Classification", STATUS_MAPS["classification"]),
                     width="stretch", hide_index=True, height=400)

    st.divider()
    st.subheader("DEFCON vs opponent territory")
    st.caption("A hypothesis test, not a feature. Tracked over time as more data arrives.")
    report = load_side(DEFCON_REPORT_PATH)
    if report is None or report.empty:
        st.info("No data yet. Run `python fpl_defcon.py` after the tracker.")
    else:
        latest_row = report.sort_values("Snapshot").iloc[-1]
        verdict_text = str(latest_row["Verdict"])
        if verdict_text.startswith("Real effect"):
            st.badge("Real effect", color=STATUS_BADGE["good"])
        elif verdict_text.startswith("Null result"):
            st.badge("Null result", color=STATUS_BADGE["critical"])
        else:
            st.badge("Insufficient data", color=STATUS_BADGE["neutral"])
        st.write(verdict_text)
        a, b, c = st.columns(3)
        a.metric("Sample size", int(latest_row["n"]))
        b.metric("Pearson r", latest_row["Pearson r"] if pd.notna(latest_row["Pearson r"]) else "—")
        c.metric("Spearman rho", latest_row["Spearman rho"] if pd.notna(latest_row["Spearman rho"]) else "—")
        with st.expander("Report history"):
            st.dataframe(report, width="stretch", hide_index=True)


def page_analysis(df, cur, squad):
    st.header("Analysis")
    view = sidebar_filters(cur, squad, "an")
    if view.empty:
        st.info("Nothing matches the sidebar filters.")
        return

    tabs = st.tabs(["Regression candidates", "Value", "Differentials",
                    "Momentum", "Custom score"])

    with tabs[0]:
        st.markdown(
            "**Goals + assists against xGI.** Below the line means the underlying "
            "numbers are there but the returns aren't yet — those are the buys. "
            "Above it means a player is outscoring his chances, which tends not to last."
        )
        m = st.slider("Minimum minutes", 0, int(view["Minutes"].max() or 1), 450, 90)
        d = view[view["Minutes"] >= m].copy()
        if d.empty:
            st.info("No players clear that minutes threshold.")
        else:
            base = scatter(d, "xGI", "G+A",
                           ["Player", "Team", "Pos", "Price", "xGI", "G+A",
                            "xGI gap", "Total pts"], size="Total pts")
            hi = max(d["xGI"].max(), d["G+A"].max())
            ref = alt.Chart(pd.DataFrame({"x": [0, hi]})).mark_line(
                strokeDash=[5, 5], color="gray").encode(x="x:Q", y="x:Q")
            st.altair_chart(base + ref, width="stretch")

            a, b = st.columns(2)
            a.caption("Underperforming — potential buys")
            a.dataframe(d.nsmallest(12, "xGI gap")[
                ["Player", "Team", "Pos", "Price", "xGI", "G+A", "xGI gap", "Owned %"]],
                width="stretch", hide_index=True)
            b.caption("Overperforming — regression risk")
            b.dataframe(d.nlargest(12, "xGI gap")[
                ["Player", "Team", "Pos", "Price", "xGI", "G+A", "xGI gap", "Owned %"]],
                width="stretch", hide_index=True)

    with tabs[1]:
        st.markdown("**Points against price.** Anything well above the cloud is "
                    "returning more than it costs.")
        st.altair_chart(scatter(view, "Price", "Total pts",
                                ["Player", "Team", "Pos", "Price", "Total pts",
                                 "Pts per £m", "Owned %"], size="Owned %"),
                        width="stretch")
        st.dataframe(view.nlargest(20, "Pts per £m")[
            ["Player", "Team", "Pos", "Price", "Total pts", "Pts per £m",
             "Owned %", "Minutes", "Status"]],
            width="stretch", hide_index=True, column_config=col_config(view))

    with tabs[2]:
        st.markdown("**Ownership against points.** Bottom-right is where differentials "
                    "live: scoring well, barely owned.")
        cap = st.slider("Max ownership %", 1.0, 50.0, 10.0, 0.5)
        st.altair_chart(scatter(view, "Owned %", "Total pts",
                                ["Player", "Team", "Pos", "Owned %", "Total pts",
                                 "Price", "Form"]), width="stretch")
        d = view[view["Owned %"] <= cap].nlargest(20, "Total pts")
        st.dataframe(d[["Player", "Team", "Pos", "Price", "Owned %", "Total pts",
                        "Form", "xGI", "Minutes", "Status"]],
                     width="stretch", hide_index=True, column_config=col_config(d))

    with tabs[3]:
        st.markdown("**Net transfers this gameweek.** Heavy inflow usually precedes a "
                    "price rise — get in before it, not after.")
        if "Net transfers" not in view.columns or view["Net transfers"].isna().all():
            st.info("No transfer data in this snapshot.")
        else:
            a, b = st.columns(2)
            cols = ["Player", "Team", "Pos", "Price", "Owned %",
                    "Transfers in (GW)", "Transfers out (GW)", "Net transfers", "Form"]
            a.caption("Most transferred in")
            a.dataframe(view.nlargest(15, "Net transfers")[cols],
                        width="stretch", hide_index=True)
            b.caption("Most transferred out")
            b.dataframe(view.nsmallest(15, "Net transfers")[cols],
                        width="stretch", hide_index=True)

    with tabs[4]:
        st.markdown("**Build your own ranking.** Each metric is scaled 0–1 across the "
                    "filtered pool, then weighted. Negative weights penalise.")
        opts = ["Form", "Total pts", "xGI", "xGI per 90", "DEFCON per 90",
                "Exp pts next", "Minutes", "Pts per £m", "Owned %", "Price"]
        chosen = st.multiselect("Metrics", opts,
                                default=["Form", "xGI per 90", "Exp pts next", "Pts per £m"])
        if not chosen:
            st.info("Choose at least one metric.")
        else:
            w = {m: st.slider(m, -1.0, 1.0, 1.0, 0.1, key=f"w_{m}") for m in chosen}
            d = view.copy()
            d["Score"] = 0.0
            for m, weight in w.items():
                col = d[m].astype(float)
                rng = col.max() - col.min()
                d["Score"] += weight * (0 if rng == 0 else (col - col.min()) / rng)
            d["Score"] = d["Score"].round(3)
            st.dataframe(d.nlargest(30, "Score")[
                ["Player", "Team", "Pos", "Price", "Score"] + chosen + ["Status"]],
                width="stretch", hide_index=True, column_config=col_config(d))


def page_live(df, cur, squad):
    st.header("Live from the API")
    st.caption("Fetched directly from fantasy.premierleague.com. The static dashboard "
               "can't do this — browsers are blocked by CORS, a Python process isn't.")

    if st.button("Refresh"):
        st.cache_data.clear()

    try:
        boot = fetch_bootstrap()
        fixtures = fetch_fixtures()
    except Exception as exc:
        st.error(f"Couldn't reach the API: {exc}")
        return

    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    nxt = next((e for e in boot["events"] if e["is_next"]), None)
    if nxt:
        a, b = st.columns(2)
        a.metric("Next deadline", nxt["deadline_time"].replace("T", " ")[:16] + " UTC")
        b.metric("Gameweek", nxt["name"])

    st.subheader("Injuries and doubts")
    flagged = [{
        "Player": e["web_name"], "Team": teams.get(e["team"], "?"),
        "Price": e["now_cost"] / 10, "Owned %": float(e["selected_by_percent"]),
        "Chance": e["chance_of_playing_next_round"], "News": e["news"],
    } for e in boot["elements"] if e["status"] != "a" and float(e["selected_by_percent"]) > 1]
    fl = pd.DataFrame(flagged).sort_values("Owned %", ascending=False) if flagged \
        else pd.DataFrame()
    if squad and not fl.empty:
        ids = {e["web_name"] for e in boot["elements"] if e["id"] in squad}
        yours = fl[fl["Player"].isin(ids)]
        if not yours.empty:
            st.error("In your squad:")
            st.dataframe(yours, width="stretch", hide_index=True)
    st.dataframe(fl if not fl.empty else pd.DataFrame([{"": "Nobody flagged"}]),
                 width="stretch", hide_index=True)

    st.subheader("Price change watch")
    proj = []
    for e in boot["elements"]:
        p = (e.get("price_change_projections") or [{}])[0]
        pct = p.get("projected_percent")
        if pct in (None, "", "0"):
            continue
        proj.append({"Player": e["web_name"], "Team": teams.get(e["team"], "?"),
                     "Price": e["now_cost"] / 10, "Owned %": float(e["selected_by_percent"]),
                     "Progress %": float(pct), "Likelihood": p.get("likelihood")})
    if proj:
        pr = pd.DataFrame(proj)
        a, b = st.columns(2)
        a.caption("Closest to a rise"); b.caption("Closest to a fall")
        a.dataframe(pr.nlargest(15, "Progress %"), width="stretch", hide_index=True)
        b.dataframe(pr.nsmallest(15, "Progress %"), width="stretch", hide_index=True)
    else:
        st.info("No price movement projected right now.")

    st.subheader("Fixture ticker")
    horizon = st.slider("Gameweeks ahead", 1, 10, 6)
    start = nxt["id"] if nxt else 1
    grid: dict[str, dict] = {s: {"Team": s} for s in teams.values()}
    fdr: dict[str, list] = {s: [] for s in teams.values()}
    for f in fixtures:
        gw = f.get("event")
        if gw is None or not (start <= gw < start + horizon):
            continue
        h, a = teams.get(f["team_h"]), teams.get(f["team_a"])
        for side, opp, diff in ((h, a, f["team_h_difficulty"]),
                                (a, h, f["team_a_difficulty"])):
            cell = f"{opp} ({'H' if side == h else 'A'}) {diff}"
            key = f"GW{gw}"
            grid[side][key] = f"{grid[side][key]} + {cell}" if key in grid[side] else cell
            fdr[side].append(diff)
    rows = []
    for s, row in grid.items():
        row["Avg FDR"] = round(np.mean(fdr[s]), 2) if fdr[s] else None
        row["Games"] = len(fdr[s])
        rows.append(row)
    tick = pd.DataFrame(rows)
    order = ["Team", "Avg FDR", "Games"] + [f"GW{g}" for g in range(start, start + horizon)]
    st.dataframe(tick[[c for c in order if c in tick.columns]].sort_values("Avg FDR"),
                 width="stretch", hide_index=True)


@st.cache_data(ttl=900, show_spinner="Fetching from the FPL API…")
def fetch_bootstrap():
    r = requests.get(f"{API}/bootstrap-static/", timeout=30,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fixtures():
    r = requests.get(f"{API}/fixtures/", timeout=30,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("⚽ FPL Tracker")
    df = get_data()

    if df is None or df.empty:
        st.title("FPL Tracker")
        st.info(
            "No data loaded yet.\n\n"
            "Run `python fpl_tracker.py` to build `fpl_history.csv`, then pick a "
            "data source in the sidebar. You can also paste a raw GitHub URL or "
            "upload the file directly."
        )
        return

    squad = load_squad()
    cur = latest_with_deltas(df)
    cur = attach_projections(cur)
    snaps = sorted(df["Snapshot"].unique())

    st.sidebar.caption(
        f"**GW{int(cur['GW'].iloc[0])}** · {len(cur)} players\n\n"
        f"{len(snaps)} snapshot{'s' if len(snaps) != 1 else ''}\n\n"
        f"{snaps[0]} → {snaps[-1]}"
    )
    if len(snaps) == 1:
        st.sidebar.info("First snapshot. Trends and Movers fill in from next week.")

    # Real URL routing (browser back/forward, shareable links) via
    # st.navigation -- each page function is unchanged, just wrapped so it
    # can be called with no arguments the way st.Page requires.
    pages = {
        "Core": [
            st.Page(lambda: page_players(df, cur, squad), title="Players",
                    url_path="players", default=True),
            st.Page(lambda: page_player(df, cur, squad), title="Player detail",
                    url_path="player-detail"),
            st.Page(lambda: page_compare(df, cur, squad), title="Compare", url_path="compare"),
            st.Page(lambda: page_movers(df, cur, squad), title="Movers", url_path="movers"),
            st.Page(lambda: page_teams(df, cur, squad), title="Teams", url_path="teams"),
            st.Page(lambda: page_squad(df, cur, squad), title="My squad", url_path="my-squad"),
        ],
        "League & market": [
            st.Page(lambda: page_rivals(df, cur, squad), title="Rivals", url_path="rivals"),
            st.Page(lambda: page_signals(df, cur, squad), title="Signals", url_path="signals"),
            st.Page(lambda: page_odds(df, cur, squad), title="Fixture projections",
                    url_path="fixture-projections"),
        ],
        "Models": [
            st.Page(lambda: page_projections(df, cur, squad), title="Projections",
                    url_path="projections"),
            st.Page(lambda: page_models(df, cur, squad), title="Models", url_path="models"),
        ],
        "Explore": [
            st.Page(lambda: page_analysis(df, cur, squad), title="Analysis", url_path="analysis"),
            st.Page(lambda: page_live(df, cur, squad), title="Live", url_path="live"),
        ],
    }
    pg = st.navigation(pages, position="top")
    pg.run()


if __name__ == "__main__":
    main()