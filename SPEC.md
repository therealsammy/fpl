# Football Data Platform — Product Spec

**Repo:** `github.com/therealsammy/fpl`
**Status:** FPL pipeline live since GW1 2026/27. This spec extends it into a four-door platform.
**Date:** August 2026

---

## 1. The one-liner

An open football data archive with four front doors — FPL, the WSL, an event-data vault, and a 30-year odds ledger — where every chart is one click from being a shareable image, and every forecast gets scored in public.

---

## 2. Why now

In January 2026, Opta (Stats Perform) terminated Sports Reference's access to advanced soccer data and required immediate deletion. FBref's advanced stats are gone. The termination came shortly after Opta's parent became FIFA's official betting data distributor.

Three consequences:

1. **The community's single unified source vanished.** Everyone scattered across five feeds with incompatible IDs. Nobody has rebuilt the join.
2. **Women's football took the worst hit.** The FBref/Opta deal had specifically prioritised women's data. Its loss was described as a massive step back for women's advanced data access.
3. **Tools built on that feed died.** McLachApp is the cautionary example — a well-built, well-loved Streamlit app that could not survive its upstream disappearing.

The strategic lesson: **build only on feeds that cannot be revoked, and archive everything.**

---

## 3. Data sources and revocation risk

| Source | What it gives | Coverage | Risk | Notes |
|---|---|---|---|---|
| **FPL API** | Player-week points, BPS components, ICT, defensive contributions, ownership, price, `ep_next`, fixtures, any manager's squad | Premier League, current season | **None** | First-party. The Premier League runs it for their own game. No partner contract exists to terminate. |
| **Football-Data.co.uk** | Results + opening/closing odds | ~22 leagues, 1993–present | **Very low** | Static CSV downloads. Been up for decades. |
| **StatsBomb Open Data** | Full event data + 360 freeze-frames | Selected competitions, historical | **Low** | Explicit free licence for research/public use. Attribution required. Not a live feed. |
| **Understat** | Shot-level xG with coordinates | Top five men's leagues, 2014/15– | **Medium** | Scraped from embedded JSON. No licence. Could break or be blocked. |
| **Club Elo** | Daily Elo ratings | European clubs, 1939– | **Low** | Free API. |
| **FBref (basic only)** | Historic basic stats | 100+ competitions | **Medium** | Advanced data deleted Jan 2026. Basic data remains. Scraping is grey-area — rate-limit hard. |

**Design rule:** every Medium-risk source lives behind an isolated collector module that can fail without taking the app down. Anything it fetches is archived on first retrieval and never re-fetched for historical periods.

---

## 4. The spine

Everything shared by all four doors.

### 4.1 Collectors

One module per source. Each writes into the same snapshot format. Each is independently runnable, independently failable, and idempotent.

```
collectors/
  fpl.py             # extends existing fpl_tracker.py
  football_data.py   # results + odds, ~22 leagues
  statsbomb.py       # open-data repo mirror + cache
  understat.py       # shots, isolated, failure-tolerant
  clubelo.py         # optional
```

### 4.2 The archive

Append-only. Timestamped. Committed and published. This is the asset.

```
data/
  fpl/
    snapshots/YYYY-MM-DD.parquet     # DAILY (see §7)
    fpl_history.csv                  # existing, preserved
  odds/
    football_data/{league}/{season}.csv
  understat/
    shots/{league}/{season}.parquet
  statsbomb/                         # cached, gitignored if large
  ids/
    players.csv
    teams.csv
    unresolved.csv                   # human review queue
  forecasts/
    YYYY-MM-DD/{source}.parquet      # THE CROWN JEWEL — see below
  scores/
    scoreboard.parquet
```

**`forecasts/` is the most valuable directory in the repo.** It stores predictions *timestamped before the outcome was known* — FPL's `ep_next` for every player every week, the market's closing odds for every match, and your own model output. Nobody archives the first. Almost nobody scores the second properly. This directory is what makes a public forecast scoreboard possible, and it cannot be reconstructed retroactively.

### 4.3 ID resolution

The single hardest and most valuable piece of the spine. Everything downstream depends on it.

**`data/ids/players.csv`**

| column | notes |
|---|---|
| `canonical_id` | Stable internal ID, never reused |
| `display_name` | Human-facing |
| `fpl_id` | FPL API element id |
| `understat_id` | |
| `statsbomb_id` | |
| `birth_date` | Primary disambiguator |
| `confidence` | 0–1 |
| `method` | `exact` / `normalized` / `fuzzy` / `manual` |
| `verified_at` | Timestamp of human confirmation, null if unverified |

**`data/ids/teams.csv`** — same shape, plus `country`, `tier`, and the naming variants (`football_data_name` = "Man United", `understat_name` = "Manchester United", etc.).

**Matching cascade:**

1. Exact string match
2. Normalized match (unicode fold, strip accents and punctuation, lowercase)
3. Token-set fuzzy match, **constrained** by birth date, current team, and position
4. Anything below threshold → `unresolved.csv` for human review

**Hard rule:** never auto-accept a fuzzy match below threshold. A wrong join silently corrupts every downstream number and is nearly impossible to detect later. Unresolved rows are a normal, healthy output — not a failure.

### 4.4 Export

PNG download on every single chart. Branded, sized for social.

This is not a nice-to-have. McLachBot's entire distribution model was posting images — the pictures travelled, and the app got visited because of them. It is roughly half a day of work and it is most of the reach.

```
core/export.py    # matplotlib figure → branded PNG → st.download_button
```

---

## 5. The four doors

Four clearly distinct tools sharing a look, a data layer, and a house style.

---

### Door 1 — FPL

**Job:** the weekly habit. This is where repeat usage lives.
**Data:** FPL API (+ odds for clean sheet probabilities)
**Audience:** FPL managers — large, motivated, returning every week.

| Page | What it answers |
|---|---|
| **My Squad** | What's my projected score, who do I captain, what's my best transfer |
| **Players** | Browse, filter, sort — the workhorse |
| **Player Detail** | Full history, form, fixtures, price and ownership trajectory |
| **Compare** | Head-to-head on any metrics |
| **Flow** | Price and ownership movement over time — *requires daily snapshots* |
| **BPS Forensics** | Why did my player get zero bonus. Component-by-component breakdown. Includes **"Robbed Bonus"** — players who missed by 1–2 BPS |
| **DEFCON** | Defensive contribution tracker, threshold proximity, who's a machine |
| **Mini-League** | Paste any entry ID — squad card, captaincy history, rank trajectory, rival transfers |
| **Fixtures** | FDR ticker + clean sheet probabilities derived from odds |
| **Projections** | Your player-level expected points, published weekly, scored against `ep_next` |

**Differentiators no one else has:** the ownership/price time series (the FPL API never serves it retroactively), BPS component forensics, and the defensive-contribution coverage.

---

### Door 2 — WSL

**Job:** deep player-level modelling. The flagship for credibility.
**Data:** StatsBomb Open Data — FA WSL, Women's World Cup 2023
**Audience:** underserved and currently abandoned.

**Why this door is the sharpest wedge:** the FBref loss hit women's advanced data hardest, and the deal that died had specifically prioritised it. Meanwhile StatsBomb's free women's coverage is genuinely deep — full event data, multiple seasons. This is the one area where the *free* data is now better than what most people can otherwise get, the audience is underserved, and essentially nobody is building.

| Page | What it does |
|---|---|
| **Player Profiles** | Radars built from *raw events* — you define the metrics, not a vendor |
| **Shot Maps** | With xG, from event coordinates |
| **Possession Value** | Every action rated by how much it shifted scoring probability |
| **Passing Networks** | Team structure, average positions, link strength |
| **Match Explorer** | Any match, full event breakdown |
| **Careers** | Ageing curves, per-90 with proper shrinkage, "is this streak real" |

**On prediction here:** ageing curves, minutes/availability modelling, and regression-to-mean all work fine on historical event data. **Next-weekend match predictions do not** — there is no live feed. Be explicit about this in the UI rather than implying currency you don't have.

---

### Door 3 — The Vault

**Job:** the shareable, viral surface. A museum, not a feed.
**Data:** StatsBomb Open Data — historic World Cups, selected Champions League finals, Messi-era La Liga

| Page | What it does |
|---|---|
| **Match Story** | One iconic match, full treatment: xG race, passing networks, momentum swings, shot maps — as a single shareable card |
| **Possession Value Replay** | Watch the model rate a famous match action by action |
| **Legends** | Deep dives where coverage is thick — the Messi Barcelona data especially |
| **Eras** | Pressing, build-up, passing distance compared across decades — possible *only* because one provider's archive is consistent across eras |

**Why iconic matches specifically:** people already know what happened, so they can sanity-check your model's output. A possession-value model that says something surprising about a game everyone remembers is far more legible — and far more shareable — than the same model run on an anonymous mid-table fixture.

---

### Door 4 — The Ledger

**Job:** breadth, history, and the public scoreboard.
**Data:** Football-Data.co.uk — ~22 leagues, 1993–present, results + closing odds

| Page | What it does |
|---|---|
| **Elo Explorer** | Your own ratings, any league, any era; cross-league strength via European ties |
| **Title Races** | Live probabilities, plus what they were at every point in every past season |
| **Upsets** | Biggest shocks in history, ranked by implied probability |
| **On This Day** | Every match played on today's date since 1993 |
| **Forecast Scoreboard** | **The flagship.** See below |
| **Market Efficiency** | How good the closing line actually is, by league and era |

#### The Forecast Scoreboard

The thing that ties the whole platform together.

A permanently visible, weekly-updated public leaderboard where forecasts get graded on the same archive:

- Your match model
- The closing line (devigged)
- FPL's `ep_next`
- Your player projections
- Eventually: anyone else who wants to submit

Scored by log loss and RPS for match outcomes, MAE/calibration for player points. Sliced by league, position, price, and minutes certainty.

**Football has no public equivalent.** This is the page that turns the project from a dashboard into the scoreboard other people get measured on.

**The honest-result requirement:** you will probably lose to the closing line, certainly in the top five leagues. That is the expected outcome and it is fine — but only if you *first* demonstrate the model is well-calibrated and beats simple baselines (home advantage, Elo). Show calibration, show it beating baselines, *then* show the market beating you. Skip that ordering and it reads as "built a bad model" rather than "measured honestly against a hard benchmark."

---

## 6. Repo structure

```
fpl/
  collectors/
    fpl.py
    football_data.py
    statsbomb.py
    understat.py
  core/
    archive.py          # append-only write, schema enforcement
    ids.py              # resolution cascade
    export.py           # branded PNG
    theme.py            # shared house style
  models/
    minutes.py          # start probability
    prices.py           # price change prediction
    match.py            # Dixon-Coles / bivariate Poisson
    projections.py      # player expected points
    possession_value.py # action valuation
    elo.py
  validation/
    scoreboard.py       # the public leaderboard
  app/
    Home.py
    pages/
      1_FPL_Squad.py
      ...
      20_WSL_Players.py
      ...
      40_Vault_Match_Story.py
      ...
      60_Ledger_Scoreboard.py
  data/                 # the archive (§4.2)
  .github/workflows/
    daily.yml
    weekly.yml
  index.html            # existing static dashboard, preserved
```

---

## 7. Do this first

**Change the cron from weekly to daily.**

Prices and ownership move every day. A Tuesday-only snapshot permanently destroys the resolution needed for a price-change model — and every day that passes at weekly cadence is data that can never be recovered. It is a one-line change and it is the single highest-value item in this document.

Keep the existing Tuesday run for the heavier weekly work; add a light daily snapshot alongside it.

---

## 8. Build order

| # | Phase | Why here |
|---|---|---|
| 0 | **Daily cron** | One line. Every day of delay is unrecoverable data. |
| 1 | **`forecasts/` archiving** | Start capturing `ep_next` and closing odds immediately. Same logic — unrecoverable. |
| 2 | **ID resolution** | Everything cross-source depends on it. |
| 3 | **Football-Data collector** | Cheapest large win. Static CSVs, 30 years, 22 leagues. |
| 4 | **Door 4 (Ledger)** | Elo, upsets, title races — builds directly on #3. |
| 5 | **PNG export** | Half a day. Unlocks distribution for everything already built. |
| 6 | **Match model + Scoreboard** | Dixon-Coles, scored against closing. |
| 7 | **FPL new pages** | BPS forensics, DEFCON, Flow, Mini-League. |
| 8 | **Minutes model** | Multiplies everything on the FPL side. |
| 9 | **StatsBomb collector** | Sets up doors 2 and 3. |
| 10 | **Door 2 (WSL)** | The credibility flagship. |
| 11 | **Possession value** | Serves doors 2 and 3. |
| 12 | **Door 3 (Vault)** | The viral surface. |
| 13 | **Price model** | Needs ~3 months of daily snapshots first. |
| 14 | **FPL projections** | Behind the readiness gate. |

**The FPL side already works. Don't touch it until the second door is standing.**

---

## 9. Retained from the original seven phases

| Original | Fate |
|---|---|
| `fpl_minutes.py` | **Keep** — phase 8. Highest-leverage FPL model. |
| `fpl_odds.py` | **Promote** — now serves both FPL and the Ledger. Becomes `models/match.py` + odds collection. |
| `validate_projections.py` | **Promote** — becomes the public Forecast Scoreboard. The thing keeping you honest. |
| `fpl_projections.py` | **Keep** — phase 14, readiness gate intact. |
| `fpl_rivals.py` | **Fold in** — becomes the Mini-League page. |
| `fpl_signals.py` | **Fold in** — becomes Flow + alerts. |
| `fpl_defcon.py` | **Keep, deprioritise** — hypothesis test, report null results honestly. |

---

## 10. Honesty gates

Non-negotiable, carried from the original build:

1. **Projections readiness gate.** No player-level output until ≥6 snapshots, ≥70% minutes coverage, and fixture projections exist.
2. **Null results published.** If DEFCON shows no signal, the page says so.
3. **No implied currency.** WSL and Vault pages state plainly that data is historical, with the coverage window shown.
4. **Unresolved IDs surfaced,** not silently dropped.
5. **The scoreboard shows losses.** Including yours.

---

## 11. Open decisions

- **Which door launches publicly first?** Recommendation: FPL — it already works, and it has the only weekly habit.
- **Streamlit Cloud or self-hosted?** Cloud is free and was good enough for McLachApp; watch resource limits once StatsBomb event data is loaded.
- **Does the archive get its own read API,** or is the repo the API? Repo-as-API is free and sufficient to start.
- **Understat: include or skip at launch?** It's the only free live men's player-level data, and it's also the highest-risk source in the stack.

---

## 12. Expectation setting

The FPL door will show habitual weekly usage from a large audience. The other three will spike when something gets shared, then go quiet. Both patterns are fine — but don't judge doors 2–4 by daily actives, and don't let their quiet periods pull effort away from the archive.

The archive is the only thing that compounds. If three other people build something on it, the project stops being a site and becomes infrastructure — which is much harder to displace than a dashboard.
