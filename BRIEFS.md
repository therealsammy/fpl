# Claude Code Briefs

Companion to `SPEC.md`. One brief per phase, in build order. Each is self-contained enough to hand over on its own.

**Repo:** `github.com/therealsammy/fpl`
**Existing files to preserve:** `fpl_tracker.py`, `app.py`, `index.html`, `fpl_history.csv`, `.github/workflows/tracker.yml`

**Global rules for every phase:**
- Never break the existing FPL pipeline. It is live and its output is irreplaceable.
- All new data writes are append-only. Never overwrite or rewrite history.
- Every collector is idempotent — re-running for the same date produces the same result.
- Fail loudly on schema drift, quietly on network errors (retry, then skip).
- No new CDN dependencies in the static dashboard.

---

## Phase 0 — Daily snapshots

**Priority: do this before reading the rest of this document.**

**Goal:** capture FPL state daily instead of weekly.

**Why:** prices and ownership move every day. Weekly snapshots make a price-change model impossible and permanently destroy resolution. Every day at the current cadence is unrecoverable.

**Tasks:**
1. Add `.github/workflows/daily.yml` — runs 06:00 UTC daily.
2. Light snapshot only: `bootstrap-static/` elements → price, ownership, form, `ep_next`, transfers in/out, status.
3. Write to `data/fpl/snapshots/YYYY-MM-DD.parquet`.
4. Keep `tracker.yml` (Tuesday) for the full heavy run, unchanged.
5. Reuse the existing `_entry_id()` validation helper — GitHub Actions sets missing secrets to empty strings, not unset.

**Acceptance:** two consecutive days produce two files with identical schema and different values.

---

## Phase 1 — Forecast archiving

**Goal:** start storing predictions timestamped before outcomes are known.

**Why:** this is the most valuable data in the project and it cannot be reconstructed retroactively. FPL's `ep_next` is overwritten every week and never served historically.

**Tasks:**
1. `core/archive.py` — `write_forecast(source, as_of, df)` → `data/forecasts/YYYY-MM-DD/{source}.parquet`.
2. Hook into the daily job: archive `ep_next` for every player, every day, with the gameweek it refers to.
3. Archive closing odds for upcoming fixtures once the odds collector exists (Phase 3).
4. Schema: `as_of`, `target_event`, `entity_id`, `entity_type`, `metric`, `value`, `source`.

**Acceptance:** a week of `ep_next` forecasts is on disk with the gameweek each refers to, and re-running a day is a no-op.

---

## Phase 2 — ID resolution

**Goal:** a maintained crosswalk between FPL, Understat, StatsBomb, and Football-Data naming.

**Why:** the single hardest piece of the spine. Everything cross-source depends on it. A wrong join silently corrupts every downstream number and is nearly undetectable later.

**Tasks:**
1. `core/ids.py` implementing the cascade:
   - exact → normalized (unicode fold, strip accents/punctuation, lowercase) → constrained fuzzy (token-set, gated on birth date + team + position) → unresolved
2. Write `data/ids/players.csv` and `data/ids/teams.csv` per the schema in SPEC §4.3.
3. Write everything below threshold to `data/ids/unresolved.csv` for human review.
4. Provide `resolve(name, source, hints) -> canonical_id | None`.
5. Teams first — there are ~20 per league and they're mostly a fixed lookup. Players second.

**Hard constraint:** never auto-accept a fuzzy match below threshold. Unresolved rows are a normal output, not a failure. Report the count.

**Acceptance:** every current Premier League team resolves across all four sources. Player resolution rate reported honestly, with unresolved names listed.

---

## Phase 3 — Football-Data collector

**Goal:** ~22 leagues, results and odds, 1993–present.

**Why:** cheapest large win in the project. Static CSVs, no scraping, decades of history including closing odds.

**Tasks:**
1. `collectors/football_data.py` — download per-league per-season CSVs.
2. Normalize columns across eras (schema changed over the years — older files have fewer odds columns).
3. Map team names through `core/ids.py`.
4. Write to `data/odds/football_data/{league}/{season}.csv`.
5. Devig closing odds to implied probabilities (overround removal — use both the multiplicative and Shin methods, store both).
6. Historical seasons fetched once and never re-fetched.

**Acceptance:** full history for at least the top two English tiers plus the big five European leagues, with devigged probabilities summing to 1.

---

## Phase 4 — Door 4: The Ledger

**Goal:** the history and breadth pages.

**Tasks:**
1. `models/elo.py` — standard Elo with configurable K and home advantage, computed from results alone. Cross-league linking via European ties where fixtures exist.
2. Pages:
   - `Ledger_Elo.py` — ratings over time, any league, any era
   - `Ledger_Title_Races.py` — Monte Carlo season simulation from current ratings; for past seasons, the probability path at every matchday
   - `Ledger_Upsets.py` — results ranked by implied probability of the winner
   - `Ledger_On_This_Day.py` — every match on today's date since 1993
   - `Ledger_Market_Efficiency.py` — closing-line calibration by league and era

**Acceptance:** Elo ratings are sane (top clubs at the top), title race probabilities sum to 1, and On This Day returns results for an arbitrary date.

---

## Phase 5 — PNG export

**Goal:** every chart one click from a shareable image.

**Why:** this was McLachBot's entire distribution model. Roughly half a day of work for most of the reach.

**Tasks:**
1. `core/theme.py` — shared matplotlib style: fonts, palette, pitch styling.
2. `core/export.py` — `to_png(fig, title, subtitle)` returning branded bytes: title block, source attribution (StatsBomb attribution is licence-required), site handle, sized for social.
3. `st.download_button` on every chart in the app.
4. Standard aspect ratios: 16:9 and 1:1.

**Acceptance:** any chart in any door downloads as a branded PNG with correct attribution.

---

## Phase 6 — Match model + Forecast Scoreboard

**Goal:** a match outcome model, and the public leaderboard that grades it.

**Tasks:**
1. `models/match.py` — Dixon-Coles with time-decayed weighting and the low-score correlation term. Bivariate Poisson as an alternative. Produce 1X2, over/under, both-teams-to-score, and clean sheet probabilities.
2. Archive every prediction via Phase 1 before kickoff.
3. `validation/scoreboard.py` — score all archived forecasts: log loss and RPS for match outcomes, MAE and calibration for player points.
4. `Ledger_Scoreboard.py` — the flagship page. Your model, the closing line, `ep_next`, and simple baselines (home advantage, Elo), all graded on the same archive, updated weekly, permanently visible.

**Presentation order is mandatory:** show calibration first, then the model beating simple baselines, *then* the market beating the model. That ordering is the difference between "measured honestly against a hard benchmark" and "built a bad model."

**Acceptance:** the scoreboard renders with at least the baselines and closing line scored, and reports negative results without softening them.

---

## Phase 7 — FPL new pages

**Goal:** the differentiated FPL surface.

**Tasks:**
1. `FPL_BPS_Forensics.py` — full BPS component breakdown per player per match. **"Robbed Bonus"** leaderboard: players who missed a bonus point by 1–2 BPS.
2. `FPL_DEFCON.py` — defensive contribution tracking, threshold proximity, league-wide leaderboards.
3. `FPL_Flow.py` — price and ownership movement over time from the daily snapshots. Animated where it helps. Template drift, ownership cliffs.
4. `FPL_Mini_League.py` — paste any entry ID: squad card, captaincy history, rank trajectory, transfer log. Works for any manager, not just yours.

**Acceptance:** BPS components reconcile against actual awarded bonus for a sampled gameweek.

---

## Phase 8 — Minutes model

**Goal:** start probability per player per gameweek.

**Why:** multiplies everything downstream on the FPL side. Highest-leverage single model in the project.

**Tasks:**
1. `models/minutes.py` — features: recent minutes pattern, rotation history, fixture congestion, `status` and `chance_of_playing`, price tier, positional competition.
2. Output a calibrated probability, not a binary.
3. Archive predictions via Phase 1.
4. Score on the Scoreboard.

**Known limitation to state in the UI:** Friday press conferences are the single most important rotation-risk signal and no API field captures them. The model is a prior, not an oracle.

---

## Phase 9 — StatsBomb collector

**Goal:** local mirror and cache of the open-data repo.

**Tasks:**
1. `collectors/statsbomb.py` — use `statsbombpy`, or read the GitHub JSON directly.
2. Enumerate available competitions/seasons from `competitions.json`; detect 360 availability.
3. Cache locally; gitignore if size is a problem. Cache once, never re-fetch.
4. Normalize events into a flat table via `core/ids.py`.
5. **Attribution is licence-required** — surface it in the app and on every exported PNG.

**Acceptance:** competition inventory printed with match counts and 360 coverage flags; one full match loads and parses.

---

## Phase 10 — Door 2: WSL

**Goal:** the deep player-level door.

**Tasks:**
1. `WSL_Players.py` — radars built from raw events, metrics defined by you, with per-90 shrinkage toward positional means.
2. `WSL_Shot_Maps.py` — coordinates + xG.
3. `WSL_Passing_Networks.py` — average positions, link strength, by match or aggregated.
4. `WSL_Match_Explorer.py` — any match, full event breakdown.
5. `WSL_Careers.py` — ageing curves. **Correct for survivorship**: players who decline leave the sample, so naive delta-method curves are biased upward. Note the correction in the UI.

**Mandatory UI element:** state the coverage window plainly on every page. This data is historical. Do not imply currency you don't have.

---

## Phase 11 — Possession value

**Goal:** rate every action by how much it changed scoring probability.

**Tasks:**
1. `models/possession_value.py` — VAEP-style: model P(score) and P(concede) in the next N actions, value each action as the change in the difference.
2. Consider `socceraction` rather than writing from scratch.
3. Where 360 frames exist, add defensive positioning features — most public work ignores this, and it's the clearest available edge.
4. Validate: does action value predict next-season goal contribution better than xG alone?

**Honesty note:** outcome labels are extremely noisy and cross-competition calibration is poor. Report uncertainty. Don't present a single number as truth.

---

## Phase 12 — Door 3: The Vault

**Goal:** the shareable surface.

**Tasks:**
1. `Vault_Match_Story.py` — one iconic match, full treatment in a single scrollable page ending in one shareable card: xG race, momentum, passing networks, shot maps, key moments.
2. `Vault_Possession_Value.py` — action-by-action model output over a famous match.
3. `Vault_Legends.py` — deep dives where coverage is thick, especially Messi-era Barcelona.
4. `Vault_Eras.py` — pressing, build-up, passing distance compared across decades. Possible only because one provider's archive is consistent across eras.

**Design note:** iconic matches are chosen deliberately. People already know what happened, so they can sanity-check the model — which makes surprising output legible and shareable rather than suspicious.

---

## Phase 13 — Price model

**Gate: do not start until ~3 months of daily snapshots exist.**

**Goal:** predict FPL price rises and falls.

**Tasks:**
1. `models/prices.py` — features: net transfers, ownership, transfer velocity, days since last change, price relative to purchase.
2. Your archive is the only possible training set. Nobody can reproduce it.
3. Archive predictions; score on the Scoreboard.

**Acceptance:** beats the naive "high net transfers → rise" heuristic on held-out days.

---

## Phase 14 — FPL projections

**Gate intact from the original build: ≥6 snapshots, ≥70% minutes coverage, and fixture projections must exist before any output is produced.**

**Goal:** player-level expected points, published weekly, targeting `ep_next`.

**Tasks:**
1. `models/projections.py` — combine minutes model (Phase 8), clean sheet probabilities from the match model (Phase 6), attacking returns, and BPS expectation.
2. Output distributions, not point estimates.
3. Archive every projection before deadline.
4. Score against `ep_next` on the Scoreboard.

**Expected outcome:** beating `ep_next` by a small margin, with a season of ~30 scored gameweeks giving very little power to distinguish real skill from luck. Say so on the page. Year one is likely inconclusive, and that is the correct thing to report.

---

## Deferred

- `fpl_signals.py` — set-piece change alerts, fixture-adjusted form. Fold into Phase 7 later.
- `fpl_defcon.py` hypothesis test — keep, deprioritise, report null results honestly.
- Understat collector — highest-risk source. Add only when the rest is stable, fully isolated behind a failure-tolerant module.
- Public read API for the archive — repo-as-API is sufficient until someone asks.
