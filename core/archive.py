"""
Forecast Archive
==================
The single most valuable data this project produces: predictions
timestamped BEFORE the outcome they predict is known. FPL's `ep_next` is
overwritten every week and the API never serves it historically -- once
today's value is gone, it is gone forever, and no amount of later work can
reconstruct it. This module exists to stop that loss.

`write_forecast()` is deliberately narrow: it takes source-specific
forecast rows, stamps them with a common schema, and writes ONE file per
(as_of date, source). It never touches any other day's file -- the
archive as a whole is append-only across days, even though a same-day
re-run overwrites that day's own file with a fresh snapshot (the same
idempotent-same-day convention used by every collector in this project).

Schema (SPEC.md Section 4.2):
    as_of         the date the forecast was MADE, YYYY-MM-DD
    target_event  what future event/gameweek/fixture this forecast is about
    entity_id     the player, team, or fixture id being forecast
    entity_type   "player" | "team" | "fixture"
    metric        which metric this is a forecast OF ("ep_next", "1x2", ...)
    value         the forecast value itself
    source        who/what produced it ("fpl_ep_next", "closing_odds", ...)
"""

from pathlib import Path

import pandas as pd

ARCHIVE_ROOT = Path("data/forecasts")

# Caller-supplied columns. as_of and source come from this function's own
# arguments, never from the frame -- see write_forecast's docstring for why.
REQUIRED_COLUMNS = ["target_event", "entity_id", "entity_type", "metric", "value"]
SCHEMA_COLUMNS = ["as_of", "target_event", "entity_id", "entity_type", "metric", "value", "source"]


def write_forecast(source: str, as_of: str, df: pd.DataFrame) -> Path:
    """
    Writes `df` to data/forecasts/{as_of}/{source}.parquet.

    `df` must already carry target_event, entity_id, entity_type, metric,
    and value -- everything specific to this forecast. `as_of` and
    `source` are supplied as arguments and stamped onto every row here,
    overwriting anything already in those columns -- so the file's path
    and its contents can never disagree about which day or which source
    a forecast belongs to.

    Fails loudly (raises KeyError) if a required column is missing --
    schema enforcement per SPEC.md's description of this module. A
    forecast archive with the wrong shape is worse than no archive at
    all, because it looks trustworthy and isn't.

    Idempotent same-day: overwrites that day's own file on re-run rather
    than appending or duplicating. Never touches any other day's file.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"write_forecast({source!r}): missing required column(s) {missing}. "
            f"Fix the caller rather than writing a partial row.")

    stamped = df.copy()
    stamped["as_of"] = as_of
    stamped["source"] = source
    stamped = stamped[SCHEMA_COLUMNS]

    out_dir = ARCHIVE_ROOT / as_of
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source}.parquet"
    stamped.to_parquet(out_path, index=False)
    return out_path


def read_forecasts(source: str | None = None, as_of: str | None = None) -> pd.DataFrame:
    """
    Reads archived forecasts back, optionally filtered to one source
    and/or one as_of date. An empty archive (or no match) returns an
    empty, correctly-shaped DataFrame rather than raising -- that's a
    normal state (e.g. before the first day has run), not an error.
    """
    pattern = f"{as_of or '*'}/{source or '*'}.parquet"
    paths = sorted(ARCHIVE_ROOT.glob(pattern))
    if not paths:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
