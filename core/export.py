"""
Branded PNG Export
=====================
Every interactive Altair chart in the app gets a matching downloadable
PNG: a title block, an attribution line naming the data's real source,
and the site name, sized for social (16:9 or 1:1). Rendered with
matplotlib (core/theme.py) server-side -- entirely separate from the
on-screen Altair chart, since Altair/vega-lite has no dependency-free
way to rasterize to PNG on the server, while matplotlib always does.

The three builders below (bar_png, line_png, scatter_png) mirror
app.py's existing Altair helpers (line(), scatter()) closely enough
that most call sites just pass the same data and column names to both.

Attribution note for future collectors: StatsBomb's terms require a
specific credit line wherever their data is shown (SPEC.md, BRIEFS.md
Phase 5) -- nothing in this repo uses StatsBomb data yet (that's Phase
9+), so `source` here is always a plain string the caller provides,
not a StatsBomb-specific special case. Whichever collector eventually
plots StatsBomb-derived numbers should pass its required credit line
through this same parameter.
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import theme

ASPECT_SIZES = {"16:9": (10.0, 5.625), "1:1": (8.0, 8.0)}


def new_figure(aspect: str = "16:9"):
    """A themed Figure+Axes with header/footer margin already reserved
    for to_png() to fill in -- callers just draw the chart into `ax`."""
    theme.apply()
    figsize = ASPECT_SIZES.get(aspect, ASPECT_SIZES["16:9"])
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.12, right=0.95)
    return fig, ax


def to_png(fig, title: str, subtitle: str | None = None, source: str | None = None,
          dpi: int = 200) -> bytes:
    """Fills the header/footer margin new_figure() reserved with a
    title block and an attribution footer, then rasterizes to PNG
    bytes ready for st.download_button. Does not change figure size --
    build the Figure at the target aspect ratio via new_figure() first."""
    fig.text(0.12, 0.94, title, fontsize=15, fontweight="bold", color=theme.INK, ha="left")
    if subtitle:
        fig.text(0.12, 0.87, subtitle, fontsize=10.5, color=theme.MUTED_INK, ha="left")

    footer = theme.SITE_NAME
    if source:
        footer = f"{source}  ·  {footer}"
    fig.text(0.12, 0.045, footer, fontsize=8.5, color=theme.MUTED_INK, ha="left")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def bar_png(categories, values, *, title: str, subtitle: str | None = None,
           source: str | None = None, x_label: str | None = None,
           color=None, aspect: str = "16:9") -> bytes:
    """Horizontal bar chart -- categories top-to-bottom in the order
    given (callers sort beforehand, same convention as the Altair
    bars elsewhere in the app)."""
    fig, ax = new_figure(aspect)
    y_pos = np.arange(len(categories))
    bar_colors = color if isinstance(color, list) else (color or theme.CATEGORICAL_COLORS[0])
    ax.barh(y_pos, values, color=bar_colors, height=0.65, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()   # first category at the top, matching Altair's sort="-x" bars
    if x_label:
        ax.set_xlabel(x_label)
    ax.grid(axis="y", visible=False)
    return to_png(fig, title, subtitle, source)


def line_png(data: pd.DataFrame, x_col: str, y_col: str, *, title: str,
            group_col: str | None = None, subtitle: str | None = None,
            source: str | None = None, x_label: str | None = None,
            y_label: str | None = None, aspect: str = "16:9") -> bytes:
    """Line chart over `x_col`, one line per distinct value of
    `group_col` if given (else a single line) -- mirrors app.py's
    line(data, metric, color) Altair helper."""
    fig, ax = new_figure(aspect)
    groups = list(data[group_col].unique()) if group_col else [None]
    for i, g in enumerate(groups):
        sub = data[data[group_col] == g] if group_col else data
        sub = sub.sort_values(x_col)
        color = theme.CATEGORICAL_COLORS[i % len(theme.CATEGORICAL_COLORS)]
        ax.plot(sub[x_col], sub[y_col], marker="o", markersize=3.5,
               linewidth=2, color=color, label=g, zorder=3)
    ax.set_xlabel(x_label or "")
    ax.set_ylabel(y_label or y_col)
    if group_col and len(groups) > 1:
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    fig.autofmt_xdate(rotation=30)
    return to_png(fig, title, subtitle, source)


def _marker_sizes(data: pd.DataFrame, size_col: str | None, default: float = 28) -> float | np.ndarray:
    """Scales a column to a readable bubble-size range (30-260 pt^2).
    A constant column (or all-equal values) falls back to a flat size
    rather than dividing by a zero range."""
    if not size_col or size_col not in data.columns:
        return default
    vals = data[size_col].astype(float)
    span = vals.max() - vals.min()
    if span == 0:
        return 90.0
    return 30 + (vals - vals.min()) / span * 230


def scatter_png(data: pd.DataFrame, x_col: str, y_col: str, *, title: str,
               color_col: str | None = None, color_map: dict | None = None,
               size_col: str | None = None, diagonal: bool = False,
               subtitle: str | None = None, source: str | None = None,
               x_label: str | None = None, y_label: str | None = None,
               aspect: str = "16:9") -> bytes:
    """Scatter plot, optionally colored by a categorical column and/or
    sized by a numeric one, with an optional y=x reference line --
    mirrors app.py's scatter() Altair helper."""
    fig, ax = new_figure(aspect)
    if color_col and color_col in data.columns:
        groups = list(data[color_col].unique())
        palette = color_map or {g: theme.CATEGORICAL_COLORS[i % len(theme.CATEGORICAL_COLORS)]
                                for i, g in enumerate(groups)}
        for g in groups:
            sub = data[data[color_col] == g]
            ax.scatter(sub[x_col], sub[y_col], s=_marker_sizes(sub, size_col), alpha=0.75,
                      color=palette.get(g, theme.CATEGORICAL_COLORS[0]), label=g, zorder=3)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    else:
        ax.scatter(data[x_col], data[y_col], s=_marker_sizes(data, size_col), alpha=0.75,
                  color=theme.CATEGORICAL_COLORS[0], zorder=3)
    if diagonal and not data.empty:
        hi = max(data[x_col].max(), data[y_col].max())
        lo = min(data[x_col].min(), data[y_col].min(), 0)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color=theme.MUTED_INK, linewidth=1, zorder=2)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    return to_png(fig, title, subtitle, source)
