"""
Shared House Style
====================
One palette for every chart in the app -- Altair on screen, matplotlib
in exported PNGs (core/export.py, Phase 5) -- so a downloaded image
looks like it came from the same product as the interactive page it
was pulled from. These are the same hex values app.py has used for its
Altair charts since the dataviz pass earlier in the project; this
module is now the single source of truth and app.py imports them
rather than redefining them.

Pitch styling (pitch()) is here ahead of Door 3 (Vault, SPEC.md Phase
12) actually needing it. Nothing calls it yet -- it's not a guess at a
future requirement invented for its own sake, it's the natural home
for it once match-event charts exist, per the repo layout SPEC.md
already lays out for core/theme.py.
"""

import matplotlib.pyplot as plt

# Categorical hues, fixed order -- never cycled or reassigned by filter.
CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                      "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]
DIVERGING_BLUE_RED = ["#104281", "#86b6ef", "#f0efec", "#f0a68a", "#d03b3b"]

STATUS_COLORS = {"good": "#0ca30c", "warning": "#fab219",
                 "serious": "#ec835a", "critical": "#d03b3b", "neutral": "#898781"}
STATUS_BADGE = {"good": "green", "warning": "orange",
                "serious": "orange", "critical": "red", "neutral": "gray"}

INK = "#1a1a1a"
MUTED_INK = "#6b6b6b"
SURFACE = "#ffffff"
GRID_COLOR = "#e4e4e4"

# DejaVu Sans ships with matplotlib itself -- renders identically
# wherever the app runs, with no dependency on a system font being
# installed (unlike relying on e.g. Helvetica or a Google Font).
FONT_FAMILY = "DejaVu Sans"

# No social handle or brand has been established anywhere else in the
# project (checked index.html, README) -- reusing the one piece of
# branding that already exists (st.set_page_config's page_title /
# the sidebar title) rather than inventing a handle that doesn't exist.
SITE_NAME = "FPL Tracker"


def apply() -> None:
    """Sets matplotlib's rcParams for every exported chart. Idempotent
    and cheap -- safe to call at the top of every chart-building
    function rather than once globally, so export functions don't
    depend on import order."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "text.color": INK,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": MUTED_INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


# ---------------------------------------------------------------------------
# PITCH STYLING (Door 3 / Vault -- not consumed anywhere yet)
# ---------------------------------------------------------------------------

PITCH_LENGTH = 105.0   # metres, standard FIFA pitch
PITCH_WIDTH = 68.0
PITCH_LINE_COLOR = "#4a4a4a"
PITCH_FILL_COLOR = "#f4f6f2"


def pitch(ax) -> None:
    """Draws a standard football pitch outline (full pitch markings:
    touchlines, halfway line, centre circle, penalty areas, six-yard
    boxes, penalty spots) on a matplotlib Axes in real-world metres, so
    x/y coordinates from StatsBomb-style event data plot directly onto
    it with no unit conversion. Orientation: (0, 0) at the bottom-left
    corner flag, attacking left-to-right."""
    L, W = PITCH_LENGTH, PITCH_WIDTH
    ax.set_facecolor(PITCH_FILL_COLOR)

    def rect(x, y, w, h):
        ax.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y],
                color=PITCH_LINE_COLOR, linewidth=1.2, zorder=2)

    rect(0, 0, L, W)                                    # touchlines
    ax.plot([L / 2, L / 2], [0, W], color=PITCH_LINE_COLOR, linewidth=1.2, zorder=2)
    ax.add_patch(plt.Circle((L / 2, W / 2), 9.15, fill=False,
                            color=PITCH_LINE_COLOR, linewidth=1.2, zorder=2))
    ax.plot(L / 2, W / 2, marker="o", markersize=2, color=PITCH_LINE_COLOR, zorder=2)

    for x0, direction in [(0, 1), (L, -1)]:
        rect(x0 if direction == 1 else x0 - 16.5, (W - 40.32) / 2, 16.5, 40.32)   # penalty area
        rect(x0 if direction == 1 else x0 - 5.5, (W - 18.32) / 2, 5.5, 18.32)     # six-yard box
        spot_x = x0 + direction * 11
        ax.plot(spot_x, W / 2, marker="o", markersize=2, color=PITCH_LINE_COLOR, zorder=2)

    ax.set_xlim(-2, L + 2)
    ax.set_ylim(-2, W + 2)
    ax.set_aspect("equal")
    ax.axis("off")
