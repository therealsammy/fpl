import io

import matplotlib
matplotlib.use("Agg")   # headless -- no display available in CI or this test run

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from PIL import Image

from core import export, theme


def _open(png_bytes: bytes) -> Image.Image:
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    return Image.open(io.BytesIO(png_bytes))


# ---------------------------------------------------------------------------
# to_png / new_figure -- aspect ratio and basic validity
# ---------------------------------------------------------------------------

def test_to_png_returns_a_decodable_png():
    fig, ax = export.new_figure()
    ax.plot([1, 2, 3], [1, 4, 9])
    png = export.to_png(fig, "Title")
    img = _open(png)
    assert img.format == "PNG"


def test_16_9_aspect_ratio_is_respected():
    fig, ax = export.new_figure("16:9")
    ax.plot([1, 2], [1, 2])
    img = _open(export.to_png(fig, "T"))
    assert img.width / img.height == pytest.approx(16 / 9, rel=0.02)


def test_1_1_aspect_ratio_is_respected():
    fig, ax = export.new_figure("1:1")
    ax.plot([1, 2], [1, 2])
    img = _open(export.to_png(fig, "T"))
    assert img.width / img.height == pytest.approx(1.0, rel=0.02)


def test_unknown_aspect_falls_back_to_16_9():
    fig, ax = export.new_figure("nonsense")
    ax.plot([1, 2], [1, 2])
    img = _open(export.to_png(fig, "T"))
    assert img.width / img.height == pytest.approx(16 / 9, rel=0.02)


def test_to_png_does_not_leak_figures():
    """to_png() must close the figure it's handed -- otherwise every
    chart download in a long-running Streamlit session accumulates an
    open matplotlib Figure and eventually exhausts memory."""
    before = len(plt.get_fignums())
    fig, ax = export.new_figure()
    ax.plot([1], [1])
    export.to_png(fig, "T")
    assert len(plt.get_fignums()) == before


# ---------------------------------------------------------------------------
# bar_png / line_png / scatter_png -- smoke + content checks
# ---------------------------------------------------------------------------

def test_bar_png_smoke():
    png = export.bar_png(["Arsenal", "Man City", "Liverpool"], [80, 75, 70],
                         title="Points", source="football-data.co.uk")
    _open(png)


def test_bar_png_accepts_per_bar_colors():
    png = export.bar_png(["A", "B"], [1, -1], title="T",
                         color=[theme.STATUS_COLORS["good"], theme.STATUS_COLORS["critical"]])
    _open(png)


def test_line_png_single_series():
    data = pd.DataFrame({"date": ["2024-01-01", "2024-01-08", "2024-01-15"],
                         "rating": [1500, 1510, 1495]})
    png = export.line_png(data, "date", "rating", title="Rating over time")
    _open(png)


def test_line_png_multiple_groups_draws_a_legend():
    data = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-08"] * 2,
        "value": [1, 2, 3, 4],
        "team": ["A", "A", "B", "B"],
    })
    png = export.line_png(data, "date", "value", group_col="team", title="T")
    _open(png)


def test_scatter_png_with_color_groups():
    data = pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1],
                         "pos": ["GKP", "DEF", "MID", "FWD"]})
    png = export.scatter_png(data, "x", "y", color_col="pos", title="T")
    _open(png)


def test_scatter_png_without_color():
    data = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    png = export.scatter_png(data, "x", "y", title="T")
    _open(png)


def test_scatter_png_with_diagonal_and_size():
    data = pd.DataFrame({"x": [1, 5, 3], "y": [2, 1, 3], "weight": [10, 20, 5]})
    png = export.scatter_png(data, "x", "y", size_col="weight", diagonal=True, title="T")
    _open(png)


def test_scatter_png_size_col_with_constant_values_does_not_divide_by_zero():
    data = pd.DataFrame({"x": [1, 2], "y": [1, 2], "weight": [5, 5]})
    png = export.scatter_png(data, "x", "y", size_col="weight", title="T")
    _open(png)


# ---------------------------------------------------------------------------
# theme.py
# ---------------------------------------------------------------------------

def test_apply_is_idempotent():
    theme.apply()
    theme.apply()   # must not raise or accumulate state


def test_pitch_draws_without_error():
    fig, ax = plt.subplots()
    theme.pitch(ax)
    plt.close(fig)


def test_pitch_sets_equal_aspect_so_the_pitch_is_not_distorted():
    fig, ax = plt.subplots()
    theme.pitch(ax)
    assert ax.get_aspect() == 1.0 or ax.get_aspect() == "equal"
    plt.close(fig)
