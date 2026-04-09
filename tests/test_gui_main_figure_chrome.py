from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from ra_sim.gui import main_figure_chrome


class _FakeWidget:
    def __init__(self):
        self.configure_calls = []

    def configure(self, **kwargs):
        self.configure_calls.append(dict(kwargs))


def test_configure_matplotlib_canvas_widget_removes_visible_widget_border() -> None:
    widget = _FakeWidget()

    main_figure_chrome.configure_matplotlib_canvas_widget(widget)

    assert widget.configure_calls == [
        {"highlightthickness": 0},
        {"borderwidth": 0},
        {"bd": 0},
        {"relief": "flat"},
    ]


def test_configure_main_figure_layout_docks_full_height_colorbar_and_clears_title() -> None:
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    image_artist = ax.imshow(np.zeros((4, 4), dtype=float))

    colorbar_main, caked_cbar_ax, caked_colorbar = (
        main_figure_chrome.configure_main_figure_layout(
            fig,
            ax,
            image_artist,
        )
    )

    assert ax.get_title() == ""
    assert ax.get_position().bounds == pytest.approx(
        main_figure_chrome.MAIN_IMAGE_AX_BOUNDS
    )
    assert colorbar_main.ax.get_position().bounds == pytest.approx(
        main_figure_chrome.MAIN_COLORBAR_AX_BOUNDS
    )
    assert caked_cbar_ax.get_position().bounds == pytest.approx(
        main_figure_chrome.MAIN_COLORBAR_AX_BOUNDS
    )
    assert caked_cbar_ax.get_visible() is False
    assert caked_colorbar.ax.get_visible() is False
    assert colorbar_main.ax.get_position().y0 == pytest.approx(ax.get_position().y0)
    assert colorbar_main.ax.get_position().height == pytest.approx(
        ax.get_position().height
    )
    assert colorbar_main.ax.get_position().x0 > ax.get_position().x1
    assert colorbar_main.ax.yaxis.get_ticks_position() == "right"
    assert colorbar_main.ax.yaxis.get_label_position() == "right"


def test_apply_main_figure_axes_chrome_hides_spines_and_keeps_left_bottom_ticks() -> None:
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_title("Simulated Diffraction Pattern")

    main_figure_chrome.apply_main_figure_axes_chrome(ax)

    assert ax.get_title() == ""
    assert all(not spine.get_visible() for spine in ax.spines.values())
    assert ax.xaxis.get_ticks_position() == "bottom"
    assert ax.yaxis.get_ticks_position() == "left"
