"""Helpers for the main 2D diffraction figure layout and chrome."""

from __future__ import annotations

from typing import Any

MAIN_IMAGE_AX_BOUNDS = (0.065, 0.075, 0.845, 0.90)
MAIN_COLORBAR_AX_BOUNDS = (0.93, 0.075, 0.023, 0.90)


def _safe_widget_configure(widget: object, **kwargs: object) -> None:
    configure = getattr(widget, "configure", None)
    if not callable(configure):
        return
    try:
        configure(**kwargs)
    except Exception:
        pass


def configure_matplotlib_canvas_widget(widget: object) -> None:
    """Remove visible widget chrome from the embedded TkAgg canvas host."""

    _safe_widget_configure(widget, highlightthickness=0)
    _safe_widget_configure(widget, borderwidth=0)
    _safe_widget_configure(widget, bd=0)
    _safe_widget_configure(widget, relief="flat")


def set_main_figure_axes_axis_visibility(ax: Any, *, visible: bool) -> None:
    """Toggle visible left/bottom axes for angle-space views."""

    visible = bool(visible)
    spines = getattr(ax, "spines", {})
    for name, spine in getattr(spines, "items", lambda: ())():
        try:
            spine.set_visible(visible and name in {"bottom", "left"})
        except Exception:
            continue

    try:
        ax.tick_params(
            axis="both",
            which="both",
            direction="out",
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelbottom=True,
            labeltop=False,
            labelleft=True,
            labelright=False,
            pad=2,
        )
    except Exception:
        pass

    xaxis = getattr(ax, "xaxis", None)
    if xaxis is not None:
        try:
            xaxis.set_ticks_position("bottom")
        except Exception:
            pass
    yaxis = getattr(ax, "yaxis", None)
    if yaxis is not None:
        try:
            yaxis.set_ticks_position("left")
        except Exception:
            pass


def apply_main_figure_axes_chrome(ax: Any, *, axes_visible: bool = False) -> None:
    """Apply the main-image axes styling for detector or angle-space views."""

    fig = getattr(ax, "figure", None)
    if fig is not None:
        try:
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        except Exception:
            pass
        try:
            fig.patch.set_edgecolor("none")
            fig.patch.set_linewidth(0.0)
        except Exception:
            pass

    try:
        ax.set_position(MAIN_IMAGE_AX_BOUNDS)
    except Exception:
        pass
    try:
        ax.set_aspect("auto")
    except Exception:
        pass
    try:
        ax.set_title("")
    except Exception:
        pass
    try:
        ax.patch.set_edgecolor("none")
        ax.patch.set_linewidth(0.0)
    except Exception:
        pass

    set_main_figure_axes_axis_visibility(ax, visible=axes_visible)


def _apply_colorbar_chrome(colorbar: Any) -> None:
    try:
        colorbar.ax.set_position(MAIN_COLORBAR_AX_BOUNDS)
    except Exception:
        pass
    axis = getattr(colorbar, "ax", None)
    if axis is None:
        return
    try:
        axis.yaxis.set_ticks_position("right")
    except Exception:
        pass
    try:
        axis.yaxis.set_label_position("right")
    except Exception:
        pass
    try:
        axis.tick_params(direction="out", left=False, right=True, pad=2)
    except Exception:
        pass


def configure_main_figure_layout(
    fig: Any,
    ax: Any,
    image_artist: Any,
) -> tuple[Any, Any, Any]:
    """Configure the main image axes plus right-docked colorbar layout."""

    apply_main_figure_axes_chrome(ax)

    colorbar_ax = fig.add_axes(MAIN_COLORBAR_AX_BOUNDS)
    colorbar_main = fig.colorbar(image_artist, cax=colorbar_ax)
    colorbar_main.set_label("Intensity")
    _apply_colorbar_chrome(colorbar_main)

    caked_cbar_ax = fig.add_axes(MAIN_COLORBAR_AX_BOUNDS)
    caked_cbar_ax.set_visible(False)
    caked_colorbar = fig.colorbar(image_artist, cax=caked_cbar_ax)
    caked_colorbar.set_label("Intensity (binned)")
    _apply_colorbar_chrome(caked_colorbar)
    caked_colorbar.ax.set_visible(False)

    return colorbar_main, caked_cbar_ax, caked_colorbar
