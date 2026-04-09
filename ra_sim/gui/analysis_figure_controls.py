"""Import-safe helpers for interactive Analyze 1D Matplotlib figures."""

from __future__ import annotations

from collections.abc import Callable


def reset_analysis_axes_view(*axes: object) -> bool:
    """Autoscale each provided axis back to its full data extents."""

    updated = False
    for axis in axes:
        relim = getattr(axis, "relim", None)
        autoscale_view = getattr(axis, "autoscale_view", None)
        if not callable(relim) or not callable(autoscale_view):
            continue
        try:
            relim()
            autoscale_view()
        except Exception:
            continue
        updated = True
    return updated


def create_analysis_figure_toolbar(
    *,
    parent,
    canvas,
    ttk_module,
    backend_tkagg_module,
    on_reset_view: Callable[[], None],
):
    """Create a Tk toolbar row with Matplotlib navigation and reset control."""

    frame = ttk_module.Frame(parent)
    frame.pack(side="top", fill="x")

    toolbar = backend_tkagg_module.NavigationToolbar2Tk(
        canvas,
        frame,
        pack_toolbar=False,
    )
    toolbar.update()
    toolbar.pack(side="left", fill="x", expand=True)

    reset_button = ttk_module.Button(
        frame,
        text="Reset View",
        command=on_reset_view,
    )
    reset_button.pack(side="right", padx=(6, 0), pady=2)

    return frame, toolbar, reset_button
