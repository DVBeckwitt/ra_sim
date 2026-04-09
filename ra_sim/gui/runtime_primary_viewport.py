"""Import-safe helpers for selecting and activating the main GUI viewport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RuntimePrimaryViewportSelection:
    """Describe the active main viewport after startup selection."""

    requested_backend: str
    active_backend: str
    backend: object | None
    canvas_proxy: object
    fallback_reason: str | None = None


def _safe_set_progress_text(
    callback: Callable[[str], None] | None,
    text: str,
) -> None:
    if not callable(callback):
        return
    try:
        callback(str(text))
    except Exception:
        pass


def _show_matplotlib_canvas(*, tk_module, matplotlib_canvas) -> None:
    widget = matplotlib_canvas.get_tk_widget()
    try:
        if str(widget.winfo_manager()) == "pack":
            return
    except Exception:
        pass
    try:
        widget.pack(
            side=tk_module.TOP,
            fill=tk_module.BOTH,
            expand=True,
        )
    except Exception:
        pass


def activate_runtime_primary_viewport(
    *,
    requested_backend: str,
    tk_primary_viewport_module,
    tk_module,
    canvas_frame,
    matplotlib_canvas,
    ax,
    image_artist,
    background_artist,
    overlay_artist,
    marker_artist_factory: object = None,
    overlay_model_factory: object = None,
    overlay_artist_groups_factory: object = None,
    layer_versions_factory: object = None,
    peak_cache_factory: object = None,
    qgroup_cache_factory: object = None,
    draw_interval_s: float = 1.0 / 60.0,
    set_progress_text: Callable[[str], None] | None = None,
) -> RuntimePrimaryViewportSelection:
    """Activate the requested main viewport, falling back to Matplotlib when needed."""

    requested_backend = str(requested_backend or "matplotlib")
    fallback_message = "Tk canvas main viewport unavailable; using Matplotlib fallback."

    if requested_backend != "tk_canvas":
        _show_matplotlib_canvas(
            tk_module=tk_module,
            matplotlib_canvas=matplotlib_canvas,
        )
        return RuntimePrimaryViewportSelection(
            requested_backend=requested_backend,
            active_backend="matplotlib",
            backend=None,
            canvas_proxy=matplotlib_canvas,
        )

    runtime_available = getattr(
        tk_primary_viewport_module,
        "primary_viewport_runtime_available",
        None,
    )
    runtime_reason = getattr(
        tk_primary_viewport_module,
        "primary_viewport_unavailable_reason",
        None,
    )
    if callable(runtime_available) and not bool(runtime_available()):
        reason = runtime_reason() if callable(runtime_reason) else None
        _safe_set_progress_text(set_progress_text, fallback_message)
        _show_matplotlib_canvas(
            tk_module=tk_module,
            matplotlib_canvas=matplotlib_canvas,
        )
        return RuntimePrimaryViewportSelection(
            requested_backend=requested_backend,
            active_backend="matplotlib",
            backend=None,
            canvas_proxy=matplotlib_canvas,
            fallback_reason=str(reason) if reason else None,
        )

    backend = None
    try:
        backend = tk_primary_viewport_module.build_tk_primary_viewport_backend(
            tk_module=tk_module,
            canvas_frame=canvas_frame,
            matplotlib_canvas=matplotlib_canvas,
            ax=ax,
            image_artist=image_artist,
            background_artist=background_artist,
            overlay_artist=overlay_artist,
            marker_artist_factory=marker_artist_factory,
            overlay_model_factory=overlay_model_factory,
            overlay_artist_groups_factory=overlay_artist_groups_factory,
            layer_versions_factory=layer_versions_factory,
            peak_cache_factory=peak_cache_factory,
            qgroup_cache_factory=qgroup_cache_factory,
            draw_interval_s=float(draw_interval_s),
        )
        backend.activate()
    except Exception as exc:
        if backend is not None:
            try:
                backend.deactivate()
            except Exception:
                pass
            try:
                backend.shutdown()
            except Exception:
                pass
        _safe_set_progress_text(set_progress_text, fallback_message)
        _show_matplotlib_canvas(
            tk_module=tk_module,
            matplotlib_canvas=matplotlib_canvas,
        )
        reason = str(exc).strip() or exc.__class__.__name__
        return RuntimePrimaryViewportSelection(
            requested_backend=requested_backend,
            active_backend="matplotlib",
            backend=None,
            canvas_proxy=matplotlib_canvas,
            fallback_reason=reason,
        )

    return RuntimePrimaryViewportSelection(
        requested_backend=requested_backend,
        active_backend="tk_canvas",
        backend=backend,
        canvas_proxy=backend.canvas_proxy,
    )
