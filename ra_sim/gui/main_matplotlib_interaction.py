"""Import-safe helpers for throttled legacy Matplotlib main-figure redraws."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any


def cancel_pending_main_matplotlib_redraw(
    widget: object,
    runtime_state: object,
) -> None:
    """Cancel one queued legacy-Matplotlib redraw when a token is present."""

    token = getattr(runtime_state, "main_matplotlib_redraw_token", None)
    setattr(runtime_state, "main_matplotlib_redraw_token", None)
    if token is None:
        return
    after_cancel = getattr(widget, "after_cancel", None)
    if callable(after_cancel):
        try:
            after_cancel(token)
        except Exception:
            pass


def request_main_matplotlib_redraw(
    *,
    widget: object,
    runtime_state: object,
    interval_s: float,
    perf_counter_fn: Callable[[], float],
    draw_now: Callable[[], bool],
    force: bool = False,
) -> bool:
    """Draw immediately or coalesce one pending redraw on the Tk event loop."""

    def _record_draw_timestamp() -> None:
        try:
            timestamp = float(perf_counter_fn())
        except Exception:
            timestamp = math.nan
        setattr(runtime_state, "main_matplotlib_last_draw_ts", timestamp)

    def _flush() -> None:
        setattr(runtime_state, "main_matplotlib_redraw_token", None)
        if bool(draw_now()):
            _record_draw_timestamp()

    if bool(force):
        cancel_pending_main_matplotlib_redraw(widget, runtime_state)
        if bool(draw_now()):
            _record_draw_timestamp()
            return True
        return False

    if getattr(runtime_state, "main_matplotlib_redraw_token", None) is not None:
        return False

    delay_ms = 0
    if float(interval_s) > 0.0:
        try:
            now = float(perf_counter_fn())
        except Exception:
            now = math.nan
        try:
            last_draw_ts = float(
                getattr(runtime_state, "main_matplotlib_last_draw_ts", math.nan)
            )
        except Exception:
            last_draw_ts = math.nan
        if math.isfinite(now) and math.isfinite(last_draw_ts):
            delay_s = max(0.0, float(interval_s) - max(0.0, now - last_draw_ts))
            delay_ms = max(0, int(round(delay_s * 1000.0)))

    after = getattr(widget, "after", None)
    if delay_ms <= 0 or not callable(after):
        if bool(draw_now()):
            _record_draw_timestamp()
            return True
        return False

    token = after(delay_ms, _flush)
    setattr(runtime_state, "main_matplotlib_redraw_token", token)
    return True


def suspend_main_matplotlib_overlays(
    runtime_state: object,
    *,
    suspend_callback: Callable[[], object] | None,
) -> bool:
    """Hide deferred overlays once at the start of one live interaction."""

    if bool(getattr(runtime_state, "main_matplotlib_overlays_suspended", False)):
        return False
    if callable(suspend_callback):
        suspend_callback()
    setattr(runtime_state, "main_matplotlib_overlays_suspended", True)
    return True


def restore_main_matplotlib_overlays(
    runtime_state: object,
    *,
    restore_callback: Callable[[], object] | None,
) -> bool:
    """Restore deferred overlays once after live interaction settles."""

    if not bool(getattr(runtime_state, "main_matplotlib_overlays_suspended", False)):
        return False
    setattr(runtime_state, "main_matplotlib_overlays_suspended", False)
    if callable(restore_callback):
        restore_callback()
    return True
