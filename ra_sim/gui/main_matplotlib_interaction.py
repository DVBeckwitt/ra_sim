"""Helpers for legacy Matplotlib redraw throttling and live blit preview."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any


def _finite_limits(
    limits: tuple[float, float] | list[float] | None,
) -> tuple[float, float] | None:
    if not isinstance(limits, (tuple, list)) or len(limits) != 2:
        return None
    try:
        start = float(limits[0])
        end = float(limits[1])
    except Exception:
        return None
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    return (start, end)


def _axis_limits(
    axis: object,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    get_xlim = getattr(axis, "get_xlim", None)
    get_ylim = getattr(axis, "get_ylim", None)
    if not callable(get_xlim) or not callable(get_ylim):
        return None
    xlim = _finite_limits(get_xlim())
    ylim = _finite_limits(get_ylim())
    if xlim is None or ylim is None:
        return None
    return (xlim, ylim)


def _set_axis_limits(
    axis: object,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> bool:
    set_xlim = getattr(axis, "set_xlim", None)
    set_ylim = getattr(axis, "set_ylim", None)
    if not callable(set_xlim) or not callable(set_ylim):
        return False
    try:
        set_xlim(float(xlim[0]), float(xlim[1]))
        set_ylim(float(ylim[0]), float(ylim[1]))
    except Exception:
        return False
    return True


def _iter_artists(factory: Callable[[], Iterable[object | None]] | None) -> list[object]:
    if not callable(factory):
        return []
    try:
        items = factory()
    except Exception:
        return []
    artists: list[object] = []
    for artist in items or ():
        if artist is not None:
            artists.append(artist)
    return artists


def _artist_visible(artist: object) -> bool:
    getter = getattr(artist, "get_visible", None)
    if callable(getter):
        try:
            return bool(getter())
        except Exception:
            return False
    return True


def _set_artist_visible(artist: object, visible: bool) -> None:
    setter = getattr(artist, "set_visible", None)
    if callable(setter):
        try:
            setter(bool(visible))
        except Exception:
            pass


def _artist_animated(artist: object) -> bool:
    getter = getattr(artist, "get_animated", None)
    if callable(getter):
        try:
            return bool(getter())
        except Exception:
            return False
    return False


def _set_artist_animated(artist: object, animated: bool) -> None:
    setter = getattr(artist, "set_animated", None)
    if callable(setter):
        try:
            setter(bool(animated))
        except Exception:
            pass


def _artist_transform(artist: object) -> object | None:
    getter = getattr(artist, "get_transform", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return None


def _set_artist_transform(artist: object, transform: object) -> None:
    setter = getattr(artist, "set_transform", None)
    if callable(setter):
        try:
            setter(transform)
        except Exception:
            pass


def _preview_transform(
    axis: object,
    *,
    base_xlim: tuple[float, float],
    base_ylim: tuple[float, float],
    target_xlim: tuple[float, float],
    target_ylim: tuple[float, float],
) -> object | None:
    try:
        from matplotlib.transforms import Affine2D
    except Exception:
        return None
    trans_data = getattr(axis, "transData", None)
    if trans_data is None:
        return None

    x_span = float(target_xlim[1]) - float(target_xlim[0])
    y_span = float(target_ylim[1]) - float(target_ylim[0])
    if abs(x_span) <= 1.0e-12 or abs(y_span) <= 1.0e-12:
        return None

    x_scale = (float(base_xlim[1]) - float(base_xlim[0])) / x_span
    y_scale = (float(base_ylim[1]) - float(base_ylim[0])) / y_span
    x_translate = float(base_xlim[0]) - x_scale * float(target_xlim[0])
    y_translate = float(base_ylim[0]) - y_scale * float(target_ylim[0])
    try:
        return Affine2D().scale(x_scale, y_scale).translate(
            x_translate,
            y_translate,
        ) + trans_data
    except Exception:
        return None


class MainMatplotlibPreviewController:
    """Own one live-preview session for the embedded legacy Matplotlib canvas."""

    def __init__(
        self,
        *,
        axis: object,
        canvas: object,
        runtime_state: object,
        preview_artist_factory: Callable[[], Iterable[object | None]] | None,
        hidden_artist_factory: Callable[[], Iterable[object | None]] | None = None,
    ) -> None:
        self._axis = axis
        self._canvas = canvas
        self._runtime_state = runtime_state
        self._preview_artist_factory = preview_artist_factory
        self._hidden_artist_factory = hidden_artist_factory
        self._background: object | None = None
        self._preview_artist_state: list[dict[str, object]] = []
        self._hidden_artist_state: list[dict[str, object]] = []

    def preview_active(self) -> bool:
        return bool(
            getattr(self._runtime_state, "main_matplotlib_preview_active", False)
        )

    def preview_limits(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        value = getattr(
            self._runtime_state,
            "main_matplotlib_preview_target_limits",
            None,
        )
        if not isinstance(value, tuple) or len(value) != 2:
            return None
        xlim = _finite_limits(value[0])
        ylim = _finite_limits(value[1])
        if xlim is None or ylim is None:
            return None
        return (xlim, ylim)

    def invalidate_background(self) -> None:
        self._background = None
        setattr(
            self._runtime_state,
            "main_matplotlib_preview_background_valid",
            False,
        )

    def _clear_preview_state(self) -> None:
        setattr(self._runtime_state, "main_matplotlib_preview_active", False)
        setattr(self._runtime_state, "main_matplotlib_preview_base_limits", None)
        setattr(self._runtime_state, "main_matplotlib_preview_target_limits", None)

    def _restore_hidden_artists(self) -> None:
        for state in reversed(self._hidden_artist_state):
            artist = state.get("artist")
            if artist is None:
                continue
            _set_artist_visible(artist, bool(state.get("visible", True)))
        self._hidden_artist_state.clear()

    def _restore_preview_artists(self) -> None:
        for state in reversed(self._preview_artist_state):
            artist = state.get("artist")
            if artist is None:
                continue
            _set_artist_transform(artist, state.get("transform"))
            _set_artist_visible(artist, bool(state.get("visible", True)))
            _set_artist_animated(artist, bool(state.get("animated", False)))
        self._preview_artist_state.clear()

    def _restore_committed_artists(self, *, redraw: bool) -> None:
        self._restore_hidden_artists()
        self._restore_preview_artists()
        self.invalidate_background()
        self._clear_preview_state()
        if redraw:
            draw = getattr(self._canvas, "draw", None)
            if callable(draw):
                try:
                    draw()
                except Exception:
                    pass

    def _prepare_background(self) -> bool:
        if not bool(getattr(self._canvas, "supports_blit", False)):
            return False
        draw = getattr(self._canvas, "draw", None)
        copy_from_bbox = getattr(self._canvas, "copy_from_bbox", None)
        if not callable(draw) or not callable(copy_from_bbox):
            return False

        self._preview_artist_state = []
        for artist in _iter_artists(self._preview_artist_factory):
            self._preview_artist_state.append(
                {
                    "artist": artist,
                    "transform": _artist_transform(artist),
                    "visible": _artist_visible(artist),
                    "animated": _artist_animated(artist),
                }
            )
            _set_artist_visible(artist, False)
            _set_artist_animated(artist, False)

        self._hidden_artist_state = []
        for artist in _iter_artists(self._hidden_artist_factory):
            self._hidden_artist_state.append(
                {
                    "artist": artist,
                    "visible": _artist_visible(artist),
                }
            )
            _set_artist_visible(artist, False)

        try:
            draw()
            self._background = copy_from_bbox(getattr(self._axis, "bbox", None))
        except Exception:
            self._background = None
            self._restore_hidden_artists()
            self._restore_preview_artists()
            try:
                draw()
            except Exception:
                pass
            self.invalidate_background()
            return False

        setattr(self._runtime_state, "main_matplotlib_preview_background_valid", True)
        return True

    def _ensure_preview_ready(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if self.preview_active():
            base_limits = getattr(
                self._runtime_state,
                "main_matplotlib_preview_base_limits",
                None,
            )
            if (
                isinstance(base_limits, tuple)
                and len(base_limits) == 2
                and _finite_limits(base_limits[0]) is not None
                and _finite_limits(base_limits[1]) is not None
                and bool(
                    getattr(
                        self._runtime_state,
                        "main_matplotlib_preview_background_valid",
                        False,
                    )
                )
                and self._background is not None
            ):
                return (_finite_limits(base_limits[0]), _finite_limits(base_limits[1]))
            self._restore_committed_artists(redraw=True)
            return None

        base_limits = _axis_limits(self._axis)
        if base_limits is None:
            return None
        if not self._prepare_background():
            return None
        setattr(self._runtime_state, "main_matplotlib_preview_active", True)
        setattr(
            self._runtime_state,
            "main_matplotlib_preview_base_limits",
            base_limits,
        )
        return base_limits

    def preview_view_limits(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> bool:
        target_xlim = _finite_limits(xlim)
        target_ylim = _finite_limits(ylim)
        if target_xlim is None or target_ylim is None:
            return False

        base_limits = self._ensure_preview_ready()
        if base_limits is None or self._background is None:
            return False
        base_xlim, base_ylim = base_limits
        transform = _preview_transform(
            self._axis,
            base_xlim=base_xlim,
            base_ylim=base_ylim,
            target_xlim=target_xlim,
            target_ylim=target_ylim,
        )
        restore_region = getattr(self._canvas, "restore_region", None)
        draw_artist = getattr(self._axis, "draw_artist", None)
        blit = getattr(self._canvas, "blit", None)
        if transform is None:
            self._restore_committed_artists(redraw=True)
            return False
        if not callable(restore_region) or not callable(draw_artist) or not callable(blit):
            self._restore_committed_artists(redraw=True)
            return False

        for state in self._preview_artist_state:
            artist = state.get("artist")
            if artist is None:
                continue
            _set_artist_transform(artist, transform)
            _set_artist_visible(artist, bool(state.get("visible", True)))
            _set_artist_animated(artist, True)

        try:
            restore_region(self._background)
            for state in self._preview_artist_state:
                artist = state.get("artist")
                if artist is None or not bool(state.get("visible", True)):
                    continue
                draw_artist(artist)
            blit(getattr(self._axis, "bbox", None))
        except Exception:
            self._restore_committed_artists(redraw=True)
            return False

        setattr(
            self._runtime_state,
            "main_matplotlib_preview_target_limits",
            (target_xlim, target_ylim),
        )
        return True

    def commit_preview_view(self) -> bool:
        if not self.preview_active():
            return False
        limits = self.preview_limits()
        if limits is None:
            self._restore_committed_artists(redraw=True)
            return False
        if not _set_axis_limits(self._axis, xlim=limits[0], ylim=limits[1]):
            self._restore_committed_artists(redraw=True)
            return False
        self._restore_committed_artists(redraw=False)
        return True

    def clear_preview_view(self, *, redraw: bool = True) -> bool:
        if not self.preview_active():
            return False
        self._restore_committed_artists(redraw=bool(redraw))
        return True


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
