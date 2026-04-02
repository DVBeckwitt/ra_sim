"""Fast optional 2D viewer for high-frequency image updates.

This module is safe to import without Qt/PyQtGraph installed. GUI imports are
done lazily at runtime.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable
from types import SimpleNamespace

import numpy as np


def _qt_mouse_button_to_mpl(value: object) -> int | None:
    try:
        raw = int(value)
    except Exception:
        return None
    # Qt uses 1=Left, 2=Right, 4=Middle.
    if raw == 1:
        return 1
    if raw == 2:
        return 3
    if raw == 4:
        return 2
    return None


def _qt_button_mask_matches(button: object, target: object) -> bool:
    try:
        return bool(int(button) & int(target))
    except Exception:
        return button == target


def _should_consume_fast_viewer_viewport_event(
    event_type: object,
    *,
    button: object,
    buttons: object,
    left_button: object,
) -> bool:
    try:
        normalized_type = int(event_type)
    except Exception:
        return False
    if normalized_type in {2, 3, 4}:
        return _qt_button_mask_matches(button, left_button)
    if normalized_type == 5:
        return _qt_button_mask_matches(buttons, left_button)
    return False


def _windows_client_point_from_lparam(lparam: object) -> tuple[int, int] | None:
    try:
        raw = int(lparam)
    except Exception:
        return None
    x_value = (raw & 0xFFFF)
    y_value = ((raw >> 16) & 0xFFFF)
    if x_value >= 0x8000:
        x_value -= 0x10000
    if y_value >= 0x8000:
        y_value -= 0x10000
    return (int(x_value), int(y_value))


def _build_turbo_lut(zero_white: bool) -> np.ndarray:
    """Return a 256-entry uint8 RGBA lookup table."""
    try:
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap("turbo")
        lut = np.asarray(cmap(np.linspace(0.0, 1.0, 256)) * 255.0, dtype=np.uint8)
    except Exception:
        gray = np.linspace(0, 255, 256, dtype=np.uint8)
        lut = np.stack([gray, gray, gray, np.full_like(gray, 255)], axis=1)

    if zero_white:
        lut[0] = np.array([255, 255, 255, 255], dtype=np.uint8)
    return lut


def _coerce_2d_array(array_like) -> np.ndarray | None:
    if array_like is None:
        return None
    arr = np.asarray(array_like)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.size == 0:
        return None
    if np.ma.isMaskedArray(arr):
        arr = np.ma.filled(arr, 0.0)
    return np.asarray(arr)


def _to_2d_float32(array_like) -> np.ndarray | None:
    arr = _coerce_2d_array(array_like)
    if arr is None:
        return None
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


@dataclass(frozen=True)
class _RectangleOverlaySpec:
    x: float
    y: float
    width: float
    height: float
    edge_rgba: tuple[int, int, int, int]
    linewidth: float
    linestyle: object
    zorder: float


@dataclass(frozen=True)
class _MarkerOverlaySpec:
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    symbol: str
    size: float
    edge_rgba: tuple[int, int, int, int]
    face_rgba: tuple[int, int, int, int]
    pen_width: float
    zorder: float


@dataclass(frozen=True)
class FastViewerCurveSpec:
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    edge_rgba: tuple[int, int, int, int]
    linewidth: float
    linestyle: object
    zorder: float


@dataclass(frozen=True)
class FastViewerOverlayModel:
    rectangle_specs: tuple[_RectangleOverlaySpec, ...] = ()
    transient_rectangle_specs: tuple[_RectangleOverlaySpec, ...] = ()
    transient_marker_specs: tuple[_MarkerOverlaySpec, ...] = ()
    transient_curve_specs: tuple[FastViewerCurveSpec, ...] = ()
    suppress_overlay_image: bool = False


@dataclass(frozen=True)
class _ImageSyncState:
    visible: bool
    version: object
    levels: tuple[float, float] | None
    alpha: float
    extent: tuple[float, ...] | None


def _to_rgba_uint8(color: object) -> tuple[int, int, int, int]:
    try:
        import matplotlib.colors as mcolors

        rgba = mcolors.to_rgba(color)
        return tuple(int(round(float(value) * 255.0)) for value in rgba)
    except Exception:
        pass

    try:
        values = np.asarray(color, dtype=float).reshape(-1)
    except Exception:
        return (255, 255, 255, 255)

    if values.size == 0:
        return (255, 255, 255, 255)
    if values.size == 1:
        values = np.repeat(values[:1], 4)
    elif values.size == 2:
        values = np.array([values[0], values[0], values[0], values[1]], dtype=float)
    elif values.size == 3:
        values = np.concatenate([values[:3], [1.0]])
    else:
        values = values[:4]

    if np.nanmax(np.abs(values)) <= 1.0:
        values = values * 255.0
    values = np.clip(
        np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0),
        0.0,
        255.0,
    )
    return tuple(int(round(float(value))) for value in values)


def _artist_visible(artist: object) -> bool:
    visible_getter = getattr(artist, "get_visible", None)
    if callable(visible_getter):
        try:
            return bool(visible_getter())
        except Exception:
            return False
    return True


def _normalize_levels(levels: object) -> tuple[float, float] | None:
    if levels is None:
        return None
    try:
        lo, hi = levels
    except Exception:
        return None
    try:
        lo = float(lo)
        hi = float(hi)
    except Exception:
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return (lo, hi)


def _normalize_extent(extent: object) -> tuple[float, ...] | None:
    if extent is None:
        return None
    try:
        values = tuple(float(value) for value in extent)
    except Exception:
        return None
    if len(values) != 4 or not all(np.isfinite(value) for value in values):
        return None
    return values


def _raw_array_signature(array_like, *, version: object = None) -> object:
    if version is not None:
        return version
    arr = _coerce_2d_array(array_like)
    if arr is None:
        return None
    interface = getattr(arr, "__array_interface__", {})
    data = interface.get("data", ())
    pointer = None
    if isinstance(data, tuple) and data:
        pointer = data[0]
    return (
        id(arr),
        pointer,
        tuple(int(value) for value in arr.shape),
        str(arr.dtype),
        tuple(int(value) for value in getattr(arr, "strides", ()) or ()),
    )


def _extract_visible_rectangle_specs_from_artists(
    artists: object,
) -> tuple[_RectangleOverlaySpec, ...]:
    specs: list[_RectangleOverlaySpec] = []
    for patch in _flatten_artist_collection(artists):
        if not _artist_visible(patch):
            continue
        if not all(
            callable(getattr(patch, attr_name, None))
            for attr_name in (
                "get_xy",
                "get_width",
                "get_height",
                "get_edgecolor",
                "get_linewidth",
                "get_linestyle",
            )
        ):
            continue
        try:
            x0, y0 = patch.get_xy()
            width = float(patch.get_width())
            height = float(patch.get_height())
        except Exception:
            continue
        if not all(np.isfinite(value) for value in (x0, y0, width, height)):
            continue

        x1 = float(x0) + width
        y1 = float(y0) + height
        x_min, x_max = sorted((float(x0), float(x1)))
        y_min, y_max = sorted((float(y0), float(y1)))
        edge_rgba = _to_rgba_uint8(patch.get_edgecolor())
        if edge_rgba[3] <= 0:
            continue

        try:
            linewidth = max(float(patch.get_linewidth()), 0.5)
        except Exception:
            linewidth = 1.0
        try:
            zorder = float(patch.get_zorder())
        except Exception:
            zorder = 6.0

        specs.append(
            _RectangleOverlaySpec(
                x=x_min,
                y=y_min,
                width=x_max - x_min,
                height=y_max - y_min,
                edge_rgba=edge_rgba,
                linewidth=linewidth,
                linestyle=patch.get_linestyle(),
                zorder=zorder,
            )
        )

    specs.sort(key=lambda spec: spec.zorder)
    return tuple(specs)


def _extract_visible_rectangle_specs(ax) -> tuple[_RectangleOverlaySpec, ...]:
    return _extract_visible_rectangle_specs_from_artists(
        getattr(ax, "patches", ()) or (),
    )


def _iter_artist_collection(artist_or_artists: object) -> tuple[object, ...]:
    if artist_or_artists is None:
        return ()
    if isinstance(artist_or_artists, (list, tuple, set, frozenset)):
        return tuple(item for item in artist_or_artists if item is not None)
    return (artist_or_artists,)


def _flatten_artist_collection(artist_or_artists: object) -> tuple[object, ...]:
    flattened: list[object] = []
    for item in _iter_artist_collection(artist_or_artists):
        if isinstance(item, (list, tuple, set, frozenset)):
            flattened.extend(_flatten_artist_collection(item))
        else:
            flattened.append(item)
    return tuple(flattened)


def _marker_symbol_for_pg(marker: object) -> str | None:
    normalized = str(marker or "").strip().lower()
    if normalized in {"", "none", "null", " ", "nan"}:
        return None
    symbol_map = {
        "o": "o",
        "s": "s",
        "square": "s",
        "d": "d",
        "diamond": "d",
        "+": "+",
        "plus": "+",
        "x": "x",
        "^": "t",
        "triangle_up": "t",
        "triangle-up": "t",
        "t": "t",
        "*": "star",
        "star": "star",
        "p": "p",
        "pentagon": "p",
        "h": "h",
        "hexagon": "h",
    }
    return symbol_map.get(normalized, "o")


def _extract_visible_marker_specs(marker_artist) -> tuple[_MarkerOverlaySpec, ...]:
    specs: list[_MarkerOverlaySpec] = []
    for artist in _flatten_artist_collection(marker_artist):
        if not _artist_visible(artist):
            continue
        if not all(
            callable(getattr(artist, attr_name, None))
            for attr_name in (
                "get_xdata",
                "get_ydata",
                "get_marker",
            )
        ):
            continue
        symbol = _marker_symbol_for_pg(artist.get_marker())
        if symbol is None:
            continue

        try:
            x_raw = np.asarray(artist.get_xdata(), dtype=float).reshape(-1)
            y_raw = np.asarray(artist.get_ydata(), dtype=float).reshape(-1)
        except Exception:
            continue
        if x_raw.size == 0 or y_raw.size == 0:
            continue
        point_count = min(int(x_raw.size), int(y_raw.size))
        if point_count <= 0:
            continue
        x_vals = x_raw[:point_count]
        y_vals = y_raw[:point_count]
        finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not np.any(finite_mask):
            continue

        markeredgecolor_getter = getattr(artist, "get_markeredgecolor", None)
        markerfacecolor_getter = getattr(artist, "get_markerfacecolor", None)
        color_getter = getattr(artist, "get_color", None)
        markeredgewidth_getter = getattr(artist, "get_markeredgewidth", None)
        linewidth_getter = getattr(artist, "get_linewidth", None)
        markersize_getter = getattr(artist, "get_markersize", None)
        zorder_getter = getattr(artist, "get_zorder", None)

        edge_rgba = _to_rgba_uint8(
            markeredgecolor_getter() if callable(markeredgecolor_getter) else None
        )
        if edge_rgba[3] <= 0 and callable(color_getter):
            edge_rgba = _to_rgba_uint8(color_getter())
        face_rgba = _to_rgba_uint8(
            markerfacecolor_getter() if callable(markerfacecolor_getter) else None
        )
        if face_rgba[3] <= 0 and edge_rgba[3] <= 0:
            continue

        try:
            pen_width = float(
                markeredgewidth_getter()
                if callable(markeredgewidth_getter)
                else linewidth_getter()
            )
        except Exception:
            pen_width = 1.0
        try:
            size = float(markersize_getter()) if callable(markersize_getter) else 8.0
        except Exception:
            size = 8.0
        try:
            zorder = float(zorder_getter()) if callable(zorder_getter) else 10.0
        except Exception:
            zorder = 10.0

        specs.append(
            _MarkerOverlaySpec(
                x_values=tuple(float(value) for value in x_vals[finite_mask]),
                y_values=tuple(float(value) for value in y_vals[finite_mask]),
                symbol=str(symbol),
                size=max(float(size), 1.0),
                edge_rgba=edge_rgba,
                face_rgba=face_rgba,
                pen_width=max(float(pen_width), 0.5),
                zorder=zorder,
            )
        )

    specs.sort(key=lambda spec: spec.zorder)
    return tuple(specs)


def _linestyle_is_visible(linestyle: object) -> bool:
    normalized = str(linestyle or "").strip().lower()
    return normalized not in {"", "none", "null", " ", "nan"}


def _extract_visible_curve_specs(curve_artists) -> tuple[FastViewerCurveSpec, ...]:
    specs: list[FastViewerCurveSpec] = []
    for artist in _flatten_artist_collection(curve_artists):
        if not _artist_visible(artist):
            continue
        if not all(
            callable(getattr(artist, attr_name, None))
            for attr_name in (
                "get_xdata",
                "get_ydata",
                "get_linestyle",
            )
        ):
            continue
        linestyle = artist.get_linestyle()
        if not _linestyle_is_visible(linestyle):
            continue
        try:
            x_raw = np.asarray(artist.get_xdata(), dtype=float).reshape(-1)
            y_raw = np.asarray(artist.get_ydata(), dtype=float).reshape(-1)
        except Exception:
            continue
        if x_raw.size == 0 or y_raw.size == 0:
            continue
        point_count = min(int(x_raw.size), int(y_raw.size))
        if point_count <= 1:
            continue
        x_vals = x_raw[:point_count]
        y_vals = y_raw[:point_count]
        finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if np.count_nonzero(finite_mask) <= 1:
            continue

        color_getter = getattr(artist, "get_color", None)
        linewidth_getter = getattr(artist, "get_linewidth", None)
        zorder_getter = getattr(artist, "get_zorder", None)
        edge_rgba = _to_rgba_uint8(color_getter() if callable(color_getter) else None)
        if edge_rgba[3] <= 0:
            continue
        try:
            linewidth = (
                float(linewidth_getter()) if callable(linewidth_getter) else 1.0
            )
        except Exception:
            linewidth = 1.0
        try:
            zorder = float(zorder_getter()) if callable(zorder_getter) else 9.0
        except Exception:
            zorder = 9.0
        specs.append(
            FastViewerCurveSpec(
                x_values=tuple(float(value) for value in x_vals[finite_mask]),
                y_values=tuple(float(value) for value in y_vals[finite_mask]),
                edge_rgba=edge_rgba,
                linewidth=max(float(linewidth), 0.5),
                linestyle=linestyle,
                zorder=zorder,
            )
        )

    specs.sort(key=lambda spec: spec.zorder)
    return tuple(specs)


def build_artist_overlay_model(
    *,
    rectangle_artists: object = None,
    transient_artists: object = None,
    transient_curve_specs: object = None,
    suppress_overlay_image: bool = False,
) -> FastViewerOverlayModel:
    explicit_curve_specs: list[FastViewerCurveSpec] = []
    for spec in _flatten_artist_collection(transient_curve_specs):
        if isinstance(spec, FastViewerCurveSpec):
            explicit_curve_specs.append(spec)
            continue
        if not isinstance(spec, dict):
            continue
        try:
            explicit_curve_specs.append(
                FastViewerCurveSpec(
                    x_values=tuple(float(value) for value in spec.get("x_values", ()) or ()),
                    y_values=tuple(float(value) for value in spec.get("y_values", ()) or ()),
                    edge_rgba=tuple(int(value) for value in spec.get("edge_rgba", (0, 255, 255, 255))),
                    linewidth=max(float(spec.get("linewidth", 1.0)), 0.5),
                    linestyle=spec.get("linestyle", "-"),
                    zorder=float(spec.get("zorder", 9.0)),
                )
            )
        except Exception:
            continue

    transient_curve_values = list(_extract_visible_curve_specs(transient_artists))
    transient_curve_values.extend(explicit_curve_specs)
    transient_curve_values.sort(key=lambda spec: spec.zorder)

    return FastViewerOverlayModel(
        rectangle_specs=_extract_visible_rectangle_specs_from_artists(rectangle_artists),
        transient_rectangle_specs=_extract_visible_rectangle_specs_from_artists(
            transient_artists
        ),
        transient_marker_specs=_extract_visible_marker_specs(transient_artists),
        transient_curve_specs=tuple(transient_curve_values),
        suppress_overlay_image=bool(suppress_overlay_image),
    )


def _coerce_overlay_model(overlay_model: object) -> FastViewerOverlayModel:
    if isinstance(overlay_model, FastViewerOverlayModel):
        return overlay_model
    return FastViewerOverlayModel()


def _normalize_view_range(
    x0: object,
    x1: object,
    y0: object,
    y1: object,
) -> tuple[float, float, float, float] | None:
    try:
        normalized = tuple(float(value) for value in (x0, x1, y0, y1))
    except Exception:
        return None
    if not all(np.isfinite(value) for value in normalized):
        return None
    return normalized


def _view_ranges_match(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
    *,
    atol: float = 1e-6,
) -> bool:
    if left is None or right is None:
        return left is right
    return all(abs(float(a) - float(b)) <= float(atol) for a, b in zip(left, right))


def _should_apply_matplotlib_view_range(
    *,
    current_range: tuple[float, float, float, float] | None,
    last_synced_range: tuple[float, float, float, float] | None,
    target_range: tuple[float, float, float, float] | None,
) -> bool:
    del current_range
    if target_range is None:
        return False
    if last_synced_range is None:
        return True
    if not _view_ranges_match(target_range, last_synced_range):
        return True
    return False


class FastPlotViewer:
    """High-performance 2D viewer powered by PyQtGraph."""

    available: bool = False

    def __init__(
        self,
        *,
        title: str = "RA-SIM Fast Viewer",
        use_opengl: bool = True,
        show_window: bool = True,
    ) -> None:
        self.available = False
        self.error_message: str | None = None

        self._pg = None
        self._QtGui = None
        self._QtCore = None
        self._QtWidgets = None
        self._app = None
        self._window = None
        self._plot_widget = None
        self._plot_item = None
        self._background_item = None
        self._simulation_item = None
        self._overlay_item = None
        self._marker_items: list[object] = []
        self._rectangle_items: list[object] = []
        self._transient_marker_items: list[object] = []
        self._transient_rectangle_items: list[object] = []
        self._transient_curve_items: list[object] = []
        self._last_synced_view_range: tuple[float, float, float, float] | None = None
        self._image_sync_states: dict[str, _ImageSyncState] = {}
        self._image_buffers: dict[str, np.ndarray] = {}
        self._marker_specs: tuple[_MarkerOverlaySpec, ...] = ()
        self._rectangle_specs: tuple[_RectangleOverlaySpec, ...] = ()
        self._transient_marker_specs: tuple[_MarkerOverlaySpec, ...] = ()
        self._transient_rectangle_specs: tuple[_RectangleOverlaySpec, ...] = ()
        self._transient_curve_specs: tuple[FastViewerCurveSpec, ...] = ()
        self._last_title: str | None = None
        self._last_xlabel: str | None = None
        self._last_ylabel: str | None = None
        self._mouse_event_callback: Callable[..., None] | None = None
        self._mouse_capture_overlay = None
        self._embedded_tk_host = None
        self._embedded_parent_hwnd: int | None = None
        self._native_hwnd: int | None = None
        self._native_input_enabled = True
        self._embedded_native_left_drag_active = False
        self._native_mouse_bridge_prev_wndproc = None
        self._native_mouse_bridge_callback = None
        if not bool(show_window):
            # Embedded Qt child windows are materially less stable with
            # OpenGL-backed viewports on some Windows driver stacks.
            use_opengl = False

        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
        except Exception as exc:
            self.error_message = str(exc)
            return

        self._pg = pg
        self._QtGui = QtGui
        self._QtCore = QtCore
        self._QtWidgets = QtWidgets

        try:
            pg.setConfigOptions(antialias=False, useOpenGL=bool(use_opengl))
        except Exception:
            pg.setConfigOptions(antialias=False)

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        self._app = app

        self._window = QtWidgets.QWidget()
        self._window.setWindowTitle(title)
        self._window.resize(1050, 850)
        layout = QtWidgets.QVBoxLayout(self._window)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._plot_item = pg.PlotItem(viewBox=self._new_view_box())
        self._plot_widget = pg.PlotWidget(
            background="w",
            plotItem=self._plot_item,
        )
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self._plot_widget.getPlotItem().getViewBox().setAspectLocked(False)
        self._plot_widget.setMouseTracking(True)
        layout.addWidget(self._plot_widget)
        self._install_mouse_capture_overlay()

        self._plot_item.setLabel("bottom", "X (pixels)")
        self._plot_item.setLabel("left", "Y (pixels)")
        self._plot_item.setTitle("Simulated Diffraction Pattern")

        self._background_item = self._new_image_item()
        self._simulation_item = self._new_image_item()
        self._overlay_item = self._new_image_item()

        self._background_item.setZValue(0)
        self._simulation_item.setZValue(1)
        self._overlay_item.setZValue(5)

        self._plot_item.addItem(self._background_item)
        self._plot_item.addItem(self._simulation_item)
        self._plot_item.addItem(self._overlay_item)

        self._background_item.setLookupTable(_build_turbo_lut(zero_white=False))
        self._simulation_item.setLookupTable(_build_turbo_lut(zero_white=True))
        self._overlay_item.setLookupTable(
            np.asarray(
                [
                    [0, 0, 0, 0],
                    [255, 0, 168, 136],
                ],
                dtype=np.uint8,
            )
        )

        self.available = True
        if bool(show_window):
            self.show_window()
        self.process_events()

    def _new_view_box(self):
        assert self._pg is not None
        assert self._QtCore is not None

        pg = self._pg
        QtCore = self._QtCore
        right_button = getattr(
            getattr(QtCore.Qt, "MouseButton", None),
            "RightButton",
            getattr(QtCore.Qt, "RightButton", None),
        )
        left_button = getattr(
            getattr(QtCore.Qt, "MouseButton", None),
            "LeftButton",
            getattr(QtCore.Qt, "LeftButton", None),
        )
        middle_button = getattr(
            getattr(QtCore.Qt, "MouseButton", None),
            "MiddleButton",
            getattr(QtCore.Qt, "MiddleButton", None),
        )

        class _FastViewBox(pg.ViewBox):  # pragma: no cover - exercised via GUI
            def __init__(self):
                super().__init__(enableMenu=False)

            def mouseDragEvent(self, ev, axis=None):
                button = ev.button()
                if _qt_button_mask_matches(button, left_button):
                    ev.accept()
                    return
                if not (
                    _qt_button_mask_matches(button, right_button)
                    or _qt_button_mask_matches(button, middle_button)
                ):
                    return super().mouseDragEvent(ev, axis=axis)

                ev.accept()
                pos = ev.pos()
                last_pos = ev.lastPos()
                delta = (pos - last_pos) * -1

                mouse_enabled = np.array(self.state["mouseEnabled"], dtype=np.float64)
                mask = mouse_enabled.copy()
                if axis is not None:
                    mask[1 - axis] = 0.0

                transform = self.childGroup.transform()
                transform = pg.functions.invertQTransform(transform)
                translated = transform.map(delta * mask) - transform.map(pg.Point(0, 0))

                x_delta = translated.x() if mask[0] == 1 else None
                y_delta = translated.y() if mask[1] == 1 else None

                self._resetTarget()
                if x_delta is not None or y_delta is not None:
                    self.translateBy(x=x_delta, y=y_delta)
                self.sigRangeChangedManually.emit(self.state["mouseEnabled"])

        return _FastViewBox()

    def show_window(self) -> None:
        if not self.available or self._window is None:
            return
        try:
            self._window.show()
        except Exception:
            return
        self.process_events()

    def _window_handle(self) -> int | None:
        if self._window is None or self._QtCore is None:
            return None
        try:
            widget_attr = getattr(self._QtCore.Qt, "WidgetAttribute", None)
            native_flag = (
                getattr(widget_attr, "WA_NativeWindow", None)
                if widget_attr is not None
                else getattr(self._QtCore.Qt, "WA_NativeWindow", None)
            )
            if native_flag is not None:
                self._window.setAttribute(native_flag, True)
        except Exception:
            pass
        try:
            return int(self._window.winId())
        except Exception:
            return None

    def _resize_native_window(self, width: int, height: int) -> None:
        if self._native_hwnd is None or not sys.platform.startswith("win"):
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.MoveWindow.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.BOOL,
            ]
            user32.MoveWindow.restype = wintypes.BOOL
            user32.MoveWindow(
                wintypes.HWND(self._native_hwnd),
                0,
                0,
                max(int(width), 1),
                max(int(height), 1),
                True,
            )
        except Exception:
            pass

    def _apply_native_input_enabled(self) -> None:
        if self._native_hwnd is None or not sys.platform.startswith("win"):
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.EnableWindow.argtypes = [wintypes.HWND, wintypes.BOOL]
            user32.EnableWindow.restype = wintypes.BOOL
            user32.EnableWindow(
                wintypes.HWND(self._native_hwnd),
                bool(self._native_input_enabled),
            )
        except Exception:
            pass

    def set_native_input_enabled(self, enabled: bool) -> None:
        self._native_input_enabled = bool(enabled)
        self._apply_native_input_enabled()

    def _set_native_mouse_capture(self, enabled: bool) -> None:
        if self._native_hwnd is None or not sys.platform.startswith("win"):
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            if bool(enabled):
                user32.SetCapture.argtypes = [wintypes.HWND]
                user32.SetCapture.restype = wintypes.HWND
                user32.SetCapture(wintypes.HWND(self._native_hwnd))
            else:
                user32.ReleaseCapture.argtypes = []
                user32.ReleaseCapture.restype = wintypes.BOOL
                user32.ReleaseCapture()
        except Exception:
            pass

    def _handle_embedded_native_mouse_message(
        self,
        message: object,
        wparam: object,
        lparam: object,
    ) -> bool:
        WM_MOUSEMOVE = 0x0200
        WM_LBUTTONDOWN = 0x0201
        WM_LBUTTONUP = 0x0202
        WM_LBUTTONDBLCLK = 0x0203
        MK_LBUTTON = 0x0001

        try:
            message_value = int(message)
        except Exception:
            return False
        if message_value not in {
            WM_MOUSEMOVE,
            WM_LBUTTONDOWN,
            WM_LBUTTONUP,
            WM_LBUTTONDBLCLK,
        }:
            return False
        if self._mouse_event_callback is None:
            return False
        point = _windows_client_point_from_lparam(lparam)
        if point is None:
            return False
        qpoint = None
        qpoint_cls = getattr(self._QtCore, "QPoint", None) if self._QtCore is not None else None
        if callable(qpoint_cls):
            try:
                qpoint = qpoint_cls(int(point[0]), int(point[1]))
            except Exception:
                qpoint = None
        if qpoint is None:
            qpoint = type(
                "_NativePoint",
                (),
                {
                    "x": lambda self: point[0],
                    "y": lambda self: point[1],
                },
            )()

        if message_value == WM_LBUTTONDOWN:
            self._embedded_native_left_drag_active = True
            self._set_native_mouse_capture(True)
            self._forward_mouse_event_from_qpoint(
                "button_press_event",
                qpoint,
                button=1,
                dblclick=False,
            )
            return True
        if message_value == WM_LBUTTONDBLCLK:
            self._embedded_native_left_drag_active = True
            self._set_native_mouse_capture(True)
            self._forward_mouse_event_from_qpoint(
                "button_press_event",
                qpoint,
                button=1,
                dblclick=True,
            )
            return True
        if message_value == WM_MOUSEMOVE:
            try:
                buttons_mask = int(wparam)
            except Exception:
                buttons_mask = 0
            if not (
                self._embedded_native_left_drag_active
                or bool(buttons_mask & MK_LBUTTON)
            ):
                return False
            self._forward_mouse_event_from_qpoint(
                "motion_notify_event",
                qpoint,
                button=0,
                dblclick=False,
            )
            return True

        self._forward_mouse_event_from_qpoint(
            "button_release_event",
            qpoint,
            button=1,
            dblclick=False,
        )
        self._embedded_native_left_drag_active = False
        self._set_native_mouse_capture(False)
        return True

    def _install_embedded_native_mouse_bridge(self) -> None:
        if (
            self._native_hwnd is None
            or not sys.platform.startswith("win")
            or self._native_mouse_bridge_callback is not None
        ):
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.SetWindowLongPtrW.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_ssize_t,
            ]
            user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
            user32.CallWindowProcW.argtypes = [
                ctypes.c_ssize_t,
                wintypes.HWND,
                ctypes.c_uint,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            user32.CallWindowProcW.restype = ctypes.c_ssize_t
            GWLP_WNDPROC = -4
            WNDPROC = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t,
                wintypes.HWND,
                ctypes.c_uint,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )

            def _wndproc(hwnd, msg, wparam, lparam):
                if self._handle_embedded_native_mouse_message(msg, wparam, lparam):
                    return 0
                return user32.CallWindowProcW(
                    ctypes.c_ssize_t(self._native_mouse_bridge_prev_wndproc),
                    hwnd,
                    msg,
                    wparam,
                    lparam,
                )

            callback = WNDPROC(_wndproc)
            previous = user32.SetWindowLongPtrW(
                wintypes.HWND(self._native_hwnd),
                GWLP_WNDPROC,
                ctypes.cast(callback, ctypes.c_void_p).value,
            )
            if not previous:
                return
            self._native_mouse_bridge_prev_wndproc = int(previous)
            self._native_mouse_bridge_callback = callback
        except Exception:
            self._native_mouse_bridge_prev_wndproc = None
            self._native_mouse_bridge_callback = None

    def _remove_embedded_native_mouse_bridge(self) -> None:
        if (
            self._native_hwnd is None
            or not sys.platform.startswith("win")
            or self._native_mouse_bridge_callback is None
            or self._native_mouse_bridge_prev_wndproc is None
        ):
            self._native_mouse_bridge_callback = None
            self._native_mouse_bridge_prev_wndproc = None
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.SetWindowLongPtrW.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_ssize_t,
            ]
            user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
            GWLP_WNDPROC = -4
            user32.SetWindowLongPtrW(
                wintypes.HWND(self._native_hwnd),
                GWLP_WNDPROC,
                ctypes.c_ssize_t(self._native_mouse_bridge_prev_wndproc),
            )
        except Exception:
            pass
        self._native_mouse_bridge_callback = None
        self._native_mouse_bridge_prev_wndproc = None
        self._embedded_native_left_drag_active = False
        self._set_native_mouse_capture(False)

    def resize_to_tk_host(self, tk_host) -> None:
        if not self.available or self._window is None:
            return
        self._embedded_tk_host = tk_host
        try:
            updater = getattr(tk_host, "update_idletasks", None)
            if callable(updater):
                updater()
        except Exception:
            pass
        try:
            width = max(int(tk_host.winfo_width()), 1)
            height = max(int(tk_host.winfo_height()), 1)
        except Exception:
            width = 1
            height = 1

        self._resize_native_window(width, height)
        try:
            self._window.resize(width, height)
        except Exception:
            pass
        self.process_events()

    def mount_into_tk(self, tk_host) -> bool:
        if not self.available or self._window is None:
            return False
        if not sys.platform.startswith("win"):
            self.error_message = (
                "in-place fast viewer embedding is currently supported on Windows only"
            )
            return False
        winfo_id = getattr(tk_host, "winfo_id", None)
        if not callable(winfo_id):
            self.error_message = "Tk host does not expose a native window handle"
            return False
        try:
            parent_hwnd = int(winfo_id())
        except Exception as exc:
            self.error_message = str(exc)
            return False
        if parent_hwnd <= 0:
            self.error_message = "Tk host returned an invalid native window handle"
            return False

        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
            user32.SetParent.argtypes = [wintypes.HWND, wintypes.HWND]
            user32.SetParent.restype = wintypes.HWND
            user32.SetWindowLongPtrW.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_ssize_t,
            ]
            user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
            user32.SetWindowPos.argtypes = [
                wintypes.HWND,
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint,
            ]
            user32.SetWindowPos.restype = wintypes.BOOL

            hwnd = self._window_handle()
            if hwnd is None or hwnd <= 0:
                raise RuntimeError("failed to obtain a native fast-viewer window handle")

            GWL_STYLE = -16
            GWL_EXSTYLE = -20
            WS_CHILD = 0x40000000
            WS_VISIBLE = 0x10000000
            WS_POPUP = 0x80000000
            WS_CAPTION = 0x00C00000
            WS_THICKFRAME = 0x00040000
            WS_MINIMIZEBOX = 0x00020000
            WS_MAXIMIZEBOX = 0x00010000
            WS_SYSMENU = 0x00080000
            WS_EX_APPWINDOW = 0x00040000
            WS_EX_WINDOWEDGE = 0x00000100
            WS_EX_CLIENTEDGE = 0x00000200
            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            SWP_FRAMECHANGED = 0x0020
            SWP_SHOWWINDOW = 0x0040

            style = int(user32.GetWindowLongPtrW(wintypes.HWND(hwnd), GWL_STYLE) or 0)
            exstyle = int(
                user32.GetWindowLongPtrW(wintypes.HWND(hwnd), GWL_EXSTYLE) or 0
            )
            style = (style | WS_CHILD | WS_VISIBLE) & ~(
                WS_POPUP
                | WS_CAPTION
                | WS_THICKFRAME
                | WS_MINIMIZEBOX
                | WS_MAXIMIZEBOX
                | WS_SYSMENU
            )
            exstyle = exstyle & ~(
                WS_EX_APPWINDOW | WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE
            )

            user32.SetParent(wintypes.HWND(hwnd), wintypes.HWND(parent_hwnd))
            user32.SetWindowLongPtrW(wintypes.HWND(hwnd), GWL_STYLE, style)
            user32.SetWindowLongPtrW(wintypes.HWND(hwnd), GWL_EXSTYLE, exstyle)
            self._window.show()
            user32.SetWindowPos(
                wintypes.HWND(hwnd),
                wintypes.HWND(0),
                0,
                0,
                1,
                1,
                SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED | SWP_SHOWWINDOW,
            )

            self._embedded_tk_host = tk_host
            self._embedded_parent_hwnd = parent_hwnd
            self._native_hwnd = hwnd
            self.error_message = None
            self._apply_native_input_enabled()
            self._install_embedded_native_mouse_bridge()
            self.resize_to_tk_host(tk_host)
            return True
        except Exception as exc:
            self.error_message = str(exc)
            return False

    def _new_image_item(self):
        assert self._pg is not None
        try:
            return self._pg.ImageItem(axisOrder="row-major")
        except TypeError:
            return self._pg.ImageItem()

    def _set_item_transform(self, item, shape: tuple[int, int], extent: tuple[float, ...] | None) -> None:
        if self._QtGui is None:
            return
        rows, cols = shape
        if rows <= 0 or cols <= 0:
            return
        if extent is None:
            x0, x1, y0, y1 = 0.0, float(cols), float(rows), 0.0
        else:
            x0, x1, y0, y1 = map(float, extent)

        scale_x = (x1 - x0) / float(cols)
        scale_y = (y1 - y0) / float(rows)

        transform = self._QtGui.QTransform()
        transform.translate(float(x0), float(y0))
        transform.scale(float(scale_x), float(scale_y))
        item.setTransform(transform)

    def _float32_image_buffer(
        self,
        array_like,
        *,
        buffer_key: str,
    ) -> np.ndarray | None:
        arr = _coerce_2d_array(array_like)
        if arr is None:
            return None
        shape = tuple(int(value) for value in arr.shape)
        buffer = self._image_buffers.get(buffer_key)
        if buffer is None or tuple(buffer.shape) != shape:
            buffer = np.empty(shape, dtype=np.float32)
            self._image_buffers[buffer_key] = buffer
        np.copyto(buffer, arr, casting="unsafe")
        np.nan_to_num(buffer, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return buffer

    def _set_image_item(
        self,
        *,
        cache_key: str,
        item,
        array,
        visible: bool,
        levels: tuple[float, float] | None,
        alpha: float,
        extent: tuple[float, ...] | None,
        version: object = None,
    ) -> None:
        if not self.available:
            return

        normalized_levels = _normalize_levels(levels)
        normalized_extent = _normalize_extent(extent)
        state = _ImageSyncState(
            visible=bool(visible),
            version=_raw_array_signature(array, version=version),
            levels=normalized_levels,
            alpha=float(np.clip(alpha, 0.0, 1.0)),
            extent=normalized_extent,
        )
        if self._image_sync_states.get(cache_key) == state:
            return

        image = self._float32_image_buffer(array, buffer_key=cache_key)
        if not visible or image is None:
            item.setVisible(False)
            self._image_sync_states[cache_key] = state
            return

        if normalized_levels is not None:
            lo, hi = normalized_levels
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                item.setImage(image, autoLevels=False, levels=(lo, hi))
            else:
                item.setImage(image, autoLevels=True)
        else:
            item.setImage(image, autoLevels=True)

        item.setOpacity(state.alpha)
        self._set_item_transform(item, image.shape, normalized_extent)
        item.setVisible(True)
        self._image_sync_states[cache_key] = state

    def _qt_pen_style(self, linestyle: object):
        qt = getattr(self._QtCore, "Qt", None)
        if qt is None:
            return None
        pen_style = getattr(qt, "PenStyle", None)

        def _enum(name: str):
            if pen_style is not None:
                value = getattr(pen_style, name, None)
                if value is not None:
                    return value
            return getattr(qt, name, None)

        normalized = str(linestyle or "-").strip().lower()
        if normalized in {"--", "dashed", "dash"}:
            return _enum("DashLine")
        if normalized in {":", "dotted", "dot"}:
            return _enum("DotLine")
        if normalized in {"-.", "dashdot", "dash-dot"}:
            return _enum("DashDotLine")
        return _enum("SolidLine")

    def _make_pen(self, *, color: tuple[int, int, int, int], width: float, linestyle: object):
        assert self._pg is not None
        pen = self._pg.mkPen(color=color, width=max(float(width), 0.5))
        style = self._qt_pen_style(linestyle)
        if style is not None:
            try:
                pen.setStyle(style)
            except Exception:
                pass
        return pen

    def _ensure_rectangle_items(self, count: int) -> None:
        assert self._pg is not None
        while len(self._rectangle_items) < int(count):
            item = self._pg.PlotCurveItem()
            item.setVisible(False)
            self._plot_item.addItem(item)
            self._rectangle_items.append(item)

    def _ensure_transient_rectangle_items(self, count: int) -> None:
        assert self._pg is not None
        while len(self._transient_rectangle_items) < int(count):
            item = self._pg.PlotCurveItem()
            item.setVisible(False)
            self._plot_item.addItem(item)
            self._transient_rectangle_items.append(item)

    def _sync_rectangle_specs(self, specs: tuple[_RectangleOverlaySpec, ...], items: list[object]) -> None:
        if not self.available or self._pg is None or self._plot_item is None:
            return

        for item, spec in zip(items, specs):
            x0 = float(spec.x)
            y0 = float(spec.y)
            x1 = x0 + float(spec.width)
            y1 = y0 + float(spec.height)
            item.setData(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
            )
            item.setPen(
                self._make_pen(
                    color=spec.edge_rgba,
                    width=spec.linewidth,
                    linestyle=spec.linestyle,
                )
            )
            item.setZValue(float(spec.zorder))
            item.setVisible(True)

        for item in items[len(specs) :]:
            item.setVisible(False)

    def _sync_rectangle_items(self, ax) -> None:
        specs = _extract_visible_rectangle_specs(ax)
        if self._rectangle_specs == specs:
            return
        self._ensure_rectangle_items(len(specs))
        self._sync_rectangle_specs(specs, self._rectangle_items)
        self._rectangle_specs = specs

    def _ensure_marker_items(self, count: int) -> None:
        assert self._pg is not None
        while len(self._marker_items) < int(count):
            item = self._pg.ScatterPlotItem()
            item.setVisible(False)
            self._plot_item.addItem(item)
            self._marker_items.append(item)

    def _ensure_transient_marker_items(self, count: int) -> None:
        assert self._pg is not None
        while len(self._transient_marker_items) < int(count):
            item = self._pg.ScatterPlotItem()
            item.setVisible(False)
            self._plot_item.addItem(item)
            self._transient_marker_items.append(item)

    def _sync_marker_specs(self, specs: tuple[_MarkerOverlaySpec, ...], items: list[object]) -> None:
        if not self.available or self._pg is None or self._plot_item is None:
            return

        for item, spec in zip(items, specs):
            item.setData(
                x=list(spec.x_values),
                y=list(spec.y_values),
                symbol=spec.symbol,
                size=float(spec.size),
                pen=self._make_pen(
                    color=spec.edge_rgba,
                    width=spec.pen_width,
                    linestyle="-",
                ),
                brush=self._pg.mkBrush(spec.face_rgba),
            )
            item.setZValue(float(spec.zorder))
            item.setVisible(True)

        for item in items[len(specs) :]:
            item.setVisible(False)

    def _sync_marker_items(self, marker_artist) -> None:
        specs = _extract_visible_marker_specs(marker_artist)
        if self._marker_specs == specs:
            return
        self._ensure_marker_items(len(specs))
        self._sync_marker_specs(specs, self._marker_items)
        self._marker_specs = specs

    def _ensure_transient_curve_items(self, count: int) -> None:
        assert self._pg is not None
        while len(self._transient_curve_items) < int(count):
            item = self._pg.PlotCurveItem()
            item.setVisible(False)
            self._plot_item.addItem(item)
            self._transient_curve_items.append(item)

    def _sync_curve_specs(
        self,
        specs: tuple[FastViewerCurveSpec, ...],
        items: list[object],
    ) -> None:
        if not self.available or self._pg is None or self._plot_item is None:
            return
        for item, spec in zip(items, specs):
            item.setData(x=list(spec.x_values), y=list(spec.y_values))
            item.setPen(
                self._make_pen(
                    color=spec.edge_rgba,
                    width=spec.linewidth,
                    linestyle=spec.linestyle,
                )
            )
            item.setZValue(float(spec.zorder))
            item.setVisible(True)

        for item in items[len(specs) :]:
            item.setVisible(False)

    def _sync_overlay_model(self, overlay_model: FastViewerOverlayModel) -> None:
        if self._rectangle_specs != overlay_model.rectangle_specs:
            self._ensure_rectangle_items(len(overlay_model.rectangle_specs))
            self._sync_rectangle_specs(
                overlay_model.rectangle_specs,
                self._rectangle_items,
            )
            self._rectangle_specs = overlay_model.rectangle_specs

        if self._transient_rectangle_specs != overlay_model.transient_rectangle_specs:
            self._ensure_transient_rectangle_items(
                len(overlay_model.transient_rectangle_specs)
            )
            self._sync_rectangle_specs(
                overlay_model.transient_rectangle_specs,
                self._transient_rectangle_items,
            )
            self._transient_rectangle_specs = overlay_model.transient_rectangle_specs

        if self._transient_marker_specs != overlay_model.transient_marker_specs:
            self._ensure_transient_marker_items(
                len(overlay_model.transient_marker_specs)
            )
            self._sync_marker_specs(
                overlay_model.transient_marker_specs,
                self._transient_marker_items,
            )
            self._transient_marker_specs = overlay_model.transient_marker_specs

        if self._transient_curve_specs != overlay_model.transient_curve_specs:
            self._ensure_transient_curve_items(len(overlay_model.transient_curve_specs))
            self._sync_curve_specs(
                overlay_model.transient_curve_specs,
                self._transient_curve_items,
            )
            self._transient_curve_specs = overlay_model.transient_curve_specs

    def _current_view_range(self) -> tuple[float, float, float, float] | None:
        if self._plot_item is None:
            return None
        try:
            x_range, y_range = self._plot_item.getViewBox().viewRange()
        except Exception:
            return None
        if len(x_range) < 2 or len(y_range) < 2:
            return None
        return _normalize_view_range(x_range[0], x_range[1], y_range[0], y_range[1])

    def current_view_range(self) -> tuple[float, float, float, float] | None:
        return self._current_view_range()

    def viewport_global_geometry(self) -> tuple[int, int, int, int] | None:
        viewport = self._plot_viewport()
        if viewport is None:
            return None
        try:
            rect = viewport.rect()
            top_left = viewport.mapToGlobal(rect.topLeft())
        except Exception:
            return None
        try:
            width = max(int(rect.width()), 1)
            height = max(int(rect.height()), 1)
            return (
                int(top_left.x()),
                int(top_left.y()),
                width,
                height,
            )
        except Exception:
            return None

    def map_viewport_pixels_to_view_coords(
        self,
        x_pixel: float,
        y_pixel: float,
    ) -> tuple[float, float, bool] | None:
        """Map viewport-local pixels into view coordinates and plot hit state."""

        if self._plot_widget is None or self._plot_item is None:
            return None

        try:
            x_value = int(round(float(x_pixel)))
            y_value = int(round(float(y_pixel)))
        except Exception:
            return None

        try:
            scene_pos = self._plot_widget.mapToScene(x_value, y_value)
        except TypeError:
            qtcore = self._QtCore
            point_cls = getattr(qtcore, "QPoint", None) if qtcore is not None else None
            if point_cls is None:
                point_cls = getattr(qtcore, "QPointF", None) if qtcore is not None else None
            if point_cls is None:
                return None
            try:
                scene_pos = self._plot_widget.mapToScene(point_cls(x_value, y_value))
            except Exception:
                return None
        except Exception:
            return None

        try:
            view_box = getattr(self._plot_item, "vb", None)
            if view_box is None:
                return None
            view_pos = view_box.mapSceneToView(scene_pos)
            in_plot = bool(view_box.sceneBoundingRect().contains(scene_pos))
            return (float(view_pos.x()), float(view_pos.y()), in_plot)
        except Exception:
            return None

    def set_mouse_event_callback(self, callback: Callable[..., None] | None) -> None:
        """Set callback for normalized mouse events.

        Callback signature:
        ``callback(event_name, x_data, y_data, x_pixel, y_pixel, button, dblclick, in_plot)``
        """
        self._mouse_event_callback = callback

    def _plot_viewport(self):
        if self._plot_widget is None:
            return None
        try:
            return self._plot_widget.viewport()
        except Exception:
            return None

    def _sync_mouse_capture_overlay_geometry(self) -> None:
        overlay = getattr(self, "_mouse_capture_overlay", None)
        viewport = self._plot_viewport()
        if overlay is None or viewport is None:
            return
        try:
            overlay.setGeometry(viewport.rect())
            overlay.raise_()
            overlay.show()
        except Exception:
            return

    def _install_mouse_capture_overlay(self) -> None:
        QtCore = self._QtCore
        QtWidgets = self._QtWidgets
        viewport = self._plot_viewport()
        if QtCore is None or QtWidgets is None or viewport is None:
            return

        left_button = getattr(
            getattr(QtCore.Qt, "MouseButton", None),
            "LeftButton",
            getattr(QtCore.Qt, "LeftButton", None),
        )
        widget_attribute = getattr(QtCore.Qt, "WidgetAttribute", None)
        translucent_attr = (
            getattr(widget_attribute, "WA_TranslucentBackground", None)
            if widget_attribute is not None
            else getattr(QtCore.Qt, "WA_TranslucentBackground", None)
        )
        no_system_background_attr = (
            getattr(widget_attribute, "WA_NoSystemBackground", None)
            if widget_attribute is not None
            else getattr(QtCore.Qt, "WA_NoSystemBackground", None)
        )
        qevent = getattr(QtCore, "QEvent", None)
        resize_event_type = getattr(qevent, "Resize", None) if qevent is not None else None
        show_event_type = getattr(qevent, "Show", None) if qevent is not None else None

        viewer = self

        class _MouseCaptureOverlay(QtWidgets.QWidget):  # pragma: no cover - UI
            def __init__(self, parent) -> None:
                super().__init__(parent)
                self._left_drag_active = False
                self.setMouseTracking(True)
                self.setAutoFillBackground(False)
                if translucent_attr is not None:
                    self.setAttribute(translucent_attr, True)
                if no_system_background_attr is not None:
                    self.setAttribute(no_system_background_attr, True)
                self.setStyleSheet("background: transparent;")

            def eventFilter(self, watched, event):
                if watched is viewport:
                    try:
                        event_type = event.type()
                    except Exception:
                        return False
                    if event_type == resize_event_type or event_type == show_event_type:
                        viewer._sync_mouse_capture_overlay_geometry()
                return False

            def _dispatch_left_mouse_event(
                self,
                event_name: str,
                event,
                *,
                dblclick: bool,
            ) -> None:
                if viewer._mouse_event_callback is None:
                    event.ignore()
                    return
                viewer._forward_mouse_event_from_qpoint(
                    event_name,
                    event.pos(),
                    button=getattr(event, "button", lambda: None)(),
                    dblclick=dblclick,
                )
                event.accept()

            def mousePressEvent(self, event) -> None:
                if not _qt_button_mask_matches(
                    getattr(event, "button", lambda: None)(),
                    left_button,
                ):
                    event.ignore()
                    return
                self._left_drag_active = True
                self._dispatch_left_mouse_event(
                    "button_press_event",
                    event,
                    dblclick=False,
                )

            def mouseDoubleClickEvent(self, event) -> None:
                if not _qt_button_mask_matches(
                    getattr(event, "button", lambda: None)(),
                    left_button,
                ):
                    event.ignore()
                    return
                self._left_drag_active = True
                self._dispatch_left_mouse_event(
                    "button_press_event",
                    event,
                    dblclick=True,
                )

            def mouseMoveEvent(self, event) -> None:
                buttons = getattr(event, "buttons", lambda: None)()
                dragging = _qt_button_mask_matches(buttons, left_button)
                if not (self._left_drag_active or dragging):
                    event.ignore()
                    return
                self._left_drag_active = dragging
                self._dispatch_left_mouse_event(
                    "motion_notify_event",
                    event,
                    dblclick=False,
                )

            def mouseReleaseEvent(self, event) -> None:
                if not (
                    self._left_drag_active
                    or _qt_button_mask_matches(
                        getattr(event, "button", lambda: None)(),
                        left_button,
                    )
                ):
                    event.ignore()
                    return
                self._left_drag_active = False
                self._dispatch_left_mouse_event(
                    "button_release_event",
                    event,
                    dblclick=False,
                )

            def wheelEvent(self, event) -> None:
                event.ignore()

        overlay = _MouseCaptureOverlay(viewport)
        viewport.installEventFilter(overlay)
        self._mouse_capture_overlay = overlay
        self._sync_mouse_capture_overlay_geometry()

    def _extract_mouse_qpoint(self, event):
        try:
            pos = event.position()
            return pos.toPoint()
        except Exception:
            pass
        try:
            return event.pos()
        except Exception:
            return None

    def _forward_mouse_event_from_qpoint(
        self,
        event_name: str,
        qpoint,
        *,
        button: object,
        dblclick: bool,
    ) -> None:
        if (
            self._mouse_event_callback is None
            or self._plot_widget is None
            or self._plot_item is None
            or qpoint is None
        ):
            return
        try:
            view_box = getattr(self._plot_item, "vb", None)
            if view_box is None:
                return
            scene_pos = self._plot_widget.mapToScene(qpoint)
            view_pos = view_box.mapSceneToView(scene_pos)
            in_plot = bool(view_box.sceneBoundingRect().contains(scene_pos))
            self._mouse_event_callback(
                event_name,
                float(view_pos.x()),
                float(view_pos.y()),
                float(qpoint.x()),
                float(qpoint.y()),
                _qt_mouse_button_to_mpl(button),
                bool(dblclick),
                in_plot,
            )
        except Exception:
            return

    def _make_viewport_event_filter(self):
        QtCore = self._QtCore
        viewer = self
        left_button = getattr(
            getattr(QtCore.Qt, "MouseButton", None),
            "LeftButton",
            getattr(QtCore.Qt, "LeftButton", None),
        )

        class _ViewportEventFilter(QtCore.QObject):  # pragma: no cover - UI
            def eventFilter(self, watched, event):
                if not bool(getattr(viewer, "available", False)) or getattr(
                    viewer, "_plot_widget", None
                ) is None:
                    return False
                try:
                    event_type = int(event.type())
                except Exception:
                    return False

                button = None
                buttons = None
                try:
                    button = getattr(event, "button", lambda: None)()
                except Exception:
                    button = None
                try:
                    buttons = getattr(event, "buttons", lambda: None)()
                except Exception:
                    buttons = None

                # Qt event type ints:
                # 2  -> MouseButtonPress
                # 3  -> MouseButtonRelease
                # 4  -> MouseButtonDblClick
                # 5  -> MouseMove
                if event_type == 2:
                    viewer._forward_mouse_event("button_press_event", event, dblclick=False)
                elif event_type == 3:
                    viewer._forward_mouse_event("button_release_event", event, dblclick=False)
                elif event_type == 4:
                    viewer._forward_mouse_event("button_press_event", event, dblclick=True)
                elif event_type == 5:
                    viewer._forward_mouse_event("motion_notify_event", event, dblclick=False)
                return _should_consume_fast_viewer_viewport_event(
                    event_type,
                    button=button,
                    buttons=buttons,
                    left_button=left_button,
                )

        return _ViewportEventFilter()

    def _forward_mouse_event(self, event_name: str, event, dblclick: bool) -> None:
        self._forward_mouse_event_from_qpoint(
            event_name,
            self._extract_mouse_qpoint(event),
            button=getattr(event, "button", lambda: None)(),
            dblclick=dblclick,
        )

    def update_from_matplotlib(
        self,
        *,
        ax,
        image_artist,
        background_artist,
        overlay_artist,
        marker_artist=None,
        overlay_model: object = None,
        layer_versions: dict[str, object] | None = None,
        force_view_range: bool = False,
    ) -> None:
        if not self.available:
            return

        overlay_model_provided = isinstance(overlay_model, FastViewerOverlayModel)
        overlay_model = _coerce_overlay_model(overlay_model)
        layer_versions = dict(layer_versions or {})

        sim_visible = bool(image_artist.get_visible())
        bg_visible = bool(background_artist.get_visible())
        overlay_visible = bool(overlay_artist.get_visible()) and not bool(
            overlay_model.suppress_overlay_image
        )

        sim_alpha = image_artist.get_alpha()
        bg_alpha = background_artist.get_alpha()
        if sim_alpha is None:
            sim_alpha = 1.0
        if bg_alpha is None:
            bg_alpha = 1.0

        self._set_image_item(
            cache_key="background",
            item=self._background_item,
            array=background_artist.get_array(),
            visible=bg_visible,
            levels=background_artist.get_clim(),
            alpha=float(bg_alpha),
            extent=background_artist.get_extent(),
            version=layer_versions.get("background"),
        )
        self._set_image_item(
            cache_key="simulation",
            item=self._simulation_item,
            array=image_artist.get_array(),
            visible=sim_visible,
            levels=image_artist.get_clim(),
            alpha=float(sim_alpha),
            extent=image_artist.get_extent(),
            version=layer_versions.get("simulation"),
        )
        self._set_image_item(
            cache_key="overlay",
            item=self._overlay_item,
            array=overlay_artist.get_array(),
            visible=overlay_visible,
            levels=(0.0, 1.0),
            alpha=1.0,
            extent=overlay_artist.get_extent(),
            version=layer_versions.get("overlay"),
        )

        self._sync_marker_items(marker_artist)
        if overlay_model_provided:
            self._sync_overlay_model(overlay_model)
        else:
            self._sync_rectangle_items(ax)
            if self._transient_rectangle_specs:
                self._sync_rectangle_specs((), self._transient_rectangle_items)
                self._transient_rectangle_specs = ()
            if self._transient_marker_specs:
                self._sync_marker_specs((), self._transient_marker_items)
                self._transient_marker_specs = ()
            if self._transient_curve_specs:
                self._sync_curve_specs((), self._transient_curve_items)
                self._transient_curve_specs = ()

        title = str(ax.get_title())
        xlabel = str(ax.get_xlabel() or "X (pixels)")
        ylabel = str(ax.get_ylabel() or "Y (pixels)")
        if title != self._last_title:
            self._plot_item.setTitle(title)
            self._last_title = title
        if xlabel != self._last_xlabel:
            self._plot_item.setLabel("bottom", xlabel)
            self._last_xlabel = xlabel
        if ylabel != self._last_ylabel:
            self._plot_item.setLabel("left", ylabel)
            self._last_ylabel = ylabel

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        target_range = _normalize_view_range(x0, x1, y0, y1)
        current_range = self._current_view_range()
        should_apply_range = bool(force_view_range)
        if not should_apply_range:
            should_apply_range = _should_apply_matplotlib_view_range(
                current_range=current_range,
                last_synced_range=self._last_synced_view_range,
                target_range=target_range,
            )
        if should_apply_range:
            self._plot_item.getViewBox().setRange(
                xRange=(float(x0), float(x1)),
                yRange=(float(y0), float(y1)),
                padding=0.0,
                disableAutoRange=True,
            )
            self._last_synced_view_range = target_range

    def process_events(self) -> None:
        if not self.available or self._app is None:
            return
        try:
            self._app.processEvents()
        except Exception:
            pass

    def close(self) -> None:
        if not self.available or self._window is None:
            return
        overlay = getattr(self, "_mouse_capture_overlay", None)
        if overlay is not None:
            try:
                overlay.close()
            except Exception:
                pass
            self._mouse_capture_overlay = None
        try:
            self._window.close()
        except Exception:
            pass
        self._embedded_tk_host = None
        self._embedded_parent_hwnd = None
        self._remove_embedded_native_mouse_bridge()
        self._native_hwnd = None
        self.process_events()


class MatplotlibCanvasProxy:
    """Proxy object that mirrors Matplotlib image updates into a fast viewer."""

    def __init__(
        self,
        canvas,
        fast_viewer: FastPlotViewer,
        *,
        draw_interval_s: float = 0.12,
        sync_interval_s: float = 1.0 / 60.0,
        event_pump_interval_s: float | None = None,
        render_matplotlib: bool = True,
        event_axes=None,
    ) -> None:
        self._canvas = canvas
        self._fast_viewer = fast_viewer
        self._draw_interval_s = float(max(0.0, draw_interval_s))
        self._sync_interval_s = float(max(0.0, sync_interval_s))
        if event_pump_interval_s is None:
            event_pump_interval_s = self._sync_interval_s
        self._event_pump_interval_s = float(max(0.0, event_pump_interval_s))
        self._last_draw_ts = 0.0
        self._last_fast_viewer_sync_ts = -float("inf")
        self._last_fast_event_pump_ts = -float("inf")
        self._sync_callback: Callable[[], None] | None = None
        self._render_matplotlib = bool(render_matplotlib)
        self._event_axes = event_axes
        self._callbacks: dict[str, dict[int, Callable]] = {}
        self._next_cid = 1
        self._fast_event_dispatch_depth = 0
        self._pending_fast_viewer_sync: dict[str, object] | None = None
        self._fast_viewer_sync_scheduled = False
        if not self._render_matplotlib:
            self._fast_viewer.set_mouse_event_callback(self._dispatch_mouse_event)

    @property
    def fast_viewer(self) -> FastPlotViewer:
        return self._fast_viewer

    def set_sync_callback(self, callback: Callable[[], None]) -> None:
        self._sync_callback = callback

    def process_fast_events(self, *, force: bool = False) -> None:
        if self._fast_event_dispatch_depth > 0:
            return
        if not bool(force) and self._event_pump_interval_s > 0.0:
            now = time.perf_counter()
            if (now - self._last_fast_event_pump_ts) < self._event_pump_interval_s:
                return
            self._last_fast_event_pump_ts = now
        elif bool(force):
            self._last_fast_event_pump_ts = time.perf_counter()
        self._fast_viewer.process_events()

    def sync_from_matplotlib(
        self,
        *,
        ax,
        image_artist,
        background_artist,
        overlay_artist,
        marker_artist=None,
        overlay_model: object = None,
        layer_versions: dict[str, object] | None = None,
        force_view_range: bool = False,
    ) -> None:
        kwargs: dict[str, object] = {
            "ax": ax,
            "image_artist": image_artist,
            "background_artist": background_artist,
            "overlay_artist": overlay_artist,
            "marker_artist": marker_artist,
            "overlay_model": overlay_model,
            "layer_versions": dict(layer_versions or {}),
            "force_view_range": bool(force_view_range),
        }
        self._pending_fast_viewer_sync = self._merge_pending_sync_kwargs(
            self._pending_fast_viewer_sync,
            kwargs,
        )
        if self._fast_event_dispatch_depth > 0:
            self._schedule_deferred_fast_viewer_sync()
            return
        if not self._schedule_deferred_fast_viewer_sync():
            self._flush_deferred_fast_viewer_sync()

    def _merge_pending_sync_kwargs(
        self,
        current: dict[str, object] | None,
        incoming: dict[str, object],
    ) -> dict[str, object]:
        if current is None:
            return incoming
        merged = dict(incoming)
        merged["force_view_range"] = bool(
            current.get("force_view_range") or incoming.get("force_view_range")
        )
        return merged

    def _apply_fast_viewer_sync(self, kwargs: dict[str, object]) -> None:
        self._fast_viewer.update_from_matplotlib(
            ax=kwargs["ax"],
            image_artist=kwargs["image_artist"],
            background_artist=kwargs["background_artist"],
            overlay_artist=kwargs["overlay_artist"],
            marker_artist=kwargs["marker_artist"],
            overlay_model=kwargs.get("overlay_model"),
            layer_versions=kwargs.get("layer_versions"),
            force_view_range=bool(kwargs["force_view_range"]),
        )
        now = time.perf_counter()
        self._last_fast_viewer_sync_ts = now
        self._last_fast_event_pump_ts = now

    def _schedule_deferred_fast_viewer_sync(self) -> bool:
        if self._fast_viewer_sync_scheduled:
            return True
        qtcore = getattr(self._fast_viewer, "_QtCore", None)
        timer = getattr(qtcore, "QTimer", None) if qtcore is not None else None
        if timer is None:
            return False
        delay_ms = 0
        if self._sync_interval_s > 0.0:
            elapsed = time.perf_counter() - self._last_fast_viewer_sync_ts
            remaining_s = max(0.0, self._sync_interval_s - elapsed)
            delay_ms = max(0, int(round(remaining_s * 1000.0)))
        try:
            self._fast_viewer_sync_scheduled = True
            timer.singleShot(delay_ms, self._flush_deferred_fast_viewer_sync)
            return True
        except Exception:
            self._fast_viewer_sync_scheduled = False
            return False

    def _flush_deferred_fast_viewer_sync(self) -> None:
        self._fast_viewer_sync_scheduled = False
        if self._fast_event_dispatch_depth > 0:
            self._schedule_deferred_fast_viewer_sync()
            return
        kwargs = self._pending_fast_viewer_sync
        self._pending_fast_viewer_sync = None
        if kwargs is None:
            return
        self._apply_fast_viewer_sync(kwargs)
        self.process_fast_events(force=True)

    def _sync(self) -> None:
        if self._sync_callback is not None:
            try:
                self._sync_callback()
            except Exception:
                pass
        self.process_fast_events()

    def _should_draw_matplotlib(self, *, force: bool) -> bool:
        if force:
            return True
        if self._draw_interval_s <= 0.0:
            return True
        now = time.perf_counter()
        if (now - self._last_draw_ts) >= self._draw_interval_s:
            self._last_draw_ts = now
            return True
        return False

    def draw(self, *args, **kwargs):
        self._sync()
        if not self._render_matplotlib:
            return None
        if self._should_draw_matplotlib(force=True):
            self._last_draw_ts = time.perf_counter()
            return self._canvas.draw(*args, **kwargs)
        return None

    def draw_idle(self, *args, **kwargs):
        self._sync()
        if not self._render_matplotlib:
            return None
        if self._should_draw_matplotlib(force=False):
            return self._canvas.draw_idle(*args, **kwargs)
        return None

    def mpl_connect(self, event_name: str, callback: Callable):
        if self._render_matplotlib:
            return self._canvas.mpl_connect(event_name, callback)
        cid = self._next_cid
        self._next_cid += 1
        bucket = self._callbacks.setdefault(str(event_name), {})
        bucket[cid] = callback
        return cid

    def mpl_disconnect(self, cid: int):
        if self._render_matplotlib:
            return self._canvas.mpl_disconnect(cid)
        for bucket in self._callbacks.values():
            if cid in bucket:
                del bucket[cid]
                break
        return None

    def _dispatch_mouse_event(
        self,
        event_name: str,
        x_data: float,
        y_data: float,
        x_pixel: float,
        y_pixel: float,
        button: int | None,
        dblclick: bool,
        in_plot: bool,
    ) -> None:
        callbacks = list(self._callbacks.get(event_name, {}).values())
        if not callbacks:
            return
        inaxes = self._event_axes if in_plot else None
        event = SimpleNamespace(
            name=event_name,
            x=(float(x_pixel) if np.isfinite(float(x_pixel)) else None),
            y=(float(y_pixel) if np.isfinite(float(y_pixel)) else None),
            xdata=(float(x_data) if inaxes is not None else None),
            ydata=(float(y_data) if inaxes is not None else None),
            button=button,
            dblclick=bool(dblclick),
            inaxes=inaxes,
        )
        self._fast_event_dispatch_depth += 1
        try:
            for callback in callbacks:
                try:
                    callback(event)
                except Exception:
                    pass
        finally:
            self._fast_event_dispatch_depth = max(
                0,
                self._fast_event_dispatch_depth - 1,
            )
            if (
                self._fast_event_dispatch_depth == 0
                and self._pending_fast_viewer_sync is not None
                and not self._fast_viewer_sync_scheduled
            ):
                if not self._schedule_deferred_fast_viewer_sync():
                    self._flush_deferred_fast_viewer_sync()

    def dispatch_host_overlay_event(
        self,
        event_name: str,
        *,
        x_pixel: float,
        y_pixel: float,
        width: float,
        height: float,
        button: int | None,
        dblclick: bool,
    ) -> None:
        try:
            width_value = float(width)
            height_value = float(height)
            x_value = float(x_pixel)
            y_value = float(y_pixel)
        except Exception:
            return
        if not (
            np.isfinite(width_value)
            and np.isfinite(height_value)
            and np.isfinite(x_value)
            and np.isfinite(y_value)
        ):
            return
        if width_value <= 0.0 or height_value <= 0.0:
            return

        coordinate_mapper = getattr(
            self._fast_viewer,
            "map_viewport_pixels_to_view_coords",
            None,
        )
        if callable(coordinate_mapper):
            try:
                mapped = coordinate_mapper(x_value, y_value)
            except Exception:
                mapped = None
            if isinstance(mapped, tuple) and len(mapped) == 3:
                try:
                    x_data = float(mapped[0])
                    y_data = float(mapped[1])
                    in_plot = bool(mapped[2])
                except Exception:
                    x_data = None
                    y_data = None
                else:
                    self._dispatch_mouse_event(
                        event_name,
                        x_data,
                        y_data,
                        x_value,
                        y_value,
                        button,
                        bool(dblclick),
                        in_plot,
                    )
                    return

        in_plot = 0.0 <= x_value <= width_value and 0.0 <= y_value <= height_value
        view_range_getter = getattr(self._fast_viewer, "current_view_range", None)
        view_range = view_range_getter() if callable(view_range_getter) else None
        if view_range is None:
            axis = self._event_axes
            if axis is None:
                return
            try:
                x0, x1 = axis.get_xlim()
                y0, y1 = axis.get_ylim()
            except Exception:
                return
            view_range = _normalize_view_range(x0, x1, y0, y1)
        if view_range is None:
            return

        x_denominator = max(width_value - 1.0, 1.0)
        y_denominator = max(height_value - 1.0, 1.0)
        x_fraction = float(np.clip(x_value / x_denominator, 0.0, 1.0))
        y_fraction = float(np.clip(y_value / y_denominator, 0.0, 1.0))
        x0, x1, y0, y1 = view_range
        x_data = float(x0 + ((x1 - x0) * x_fraction))
        y_data = float(y0 + ((y1 - y0) * y_fraction))
        self._dispatch_mouse_event(
            event_name,
            x_data,
            y_data,
            x_value,
            y_value,
            button,
            bool(dblclick),
            bool(in_plot),
        )

    def __getattr__(self, name):
        return getattr(self._canvas, name)
