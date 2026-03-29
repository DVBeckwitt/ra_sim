"""Fast optional 2D viewer for high-frequency image updates.

This module is safe to import without Qt/PyQtGraph installed. GUI imports are
done lazily at runtime.
"""

from __future__ import annotations

import time
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


def _to_2d_float32(array_like) -> np.ndarray | None:
    if array_like is None:
        return None
    arr = np.asarray(array_like)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.size == 0:
        return None
    if np.ma.isMaskedArray(arr):
        arr = np.ma.filled(arr, 0.0)
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


class FastPlotViewer:
    """High-performance 2D viewer powered by PyQtGraph."""

    def __init__(self, *, title: str = "RA-SIM Fast Viewer", use_opengl: bool = True) -> None:
        self.available = False
        self.error_message: str | None = None

        self._pg = None
        self._QtGui = None
        self._app = None
        self._window = None
        self._plot_widget = None
        self._plot_item = None
        self._background_item = None
        self._simulation_item = None
        self._overlay_item = None
        self._marker_item = None
        self._mouse_event_callback: Callable[..., None] | None = None

        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
        except Exception as exc:
            self.error_message = str(exc)
            return

        self._pg = pg
        self._QtGui = QtGui
        self._QtCore = QtCore

        try:
            pg.setConfigOptions(antialias=False, useOpenGL=bool(use_opengl))
        except Exception:
            pg.setConfigOptions(antialias=False)

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        self._app = app

        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle(title)
        self._window.resize(1050, 850)

        self._plot_widget = pg.PlotWidget(background="w")
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self._plot_widget.getPlotItem().getViewBox().setAspectLocked(False)
        self._plot_widget.setMouseTracking(True)
        self._window.setCentralWidget(self._plot_widget)
        self._viewport_event_filter = self._make_viewport_event_filter()
        self._plot_widget.viewport().installEventFilter(self._viewport_event_filter)

        self._plot_item = self._plot_widget.getPlotItem()
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

        self._marker_item = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen((255, 255, 0, 255), width=2),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        self._marker_item.setZValue(10)
        self._marker_item.setVisible(False)
        self._plot_item.addItem(self._marker_item)

        self._background_item.setLookupTable(_build_turbo_lut(zero_white=False))
        self._simulation_item.setLookupTable(_build_turbo_lut(zero_white=True))
        self._overlay_item.setLookupTable(
            np.asarray(
                [
                    [0, 0, 0, 0],
                    [0, 255, 255, 110],
                ],
                dtype=np.uint8,
            )
        )

        self._window.show()
        self.available = True
        self.process_events()

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

    def _set_image_item(
        self,
        *,
        item,
        array,
        visible: bool,
        levels: tuple[float, float] | None,
        alpha: float,
        extent: tuple[float, ...] | None,
    ) -> None:
        if not self.available:
            return

        image = _to_2d_float32(array)
        if not visible or image is None:
            item.setVisible(False)
            return

        if levels is not None:
            lo, hi = float(levels[0]), float(levels[1])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                item.setImage(image, autoLevels=False, levels=(lo, hi))
            else:
                item.setImage(image, autoLevels=True)
        else:
            item.setImage(image, autoLevels=True)

        item.setOpacity(float(np.clip(alpha, 0.0, 1.0)))
        self._set_item_transform(item, image.shape, extent)
        item.setVisible(True)

    def set_mouse_event_callback(self, callback: Callable[..., None] | None) -> None:
        """Set callback for normalized mouse events.

        Callback signature:
        ``callback(event_name, x_data, y_data, x_pixel, y_pixel, button, dblclick, in_plot)``
        """
        self._mouse_event_callback = callback

    def _make_viewport_event_filter(self):
        QtCore = self._QtCore
        viewer = self

        class _ViewportEventFilter(QtCore.QObject):  # pragma: no cover - UI
            def eventFilter(self, watched, event):
                if not viewer.available or viewer._plot_widget is None:
                    return False
                try:
                    event_type = int(event.type())
                except Exception:
                    return False

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
                return False

        return _ViewportEventFilter()

    def _forward_mouse_event(self, event_name: str, event, dblclick: bool) -> None:
        if self._mouse_event_callback is None or self._plot_widget is None:
            return
        try:
            pos = event.position()
            qpoint = pos.toPoint()
        except Exception:
            try:
                qpoint = event.pos()
            except Exception:
                return
        scene_pos = self._plot_widget.mapToScene(qpoint)
        view_pos = self._plot_item.vb.mapSceneToView(scene_pos)
        in_plot = bool(self._plot_item.vb.sceneBoundingRect().contains(scene_pos))
        button = _qt_mouse_button_to_mpl(getattr(event, "button", lambda: None)())
        self._mouse_event_callback(
            event_name,
            float(view_pos.x()),
            float(view_pos.y()),
            float(qpoint.x()),
            float(qpoint.y()),
            button,
            bool(dblclick),
            in_plot,
        )

    def update_from_matplotlib(
        self,
        *,
        ax,
        image_artist,
        background_artist,
        overlay_artist,
        marker_artist=None,
    ) -> None:
        if not self.available:
            return

        sim_visible = bool(image_artist.get_visible())
        bg_visible = bool(background_artist.get_visible())
        overlay_visible = bool(overlay_artist.get_visible())

        sim_alpha = image_artist.get_alpha()
        bg_alpha = background_artist.get_alpha()
        if sim_alpha is None:
            sim_alpha = 1.0
        if bg_alpha is None:
            bg_alpha = 1.0

        self._set_image_item(
            item=self._background_item,
            array=background_artist.get_array(),
            visible=bg_visible,
            levels=background_artist.get_clim(),
            alpha=float(bg_alpha),
            extent=background_artist.get_extent(),
        )
        self._set_image_item(
            item=self._simulation_item,
            array=image_artist.get_array(),
            visible=sim_visible,
            levels=image_artist.get_clim(),
            alpha=float(sim_alpha),
            extent=image_artist.get_extent(),
        )
        self._set_image_item(
            item=self._overlay_item,
            array=overlay_artist.get_array(),
            visible=overlay_visible,
            levels=(0.0, 1.0),
            alpha=1.0,
            extent=overlay_artist.get_extent(),
        )

        if marker_artist is not None and marker_artist.get_visible():
            x_vals, y_vals = marker_artist.get_data()
            if len(x_vals) > 0 and len(y_vals) > 0:
                self._marker_item.setData([float(x_vals[0])], [float(y_vals[0])])
                self._marker_item.setVisible(True)
            else:
                self._marker_item.setVisible(False)
        else:
            self._marker_item.setVisible(False)

        self._plot_item.setTitle(str(ax.get_title()))
        self._plot_item.setLabel("bottom", str(ax.get_xlabel() or "X (pixels)"))
        self._plot_item.setLabel("left", str(ax.get_ylabel() or "Y (pixels)"))

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        self._plot_item.getViewBox().setRange(
            xRange=(float(x0), float(x1)),
            yRange=(float(y0), float(y1)),
            padding=0.0,
            disableAutoRange=True,
        )

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
        try:
            self._window.close()
        except Exception:
            pass
        self.process_events()


class MatplotlibCanvasProxy:
    """Proxy object that mirrors Matplotlib image updates into a fast viewer."""

    def __init__(
        self,
        canvas,
        fast_viewer: FastPlotViewer,
        *,
        draw_interval_s: float = 0.12,
        render_matplotlib: bool = True,
        event_axes=None,
    ) -> None:
        self._canvas = canvas
        self._fast_viewer = fast_viewer
        self._draw_interval_s = float(max(0.0, draw_interval_s))
        self._last_draw_ts = 0.0
        self._sync_callback: Callable[[], None] | None = None
        self._render_matplotlib = bool(render_matplotlib)
        self._event_axes = event_axes
        self._callbacks: dict[str, dict[int, Callable]] = {}
        self._next_cid = 1
        if not self._render_matplotlib:
            self._fast_viewer.set_mouse_event_callback(self._dispatch_mouse_event)

    @property
    def fast_viewer(self) -> FastPlotViewer:
        return self._fast_viewer

    def set_sync_callback(self, callback: Callable[[], None]) -> None:
        self._sync_callback = callback

    def process_fast_events(self) -> None:
        self._fast_viewer.process_events()

    def sync_from_matplotlib(
        self,
        *,
        ax,
        image_artist,
        background_artist,
        overlay_artist,
        marker_artist=None,
    ) -> None:
        self._fast_viewer.update_from_matplotlib(
            ax=ax,
            image_artist=image_artist,
            background_artist=background_artist,
            overlay_artist=overlay_artist,
            marker_artist=marker_artist,
        )

    def _sync(self) -> None:
        if self._sync_callback is not None:
            try:
                self._sync_callback()
            except Exception:
                pass
        self._fast_viewer.process_events()

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
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._canvas, name)
