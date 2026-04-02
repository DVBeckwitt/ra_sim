from types import SimpleNamespace

import numpy as np

from ra_sim.gui import fast_plot_viewer


class _FakeRectanglePatch:
    def __init__(
        self,
        *,
        xy=(0.0, 0.0),
        width=0.0,
        height=0.0,
        visible=True,
        edgecolor=(1.0, 1.0, 0.0, 1.0),
        linewidth=2.0,
        linestyle="-",
        zorder=6.0,
    ) -> None:
        self._xy = tuple(xy)
        self._width = float(width)
        self._height = float(height)
        self._visible = bool(visible)
        self._edgecolor = tuple(edgecolor)
        self._linewidth = float(linewidth)
        self._linestyle = linestyle
        self._zorder = float(zorder)

    def get_visible(self):
        return self._visible

    def get_xy(self):
        return self._xy

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_edgecolor(self):
        return self._edgecolor

    def get_linewidth(self):
        return self._linewidth

    def get_linestyle(self):
        return self._linestyle

    def get_zorder(self):
        return self._zorder


class _FakeMarkerArtist:
    def __init__(
        self,
        *,
        xdata=(0.0,),
        ydata=(0.0,),
        marker="o",
        visible=True,
        markeredgecolor="y",
        markerfacecolor="none",
        markersize=8.0,
        markeredgewidth=1.5,
        color=None,
        zorder=10.0,
    ) -> None:
        self._xdata = tuple(xdata)
        self._ydata = tuple(ydata)
        self._marker = marker
        self._visible = bool(visible)
        self._markeredgecolor = markeredgecolor
        self._markerfacecolor = markerfacecolor
        self._markersize = float(markersize)
        self._markeredgewidth = float(markeredgewidth)
        self._color = color
        self._zorder = float(zorder)

    def get_visible(self):
        return self._visible

    def get_xdata(self):
        return self._xdata

    def get_ydata(self):
        return self._ydata

    def get_marker(self):
        return self._marker

    def get_markeredgecolor(self):
        return self._markeredgecolor

    def get_markerfacecolor(self):
        return self._markerfacecolor

    def get_markersize(self):
        return self._markersize

    def get_markeredgewidth(self):
        return self._markeredgewidth

    def get_color(self):
        return self._color

    def get_zorder(self):
        return self._zorder


class _FakeLineArtist:
    def __init__(
        self,
        *,
        xdata=(0.0, 1.0),
        ydata=(0.0, 1.0),
        linestyle="-",
        visible=True,
        color="c",
        linewidth=1.0,
        zorder=9.0,
    ) -> None:
        self._xdata = tuple(xdata)
        self._ydata = tuple(ydata)
        self._linestyle = linestyle
        self._visible = bool(visible)
        self._color = color
        self._linewidth = float(linewidth)
        self._zorder = float(zorder)

    def get_visible(self):
        return self._visible

    def get_xdata(self):
        return self._xdata

    def get_ydata(self):
        return self._ydata

    def get_linestyle(self):
        return self._linestyle

    def get_color(self):
        return self._color

    def get_linewidth(self):
        return self._linewidth

    def get_zorder(self):
        return self._zorder


def test_extract_visible_rectangle_specs_preserves_roi_geometry_and_style() -> None:
    axis = SimpleNamespace(
        patches=[
            _FakeRectanglePatch(
                xy=(10.0, 20.0),
                width=-4.0,
                height=8.0,
                edgecolor=(0.0, 1.0, 1.0, 0.5),
                linewidth=1.75,
                linestyle="--",
                zorder=7.0,
            ),
            _FakeRectanglePatch(visible=False),
            object(),
        ]
    )

    specs = fast_plot_viewer._extract_visible_rectangle_specs(axis)

    assert len(specs) == 1
    assert specs[0].x == 6.0
    assert specs[0].y == 20.0
    assert specs[0].width == 4.0
    assert specs[0].height == 8.0
    assert specs[0].edge_rgba == (0, 255, 255, 128)
    assert specs[0].linewidth == 1.75
    assert specs[0].linestyle == "--"
    assert specs[0].zorder == 7.0


def test_qt_button_mask_matches_handles_exact_and_mask_values() -> None:
    assert fast_plot_viewer._qt_button_mask_matches(1, 1) is True
    assert fast_plot_viewer._qt_button_mask_matches(2, 1) is False
    assert fast_plot_viewer._qt_button_mask_matches(3, 1) is True
    assert fast_plot_viewer._qt_button_mask_matches(3, 2) is True


def test_should_consume_fast_viewer_viewport_event_for_left_drag_sequence() -> None:
    assert (
        fast_plot_viewer._should_consume_fast_viewer_viewport_event(
            2,
            button=1,
            buttons=1,
            left_button=1,
        )
        is True
    )
    assert (
        fast_plot_viewer._should_consume_fast_viewer_viewport_event(
            5,
            button=0,
            buttons=1,
            left_button=1,
        )
        is True
    )
    assert (
        fast_plot_viewer._should_consume_fast_viewer_viewport_event(
            3,
            button=1,
            buttons=0,
            left_button=1,
        )
        is True
    )
    assert (
        fast_plot_viewer._should_consume_fast_viewer_viewport_event(
            5,
            button=0,
            buttons=2,
            left_button=1,
        )
        is False
    )


def test_windows_client_point_from_lparam_decodes_signed_coordinates() -> None:
    lparam = ((0xFFF6 & 0xFFFF) << 16) | (0x0012 & 0xFFFF)

    assert fast_plot_viewer._windows_client_point_from_lparam(lparam) == (18, -10)


def test_viewport_event_filter_ignores_partially_initialized_viewer_instances() -> None:
    viewer = fast_plot_viewer.FastPlotViewer.__new__(fast_plot_viewer.FastPlotViewer)
    viewer._QtCore = SimpleNamespace(
        Qt=None,
        QObject=type("_FakeQObject", (), {}),
    )

    event_filter = viewer._make_viewport_event_filter()

    assert event_filter.eventFilter(None, object()) is False


def test_forward_mouse_event_ignores_view_mapping_failures() -> None:
    class _FakePoint:
        def toPoint(self):
            return self

        def x(self):
            return 10.0

        def y(self):
            return 20.0

    class _FakeEvent:
        def position(self):
            return _FakePoint()

    viewer = fast_plot_viewer.FastPlotViewer.__new__(fast_plot_viewer.FastPlotViewer)
    viewer._mouse_event_callback = lambda *_args, **_kwargs: (_args, _kwargs)
    viewer._plot_widget = SimpleNamespace(
        mapToScene=lambda _point: (_ for _ in ()).throw(RuntimeError("deleted")),
    )
    viewer._plot_item = SimpleNamespace(vb=SimpleNamespace())

    viewer._forward_mouse_event("button_release_event", _FakeEvent(), dblclick=False)


def test_forward_mouse_event_from_qpoint_dispatches_normalized_callback_payload() -> None:
    class _FakePoint:
        def __init__(self, x: float, y: float) -> None:
            self._x = float(x)
            self._y = float(y)

        def x(self):
            return self._x

        def y(self):
            return self._y

    received = []
    viewer = fast_plot_viewer.FastPlotViewer.__new__(fast_plot_viewer.FastPlotViewer)
    viewer._mouse_event_callback = lambda *args: received.append(args)
    viewer._plot_widget = SimpleNamespace(
        mapToScene=lambda point: _FakePoint(point.x() + 5.0, point.y() + 5.0),
    )
    viewer._plot_item = SimpleNamespace(
        vb=SimpleNamespace(
            mapSceneToView=lambda point: _FakePoint(point.x() * 2.0, point.y() * 3.0),
            sceneBoundingRect=lambda: SimpleNamespace(
                contains=lambda _point: True,
            ),
        )
    )

    viewer._forward_mouse_event_from_qpoint(
        "button_press_event",
        _FakePoint(10.0, 20.0),
        button=1,
        dblclick=False,
    )

    assert received == [
        (
            "button_press_event",
            30.0,
            75.0,
            10.0,
            20.0,
            1,
            False,
            True,
        )
    ]


def test_handle_embedded_native_mouse_message_consumes_left_drag_sequence() -> None:
    class _FakePoint:
        def __init__(self, x: int, y: int) -> None:
            self._x = int(x)
            self._y = int(y)

        def x(self):
            return self._x

        def y(self):
            return self._y

    received = []
    capture_states = []
    viewer = fast_plot_viewer.FastPlotViewer.__new__(fast_plot_viewer.FastPlotViewer)
    viewer._mouse_event_callback = lambda *args: received.append(args)
    viewer._QtCore = SimpleNamespace(QPoint=lambda x, y: _FakePoint(x, y))
    viewer._plot_widget = SimpleNamespace(mapToScene=lambda point: point)
    viewer._plot_item = SimpleNamespace(
        vb=SimpleNamespace(
            mapSceneToView=lambda point: point,
            sceneBoundingRect=lambda: SimpleNamespace(contains=lambda _point: True),
        )
    )
    viewer._embedded_native_left_drag_active = False
    viewer._set_native_mouse_capture = lambda enabled: capture_states.append(bool(enabled))

    press_lparam = ((34 & 0xFFFF) << 16) | (12 & 0xFFFF)
    move_lparam = ((36 & 0xFFFF) << 16) | (18 & 0xFFFF)
    release_lparam = ((40 & 0xFFFF) << 16) | (24 & 0xFFFF)

    assert viewer._handle_embedded_native_mouse_message(0x0201, 0, press_lparam) is True
    assert viewer._handle_embedded_native_mouse_message(0x0200, 0x0001, move_lparam) is True
    assert viewer._handle_embedded_native_mouse_message(0x0202, 0, release_lparam) is True
    assert viewer._handle_embedded_native_mouse_message(0x0204, 0, press_lparam) is False

    assert capture_states == [True, False]
    assert [call[0] for call in received] == [
        "button_press_event",
        "motion_notify_event",
        "button_release_event",
    ]
    assert received[0][3:6] == (12.0, 34.0, 1)
    assert received[1][3:6] == (18.0, 36.0, None)
    assert received[2][3:6] == (24.0, 40.0, 1)


def test_extract_visible_rectangle_specs_skips_transparent_edges_and_sorts_by_zorder() -> None:
    axis = SimpleNamespace(
        patches=[
            _FakeRectanglePatch(xy=(2.0, 2.0), zorder=9.0),
            _FakeRectanglePatch(xy=(1.0, 1.0), zorder=5.0),
            _FakeRectanglePatch(
                xy=(3.0, 3.0),
                edgecolor=(1.0, 1.0, 0.0, 0.0),
                zorder=1.0,
            ),
        ]
    )

    specs = fast_plot_viewer._extract_visible_rectangle_specs(axis)

    assert [spec.zorder for spec in specs] == [5.0, 9.0]
    assert [(spec.x, spec.y) for spec in specs] == [(1.0, 1.0), (2.0, 2.0)]


def test_extract_visible_marker_specs_supports_multiple_fast_viewer_pick_markers() -> None:
    specs = fast_plot_viewer._extract_visible_marker_specs(
        (
            _FakeMarkerArtist(
                xdata=(12.0,),
                ydata=(24.0,),
                marker="o",
                markeredgecolor="r",
                markerfacecolor="none",
                markersize=5.0,
                zorder=8.0,
            ),
            _FakeMarkerArtist(
                xdata=(30.0,),
                ydata=(40.0,),
                marker="s",
                markeredgecolor="y",
                markerfacecolor="none",
                markersize=8.0,
                markeredgewidth=1.5,
                zorder=9.0,
            ),
            _FakeMarkerArtist(visible=False),
            object(),
        )
    )

    assert [spec.symbol for spec in specs] == ["o", "s"]
    assert [spec.x_values for spec in specs] == [(12.0,), (30.0,)]
    assert [spec.y_values for spec in specs] == [(24.0,), (40.0,)]
    assert specs[0].edge_rgba == (255, 0, 0, 255)
    assert specs[1].edge_rgba == (191, 191, 0, 255)
    assert specs[1].face_rgba == (0, 0, 0, 0)
    assert specs[1].pen_width == 1.5
    assert specs[1].size == 8.0


def test_build_artist_overlay_model_uses_explicit_runtime_overlay_groups() -> None:
    overlay_model = fast_plot_viewer.build_artist_overlay_model(
        transient_artists=(
            _FakeRectanglePatch(xy=(2.0, 3.0), width=4.0, height=5.0, zorder=4.0),
            (
                _FakeMarkerArtist(
                    xdata=(12.0,),
                    ydata=(24.0,),
                    marker="d",
                    markeredgecolor="y",
                    markerfacecolor="none",
                    markersize=6.0,
                    zorder=8.0,
                ),
                _FakeLineArtist(
                    xdata=(1.0, 2.0, 3.0),
                    ydata=(4.0, 5.0, 6.0),
                    color="c",
                    linewidth=1.2,
                    zorder=7.0,
                ),
            ),
        ),
        transient_curve_specs=(
            {
                "x_values": (10.0, 20.0),
                "y_values": (30.0, 40.0),
                "edge_rgba": (255, 0, 255, 255),
                "linewidth": 1.5,
                "linestyle": "--",
                "zorder": 11.0,
            },
        ),
        suppress_overlay_image=True,
    )

    assert overlay_model.suppress_overlay_image is True
    assert [(spec.x, spec.y) for spec in overlay_model.transient_rectangle_specs] == [
        (2.0, 3.0)
    ]
    assert [spec.symbol for spec in overlay_model.transient_marker_specs] == ["d"]
    assert [spec.zorder for spec in overlay_model.transient_curve_specs] == [7.0, 11.0]
    assert overlay_model.transient_curve_specs[0].edge_rgba == (0, 191, 191, 255)
    assert overlay_model.transient_curve_specs[1].linestyle == "--"


def test_should_apply_matplotlib_view_range_on_initial_sync() -> None:
    assert (
        fast_plot_viewer._should_apply_matplotlib_view_range(
            current_range=None,
            last_synced_range=None,
            target_range=(0.0, 100.0, 200.0, 0.0),
        )
        is True
    )


def test_should_not_reset_fast_viewer_zoom_when_matplotlib_range_is_unchanged() -> None:
    last_synced = (0.0, 100.0, 200.0, 0.0)
    current_zoomed = (20.0, 40.0, 80.0, 20.0)

    assert (
        fast_plot_viewer._should_apply_matplotlib_view_range(
            current_range=current_zoomed,
            last_synced_range=last_synced,
            target_range=last_synced,
        )
        is False
    )


def test_should_apply_matplotlib_view_range_when_source_axes_change() -> None:
    last_synced = (0.0, 100.0, 200.0, 0.0)
    current_zoomed = (20.0, 40.0, 80.0, 20.0)
    target_changed = (0.0, 360.0, 90.0, -90.0)

    assert (
        fast_plot_viewer._should_apply_matplotlib_view_range(
            current_range=current_zoomed,
            last_synced_range=last_synced,
            target_range=target_changed,
        )
        is True
    )


class _FakeCanvas:
    def draw(self, *args, **kwargs):
        return None

    def draw_idle(self, *args, **kwargs):
        return None


class _FakeFastViewer:
    def __init__(self, *, timer_cls=None) -> None:
        self.process_events_calls = 0
        self.mouse_event_callback = None
        self.update_from_matplotlib_calls = []
        self.view_range = (0.0, 100.0, 200.0, 0.0)
        self._QtCore = (
            SimpleNamespace(QTimer=timer_cls) if timer_cls is not None else None
        )

    def set_mouse_event_callback(self, callback) -> None:
        self.mouse_event_callback = callback

    def process_events(self) -> None:
        self.process_events_calls += 1

    def update_from_matplotlib(self, **kwargs) -> None:
        self.update_from_matplotlib_calls.append(kwargs)

    def current_view_range(self):
        return self.view_range


class _FakeQTimer:
    scheduled = []

    @classmethod
    def singleShot(cls, delay, callback) -> None:
        cls.scheduled.append((int(delay), callback))

    @classmethod
    def reset(cls) -> None:
        cls.scheduled = []


class _FakeTransform:
    def translate(self, *_args, **_kwargs) -> None:
        return None

    def scale(self, *_args, **_kwargs) -> None:
        return None


class _FakeImageItem:
    def __init__(self) -> None:
        self.set_image_calls = []
        self.opacity_calls = []
        self.transform_calls = 0
        self.visible = None

    def setImage(self, image, **kwargs) -> None:
        self.set_image_calls.append((tuple(image.shape), dict(kwargs)))

    def setOpacity(self, value) -> None:
        self.opacity_calls.append(float(value))

    def setTransform(self, _transform) -> None:
        self.transform_calls += 1

    def setVisible(self, visible) -> None:
        self.visible = bool(visible)


class _FakeCurveItem:
    def __init__(self) -> None:
        self.visible = False
        self.data_calls = []
        self.pen_calls = []
        self.z_values = []

    def setVisible(self, visible) -> None:
        self.visible = bool(visible)

    def setData(self, **kwargs) -> None:
        self.data_calls.append(dict(kwargs))

    def setPen(self, pen) -> None:
        self.pen_calls.append(pen)

    def setZValue(self, value) -> None:
        self.z_values.append(float(value))


class _FakeScatterItem(_FakeCurveItem):
    pass


class _FakePg:
    @staticmethod
    def mkPen(**kwargs):
        return dict(kwargs)

    @staticmethod
    def mkBrush(rgba):
        return tuple(rgba)

    @staticmethod
    def PlotCurveItem():
        return _FakeCurveItem()

    @staticmethod
    def ScatterPlotItem():
        return _FakeScatterItem()


class _FakeViewBox:
    def __init__(self) -> None:
        self.range = ((0.0, 1.0), (1.0, 0.0))
        self.set_range_calls = []

    def viewRange(self):
        return self.range

    def setRange(self, *, xRange, yRange, padding, disableAutoRange) -> None:
        self.set_range_calls.append(
            {
                "xRange": tuple(xRange),
                "yRange": tuple(yRange),
                "padding": float(padding),
                "disableAutoRange": bool(disableAutoRange),
            }
        )
        self.range = (tuple(xRange), tuple(yRange))


class _FakePlotItem:
    def __init__(self) -> None:
        self.title_calls = []
        self.label_calls = []
        self.view_box = _FakeViewBox()
        self.items = []

    def addItem(self, item) -> None:
        self.items.append(item)

    def setTitle(self, title) -> None:
        self.title_calls.append(str(title))

    def setLabel(self, axis, label) -> None:
        self.label_calls.append((str(axis), str(label)))

    def getViewBox(self):
        return self.view_box


class _FakeImageArtist:
    def __init__(
        self,
        array,
        *,
        visible=True,
        alpha=1.0,
        clim=(0.0, 1.0),
        extent=(0.0, 4.0, 4.0, 0.0),
    ) -> None:
        self._array = np.asarray(array, dtype=float)
        self._visible = bool(visible)
        self._alpha = alpha
        self._clim = tuple(clim)
        self._extent = tuple(extent)

    def get_array(self):
        return self._array

    def get_visible(self):
        return self._visible

    def get_alpha(self):
        return self._alpha

    def get_clim(self):
        return self._clim

    def get_extent(self):
        return self._extent


class _FakeAxes:
    def __init__(self) -> None:
        self.patches = []

    def get_title(self):
        return "Simulated Diffraction Pattern"

    def get_xlabel(self):
        return "X (pixels)"

    def get_ylabel(self):
        return "Y (pixels)"

    def get_xlim(self):
        return (0.0, 4.0)

    def get_ylim(self):
        return (4.0, 0.0)


def _build_stub_fast_viewer():
    viewer = fast_plot_viewer.FastPlotViewer.__new__(fast_plot_viewer.FastPlotViewer)
    viewer.available = True
    viewer._pg = _FakePg()
    viewer._QtGui = SimpleNamespace(QTransform=_FakeTransform)
    viewer._QtCore = SimpleNamespace(Qt=None)
    viewer._QtWidgets = None
    viewer._app = None
    viewer._window = None
    viewer._plot_widget = None
    viewer._plot_item = _FakePlotItem()
    viewer._background_item = _FakeImageItem()
    viewer._simulation_item = _FakeImageItem()
    viewer._overlay_item = _FakeImageItem()
    viewer._marker_items = []
    viewer._rectangle_items = []
    viewer._transient_marker_items = []
    viewer._transient_rectangle_items = []
    viewer._transient_curve_items = []
    viewer._last_synced_view_range = None
    viewer._image_sync_states = {}
    viewer._image_buffers = {}
    viewer._marker_specs = ()
    viewer._rectangle_specs = ()
    viewer._transient_marker_specs = ()
    viewer._transient_rectangle_specs = ()
    viewer._transient_curve_specs = ()
    viewer._last_title = None
    viewer._last_xlabel = None
    viewer._last_ylabel = None
    viewer._mouse_event_callback = None
    viewer._mouse_capture_overlay = None
    viewer._embedded_tk_host = None
    viewer._embedded_parent_hwnd = None
    viewer._native_hwnd = None
    return viewer


def test_canvas_proxy_skips_reentrant_fast_event_pumping_during_mouse_dispatch() -> None:
    event_axes = object()
    viewer = _FakeFastViewer()
    proxy = fast_plot_viewer.MatplotlibCanvasProxy(
        _FakeCanvas(),
        viewer,
        render_matplotlib=False,
        event_axes=event_axes,
    )
    sync_calls: list[str] = []
    received = []

    proxy.set_sync_callback(lambda: sync_calls.append("sync"))

    def _on_motion(event) -> None:
        received.append(event)
        proxy.draw_idle()
        proxy.process_fast_events()

    proxy.mpl_connect("motion_notify_event", _on_motion)

    proxy._dispatch_mouse_event(
        "motion_notify_event",
        12.5,
        24.5,
        125.0,
        245.0,
        None,
        False,
        True,
    )

    assert len(received) == 1
    assert received[0].inaxes is event_axes
    assert received[0].xdata == 12.5
    assert received[0].ydata == 24.5
    assert sync_calls == ["sync"]
    assert viewer.process_events_calls == 0

    proxy.process_fast_events()

    assert viewer.process_events_calls == 1


def test_canvas_proxy_defers_fast_viewer_sync_until_mouse_dispatch_finishes() -> None:
    event_axes = object()
    viewer = _FakeFastViewer()
    proxy = fast_plot_viewer.MatplotlibCanvasProxy(
        _FakeCanvas(),
        viewer,
        render_matplotlib=False,
        event_axes=event_axes,
    )
    update_counts = []

    def _on_motion(_event) -> None:
        update_counts.append(len(viewer.update_from_matplotlib_calls))
        proxy.sync_from_matplotlib(
            ax="ax",
            image_artist="image",
            background_artist="background",
            overlay_artist="overlay",
            marker_artist="marker",
            force_view_range=True,
        )
        update_counts.append(len(viewer.update_from_matplotlib_calls))

    proxy.mpl_connect("motion_notify_event", _on_motion)

    proxy._dispatch_mouse_event(
        "motion_notify_event",
        10.0,
        20.0,
        100.0,
        200.0,
        None,
        False,
        True,
    )

    assert update_counts == [0, 0]
    assert len(viewer.update_from_matplotlib_calls) == 1
    assert viewer.update_from_matplotlib_calls[0] == {
        "ax": "ax",
        "image_artist": "image",
        "background_artist": "background",
        "overlay_artist": "overlay",
        "marker_artist": "marker",
        "overlay_model": None,
        "layer_versions": {},
        "force_view_range": True,
    }


def test_canvas_proxy_coalesces_latest_sync_payload_on_qt_timer_boundary() -> None:
    _FakeQTimer.reset()
    viewer = _FakeFastViewer(timer_cls=_FakeQTimer)
    proxy = fast_plot_viewer.MatplotlibCanvasProxy(
        _FakeCanvas(),
        viewer,
        render_matplotlib=False,
    )

    proxy.sync_from_matplotlib(
        ax="ax-1",
        image_artist="image-1",
        background_artist="background-1",
        overlay_artist="overlay-1",
        marker_artist="marker-1",
        layer_versions={"simulation": 1},
        force_view_range=False,
    )
    proxy.sync_from_matplotlib(
        ax="ax-2",
        image_artist="image-2",
        background_artist="background-2",
        overlay_artist="overlay-2",
        marker_artist="marker-2",
        overlay_model="explicit-overlay",
        layer_versions={"simulation": 2},
        force_view_range=True,
    )

    assert viewer.update_from_matplotlib_calls == []
    assert len(_FakeQTimer.scheduled) == 1

    _delay, callback = _FakeQTimer.scheduled.pop()
    callback()

    assert viewer.update_from_matplotlib_calls == [
        {
            "ax": "ax-2",
            "image_artist": "image-2",
            "background_artist": "background-2",
            "overlay_artist": "overlay-2",
            "marker_artist": "marker-2",
            "overlay_model": "explicit-overlay",
            "layer_versions": {"simulation": 2},
            "force_view_range": True,
        }
    ]


def test_canvas_proxy_throttles_fast_event_pumping(monkeypatch) -> None:
    viewer = _FakeFastViewer()
    proxy = fast_plot_viewer.MatplotlibCanvasProxy(
        _FakeCanvas(),
        viewer,
        render_matplotlib=False,
        event_pump_interval_s=0.1,
    )
    times = iter((0.0, 0.05, 0.15))
    monkeypatch.setattr(
        fast_plot_viewer.time,
        "perf_counter",
        lambda: next(times),
    )

    proxy.process_fast_events()
    proxy.process_fast_events()
    proxy.process_fast_events()

    assert viewer.process_events_calls == 2


def test_canvas_proxy_maps_host_overlay_pixels_into_event_axes_coordinates() -> None:
    event_axes = object()
    viewer = _FakeFastViewer()
    proxy = fast_plot_viewer.MatplotlibCanvasProxy(
        _FakeCanvas(),
        viewer,
        render_matplotlib=False,
        event_axes=event_axes,
    )
    received = []

    proxy.mpl_connect("button_press_event", lambda event: received.append(event))
    proxy.dispatch_host_overlay_event(
        "button_press_event",
        x_pixel=50.0,
        y_pixel=25.0,
        width=101.0,
        height=51.0,
        button=1,
        dblclick=False,
    )

    assert len(received) == 1
    assert received[0].inaxes is event_axes
    assert received[0].button == 1
    assert received[0].x == 50.0
    assert received[0].y == 25.0
    assert received[0].xdata == 50.0
    assert received[0].ydata == 100.0


def test_canvas_proxy_prefers_fast_viewer_viewport_mapping_for_host_overlay_events() -> None:
    event_axes = object()
    viewer = _FakeFastViewer()
    viewer.map_viewport_pixels_to_view_coords = lambda x, y: (x + 3.5, y + 7.5, True)
    proxy = fast_plot_viewer.MatplotlibCanvasProxy(
        _FakeCanvas(),
        viewer,
        render_matplotlib=False,
        event_axes=event_axes,
    )
    received = []

    proxy.mpl_connect("button_press_event", lambda event: received.append(event))
    proxy.dispatch_host_overlay_event(
        "button_press_event",
        x_pixel=10.0,
        y_pixel=20.0,
        width=101.0,
        height=51.0,
        button=1,
        dblclick=False,
    )

    assert len(received) == 1
    assert received[0].inaxes is event_axes
    assert received[0].x == 10.0
    assert received[0].y == 20.0
    assert received[0].xdata == 13.5
    assert received[0].ydata == 27.5


def test_fast_viewer_maps_viewport_pixels_into_plot_view_coordinates() -> None:
    class _FakePoint:
        def __init__(self, x: float, y: float) -> None:
            self._x = float(x)
            self._y = float(y)

        def x(self):
            return self._x

        def y(self):
            return self._y

    viewer = fast_plot_viewer.FastPlotViewer.__new__(fast_plot_viewer.FastPlotViewer)
    viewer._QtCore = SimpleNamespace()
    viewer._plot_widget = SimpleNamespace(
        mapToScene=lambda x, y: _FakePoint(x + 5.0, y + 7.0),
    )
    viewer._plot_item = SimpleNamespace(
        vb=SimpleNamespace(
            mapSceneToView=lambda point: _FakePoint(point.x() * 2.0, point.y() * 3.0),
            sceneBoundingRect=lambda: SimpleNamespace(
                contains=lambda point: point.x() <= 20.0,
            ),
        )
    )

    mapped = viewer.map_viewport_pixels_to_view_coords(10.2, 20.7)

    assert mapped == (30.0, 84.0, True)


def test_fast_viewer_update_from_matplotlib_skips_unchanged_layer_pushes() -> None:
    viewer = _build_stub_fast_viewer()
    ax = _FakeAxes()
    background_artist = _FakeImageArtist(np.ones((4, 4)), clim=(0.0, 2.0))
    image_artist = _FakeImageArtist(np.full((4, 4), 3.0), clim=(1.0, 5.0))
    overlay_artist = _FakeImageArtist(
        np.zeros((4, 4)),
        clim=(0.0, 1.0),
        alpha=1.0,
    )

    viewer.update_from_matplotlib(
        ax=ax,
        image_artist=image_artist,
        background_artist=background_artist,
        overlay_artist=overlay_artist,
        marker_artist=None,
        overlay_model=fast_plot_viewer.FastViewerOverlayModel(),
        layer_versions={"background": 1, "simulation": 2, "overlay": 3},
        force_view_range=False,
    )
    viewer.update_from_matplotlib(
        ax=ax,
        image_artist=image_artist,
        background_artist=background_artist,
        overlay_artist=overlay_artist,
        marker_artist=None,
        overlay_model=fast_plot_viewer.FastViewerOverlayModel(),
        layer_versions={"background": 1, "simulation": 2, "overlay": 3},
        force_view_range=False,
    )

    assert len(viewer._background_item.set_image_calls) == 1
    assert len(viewer._simulation_item.set_image_calls) == 1
    assert len(viewer._overlay_item.set_image_calls) == 1
    assert viewer._plot_item.title_calls == ["Simulated Diffraction Pattern"]
    assert viewer._plot_item.label_calls == [
        ("bottom", "X (pixels)"),
        ("left", "Y (pixels)"),
    ]
    assert len(viewer._plot_item.view_box.set_range_calls) == 1

    viewer.update_from_matplotlib(
        ax=ax,
        image_artist=image_artist,
        background_artist=background_artist,
        overlay_artist=overlay_artist,
        marker_artist=None,
        overlay_model=fast_plot_viewer.FastViewerOverlayModel(
            suppress_overlay_image=True
        ),
        layer_versions={"background": 1, "simulation": 2, "overlay": 4},
        force_view_range=False,
    )

    assert viewer._overlay_item.visible is False
    assert len(viewer._simulation_item.set_image_calls) == 1
