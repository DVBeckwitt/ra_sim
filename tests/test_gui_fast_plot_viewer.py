from types import SimpleNamespace

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
    def __init__(self) -> None:
        self.process_events_calls = 0
        self.mouse_event_callback = None
        self.update_from_matplotlib_calls = []

    def set_mouse_event_callback(self, callback) -> None:
        self.mouse_event_callback = callback

    def process_events(self) -> None:
        self.process_events_calls += 1

    def update_from_matplotlib(self, **kwargs) -> None:
        self.update_from_matplotlib_calls.append(kwargs)


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
        "force_view_range": True,
    }
