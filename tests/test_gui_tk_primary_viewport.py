from types import SimpleNamespace

from ra_sim.gui import tk_primary_viewport


class _FakeWidget:
    def __init__(self):
        self.bindings = {}
        self.after_calls = []
        self.after_cancelled = []
        self.pack_calls = []
        self.pack_forget_calls = 0
        self.destroy_calls = 0

    def bind(self, sequence, callback, add=None):
        self.bindings[sequence] = (callback, add)

    def after(self, delay_ms, callback):
        token = len(self.after_calls) + 1
        self.after_calls.append((token, delay_ms, callback))
        return token

    def after_cancel(self, token):
        self.after_cancelled.append(token)

    def pack(self, **kwargs):
        self.pack_calls.append(dict(kwargs))

    def pack_forget(self):
        self.pack_forget_calls += 1

    def destroy(self):
        self.destroy_calls += 1

    def winfo_reqwidth(self):
        return 640

    def winfo_reqheight(self):
        return 480


class _FakeViewport:
    def __init__(self, view_state):
        self.widget = _FakeWidget()
        self._view_state = view_state

    def contains_screen_point(self, x_pixel, y_pixel):
        return (
            0.0 <= float(x_pixel) <= float(self._view_state.width)
            and 0.0 <= float(y_pixel) <= float(self._view_state.height)
        )

    def screen_to_world(self, x_pixel, y_pixel):
        return tk_primary_viewport.screen_to_world(
            self._view_state,
            float(x_pixel),
            float(y_pixel),
        )


def test_primary_viewport_backend_parser_accepts_matplotlib_and_tk_canvas() -> None:
    assert tk_primary_viewport.parse_primary_viewport_backend(None) == "matplotlib"
    assert tk_primary_viewport.parse_primary_viewport_backend("matplotlib") == "matplotlib"
    assert tk_primary_viewport.parse_primary_viewport_backend("tk_canvas") == "tk_canvas"
    assert tk_primary_viewport.parse_primary_viewport_backend("tk-canvas") == "tk_canvas"
    assert tk_primary_viewport.parse_primary_viewport_backend("something-unknown") == "matplotlib"


def test_screen_and_world_transforms_round_trip_detector_view() -> None:
    view_state = tk_primary_viewport.ViewportViewState(
        width=200,
        height=100,
        xlim=(0.0, 100.0),
        ylim=(100.0, 0.0),
    )

    world = tk_primary_viewport.screen_to_world(view_state, 50.0, 25.0)
    assert world == (25.0, 75.0)
    assert tk_primary_viewport.world_to_screen(view_state, *world) == (50.0, 25.0)


def test_screen_and_world_transforms_round_trip_caked_view() -> None:
    view_state = tk_primary_viewport.ViewportViewState(
        width=240,
        height=120,
        xlim=(10.0, 34.0),
        ylim=(-30.0, 30.0),
    )

    world = tk_primary_viewport.screen_to_world(view_state, 120.0, 30.0)
    assert world == (22.0, -15.0)
    assert tk_primary_viewport.world_to_screen(view_state, *world) == (120.0, 30.0)


def test_build_peak_cache_filters_visible_points_in_current_view() -> None:
    view_state = tk_primary_viewport.ViewportViewState(
        width=200,
        height=100,
        xlim=(0.0, 100.0),
        ylim=(100.0, 0.0),
    )

    cache = tk_primary_viewport._build_peak_cache(
        [(10.0, 10.0), (50.0, 50.0), (125.0, 60.0), ("bad", 0.0)],
        view_state,
    )

    assert cache.positions == ((10.0, 10.0), (50.0, 50.0), (125.0, 60.0))
    assert cache.visible_positions == ((10.0, 10.0), (50.0, 50.0))


def test_build_q_group_cache_uses_detector_display_points() -> None:
    view_state = tk_primary_viewport.ViewportViewState(
        width=200,
        height=100,
        xlim=(0.0, 100.0),
        ylim=(100.0, 0.0),
    )
    grouped_candidates = {
        ("q", 1): [
            {"display_col": 40.0, "display_row": 35.0, "label": "visible"},
            {"display_col": 140.0, "display_row": 35.0, "label": "hidden"},
        ]
    }

    cache = tk_primary_viewport._build_q_group_cache(
        grouped_candidates,
        view_state,
        view_mode="detector",
    )

    assert len(cache.visible_entries) == 1
    assert cache.visible_entries[0]["label"] == "visible"
    assert cache.visible_entries[0]["q_group_key"] == ("q", 1)
    assert cache.visible_entries[0]["_viewport_point"] == (40.0, 35.0)


def test_build_q_group_cache_uses_caked_angles_when_present() -> None:
    view_state = tk_primary_viewport.ViewportViewState(
        width=240,
        height=120,
        xlim=(10.0, 34.0),
        ylim=(-30.0, 30.0),
    )
    grouped_candidates = {
        ("q", 2): [
            {"two_theta_deg": 21.0, "phi_deg": -8.0, "display_col": 500.0, "display_row": 500.0}
        ]
    }

    cache = tk_primary_viewport._build_q_group_cache(
        grouped_candidates,
        view_state,
        view_mode="caked",
    )

    assert len(cache.visible_entries) == 1
    assert cache.visible_entries[0]["_viewport_point"] == (21.0, -8.0)


def test_tk_canvas_proxy_dispatches_click_with_axis_space_coordinates() -> None:
    view_state = tk_primary_viewport.ViewportViewState(
        width=200,
        height=100,
        xlim=(0.0, 100.0),
        ylim=(100.0, 0.0),
    )
    viewport = _FakeViewport(view_state)
    proxy = tk_primary_viewport.TkCanvasViewportProxy(viewport, event_axes="AX", draw_interval_s=0.0)
    events = []
    proxy.mpl_connect("button_press_event", lambda event: events.append(event))

    proxy.dispatch_tk_event(
        "button_press_event",
        SimpleNamespace(x=60.0, y=25.0),
        button=1,
    )

    assert len(events) == 1
    assert events[0].button == 1
    assert events[0].inaxes == "AX"
    assert events[0].xdata == 30.0
    assert events[0].ydata == 75.0


def test_tk_canvas_proxy_dispatches_scroll_step_from_mousewheel_delta() -> None:
    view_state = tk_primary_viewport.ViewportViewState(
        width=200,
        height=100,
        xlim=(0.0, 100.0),
        ylim=(100.0, 0.0),
    )
    viewport = _FakeViewport(view_state)
    proxy = tk_primary_viewport.TkCanvasViewportProxy(viewport, event_axes="AX", draw_interval_s=0.0)
    events = []
    proxy.mpl_connect("scroll_event", lambda event: events.append(event))

    proxy._dispatch_mousewheel(SimpleNamespace(x=30.0, y=40.0, delta=120.0))

    assert len(events) == 1
    assert events[0].button == "up"
    assert events[0].step == 1.0
    assert events[0].xdata == 15.0
    assert events[0].ydata == 60.0


def test_tk_canvas_proxy_draw_idle_schedules_one_sync() -> None:
    view_state = tk_primary_viewport.ViewportViewState(
        width=200,
        height=100,
        xlim=(0.0, 100.0),
        ylim=(100.0, 0.0),
    )
    viewport = _FakeViewport(view_state)
    sync_calls = []
    proxy = tk_primary_viewport.TkCanvasViewportProxy(
        viewport,
        event_axes="AX",
        draw_interval_s=0.0,
        sync_callback=lambda: sync_calls.append("sync"),
    )

    proxy.draw_idle()
    proxy.draw_idle()

    assert len(viewport.widget.after_calls) == 1
    _, _, callback = viewport.widget.after_calls[0]
    callback()

    assert sync_calls == ["sync"]


def test_build_tk_primary_viewport_backend_swaps_widgets_on_activate_and_deactivate(
    monkeypatch,
) -> None:
    viewport_widget = _FakeWidget()
    matplotlib_widget = _FakeWidget()
    sync_calls = []

    class _FakeViewportRuntime:
        def __init__(self):
            self.widget = viewport_widget

        def sync_from_matplotlib(
            self,
            *,
            image_artist,
            background_artist,
            overlay_artist,
            force_view_range=False,
        ):
            sync_calls.append(
                {
                    "image_artist": image_artist,
                    "background_artist": background_artist,
                    "overlay_artist": overlay_artist,
                    "force_view_range": bool(force_view_range),
                }
            )

    monkeypatch.setattr(
        tk_primary_viewport,
        "_TkPrimaryViewport",
        lambda **kwargs: _FakeViewportRuntime(),
    )

    backend = tk_primary_viewport.build_tk_primary_viewport_backend(
        tk_module=SimpleNamespace(TOP="top", BOTH="both"),
        canvas_frame="canvas-parent",
        matplotlib_canvas=SimpleNamespace(get_tk_widget=lambda: matplotlib_widget),
        ax="AX",
        image_artist="image",
        background_artist="background",
        overlay_artist="overlay",
    )

    backend.activate()
    backend.deactivate()
    backend.shutdown()

    assert matplotlib_widget.pack_forget_calls == 1
    assert viewport_widget.pack_calls == [{"side": "top", "fill": "both", "expand": True}]
    assert viewport_widget.pack_forget_calls == 1
    assert matplotlib_widget.pack_calls == [{"side": "top", "fill": "both", "expand": True}]
    assert viewport_widget.destroy_calls == 1
    assert sync_calls == [
        {
            "image_artist": "image",
            "background_artist": "background",
            "overlay_artist": "overlay",
            "force_view_range": True,
        }
    ]
