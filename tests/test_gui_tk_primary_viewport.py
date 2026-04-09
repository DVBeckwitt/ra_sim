from types import SimpleNamespace

import numpy as np
import pytest

from ra_sim.gui import tk_primary_viewport


class _FakeWidget:
    def __init__(self):
        self.bindings = {}
        self.after_calls = []
        self.after_cancelled = []
        self.pack_calls = []
        self.pack_forget_calls = 0
        self.destroy_calls = 0
        self.create_calls = []
        self.itemconfigure_calls = []
        self.delete_calls = []
        self._next_item_id = 1
        self._items = {}

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

    def _create_item(self, kind, **kwargs):
        item_id = self._next_item_id
        self._next_item_id += 1
        record = {
            "id": item_id,
            "kind": kind,
            **dict(kwargs),
        }
        self._items[item_id] = record
        self.create_calls.append(record)
        return item_id

    def create_image(self, x, y, **kwargs):
        return self._create_item("image", x=x, y=y, **kwargs)

    def create_rectangle(self, *coords, **kwargs):
        return self._create_item("rectangle", coords=tuple(coords), **kwargs)

    def create_line(self, *coords, **kwargs):
        return self._create_item("line", coords=tuple(coords), **kwargs)

    def create_oval(self, *coords, **kwargs):
        return self._create_item("oval", coords=tuple(coords), **kwargs)

    def create_text(self, x, y, **kwargs):
        return self._create_item("text", x=x, y=y, **kwargs)

    def itemconfigure(self, item, **kwargs):
        self.itemconfigure_calls.append((item, dict(kwargs)))
        if item in self._items:
            self._items[item].update(dict(kwargs))

    def delete(self, target):
        self.delete_calls.append(target)
        if isinstance(target, str):
            to_delete = [
                item_id
                for item_id, record in self._items.items()
                if target in tuple(record.get("tags", ()))
            ]
            for item_id in to_delete:
                self._items.pop(item_id, None)
            return
        self._items.pop(target, None)

    def winfo_reqwidth(self):
        return 640

    def winfo_reqheight(self):
        return 480

    def winfo_width(self):
        return 640

    def winfo_height(self):
        return 480


class _FakeTkModule:
    CENTER = "center"
    NW = "nw"

    @staticmethod
    def Canvas(_parent, **_kwargs):
        return _FakeWidget()


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


class _FakeAxes:
    def __init__(self, *, xlim, ylim):
        self._xlim = tuple(xlim)
        self._ylim = tuple(ylim)

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim


class _FakePhotoImage:
    instances = []

    def __init__(self, image):
        self.image = image
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []


def _make_layer(
    name: str,
    rgba: np.ndarray,
    *,
    extent: tuple[float, float, float, float] = (0.0, 2.0, 2.0, 0.0),
    visible: bool = True,
    origin: str = "upper",
) -> tk_primary_viewport._ViewportImageLayer:
    return tk_primary_viewport._ViewportImageLayer(
        name=name,
        visible=visible,
        extent=extent,
        interpolation="nearest",
        source_rgba=np.asarray(rgba, dtype=np.uint8),
        origin=origin,
    )


def _make_scene(
    *,
    view_state: tk_primary_viewport.ViewportViewState | None = None,
    view_mode: str = "detector",
    background_layer: tk_primary_viewport._ViewportImageLayer | None = None,
    simulation_layer: tk_primary_viewport._ViewportImageLayer | None = None,
    overlay_layer: tk_primary_viewport._ViewportImageLayer | None = None,
    text_specs: tuple[tk_primary_viewport._ViewportTextSpec, ...] = (),
) -> tk_primary_viewport.ViewportScene:
    if view_state is None:
        view_state = tk_primary_viewport.ViewportViewState(
            width=4,
            height=4,
            xlim=(0.0, 2.0),
            ylim=(2.0, 0.0),
        )
    overlay_model = tk_primary_viewport.fast_plot_viewer.FastViewerOverlayModel()
    return tk_primary_viewport.ViewportScene(
        view_state=view_state,
        view_mode=view_mode,
        background_layer=background_layer,
        simulation_layer=simulation_layer,
        overlay_layer=overlay_layer,
        overlay_model=overlay_model,
        text_specs=text_specs,
        raster_signature=tk_primary_viewport._scene_raster_signature(
            view_state,
            view_mode=view_mode,
            background_layer=background_layer,
            simulation_layer=simulation_layer,
        ),
        overlay_signature=tk_primary_viewport._scene_overlay_signature(
            view_state,
            view_mode=view_mode,
            overlay_layer=overlay_layer,
            overlay_model=overlay_model,
            text_specs=text_specs,
        ),
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


def test_render_layer_patch_restores_original_detector_orientation() -> None:
    viewport = tk_primary_viewport._TkPrimaryViewport(
        tk_module=_FakeTkModule(),
        parent="canvas-parent",
        ax=_FakeAxes(xlim=(0.0, 1.0), ylim=(2.0, 0.0)),
        initial_width=1,
        initial_height=2,
    )
    view_state = tk_primary_viewport.ViewportViewState(
        width=1,
        height=2,
        xlim=(0.0, 1.0),
        ylim=(2.0, 0.0),
    )
    layer = tk_primary_viewport._ViewportImageLayer(
        name="simulation",
        visible=True,
        extent=(0.0, 1.0, 2.0, 0.0),
        interpolation="nearest",
        source_rgba=np.asarray(
            [
                [[255, 0, 0, 255]],
                [[0, 0, 255, 255]],
            ],
            dtype=np.uint8,
        ),
    )

    rendered = viewport._render_layer_patch(layer, view_state)

    assert rendered is not None
    patch_image, left, top = rendered
    assert (left, top) == (0, 0)
    assert patch_image.size == (1, 2)
    assert patch_image.getpixel((0, 0)) == (0, 0, 255, 255)
    assert patch_image.getpixel((0, 1)) == (255, 0, 0, 255)


def test_render_layer_patch_restores_original_caked_orientation() -> None:
    viewport = tk_primary_viewport._TkPrimaryViewport(
        tk_module=_FakeTkModule(),
        parent="canvas-parent",
        ax=_FakeAxes(xlim=(10.0, 20.0), ylim=(-30.0, 30.0)),
        initial_width=1,
        initial_height=3,
    )
    view_state = tk_primary_viewport.ViewportViewState(
        width=1,
        height=3,
        xlim=(10.0, 20.0),
        ylim=(-30.0, 30.0),
    )
    layer = _make_layer(
        "simulation",
        np.asarray(
            [
                [[255, 0, 0, 255]],
                [[0, 255, 0, 255]],
                [[0, 0, 255, 255]],
            ],
            dtype=np.uint8,
        ),
        extent=(10.0, 20.0, -30.0, 30.0),
        origin="lower",
    )

    rendered = viewport._render_layer_patch(layer, view_state)

    assert rendered is not None
    patch_image, left, top = rendered
    assert (left, top) == (0, 0)
    assert patch_image.size == (1, 3)
    assert patch_image.getpixel((0, 0)) == (255, 0, 0, 255)
    assert patch_image.getpixel((0, 1)) == (0, 255, 0, 255)
    assert patch_image.getpixel((0, 2)) == (0, 0, 255, 255)


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


def test_render_scene_overlay_only_redraw_reuses_base_raster(monkeypatch) -> None:
    if tk_primary_viewport.Image is None:
        pytest.skip("Pillow is unavailable")
    _FakePhotoImage.reset()
    monkeypatch.setattr(
        tk_primary_viewport,
        "ImageTk",
        SimpleNamespace(PhotoImage=_FakePhotoImage),
    )
    viewport = tk_primary_viewport._TkPrimaryViewport(
        tk_module=_FakeTkModule(),
        parent="canvas-parent",
        ax=_FakeAxes(xlim=(0.0, 2.0), ylim=(2.0, 0.0)),
        initial_width=4,
        initial_height=4,
    )
    background_layer = _make_layer(
        "background",
        np.full((2, 2, 4), [255, 255, 255, 255], dtype=np.uint8),
    )
    simulation_layer = _make_layer(
        "simulation",
        np.full((2, 2, 4), [255, 0, 0, 255], dtype=np.uint8),
    )
    scene_one = _make_scene(
        background_layer=background_layer,
        simulation_layer=simulation_layer,
        text_specs=(
            tk_primary_viewport._ViewportTextSpec(
                x=0.5,
                y=0.5,
                text="A",
                fill_rgba=(255, 255, 255, 255),
                font_size=10,
                zorder=5.0,
            ),
        ),
    )
    viewport.render_scene(scene_one)

    base_photo = viewport._photo_image
    base_itemconfigure_count = len(
        [
            call
            for call in viewport.widget.itemconfigure_calls
            if call[0] == viewport._photo_item
        ]
    )

    scene_two = _make_scene(
        background_layer=background_layer,
        simulation_layer=simulation_layer,
        text_specs=(
            tk_primary_viewport._ViewportTextSpec(
                x=1.0,
                y=1.0,
                text="B",
                fill_rgba=(255, 255, 0, 255),
                font_size=12,
                zorder=5.0,
            ),
        ),
    )
    viewport.render_scene(scene_two)

    assert viewport._photo_image is base_photo
    assert len(
        [
            call
            for call in viewport.widget.itemconfigure_calls
            if call[0] == viewport._photo_item
        ]
    ) == base_itemconfigure_count


def test_render_scene_noops_when_scene_signatures_match(monkeypatch) -> None:
    if tk_primary_viewport.Image is None:
        pytest.skip("Pillow is unavailable")
    _FakePhotoImage.reset()
    monkeypatch.setattr(
        tk_primary_viewport,
        "ImageTk",
        SimpleNamespace(PhotoImage=_FakePhotoImage),
    )
    viewport = tk_primary_viewport._TkPrimaryViewport(
        tk_module=_FakeTkModule(),
        parent="canvas-parent",
        ax=_FakeAxes(xlim=(0.0, 2.0), ylim=(2.0, 0.0)),
        initial_width=4,
        initial_height=4,
    )
    scene = _make_scene(
        background_layer=_make_layer(
            "background",
            np.full((2, 2, 4), [255, 255, 255, 255], dtype=np.uint8),
        ),
        simulation_layer=_make_layer(
            "simulation",
            np.full((2, 2, 4), [0, 0, 255, 255], dtype=np.uint8),
        ),
    )
    viewport.render_scene(scene)
    snapshot = (
        len(viewport.widget.create_calls),
        len(viewport.widget.itemconfigure_calls),
        len(viewport.widget.delete_calls),
        len(_FakePhotoImage.instances),
    )

    viewport.render_scene(scene)

    assert (
        len(viewport.widget.create_calls),
        len(viewport.widget.itemconfigure_calls),
        len(viewport.widget.delete_calls),
        len(_FakePhotoImage.instances),
    ) == snapshot


def test_render_scene_rebuilds_raster_when_bounds_or_layer_change(monkeypatch) -> None:
    if tk_primary_viewport.Image is None:
        pytest.skip("Pillow is unavailable")
    _FakePhotoImage.reset()
    monkeypatch.setattr(
        tk_primary_viewport,
        "ImageTk",
        SimpleNamespace(PhotoImage=_FakePhotoImage),
    )
    viewport = tk_primary_viewport._TkPrimaryViewport(
        tk_module=_FakeTkModule(),
        parent="canvas-parent",
        ax=_FakeAxes(xlim=(0.0, 2.0), ylim=(2.0, 0.0)),
        initial_width=4,
        initial_height=4,
    )
    base_scene = _make_scene(
        background_layer=_make_layer(
            "background",
            np.full((2, 2, 4), [255, 255, 255, 255], dtype=np.uint8),
        ),
        simulation_layer=_make_layer(
            "simulation",
            np.full((2, 2, 4), [255, 0, 0, 255], dtype=np.uint8),
        ),
    )
    viewport.render_scene(base_scene)

    initial_photo = viewport._photo_image
    updated_bounds_scene = _make_scene(
        view_state=tk_primary_viewport.ViewportViewState(
            width=5,
            height=4,
            xlim=(0.0, 2.0),
            ylim=(2.0, 0.0),
        ),
        background_layer=base_scene.background_layer,
        simulation_layer=base_scene.simulation_layer,
    )
    viewport.render_scene(updated_bounds_scene)

    assert viewport._photo_image is not initial_photo

    bounds_photo = viewport._photo_image
    updated_layer_scene = _make_scene(
        background_layer=base_scene.background_layer,
        simulation_layer=_make_layer(
            "simulation",
            np.full((2, 2, 4), [0, 255, 0, 255], dtype=np.uint8),
        ),
    )
    viewport.render_scene(updated_layer_scene)

    assert viewport._photo_image is not bounds_photo
