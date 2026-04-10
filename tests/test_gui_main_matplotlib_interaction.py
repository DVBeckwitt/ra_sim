from types import SimpleNamespace

from matplotlib.transforms import IdentityTransform

from ra_sim.gui import main_matplotlib_interaction


class _FakeWidget:
    def __init__(self) -> None:
        self._after_callbacks = {}
        self._next_after_token = 0

    def after(self, _delay, callback):
        self._next_after_token += 1
        token = f"after-{self._next_after_token}"
        self._after_callbacks[token] = callback
        return token

    def after_cancel(self, token) -> None:
        self._after_callbacks.pop(token, None)

    def run_after_callbacks(self) -> None:
        callbacks = list(self._after_callbacks.items())
        self._after_callbacks = {}
        for _token, callback in callbacks:
            callback()


class _FakeBbox:
    def __init__(self) -> None:
        self.x0 = 0.0
        self.y0 = 0.0
        self.width = 200.0
        self.height = 100.0


class _FakeAxis:
    def __init__(self, *, xlim=(0.0, 10.0), ylim=(20.0, 0.0)) -> None:
        self._xlim = tuple(float(value) for value in xlim)
        self._ylim = tuple(float(value) for value in ylim)
        self.bbox = _FakeBbox()
        self.transData = IdentityTransform()
        self.draw_artist_calls = []

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def set_xlim(self, left, right):
        self._xlim = (float(left), float(right))

    def set_ylim(self, bottom, top):
        self._ylim = (float(bottom), float(top))

    def draw_artist(self, artist) -> None:
        self.draw_artist_calls.append(artist)


class _FakeArtist:
    def __init__(self, *, visible=True) -> None:
        self._visible = bool(visible)
        self._animated = False
        self._transform = IdentityTransform()

    def get_visible(self):
        return self._visible

    def set_visible(self, visible) -> None:
        self._visible = bool(visible)

    def get_animated(self):
        return self._animated

    def set_animated(self, animated) -> None:
        self._animated = bool(animated)

    def get_transform(self):
        return self._transform

    def set_transform(self, transform) -> None:
        self._transform = transform


class _FakeCanvas:
    def __init__(self, *, supports_blit=True) -> None:
        self.supports_blit = bool(supports_blit)
        self.draw_calls = 0
        self.copy_calls = 0
        self.restore_calls = []
        self.blit_calls = []

    def draw(self) -> None:
        self.draw_calls += 1

    def copy_from_bbox(self, bbox):
        self.copy_calls += 1
        return ("background", bbox, self.copy_calls)

    def restore_region(self, background) -> None:
        self.restore_calls.append(background)

    def blit(self, bbox) -> None:
        self.blit_calls.append(bbox)


def test_request_main_matplotlib_redraw_coalesces_pending_draws() -> None:
    widget = _FakeWidget()
    runtime_state = SimpleNamespace(
        main_matplotlib_redraw_token=None,
        main_matplotlib_last_draw_ts=None,
    )
    clock = {"value": 100.0}
    draw_calls = []

    def _draw() -> bool:
        draw_calls.append(clock["value"])
        return True

    assert (
        main_matplotlib_interaction.request_main_matplotlib_redraw(
            widget=widget,
            runtime_state=runtime_state,
            interval_s=0.033,
            perf_counter_fn=lambda: clock["value"],
            draw_now=_draw,
            force=False,
        )
        is True
    )
    assert draw_calls == [100.0]
    assert runtime_state.main_matplotlib_redraw_token is None

    clock["value"] = 100.01
    assert (
        main_matplotlib_interaction.request_main_matplotlib_redraw(
            widget=widget,
            runtime_state=runtime_state,
            interval_s=0.033,
            perf_counter_fn=lambda: clock["value"],
            draw_now=_draw,
            force=False,
        )
        is True
    )
    first_token = runtime_state.main_matplotlib_redraw_token
    assert first_token is not None
    assert draw_calls == [100.0]

    clock["value"] = 100.02
    assert (
        main_matplotlib_interaction.request_main_matplotlib_redraw(
            widget=widget,
            runtime_state=runtime_state,
            interval_s=0.033,
            perf_counter_fn=lambda: clock["value"],
            draw_now=_draw,
            force=False,
        )
        is False
    )
    assert runtime_state.main_matplotlib_redraw_token == first_token
    assert draw_calls == [100.0]

    clock["value"] = 100.04
    widget.run_after_callbacks()
    assert runtime_state.main_matplotlib_redraw_token is None
    assert draw_calls == [100.0, 100.04]


def test_force_main_matplotlib_redraw_cancels_pending_token() -> None:
    widget = _FakeWidget()
    runtime_state = SimpleNamespace(
        main_matplotlib_redraw_token=None,
        main_matplotlib_last_draw_ts=50.0,
    )
    clock = {"value": 50.01}
    draw_calls = []

    def _draw() -> bool:
        draw_calls.append(clock["value"])
        return True

    main_matplotlib_interaction.request_main_matplotlib_redraw(
        widget=widget,
        runtime_state=runtime_state,
        interval_s=0.05,
        perf_counter_fn=lambda: clock["value"],
        draw_now=_draw,
        force=False,
    )
    assert runtime_state.main_matplotlib_redraw_token is not None
    assert draw_calls == []

    clock["value"] = 50.02
    assert (
        main_matplotlib_interaction.request_main_matplotlib_redraw(
            widget=widget,
            runtime_state=runtime_state,
            interval_s=0.05,
            perf_counter_fn=lambda: clock["value"],
            draw_now=_draw,
            force=True,
        )
        is True
    )
    assert runtime_state.main_matplotlib_redraw_token is None
    assert draw_calls == [50.02]


def test_suspend_and_restore_main_matplotlib_overlays_are_one_shot() -> None:
    runtime_state = SimpleNamespace(main_matplotlib_overlays_suspended=False)
    events = []

    assert (
        main_matplotlib_interaction.suspend_main_matplotlib_overlays(
            runtime_state,
            suspend_callback=lambda: events.append("suspend"),
        )
        is True
    )
    assert runtime_state.main_matplotlib_overlays_suspended is True
    assert events == ["suspend"]

    assert (
        main_matplotlib_interaction.suspend_main_matplotlib_overlays(
            runtime_state,
            suspend_callback=lambda: events.append("suspend-again"),
        )
        is False
    )
    assert events == ["suspend"]

    assert (
        main_matplotlib_interaction.restore_main_matplotlib_overlays(
            runtime_state,
            restore_callback=lambda: events.append("restore"),
        )
        is True
    )
    assert runtime_state.main_matplotlib_overlays_suspended is False
    assert events == ["suspend", "restore"]

    assert (
        main_matplotlib_interaction.restore_main_matplotlib_overlays(
            runtime_state,
            restore_callback=lambda: events.append("restore-again"),
        )
        is False
    )
    assert events == ["suspend", "restore"]


def _preview_runtime_state() -> SimpleNamespace:
    return SimpleNamespace(
        main_matplotlib_preview_active=False,
        main_matplotlib_preview_base_limits=None,
        main_matplotlib_preview_target_limits=None,
        main_matplotlib_preview_background_valid=False,
    )


def test_preview_update_uses_blit_path_when_background_is_available() -> None:
    axis = _FakeAxis()
    canvas = _FakeCanvas()
    background_artist = _FakeArtist()
    image_artist = _FakeArtist()
    center_marker = _FakeArtist()
    selected_peak_marker = _FakeArtist(visible=False)
    overlay_artist = _FakeArtist()
    runtime_state = _preview_runtime_state()

    controller = main_matplotlib_interaction.MainMatplotlibPreviewController(
        axis=axis,
        canvas=canvas,
        runtime_state=runtime_state,
        preview_artist_factory=lambda: (
            background_artist,
            image_artist,
            center_marker,
            selected_peak_marker,
        ),
        hidden_artist_factory=lambda: (overlay_artist,),
    )

    assert controller.preview_view_limits((2.0, 8.0), (18.0, 6.0)) is True
    assert runtime_state.main_matplotlib_preview_active is True
    assert runtime_state.main_matplotlib_preview_target_limits == (
        (2.0, 8.0),
        (18.0, 6.0),
    )
    assert canvas.draw_calls == 1
    assert canvas.copy_calls == 1
    assert canvas.restore_calls == [("background", axis.bbox, 1)]
    assert canvas.blit_calls == [axis.bbox]
    assert axis.draw_artist_calls == [background_artist, image_artist, center_marker]
    assert overlay_artist.get_visible() is False
    assert background_artist.get_animated() is True
    assert image_artist.get_animated() is True
    assert center_marker.get_animated() is True
    assert selected_peak_marker.get_animated() is True


def test_preview_invalidation_falls_back_without_leaving_artists_hidden() -> None:
    axis = _FakeAxis()
    canvas = _FakeCanvas()
    background_artist = _FakeArtist()
    image_artist = _FakeArtist()
    overlay_artist = _FakeArtist()
    runtime_state = _preview_runtime_state()

    controller = main_matplotlib_interaction.MainMatplotlibPreviewController(
        axis=axis,
        canvas=canvas,
        runtime_state=runtime_state,
        preview_artist_factory=lambda: (
            background_artist,
            image_artist,
        ),
        hidden_artist_factory=lambda: (overlay_artist,),
    )

    assert controller.preview_view_limits((2.0, 8.0), (18.0, 6.0)) is True
    controller.invalidate_background()

    assert controller.preview_view_limits((1.0, 9.0), (19.0, 5.0)) is False
    assert runtime_state.main_matplotlib_preview_active is False
    assert runtime_state.main_matplotlib_preview_target_limits is None
    assert overlay_artist.get_visible() is True
    assert background_artist.get_animated() is False
    assert image_artist.get_animated() is False
    assert canvas.draw_calls == 2


def test_preview_commit_restores_artist_state_and_applies_limits() -> None:
    axis = _FakeAxis()
    canvas = _FakeCanvas()
    background_artist = _FakeArtist()
    image_artist = _FakeArtist()
    center_marker = _FakeArtist()
    overlay_artist = _FakeArtist()
    runtime_state = _preview_runtime_state()

    original_background_transform = background_artist.get_transform()
    original_image_transform = image_artist.get_transform()
    original_marker_transform = center_marker.get_transform()

    controller = main_matplotlib_interaction.MainMatplotlibPreviewController(
        axis=axis,
        canvas=canvas,
        runtime_state=runtime_state,
        preview_artist_factory=lambda: (
            background_artist,
            image_artist,
            center_marker,
        ),
        hidden_artist_factory=lambda: (overlay_artist,),
    )

    assert controller.preview_view_limits((2.0, 8.0), (18.0, 6.0)) is True
    assert controller.commit_preview_view() is True

    assert axis.get_xlim() == (2.0, 8.0)
    assert axis.get_ylim() == (18.0, 6.0)
    assert runtime_state.main_matplotlib_preview_active is False
    assert runtime_state.main_matplotlib_preview_base_limits is None
    assert runtime_state.main_matplotlib_preview_target_limits is None
    assert runtime_state.main_matplotlib_preview_background_valid is False
    assert overlay_artist.get_visible() is True
    assert background_artist.get_transform() is original_background_transform
    assert image_artist.get_transform() is original_image_transform
    assert center_marker.get_transform() is original_marker_transform
    assert background_artist.get_animated() is False
    assert image_artist.get_animated() is False
    assert center_marker.get_animated() is False
