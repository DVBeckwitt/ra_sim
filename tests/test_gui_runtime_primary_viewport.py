from __future__ import annotations

from types import SimpleNamespace

from ra_sim.gui import runtime_primary_viewport


class _FakeWidget:
    def __init__(self) -> None:
        self.manager = ""
        self.pack_calls: list[dict[str, object]] = []

    def winfo_manager(self) -> str:
        return self.manager

    def pack(self, **kwargs) -> None:
        self.manager = "pack"
        self.pack_calls.append(dict(kwargs))


class _FakeMatplotlibCanvas:
    def __init__(self) -> None:
        self.widget = _FakeWidget()

    def get_tk_widget(self):
        return self.widget


class _FakeBackend:
    def __init__(self, *, activate_error: Exception | None = None) -> None:
        self.canvas_proxy = object()
        self.activate_error = activate_error
        self.activate_calls = 0
        self.deactivate_calls = 0
        self.shutdown_calls = 0

    def activate(self) -> None:
        self.activate_calls += 1
        if self.activate_error is not None:
            raise self.activate_error

    def deactivate(self) -> None:
        self.deactivate_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_activate_runtime_primary_viewport_uses_matplotlib_when_requested() -> None:
    matplotlib_canvas = _FakeMatplotlibCanvas()
    progress_texts: list[str] = []

    selection = runtime_primary_viewport.activate_runtime_primary_viewport(
        requested_backend="matplotlib",
        tk_primary_viewport_module=SimpleNamespace(
            build_tk_primary_viewport_backend=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("Tk backend should not be built")
            )
        ),
        tk_module=SimpleNamespace(TOP="top", BOTH="both"),
        canvas_frame="canvas-frame",
        matplotlib_canvas=matplotlib_canvas,
        ax="AX",
        image_artist="image",
        background_artist="background",
        overlay_artist="overlay",
        set_progress_text=progress_texts.append,
    )

    assert selection.active_backend == "matplotlib"
    assert selection.backend is None
    assert selection.canvas_proxy is matplotlib_canvas
    assert selection.fallback_reason is None
    assert matplotlib_canvas.widget.pack_calls == [
        {"side": "top", "fill": "both", "expand": True}
    ]
    assert progress_texts == []


def test_activate_runtime_primary_viewport_activates_tk_canvas_when_available() -> None:
    matplotlib_canvas = _FakeMatplotlibCanvas()
    backend = _FakeBackend()
    build_calls: list[dict[str, object]] = []
    progress_texts: list[str] = []

    selection = runtime_primary_viewport.activate_runtime_primary_viewport(
        requested_backend="tk_canvas",
        tk_primary_viewport_module=SimpleNamespace(
            primary_viewport_runtime_available=lambda: True,
            build_tk_primary_viewport_backend=lambda **kwargs: (
                build_calls.append(dict(kwargs)) or backend
            ),
        ),
        tk_module=SimpleNamespace(TOP="top", BOTH="both"),
        canvas_frame="canvas-frame",
        matplotlib_canvas=matplotlib_canvas,
        ax="AX",
        image_artist="image",
        background_artist="background",
        overlay_artist="overlay",
        draw_interval_s=0.25,
        set_progress_text=progress_texts.append,
    )

    assert selection.active_backend == "tk_canvas"
    assert selection.backend is backend
    assert selection.canvas_proxy is backend.canvas_proxy
    assert selection.fallback_reason is None
    assert backend.activate_calls == 1
    assert build_calls == [
        {
            "tk_module": SimpleNamespace(TOP="top", BOTH="both"),
            "canvas_frame": "canvas-frame",
            "matplotlib_canvas": matplotlib_canvas,
            "ax": "AX",
            "image_artist": "image",
            "background_artist": "background",
            "overlay_artist": "overlay",
            "marker_artist_factory": None,
            "overlay_model_factory": None,
            "overlay_artist_groups_factory": None,
            "layer_versions_factory": None,
            "peak_cache_factory": None,
            "qgroup_cache_factory": None,
            "draw_interval_s": 0.25,
        }
    ]
    assert progress_texts == []


def test_activate_runtime_primary_viewport_falls_back_when_tk_runtime_unavailable() -> None:
    matplotlib_canvas = _FakeMatplotlibCanvas()
    progress_texts: list[str] = []

    selection = runtime_primary_viewport.activate_runtime_primary_viewport(
        requested_backend="tk_canvas",
        tk_primary_viewport_module=SimpleNamespace(
            primary_viewport_runtime_available=lambda: False,
            primary_viewport_unavailable_reason=lambda: "Pillow ImageTk is unavailable",
            build_tk_primary_viewport_backend=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("Tk backend should not be built")
            ),
        ),
        tk_module=SimpleNamespace(TOP="top", BOTH="both"),
        canvas_frame="canvas-frame",
        matplotlib_canvas=matplotlib_canvas,
        ax="AX",
        image_artist="image",
        background_artist="background",
        overlay_artist="overlay",
        set_progress_text=progress_texts.append,
    )

    assert selection.active_backend == "matplotlib"
    assert selection.backend is None
    assert selection.canvas_proxy is matplotlib_canvas
    assert selection.fallback_reason == "Pillow ImageTk is unavailable"
    assert matplotlib_canvas.widget.pack_calls == [
        {"side": "top", "fill": "both", "expand": True}
    ]
    assert progress_texts == [
        "Tk canvas main viewport unavailable; using Matplotlib fallback."
    ]


def test_activate_runtime_primary_viewport_falls_back_after_activation_failure() -> None:
    matplotlib_canvas = _FakeMatplotlibCanvas()
    backend = _FakeBackend(activate_error=RuntimeError("boom"))
    progress_texts: list[str] = []

    selection = runtime_primary_viewport.activate_runtime_primary_viewport(
        requested_backend="tk_canvas",
        tk_primary_viewport_module=SimpleNamespace(
            primary_viewport_runtime_available=lambda: True,
            build_tk_primary_viewport_backend=lambda **_kwargs: backend,
        ),
        tk_module=SimpleNamespace(TOP="top", BOTH="both"),
        canvas_frame="canvas-frame",
        matplotlib_canvas=matplotlib_canvas,
        ax="AX",
        image_artist="image",
        background_artist="background",
        overlay_artist="overlay",
        set_progress_text=progress_texts.append,
    )

    assert selection.active_backend == "matplotlib"
    assert selection.backend is None
    assert selection.canvas_proxy is matplotlib_canvas
    assert selection.fallback_reason == "boom"
    assert backend.activate_calls == 1
    assert backend.deactivate_calls == 1
    assert backend.shutdown_calls == 1
    assert matplotlib_canvas.widget.pack_calls == [
        {"side": "top", "fill": "both", "expand": True}
    ]
    assert progress_texts == [
        "Tk canvas main viewport unavailable; using Matplotlib fallback."
    ]
