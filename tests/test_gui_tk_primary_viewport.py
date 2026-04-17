from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

RUNTIME_SESSION_SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "ra_sim" / "gui" / "_runtime" / "runtime_session.py"
)


def test_primary_viewport_source_uses_embedded_matplotlib_tk_canvas_only() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "FigureCanvasTkAgg" in source
    assert "matplotlib_canvas = figure_canvas_cls(" in source
    assert "master=app_shell_view_state.canvas_frame" in source
    assert "canvas = matplotlib_canvas" in source
    assert "_set_runtime_canvas(matplotlib_canvas)" in source
    assert "activate_runtime_primary_viewport(" not in source
    assert "PRIMARY_VIEWPORT_BACKEND" not in source
    assert "tk_canvas" not in source


def test_get_tk_figure_canvas_cls_returns_figure_canvas_tkagg(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(runtime_session, "_TK_FIGURE_CANVAS_CLS", None, raising=False)

    cls = runtime_session._get_tk_figure_canvas_cls()

    assert cls.__name__ == "FigureCanvasTkAgg"
    assert cls.__module__ == "matplotlib.backends.backend_tkagg"
    assert runtime_session._TK_FIGURE_CANVAS_CLS is cls


def test_configure_primary_viewport_redraw_helpers_use_matplotlib_redraw(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    redraw_calls: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda *, force=False: redraw_calls.append(bool(force)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_defer_nonessential_redraw",
        lambda: True,
        raising=False,
    )

    runtime_session._configure_primary_viewport_redraw_helpers()
    runtime_session._request_main_canvas_redraw(force_matplotlib=False)
    runtime_session._request_main_canvas_redraw(force_matplotlib=True)
    runtime_session._request_overlay_canvas_redraw(force=False)
    runtime_session._request_overlay_canvas_redraw(force=True)

    assert redraw_calls == [False, True, True]


def test_schedule_post_idle_main_canvas_redraw_uses_after_idle(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    scheduled_callbacks: list[object] = []
    redraw_calls: list[dict[str, object]] = []
    flush_calls: list[str] = []

    class _Root:
        def after_idle(self, callback) -> str:
            scheduled_callbacks.append(callback)
            return "idle-token"

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **kwargs: redraw_calls.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_flush_main_canvas_tk_present",
        lambda: flush_calls.append("flush"),
        raising=False,
    )

    runtime_session._schedule_post_idle_main_canvas_redraw()

    assert redraw_calls == []
    assert flush_calls == []
    assert len(scheduled_callbacks) == 1
    scheduled_callbacks[0]()
    assert redraw_calls == [{"force_matplotlib": True}]
    assert flush_calls == ["flush"]


def test_schedule_post_idle_main_canvas_redraw_falls_back_to_immediate_force(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    redraw_calls: list[dict[str, object]] = []
    flush_calls: list[str] = []

    class _Root:
        def after_idle(self, _callback) -> None:
            raise RuntimeError("idle-unavailable")

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **kwargs: redraw_calls.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_flush_main_canvas_tk_present",
        lambda: flush_calls.append("flush"),
        raising=False,
    )

    runtime_session._schedule_post_idle_main_canvas_redraw()

    assert redraw_calls == [{"force_matplotlib": True}]
    assert flush_calls == ["flush"]


def test_schedule_post_idle_main_canvas_redraw_falls_back_when_after_idle_missing(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    redraw_calls: list[dict[str, object]] = []
    flush_calls: list[str] = []

    class _Root:
        after_idle = None

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **kwargs: redraw_calls.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_flush_main_canvas_tk_present",
        lambda: flush_calls.append("flush"),
        raising=False,
    )

    runtime_session._schedule_post_idle_main_canvas_redraw()

    assert redraw_calls == [{"force_matplotlib": True}]
    assert flush_calls == ["flush"]


def test_flush_main_canvas_tk_present_prefers_canvas_widget(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    widget_calls: list[str] = []
    root_calls: list[str] = []

    class _Widget:
        def update_idletasks(self) -> None:
            widget_calls.append("widget")

    class _Canvas:
        def get_tk_widget(self):
            return _Widget()

    class _Root:
        def update_idletasks(self) -> None:
            root_calls.append("root")

    monkeypatch.setattr(runtime_session, "matplotlib_canvas", _Canvas(), raising=False)
    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)

    runtime_session._flush_main_canvas_tk_present()

    assert widget_calls == ["widget"]
    assert root_calls == []


def test_flush_main_canvas_tk_present_falls_back_to_root(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    root_calls: list[str] = []

    class _Canvas:
        def get_tk_widget(self):
            raise RuntimeError("widget-unavailable")

    class _Root:
        def update_idletasks(self) -> None:
            root_calls.append("root")

    monkeypatch.setattr(runtime_session, "matplotlib_canvas", _Canvas(), raising=False)
    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)

    runtime_session._flush_main_canvas_tk_present()

    assert root_calls == ["root"]


def test_clear_pending_main_figure_preview_interaction_clears_canvas_preview_cache(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_runtime_state = SimpleNamespace(
        _canvas_preview_limits=((4.0, 8.0), (9.0, 2.0)),
        _canvas_pan_session={"drag": True},
    )
    clear_calls: list[bool] = []
    cleared_tokens: list[object] = []
    refresh_calls: list[tuple[bool, bool, object, bool]] = []
    restore_callback_flags: list[bool] = []
    settled_overlay_calls: list[str] = []
    redraw_calls: list[bool] = []
    simulation_runtime_state = SimpleNamespace(
        interaction_drag_active=True,
        interaction_drag_requires_settled_update=True,
        interaction_settle_token="settle-token",
        main_matplotlib_overlays_suspended=True,
    )

    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        geometry_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "root", object(), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_clear_legacy_main_matplotlib_preview_view",
        lambda *, redraw=True: clear_calls.append(bool(redraw)) or False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda _root, token: cleared_tokens.append(token),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: settled_overlay_calls.append("settled"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda *, force=False: redraw_calls.append(bool(force)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_matplotlib_interaction,
        "restore_main_matplotlib_overlays",
        lambda runtime_state, *, restore_callback: (
            restore_callback_flags.append(callable(restore_callback))
            or setattr(runtime_state, "main_matplotlib_overlays_suspended", False)
            or (restore_callback() if callable(restore_callback) else None)
            or True
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_run_status_bar",
        lambda: refresh_calls.append(
            (
                bool(simulation_runtime_state.interaction_drag_active),
                bool(simulation_runtime_state.interaction_drag_requires_settled_update),
                simulation_runtime_state.interaction_settle_token,
                bool(simulation_runtime_state.main_matplotlib_overlays_suspended),
            )
        ),
        raising=False,
    )

    assert runtime_session._clear_pending_main_figure_preview_interaction() is True
    assert clear_calls == [False]
    assert cleared_tokens == ["settle-token"]
    assert restore_callback_flags == [False]
    assert settled_overlay_calls == []
    assert redraw_calls == []
    assert geometry_runtime_state._canvas_preview_limits is None
    assert geometry_runtime_state._canvas_pan_session is None
    assert simulation_runtime_state.interaction_drag_active is False
    assert simulation_runtime_state.interaction_drag_requires_settled_update is False
    assert simulation_runtime_state.interaction_settle_token is None
    assert simulation_runtime_state.main_matplotlib_overlays_suspended is False
    assert refresh_calls == [(False, False, None, False)]


def test_clear_pending_main_figure_preview_interaction_uses_no_redraw_preview_clear_path(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_runtime_state = SimpleNamespace(
        _canvas_preview_limits=((4.0, 8.0), (9.0, 2.0)),
        _canvas_pan_session={"drag": True},
    )
    preview_clear_calls: list[bool] = []
    cleared_tokens: list[object] = []
    restore_callback_flags: list[bool] = []
    settled_overlay_calls: list[str] = []
    redraw_calls: list[bool] = []
    refresh_calls: list[tuple[bool, bool, object, bool]] = []
    simulation_runtime_state = SimpleNamespace(
        interaction_drag_active=True,
        interaction_drag_requires_settled_update=True,
        interaction_settle_token="preview-token",
        main_matplotlib_overlays_suspended=True,
    )

    class _PreviewController:
        def clear_preview_view(self, *, redraw=True) -> bool:
            preview_clear_calls.append(bool(redraw))
            return True

    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        geometry_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "root", object(), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_legacy_main_matplotlib_preview_controller",
        lambda: _PreviewController(),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda _root, token: cleared_tokens.append(token),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: settled_overlay_calls.append("settled"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda *, force=False: redraw_calls.append(bool(force)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_matplotlib_interaction,
        "restore_main_matplotlib_overlays",
        lambda runtime_state, *, restore_callback: (
            restore_callback_flags.append(callable(restore_callback))
            or setattr(runtime_state, "main_matplotlib_overlays_suspended", False)
            or (restore_callback() if callable(restore_callback) else None)
            or True
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_run_status_bar",
        lambda: refresh_calls.append(
            (
                bool(simulation_runtime_state.interaction_drag_active),
                bool(simulation_runtime_state.interaction_drag_requires_settled_update),
                simulation_runtime_state.interaction_settle_token,
                bool(simulation_runtime_state.main_matplotlib_overlays_suspended),
            )
        ),
        raising=False,
    )

    assert runtime_session._clear_pending_main_figure_preview_interaction() is True
    assert preview_clear_calls == [False]
    assert cleared_tokens == ["preview-token"]
    assert restore_callback_flags == [False]
    assert settled_overlay_calls == []
    assert redraw_calls == []
    assert geometry_runtime_state._canvas_preview_limits is None
    assert geometry_runtime_state._canvas_pan_session is None
    assert simulation_runtime_state.interaction_drag_active is False
    assert simulation_runtime_state.interaction_drag_requires_settled_update is False
    assert simulation_runtime_state.interaction_settle_token is None
    assert simulation_runtime_state.main_matplotlib_overlays_suspended is False
    assert refresh_calls == [(False, False, None, False)]


def test_clear_legacy_main_matplotlib_preview_view_restores_overlays_when_preview_clears(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    cleared_tokens: list[object] = []
    restore_callback_flags: list[bool] = []
    settled_overlay_calls: list[str] = []
    redraw_calls: list[bool] = []
    refresh_calls: list[tuple[bool, bool, object, bool]] = []
    simulation_runtime_state = SimpleNamespace(
        interaction_drag_active=True,
        interaction_drag_requires_settled_update=True,
        interaction_settle_token="preview-token",
        main_matplotlib_overlays_suspended=True,
    )

    class _PreviewController:
        def clear_preview_view(self, *, redraw=True) -> bool:
            return True

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "root", object(), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_legacy_main_matplotlib_preview_controller",
        lambda: _PreviewController(),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda _root, token: cleared_tokens.append(token),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: settled_overlay_calls.append("settled"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda *, force=False: redraw_calls.append(bool(force)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_matplotlib_interaction,
        "restore_main_matplotlib_overlays",
        lambda runtime_state, *, restore_callback: (
            restore_callback_flags.append(callable(restore_callback))
            or setattr(runtime_state, "main_matplotlib_overlays_suspended", False)
            or (restore_callback() if callable(restore_callback) else None)
            or True
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_run_status_bar",
        lambda: refresh_calls.append(
            (
                bool(simulation_runtime_state.interaction_drag_active),
                bool(simulation_runtime_state.interaction_drag_requires_settled_update),
                simulation_runtime_state.interaction_settle_token,
                bool(simulation_runtime_state.main_matplotlib_overlays_suspended),
            )
        ),
        raising=False,
    )

    assert runtime_session._clear_legacy_main_matplotlib_preview_view(redraw=False) is True
    assert cleared_tokens == ["preview-token"]
    assert restore_callback_flags == [False]
    assert settled_overlay_calls == []
    assert redraw_calls == []
    assert simulation_runtime_state.interaction_drag_active is False
    assert simulation_runtime_state.interaction_drag_requires_settled_update is False
    assert simulation_runtime_state.interaction_settle_token is None
    assert simulation_runtime_state.main_matplotlib_overlays_suspended is False
    assert refresh_calls == [(False, False, None, False)]


def test_reset_main_figure_live_interaction_state_clears_overlay_flag_when_restore_fails(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    cleared_tokens: list[object] = []
    settled_overlay_calls: list[str] = []
    redraw_calls: list[bool] = []
    refresh_calls: list[tuple[bool, bool, object, bool]] = []
    simulation_runtime_state = SimpleNamespace(
        interaction_drag_active=True,
        interaction_drag_requires_settled_update=True,
        interaction_settle_token="preview-token",
        main_matplotlib_overlays_suspended=True,
    )

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "root", object(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda _root, token: cleared_tokens.append(token),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_legacy_main_matplotlib_interaction_active",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: settled_overlay_calls.append("settled"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda *, force=False: redraw_calls.append(bool(force)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_matplotlib_interaction,
        "restore_main_matplotlib_overlays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_run_status_bar",
        lambda: refresh_calls.append(
            (
                bool(simulation_runtime_state.interaction_drag_active),
                bool(simulation_runtime_state.interaction_drag_requires_settled_update),
                simulation_runtime_state.interaction_settle_token,
                bool(simulation_runtime_state.main_matplotlib_overlays_suspended),
            )
        ),
        raising=False,
    )

    runtime_session._reset_main_figure_live_interaction_state(redraw=False)

    assert cleared_tokens == ["preview-token"]
    assert settled_overlay_calls == []
    assert redraw_calls == []
    assert simulation_runtime_state.interaction_drag_active is False
    assert simulation_runtime_state.interaction_drag_requires_settled_update is False
    assert simulation_runtime_state.interaction_settle_token is None
    assert simulation_runtime_state.main_matplotlib_overlays_suspended is False
    assert refresh_calls == [(False, False, None, False)]


def test_clear_legacy_main_matplotlib_preview_view_redraw_true_restores_overlays_with_redraw(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    cleared_tokens: list[object] = []
    restore_callback_flags: list[bool] = []
    settled_overlay_calls: list[str] = []
    redraw_calls: list[bool] = []
    refresh_calls: list[tuple[bool, bool, object, bool]] = []
    simulation_runtime_state = SimpleNamespace(
        interaction_drag_active=True,
        interaction_drag_requires_settled_update=True,
        interaction_settle_token="preview-token",
        main_matplotlib_overlays_suspended=True,
    )

    class _PreviewController:
        def clear_preview_view(self, *, redraw=True) -> bool:
            return bool(redraw)

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "root", object(), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_legacy_main_matplotlib_preview_controller",
        lambda: _PreviewController(),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda _root, token: cleared_tokens.append(token),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: settled_overlay_calls.append("settled"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda *, force=False: redraw_calls.append(bool(force)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_matplotlib_interaction,
        "restore_main_matplotlib_overlays",
        lambda runtime_state, *, restore_callback: (
            restore_callback_flags.append(callable(restore_callback))
            or setattr(runtime_state, "main_matplotlib_overlays_suspended", False)
            or (restore_callback() if callable(restore_callback) else None)
            or True
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_run_status_bar",
        lambda: refresh_calls.append(
            (
                bool(simulation_runtime_state.interaction_drag_active),
                bool(simulation_runtime_state.interaction_drag_requires_settled_update),
                simulation_runtime_state.interaction_settle_token,
                bool(simulation_runtime_state.main_matplotlib_overlays_suspended),
            )
        ),
        raising=False,
    )

    assert runtime_session._clear_legacy_main_matplotlib_preview_view(redraw=True) is True
    assert cleared_tokens == ["preview-token"]
    assert restore_callback_flags == [True]
    assert settled_overlay_calls == ["settled"]
    assert redraw_calls == [True]
    assert simulation_runtime_state.interaction_drag_active is False
    assert simulation_runtime_state.interaction_drag_requires_settled_update is False
    assert simulation_runtime_state.interaction_settle_token is None
    assert simulation_runtime_state.main_matplotlib_overlays_suspended is False
    assert refresh_calls == [(False, False, None, False)]


def test_apply_main_caked_view_toggle_clears_pending_preview_before_redraw(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_runtime_state = SimpleNamespace(
        _canvas_preview_limits=((4.0, 8.0), (9.0, 2.0)),
        _canvas_pan_session={"drag": True},
    )
    captured_states: list[tuple[object, object]] = []
    redraw_states: list[tuple[object, object, bool, bool, dict[str, object]]] = []
    simulation_runtime_state = SimpleNamespace(
        unscaled_image=np.ones((1, 1), dtype=np.float64),
        interaction_drag_active=True,
        interaction_drag_requires_settled_update=True,
        interaction_settle_token="pending",
        main_matplotlib_overlays_suspended=True,
    )

    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        geometry_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_caked_2d_var=SimpleNamespace(get=lambda: False)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "root", object(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 8, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_clear_legacy_main_matplotlib_preview_view",
        lambda *, redraw=True: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_restore_combined_detector_intersection_cache",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_primary_figure_mode",
        lambda: "caked",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_canvas_interactions,
        "capture_axis_limits",
        lambda axis: (
            captured_states.append(
                (
                    geometry_runtime_state._canvas_preview_limits,
                    geometry_runtime_state._canvas_pan_session,
                )
            )
            or ((1.0, 7.0), (8.0, 2.0))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "apply_scale_factor_to_existing_results",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_canvas_interactions,
        "restore_axis_view",
        lambda *args, **kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            set_aspect=lambda *_args, **_kwargs: None,
            set_xlabel=lambda *_args, **_kwargs: None,
            set_ylabel=lambda *_args, **_kwargs: None,
            set_title=lambda *_args, **_kwargs: None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "set_main_figure_axes_axis_visibility",
        lambda axis, *, visible: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **kwargs: redraw_states.append(
            (
                geometry_runtime_state._canvas_preview_limits,
                geometry_runtime_state._canvas_pan_session,
                bool(simulation_runtime_state.interaction_drag_active),
                bool(simulation_runtime_state.main_matplotlib_overlays_suspended),
                dict(kwargs),
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_run_status_bar",
        lambda: None,
        raising=False,
    )

    runtime_session._apply_main_caked_view_toggle()

    assert geometry_runtime_state._canvas_preview_limits is None
    assert geometry_runtime_state._canvas_pan_session is None
    assert captured_states == [(None, None)]
    assert simulation_runtime_state.interaction_drag_active is False
    assert simulation_runtime_state.interaction_drag_requires_settled_update is False
    assert simulation_runtime_state.interaction_settle_token is None
    assert simulation_runtime_state.main_matplotlib_overlays_suspended is False
    assert redraw_states == [(None, None, False, False, {"force_matplotlib": False})]


def test_reset_primary_figure_view_forces_embedded_matplotlib_redraw(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    xlim_calls: list[tuple[float, float]] = []
    ylim_calls: list[tuple[float, float]] = []
    redraw_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "_default_primary_view_limits",
        lambda: (1.0, 8.0, 9.0, 2.0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            set_xlim=lambda left, right: xlim_calls.append((float(left), float(right))),
            set_ylim=lambda bottom, top: ylim_calls.append((float(bottom), float(top))),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **kwargs: redraw_calls.append(dict(kwargs)),
        raising=False,
    )

    runtime_session._reset_primary_figure_view()

    assert xlim_calls == [(1.0, 8.0)]
    assert ylim_calls == [(9.0, 2.0)]
    assert redraw_calls == [{"force_matplotlib": True}]


def test_reset_primary_figure_view_clears_pending_preview_before_force_redraw(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_runtime_state = SimpleNamespace(
        _canvas_preview_limits=((4.0, 8.0), (9.0, 2.0)),
        _canvas_pan_session={"drag": True},
    )
    redraw_states: list[tuple[object, object, dict[str, object]]] = []
    simulation_runtime_state = SimpleNamespace(
        interaction_drag_active=True,
        interaction_drag_requires_settled_update=True,
        interaction_settle_token="reset-pending",
        main_matplotlib_overlays_suspended=True,
    )

    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        geometry_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "root", object(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_run_status_bar",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_legacy_main_matplotlib_preview_view",
        lambda *, redraw=True: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_default_primary_view_limits",
        lambda: (1.0, 8.0, 9.0, 2.0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            set_xlim=lambda *_args, **_kwargs: None,
            set_ylim=lambda *_args, **_kwargs: None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **kwargs: redraw_states.append(
            (
                geometry_runtime_state._canvas_preview_limits,
                geometry_runtime_state._canvas_pan_session,
                dict(kwargs),
            )
        ),
        raising=False,
    )

    runtime_session._reset_primary_figure_view()

    assert geometry_runtime_state._canvas_preview_limits is None
    assert geometry_runtime_state._canvas_pan_session is None
    assert redraw_states == [(None, None, {"force_matplotlib": True})]


class _FakeTkWidget:
    def __init__(self) -> None:
        self.manager = ""
        self.pack_calls: list[dict[str, object]] = []

    def winfo_manager(self) -> str:
        return self.manager

    def pack(self, **kwargs) -> None:
        self.pack_calls.append(dict(kwargs))
        self.manager = "pack"


class _FakeFigureCanvasTkAgg(FigureCanvasAgg):
    def __init__(self, figure, master=None) -> None:
        self.master = master
        self.widget = _FakeTkWidget()
        super().__init__(figure)

    def get_tk_widget(self):
        return self.widget


def test_initialize_runtime_plot_block_packs_embedded_canvas_widget(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    configured_widgets = []

    monkeypatch.setattr(
        runtime_session,
        "_get_tk_figure_canvas_cls",
        lambda: _FakeFigureCanvasTkAgg,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "app_shell_view_state",
        SimpleNamespace(canvas_frame=object()),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(main_display_raster_limit=8, unscaled_image="before"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_display=np.zeros((1, 1), dtype=np.float32)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "configure_matplotlib_canvas_widget",
        lambda widget: configured_widgets.append(widget),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_integration_range_drag,
        "create_integration_region_highlight_cmap",
        lambda *, listed_colormap_cls: listed_colormap_cls(
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0],
                ]
            )
        ),
        raising=False,
    )

    runtime_session._initialize_runtime_plot_block_01()

    widget = runtime_session.matplotlib_canvas_widget
    assert configured_widgets == [widget]
    assert widget.pack_calls == [
        {"side": runtime_session.tk.TOP, "fill": runtime_session.tk.BOTH, "expand": True}
    ]
    assert runtime_session.canvas is runtime_session.matplotlib_canvas
