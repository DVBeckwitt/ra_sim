from __future__ import annotations

from types import SimpleNamespace

from ra_sim.gui import runtime_geometry_preview


def test_build_runtime_geometry_q_group_workflow_exposes_runtime_aliases_and_late_bound_schedule() -> None:
    calls: list[tuple[str, object]] = []
    callbacks = SimpleNamespace(
        live_preview_enabled="live-preview-enabled",
        render_live_preview_state="render-live-preview-state",
        set_preview_exclude_mode="set-preview-exclude-mode",
        clear_preview_exclusions="clear-preview-exclusions",
        toggle_preview_exclusion_at="toggle-preview-exclusion-at",
        toggle_live_preview="toggle-live-preview",
    )
    runtime_obj = SimpleNamespace(
        bindings_factory="geometry-q-group-bindings-factory",
        callbacks=callbacks,
    )
    bootstrap_module = SimpleNamespace(
        build_runtime_geometry_q_group_bootstrap=lambda **kwargs: (
            calls.append(("geometry-q-group", kwargs)) or runtime_obj
        )
    )
    schedule_state = {"value": "schedule-update-1"}

    workflow = runtime_geometry_preview.build_runtime_geometry_q_group_workflow(
        bootstrap_module=bootstrap_module,
        geometry_q_group_manager_module="geometry-q-group-manager-module",
        root="root-window",
        schedule_update_resolver=lambda: schedule_state["value"],
        fit_config="fit-config",
        view_state="view-state",
    )

    assert workflow.runtime is runtime_obj
    assert workflow.bindings_factory == "geometry-q-group-bindings-factory"
    assert workflow.callbacks is callbacks
    assert workflow.live_preview_enabled == "live-preview-enabled"
    assert workflow.render_live_preview_state == "render-live-preview-state"
    assert workflow.set_preview_exclude_mode == "set-preview-exclude-mode"
    assert workflow.clear_preview_exclusions == "clear-preview-exclusions"
    assert workflow.toggle_preview_exclusion_at == "toggle-preview-exclusion-at"
    assert workflow.toggle_live_preview == "toggle-live-preview"
    assert len(calls) == 1
    geometry_call = calls[0]
    assert geometry_call[0] == "geometry-q-group"
    assert geometry_call[1]["geometry_q_group_manager_module"] == (
        "geometry-q-group-manager-module"
    )
    assert geometry_call[1]["root"] == "root-window"
    assert geometry_call[1]["fit_config"] == "fit-config"
    assert geometry_call[1]["view_state"] == "view-state"
    forwarded_schedule_factory = geometry_call[1]["schedule_update_factory"]
    assert callable(forwarded_schedule_factory)
    assert forwarded_schedule_factory() == "schedule-update-1"
    schedule_state["value"] = "schedule-update-2"
    assert forwarded_schedule_factory() == "schedule-update-2"


def test_build_runtime_canvas_interaction_workflow_uses_late_bound_preview_callbacks() -> None:
    calls: list[tuple[str, object]] = []
    runtime_callbacks = SimpleNamespace(
        on_click="on-click",
        on_press="on-press",
        on_motion="on-motion",
        on_release="on-release",
    )
    runtime_obj = SimpleNamespace(
        bindings_factory="canvas-bindings-factory",
        callbacks=runtime_callbacks,
    )
    bootstrap_module = SimpleNamespace(
        build_runtime_canvas_interaction_bootstrap=lambda **kwargs: (
            calls.append(("canvas", kwargs)) or runtime_obj
        )
    )
    manager_events: list[tuple[str, object, object, object | None]] = []
    geometry_q_group_manager_module = SimpleNamespace(
        set_runtime_geometry_preview_exclude_mode=(
            lambda bindings, enabled, message=None: (
                manager_events.append(("set", bindings, enabled, message)) or "set-result"
            )
        ),
        toggle_runtime_live_geometry_preview_exclusion_at=(
            lambda bindings, col, row: (
                manager_events.append(("toggle", bindings, col, row)) or "toggle-result"
            )
        ),
    )
    bindings_factory_state = {
        "value": (lambda: "geometry-bindings-1"),
    }

    workflow = runtime_geometry_preview.build_runtime_canvas_interaction_workflow(
        bootstrap_module=bootstrap_module,
        canvas_interactions_module="canvas-interactions-module",
        geometry_q_group_manager_module=geometry_q_group_manager_module,
        geometry_q_group_runtime_bindings_factory_resolver=(
            lambda: bindings_factory_state["value"]
        ),
        axis="axis",
        peak_selection_callbacks="peak-selection-callbacks",
    )

    assert workflow.runtime is runtime_obj
    assert workflow.bindings_factory == "canvas-bindings-factory"
    assert workflow.callbacks is runtime_callbacks
    assert len(calls) == 1
    canvas_call = calls[0]
    assert canvas_call[0] == "canvas"
    assert canvas_call[1]["canvas_interactions_module"] == "canvas-interactions-module"
    assert canvas_call[1]["axis"] == "axis"
    assert canvas_call[1]["peak_selection_callbacks"] == "peak-selection-callbacks"

    bindings_factory_state["value"] = lambda: "geometry-bindings-2"
    set_preview = canvas_call[1]["set_geometry_preview_exclude_mode"]
    toggle_preview = canvas_call[1]["toggle_live_geometry_preview_exclusion_at"]
    assert callable(set_preview)
    assert callable(toggle_preview)
    assert set_preview(True, message="armed") == "set-result"
    assert toggle_preview(12.5, 6.0) == "toggle-result"
    assert manager_events == [
        ("set", "geometry-bindings-2", True, "armed"),
        ("toggle", "geometry-bindings-2", 12.5, 6.0),
    ]


def test_initialize_runtime_canvas_interaction_bindings_connects_canvas_events() -> None:
    events: list[tuple[str, object]] = []
    canvas = SimpleNamespace(
        mpl_connect=lambda event_name, callback: events.append((event_name, callback))
    )
    callbacks = SimpleNamespace(
        on_click="on-click",
        on_press="on-press",
        on_motion="on-motion",
        on_release="on-release",
        on_scroll="on-scroll",
    )

    runtime_geometry_preview.initialize_runtime_canvas_interaction_bindings(
        canvas=canvas,
        callbacks=callbacks,
    )

    assert events == [
        ("button_press_event", "on-click"),
        ("button_press_event", "on-press"),
        ("motion_notify_event", "on-motion"),
        ("button_release_event", "on-release"),
        ("scroll_event", "on-scroll"),
    ]
