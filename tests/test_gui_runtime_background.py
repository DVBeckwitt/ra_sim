from __future__ import annotations

from types import SimpleNamespace

from ra_sim.gui import runtime_background


def test_build_runtime_background_status_refreshers_prefers_controls_then_callbacks() -> None:
    refreshers = runtime_background.build_runtime_background_status_refreshers(
        background_controls_runtime_factory=lambda: SimpleNamespace(
            refresh_status=lambda: "controls-status",
            refresh_backend_status=lambda: "controls-backend",
        ),
        background_runtime_callbacks_factory=lambda: SimpleNamespace(
            refresh_status=lambda: "callbacks-status",
            refresh_backend_status=lambda: "callbacks-backend",
        ),
    )

    assert refreshers.refresh_status() == "controls-status"
    assert refreshers.refresh_backend_status() == "controls-backend"


def test_build_runtime_background_status_refreshers_falls_back_to_callbacks() -> None:
    refreshers = runtime_background.build_runtime_background_status_refreshers(
        background_controls_runtime_factory=lambda: object(),
        background_runtime_callbacks_factory=lambda: SimpleNamespace(
            refresh_status=lambda: "callbacks-status",
            refresh_backend_status=lambda: "callbacks-backend",
        ),
    )

    assert refreshers.refresh_status() == "callbacks-status"
    assert refreshers.refresh_backend_status() == "callbacks-backend"


def test_build_runtime_background_theta_workflow_exposes_bootstrap_callbacks() -> None:
    calls: list[tuple[str, object]] = []
    callbacks = SimpleNamespace(
        current_geometry_fit_background_indices=lambda strict=False: [0, 2],
        geometry_fit_uses_shared_theta_offset=lambda: True,
        current_geometry_theta_offset=lambda strict=False: 0.25,
        current_background_theta_values=lambda strict_count=False: [4.0, 7.5],
        background_theta_for_index=lambda idx, strict_count=False: 4.0 + idx,
        sync_background_theta_controls=lambda **kwargs: None,
        apply_background_theta_metadata=lambda **kwargs: True,
        apply_geometry_fit_background_selection=lambda **kwargs: True,
        sync_geometry_fit_background_selection=lambda **kwargs: None,
    )
    runtime_obj = SimpleNamespace(bindings_factory="bindings", callbacks=callbacks)

    bootstrap_module = SimpleNamespace(
        build_runtime_background_theta_bootstrap=lambda **kwargs: (
            calls.append(("theta", kwargs)) or runtime_obj
        )
    )

    workflow = runtime_background.build_runtime_background_theta_workflow(
        bootstrap_module=bootstrap_module,
        background_theta_module="background-theta-module",
        osc_files_factory="osc-factory",
        schedule_update_factory="schedule-update-factory",
    )

    assert workflow.runtime is runtime_obj
    assert workflow.callbacks is callbacks
    assert workflow.current_geometry_fit_background_indices is (
        callbacks.current_geometry_fit_background_indices
    )
    assert workflow.geometry_fit_uses_shared_theta_offset is (
        callbacks.geometry_fit_uses_shared_theta_offset
    )
    assert workflow.current_geometry_theta_offset is callbacks.current_geometry_theta_offset
    assert workflow.current_background_theta_values is callbacks.current_background_theta_values
    assert workflow.background_theta_for_index is callbacks.background_theta_for_index
    assert workflow.sync_background_theta_controls is callbacks.sync_background_theta_controls
    assert workflow.apply_background_theta_metadata is callbacks.apply_background_theta_metadata
    assert workflow.apply_geometry_fit_background_selection is (
        callbacks.apply_geometry_fit_background_selection
    )
    assert workflow.sync_geometry_fit_background_selection is (
        callbacks.sync_geometry_fit_background_selection
    )
    assert calls == [
        (
            "theta",
            {
                "background_theta_module": "background-theta-module",
                "osc_files_factory": "osc-factory",
                "schedule_update_factory": "schedule-update-factory",
            },
        )
    ]


def test_build_runtime_background_workflow_composes_runtime_and_controls() -> None:
    calls: list[tuple[str, object]] = []
    runtime_callbacks = SimpleNamespace(
        refresh_status=lambda: "status",
        switch_background=lambda: "switched",
    )
    runtime_obj = SimpleNamespace(
        bindings_factory="background-bindings",
        callbacks=runtime_callbacks,
    )
    controls_runtime = SimpleNamespace(
        toggle_visibility=lambda: True,
        refresh_status=lambda: "controls-status",
        refresh_backend_status=lambda: "controls-backend",
    )

    bootstrap_module = SimpleNamespace(
        build_runtime_background_bootstrap=lambda **kwargs: (
            calls.append(("background", kwargs)) or runtime_obj
        ),
        build_runtime_background_controls_bootstrap=lambda **kwargs: (
            calls.append(("controls", kwargs)) or controls_runtime
        ),
    )

    workflow = runtime_background.build_runtime_background_workflow(
        bootstrap_module=bootstrap_module,
        background_manager_module="background-manager-module",
        views_module="views-module",
        workspace_view_state="workspace-view-state",
        background_backend_debug_view_state="backend-view-state",
        background_state="background-state",
        image_size=2048,
        schedule_update_factory="schedule-update-factory",
    )

    assert workflow.runtime is runtime_obj
    assert workflow.bindings_factory == "background-bindings"
    assert workflow.callbacks is runtime_callbacks
    assert workflow.controls_runtime is controls_runtime
    assert workflow.toggle_visibility() is True
    assert workflow.switch_background() == "switched"
    assert calls == [
        (
            "background",
            {
                "background_manager_module": "background-manager-module",
                "view_state": "workspace-view-state",
                "background_state": "background-state",
                "image_size": 2048,
                "schedule_update_factory": "schedule-update-factory",
            },
        ),
        (
            "controls",
            {
                "views_module": "views-module",
                "workspace_view_state": "workspace-view-state",
                "background_backend_debug_view_state": "backend-view-state",
                "background_callbacks": runtime_callbacks,
            },
        ),
    ]
