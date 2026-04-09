from __future__ import annotations

from types import SimpleNamespace

from ra_sim.gui import runtime_geometry_interaction


def test_build_runtime_peak_selection_workflow_composes_hkl_lookup_controls() -> None:
    calls: list[tuple[str, object]] = []
    callbacks = SimpleNamespace(name="peak-callbacks")
    runtime_obj = SimpleNamespace(
        bindings_factory="peak-bindings",
        callbacks=callbacks,
        maintenance_callbacks="peak-maintenance",
        ensure_peak_overlay_data="peak-overlay-data",
    )
    hkl_lookup_controls_runtime = SimpleNamespace(name="hkl-controls")
    bootstrap_module = SimpleNamespace(
        build_runtime_selected_peak_bootstrap=lambda **kwargs: (
            calls.append(("selected_peak", kwargs)) or runtime_obj
        ),
        build_runtime_hkl_lookup_controls_bootstrap=lambda **kwargs: (
            calls.append(("hkl_lookup", kwargs)) or hkl_lookup_controls_runtime
        ),
    )

    workflow = runtime_geometry_interaction.build_runtime_peak_selection_workflow(
        bootstrap_module=bootstrap_module,
        peak_selection_module="peak-selection-module",
        views_module="views-module",
        hkl_lookup_view_state="hkl-view-state",
        open_bragg_qr_groups="open-bragg-qr-groups",
        simulation_runtime_state="simulation-runtime-state",
        max_distance_px=17.0,
        min_separation_px=5.0,
    )

    assert workflow.runtime is runtime_obj
    assert workflow.bindings_factory == "peak-bindings"
    assert workflow.callbacks is callbacks
    assert workflow.maintenance_callbacks == "peak-maintenance"
    assert workflow.ensure_peak_overlay_data == "peak-overlay-data"
    assert workflow.hkl_lookup_controls_runtime is hkl_lookup_controls_runtime
    assert calls == [
        (
            "selected_peak",
            {
                "peak_selection_module": "peak-selection-module",
                "simulation_runtime_state": "simulation-runtime-state",
                "max_distance_px": 17.0,
                "min_separation_px": 5.0,
            },
        ),
        (
            "hkl_lookup",
            {
                "views_module": "views-module",
                "view_state": "hkl-view-state",
                "peak_selection_callbacks": callbacks,
                "open_bragg_qr_groups": "open-bragg-qr-groups",
            },
        ),
    ]


def test_build_runtime_geometry_manual_projection_workflow_exposes_callbacks() -> None:
    calls: list[tuple[str, object]] = []
    callbacks = SimpleNamespace(
        pick_uses_caked_space="pick-uses-caked-space",
        current_background_image="current-background-image",
        entry_display_coords="entry-display-coords",
        caked_angles_to_background_display_coords="caked-angles",
        background_display_to_native_detector_coords="display-to-native",
        native_detector_coords_to_caked_display_coords="native-to-caked",
        project_peaks_to_current_view="project-peaks",
        simulated_peaks_for_params="simulated-peaks",
        pick_candidates="pick-candidates",
        simulated_lookup="simulated-lookup",
    )
    runtime_obj = SimpleNamespace(callbacks=callbacks)
    bootstrap_module = SimpleNamespace(
        build_runtime_geometry_manual_projection_bootstrap=lambda **kwargs: (
            calls.append(("projection", kwargs)) or runtime_obj
        )
    )

    workflow = (
        runtime_geometry_interaction.build_runtime_geometry_manual_projection_workflow(
            bootstrap_module=bootstrap_module,
            manual_geometry_module="manual-geometry-module",
            display_rotate_k=3,
            pixel_size=1.5e-4,
        )
    )

    assert workflow.runtime is runtime_obj
    assert workflow.callbacks is callbacks
    assert workflow.pick_uses_caked_space == "pick-uses-caked-space"
    assert workflow.current_background_image == "current-background-image"
    assert workflow.entry_display_coords == "entry-display-coords"
    assert workflow.caked_angles_to_background_display_coords == "caked-angles"
    assert workflow.background_display_to_native_detector_coords == "display-to-native"
    assert workflow.native_detector_coords_to_caked_display_coords == "native-to-caked"
    assert workflow.project_peaks_to_current_view == "project-peaks"
    assert workflow.simulated_peaks_for_params == "simulated-peaks"
    assert workflow.pick_candidates == "pick-candidates"
    assert workflow.simulated_lookup == "simulated-lookup"
    assert calls == [
        (
            "projection",
            {
                "manual_geometry_module": "manual-geometry-module",
                "display_rotate_k": 3,
                "pixel_size": 1.5e-4,
            },
        )
    ]


def test_build_runtime_geometry_manual_cache_workflow_exposes_callbacks() -> None:
    calls: list[tuple[str, object]] = []
    callbacks = SimpleNamespace(
        current_match_config="current-match-config",
        pick_cache_signature="pick-cache-signature",
        get_pick_cache="get-pick-cache",
        build_initial_pairs_display="build-initial-pairs-display",
    )
    runtime_obj = SimpleNamespace(callbacks=callbacks)
    bootstrap_module = SimpleNamespace(
        build_runtime_geometry_manual_cache_bootstrap=lambda **kwargs: (
            calls.append(("cache", kwargs)) or runtime_obj
        )
    )

    workflow = runtime_geometry_interaction.build_runtime_geometry_manual_cache_workflow(
        bootstrap_module=bootstrap_module,
        manual_geometry_module="manual-geometry-module",
        fit_config="fit-config",
        pairs_for_index="pairs-for-index",
        simulated_peaks_for_params="manual-simulated-peaks",
    )

    assert workflow.runtime is runtime_obj
    assert workflow.callbacks is callbacks
    assert workflow.current_match_config == "current-match-config"
    assert workflow.pick_cache_signature == "pick-cache-signature"
    assert workflow.get_pick_cache == "get-pick-cache"
    assert workflow.build_initial_pairs_display == "build-initial-pairs-display"
    assert calls == [
        (
            "cache",
            {
                "manual_geometry_module": "manual-geometry-module",
                "fit_config": "fit-config",
                "pairs_for_index": "pairs-for-index",
                "simulated_peaks_for_params": "manual-simulated-peaks",
            },
        )
    ]


def test_build_runtime_geometry_manual_workflow_exposes_callbacks() -> None:
    calls: list[tuple[str, object]] = []
    callbacks = SimpleNamespace(
        render_current_pairs="render-current-pairs",
        toggle_selection_at="toggle-selection-at",
        place_selection_at="place-selection-at",
        update_pick_preview="update-pick-preview",
        cancel_pick_session="cancel-pick-session",
    )
    runtime_obj = SimpleNamespace(callbacks=callbacks)
    bootstrap_module = SimpleNamespace(
        build_runtime_geometry_manual_bootstrap=lambda **kwargs: (
            calls.append(("manual", kwargs)) or runtime_obj
        )
    )

    workflow = runtime_geometry_interaction.build_runtime_geometry_manual_workflow(
        bootstrap_module=bootstrap_module,
        manual_geometry_module="manual-geometry-module",
        build_initial_pairs_display="build-initial-pairs-display",
        pick_search_window_px=42.0,
    )

    assert workflow.runtime is runtime_obj
    assert workflow.callbacks is callbacks
    assert workflow.render_current_pairs == "render-current-pairs"
    assert workflow.toggle_selection_at == "toggle-selection-at"
    assert workflow.place_selection_at == "place-selection-at"
    assert workflow.update_pick_preview == "update-pick-preview"
    assert workflow.cancel_pick_session == "cancel-pick-session"
    assert calls == [
        (
            "manual",
            {
                "manual_geometry_module": "manual-geometry-module",
                "build_initial_pairs_display": "build-initial-pairs-display",
                "pick_search_window_px": 42.0,
            },
        )
    ]


def test_build_runtime_geometry_tool_action_workflow_exposes_callbacks() -> None:
    calls: list[tuple[str, object]] = []
    callbacks = SimpleNamespace(
        update_fit_history_button_state="update-fit-history-button-state",
        update_manual_pick_button_label="update-manual-pick-button-label",
        set_manual_pick_mode="set-manual-pick-mode",
        toggle_manual_pick_mode="toggle-manual-pick-mode",
        clear_current_manual_pairs="clear-current-manual-pairs",
    )
    runtime_obj = SimpleNamespace(callbacks=callbacks)
    bootstrap_module = SimpleNamespace(
        build_runtime_geometry_tool_action_callbacks_bootstrap=lambda **kwargs: (
            calls.append(("tool_action", kwargs)) or runtime_obj
        )
    )

    workflow = runtime_geometry_interaction.build_runtime_geometry_tool_action_workflow(
        bootstrap_module=bootstrap_module,
        geometry_fit_module="geometry-fit-module",
        set_hkl_pick_mode="set-hkl-pick-mode",
        refresh_status="refresh-status",
    )

    assert workflow.runtime is runtime_obj
    assert workflow.callbacks is callbacks
    assert workflow.update_fit_history_button_state == (
        "update-fit-history-button-state"
    )
    assert workflow.update_manual_pick_button_label == (
        "update-manual-pick-button-label"
    )
    assert workflow.set_manual_pick_mode == "set-manual-pick-mode"
    assert workflow.toggle_manual_pick_mode == "toggle-manual-pick-mode"
    assert workflow.clear_current_manual_pairs == "clear-current-manual-pairs"
    assert calls == [
        (
            "tool_action",
            {
                "geometry_fit_module": "geometry-fit-module",
                "set_hkl_pick_mode": "set-hkl-pick-mode",
                "refresh_status": "refresh-status",
            },
        )
    ]


def test_build_runtime_geometry_tool_action_controls_runtime_delegates_to_bootstrap() -> None:
    calls: list[tuple[str, object]] = []
    runtime_obj = SimpleNamespace(name="geometry-tool-controls")
    bootstrap_module = SimpleNamespace(
        build_runtime_geometry_tool_action_controls_bootstrap=lambda **kwargs: (
            calls.append(("controls", kwargs)) or runtime_obj
        )
    )

    result = (
        runtime_geometry_interaction.build_runtime_geometry_tool_action_controls_runtime(
            bootstrap_module=bootstrap_module,
            views_module="views-module",
            view_state="geometry-tool-view-state",
            on_toggle_manual_pick="toggle-pick",
        )
    )

    assert result is runtime_obj
    assert calls == [
        (
            "controls",
            {
                "views_module": "views-module",
                "view_state": "geometry-tool-view-state",
                "on_toggle_manual_pick": "toggle-pick",
            },
        )
    ]


def test_initialize_runtime_geometry_interaction_controls_creates_controls_and_refreshes_labels() -> None:
    events: list[tuple[str, object | None]] = []
    geometry_tool_actions_runtime = SimpleNamespace(
        create_controls=lambda parent: events.append(("geometry", parent))
    )
    hkl_lookup_controls_runtime = SimpleNamespace(
        create_controls=lambda parent: events.append(("hkl", parent))
    )

    runtime_geometry_interaction.initialize_runtime_geometry_interaction_controls(
        fit_actions_parent="fit-actions-parent",
        geometry_tool_actions_runtime=geometry_tool_actions_runtime,
        hkl_lookup_controls_runtime=hkl_lookup_controls_runtime,
        update_fit_history_button_state=lambda: events.append(("fit-history", None)),
        update_manual_pick_button_label=lambda: events.append(("manual-pick", None)),
        update_preview_exclude_button_label=lambda: events.append(("preview-exclude", None)),
    )

    assert events == [
        ("geometry", "fit-actions-parent"),
        ("fit-history", None),
        ("manual-pick", None),
        ("preview-exclude", None),
        ("hkl", "fit-actions-parent"),
    ]


def test_refresh_runtime_peak_selection_after_update_uses_maintenance_bundle() -> None:
    calls: list[bool] = []
    maintenance_callbacks = SimpleNamespace(
        refresh_after_simulation_update=lambda enabled: (
            calls.append(bool(enabled)) or "overlay-ready"
        )
    )

    result = runtime_geometry_interaction.refresh_runtime_peak_selection_after_update(
        maintenance_callbacks=maintenance_callbacks,
        live_geometry_preview_enabled=1,
    )

    assert result is True
    assert calls == [True]


def test_apply_restored_runtime_selected_hkl_target_uses_maintenance_bundle() -> None:
    calls: list[object] = []
    maintenance_callbacks = SimpleNamespace(
        apply_restored_selected_hkl_target=lambda target: (
            calls.append(target) or (1, 2, 3)
        )
    )

    result = runtime_geometry_interaction.apply_restored_runtime_selected_hkl_target(
        maintenance_callbacks=maintenance_callbacks,
        selected_hkl_target=[1, 2, 3],
    )

    assert result == (1, 2, 3)
    assert calls == [[1, 2, 3]]
