import numpy as np

from ra_sim.gui import controllers, state


def test_app_state_has_isolated_manual_geometry_state() -> None:
    app_state = state.AppState()
    other_state = state.AppState()

    assert isinstance(app_state.manual_geometry, state.ManualGeometryState)
    assert isinstance(app_state.geometry_fit_history, state.GeometryFitHistoryState)
    assert isinstance(app_state.background_runtime, state.BackgroundRuntimeState)
    assert isinstance(app_state.peak_selection, state.PeakSelectionState)
    assert isinstance(app_state.atom_site_override, state.AtomSiteOverrideState)
    assert isinstance(app_state.geometry_runtime, state.GeometryRuntimeState)
    assert isinstance(app_state.simulation_runtime, state.SimulationRuntimeState)
    assert isinstance(
        app_state.geometry_fit_parameter_controls_view,
        state.GeometryFitParameterControlsViewState,
    )
    assert isinstance(
        app_state.geometry_fit_constraints_view,
        state.GeometryFitConstraintsViewState,
    )
    assert isinstance(
        app_state.background_theta_controls_view,
        state.BackgroundThetaControlsViewState,
    )
    assert isinstance(app_state.workspace_panels_view, state.WorkspacePanelsViewState)
    assert isinstance(
        app_state.background_backend_debug_view,
        state.BackgroundBackendDebugViewState,
    )
    assert isinstance(
        app_state.primary_cif_controls_view,
        state.PrimaryCifControlsViewState,
    )
    assert isinstance(
        app_state.cif_weight_controls_view,
        state.CifWeightControlsViewState,
    )
    assert isinstance(
        app_state.display_controls_state,
        state.DisplayControlsState,
    )
    assert isinstance(
        app_state.display_controls_view,
        state.DisplayControlsViewState,
    )
    assert isinstance(
        app_state.structure_factor_pruning_controls_view,
        state.StructureFactorPruningControlsViewState,
    )
    assert isinstance(
        app_state.beam_mosaic_parameter_sliders_view,
        state.BeamMosaicParameterSlidersViewState,
    )
    assert isinstance(
        app_state.sampling_optics_controls_view,
        state.SamplingOpticsControlsViewState,
    )
    assert isinstance(
        app_state.finite_stack_controls_view,
        state.FiniteStackControlsViewState,
    )
    assert isinstance(
        app_state.stacking_parameter_controls_view,
        state.StackingParameterControlsViewState,
    )
    assert isinstance(
        app_state.geometry_tool_actions_view,
        state.GeometryToolActionsViewState,
    )
    assert isinstance(
        app_state.hkl_lookup_view,
        state.HklLookupViewState,
    )
    assert isinstance(
        app_state.geometry_overlay_actions_view,
        state.GeometryOverlayActionsViewState,
    )
    assert isinstance(
        app_state.analysis_view_controls_view,
        state.AnalysisViewControlsViewState,
    )
    assert isinstance(
        app_state.analysis_export_controls_view,
        state.AnalysisExportControlsViewState,
    )
    assert isinstance(
        app_state.integration_range_controls_view,
        state.IntegrationRangeControlsViewState,
    )
    assert isinstance(app_state.geometry_preview, state.GeometryPreviewState)
    assert isinstance(
        app_state.geometry_preview.overlay,
        state.GeometryPreviewOverlayState,
    )
    assert isinstance(app_state.geometry_q_groups, state.GeometryQGroupState)
    assert isinstance(app_state.geometry_q_group_view, state.GeometryQGroupViewState)
    assert isinstance(app_state.bragg_qr_manager, state.BraggQrManagerState)
    assert isinstance(app_state.bragg_qr_manager_view, state.BraggQrManagerViewState)
    assert isinstance(
        app_state.hbn_geometry_debug_view,
        state.HbnGeometryDebugViewState,
    )
    assert isinstance(app_state.app_shell_view, state.AppShellViewState)
    assert isinstance(app_state.status_panel_view, state.StatusPanelViewState)
    assert app_state.manual_geometry is not other_state.manual_geometry
    assert app_state.geometry_fit_history is not other_state.geometry_fit_history
    assert app_state.background_runtime is not other_state.background_runtime
    assert app_state.peak_selection is not other_state.peak_selection
    assert app_state.atom_site_override is not other_state.atom_site_override
    assert app_state.geometry_runtime is not other_state.geometry_runtime
    assert app_state.simulation_runtime is not other_state.simulation_runtime
    assert (
        app_state.geometry_fit_parameter_controls_view
        is not other_state.geometry_fit_parameter_controls_view
    )
    assert (
        app_state.geometry_fit_constraints_view
        is not other_state.geometry_fit_constraints_view
    )
    assert app_state.background_theta_controls_view is not other_state.background_theta_controls_view
    assert app_state.workspace_panels_view is not other_state.workspace_panels_view
    assert app_state.background_backend_debug_view is not other_state.background_backend_debug_view
    assert app_state.primary_cif_controls_view is not other_state.primary_cif_controls_view
    assert app_state.cif_weight_controls_view is not other_state.cif_weight_controls_view
    assert app_state.display_controls_state is not other_state.display_controls_state
    assert app_state.display_controls_view is not other_state.display_controls_view
    assert (
        app_state.structure_factor_pruning_controls_view
        is not other_state.structure_factor_pruning_controls_view
    )
    assert (
        app_state.beam_mosaic_parameter_sliders_view
        is not other_state.beam_mosaic_parameter_sliders_view
    )
    assert (
        app_state.sampling_optics_controls_view
        is not other_state.sampling_optics_controls_view
    )
    assert (
        app_state.finite_stack_controls_view
        is not other_state.finite_stack_controls_view
    )
    assert (
        app_state.stacking_parameter_controls_view
        is not other_state.stacking_parameter_controls_view
    )
    assert app_state.geometry_tool_actions_view is not other_state.geometry_tool_actions_view
    assert app_state.hkl_lookup_view is not other_state.hkl_lookup_view
    assert (
        app_state.geometry_overlay_actions_view
        is not other_state.geometry_overlay_actions_view
    )
    assert (
        app_state.analysis_view_controls_view
        is not other_state.analysis_view_controls_view
    )
    assert (
        app_state.analysis_export_controls_view
        is not other_state.analysis_export_controls_view
    )
    assert (
        app_state.integration_range_controls_view
        is not other_state.integration_range_controls_view
    )
    assert app_state.geometry_preview is not other_state.geometry_preview
    assert app_state.geometry_preview.overlay is not other_state.geometry_preview.overlay
    assert app_state.geometry_q_groups is not other_state.geometry_q_groups
    assert app_state.geometry_q_group_view is not other_state.geometry_q_group_view
    assert app_state.bragg_qr_manager is not other_state.bragg_qr_manager
    assert app_state.bragg_qr_manager_view is not other_state.bragg_qr_manager_view
    assert app_state.hbn_geometry_debug_view is not other_state.hbn_geometry_debug_view
    assert app_state.app_shell_view is not other_state.app_shell_view
    assert app_state.status_panel_view is not other_state.status_panel_view

    app_state.background_runtime.current_background_index = 3
    app_state.background_runtime.visible = False
    app_state.background_runtime.backend_flip_x = True
    app_state.peak_selection.selected_hkl_target = (1, 2, 3)
    app_state.peak_selection.hkl_pick_armed = True
    app_state.atom_site_override.source_path = "example.cif"
    app_state.geometry_runtime.manual_pick_armed = True
    app_state.simulation_runtime.last_simulation_signature = ("sig", 1)
    app_state.simulation_runtime.sf_prune_stats["qr_kept"] = 12
    app_state.manual_geometry.pick_session["group_key"] = ("q_group", "primary", 1, 0)
    app_state.geometry_preview.skip_once = True
    app_state.geometry_preview.overlay.pairs.append({"x": 1.0})
    app_state.geometry_q_groups.refresh_requested = True
    app_state.bragg_qr_manager.selected_group_key = ("primary", 1)
    assert other_state.background_runtime.current_background_index == 0
    assert other_state.background_runtime.visible is True
    assert other_state.background_runtime.backend_flip_x is False
    assert other_state.peak_selection.selected_hkl_target is None
    assert other_state.peak_selection.hkl_pick_armed is False
    assert other_state.atom_site_override.source_path is None
    assert other_state.geometry_runtime.manual_pick_armed is False
    assert other_state.simulation_runtime.last_simulation_signature is None
    assert other_state.simulation_runtime.sf_prune_stats["qr_kept"] == 0
    assert other_state.manual_geometry.pick_session == {}
    assert other_state.geometry_preview.skip_once is False
    assert other_state.geometry_preview.overlay.pairs == []
    assert other_state.geometry_q_groups.refresh_requested is False
    assert other_state.bragg_qr_manager.selected_group_key is None


def test_build_initial_state_reads_instrument_config(monkeypatch) -> None:
    monkeypatch.setattr(
        controllers,
        "get_instrument_config",
        lambda: {"instrument": {"detector": {"image_size": 4096}}},
    )

    app_state = controllers.build_initial_state()

    assert app_state.instrument_config == {
        "instrument": {"detector": {"image_size": 4096}}
    }
    assert app_state.image_size == 4096


def test_display_control_controller_helpers_normalize_scale_and_ranges() -> None:
    assert controllers.ensure_display_intensity_range(float("nan"), 5.0) == (0.0, 5.0)
    assert controllers.ensure_display_intensity_range(5.0, 4.0) == (5.0, 6.0)
    assert controllers.normalize_display_scale_factor("2.5", fallback=1.0) == 2.5
    assert controllers.normalize_display_scale_factor("-3", fallback=1.0) == 0.0
    assert controllers.normalize_display_scale_factor("bad", fallback="4.5") == 4.5


def test_structure_factor_pruning_controller_helpers_clip_and_normalize_inputs() -> None:
    assert controllers.clip_structure_factor_prune_bias("1.5", fallback=0.0, minimum=-2.0, maximum=2.0) == 1.5
    assert controllers.clip_structure_factor_prune_bias("bad", fallback=0.25, minimum=-2.0, maximum=2.0) == 0.25
    assert controllers.clip_solve_q_steps("12.6", fallback=8, minimum=4, maximum=20) == 13
    assert controllers.clip_solve_q_steps("bad", fallback=8, minimum=4, maximum=20) == 8
    assert controllers.clip_solve_q_rel_tol("1e-4", fallback=1e-3, minimum=1e-6, maximum=1e-2) == 1e-4
    assert controllers.clip_solve_q_rel_tol("bad", fallback=1e-3, minimum=1e-6, maximum=1e-2) == 1e-3
    assert controllers.normalize_solve_q_mode_label("fast") == "uniform"
    assert controllers.normalize_solve_q_mode_label("robust") == "adaptive"
    assert controllers.solve_q_mode_flag_from_label("uniform", uniform_flag=7, adaptive_flag=9) == 7
    assert controllers.solve_q_mode_flag_from_label("adaptive", uniform_flag=7, adaptive_flag=9) == 9


def test_beam_mosaic_slider_controller_helper_clamps_to_bounds() -> None:
    assert controllers.clamp_slider_value_to_bounds(2.5, lower_bound=0.0, upper_bound=2.0, fallback=1.0) == 2.0
    assert controllers.clamp_slider_value_to_bounds(-1.0, lower_bound=0.0, upper_bound=2.0, fallback=1.0) == 0.0
    assert controllers.clamp_slider_value_to_bounds("bad", lower_bound=0.0, upper_bound=2.0, fallback=1.25) == 1.25
    assert controllers.clamp_slider_value_to_bounds(1.0, lower_bound=3.0, upper_bound=2.0, fallback=1.0) == 2.0


def test_sampling_resolution_controller_helpers_normalize_parse_and_format() -> None:
    assert controllers.parse_sampling_count("1,250", 10) == 1250
    assert controllers.parse_sampling_count("bad", 10) == 10

    assert controllers.normalize_sampling_resolution_choice(
        "Custom",
        allowed_options=["Low", "High", "Custom"],
        fallback="Low",
    ) == "Custom"
    assert controllers.normalize_sampling_resolution_choice(
        "unexpected",
        allowed_options=["Low", "High", "Custom"],
        fallback="High",
    ) == "High"

    assert controllers.resolve_sampling_count(
        "Custom",
        custom_option="Custom",
        custom_value="3,600",
        preset_counts={"Low": 32, "High": 128},
        fallback_resolution="Low",
        fallback_count=16,
    ) == 3600
    assert controllers.resolve_sampling_count(
        "High",
        custom_option="Custom",
        custom_value="3,600",
        preset_counts={"Low": 32, "High": 128},
        fallback_resolution="Low",
        fallback_count=16,
    ) == 128

    assert controllers.format_sampling_resolution_summary(
        "Custom",
        custom_option="Custom",
        custom_value="3,600",
        preset_counts={"Low": 32, "High": 128},
        fallback_resolution="Low",
        fallback_count=16,
    ) == "3,600 samples (custom)"
    assert controllers.format_sampling_resolution_summary(
        "Low",
        custom_option="Custom",
        custom_value="999",
        preset_counts={"Low": 32, "High": 128},
        fallback_resolution="Low",
        fallback_count=16,
    ) == "32 samples"


def test_finite_stack_controller_helpers_normalize_and_format() -> None:
    assert controllers.normalize_finite_stack_layer_count("72", 10) == 72
    assert controllers.normalize_finite_stack_layer_count("bad", 10) == 10
    assert controllers.format_finite_stack_layer_count(9.8) == "10"

    assert controllers.normalize_finite_stack_phase_delta_expression(
        "pi*L/2",
        fallback="0",
    ) == "pi*L/2"

    assert controllers.normalize_finite_stack_phi_l_divisor(
        "3.5",
        fallback=1.0,
    ) == 3.5
    assert controllers.format_finite_stack_phi_l_divisor(4.0) == "4"


def test_stacking_parameter_controller_helpers_clamp_and_normalize() -> None:
    assert controllers.clamp_site_occupancy_values([1.2, -0.5, "bad"]) == [1.0, 0.0, 1.0]
    assert controllers.clamp_site_occupancy_values(
        [float("nan"), "bad"],
        fallback_values=[0.25, 0.75],
    ) == [0.25, 0.75]
    assert controllers.normalize_stacking_weight_values([20.0, 30.0, 50.0]) == [
        0.2,
        0.3,
        0.5,
    ]
    assert controllers.normalize_stacking_weight_values([0.0, 0.0, 0.0]) == [
        0.0,
        0.0,
        0.0,
    ]


def test_cif_weight_controller_helper_combines_and_renormalizes() -> None:
    combined = controllers.combine_cif_weighted_intensities(
        np.array([2.0, 4.0], dtype=np.float64),
        np.array([3.0, 1.0], dtype=np.float64),
        weight1=0.5,
        weight2=0.25,
    )
    np.testing.assert_allclose(combined, [77.77777778, 100.0], rtol=1e-8, atol=1e-8)

    fallback = controllers.combine_cif_weighted_intensities(
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        weight1="bad",
        weight2=float("nan"),
    )
    np.testing.assert_allclose(fallback, [100.0, 0.0], rtol=1e-8, atol=1e-8)


def test_replace_manual_geometry_state_updates_in_place() -> None:
    manual_state = state.ManualGeometryState()
    pairs_alias = manual_state.pairs_by_background
    session_alias = manual_state.pick_session

    source_pairs = {
        "1": [
            {"label": "1,0,0", "x": 5.0, "y": 6.0},
        ]
    }
    controllers.replace_manual_geometry_pairs_by_background(
        manual_state,
        source_pairs,
    )
    controllers.replace_manual_geometry_pick_session(
        manual_state,
        {"background_index": 1, "group_key": ("q_group", "primary", 1, 0)},
    )

    assert manual_state.pairs_by_background is pairs_alias
    assert manual_state.pick_session is session_alias
    assert pairs_alias == {1: [{"label": "1,0,0", "x": 5.0, "y": 6.0}]}
    assert session_alias["background_index"] == 1

    source_pairs["1"][0]["x"] = 99.0
    assert manual_state.pairs_by_background[1][0]["x"] == 5.0


def test_manual_geometry_undo_controller_restores_latest_snapshot_and_limit() -> None:
    manual_state = state.ManualGeometryState()

    controllers.replace_manual_geometry_pairs_by_background(
        manual_state,
        {0: [{"label": "a", "x": 1.0, "y": 2.0}]},
    )
    controllers.replace_manual_geometry_pick_session(
        manual_state,
        {"group_key": "a"},
    )
    controllers.push_manual_geometry_undo_state(manual_state, limit=2)

    controllers.replace_manual_geometry_pairs_by_background(
        manual_state,
        {1: [{"label": "b", "x": 3.0, "y": 4.0}]},
    )
    controllers.replace_manual_geometry_pick_session(
        manual_state,
        {"group_key": "b"},
    )
    controllers.push_manual_geometry_undo_state(manual_state, limit=2)

    controllers.replace_manual_geometry_pairs_by_background(
        manual_state,
        {2: [{"label": "c", "x": 5.0, "y": 6.0}]},
    )
    controllers.replace_manual_geometry_pick_session(
        manual_state,
        {"group_key": "c"},
    )
    controllers.push_manual_geometry_undo_state(manual_state, limit=2)

    assert len(manual_state.undo_stack) == 2

    controllers.replace_manual_geometry_pairs_by_background(
        manual_state,
        {9: [{"label": "live", "x": 9.0, "y": 10.0}]},
    )
    controllers.replace_manual_geometry_pick_session(
        manual_state,
        {"group_key": "live"},
    )

    restored = controllers.restore_last_manual_geometry_undo_state(manual_state)
    assert restored is not None
    assert manual_state.pick_session["group_key"] == "c"
    assert manual_state.pairs_by_background == {2: [{"label": "c", "x": 5.0, "y": 6.0}]}

    restored = controllers.restore_last_manual_geometry_undo_state(manual_state)
    assert restored is not None
    assert manual_state.pick_session["group_key"] == "b"
    assert manual_state.pairs_by_background == {1: [{"label": "b", "x": 3.0, "y": 4.0}]}

    assert controllers.restore_last_manual_geometry_undo_state(manual_state) is None


def test_clear_manual_geometry_undo_stack_discards_history() -> None:
    manual_state = state.ManualGeometryState()
    controllers.push_manual_geometry_undo_state(manual_state, limit=2)

    assert len(manual_state.undo_stack) == 1
    controllers.clear_manual_geometry_undo_stack(manual_state)
    assert manual_state.undo_stack == []


def test_geometry_fit_history_controller_tracks_overlay_and_undo_redo() -> None:
    fit_state = state.GeometryFitHistoryState()

    controllers.replace_geometry_fit_last_overlay_state(
        fit_state,
        {"overlay_records": [{"x": 1.0}], "max_display_markers": 4},
    )
    assert fit_state.last_overlay_state == {
        "overlay_records": [{"x": 1.0}],
        "max_display_markers": 4,
    }

    controllers.push_geometry_fit_undo_state(
        fit_state,
        {"ui_params": {"gamma": 1.0}},
        copy_state_value=lambda value: dict(value),
        limit=2,
    )
    controllers.push_geometry_fit_undo_state(
        fit_state,
        {"ui_params": {"gamma": 2.0}},
        copy_state_value=lambda value: dict(value),
        limit=2,
    )
    controllers.push_geometry_fit_undo_state(
        fit_state,
        {"ui_params": {"gamma": 3.0}},
        copy_state_value=lambda value: dict(value),
        limit=2,
    )

    assert len(fit_state.undo_stack) == 2
    assert controllers.peek_last_geometry_fit_undo_state(
        fit_state,
        copy_state_value=lambda value: dict(value),
    ) == {"ui_params": {"gamma": 3.0}}

    controllers.commit_geometry_fit_undo(
        fit_state,
        {"ui_params": {"gamma": 9.0}},
        copy_state_value=lambda value: dict(value),
        limit=2,
    )
    assert len(fit_state.undo_stack) == 1
    assert fit_state.redo_stack == [{"ui_params": {"gamma": 9.0}}]

    assert controllers.peek_last_geometry_fit_redo_state(
        fit_state,
        copy_state_value=lambda value: dict(value),
    ) == {"ui_params": {"gamma": 9.0}}

    controllers.commit_geometry_fit_redo(
        fit_state,
        {"ui_params": {"gamma": 5.0}},
        copy_state_value=lambda value: dict(value),
        limit=2,
    )
    assert fit_state.redo_stack == []
    assert fit_state.undo_stack[-1] == {"ui_params": {"gamma": 5.0}}

    controllers.clear_geometry_fit_history(fit_state)
    assert fit_state.undo_stack == []
    assert fit_state.redo_stack == []
    assert fit_state.last_overlay_state is None


def test_geometry_preview_controller_tracks_exclusions_skip_flag_and_cache() -> None:
    preview_state = state.GeometryPreviewState()
    excluded_keys_alias = preview_state.excluded_keys
    excluded_alias = preview_state.excluded_q_groups

    controllers.replace_geometry_preview_excluded_q_groups(
        preview_state,
        [
            ("q_group", "primary", 1, 0),
            ["q_group", "secondary", 2, 1],
        ],
    )
    assert preview_state.excluded_q_groups is excluded_alias
    assert preview_state.excluded_q_groups == {
        ("q_group", "primary", 1, 0),
        ("q_group", "secondary", 2, 1),
    }

    controllers.retain_geometry_preview_excluded_q_groups(
        preview_state,
        {("q_group", "secondary", 2, 1)},
    )
    assert preview_state.excluded_q_groups == {("q_group", "secondary", 2, 1)}
    assert controllers.count_geometry_preview_excluded_q_groups(preview_state) == 1
    assert (
        controllers.count_geometry_preview_excluded_q_groups(
            preview_state,
            [
                ("q_group", "secondary", 2, 1),
                ("q_group", "primary", 1, 0),
            ],
        )
        == 1
    )

    controllers.set_geometry_preview_q_group_included(
        preview_state,
        ("q_group", "secondary", 2, 1),
        included=True,
    )
    controllers.set_geometry_preview_q_group_included(
        preview_state,
        ("q_group", "primary", 1, 0),
        included=False,
    )
    assert preview_state.excluded_q_groups == {("q_group", "primary", 1, 0)}
    assert preview_state.excluded_keys is excluded_keys_alias

    controllers.set_geometry_preview_match_included(
        preview_state,
        ("hkl", 1, 0, 0),
        included=False,
    )
    assert preview_state.excluded_keys == {("hkl", 1, 0, 0)}
    controllers.set_geometry_preview_match_included(
        preview_state,
        ("hkl", 1, 0, 0),
        included=True,
    )
    assert preview_state.excluded_keys == set()
    controllers.set_geometry_preview_exclude_mode(preview_state, True)
    assert preview_state.exclude_armed is True
    controllers.set_geometry_preview_exclude_mode(preview_state, False)
    assert preview_state.exclude_armed is False

    overlay_alias = preview_state.overlay
    pairs_alias = preview_state.overlay.pairs
    attempts_alias = preview_state.overlay.auto_match_attempts
    controllers.replace_geometry_preview_overlay_state(
        preview_state,
        {
            "signature": ("sig", 1),
            "pairs": [{"x": 1.0, "y": 2.0}],
            "simulated_count": 8,
            "min_matches": 6,
            "best_radius": 12.5,
            "mean_dist": 3.0,
            "p90_dist": 4.5,
            "quality_fail": True,
            "max_display_markers": 5,
            "auto_match_attempts": [{"name": "attempt-a"}],
            "q_group_total": 10,
            "q_group_excluded": 2,
            "excluded_q_peaks": 4,
            "collapsed_degenerate_peaks": 1,
        },
    )
    assert preview_state.overlay is overlay_alias
    assert preview_state.overlay.pairs is pairs_alias
    assert preview_state.overlay.auto_match_attempts is attempts_alias
    assert preview_state.overlay.signature == ("sig", 1)
    assert preview_state.overlay.pairs == [{"x": 1.0, "y": 2.0}]
    assert preview_state.overlay.simulated_count == 8
    assert preview_state.overlay.min_matches == 6
    assert preview_state.overlay.best_radius == 12.5
    assert preview_state.overlay.quality_fail is True
    assert preview_state.overlay.max_display_markers == 5
    assert preview_state.overlay.auto_match_attempts == [{"name": "attempt-a"}]
    assert preview_state.overlay.q_group_total == 10
    assert preview_state.overlay.q_group_excluded == 2
    assert preview_state.overlay.excluded_q_peaks == 4
    assert preview_state.overlay.collapsed_degenerate_peaks == 1

    controllers.request_geometry_preview_skip_once(preview_state)
    assert preview_state.skip_once is True
    assert controllers.consume_geometry_preview_skip_once(preview_state) is True
    assert preview_state.skip_once is False
    assert controllers.consume_geometry_preview_skip_once(preview_state) is False
    controllers.request_geometry_preview_skip_once(preview_state)
    controllers.clear_geometry_preview_skip_once(preview_state)
    assert preview_state.skip_once is False

    cache_key = ("bg", 1, 2, 3)
    cache_data = {"summit_records": [1, 2, 3]}
    controllers.replace_geometry_auto_match_background_cache(
        preview_state,
        cache_key,
        cache_data,
    )
    assert (
        controllers.get_geometry_auto_match_background_cache(preview_state, cache_key)
        is cache_data
    )
    assert (
        controllers.get_geometry_auto_match_background_cache(
            preview_state,
            ("bg", 9),
        )
        is None
    )
    controllers.clear_geometry_auto_match_background_cache(preview_state)
    assert preview_state.auto_match_background_cache_key is None
    assert preview_state.auto_match_background_cache_data is None
    controllers.clear_geometry_preview_excluded_keys(preview_state)
    assert preview_state.excluded_keys == set()
    controllers.clear_geometry_preview_excluded_q_groups(preview_state)
    assert preview_state.excluded_q_groups == set()


def test_geometry_q_group_controller_replaces_entries_row_vars_and_refresh_flag() -> None:
    q_state = state.GeometryQGroupState()
    row_vars_alias = q_state.row_vars
    cached_alias = q_state.cached_entries

    listed = controllers.replace_geometry_q_group_cached_entries(
        q_state,
        [
            {
                "key": ("q_group", "primary", 1, 0),
                "hkl_preview": [(1, 0, 0)],
                "peak_count": 2,
            }
        ],
    )

    assert q_state.row_vars is row_vars_alias
    assert q_state.cached_entries is cached_alias
    assert listed == [
        {
            "key": ("q_group", "primary", 1, 0),
            "hkl_preview": [(1, 0, 0)],
            "peak_count": 2,
        }
    ]
    assert controllers.listed_geometry_q_group_keys(q_state) == {
        ("q_group", "primary", 1, 0)
    }

    controllers.set_geometry_q_group_row_var(
        q_state,
        ("q_group", "primary", 1, 0),
        "var-a",
    )
    assert q_state.row_vars == {("q_group", "primary", 1, 0): "var-a"}
    controllers.clear_geometry_q_group_row_vars(q_state)
    assert q_state.row_vars == {}

    controllers.request_geometry_q_group_refresh(q_state)
    assert q_state.refresh_requested is True
    assert controllers.consume_geometry_q_group_refresh_request(q_state) is True
    assert q_state.refresh_requested is False
    assert controllers.consume_geometry_q_group_refresh_request(q_state) is False


def test_bragg_qr_manager_controller_tracks_indices_selection_and_mutations() -> None:
    bragg_state = state.BraggQrManagerState()
    qr_alias = bragg_state.qr_index_keys
    l_alias = bragg_state.l_index_keys

    assert controllers.replace_bragg_qr_index_keys(
        bragg_state,
        [("primary", 1), ["secondary", 2], "bad"],
    ) == [("primary", 1), ("secondary", 2)]
    assert bragg_state.qr_index_keys is qr_alias
    assert controllers.replace_bragg_qr_l_index_keys(
        bragg_state,
        [0, "3", None],
    ) == [0, 3]
    assert bragg_state.l_index_keys is l_alias

    assert controllers.selected_bragg_qr_keys(bragg_state, [1, 9]) == [
        ("secondary", 2)
    ]
    assert controllers.selected_bragg_qr_l_keys(bragg_state, [0, 1, 7]) == [0, 3]
    assert controllers.set_bragg_qr_selected_group_key(
        bragg_state,
        ["primary", "4"],
    ) == ("primary", 4)
    assert bragg_state.selected_group_key == ("primary", 4)

    disabled_groups: set[tuple[str, int]] = set()
    assert (
        controllers.set_bragg_qr_groups_enabled(
            disabled_groups,
            [("primary", 1), ("secondary", 2)],
            enabled=False,
        )
        == 2
    )
    assert disabled_groups == {("primary", 1), ("secondary", 2)}
    assert controllers.toggle_bragg_qr_groups(disabled_groups, [("primary", 1)]) == 1
    assert disabled_groups == {("secondary", 2)}

    disabled_l_values: set[tuple[str, int, int]] = set()
    assert (
        controllers.set_bragg_qr_l_values_enabled(
            disabled_l_values,
            ("primary", 4),
            [0, 2, -99],
            enabled=False,
            invalid_key=-99,
        )
        == 2
    )
    assert disabled_l_values == {("primary", 4, 0), ("primary", 4, 2)}
    assert (
        controllers.toggle_bragg_qr_l_values(
            disabled_l_values,
            ("primary", 4),
            [2, 5],
            invalid_key=-99,
        )
        == 2
    )
    assert disabled_l_values == {("primary", 4, 0), ("primary", 4, 5)}
    assert (
        controllers.clear_bragg_qr_l_values_for_group(
            disabled_l_values,
            ("primary", 4),
        )
        is True
    )
    assert disabled_l_values == set()

    controllers.clear_bragg_qr_manager_state(bragg_state)
    assert bragg_state.qr_index_keys == []
    assert bragg_state.l_index_keys == []
    assert bragg_state.selected_group_key is None


def test_bragg_qr_filter_controller_applies_group_and_l_disables() -> None:
    sim_state = state.SimulationRuntimeState(
        sim_primary_qr_all={
            1: {
                "L": np.array([0.0, 1.0, 2.0]),
                "I": np.array([5.0, 4.0, 3.0]),
                "hk": (1, 0),
                "deg": 1,
            },
            3: {
                "L": np.array([0.0]),
                "I": np.array([2.5]),
                "hk": (1, 1),
                "deg": 1,
            },
        },
        sim_miller1_all=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        sim_intens1_all=np.array([10.0, 5.0, 1.0, 8.0]),
        sim_miller2_all=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        ),
        sim_intens2_all=np.array([7.0, 6.0]),
    )
    bragg_state = state.BraggQrManagerState(
        disabled_groups={("primary", 3), ("primary", 99)},
        disabled_l_values={
            ("primary", 1, controllers.bragg_qr_l_value_to_key(1.0)),
            ("primary", 7, controllers.bragg_qr_l_value_to_key(0.0)),
        },
    )

    stats = controllers.apply_bragg_qr_filters(
        sim_state,
        bragg_state,
        prune_bias=0.0,
    )

    assert bragg_state.disabled_groups == {("primary", 3)}
    assert bragg_state.disabled_l_values == {
        ("primary", 1, controllers.bragg_qr_l_value_to_key(1.0))
    }
    assert set(sim_state.sim_primary_qr) == {1}
    assert np.array_equal(sim_state.sim_primary_qr[1]["L"], np.array([0.0, 2.0]))
    assert np.array_equal(sim_state.sim_primary_qr[1]["I"], np.array([5.0, 3.0]))
    assert np.array_equal(
        sim_state.sim_miller1,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
            ]
        ),
    )
    assert np.array_equal(sim_state.sim_intens1, np.array([10.0, 1.0]))
    assert np.array_equal(sim_state.sim_miller2, sim_state.sim_miller2_all)
    assert np.array_equal(sim_state.sim_intens2, sim_state.sim_intens2_all)
    assert stats == {
        "qr_total": 2,
        "qr_kept": 2,
        "hkl_primary_total": 2,
        "hkl_primary_kept": 2,
        "hkl_secondary_total": 2,
        "hkl_secondary_kept": 2,
    }
    assert sim_state.sf_prune_stats == stats


def test_bragg_qr_controller_helpers_format_status_and_keys() -> None:
    assert controllers.normalize_bragg_qr_source_label("SECONDARY") == "secondary"
    assert controllers.normalize_bragg_qr_source_label("anything else") == "primary"
    assert controllers.bragg_qr_l_value_to_key(1.25) == 1250000
    assert controllers.bragg_qr_l_key_to_value(1250000) == 1.25
    assert np.array_equal(
        controllers.bragg_qr_l_keys_from_l_array(np.array([0.0, np.nan, 1.5])),
        np.array(
            [
                0,
                controllers.BRAGG_QR_L_INVALID_KEY,
                1500000,
            ],
            dtype=np.int64,
        ),
    )
    assert np.array_equal(
        controllers.bragg_qr_m_indices_from_miller_array(
            np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        ),
        np.array([1, 3], dtype=np.int64),
    )
    assert (
        controllers.format_structure_factor_pruning_status(
            {
                "qr_total": 10,
                "qr_kept": 4,
                "hkl_primary_total": 0,
                "hkl_primary_kept": 0,
            },
            prune_bias=0.5,
        )
        == "SF pruning keeps 4/10 rod points (40.0%), bias=+0.50"
    )


def test_bragg_qr_manager_entry_and_l_value_helpers_build_expected_data() -> None:
    sim_state = state.SimulationRuntimeState(
        sim_primary_qr_all={
            2: {
                "L": np.array([0.25, 0.75]),
                "I": np.array([4.0, 2.0]),
                "hk": (1, 0),
                "deg": 1,
            }
        },
        sim_miller1_all=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.5],
            ]
        ),
        sim_miller2_all=np.array(
            [
                [1.0, 0.0, 0.25],
                [1.0, 0.0, 0.5],
            ]
        ),
    )

    entries = controllers.build_bragg_qr_entries(
        sim_state,
        primary_a=3.0,
        secondary_a=4.0,
    )

    assert [entry["key"] for entry in entries] == [
        ("primary", 1),
        ("primary", 2),
        ("primary", 3),
        ("secondary", 1),
    ]
    assert entries[1]["hk_preview"] == "n/a"
    assert entries[0]["hk_preview"] == "(1 0)"
    assert entries[2]["hk_preview"] == "(1 1)"
    assert entries[0]["qr"] == controllers.qr_value_for_m(1, 3.0)
    assert entries[3]["qr"] == controllers.qr_value_for_m(1, 4.0)

    assert controllers.build_bragg_qr_l_value_map(sim_state, "primary", 2) == {
        controllers.bragg_qr_l_value_to_key(0.25): 0.25,
        controllers.bragg_qr_l_value_to_key(0.75): 0.75,
    }
    assert controllers.build_bragg_qr_l_value_map(sim_state, "secondary", 1) == {
        controllers.bragg_qr_l_value_to_key(0.25): 0.25,
        controllers.bragg_qr_l_value_to_key(0.5): 0.5,
    }


def test_bragg_qr_manager_list_models_format_selection_and_status() -> None:
    bragg_state = state.BraggQrManagerState(
        selected_group_key=("secondary", 1),
        disabled_groups={("secondary", 1)},
        disabled_l_values={
            ("secondary", 1, controllers.bragg_qr_l_value_to_key(0.5))
        },
    )
    entries = [
        {
            "key": ("primary", 1),
            "source": "primary",
            "m": 1,
            "qr": 1.23456,
            "hk_preview": "(1 0)",
        },
        {
            "key": ("secondary", 1),
            "source": "secondary",
            "m": 1,
            "qr": 0.98765,
            "hk_preview": "(1 0), (0 1)",
        },
    ]

    qr_model = controllers.build_bragg_qr_qr_list_model(
        bragg_state,
        entries,
        selected_keys=[],
    )

    assert qr_model["index_keys"] == [("primary", 1), ("secondary", 1)]
    assert qr_model["selected_indices"] == [1]
    assert qr_model["see_index"] == 1
    assert qr_model["status_text"] == "Enabled: 1 / 2"
    assert qr_model["lines"][0] == "[ON ] primary   Qr=1.2346 A^-1  m=  1  HK=(1 0)"
    assert qr_model["lines"][1] == "[OFF] secondary Qr=0.9877 A^-1  m=  1  HK=(1 0), (0 1)"

    l_model = controllers.build_bragg_qr_l_list_model(
        bragg_state,
        group_key=("secondary", 1),
        l_value_map={
            controllers.bragg_qr_l_value_to_key(0.25): 0.25,
            controllers.bragg_qr_l_value_to_key(0.5): 0.5,
        },
        selected_l_keys=[controllers.bragg_qr_l_value_to_key(0.5)],
    )

    assert l_model["selected_group_key"] == ("secondary", 1)
    assert l_model["index_keys"] == [
        controllers.bragg_qr_l_value_to_key(0.25),
        controllers.bragg_qr_l_value_to_key(0.5),
    ]
    assert l_model["selected_indices"] == [1]
    assert l_model["status_text"] == (
        "Selected: secondary m=1 | Enabled L: 0 / 2 | Qr group disabled"
    )
    assert l_model["lines"] == ["[OFF] L=0.2500", "[OFF] L=0.5000"]
