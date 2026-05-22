from __future__ import annotations

from dataclasses import replace

from ra_sim.gui.runtime_update_dependencies import (
    RuntimeCacheState,
    SimulationDependencySignatures,
    UpdateAction,
    classify_update,
)


def _signatures() -> SimulationDependencySignatures:
    return SimulationDependencySignatures(
        source_sig=("source", 1),
        physics_sig=(
            "physics",
            ("distance_m", 0.1),
            ("rotations", 0.0, 0.0, 0.0),
            ("wavelength", 1.54),
            ("pixel_size", 1.0e-4),
            ("sample_geometry", 1.0, 2.0, 3.0),
            ("solve_q", "adaptive", 128, 1.0e-6),
            ("beam_sampling", 1),
            ("mosaic_sampling", 1),
            ("optics_mode", "exact"),
        ),
        detector_projection_sig=("projection", 1),
        detector_center_sig=("center", 512.0, 512.0),
        primary_filter_sig=("primary_filter", ("all",)),
        combine_sig=("combine", 1.0, 0.0),
        analysis_geometry_sig=("analysis", 360, 720),
        display_sig=("display", "linear", 1.0),
        hit_table_sig=("hit_table", 0),
        full_image_sig=("full_image", 1),
    )


def test_initial_update_requires_full_simulation() -> None:
    decision = classify_update(None, _signatures(), RuntimeCacheState())

    assert decision.action is UpdateAction.FULL_SIMULATION
    assert decision.requires_worker is True
    assert decision.requires_analysis is False


def test_detector_center_only_uses_remap_when_cache_allows() -> None:
    previous = _signatures()
    current = replace(previous, detector_center_sig=("center", 520.0, 512.0))

    decision = classify_update(
        previous,
        current,
        RuntimeCacheState(can_remap_detector_center=True),
    )

    assert decision.action is UpdateAction.DETECTOR_CENTER_REMAP
    assert decision.requires_worker is False


def test_detector_center_with_hit_table_refresh_uses_remap_when_cache_allows() -> None:
    previous = _signatures()
    current = replace(
        previous,
        detector_center_sig=("center", 520.0, 512.0),
        hit_table_sig=("hit_table", 1),
    )

    decision = classify_update(
        previous,
        current,
        RuntimeCacheState(can_remap_detector_center=True),
    )

    assert decision.action is UpdateAction.DETECTOR_CENTER_REMAP
    assert decision.requires_worker is False


def test_detector_center_only_falls_back_without_exact_cache() -> None:
    previous = _signatures()
    current = replace(previous, detector_center_sig=("center", 520.0, 512.0))

    decision = classify_update(previous, current, RuntimeCacheState())

    assert decision.action is UpdateAction.FULL_SIMULATION
    assert decision.requires_worker is True


def test_distance_change_requires_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        physics_sig=(
            "physics",
            ("distance_m", 0.2),
            ("rotations", 0.0, 0.0, 0.0),
            ("wavelength", 1.54),
            ("pixel_size", 1.0e-4),
            ("sample_geometry", 1.0, 2.0, 3.0),
            ("solve_q", "adaptive", 128, 1.0e-6),
            ("beam_sampling", 1),
            ("mosaic_sampling", 1),
            ("optics_mode", "exact"),
        ),
        full_image_sig=("full_image", 2),
    )

    decision = classify_update(previous, current, RuntimeCacheState())

    assert decision.action is UpdateAction.FULL_SIMULATION
    assert decision.requires_worker is True


def test_orientation_change_requires_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        physics_sig=(
            "physics",
            ("distance_m", 0.1),
            ("rotations", 0.0, 1.0, 0.0),
            ("wavelength", 1.54),
            ("pixel_size", 1.0e-4),
            ("sample_geometry", 1.0, 2.0, 3.0),
            ("solve_q", "adaptive", 128, 1.0e-6),
            ("beam_sampling", 1),
            ("mosaic_sampling", 1),
            ("optics_mode", "exact"),
        ),
        full_image_sig=("full_image", 2),
    )

    decision = classify_update(previous, current, RuntimeCacheState())

    assert decision.action is UpdateAction.FULL_SIMULATION
    assert decision.requires_worker is True


def test_theta_initial_change_requires_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        physics_sig=(
            "physics",
            ("distance_m", 0.1),
            ("rotations", 0.0, 0.0, 0.0),
            ("wavelength", 1.54),
            ("pixel_size", 1.0e-4),
            ("sample_geometry", 1.0, 2.0, 3.0),
            ("theta_initial", 7.5),
            ("solve_q", "adaptive", 128, 1.0e-6),
            ("beam_sampling", 1),
            ("mosaic_sampling", 1),
            ("optics_mode", "exact"),
        ),
        full_image_sig=("full_image", "theta-initial-7.5"),
    )

    decision = classify_update(previous, current, RuntimeCacheState())

    assert decision.action is UpdateAction.FULL_SIMULATION
    assert decision.requires_worker is True


def test_center_plus_distance_change_requires_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        detector_center_sig=("center", 520.0, 512.0),
        physics_sig=(
            "physics",
            ("distance_m", 0.2),
            ("rotations", 0.0, 0.0, 0.0),
            ("wavelength", 1.54),
            ("pixel_size", 1.0e-4),
            ("sample_geometry", 1.0, 2.0, 3.0),
            ("solve_q", "adaptive", 128, 1.0e-6),
            ("beam_sampling", 1),
            ("mosaic_sampling", 1),
            ("optics_mode", "exact"),
        ),
        full_image_sig=("full_image", 2),
    )

    decision = classify_update(
        previous,
        current,
        RuntimeCacheState(can_remap_detector_center=True),
    )

    assert decision.action is UpdateAction.FULL_SIMULATION
    assert decision.requires_worker is True


def test_center_plus_rotation_change_requires_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        detector_center_sig=("center", 520.0, 512.0),
        physics_sig=(
            "physics",
            ("distance_m", 0.1),
            ("rotations", 1.0, 0.0, 0.0),
            ("wavelength", 1.54),
            ("pixel_size", 1.0e-4),
            ("sample_geometry", 1.0, 2.0, 3.0),
            ("solve_q", "adaptive", 128, 1.0e-6),
            ("beam_sampling", 1),
            ("mosaic_sampling", 1),
            ("optics_mode", "exact"),
        ),
        full_image_sig=("full_image", 2),
    )

    decision = classify_update(
        previous,
        current,
        RuntimeCacheState(can_remap_detector_center=True),
    )

    assert decision.action is UpdateAction.FULL_SIMULATION
    assert decision.requires_worker is True


def test_center_plus_source_change_requires_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        detector_center_sig=("center", 520.0, 512.0),
        source_sig=("source", 2),
        full_image_sig=("full_image", 2),
    )

    decision = classify_update(
        previous,
        current,
        RuntimeCacheState(can_remap_detector_center=True),
    )

    assert decision.action is UpdateAction.FULL_SIMULATION
    assert decision.requires_worker is True


def test_physics_signature_changes_require_full_simulation() -> None:
    previous = _signatures()
    physics_cases = [
        ("wavelength", 1.55),
        ("pixel_size", 2.0e-4),
        ("sample_geometry", 2.0, 2.0, 3.0),
        ("solve_q", "adaptive", 256, 1.0e-6),
        ("beam_sampling", 2),
        ("mosaic_sampling", 2),
    ]

    for idx, changed_component in enumerate(physics_cases):
        current = replace(
            previous,
            physics_sig=(
                "physics",
                ("distance_m", 0.1),
                ("rotations", 0.0, 0.0, 0.0),
                ("wavelength", 1.54),
                ("pixel_size", 1.0e-4),
                ("sample_geometry", 1.0, 2.0, 3.0),
                ("solve_q", "adaptive", 128, 1.0e-6),
                ("beam_sampling", 1),
                ("mosaic_sampling", 1),
                ("optics_mode", "exact"),
                changed_component,
            ),
            full_image_sig=("full_image", idx + 10),
        )

        decision = classify_update(
            previous,
            current,
            RuntimeCacheState(can_remap_detector_center=True),
        )

        assert decision.action is UpdateAction.FULL_SIMULATION
        assert decision.requires_worker is True


def test_detector_projection_signature_changes_require_full_simulation() -> None:
    previous = _signatures()
    projection_cases = [
        ("projection", "pixel_size", 2.0e-4),
        ("projection", "wavelength", 1.55),
        ("projection", "distance", 0.2),
    ]

    for current_projection in projection_cases:
        current = replace(
            previous,
            detector_projection_sig=current_projection,
            full_image_sig=("full_image", current_projection),
        )

        decision = classify_update(
            previous,
            current,
            RuntimeCacheState(can_remap_detector_center=True),
        )

        assert decision.action is UpdateAction.FULL_SIMULATION
        assert decision.requires_worker is True


def test_prune_reuse_when_cache_reports_reuse() -> None:
    previous = _signatures()
    current = replace(previous, primary_filter_sig=("primary_filter", ("strong",)))

    decision = classify_update(
        previous,
        current,
        RuntimeCacheState(prune_cache_mode="reuse"),
    )

    assert decision.action is UpdateAction.PRIMARY_PRUNE_REUSE
    assert decision.requires_worker is False
    assert decision.missing_contribution_keys == frozenset()


def test_prune_fill_when_cache_reports_missing_keys() -> None:
    previous = _signatures()
    current = replace(previous, primary_filter_sig=("primary_filter", ("strong", "weak")))
    missing_keys = frozenset({"weak"})

    decision = classify_update(
        previous,
        current,
        RuntimeCacheState(
            prune_cache_mode="fill",
            missing_contribution_keys=missing_keys,
        ),
    )

    assert decision.action is UpdateAction.PRIMARY_PRUNE_FILL
    assert decision.requires_worker is True
    assert decision.missing_contribution_keys == missing_keys


def test_display_only_change_requires_no_worker() -> None:
    previous = _signatures()
    current = replace(previous, display_sig=("display", "log", 1.0))

    decision = classify_update(previous, current, RuntimeCacheState())

    assert decision.action is UpdateAction.DISPLAY_ONLY
    assert decision.requires_worker is False
    assert decision.requires_analysis is False


def test_display_visibility_toggle_change_requires_no_worker() -> None:
    previous = _signatures()
    current = replace(previous, display_sig=("display", "overlays_visible", False))

    decision = classify_update(previous, current, RuntimeCacheState())

    assert decision.action is UpdateAction.DISPLAY_ONLY
    assert decision.requires_worker is False


def test_combine_only_change_requires_no_worker() -> None:
    previous = _signatures()
    current = replace(previous, combine_sig=("combine", 1.0, 0.5))

    decision = classify_update(previous, current, RuntimeCacheState())

    assert decision.action is UpdateAction.COMBINE_ONLY
    assert decision.requires_worker is False
    assert decision.requires_analysis is False


def test_analysis_geometry_only_change_requires_analysis_not_simulation_worker() -> None:
    previous = _signatures()
    current = replace(previous, analysis_geometry_sig=("analysis", 180, 360))

    decision = classify_update(previous, current, RuntimeCacheState())

    assert decision.action is UpdateAction.ANALYSIS_ONLY
    assert decision.requires_worker is False
    assert decision.requires_analysis is True


def test_mixed_nonphysics_changes_fail_closed_to_full_simulation() -> None:
    previous = _signatures()
    current = replace(
        previous,
        display_sig=("display", "log", 1.0),
        analysis_geometry_sig=("analysis", 180, 360),
    )

    decision = classify_update(previous, current, RuntimeCacheState())

    assert decision.action is UpdateAction.FULL_SIMULATION
    assert decision.requires_worker is True
