import pytest
import numpy as np

from ra_sim.gui.geometry_overlay import (
    build_geometry_fit_overlay_records,
    compose_orientation_transforms,
    compare_holistic_sim_residuals,
    compute_holistic_sim_residual,
    compute_geometry_overlay_frame_diagnostics,
    build_geometry_fit_visual_probe_records,
    inverse_transform_points_orientation,
    probe_display_image_peak_near_point,
    rotate_point_for_display,
    summarize_geometry_fit_overlay_visual_distances,
    transform_points_orientation,
)


def test_compute_holistic_sim_residual_prefers_aligned_simulation():
    background = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 8.0, 0.0],
        [0.0, 2.0, 6.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    aligned_sim = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 4.0, 0.0],
        [0.0, 1.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    shifted_sim = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 4.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    aligned = compute_holistic_sim_residual(background, aligned_sim)
    shifted = compute_holistic_sim_residual(background, shifted_sim)

    assert aligned["status"] == "ok"
    assert shifted["status"] == "ok"
    assert aligned["scale"] == pytest.approx(2.0)
    assert aligned["rmse"] < shifted["rmse"]
    assert aligned["mae"] < shifted["mae"]


def test_compare_holistic_sim_residuals_flags_worse_fit():
    initial = compute_holistic_sim_residual([[1.0, 2.0]], [[1.0, 2.0]])
    final = compute_holistic_sim_residual([[1.0, 2.0]], [[2.0, 1.0]])

    comparison = compare_holistic_sim_residuals(initial, final)

    assert comparison["holistic_fit_suspicious"] is True
    assert comparison["holistic_residual_delta_rmse"] > 0.0


def test_probe_display_image_peak_near_point_finds_peak_in_display_coordinates():
    image = np.zeros((7, 9), dtype=np.float64)
    image[2, 6] = 12.0

    probe = probe_display_image_peak_near_point(
        image,
        (6.2, 4.1),
        extent=(-0.5, 8.5, 6.5, -0.5),
        search_radius_px=3,
    )

    assert probe["status"] == "ok"
    assert probe["image_peak_point"] == pytest.approx((6.0, 4.0))
    assert probe["image_peak_index"] == (2, 6)
    assert probe["peak_value"] == pytest.approx(12.0)
    assert probe["point_to_image_peak_delta"] < 0.25


def test_probe_display_image_peak_near_point_flags_shifted_marker():
    image = np.zeros((8, 8), dtype=np.float64)
    image[1, 1] = 1.0
    image[4, 5] = 9.0

    probe = probe_display_image_peak_near_point(
        image,
        (1.0, 1.0),
        extent=(-0.5, 7.5, -0.5, 7.5),
        search_radius_px=5,
    )

    assert probe["status"] == "ok"
    assert probe["image_peak_point"] == pytest.approx((5.0, 4.0))
    assert probe["point_to_image_peak_delta"] == pytest.approx(5.0)


def test_build_geometry_fit_visual_probe_records_compares_artist_record_and_image_peak():
    image = np.zeros((8, 8), dtype=np.float64)
    image[4, 5] = 9.0
    draw_records = [
        {
            "overlay_match_index": 2,
            "hkl": (0, 0, 6),
            "record_point": (1.0, 1.0),
            "artist_point": (1.5, 1.0),
            "record_source": "fit_prediction_caked_deg",
        }
    ]

    probes = build_geometry_fit_visual_probe_records(
        draw_records,
        image,
        extent=(-0.5, 7.5, -0.5, 7.5),
        search_radius_px=6,
    )

    assert len(probes) == 1
    assert probes[0]["status"] == "ok"
    assert probes[0]["image_peak_point"] == pytest.approx((5.0, 4.0))
    assert probes[0]["artist_to_record_delta"] == pytest.approx(0.5)
    assert probes[0]["record_to_image_peak_delta"] == pytest.approx(5.0)
    assert probes[0]["artist_to_image_peak_delta"] > 4.5


def test_build_geometry_fit_overlay_records_preserves_duplicate_hkls():
    initial_pairs_display = [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 0),
            "sim_display": (100.0, 50.0),
            "bg_display": (102.0, 52.0),
        },
        {
            "overlay_match_index": 1,
            "hkl": (1, 0, 0),
            "sim_display": (200.0, 150.0),
            "bg_display": (202.0, 152.0),
        },
    ]
    point_match_diagnostics = [
        {
            "match_status": "matched",
            "overlay_match_index": 0,
            "hkl": (1, 0, 0),
            "simulated_x": 10.0,
            "simulated_y": 20.0,
            "measured_x": 12.0,
            "measured_y": 22.0,
            "distance_px": 2.0,
        },
        {
            "match_status": "matched",
            "overlay_match_index": 1,
            "hkl": (1, 0, 0),
            "simulated_x": 30.0,
            "simulated_y": 40.0,
            "measured_x": 32.0,
            "measured_y": 42.0,
            "distance_px": 2.0,
        },
    ]

    records = build_geometry_fit_overlay_records(
        initial_pairs_display,
        point_match_diagnostics,
        native_shape=(256, 256),
        orientation_choice={
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        },
        sim_display_rotate_k=0,
        background_display_rotate_k=0,
    )

    assert len(records) == 2
    assert [entry["overlay_match_index"] for entry in records] == [0, 1]
    assert records[0]["initial_bg_display"] == pytest.approx((102.0, 52.0))
    assert records[1]["initial_bg_display"] == pytest.approx((202.0, 152.0))


def test_build_geometry_fit_overlay_records_inverts_orientation_before_display():
    native_shape = (11, 11)
    orientation_choice = {
        "indexing_mode": "yx",
        "k": 1,
        "flip_x": True,
        "flip_y": False,
        "flip_order": "xy",
    }
    native_sim = (2.0, 8.0)
    native_bg = (4.0, 6.0)
    fit_bg = transform_points_orientation(
        [native_bg],
        native_shape,
        **orientation_choice,
    )[0]
    expected_sim_display = rotate_point_for_display(
        native_sim[0],
        native_sim[1],
        native_shape,
        -1,
    )
    expected_bg_display = rotate_point_for_display(
        native_bg[0],
        native_bg[1],
        native_shape,
        -1,
    )

    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 5,
                "hkl": (2, 1, 0),
                "sim_display": (999.0, 998.0),
                "bg_display": (997.0, 996.0),
                "sim_native": native_sim,
                "bg_native": native_bg,
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 5,
                "hkl": (2, 1, 0),
                "simulated_x": native_sim[0],
                "simulated_y": native_sim[1],
                "measured_x": fit_bg[0],
                "measured_y": fit_bg[1],
                "distance_px": 3.0,
            }
        ],
        native_shape=native_shape,
        orientation_choice=orientation_choice,
        sim_display_rotate_k=0,
        background_display_rotate_k=-1,
    )

    assert len(records) == 1
    assert records[0]["initial_sim_display"] == pytest.approx(expected_sim_display)
    assert records[0]["initial_bg_display"] == pytest.approx(expected_bg_display)
    assert records[0]["final_sim_display"] == pytest.approx(expected_sim_display)
    assert records[0]["final_bg_display"] == pytest.approx(expected_bg_display)

    frame_diag, frame_warning = compute_geometry_overlay_frame_diagnostics(records)
    assert frame_diag["paired_records"] == pytest.approx(1.0)
    assert frame_diag["sim_display_med_px"] == pytest.approx(0.0)
    assert frame_diag["bg_display_med_px"] == pytest.approx(0.0)
    assert frame_warning == ""


def test_build_geometry_fit_overlay_records_matches_real_world_orientation_case():
    native_shape = (3072, 3072)
    orientation_choice = {
        "indexing_mode": "xy",
        "k": 1,
        "flip_x": True,
        "flip_y": True,
        "flip_order": "yx",
    }
    native_sim = (1827.0, 1362.0)
    measured_fit = (1820.992, 1375.983)
    native_bg = inverse_transform_points_orientation(
        [measured_fit],
        native_shape,
        orientation_choice,
    )[0]
    expected_sim_display = rotate_point_for_display(
        native_sim[0],
        native_sim[1],
        native_shape,
        -1,
    )
    expected_bg_display = rotate_point_for_display(
        native_bg[0],
        native_bg[1],
        native_shape,
        -1,
    )

    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (-1, 0, 5),
                # Cached detector-view aliases from the simulator frame should
                # not override canonical native detector coordinates.
                "sim_display": native_sim,
                "bg_display": measured_fit,
                "sim_native": native_sim,
                "bg_native": native_bg,
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (-1, 0, 5),
                "simulated_x": native_sim[0],
                "simulated_y": native_sim[1],
                "measured_x": measured_fit[0],
                "measured_y": measured_fit[1],
                "distance_px": 15.219,
            }
        ],
        native_shape=native_shape,
        orientation_choice=orientation_choice,
        sim_display_rotate_k=0,
        background_display_rotate_k=-1,
    )

    assert len(records) == 1
    assert records[0]["simulated_frame"] == "sim_native"
    assert records[0]["measured_frame"] == "fit_oriented"
    assert records[0]["initial_sim_display"] == pytest.approx(expected_sim_display)
    assert records[0]["initial_bg_display"] == pytest.approx(expected_bg_display)
    assert records[0]["final_sim_display"] == pytest.approx(expected_sim_display)
    assert records[0]["final_bg_display"] == pytest.approx(expected_bg_display)

    frame_diag, frame_warning = compute_geometry_overlay_frame_diagnostics(records)
    assert frame_diag["sim_display_med_px"] == pytest.approx(0.0)
    assert frame_diag["bg_display_med_px"] == pytest.approx(0.0)
    assert frame_warning == ""


def test_build_geometry_fit_overlay_records_falls_back_to_initial_native_points():
    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "sim_native": (10.0, 20.0),
                "bg_native": (12.0, 22.0),
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "simulated_x": 10.0,
                "simulated_y": 20.0,
                "measured_x": 12.0,
                "measured_y": 22.0,
                "distance_px": 2.0,
            }
        ],
        native_shape=(256, 256),
        orientation_choice={
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        },
        sim_display_rotate_k=0,
        background_display_rotate_k=0,
    )

    assert records[0]["initial_sim_display"] == pytest.approx((10.0, 20.0))
    assert records[0]["initial_bg_display"] == pytest.approx((12.0, 22.0))
    assert records[0]["initial_sim_native"] == pytest.approx((10.0, 20.0))
    assert records[0]["initial_bg_native"] == pytest.approx((12.0, 22.0))

    frame_diag, frame_warning = compute_geometry_overlay_frame_diagnostics(records)
    assert frame_diag["paired_records"] == pytest.approx(1.0)
    assert frame_diag["sim_display_med_px"] == pytest.approx(0.0)
    assert frame_diag["bg_display_med_px"] == pytest.approx(0.0)
    assert frame_warning == ""


def test_compose_orientation_transforms_matches_sequential_application():
    native_shape = (23, 17)
    first = {
        "indexing_mode": "yx",
        "k": 1,
        "flip_x": True,
        "flip_y": False,
        "flip_order": "xy",
    }
    second = {
        "indexing_mode": "xy",
        "k": 3,
        "flip_x": False,
        "flip_y": True,
        "flip_order": "yx",
    }
    points = [(1.0, 2.0), (6.0, 10.0), (15.0, 20.0)]

    sequential = transform_points_orientation(
        transform_points_orientation(points, native_shape, **first),
        native_shape,
        **second,
    )
    combined = compose_orientation_transforms(native_shape, first, second)
    composed = transform_points_orientation(points, native_shape, **combined)

    assert composed == pytest.approx(sequential)


def test_build_geometry_fit_overlay_records_prefers_native_initial_points_over_cached_view_coords():
    native_shape = (256, 256)
    expected_sim_display = rotate_point_for_display(
        10.0,
        20.0,
        native_shape,
        -1,
    )
    expected_bg_display = rotate_point_for_display(
        12.0,
        22.0,
        native_shape,
        -1,
    )

    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                # These simulate view-specific cached coordinates captured in a
                # different projection than the current detector overlay redraw.
                "sim_display": (101.0, 202.0),
                "bg_display": (303.0, 404.0),
                "sim_native": (10.0, 20.0),
                "bg_native": (12.0, 22.0),
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "simulated_x": 10.0,
                "simulated_y": 20.0,
                "measured_x": 12.0,
                "measured_y": 22.0,
                "distance_px": 2.0,
            }
        ],
        native_shape=native_shape,
        orientation_choice={
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        },
        sim_display_rotate_k=0,
        background_display_rotate_k=-1,
    )

    assert records[0]["initial_sim_display"] == pytest.approx(expected_sim_display)
    assert records[0]["initial_bg_display"] == pytest.approx(expected_bg_display)


def test_build_geometry_fit_overlay_records_reprojects_legacy_display_fallbacks():
    native_shape = (256, 256)
    native_sim = (10.0, 20.0)
    native_bg = (12.0, 22.0)
    legacy_sim_display = rotate_point_for_display(
        native_sim[0],
        native_sim[1],
        native_shape,
        1,
    )
    legacy_bg_display = rotate_point_for_display(
        native_bg[0],
        native_bg[1],
        native_shape,
        -1,
    )
    expected_overlay_sim_display = rotate_point_for_display(
        native_sim[0],
        native_sim[1],
        native_shape,
        -1,
    )

    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "sim_display": legacy_sim_display,
                "bg_display": legacy_bg_display,
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "simulated_x": native_sim[0],
                "simulated_y": native_sim[1],
                "measured_x": native_bg[0],
                "measured_y": native_bg[1],
                "distance_px": 2.0,
            }
        ],
        native_shape=native_shape,
        orientation_choice={
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        },
        sim_display_rotate_k=1,
        background_display_rotate_k=-1,
    )

    assert records[0]["initial_sim_native"] == pytest.approx(native_sim)
    assert records[0]["initial_bg_native"] == pytest.approx(native_bg)
    assert records[0]["initial_sim_display"] == pytest.approx(expected_overlay_sim_display)
    assert records[0]["initial_bg_display"] == pytest.approx(legacy_bg_display)
    assert records[0]["final_sim_display"] == pytest.approx(expected_overlay_sim_display)
    assert records[0]["final_bg_display"] == pytest.approx(legacy_bg_display)


def test_build_geometry_fit_overlay_records_preserves_caked_display_locks():
    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "sim_display": (101.0, 202.0),
                "bg_display": (303.0, 404.0),
                "sim_caked_display": (11.0, 12.0),
                "bg_caked_display": (13.0, 14.0),
                "sim_native": (10.0, 20.0),
                "bg_native": (12.0, 22.0),
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "simulated_x": 10.0,
                "simulated_y": 20.0,
                "measured_x": 12.0,
                "measured_y": 22.0,
                "distance_px": 2.0,
            }
        ],
        native_shape=(256, 256),
        orientation_choice={
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        },
        sim_display_rotate_k=0,
        background_display_rotate_k=0,
    )

    assert records[0]["initial_sim_caked_display"] == pytest.approx((11.0, 12.0))
    assert records[0]["initial_bg_caked_display"] == pytest.approx((13.0, 14.0))
    assert records[0]["final_bg_caked_display"] == pytest.approx((13.0, 14.0))


def test_build_geometry_fit_overlay_records_uses_fit_caked_angles_for_final_caked_markers():
    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (0, 0, 6),
                "sim_caked_display": (40.0, 12.0),
                "bg_caked_display": (40.5, 12.5),
                "sim_native": (10.0, 20.0),
                "bg_native": (12.0, 22.0),
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (0, 0, 6),
                "simulated_x": 900.0,
                "simulated_y": 901.0,
                "measured_x": 12.0,
                "measured_y": 22.0,
                "simulated_two_theta_deg": 40.6,
                "simulated_phi_deg": 12.4,
                "measured_two_theta_deg": 40.5,
                "measured_phi_deg": 12.5,
                "distance_px": 2.0,
            }
        ],
        native_shape=(1024, 1024),
        orientation_choice={
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        },
        sim_display_rotate_k=0,
        background_display_rotate_k=0,
    )

    assert records[0]["final_sim_caked_display"] == pytest.approx((40.6, 12.4))
    assert records[0]["final_bg_caked_display"] == pytest.approx((40.5, 12.5))


def test_build_geometry_fit_overlay_records_prefers_fit_prediction_sim_points():
    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (0, 0, 9),
                "sim_display": (10.0, 20.0),
                "bg_display": (12.0, 22.0),
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (0, 0, 9),
                # Legacy/stale solver-display fields should not define the
                # green fit marker when fitted prediction fields are present.
                "simulated_x": 900.0,
                "simulated_y": 901.0,
                "simulated_two_theta_deg": 99.0,
                "simulated_phi_deg": -99.0,
                "fit_prediction_detector_display_px": (30.0, 40.0),
                "fit_prediction_detector_native_px": (130.0, 140.0),
                "fit_prediction_caked_deg": (40.6, 12.4),
                "measured_x": 12.0,
                "measured_y": 22.0,
                "measured_two_theta_deg": 40.5,
                "measured_phi_deg": 12.5,
                "distance_px": 2.0,
            }
        ],
        native_shape=(1024, 1024),
        orientation_choice={
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        },
        sim_display_rotate_k=0,
        background_display_rotate_k=0,
    )

    assert records[0]["final_sim_display"] == pytest.approx((30.0, 40.0))
    assert records[0]["final_sim_native"] == pytest.approx((130.0, 140.0))
    assert records[0]["final_sim_caked_display"] == pytest.approx((40.6, 12.4))
    assert records[0]["final_sim_display_source"] == "fit_prediction_detector_display_px"
    assert records[0]["final_sim_caked_display_source"] == "fit_prediction_caked_deg"


def test_build_geometry_fit_overlay_records_uses_fit_prediction_display_as_caked_fallback():
    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (0, 0, 3),
                "sim_display": (9.2, 0.0),
                "bg_display": (9.1, -1.2),
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (0, 0, 3),
                # Caked fit progress records can carry the current fit-space
                # prediction under this legacy detector-display field name.
                "objective_cache_mode": "point_only_projection",
                "fit_prediction_detector_display_px": (9.2478, -1.2446),
                "fit_prediction_detector_native_px": (130.0, 140.0),
                "simulated_x": 900.0,
                "simulated_y": 901.0,
                "simulated_two_theta_deg": 9.2526,
                "simulated_phi_deg": 0.0,
                "measured_x": 12.0,
                "measured_y": 22.0,
                "measured_two_theta_deg": 9.1,
                "measured_phi_deg": -1.2,
                "distance_px": 2.0,
            }
        ],
        native_shape=(1024, 1024),
        orientation_choice={
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        },
        sim_display_rotate_k=0,
        background_display_rotate_k=0,
    )

    assert records[0]["final_sim_caked_display"] == pytest.approx((9.2478, -1.2446))
    assert records[0]["final_sim_caked_display_source"] == "fit_prediction_detector_display_px"


def test_build_geometry_fit_overlay_records_prefers_rendered_caked_sim_over_display_alias():
    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (0, 0, 3),
                "sim_display": (9.2, 0.0),
                "bg_display": (9.1, -1.2),
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (0, 0, 3),
                "objective_cache_mode": "point_only_projection",
                # This legacy field can be stale detector/display data. The
                # rendered caked sim point is the source of truth when present.
                "fit_prediction_detector_display_px": (9.2478, -1.2446),
                "fit_prediction_caked_deg": (9.2478, -1.2446),
                "sim_visual_caked_deg": (18.0, 4.0),
                "sim_refined_caked_deg": (18.1, 4.1),
                "simulated_x": 900.0,
                "simulated_y": 901.0,
                "measured_x": 12.0,
                "measured_y": 22.0,
                "measured_two_theta_deg": 9.1,
                "measured_phi_deg": -1.2,
                "distance_px": 2.0,
            }
        ],
        native_shape=(1024, 1024),
        orientation_choice={
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        },
        sim_display_rotate_k=0,
        background_display_rotate_k=0,
    )

    assert records[0]["final_sim_caked_display"] == pytest.approx((18.0, 4.0))
    assert records[0]["final_sim_caked_display_source"] == "sim_visual_caked_deg"
    assert records[0]["final_sim_render_caked_display"] == pytest.approx((18.0, 4.0))
    assert records[0]["fit_sim_render_caked_delta"] == pytest.approx(0.0)


def test_build_geometry_fit_overlay_records_keeps_detector_display_out_of_caked_fallback():
    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (0, 0, 3),
                "sim_display": (9.2, 0.0),
                "bg_display": (9.1, -1.2),
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (0, 0, 3),
                "fit_prediction_detector_display_px": (900.0, 901.0),
                "fit_prediction_detector_native_px": (130.0, 140.0),
                "sim_refined_caked_deg": (9.2526, 0.0),
                "simulated_x": 130.0,
                "simulated_y": 140.0,
                "measured_x": 12.0,
                "measured_y": 22.0,
                "distance_px": 2.0,
            }
        ],
        native_shape=(1024, 1024),
        orientation_choice={
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        },
        sim_display_rotate_k=0,
        background_display_rotate_k=0,
    )

    assert records[0]["final_sim_caked_display"] == pytest.approx((9.2526, 0.0))
    assert records[0]["final_sim_caked_display_source"] == "sim_refined_caked_deg"


def test_build_geometry_fit_overlay_records_keeps_unmatched_initial_pairs_visible():
    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "sim_display": (10.0, 20.0),
                "bg_display": (12.0, 22.0),
            },
            {
                "overlay_match_index": 1,
                "hkl": (2, 0, 0),
                "sim_display": (30.0, 40.0),
                "bg_display": (32.0, 42.0),
            },
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "simulated_x": 10.0,
                "simulated_y": 20.0,
                "measured_x": 12.0,
                "measured_y": 22.0,
                "distance_px": 2.0,
            }
        ],
        native_shape=(256, 256),
        orientation_choice={
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        },
        sim_display_rotate_k=0,
        background_display_rotate_k=0,
    )

    assert [entry["overlay_match_index"] for entry in records] == [0, 1]
    assert records[0]["match_status"] == "matched"
    assert records[1]["match_status"] == "missing_pair"
    assert records[1]["initial_sim_display"] == pytest.approx((30.0, 40.0))
    assert records[1]["initial_bg_display"] == pytest.approx((32.0, 42.0))
    assert "final_sim_display" not in records[1]


def test_compute_geometry_overlay_frame_diagnostics_projects_native_points_in_caked_view():
    records = [
        {
            "initial_sim_display": (999.0, 999.0),
            "initial_bg_display": (777.0, 777.0),
            "final_sim_display": (10.0, 20.0),
            "final_bg_display": (12.0, 22.0),
            "initial_sim_native": (1.0, 2.0),
            "initial_bg_native": (3.0, 4.0),
            "final_sim_native": (1.0, 2.0),
            "final_bg_native": (3.0, 4.0),
        }
    ]

    frame_diag, frame_warning = compute_geometry_overlay_frame_diagnostics(
        records,
        show_caked_2d=True,
        native_detector_coords_to_caked_display_coords=(
            lambda col, row: (100.0 + float(col), 200.0 + float(row))
        ),
    )

    assert frame_diag["paired_records"] == pytest.approx(1.0)
    assert frame_diag["sim_display_med_px"] == pytest.approx(0.0)
    assert frame_diag["bg_display_med_px"] == pytest.approx(0.0)
    assert frame_warning == ""


def test_visual_distance_summary_uses_the_same_caked_points_as_overlay_drawing():
    records = [
        {
            "match_status": "matched",
            "initial_sim_display": (900.0, 900.0),
            "initial_bg_display": (910.0, 910.0),
            "final_sim_display": (950.0, 950.0),
            "final_bg_display": (10.0, 10.0),
            "initial_sim_native": (1.0, 2.0),
            "initial_bg_native": (3.0, 4.0),
            "final_sim_native": (9.0, 9.0),
            "final_bg_native": (10.0, 10.0),
            "initial_sim_caked_display": (40.0, -10.0),
            "initial_bg_caked_display": (41.0, -11.0),
            "final_sim_caked_display": (40.7, -10.4),
            "final_bg_caked_display": (40.6, -10.5),
        }
    ]

    detector_summary = summarize_geometry_fit_overlay_visual_distances(records)
    caked_summary = summarize_geometry_fit_overlay_visual_distances(
        records,
        show_caked_2d=True,
        native_detector_coords_to_caked_display_coords=(
            lambda col, row: (100.0 + float(col), 200.0 + float(row))
        ),
    )

    assert detector_summary["final_distance_median"] > 1000.0
    assert caked_summary["initial_distance_median"] == pytest.approx(2**0.5)
    assert caked_summary["final_distance_median"] == pytest.approx(0.02**0.5)
    assert caked_summary["worsened_count"] == pytest.approx(0.0)


def test_compute_geometry_overlay_frame_diagnostics_flags_fit_sim_render_mismatch():
    records = [
        {
            "initial_sim_display": (float(idx), 0.0),
            "initial_bg_display": (float(idx), 0.0),
            "final_sim_display": (float(idx), 0.0),
            "final_bg_display": (float(idx), 0.0),
            "fit_sim_render_caked_delta": 2.0 + float(idx),
        }
        for idx in range(3)
    ]

    frame_diag, frame_warning = compute_geometry_overlay_frame_diagnostics(
        records,
    )

    assert frame_diag["fit_sim_render_caked_delta_count"] == pytest.approx(3.0)
    assert frame_diag["fit_sim_render_caked_delta_median"] == pytest.approx(3.0)
    assert frame_diag["fit_sim_render_caked_delta_max"] == pytest.approx(4.0)
    assert "rendered simulation caked positions" in frame_warning
