import pytest

from ra_sim.gui.geometry_overlay import (
    build_geometry_fit_overlay_records,
    compute_geometry_overlay_frame_diagnostics,
    rotate_point_for_display,
    transform_points_orientation,
)


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
        orientation_choice={"indexing_mode": "xy", "k": 0, "flip_x": False, "flip_y": False, "flip_order": "yx"},
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
        0,
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
                "sim_display": expected_sim_display,
                "bg_display": expected_bg_display,
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
    initial_sim_display = (1827.0, 1362.0)
    initial_bg_display = (1820.992, 1375.983)

    records = build_geometry_fit_overlay_records(
        [
            {
                "overlay_match_index": 0,
                "hkl": (-1, 0, 5),
                "sim_display": initial_sim_display,
                "bg_display": initial_bg_display,
            }
        ],
        [
            {
                "match_status": "matched",
                "overlay_match_index": 0,
                "hkl": (-1, 0, 5),
                "simulated_x": 1827.0,
                "simulated_y": 1362.0,
                "measured_x": 1820.992,
                "measured_y": 1375.983,
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
    assert records[0]["final_sim_display"] == pytest.approx(initial_sim_display)
    assert records[0]["final_bg_display"] == pytest.approx(initial_bg_display)

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
        orientation_choice={"indexing_mode": "xy", "k": 0, "flip_x": False, "flip_y": False, "flip_order": "yx"},
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
