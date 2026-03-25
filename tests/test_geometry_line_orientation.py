import numpy as np

from ra_sim.gui.geometry_overlay import (
    display_to_native_sim_coords,
    inverse_orientation_transform,
    native_sim_to_display_coords,
    select_fit_orientation,
    transform_points_orientation,
)


def test_inverse_orientation_transform_roundtrips_real_world_orientation_choice() -> None:
    orientation_choice = {
        "k": 1,
        "flip_x": True,
        "flip_y": True,
        "flip_order": "yx",
        "indexing_mode": "xy",
    }
    shape = (3072, 3072)
    native_points = [
        (0.0, 0.0),
        (125.5, 331.25),
        (1499.0, 922.0),
        (2999.0, 1845.0),
    ]

    transformed = transform_points_orientation(
        native_points,
        shape,
        **orientation_choice,
    )
    inverse_choice = inverse_orientation_transform(shape, orientation_choice)
    roundtrip = transform_points_orientation(
        transformed,
        shape,
        **inverse_choice,
    )

    assert np.allclose(roundtrip, native_points, atol=1e-6)


def test_native_sim_display_coordinate_helpers_roundtrip() -> None:
    shape = (3072, 3072)
    native_points = [
        (0.0, 0.0),
        (125.5, 331.25),
        (1499.0, 922.0),
        (2999.0, 1845.0),
    ]

    for native_col, native_row in native_points:
        display_point = native_sim_to_display_coords(
            native_col,
            native_row,
            shape,
            sim_display_rotate_k=-1,
        )
        roundtrip = display_to_native_sim_coords(
            display_point[0],
            display_point[1],
            shape,
            sim_display_rotate_k=-1,
        )
        assert np.allclose(roundtrip, (native_col, native_row), atol=1e-6)


def test_select_fit_orientation_recovers_known_transform() -> None:
    expected_choice = {
        "k": 1,
        "flip_x": True,
        "flip_y": True,
        "flip_order": "yx",
        "indexing_mode": "xy",
    }
    shape = (3072, 3072)
    sim_coords = [
        (215.0, 340.0),
        (624.0, 911.0),
        (1255.0, 444.0),
        (1888.0, 1501.0),
    ]
    inverse_choice = inverse_orientation_transform(shape, expected_choice)
    meas_coords = transform_points_orientation(
        sim_coords,
        shape,
        **inverse_choice,
    )

    selected, diagnostics = select_fit_orientation(
        sim_coords,
        meas_coords,
        shape,
        cfg={"min_improvement_px": 0.0, "max_rms_px": 1.0e9},
    )

    for key, value in expected_choice.items():
        assert selected[key] == value
    assert selected["label"] == "rot90° CCW + flip_x + flip_y + order=yx + indexing=xy"
    assert diagnostics["reason"] == "selected_best"
    assert diagnostics["best_label"] == selected["label"]
