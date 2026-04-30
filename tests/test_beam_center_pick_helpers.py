import numpy as np

from ra_sim.gui import geometry_overlay, manual_geometry


def test_beam_center_refinement_uses_manual_detector_peak_refiner() -> None:
    background = np.zeros((40, 50), dtype=float)
    background[18, 27] = 100.0
    background[17, 26] = 25.0

    refined_col, refined_row = manual_geometry.geometry_manual_refine_preview_point(
        None,
        24.0,
        16.0,
        display_background=background,
        cache_data={"manual_auto_refine_search_radius_px": 8.0},
        use_caked_space=False,
    )

    assert refined_col == 27.0
    assert refined_row == 18.0


def test_beam_center_display_to_native_row_col_mapping_uses_display_shape() -> None:
    native_shape = (3142, 3092)
    native_col = 1453.0
    native_row = 1596.0
    display_col, display_row = geometry_overlay.rotate_point_for_display(
        native_col,
        native_row,
        native_shape,
        -1,
    )

    wrong_col, wrong_row = geometry_overlay.rotate_point_for_display(
        display_col,
        display_row,
        native_shape,
        1,
    )
    mapped_col, mapped_row = geometry_overlay.display_point_to_native_for_rotation(
        display_col,
        display_row,
        native_shape,
        -1,
    )

    assert wrong_col == native_col
    assert wrong_row == 1546.0
    assert mapped_col == native_col
    assert mapped_row == native_row
    center_x = mapped_row
    center_y = mapped_col
    assert center_x == 1596.0
    assert center_y == 1453.0


def test_beam_center_projection_callback_unrotates_non_square_display() -> None:
    native_shape = (3142, 3092)
    native_background = np.zeros(native_shape, dtype=float)
    display_background = np.rot90(native_background, -1)
    native_col = 1453.0
    native_row = 1596.0
    display_col, display_row = geometry_overlay.rotate_point_for_display(
        native_col,
        native_row,
        native_shape,
        -1,
    )
    callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: display_background,
        current_background_native=lambda: native_background,
        display_rotate_k=-1,
    )

    mapped_col, mapped_row = callbacks.background_display_to_native_detector_coords(
        display_col,
        display_row,
    )

    assert mapped_col == native_col
    assert mapped_row == native_row


def test_beam_center_visual_pick_maps_to_default_center_frame() -> None:
    display_shape = (3000, 3000)
    display_col = 1596.0
    display_row = 1546.0

    center_col, center_row = geometry_overlay.rotate_point_for_display(
        display_col,
        display_row,
        display_shape,
        -1,
    )

    assert center_row == 1596.0
    assert center_col == 1453.0
