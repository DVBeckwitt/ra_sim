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


def test_beam_center_display_to_native_row_col_mapping_uses_inverse_rotation() -> None:
    native_shape = (20, 20)
    native_col = 6.0
    native_row = 9.0
    display_col, display_row = geometry_overlay.rotate_point_for_display(
        native_col,
        native_row,
        native_shape,
        1,
    )

    mapped_col, mapped_row = geometry_overlay.rotate_point_for_display(
        display_col,
        display_row,
        native_shape,
        -1,
    )

    assert mapped_col == native_col
    assert mapped_row == native_row
    center_x = mapped_row
    center_y = mapped_col
    assert center_x == 9.0
    assert center_y == 6.0
