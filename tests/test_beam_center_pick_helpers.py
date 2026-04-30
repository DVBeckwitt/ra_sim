import numpy as np
import pytest

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


def test_beam_center_default_display_pick_maps_to_slider_row_col() -> None:
    row, col = geometry_overlay.beam_center_row_col_from_detector_display(
        1404.0,
        1453.0,
        (3000, 3000),
        -1,
    )

    assert row == pytest.approx(1596.0)
    assert col == pytest.approx(1453.0)


def test_beam_center_mapping_uses_extent_not_pixel_index_frame() -> None:
    pixel_col, pixel_row = geometry_overlay.detector_display_to_native_coords(
        1404.0,
        1453.0,
        (3000, 3000),
        -1,
    )
    beam_row, beam_col = geometry_overlay.beam_center_row_col_from_detector_display(
        1404.0,
        1453.0,
        (3000, 3000),
        -1,
    )

    assert (pixel_row, pixel_col) == pytest.approx((1595.0, 1453.0))
    assert (beam_row, beam_col) == pytest.approx((1596.0, 1453.0))


def test_detector_display_to_native_round_trips_non_square_pixel_indices() -> None:
    native_shape = (2400, 3000)
    native_col = 211.0
    native_row = 1777.0

    display_col, display_row = geometry_overlay.detector_native_to_display_coords(
        native_col,
        native_row,
        native_shape,
        -1,
    )
    roundtrip_col, roundtrip_row = geometry_overlay.detector_display_to_native_coords(
        display_col,
        display_row,
        native_shape,
        -1,
    )

    assert roundtrip_col == pytest.approx(native_col)
    assert roundtrip_row == pytest.approx(native_row)


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
    center_row, center_col = geometry_overlay.beam_center_row_col_from_detector_display(
        1404.0,
        1453.0,
        (3000, 3000),
        -1,
    )

    assert center_row == pytest.approx(1596.0)
    assert center_col == pytest.approx(1453.0)


def test_beam_center_poni_defaults_keep_native_row_col() -> None:
    pixel_size_m = 1.0e-4

    row, col = geometry_overlay.beam_center_row_col_from_poni(
        1596.0 * pixel_size_m,
        1453.0 * pixel_size_m,
        pixel_size_m,
    )

    assert row == pytest.approx(1596.0)
    assert col == pytest.approx(1453.0)


def test_beam_center_poni_defaults_do_not_use_rotated_display_formula() -> None:
    pixel_size_m = 1.0e-4
    image_size = 3000.0
    poni1_m = 1596.0 * pixel_size_m
    poni2_m = 1453.0 * pixel_size_m

    row, col = geometry_overlay.beam_center_row_col_from_poni(
        poni1_m,
        poni2_m,
        pixel_size_m,
    )

    assert (row, col) == pytest.approx((1596.0, 1453.0))
    assert (row, col) != pytest.approx(
        (poni2_m / pixel_size_m, image_size - poni1_m / pixel_size_m)
    )


def test_runtime_beam_center_refiner_bypasses_center_dependent_caked_wrapper() -> None:
    from pathlib import Path

    runtime_path = Path(geometry_overlay.__file__).with_name("_runtime") / "runtime_session.py"
    source = runtime_path.read_text(encoding="utf-8")
    start = source.index("def _refine_beam_center_pick_display_point(")
    end = source.index("def _update_beam_center_pick_preview(", start)
    body_source = source[start:end]

    assert "gui_manual_geometry.geometry_manual_refine_preview_point" in body_source
    assert "_geometry_manual_refine_preview_point" not in body_source
    assert "use_caked_space=False" in body_source


def test_headless_defaults_use_same_poni_row_col_frame() -> None:
    from pathlib import Path

    headless_path = Path(geometry_overlay.__file__).parents[1] / "headless_geometry_fit.py"
    source = headless_path.read_text(encoding="utf-8")
    start = source.index("def _build_runtime_defaults(")
    end = source.index("def _build_var_store(", start)
    body_source = source[start:end]

    assert "beam_center_row_col_from_poni" in body_source
    assert "float(poni2 / pixel_size_m)" not in body_source
    assert "image_size - (poni1 / pixel_size_m)" not in body_source
