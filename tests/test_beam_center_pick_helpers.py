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

    assert row == pytest.approx(1453.0)
    assert col == pytest.approx(1596.0)


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
    assert (beam_row, beam_col) == pytest.approx((1453.0, 1596.0))


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

    assert center_row == pytest.approx(1453.0)
    assert center_col == pytest.approx(1596.0)


def test_beam_center_pick_uses_clicked_display_point_exactly() -> None:
    center_row, center_col = geometry_overlay.beam_center_row_col_from_detector_display(
        1452.0,
        1599.0,
        (3000, 3000),
        -1,
    )

    assert center_row == pytest.approx(1599.0)
    assert center_col == pytest.approx(1548.0)


def test_beam_center_pick_gui_row_follows_display_row_and_col_mirrors_display_column() -> None:
    display_col = 1456.0
    display_row = 1607.0
    mapped_row, mapped_col = geometry_overlay.beam_center_row_col_from_detector_display(
        display_col,
        display_row,
        (3000, 3000),
        -1,
    )

    assert mapped_row == pytest.approx(display_row)
    assert mapped_col == pytest.approx(3000.0 - display_col)
    assert (mapped_row, mapped_col) == pytest.approx((1607.0, 1544.0))


def test_beam_center_marker_projection_round_trips_display_pick() -> None:
    display_col = 1456.0
    display_row = 1607.0
    center_row, center_col = geometry_overlay.beam_center_row_col_from_detector_display(
        display_col,
        display_row,
        (3000, 3000),
        -1,
    )

    projected_col, projected_row = geometry_overlay.beam_center_row_col_to_detector_display(
        center_row,
        center_col,
        (3000, 3000),
        -1,
    )

    assert (projected_col, projected_row) == pytest.approx((display_col, display_row))


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


def test_runtime_beam_center_preview_does_not_auto_refine_clicked_point() -> None:
    from pathlib import Path

    runtime_path = Path(geometry_overlay.__file__).with_name("_runtime") / "runtime_session.py"
    source = runtime_path.read_text(encoding="utf-8")
    start = source.index("def _update_beam_center_pick_preview(")
    end = source.index("def _ensure_slider_includes_value(", start)
    body_source = source[start:end]

    assert "refined_col = float(col)" in body_source
    assert "refined_row = float(row)" in body_source
    assert "geometry_manual_refine_preview_point" not in body_source
    assert "_geometry_manual_refine_preview_point" not in body_source


def test_runtime_beam_center_commit_writes_one_gui_row_col_pair() -> None:
    from pathlib import Path

    runtime_path = Path(geometry_overlay.__file__).with_name("_runtime") / "runtime_session.py"
    source = runtime_path.read_text(encoding="utf-8")
    shape_start = source.index("def _current_beam_center_coordinate_shape(")
    shape_end = source.index("def _beam_center_pick_session_active(", shape_start)
    shape_source = source[shape_start:shape_end]
    writer_start = source.index("def _set_beam_center_row_col_sliders(")
    writer_end = source.index("def _beam_center_display_to_center_coords(", writer_start)
    writer_source = source[writer_start:writer_end]
    mapper_end = source.index("def _commit_beam_center_pick_at(", writer_end)
    mapper_source = source[writer_end:mapper_end]
    commit_end = source.index("def _cancel_beam_center_pick(", mapper_end)
    commit_source = source[mapper_end:commit_end]

    assert "_get_current_background_native" not in shape_source
    assert "coordinate_shape = _current_beam_center_coordinate_shape()" in mapper_source
    assert "return float(center_row), float(center_col)" in mapper_source
    assert "return float(center_col), float(center_row)" not in mapper_source
    assert "gui_row_value, gui_col_value = _beam_center_display_to_center_coords(" in commit_source
    assert "gui_row_value = center_col" not in commit_source
    assert "gui_col_value = center_row" not in commit_source
    assert "_set_visible_beam_center_row(row_float)" in writer_source
    assert "_set_visible_beam_center_col(col_float)" in writer_source
    assert "_set_beam_center_row_col_sliders(" in commit_source
    assert "gui_row_value," in commit_source
    assert "gui_col_value," in commit_source
    assert (
        "Beam center set to row={gui_row_value:.2f}"
        in commit_source
    )
    assert "center_x_var.set(float(gui_row_value))" not in commit_source
    assert "center_y_var.set(float(gui_col_value))" not in commit_source
    assert "center_x_var.set(float(center_row))" not in commit_source
    assert "center_y_var.set(float(center_col))" not in commit_source
    assert "native_point" not in commit_source
    assert "native_row = float(native_point[0])" not in commit_source
    assert "native_col = float(native_point[1])" not in commit_source


def test_runtime_beam_center_marker_uses_display_projection_not_pick_swap() -> None:
    from pathlib import Path

    runtime_path = Path(geometry_overlay.__file__).with_name("_runtime") / "runtime_session.py"
    source = runtime_path.read_text(encoding="utf-8")
    start = source.index("def _sync_center_marker(")
    end = source.index("def _toggle_beam_center_spot(", start)
    marker_source = source[start:end]

    assert "view_mode = _current_app_shell_view_mode()" in marker_source
    assert 'view_mode == "q_space"' in marker_source
    assert 'view_mode == "caked"' in marker_source
    assert "beam_center_row_col_to_detector_display" in marker_source
    assert "float(center_row)," in marker_source
    assert "float(center_col)," in marker_source



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
