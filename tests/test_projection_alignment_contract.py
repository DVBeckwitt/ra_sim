from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ra_sim.gui import (
    geometry_q_group_manager,
    geometry_overlay,
    manual_geometry,
    overlays,
    peak_selection,
    qr_cylinder_overlay,
    state,
)
from ra_sim.simulation import exact_cake_portable
from ra_sim.simulation import intersection_cache_schema as schema


_SENTINEL_HKL = (1, 0, 2)


def _detector_cache_row(*, native_col: float, native_row: float) -> np.ndarray:
    row = np.full(schema.CURRENT_DETECTOR_CACHE_WIDTH, np.nan, dtype=np.float64)
    row[schema.CACHE_COL_QR] = 0.2
    row[schema.CACHE_COL_QZ] = 0.3
    row[schema.CACHE_COL_DETECTOR_COL] = native_col
    row[schema.CACHE_COL_DETECTOR_ROW] = native_row
    row[schema.CACHE_COL_INTENSITY] = 100.0
    row[schema.CACHE_COL_PHI] = 0.0
    row[schema.CACHE_COL_H] = _SENTINEL_HKL[0]
    row[schema.CACHE_COL_K] = _SENTINEL_HKL[1]
    row[schema.CACHE_COL_L] = _SENTINEL_HKL[2]
    row[schema.CACHE_COL_SOURCE_TABLE_INDEX] = 0
    row[schema.CACHE_COL_SOURCE_ROW_INDEX] = 0
    row[schema.CACHE_COL_BEST_SAMPLE_INDEX] = 0
    return row


def _caked_cache_row(
    *,
    native_col: float,
    native_row: float,
    caked_x: float,
    caked_y: float,
) -> np.ndarray:
    row = np.full(schema.CURRENT_CAKED_CACHE_WIDTH, np.nan, dtype=np.float64)
    row[: schema.CURRENT_DETECTOR_CACHE_WIDTH] = _detector_cache_row(
        native_col=native_col,
        native_row=native_row,
    )
    row[schema.CACHE_COL_CAKED_TWO_THETA] = caked_x
    row[schema.CACHE_COL_CAKED_PHI] = caked_y
    return row


def _reflection_q_group_metadata(*_args, **_kwargs):
    return ("primary", None, 0.0)


def _qr_config(*, render_in_caked_space: bool, image_size: int, display_rotate_k: int = 0):
    return qr_cylinder_overlay.build_qr_cylinder_overlay_render_config(
        render_in_caked_space=render_in_caked_space,
        image_size=image_size,
        display_rotate_k=display_rotate_k,
        center_col=3.0,
        center_row=2.0,
        distance_cor_to_detector=0.075,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        zs=0.0,
        zb=0.0,
        theta_initial_deg=0.0,
        cor_angle_deg=0.0,
        pixel_size_m=1.0e-4,
        wavelength=1.54,
        n2=1.0 + 0.0j,
    )


def _fake_trace_through_native_point(native_col: float, native_row: float):
    return SimpleNamespace(
        detector_col=np.asarray(
            [native_col - 1.0, native_col, native_col + 1.0],
            dtype=np.float64,
        ),
        detector_row=np.asarray([native_row, native_row, native_row], dtype=np.float64),
        valid_mask=np.asarray([True, True, True], dtype=bool),
    )


def _draw_qr_paths(ax, paths):
    artists: list[object] = []

    def _clear(redraw: bool = True):
        del redraw
        for artist in list(artists):
            artist.remove()
        artists.clear()

    overlays.draw_qr_cylinder_overlay_paths(
        ax,
        paths,
        qr_cylinder_overlay_artists=artists,
        clear_qr_cylinder_overlay_artists=_clear,
        draw_idle=lambda: None,
        redraw=False,
    )
    return artists


def _point_from_line(line) -> tuple[float, float]:
    return float(line.get_xdata()[0]), float(line.get_ydata()[0])


def _image_cell_center(
    *,
    image_shape: tuple[int, int],
    row: int,
    col: int,
    extent: tuple[float, float, float, float],
    origin: str,
) -> tuple[float, float]:
    n_rows, n_cols = (int(image_shape[0]), int(image_shape[1]))
    left, right, bottom, top = [float(v) for v in extent]
    x = left + (float(col) + 0.5) * (right - left) / float(n_cols)
    if origin == "upper":
        y = top + (float(row) + 0.5) * (bottom - top) / float(n_rows)
    else:
        y = bottom + (float(row) + 0.5) * (top - bottom) / float(n_rows)
    return float(x), float(y)


def _assert_overlaps_image_cell(
    point: tuple[float, float],
    *,
    image_shape: tuple[int, int],
    row: int,
    col: int,
    extent: tuple[float, float, float, float],
    origin: str,
    tolerance: float = 0.5,
) -> None:
    raster_center = _image_cell_center(
        image_shape=image_shape,
        row=int(row),
        col=int(col),
        extent=extent,
        origin=origin,
    )
    assert abs(float(point[0]) - raster_center[0]) <= float(tolerance)
    assert abs(float(point[1]) - raster_center[1]) <= float(tolerance)


def test_detector_background_hkl_and_qr_overlays_share_display_coordinates() -> None:
    native_shape = (5, 7)
    native_row = 1.0
    native_col = 5.0
    display_rotate_k = 1

    native_background = np.zeros(native_shape, dtype=np.float64)
    native_background[int(native_row), int(native_col)] = 1.0
    display_background = np.rot90(native_background, display_rotate_k)
    display_row, display_col = np.argwhere(display_background == 1.0)[0]
    expected_display = geometry_overlay.rotate_point_for_display(
        native_col,
        native_row,
        native_shape,
        display_rotate_k,
    )

    assert (float(display_col), float(display_row)) == pytest.approx(expected_display)

    runtime_state = state.SimulationRuntimeState(
        last_simulation_signature=("projection-alignment-contract", "detector"),
        stored_sim_image=np.zeros(native_shape, dtype=np.float64),
        stored_primary_intersection_cache=[
            _detector_cache_row(native_col=native_col, native_row=native_row).reshape(1, -1)
        ],
    )

    def _detector_display(col: float, row: float) -> tuple[float, float]:
        return geometry_overlay.rotate_point_for_display(
            col,
            row,
            native_shape,
            display_rotate_k,
        )

    def _poison_detector_hkl_sim_display(*_args):
        raise AssertionError("detector HKL overlays must use detector display coordinates")

    assert peak_selection.ensure_runtime_peak_overlay_data(
        runtime_state,
        primary_a=4.0,
        primary_c=6.0,
        native_sim_to_display_coords=_poison_detector_hkl_sim_display,
        native_detector_coords_to_detector_display_coords=_detector_display,
        reflection_q_group_metadata=_reflection_q_group_metadata,
        max_hits_per_reflection=1,
        min_separation_px=0.0,
        force=True,
    )
    assert runtime_state.peak_positions == pytest.approx([expected_display])

    def _detector_qr_display(
        col: float,
        row: float,
        _image_shape: tuple[int, int],
    ) -> tuple[float, float]:
        return _detector_display(col, row)

    paths = qr_cylinder_overlay.build_qr_cylinder_overlay_paths(
        [{"source": "primary", "m": 1, "qr": 0.2}],
        config=_qr_config(
            render_in_caked_space=False,
            image_size=max(native_shape),
            display_rotate_k=display_rotate_k,
        ),
        projection_context=None,
        native_sim_to_display_coords=_detector_qr_display,
        project_traces=lambda **_kwargs: [_fake_trace_through_native_point(native_col, native_row)],
    )
    assert len(paths) == 1
    assert (float(paths[0]["cols"][1]), float(paths[0]["rows"][1])) == pytest.approx(
        expected_display,
    )

    fig, ax = plt.subplots()
    try:
        extent = (0.0, float(display_background.shape[1]), float(display_background.shape[0]), 0.0)
        ax.imshow(display_background, origin="upper", extent=extent, interpolation="nearest")
        peak_line = ax.plot(
            [runtime_state.peak_positions[0][0]],
            [runtime_state.peak_positions[0][1]],
            "o",
        )[0]
        qr_artists = _draw_qr_paths(ax, paths)

        assert (float(peak_line.get_xdata()[0]), float(peak_line.get_ydata()[0])) == pytest.approx(
            expected_display,
        )
        assert (
            float(qr_artists[0].get_xdata()[1]),
            float(qr_artists[0].get_ydata()[1]),
        ) == pytest.approx(expected_display)

        # The matrix contract is exact above. The rendered raster is sampled at cell
        # centers, while overlays are drawn in pixel-index coordinates, so the visual
        # check allows the expected half-pixel offset but still catches flips/transposes.
        raster_center = _image_cell_center(
            image_shape=display_background.shape,
            row=int(display_row),
            col=int(display_col),
            extent=extent,
            origin="upper",
        )
        assert abs(raster_center[0] - expected_display[0]) <= 0.5
        assert abs(raster_center[1] - expected_display[1]) <= 0.5
    finally:
        plt.close(fig)


def test_caked_background_hkl_and_qr_overlays_share_display_coordinates(monkeypatch) -> None:
    detector_shape = (7, 7)
    native_col = 3.0
    native_row = 4.0
    target_caked = (30.0, 0.0)
    radial_axis = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    azimuth_axis = np.asarray([-20.0, -10.0, 0.0, 10.0, 20.0], dtype=np.float64)
    raw_azimuth_axis = exact_cake_portable.gui_phi_to_raw_phi(azimuth_axis)
    caked_background = np.zeros((azimuth_axis.size, radial_axis.size), dtype=np.float64)
    caked_background[2, 2] = 1.0

    bundle = qr_cylinder_overlay.CakeTransformBundle(
        detector_shape=detector_shape,
        radial_deg=radial_axis,
        raw_azimuth_deg=raw_azimuth_axis,
        gui_azimuth_deg=azimuth_axis,
        lut=SimpleNamespace(),
    )
    projection_context = {
        "detector_shape": detector_shape,
        "radial_axis": radial_axis,
        "azimuth_axis": azimuth_axis,
        "raw_azimuth_axis": raw_azimuth_axis,
        "transform_bundle": bundle,
    }

    detector_source_cache = [np.zeros((1, schema.CURRENT_DETECTOR_CACHE_WIDTH), dtype=np.float64)]
    runtime_state = state.SimulationRuntimeState(
        last_simulation_signature=("projection-alignment-contract", "caked"),
        stored_sim_image=np.zeros(detector_shape, dtype=np.float64),
        stored_intersection_cache=detector_source_cache,
        last_caked_intersection_cache=[
            _caked_cache_row(
                native_col=native_col,
                native_row=native_row,
                caked_x=target_caked[0],
                caked_y=target_caked[1],
            ).reshape(1, -1)
        ],
        last_caked_radial_values=radial_axis,
        last_caked_azimuth_values=azimuth_axis,
        last_caked_transform_bundle=bundle,
        last_caked_intersection_cache_transform_bundle=bundle,
        last_caked_intersection_cache_source_signature=(
            id(detector_source_cache),
            len(detector_source_cache),
        ),
    )

    def _poison_caked_hkl_display_fallback(*_args):
        raise AssertionError("caked HKL overlays must use cached caked coordinates")

    assert peak_selection.ensure_runtime_peak_overlay_data(
        runtime_state,
        primary_a=4.0,
        primary_c=6.0,
        caked_view_enabled_factory=True,
        native_sim_to_display_coords=_poison_caked_hkl_display_fallback,
        native_detector_coords_to_caked_display_coords=_poison_caked_hkl_display_fallback,
        native_detector_coords_to_detector_display_coords=_poison_caked_hkl_display_fallback,
        reflection_q_group_metadata=_reflection_q_group_metadata,
        max_hits_per_reflection=1,
        min_separation_px=0.0,
        force=True,
    )
    assert runtime_state.peak_positions == pytest.approx([target_caked])

    projection_map = {
        (native_col - 1.0, native_row): (20.0, target_caked[1]),
        (native_col, native_row): target_caked,
        (native_col + 1.0, native_row): (40.0, target_caked[1]),
    }

    def _fake_detector_pixel_to_caked_bin(active_bundle, col, row):
        assert active_bundle is bundle
        return projection_map.get((float(col), float(row)), (None, None))

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "detector_pixel_to_caked_bin",
        _fake_detector_pixel_to_caked_bin,
    )
    paths = qr_cylinder_overlay.build_qr_cylinder_overlay_paths(
        [{"source": "primary", "m": 1, "qr": 0.2}],
        config=_qr_config(render_in_caked_space=True, image_size=max(detector_shape)),
        projection_context=projection_context,
        native_sim_to_display_coords=lambda *_args: (_ for _ in ()).throw(
            AssertionError("caked Qr overlays must not use detector display coordinates")
        ),
        project_traces=lambda **_kwargs: [_fake_trace_through_native_point(native_col, native_row)],
    )
    assert len(paths) == 1
    assert (float(paths[0]["cols"][1]), float(paths[0]["rows"][1])) == pytest.approx(target_caked)

    fig, ax = plt.subplots()
    try:
        extent = (
            float(radial_axis.min()),
            float(radial_axis.max()),
            float(azimuth_axis.min()),
            float(azimuth_axis.max()),
        )
        ax.imshow(caked_background, origin="lower", extent=extent, interpolation="nearest")
        peak_line = ax.plot(
            [runtime_state.peak_positions[0][0]],
            [runtime_state.peak_positions[0][1]],
            "o",
        )[0]
        qr_artists = _draw_qr_paths(ax, paths)

        assert (float(peak_line.get_xdata()[0]), float(peak_line.get_ydata()[0])) == pytest.approx(
            target_caked
        )
        assert (
            float(qr_artists[0].get_xdata()[1]),
            float(qr_artists[0].get_ydata()[1]),
        ) == pytest.approx(target_caked)

        raster_center = _image_cell_center(
            image_shape=caked_background.shape,
            row=2,
            col=2,
            extent=extent,
            origin="lower",
        )
        assert abs(raster_center[0] - target_caked[0]) <= 0.5 * float(np.diff(radial_axis).max())
        assert abs(raster_center[1] - target_caked[1]) <= 0.5 * float(np.diff(azimuth_axis).max())
    finally:
        plt.close(fig)


def test_refined_caked_hkl_point_overlaps_detector_pixel_after_return_to_detector_view() -> None:
    detector_shape = (5, 7)
    display_rotate_k = 1
    original_native = (1.0, 4.0)
    refined_native = (5.0, 1.0)
    refined_caked = (10.2, 0.3)

    native_background = np.zeros(detector_shape, dtype=np.float64)
    native_background[int(refined_native[1]), int(refined_native[0])] = 1.0
    display_background = np.rot90(native_background, display_rotate_k)
    display_row, display_col = np.argwhere(display_background == 1.0)[0]
    expected_display = geometry_overlay.rotate_point_for_display(
        refined_native[0],
        refined_native[1],
        detector_shape,
        display_rotate_k,
    )
    original_display = geometry_overlay.rotate_point_for_display(
        original_native[0],
        original_native[1],
        detector_shape,
        display_rotate_k,
    )
    assert original_display != pytest.approx(expected_display)
    assert refined_caked != pytest.approx(expected_display)

    radial = np.linspace(0.0, 30.0, 301)
    azimuth = np.linspace(-5.0, 5.0, 201)
    caked_image = np.zeros((azimuth.size, radial.size), dtype=np.float64)
    caked_image[
        int(np.argmin(np.abs(azimuth - refined_caked[1]))),
        int(np.argmin(np.abs(radial - refined_caked[0]))),
    ] = 10.0
    stale_caked_record = {
        "hkl": _SENTINEL_HKL,
        "hkl_raw": _SENTINEL_HKL,
        "display_col": 10.0,
        "display_row": 0.0,
        "selected_display_col": 10.0,
        "selected_display_row": 0.0,
        "sim_col": 10.0,
        "sim_row": 0.0,
        "sim_col_raw": 10.0,
        "sim_row_raw": 0.0,
        "caked_x": 10.0,
        "caked_y": 0.0,
        "branch_id": "+x",
        "branch_source": "generated",
        "mosaic_weight": 1.0,
        "best_sample_index": 0,
        "source_row_index": 31,
    }
    runtime_state = state.SimulationRuntimeState(
        last_simulation_signature=("projection-alignment-contract", "hkl-refined-return"),
        stored_sim_image=np.zeros(detector_shape, dtype=np.float64),
        stored_primary_intersection_cache=[
            _detector_cache_row(
                native_col=original_native[0],
                native_row=original_native[1],
            ).reshape(1, -1)
        ],
        peak_positions=[(10.0, 0.0)],
        peak_millers=[_SENTINEL_HKL],
        peak_intensities=[1.0],
        peak_records=[stale_caked_record],
        last_caked_image_unscaled=caked_image,
        last_caked_radial_values=radial,
        last_caked_azimuth_values=azimuth,
    )
    peak_state = state.PeakSelectionState()

    def _detector_display(col: float, row: float) -> tuple[float, float]:
        return geometry_overlay.rotate_point_for_display(
            col,
            row,
            detector_shape,
            display_rotate_k,
        )

    def _caked_to_detector_display(two_theta: float, phi: float) -> tuple[float, float]:
        assert (float(two_theta), float(phi)) == pytest.approx(refined_caked, abs=0.12)
        return expected_display

    def _display_to_native(col: float, row: float) -> tuple[float, float]:
        assert (float(col), float(row)) == pytest.approx(expected_display)
        return refined_native

    fig, ax = plt.subplots()
    try:
        marker = ax.plot([], [], "o")[0]
        assert peak_selection.select_peak_by_hkl(
            runtime_state,
            peak_state,
            None,
            marker,
            *_SENTINEL_HKL,
            primary_a=4.0,
            ensure_peak_overlay_data=lambda **_kwargs: None,
            schedule_update=lambda: None,
            sync_peak_selection_state=lambda: None,
            set_status_text=lambda _text: None,
            draw_idle=lambda: None,
            caked_view_enabled=True,
            sync_hkl_vars=False,
            caked_angles_to_detector_display_coords=_caked_to_detector_display,
            detector_display_to_native_detector_coords=_display_to_native,
        )
        assert _point_from_line(marker) == pytest.approx(refined_caked, abs=0.12)
        selected_record = runtime_state.selected_peak_record
        assert selected_record["display_col"] == pytest.approx(expected_display[0])
        assert selected_record["display_row"] == pytest.approx(expected_display[1])
        assert selected_record["selected_display_col"] == pytest.approx(refined_caked[0], abs=0.12)
        assert selected_record["selected_display_row"] == pytest.approx(refined_caked[1], abs=0.12)

        def _ensure_detector_overlay(*, force: bool = False) -> bool:
            return peak_selection.ensure_runtime_peak_overlay_data(
                runtime_state,
                primary_a=4.0,
                primary_c=6.0,
                native_sim_to_display_coords=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("detector HKL refresh must use detector display mapping")
                ),
                native_detector_coords_to_detector_display_coords=_detector_display,
                reflection_q_group_metadata=_reflection_q_group_metadata,
                max_hits_per_reflection=1,
                min_separation_px=0.0,
                force=bool(force),
            )

        bindings = peak_selection.SelectedPeakRuntimeBindings(
            simulation_runtime_state=runtime_state,
            peak_selection_state=peak_state,
            hkl_lookup_view_state=None,
            selected_peak_marker=marker,
            current_primary_a_factory=lambda: 4.0,
            caked_view_enabled_factory=lambda: False,
            current_canvas_pick_config_factory=lambda: None,
            current_intersection_config_factory=lambda: None,
            ensure_peak_overlay_data=_ensure_detector_overlay,
            sync_peak_selection_state=lambda: None,
            set_status_text=lambda _text: None,
            draw_idle=lambda: None,
            native_sim_to_display_coords=lambda *_args: (_ for _ in ()).throw(
                AssertionError("detector HKL reselect must not use caked display fallback")
            ),
            native_detector_coords_to_detector_display_coords=_detector_display,
            caked_angles_to_detector_display_coords=lambda *_args: (_ for _ in ()).throw(
                AssertionError("detector HKL reselect must not ask for caked coordinates")
            ),
        )

        assert peak_selection.refresh_runtime_selected_peak_after_simulation_update(
            bindings,
            live_geometry_preview_enabled=False,
        )

        extent = (0.0, float(display_background.shape[1]), float(display_background.shape[0]), 0.0)
        ax.imshow(display_background, origin="upper", extent=extent, interpolation="nearest")
        detector_point = _point_from_line(marker)
        assert detector_point == pytest.approx(expected_display)
        assert detector_point != pytest.approx(refined_caked, abs=0.12)
        assert detector_point != pytest.approx(original_display)
        _assert_overlaps_image_cell(
            detector_point,
            image_shape=display_background.shape,
            row=int(display_row),
            col=int(display_col),
            extent=extent,
            origin="upper",
        )
    finally:
        plt.close(fig)


def test_refined_hkl_detector_caked_numeric_collision_still_overlaps_detector_pixel() -> None:
    detector_shape = (3, 13)
    refined_caked = (10.2, 0.3)
    expected_display = refined_caked
    signature = ("projection-alignment-contract", "numeric-collision")

    display_background = np.zeros(detector_shape, dtype=np.float64)
    display_background[0, 10] = 1.0
    display_row, display_col = np.argwhere(display_background == 1.0)[0]
    runtime_state = state.SimulationRuntimeState(
        last_simulation_signature=signature,
        selected_peak_record={
            "hkl": _SENTINEL_HKL,
            "hkl_raw": _SENTINEL_HKL,
            "refined_by": "caked_peak_center",
            "_refined_simulation_signature": signature,
            "refined_sim_caked_x": refined_caked[0],
            "refined_sim_caked_y": refined_caked[1],
            "caked_x": refined_caked[0],
            "caked_y": refined_caked[1],
            "two_theta_deg": refined_caked[0],
            "phi_deg": refined_caked[1],
            "refined_detector_display_col": expected_display[0],
            "refined_detector_display_row": expected_display[1],
            "display_col": expected_display[0],
            "display_row": expected_display[1],
            "sim_col": expected_display[0],
            "sim_row": expected_display[1],
            "sim_col_raw": expected_display[0],
            "sim_row_raw": expected_display[1],
        },
    )
    peak_state = state.PeakSelectionState(selected_hkl_target=_SENTINEL_HKL)

    fig, ax = plt.subplots()
    try:
        marker = ax.plot([], [], "o")[0]
        bindings = peak_selection.SelectedPeakRuntimeBindings(
            simulation_runtime_state=runtime_state,
            peak_selection_state=peak_state,
            hkl_lookup_view_state=None,
            selected_peak_marker=marker,
            current_primary_a_factory=lambda: 4.0,
            caked_view_enabled_factory=lambda: False,
            current_canvas_pick_config_factory=lambda: None,
            current_intersection_config_factory=lambda: None,
            ensure_peak_overlay_data=lambda *, force=False: True,
            sync_peak_selection_state=lambda: None,
            set_status_text=lambda _text: None,
            draw_idle=lambda: None,
            native_sim_to_display_coords=lambda *_args: (_ for _ in ()).throw(
                AssertionError("numeric collision must use explicit detector coords")
            ),
            caked_angles_to_detector_display_coords=lambda *_args: (_ for _ in ()).throw(
                AssertionError("detector reselect must not ask for caked coords")
            ),
        )

        assert peak_selection.refresh_runtime_selected_peak_after_simulation_update(
            bindings,
            live_geometry_preview_enabled=False,
        )

        extent = (0.0, float(detector_shape[1]), float(detector_shape[0]), 0.0)
        ax.imshow(display_background, origin="upper", extent=extent, interpolation="nearest")
        detector_point = _point_from_line(marker)
        assert detector_point == pytest.approx(expected_display)
        _assert_overlaps_image_cell(
            detector_point,
            image_shape=display_background.shape,
            row=int(display_row),
            col=int(display_col),
            extent=extent,
            origin="upper",
        )
    finally:
        plt.close(fig)


def test_caked_simulation_qr_point_overlaps_detector_pixel_after_detector_projection(
    monkeypatch,
) -> None:
    detector_shape = (5, 7)
    display_rotate_k = 1
    refined_native = (5.0, 1.0)
    refined_caked = (10.2, 0.3)
    other_branch_native = (2.0, 3.0)
    other_branch_caked = (12.4, -0.7)

    native_background = np.zeros(detector_shape, dtype=np.float64)
    native_background[int(refined_native[1]), int(refined_native[0])] = 1.0
    native_background[int(other_branch_native[1]), int(other_branch_native[0])] = 2.0
    display_background = np.rot90(native_background, display_rotate_k)
    display_row, display_col = np.argwhere(display_background == 1.0)[0]
    other_display_row, other_display_col = np.argwhere(display_background == 2.0)[0]
    expected_display = geometry_overlay.rotate_point_for_display(
        refined_native[0],
        refined_native[1],
        detector_shape,
        display_rotate_k,
    )
    expected_other_branch_display = geometry_overlay.rotate_point_for_display(
        other_branch_native[0],
        other_branch_native[1],
        detector_shape,
        display_rotate_k,
    )
    assert refined_caked != pytest.approx(expected_display)

    def _fake_caked_to_detector(two_theta: float, phi: float, **_kwargs):
        projection_map = {
            refined_caked: expected_display,
            (10.8, 0.9): (0.0, 0.0),
            other_branch_caked: expected_other_branch_display,
            (12.8, -1.1): (6.0, 4.0),
        }
        for caked_point, detector_point in projection_map.items():
            if (float(two_theta), float(phi)) == pytest.approx(caked_point):
                return detector_point
        raise AssertionError(f"unexpected caked projection point {(two_theta, phi)!r}")

    monkeypatch.setattr(
        manual_geometry,
        "caked_angles_to_background_display_coords",
        _fake_caked_to_detector,
    )

    def _raw_qr_peak(
        *,
        label: str,
        branch_id: str,
        mosaic_weight: float,
        caked_point: tuple[float, float],
        hkl: tuple[int, int, int],
        source_row_index: int,
    ) -> dict[str, object]:
        return {
            "label": label,
            "q_group_key": ("q_group", "primary", 1, 2),
            "source_table_index": 7,
            "source_row_index": source_row_index,
            "source_reflection_index": source_row_index,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "hkl": hkl,
            "branch_id": branch_id,
            "branch_source": "generated",
            "mosaic_weight": float(mosaic_weight),
            "display_col": caked_point[0],
            "display_row": caked_point[1],
            "selected_display_col": caked_point[0],
            "selected_display_row": caked_point[1],
            "sim_col": caked_point[0],
            "sim_row": caked_point[1],
            "sim_col_raw": caked_point[0],
            "sim_row_raw": caked_point[1],
            "refined_sim_caked_x": caked_point[0],
            "refined_sim_caked_y": caked_point[1],
            "caked_x": caked_point[0],
            "caked_y": caked_point[1],
        }

    raw_qr_peaks = [
        _raw_qr_peak(
            label="plus-winner",
            branch_id="+x",
            mosaic_weight=3.0,
            caked_point=refined_caked,
            hkl=(1, 0, 1),
            source_row_index=10,
        ),
        _raw_qr_peak(
            label="plus-loser",
            branch_id="+x",
            mosaic_weight=1.0,
            caked_point=(10.8, 0.9),
            hkl=(2, 0, 1),
            source_row_index=11,
        ),
        _raw_qr_peak(
            label="minus-winner",
            branch_id="-x",
            mosaic_weight=4.0,
            caked_point=other_branch_caked,
            hkl=(-1, 0, 1),
            source_row_index=12,
        ),
        _raw_qr_peak(
            label="minus-loser",
            branch_id="-x",
            mosaic_weight=1.0,
            caked_point=(12.8, -1.1),
            hkl=(-2, 0, 1),
            source_row_index=13,
        ),
    ]

    callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: np.zeros((3, 3), dtype=np.float64),
        last_caked_radial_values=lambda: np.asarray([10.0, 10.2, 10.4], dtype=np.float64),
        last_caked_azimuth_values=lambda: np.asarray([0.1, 0.3, 0.5], dtype=np.float64),
        current_background_display=lambda: display_background,
        current_background_native=lambda: native_background,
        image_size=max(detector_shape),
        rotate_point_for_display=geometry_overlay.rotate_point_for_display,
        display_rotate_k=display_rotate_k,
        native_sim_to_display_coords=lambda *_args: (_ for _ in ()).throw(
            AssertionError("caked-origin Qr point should back-project from caked angles")
        ),
        simulation_native_detector_coords_to_caked_display_coords=lambda *_args: (
            _ for _ in ()
        ).throw(AssertionError("detector projection should not need caked display refresh")),
        build_live_preview_simulated_peaks_from_cache=lambda: list(raw_qr_peaks),
        filter_simulated_peaks=lambda entries: (list(entries or ()), 0, 1),
        collapse_simulated_peaks=geometry_q_group_manager.collapse_geometry_fit_simulated_peaks,
    )
    projected = callbacks.simulated_peaks_for_params(prefer_cache=True)

    assert len(projected) == 2
    assert {entry["label"] for entry in projected} == {
        "plus-winner",
        "minus-winner",
    }
    assert {entry["branch_id"] for entry in projected} == {"+x", "-x"}
    selected_entry = next(entry for entry in projected if entry["label"] == "plus-winner")
    other_entry = next(entry for entry in projected if entry["label"] == "minus-winner")
    detector_point = (
        float(selected_entry["display_col"]),
        float(selected_entry["display_row"]),
    )
    other_detector_point = (
        float(other_entry["display_col"]),
        float(other_entry["display_row"]),
    )
    assert detector_point == pytest.approx(expected_display)
    assert detector_point != pytest.approx(refined_caked)
    assert other_detector_point == pytest.approx(expected_other_branch_display)
    assert other_detector_point != pytest.approx(other_branch_caked)

    fig, ax = plt.subplots()
    try:
        extent = (0.0, float(display_background.shape[1]), float(display_background.shape[0]), 0.0)
        ax.imshow(display_background, origin="upper", extent=extent, interpolation="nearest")
        line = ax.plot([detector_point[0]], [detector_point[1]], "o")[0]
        other_line = ax.plot([other_detector_point[0]], [other_detector_point[1]], "o")[0]
        _assert_overlaps_image_cell(
            _point_from_line(line),
            image_shape=display_background.shape,
            row=int(display_row),
            col=int(display_col),
            extent=extent,
            origin="upper",
        )
        _assert_overlaps_image_cell(
            _point_from_line(other_line),
            image_shape=display_background.shape,
            row=int(other_display_row),
            col=int(other_display_col),
            extent=extent,
            origin="upper",
        )
    finally:
        plt.close(fig)
