from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ra_sim.gui import geometry_overlay, overlays, peak_selection, qr_cylinder_overlay, state
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
