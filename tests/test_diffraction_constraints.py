import json
from pathlib import Path

import numpy as np
import pytest

from ra_sim.simulation import diffraction
from ra_sim.simulation import diffraction_debug
from ra_sim.simulation import intersection_cache_schema as cache_schema
from ra_sim.utils.calculations import (
    IndexofRefraction,
    _legacy_kernel_n2_sample_array_from_angstrom,
    resolve_index_of_refraction,
)
from ra_sim.utils import stacking_fault


def test_legacy_kernel_n2_sample_array_matches_kernel_fallback_rules():
    wavelengths = np.array([1.0, np.nan, -1.0, 1.54], dtype=np.float64)
    nominal_n2 = 0.99 + 0.01j

    actual = _legacy_kernel_n2_sample_array_from_angstrom(
        wavelengths,
        nominal_n2=nominal_n2,
        sample_count=4,
    )

    expected = np.array(
        [
            IndexofRefraction(1.0e-10),
            nominal_n2,
            nominal_n2,
            IndexofRefraction(1.54e-10),
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def test_process_peaks_parallel_debug_uses_override_sample_weight_and_pixel_size(
    monkeypatch,
):
    solve_q_calls: list[float] = []
    intersection_call = {"count": 0}

    def fake_intersect_line_plane(_p0, _k_vec, plane_point, _n_plane):
        call_idx = intersection_call["count"] % 2
        intersection_call["count"] += 1
        if call_idx == 0:
            return 0.0, 0.0, 0.0, True
        return float(plane_point[0] + 2.0e-4), float(plane_point[1]), float(plane_point[2]), True

    def fake_solve_q(k_in_crystal, k_scat, _g_vec, _sigma_rad):
        solve_q_calls.append(float(k_scat))
        assert np.isclose(np.linalg.norm(k_in_crystal), 4.0 * np.pi)
        return np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    monkeypatch.setattr(diffraction_debug, "intersect_line_plane", fake_intersect_line_plane)
    monkeypatch.setattr(diffraction_debug, "solve_q", fake_solve_q)

    image = np.zeros((24, 24), dtype=np.float64)
    image_out, max_positions, q_data, q_count = diffraction_debug.process_peaks_parallel_debug(
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        24,
        1.0,
        1.0,
        1.0,
        image,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        7.0 + 0.0j,
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        0.0,
        0.0,
        0.0,
        np.asarray([1.0], dtype=np.float64),
        0.0,
        0.0,
        [12.0, 12.0],
        0.0,
        0.0,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=1,
        sample_weights=np.asarray([2.0], dtype=np.float64),
        pixel_size_m=2.0e-4,
        n2_sample_array_override=np.asarray([2.0 + 0.0j], dtype=np.complex128),
    )

    assert solve_q_calls == pytest.approx([4.0 * np.pi])
    assert image_out[12, 13] == pytest.approx(2.0)
    np.testing.assert_allclose(max_positions[0, :3], [2.0, 13.0, 12.0])
    assert q_data.shape[0] == 1
    assert int(q_count[0]) == 1


def test_process_peaks_parallel_debug_skips_nonpositive_sample_weights(monkeypatch):
    solve_q_calls: list[float] = []

    monkeypatch.setattr(
        diffraction_debug,
        "intersect_line_plane",
        lambda *_args, **_kwargs: pytest.fail("zero-weight samples must skip projection"),
    )
    monkeypatch.setattr(
        diffraction_debug,
        "solve_q",
        lambda *_args, **_kwargs: solve_q_calls.append(1.0),
    )

    image = np.zeros((16, 16), dtype=np.float64)
    image_out, max_positions, _q_data, _q_count = diffraction_debug.process_peaks_parallel_debug(
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        16,
        1.0,
        1.0,
        1.0,
        image,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        0.0,
        0.0,
        0.0,
        np.asarray([1.0], dtype=np.float64),
        0.0,
        0.0,
        [8.0, 8.0],
        0.0,
        0.0,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
        sample_weights=np.asarray([0.0], dtype=np.float64),
    )

    assert solve_q_calls == []
    assert np.count_nonzero(image_out) == 0
    np.testing.assert_allclose(
        max_positions[0], [-1.0, np.nan, np.nan, -1.0, np.nan, np.nan], equal_nan=True
    )


def test_process_peaks_parallel_debug_clamps_mismatched_core_sample_arrays(monkeypatch):
    solve_q_calls: list[float] = []
    intersection_call = {"count": 0}

    def fake_intersect_line_plane(_p0, _k_vec, plane_point, _n_plane):
        call_idx = intersection_call["count"] % 2
        intersection_call["count"] += 1
        if call_idx == 0:
            return 0.0, 0.0, 0.0, True
        return float(plane_point[0] + 2.0e-4), float(plane_point[1]), float(plane_point[2]), True

    def fake_solve_q(k_in_crystal, k_scat, _g_vec, _sigma_rad):
        solve_q_calls.append(float(k_scat))
        assert np.isclose(np.linalg.norm(k_in_crystal), 4.0 * np.pi)
        return np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    monkeypatch.setattr(diffraction_debug, "intersect_line_plane", fake_intersect_line_plane)
    monkeypatch.setattr(diffraction_debug, "solve_q", fake_solve_q)

    image = np.zeros((24, 24), dtype=np.float64)
    image_out, max_positions, q_data, q_count = diffraction_debug.process_peaks_parallel_debug(
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        24,
        1.0,
        1.0,
        1.0,
        image,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        7.0 + 0.0j,
        np.asarray([0.0, 9.9], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0, 0.2], dtype=np.float64),
        np.asarray([0.0, 0.3], dtype=np.float64),
        0.0,
        0.0,
        0.0,
        np.asarray([1.0, 1.6], dtype=np.float64),
        0.0,
        0.0,
        [12.0, 12.0],
        0.0,
        0.0,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=1,
        sample_weights=np.asarray([3.0, 99.0], dtype=np.float64),
        pixel_size_m=2.0e-4,
        n2_sample_array_override=np.asarray([2.0 + 0.0j, 9.0 + 0.0j], dtype=np.complex128),
    )

    assert solve_q_calls == pytest.approx([4.0 * np.pi])
    assert image_out[12, 13] == pytest.approx(3.0)
    np.testing.assert_allclose(max_positions[0, :3], [3.0, 13.0, 12.0])
    assert q_data.shape[0] == 1
    assert int(q_count[0]) == 1


@pytest.mark.parametrize("pixel_size_m", [0.0, np.nan])
def test_process_peaks_parallel_debug_falls_back_for_invalid_pixel_size(
    monkeypatch,
    pixel_size_m,
):
    solve_q_calls: list[float] = []
    intersection_call = {"count": 0}

    def fake_intersect_line_plane(_p0, _k_vec, plane_point, _n_plane):
        call_idx = intersection_call["count"] % 2
        intersection_call["count"] += 1
        if call_idx == 0:
            return 0.0, 0.0, 0.0, True
        return float(plane_point[0] + 2.0e-4), float(plane_point[1]), float(plane_point[2]), True

    def fake_solve_q(k_in_crystal, k_scat, _g_vec, _sigma_rad):
        solve_q_calls.append(float(k_scat))
        assert np.isclose(np.linalg.norm(k_in_crystal), 4.0 * np.pi)
        return np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    monkeypatch.setattr(diffraction_debug, "intersect_line_plane", fake_intersect_line_plane)
    monkeypatch.setattr(diffraction_debug, "solve_q", fake_solve_q)

    image = np.zeros((24, 24), dtype=np.float64)
    image_out, max_positions, q_data, q_count = diffraction_debug.process_peaks_parallel_debug(
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        24,
        1.0,
        1.0,
        1.0,
        image,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        7.0 + 0.0j,
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        0.0,
        0.0,
        0.0,
        np.asarray([1.0], dtype=np.float64),
        0.0,
        0.0,
        [12.0, 12.0],
        0.0,
        0.0,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=1,
        pixel_size_m=pixel_size_m,
        n2_sample_array_override=np.asarray([2.0 + 0.0j], dtype=np.complex128),
    )

    assert solve_q_calls == pytest.approx([4.0 * np.pi])
    assert image_out[12, 14] == pytest.approx(1.0)
    np.testing.assert_allclose(max_positions[0, :3], [1.0, 14.0, 12.0])
    assert q_data.shape[0] == 1
    assert int(q_count[0]) == 1


def test_process_peaks_parallel_debug_reports_q_debug_truncation(monkeypatch):
    intersection_call = {"count": 0}

    def fake_intersect_line_plane(_p0, _k_vec, plane_point, _n_plane):
        call_idx = intersection_call["count"] % 2
        intersection_call["count"] += 1
        if call_idx == 0:
            return 0.0, 0.0, 0.0, True
        return float(plane_point[0] + 2.0e-4), float(plane_point[1]), float(plane_point[2]), True

    def fake_solve_q(_k_in_crystal, _k_scat, _g_vec, _sigma_rad):
        return np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(diffraction_debug, "intersect_line_plane", fake_intersect_line_plane)
    monkeypatch.setattr(diffraction_debug, "solve_q", fake_solve_q)

    image = np.zeros((24, 24), dtype=np.float64)
    image_out, max_positions, q_data, q_count = diffraction_debug.process_peaks_parallel_debug(
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        24,
        1.0,
        1.0,
        1.0,
        image,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        7.0 + 0.0j,
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        0.0,
        0.0,
        0.0,
        np.asarray([1.0], dtype=np.float64),
        0.0,
        0.0,
        [12.0, 12.0],
        0.0,
        0.0,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=1,
        pixel_size_m=2.0e-4,
        n2_sample_array_override=np.asarray([2.0 + 0.0j], dtype=np.complex128),
        q_debug_max_solutions_per_peak=1,
    )

    assert image_out[12, 13] > 0.0
    assert max_positions.shape == (1, 6)
    assert q_data.shape == (1, 1, 5)
    assert np.all(q_count <= 1)
    stats = diffraction_debug.get_last_process_peaks_debug_q_stats()
    assert stats["save_flag"] == 1
    assert stats["q_debug_capacity_per_peak"] == 1
    assert stats["q_debug_saved_solution_count"] > 0
    assert stats["q_debug_truncated_solution_count"] > 0


def test_process_qr_rods_parallel_debug_forwards_optional_debug_kwargs(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        stacking_fault,
        "qr_dict_to_arrays",
        lambda _qr_dict: (
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
            np.asarray([2.0], dtype=np.float64),
            np.asarray([3], dtype=np.int64),
            None,
        ),
    )
    monkeypatch.setattr(
        diffraction_debug,
        "process_peaks_parallel_debug",
        lambda *_args, **kwargs: captured.update(kwargs) or ("image", "maxpos", "qdata", "qcount"),
    )

    result = diffraction_debug.process_qr_rods_parallel_debug(
        {},
        16,
        1.0,
        1.0,
        1.0,
        np.zeros((16, 16), dtype=np.float64),
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        0.0,
        0.0,
        0.0,
        np.asarray([1.0], dtype=np.float64),
        0.0,
        0.0,
        [8.0, 8.0],
        0.0,
        0.0,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
        psi_z_deg=0.2,
        sample_weights=np.asarray([4.0], dtype=np.float64),
        pixel_size_m=3.0e-4,
        n2_sample_array_override=np.asarray([1.1 + 0.0j], dtype=np.complex128),
        q_debug_max_solutions_per_peak=1,
    )

    np.testing.assert_array_equal(captured["sample_weights"], [4.0])
    assert captured["pixel_size_m"] == pytest.approx(3.0e-4)
    np.testing.assert_array_equal(captured["n2_sample_array_override"], [1.1 + 0.0j])
    assert captured["q_debug_max_solutions_per_peak"] == 1
    assert result[:4] == ("image", "maxpos", "qdata", "qcount")
    np.testing.assert_array_equal(result[4], np.asarray([3], dtype=np.int64))


def test_process_peaks_parallel_skips_negative_l(monkeypatch):
    called_l_values = []

    def fake_solve_q(
        _k_in_crystal,
        _k_scat,
        _G_vec,
        _sigma,
        _gamma_pv,
        _eta_pv,
        H,
        K,
        L,
        *_args,
        **_kwargs,
    ):
        called_l_values.append(float(L))
        return np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), 0

    def fake_project_weighted_candidate(**kwargs):
        return True, 8.0, 8.0, 0.0, 1.0

    monkeypatch.setattr(
        diffraction,
        "solve_q",
        fake_solve_q,
    )
    monkeypatch.setattr(
        diffraction,
        "_project_weighted_candidate",
        fake_project_weighted_candidate,
    )

    miller = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    intensities = np.array([1.0, 1.0], dtype=np.float64)
    image_size = 16
    image = np.zeros((image_size, image_size), dtype=np.float64)

    diffraction.process_peaks_parallel.py_func(
        miller,
        intensities,
        image_size,
        1.0,
        1.0,
        1.0,
        image,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        0.5,
        0.5,
        0.0,
        np.array([1.0], dtype=np.float64),
        0.0,
        0.0,
        [8.0, 8.0],
        6.0,
        0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
    )

    assert called_l_values
    assert all(value == pytest.approx(1.0) for value in called_l_values)


def test_process_peaks_parallel_passes_wavelength_specific_n2(monkeypatch):
    captured_n2 = []

    def fake_precompute_sample_terms(
        wavelength_array,
        n2,
        n2_array,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        zb,
        thickness,
        sample_width_m,
        sample_length_m,
        optics_mode,
        theta_initial_deg,
        cor_angle_deg,
        psi_z_deg,
        R_z_R_y,
        R_ZY_n,
        P0,
    ):
        captured_n2.append(np.asarray(n2_array, dtype=np.complex128).copy())
        n_samp = beam_x_array.size
        return (
            np.eye(3, dtype=np.float64),
            np.zeros((n_samp, diffraction._SAMPLE_COLS), dtype=np.float64),
            np.asarray(n2_array, dtype=np.complex128).copy(),
            np.asarray(n2_array, dtype=np.complex128).copy() ** 2,
            0,
        )

    def fake_calculate_phi_precomputed(
        H,
        K,
        L,
        *_args,
        **_kwargs,
    ):
        return (
            np.empty((0, 7), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            0,
        )

    monkeypatch.setattr(
        diffraction,
        "_precompute_sample_terms",
        fake_precompute_sample_terms,
    )
    monkeypatch.setattr(
        diffraction,
        "_calculate_phi_from_precomputed",
        fake_calculate_phi_precomputed,
    )

    miller = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    wavelengths = np.array([1.0, 1.6], dtype=np.float64)  # Angstrom
    image_size = 16
    image = np.zeros((image_size, image_size), dtype=np.float64)

    diffraction.process_peaks_parallel.py_func(
        miller,
        intensities,
        image_size,
        1.0,
        1.0,
        1.0,
        image,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        0.5,
        0.5,
        0.0,
        wavelengths,
        0.0,
        0.0,
        [8.0, 8.0],
        6.0,
        0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
    )

    assert captured_n2, "precompute should be called with wavelength-specific n2 values"
    expected = np.array(
        [
            IndexofRefraction(wavelengths[0] * 1.0e-10),
            IndexofRefraction(wavelengths[1] * 1.0e-10),
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(captured_n2[0], expected, rtol=1e-12, atol=0.0)


def test_process_peaks_parallel_prefers_explicit_n2_override(monkeypatch):
    captured_n2 = []

    def fake_precompute_sample_terms(
        wavelength_array,
        n2,
        n2_array,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        zb,
        thickness,
        sample_width_m,
        sample_length_m,
        optics_mode,
        theta_initial_deg,
        cor_angle_deg,
        psi_z_deg,
        R_z_R_y,
        R_ZY_n,
        P0,
    ):
        captured_n2.append(np.asarray(n2_array, dtype=np.complex128).copy())
        n_samp = beam_x_array.size
        return (
            np.eye(3, dtype=np.float64),
            np.zeros((n_samp, diffraction._SAMPLE_COLS), dtype=np.float64),
            np.asarray(n2_array, dtype=np.complex128).copy(),
            np.asarray(n2_array, dtype=np.complex128).copy() ** 2,
            0,
        )

    def fake_calculate_phi_precomputed(*_args, **_kwargs):
        return (
            np.empty((0, 7), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            0,
        )

    monkeypatch.setattr(diffraction, "_precompute_sample_terms", fake_precompute_sample_terms)
    monkeypatch.setattr(
        diffraction, "_calculate_phi_from_precomputed", fake_calculate_phi_precomputed
    )

    override = np.array([1.0 + 0.1j, 1.0 + 0.2j], dtype=np.complex128)
    diffraction.process_peaks_parallel.py_func(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        16,
        1.0,
        1.0,
        1.0,
        np.zeros((16, 16), dtype=np.float64),
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        0.5,
        0.5,
        0.0,
        np.array([1.0, 1.6], dtype=np.float64),
        0.0,
        0.0,
        [8.0, 8.0],
        6.0,
        0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
        n2_sample_array_override=override,
    )

    np.testing.assert_allclose(captured_n2[0], override, rtol=0.0, atol=0.0)


def test_process_peaks_parallel_compiled_allows_missing_n2_override():
    miller = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    image_size = 16
    image = np.zeros((image_size, image_size), dtype=np.float64)

    result = diffraction.process_peaks_parallel(
        miller,
        intensities,
        image_size,
        1.0,
        1.0,
        1.0,
        image,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        0.5,
        0.5,
        0.0,
        np.array([1.0], dtype=np.float64),
        0.0,
        0.0,
        [8.0, 8.0],
        6.0,
        0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
        collect_hit_tables=False,
    )

    assert isinstance(result, tuple)
    assert len(result) == 6
    assert result[0].shape == (image_size, image_size)


def test_build_intersection_cache_uses_best_sample_only_and_drops_invalid_rows(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    hit_tables = [
        np.array([[1.0, 10.0, 20.0, 30.0, 1.0, 0.0, 1.0]], dtype=np.float64),
        np.array([[2.0, 11.0, 21.0, 31.0, 1.0, 0.0, 1.0]], dtype=np.float64),
    ]
    cache = diffraction.build_intersection_cache(
        hit_tables,
        4.0,
        7.0,
        beam_x_array=np.array([0.0, 5.0, 10.0], dtype=np.float64),
        beam_y_array=np.array([0.0, 10.0, 20.0], dtype=np.float64),
        theta_array=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        phi_array=np.array([0.0, 2.0, 4.0], dtype=np.float64),
        wavelength_array=np.array([1.0, 1.5, 2.0], dtype=np.float64),
        best_sample_indices_out=np.array([0, -1], dtype=np.int64),
    )

    assert len(cache) == 1
    first = np.asarray(cache[0], dtype=np.float64)

    np.testing.assert_allclose(first[:, 9:14], np.array([[-5.0, -10.0, -1.0, -2.0, -0.5]]))
    np.testing.assert_allclose(first[:, 14:], np.array([[0.0, 0.0, 0.0]]))


def test_build_intersection_cache_preserves_all_specular_sampled_rows(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    hit_tables = [
        np.array(
            [
                [1.0, 10.0, 10.0, 0.0, 0.0, 0.0, 3.0],
                [1.0, 20.0, 21.0, 0.0, 0.0, 0.0, 3.0],
                [1.0, 31.0, 30.0, 0.0, 0.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        )
    ]
    cache = diffraction.build_intersection_cache(hit_tables, 4.0, 7.0)

    assert len(cache) == 3
    np.testing.assert_allclose(
        np.vstack([np.asarray(table, dtype=np.float64)[0, 2:4] for table in cache]),
        np.array([[10.0, 10.0], [20.0, 21.0], [31.0, 30.0]], dtype=np.float64),
    )


def test_build_intersection_cache_keeps_explicit_zero_offset_representative_without_sample(
    monkeypatch,
):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    row = np.full(cache_schema.HIT_ROW_WITH_CONTEXT_WIDTH, np.nan, dtype=np.float64)
    row[:10] = [0.0, 30.0, 40.0, -0.25, 1.0, 0.0, 2.0, 7.0, 3.0, np.nan]
    row[cache_schema.HIT_ROW_COL_BEAM_X_OFFSET] = 0.0
    row[cache_schema.HIT_ROW_COL_BEAM_Y_OFFSET] = 0.0
    row[cache_schema.HIT_ROW_COL_THETA_OFFSET] = 0.0
    row[cache_schema.HIT_ROW_COL_PHI_OFFSET] = 0.0
    row[cache_schema.HIT_ROW_COL_WAVELENGTH_OFFSET] = 0.0

    cache = diffraction.build_branch_representative_intersection_cache(
        [row.reshape(1, -1)],
        4.0,
        7.0,
        beam_x_array=np.array([10.0, 20.0], dtype=np.float64),
        beam_y_array=np.array([11.0, 21.0], dtype=np.float64),
        theta_array=np.array([0.1, 0.2], dtype=np.float64),
        phi_array=np.array([0.3, 0.4], dtype=np.float64),
        wavelength_array=np.array([1.52, 1.56], dtype=np.float64),
        best_sample_indices_out=np.array([-1], dtype=np.int64),
    )

    assert len(cache) == 1
    cache_row = np.asarray(cache[0], dtype=np.float64)[0]
    assert float(cache_row[cache_schema.CACHE_COL_INTENSITY]) == pytest.approx(0.0)
    np.testing.assert_allclose(cache_row[9:14], np.zeros(5, dtype=np.float64))
    assert np.isnan(cache_row[cache_schema.CACHE_COL_BEST_SAMPLE_INDEX])
    np.testing.assert_allclose(cache_row[14:16], np.array([7.0, 3.0], dtype=np.float64))


def test_build_intersection_cache_preserves_all_non_specular_sampled_rows(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    hit_tables = [
        np.array(
            [
                [1.0, 9.0, 10.0, 0.0, 1.0, 0.0, 2.0],
                [1.0, 11.0, 12.0, 0.0, 1.0, 0.0, 2.0],
                [1.0, 13.0, 11.0, 0.0, 1.0, 0.0, 2.0],
                [1.0, 48.0, 50.0, 0.0, 1.0, 0.0, 2.0],
                [1.0, 50.0, 49.0, 0.0, 1.0, 0.0, 2.0],
                [1.0, 53.0, 52.0, 0.0, 1.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
    ]
    cache = diffraction.build_intersection_cache(hit_tables, 4.0, 7.0)

    assert len(cache) == 6
    np.testing.assert_allclose(
        np.vstack([np.asarray(table, dtype=np.float64)[:, 2:4] for table in cache]),
        np.array(
            [
                [9.0, 10.0],
                [11.0, 12.0],
                [13.0, 11.0],
                [48.0, 50.0],
                [50.0, 49.0],
                [53.0, 52.0],
            ],
            dtype=np.float64,
        ),
    )


def test_branch_representative_intersection_cache_preserves_preselected_rows_and_provenance(
    monkeypatch,
):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)
    monkeypatch.setattr(
        diffraction,
        "_expand_intersection_cache_group_with_metadata",
        lambda *_args, **_kwargs: pytest.fail("representative cache must not expand/reselect rows"),
    )
    monkeypatch.setattr(
        diffraction,
        "_intersection_cache_selected_row_indices",
        lambda *_args, **_kwargs: pytest.fail("representative cache must not reselect rows"),
    )

    hit_tables = [
        np.array([[1.0, 13.0, 10.0, -0.3, 1.0, 0.0, 2.0, 7.0, 3.0, 1.0]], dtype=np.float64),
        np.array([[1.0, 53.0, 10.0, 0.3, 1.0, 0.0, 2.0, 7.0, 4.0, 2.0]], dtype=np.float64),
    ]

    cache = diffraction.build_branch_representative_intersection_cache(
        hit_tables,
        4.0,
        7.0,
    )

    assert len(cache) == 2
    roundtrip_rows = np.vstack(
        [
            np.asarray(table, dtype=np.float64)
            for table in diffraction.intersection_cache_to_hit_tables(cache)
        ]
    )
    np.testing.assert_allclose(roundtrip_rows, np.vstack(hit_tables))


def test_build_intersection_cache_preserves_duplicate_sampled_rows(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    hit_tables = [
        np.array(
            [
                [1.0, 10.0, 11.0, 0.0, 1.0, 0.0, 2.0],
                [1.0, 10.0, 11.0, 0.0, 1.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
    ]

    cache = diffraction.build_intersection_cache(hit_tables, 4.0, 7.0)

    assert len(cache) == 2
    np.testing.assert_allclose(
        np.vstack([np.asarray(table, dtype=np.float64)[0, 2:4] for table in cache]),
        np.array([[10.0, 11.0], [10.0, 11.0]], dtype=np.float64),
    )


def test_branch_representative_cache_keeps_exact_hkl_rows_separate_when_qr_set_mode_is_disabled(
    monkeypatch,
):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    nan = np.nan
    hit_tables = [
        np.array([[1.0, 10.0, 11.0, 0.0, 1.0, 0.0, 2.0, nan, nan, 0.0]], dtype=np.float64),
        np.array([[1.0, 20.0, 21.0, 0.0, 0.0, 1.0, 2.0, nan, nan, 1.0]], dtype=np.float64),
    ]

    cache = diffraction.build_branch_representative_intersection_cache(hit_tables, 4.0, 7.0)

    assert len(cache) == 2
    hkls = np.vstack([np.asarray(table, dtype=np.float64)[0, 6:9] for table in cache])
    np.testing.assert_allclose(hkls, np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float64))


def test_branch_representative_cache_group_by_qr_set_flag_is_passthrough(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)
    monkeypatch.setattr(
        diffraction,
        "_expand_intersection_cache_group_with_metadata",
        lambda *_args, **_kwargs: pytest.fail(
            "group_by_qr_set must not recollapse representative rows"
        ),
    )
    monkeypatch.setattr(
        diffraction,
        "_intersection_cache_selected_row_indices",
        lambda *_args, **_kwargs: pytest.fail(
            "group_by_qr_set must not reselect representative rows"
        ),
    )

    hit_tables = [
        np.array([[1.0, 10.0, 10.0, -0.3, 1.0, 0.0, 2.0, 7.0, 0.0, 1.0]], dtype=np.float64),
        np.array([[1.0, 50.0, 10.0, 0.3, 1.0, 0.0, 2.0, 7.0, 1.0, 2.0]], dtype=np.float64),
        np.array([[1.0, 12.0, 10.0, -0.2, 0.0, 1.0, 2.0, 8.0, 0.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 52.0, 10.0, 0.2, 0.0, 1.0, 2.0, 8.0, 1.0, 3.0]], dtype=np.float64),
    ]

    cache = diffraction.build_branch_representative_intersection_cache(
        hit_tables,
        4.0,
        7.0,
        group_by_qr_set=True,
    )

    assert len(cache) == 4
    roundtrip_rows = np.vstack(
        [
            np.asarray(table, dtype=np.float64)
            for table in diffraction.intersection_cache_to_hit_tables(cache)
        ]
    )
    np.testing.assert_allclose(roundtrip_rows, np.vstack(hit_tables))


def test_build_intersection_cache_merges_specular_tables_by_nominal_l(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    hit_tables = [
        np.array([[1.0, 10.0, 10.0, 0.0, 0.0, 0.0, 2.99]], dtype=np.float64),
        np.array([[1.0, 20.0, 20.0, 0.0, 0.0, 0.0, 3.00]], dtype=np.float64),
        np.array([[1.0, 21.0, 21.0, 0.0, 0.0, 0.0, 3.01]], dtype=np.float64),
        np.array([[1.0, 40.0, 40.0, 0.0, 0.0, 0.0, 3.02]], dtype=np.float64),
    ]

    cache = diffraction.build_intersection_cache(hit_tables, 4.0, 7.0)

    assert len(cache) == 4
    for table in cache:
        assert int(np.rint(float(np.asarray(table, dtype=np.float64)[0, 8]))) == 3


def test_build_intersection_cache_merges_non_specular_tables_by_nominal_peak_and_side(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    hit_tables = [
        np.array([[1.0, 9.0, 10.0, 0.0, 1.0, 0.0, 1.98]], dtype=np.float64),
        np.array([[1.0, 11.0, 10.0, 0.0, 1.0, 0.0, 2.00]], dtype=np.float64),
        np.array([[1.0, 13.0, 10.0, 0.0, 1.0, 0.0, 2.02]], dtype=np.float64),
        np.array([[1.0, 47.0, 10.0, 0.0, 1.0, 0.0, 1.98]], dtype=np.float64),
        np.array([[1.0, 50.0, 10.0, 0.0, 1.0, 0.0, 2.00]], dtype=np.float64),
        np.array([[1.0, 53.0, 10.0, 0.0, 1.0, 0.0, 2.02]], dtype=np.float64),
    ]

    cache = diffraction.build_intersection_cache(hit_tables, 4.0, 7.0)

    assert len(cache) == 6
    for table in cache:
        assert np.asarray(table, dtype=np.float64).shape == (1, 17)
    for table in cache:
        assert int(np.rint(float(np.asarray(table, dtype=np.float64)[0, 8]))) == 2


def test_build_intersection_cache_skips_empty_tables(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    hit_tables = [
        np.empty((0, 7), dtype=np.float64),
        np.array([[1.0, 20.0, 21.0, 0.0, 0.0, 0.0, 3.0]], dtype=np.float64),
        np.empty((0, 7), dtype=np.float64),
    ]

    cache = diffraction.build_intersection_cache(hit_tables, 4.0, 7.0)

    assert len(cache) == 1
    table = np.asarray(cache[0], dtype=np.float64)
    assert table.shape == (1, 17)
    np.testing.assert_allclose(table[0, 2:4], np.array([20.0, 21.0]))


def test_build_intersection_cache_log_records_extended_cache_metadata(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: True)
    monkeypatch.setattr(
        diffraction,
        "_resolve_intersection_cache_log_root",
        lambda: tmp_path,
    )

    cache = diffraction.build_intersection_cache(
        [
            np.array([[1.0, 10.0, 10.0, 0.0, 0.0, 0.0, 2.99]], dtype=np.float64),
            np.array([[1.0, 20.0, 20.0, 0.0, 0.0, 0.0, 3.00]], dtype=np.float64),
            np.array([[1.0, 21.0, 21.0, 0.0, 0.0, 0.0, 3.01]], dtype=np.float64),
        ],
        4.0,
        7.0,
    )

    assert len(cache) == 3
    log_root = tmp_path / "intersection_cache"
    log_dirs = list(log_root.glob("intersection_cache_*"))
    assert len(log_dirs) == 1
    log_dir = log_dirs[0]
    metadata = json.loads((log_dir / "meta.json").read_text(encoding="utf-8"))

    assert metadata["reused"] is False
    assert metadata["rebuilt"] is True
    assert metadata["cache_action"] == "rebuilt"
    assert metadata["stale_reason"] is None
    assert metadata["cache_source"] == "build_intersection_cache"
    assert metadata["cache_provenance"]["grouping"] == "nominal_bragg_family"
    assert metadata["group_summary_count"] == 1
    assert metadata["table_count"] == 3
    assert len(metadata["cache_tables"]) == 3
    assert metadata["table_files"] == ["table_0000.npy", "table_0001.npy", "table_0002.npy"]
    assert (log_dir / "table_0000.npy").exists()

    group_summary = metadata["group_summaries"][0]
    assert group_summary["group_key"] == ["specular", 3]
    assert "q_group_key" in group_summary
    assert group_summary["row_count_before_grouping"] == 3
    assert group_summary["row_count_after_grouping"] == 3
    assert isinstance(group_summary["representative_row_indices_kept"], list)

    table_summary = metadata["table_summaries"][0]
    assert table_summary["row_count_before_grouping"] == 3
    assert table_summary["row_count_after_grouping"] == 1
    assert isinstance(table_summary["representative_row_indices_kept"], list)

    reloaded_cache, reloaded_metadata = diffraction.load_most_recent_logged_intersection_cache(
        log_root=log_root
    )
    assert len(reloaded_cache) == 3
    np.testing.assert_allclose(
        np.asarray(reloaded_cache[0], dtype=np.float64),
        np.asarray(cache[0], dtype=np.float64),
    )
    assert reloaded_metadata["table_count"] == 3
    assert reloaded_metadata["log_dir"] == str(log_dir)


def test_build_intersection_cache_preserves_duplicate_sampled_rows(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    hit_tables = [
        np.array(
            [
                [0.5, 10.0, 11.0, -1.0, 1.0, 0.0, 1.0, np.nan, np.nan, 0.0],
                [0.5, 10.0, 11.0, -1.0, 1.0, 0.0, 1.0, np.nan, np.nan, 0.0],
            ],
            dtype=np.float64,
        )
    ]

    cache = diffraction.build_intersection_cache(hit_tables, 4.0, 7.0)

    assert len(cache) == 2
    first = np.asarray(cache[0], dtype=np.float64)
    second = np.asarray(cache[1], dtype=np.float64)
    np.testing.assert_allclose(first[:, :15], second[:, :15])
    np.testing.assert_allclose(first[:, 16:], second[:, 16:])
    assert int(first[0, 15]) != int(second[0, 15])


def test_load_most_recent_logged_intersection_cache_metadata_avoids_dir_stat_sort(
    tmp_path,
    monkeypatch,
):
    log_root = tmp_path / "intersection_cache"
    older_dir = log_root / "intersection_cache_20260422_000000_000000"
    newer_dir = log_root / "intersection_cache_20260422_000001_000000"
    older_dir.mkdir(parents=True)
    newer_dir.mkdir(parents=True)
    (older_dir / "meta.json").write_text(json.dumps({"created_at": "older"}), encoding="utf-8")
    (newer_dir / "meta.json").write_text(json.dumps({"created_at": "newer"}), encoding="utf-8")
    real_stat = Path.stat

    def fail_log_dir_stat(path, *args, **kwargs):
        if path.parent == log_root and path.name.startswith("intersection_cache_"):
            raise AssertionError("loader should not stat-sort every cache directory")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_log_dir_stat)

    metadata = diffraction.load_most_recent_logged_intersection_cache_metadata(log_root=log_root)

    assert metadata["created_at"] == "newer"
    assert metadata["log_dir"] == str(newer_dir)


def test_load_most_recent_logged_intersection_cache_avoids_dir_stat_sort(
    tmp_path,
    monkeypatch,
):
    log_root = tmp_path / "intersection_cache"
    older_dir = log_root / "intersection_cache_20260422_000000_000000"
    newer_dir = log_root / "intersection_cache_20260422_000001_000000"
    older_dir.mkdir(parents=True)
    newer_dir.mkdir(parents=True)
    (older_dir / "meta.json").write_text(
        json.dumps({"table_files": ["table_0000.npy"]}),
        encoding="utf-8",
    )
    (newer_dir / "meta.json").write_text(
        json.dumps({"table_files": ["table_0000.npy"]}),
        encoding="utf-8",
    )
    np.save(older_dir / "table_0000.npy", np.asarray([[1.0, 2.0]], dtype=np.float64))
    np.save(newer_dir / "table_0000.npy", np.asarray([[3.0, 4.0]], dtype=np.float64))
    real_stat = Path.stat

    def fail_log_dir_stat(path, *args, **kwargs):
        if path.parent == log_root and path.name.startswith("intersection_cache_"):
            raise AssertionError("loader should not stat-sort every cache directory")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_log_dir_stat)

    cache, metadata = diffraction.load_most_recent_logged_intersection_cache(log_root=log_root)

    assert metadata["log_dir"] == str(newer_dir)
    assert len(cache) == 1
    np.testing.assert_allclose(cache[0], np.asarray([[3.0, 4.0]], dtype=np.float64))


def test_load_most_recent_logged_intersection_cache_scans_past_empty_recent_dirs(
    tmp_path,
    monkeypatch,
):
    log_root = tmp_path / "intersection_cache"
    valid_dir = log_root / "intersection_cache_20260422_000000_000000"
    valid_dir.mkdir(parents=True)
    (valid_dir / "meta.json").write_text(
        json.dumps({"table_files": ["table_0000.npy"]}),
        encoding="utf-8",
    )
    np.save(valid_dir / "table_0000.npy", np.asarray([[1.0, 2.0]], dtype=np.float64))
    for index in range(1, 40):
        empty_dir = log_root / f"intersection_cache_20260422_0000{index:02d}_000000"
        empty_dir.mkdir(parents=True)
        (empty_dir / "meta.json").write_text(
            json.dumps({"table_files": ["missing.npy"]}),
            encoding="utf-8",
        )
    real_stat = Path.stat

    def fail_log_dir_stat(path, *args, **kwargs):
        if path.parent == log_root and path.name.startswith("intersection_cache_"):
            raise AssertionError("loader should not stat-sort every cache directory")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_log_dir_stat)

    cache, metadata = diffraction.load_most_recent_logged_intersection_cache(log_root=log_root)

    assert metadata["log_dir"] == str(valid_dir)
    assert len(cache) == 1
    np.testing.assert_allclose(cache[0], np.asarray([[1.0, 2.0]], dtype=np.float64))


def test_precompute_sample_terms_rejects_hits_outside_finite_sample_bounds(monkeypatch):
    monkeypatch.setattr(
        diffraction,
        "_build_sample_rotation",
        lambda *_args: (
            np.eye(3, dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        diffraction,
        "intersect_line_plane",
        lambda *_args: (0.6, 0.0, 0.0, True),
    )

    _, sample_terms, *_ = diffraction._precompute_sample_terms.py_func(
        np.array([1.0], dtype=np.float64),
        1.0 + 0.0j,
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        0.0,
        0.0,
        1.0,
        0.0,
        diffraction.OPTICS_MODE_EXACT,
        0.0,
        0.0,
        0.0,
        np.eye(3, dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )

    assert float(sample_terms[0, diffraction._SAMPLE_COL_VALID]) == 0.0


def test_calculate_phi_from_precomputed_uses_pixel_size(monkeypatch):
    monkeypatch.setattr(
        diffraction,
        "_nominal_reflection_visible",
        lambda *_args, **_kwargs: (True, 0, False),
    )
    monkeypatch.setattr(
        diffraction,
        "solve_q",
        lambda *_args, **_kwargs: (np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), 0),
    )
    monkeypatch.setattr(
        diffraction,
        "intersect_line_plane",
        lambda *_args, **_kwargs: (2.0e-4, 0.0, 0.0, True),
    )

    sample_terms = np.zeros((1, diffraction._SAMPLE_COLS), dtype=np.float64)
    sample_terms[0, diffraction._SAMPLE_COL_VALID] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_K_SCAT] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_K0] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_TI2] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_L_IN] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_N2_REAL] = 1.0

    common_args = (
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        np.zeros((16, 16), dtype=np.float64),
        16,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        np.array([8.0, 8.0], dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        sample_terms,
        np.array([1.0 + 0.0j], dtype=np.complex128),
        0,
        0,
        np.zeros((1, 1, 5), dtype=np.float64),
        np.zeros(1, dtype=np.int64),
        0,
    )

    hits_100, *_ = diffraction._calculate_phi_from_precomputed.py_func(
        *common_args,
        pixel_size_m=100e-6,
    )
    hits_200, *_ = diffraction._calculate_phi_from_precomputed.py_func(
        *common_args,
        pixel_size_m=200e-6,
    )

    assert hits_100.shape == (1, diffraction.HIT_ROW_WITH_PROVENANCE_WIDTH)
    assert hits_200.shape == (1, diffraction.HIT_ROW_WITH_PROVENANCE_WIDTH)
    assert int(np.rint(float(hits_100[0, diffraction.HIT_ROW_COL_BEST_SAMPLE_INDEX]))) == 0
    assert int(np.rint(float(hits_200[0, diffraction.HIT_ROW_COL_BEST_SAMPLE_INDEX]))) == 0
    assert float(hits_100[0, 1]) > float(hits_200[0, 1])
    np.testing.assert_allclose(float(hits_100[0, 1]), 10.0, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(float(hits_200[0, 1]), 9.0, atol=1.0e-12, rtol=0.0)


def test_calculate_phi_from_precomputed_samples_one_ring_point_using_total_ring_mass(monkeypatch):
    monkeypatch.setattr(
        diffraction,
        "_nominal_reflection_visible",
        lambda *_args, **_kwargs: (True, 0, False),
    )
    monkeypatch.setattr(
        diffraction,
        "solve_q",
        lambda *_args, **_kwargs: (
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 3.0],
                ],
                dtype=np.float64,
            ),
            0,
        ),
    )
    monkeypatch.setattr(
        diffraction,
        "_sample_q_ring_solution",
        lambda *_args, **_kwargs: (1, 4.0),
    )
    monkeypatch.setattr(
        diffraction,
        "intersect_line_plane",
        lambda *_args, **_kwargs: (0.0, 0.0, 0.0, True),
    )

    sample_terms = np.zeros((1, diffraction._SAMPLE_COLS), dtype=np.float64)
    sample_terms[0, diffraction._SAMPLE_COL_VALID] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_K_SCAT] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_K0] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_TI2] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_L_IN] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_N2_REAL] = 1.0

    hits, *_ = diffraction._calculate_phi_from_precomputed.py_func(
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        np.zeros((16, 16), dtype=np.float64),
        16,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        np.array([8.0, 8.0], dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        sample_terms,
        np.array([1.0 + 0.0j], dtype=np.complex128),
        0,
        0,
        np.zeros((1, 1, 5), dtype=np.float64),
        np.zeros(1, dtype=np.int64),
        0,
        sample_qr_ring_once=True,
    )

    assert hits.shape == (1, diffraction.HIT_ROW_WITH_PROVENANCE_WIDTH)
    assert int(np.rint(float(hits[0, diffraction.HIT_ROW_COL_BEST_SAMPLE_INDEX]))) == 0
    np.testing.assert_allclose(float(hits[0, 0]), 8.0, atol=1.0e-12, rtol=0.0)


def test_resolve_index_of_refraction_uses_cif_when_available():
    pytest.importorskip("Dans_Diffraction")
    pytest.importorskip("xraydb")
    cif_n2 = resolve_index_of_refraction(1.54e-10, cif_path="tests/Diffuse/PbI2_2H.cif")
    default_n2 = IndexofRefraction(1.54e-10)
    assert not np.isclose(cif_n2.real, default_n2.real, rtol=1e-9, atol=0.0)


def test_debug_detector_paths_ignores_cor_angle_for_theta_i():
    common_kwargs = dict(
        beam_x_array=np.array([0.0], dtype=np.float64),
        beam_y_array=np.array([0.0], dtype=np.float64),
        theta_array=np.array([0.0], dtype=np.float64),
        phi_array=np.array([0.0], dtype=np.float64),
        theta_initial_deg=6.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        zb=0.0,
        zs=0.0,
        Distance_CoR_to_Detector=0.075,
        gamma_deg=0.0,
        Gamma_deg=0.0,
    )

    out_cor_0 = diffraction.debug_detector_paths(cor_angle_deg=0.0, **common_kwargs)
    out_cor_5 = diffraction.debug_detector_paths(cor_angle_deg=5.0, **common_kwargs)

    np.testing.assert_allclose(out_cor_0, out_cor_5, atol=1e-12, rtol=0.0)


def test_build_sample_rotation_psi_z_yaws_cor_axis():
    r_z_r_y = np.eye(3, dtype=np.float64)
    r_zy_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    p0 = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    _, n_surf_0, _ = diffraction._build_sample_rotation.py_func(
        10.0,
        20.0,
        0.0,
        r_z_r_y,
        r_zy_n,
        p0,
    )
    _, n_surf_90, _ = diffraction._build_sample_rotation.py_func(
        10.0,
        20.0,
        90.0,
        r_z_r_y,
        r_zy_n,
        p0,
    )

    assert np.isclose(float(n_surf_0[2]), float(n_surf_90[2]), atol=1e-12, rtol=0.0)
    assert abs(float(n_surf_90[0])) > abs(float(n_surf_0[0])) + 1e-6
    assert abs(float(n_surf_90[1])) < abs(float(n_surf_0[1])) - 1e-6
    assert not np.allclose(n_surf_0, n_surf_90)


def test_build_sample_rotation_zero_tilt_ignores_axis_yaw():
    r_z_r_y = np.eye(3, dtype=np.float64)
    r_zy_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    p0 = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    r_sample_0, n_surf_0, p0_rot_0 = diffraction._build_sample_rotation.py_func(
        0.0,
        20.0,
        0.0,
        r_z_r_y,
        r_zy_n,
        p0,
    )
    r_sample_45, n_surf_45, p0_rot_45 = diffraction._build_sample_rotation.py_func(
        0.0,
        20.0,
        45.0,
        r_z_r_y,
        r_zy_n,
        p0,
    )

    np.testing.assert_allclose(r_sample_0, r_sample_45, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(n_surf_0, n_surf_45, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(p0_rot_0, p0_rot_45, atol=1e-12, rtol=0.0)


def test_intersect_infinite_line_plane_allows_negative_t():
    p0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # Plane y=1, while the direction initially points toward -y.
    k_vec = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    p_plane = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n_plane = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    ix, iy, iz, valid = diffraction.intersect_infinite_line_plane(p0, k_vec, p_plane, n_plane)
    assert valid
    np.testing.assert_allclose([ix, iy, iz], [0.0, 1.0, 0.0], atol=1e-12, rtol=0.0)


def test_intersect_infinite_line_plane_parallel_projects_to_plane():
    p0 = np.array([2.0, 0.0, -3.0], dtype=np.float64)
    # Direction parallel to plane y=1.
    k_vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    p_plane = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n_plane = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    ix, iy, iz, valid = diffraction.intersect_infinite_line_plane(p0, k_vec, p_plane, n_plane)
    assert valid
    np.testing.assert_allclose([ix, iy, iz], [2.0, 1.0, -3.0], atol=1e-12, rtol=0.0)
