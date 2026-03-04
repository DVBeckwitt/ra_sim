import numpy as np

from ra_sim.simulation import diffraction
from ra_sim.utils.calculations import IndexofRefraction


def test_process_peaks_parallel_skips_negative_l(monkeypatch):
    called_l_values = []

    def fake_calculate_phi_precomputed(
        H,
        K,
        L,
        av,
        cv,
        image,
        image_size,
        reflection_intensity,
        sigma_rad,
        gamma_pv,
        eta_pv,
        debye_x,
        debye_y,
        center,
        R_sample,
        n_det_rot,
        Detector_Pos,
        e1_det,
        e2_det,
        sample_terms,
        n2_samp_array,
        eps2_array,
        best_idx,
        save_flag,
        q_data,
        q_count,
        i_peaks_index,
        record_status=False,
        thickness=0.0,
        optics_mode=0,
        solve_q_steps=1000,
        solve_q_rel_tol=5e-4,
        solve_q_mode=0,
        forced_sample_idx=-1,
    ):
        called_l_values.append(float(L))
        return (
            np.empty((0, 7), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            0,
        )

    monkeypatch.setattr(
        diffraction,
        "_calculate_phi_from_precomputed",
        fake_calculate_phi_precomputed,
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

    assert called_l_values == [1.0]


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
        optics_mode,
        theta_initial_deg,
        cor_angle_deg,
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
        av,
        cv,
        image,
        image_size,
        reflection_intensity,
        sigma_rad,
        gamma_pv,
        eta_pv,
        debye_x,
        debye_y,
        center,
        R_sample,
        n_det_rot,
        Detector_Pos,
        e1_det,
        e2_det,
        sample_terms,
        n2_samp_array,
        eps2_array,
        best_idx,
        save_flag,
        q_data,
        q_count,
        i_peaks_index,
        record_status=False,
        thickness=0.0,
        optics_mode=0,
        solve_q_steps=1000,
        solve_q_rel_tol=5e-4,
        solve_q_mode=0,
        forced_sample_idx=-1,
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


def test_intersect_infinite_line_plane_allows_negative_t():
    p0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # Plane y=1, while the direction initially points toward -y.
    k_vec = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    p_plane = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n_plane = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    ix, iy, iz, valid = diffraction.intersect_infinite_line_plane(
        p0, k_vec, p_plane, n_plane
    )
    assert valid
    np.testing.assert_allclose([ix, iy, iz], [0.0, 1.0, 0.0], atol=1e-12, rtol=0.0)


def test_intersect_infinite_line_plane_parallel_projects_to_plane():
    p0 = np.array([2.0, 0.0, -3.0], dtype=np.float64)
    # Direction parallel to plane y=1.
    k_vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    p_plane = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n_plane = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    ix, iy, iz, valid = diffraction.intersect_infinite_line_plane(
        p0, k_vec, p_plane, n_plane
    )
    assert valid
    np.testing.assert_allclose([ix, iy, iz], [2.0, 1.0, -3.0], atol=1e-12, rtol=0.0)
