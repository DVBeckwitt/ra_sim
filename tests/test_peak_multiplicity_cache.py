import numpy as np

from ra_sim.simulation import diffraction


def _run_process(
    miller: np.ndarray,
    intensities: np.ndarray,
    *,
    single_sample_indices: np.ndarray | None = None,
    n_samp: int = 2,
):
    image_size = 16
    image = np.zeros((image_size, image_size), dtype=np.float64)

    beam_x = np.zeros(n_samp, dtype=np.float64)
    beam_y = np.zeros(n_samp, dtype=np.float64)
    theta = np.zeros(n_samp, dtype=np.float64)
    phi = np.zeros(n_samp, dtype=np.float64)
    wavelength = np.full(n_samp, 1.0, dtype=np.float64)

    return diffraction.process_peaks_parallel.py_func(
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
        beam_x,
        beam_y,
        theta,
        phi,
        0.5,
        0.5,
        0.0,
        wavelength,
        0.0,
        0.0,
        [8.0, 8.0],
        6.0,
        0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
        single_sample_indices=single_sample_indices,
    )


def test_process_peaks_parallel_reuses_duplicate_gr_gz_sf(monkeypatch):
    call_count = 0

    def fake_calculate_phi(
        H,
        K,
        L,
        av,
        cv,
        wavelength_array,
        image,
        image_size,
        gamma_rad,
        Gamma_rad,
        chi_rad,
        psi_rad,
        zs,
        zb,
        n2,
        n2_array,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        reflection_intensity,
        sigma_rad,
        gamma_pv,
        eta_pv,
        debye_x,
        debye_y,
        center,
        theta_initial_deg,
        cor_angle_deg,
        R_x_detector,
        R_z_detector,
        n_det_rot,
        Detector_Pos,
        e1_det,
        e2_det,
        R_z_R_y,
        R_ZY_n,
        P0,
        unit_x,
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
        nonlocal call_count
        call_count += 1
        return (
            np.array([[10.0, 5.0, 6.0, 0.2, H, K, L]], dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            0,
        )

    monkeypatch.setattr(diffraction, "calculate_phi", fake_calculate_phi)

    # (1,0,1) and (0,1,1) have identical Gr and Gz for hexagonal metric.
    miller = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    intensities = np.array([2.5, 2.5], dtype=np.float64)

    _, hit_tables, *_ = _run_process(miller, intensities)

    assert call_count == 1
    first = np.asarray(hit_tables[0])
    second = np.asarray(hit_tables[1])
    assert first.shape == (1, 7)
    assert second.shape == (1, 7)

    # Cached hit keeps geometry/intensity columns but updates HKL for the row.
    np.testing.assert_allclose(first[0, 0:4], second[0, 0:4])
    np.testing.assert_allclose(first[0, 4:7], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(second[0, 4:7], [0.0, 1.0, 1.0])


def test_process_peaks_parallel_cache_respects_forced_sample_index(monkeypatch):
    call_count = 0

    def fake_calculate_phi(
        H,
        K,
        L,
        av,
        cv,
        wavelength_array,
        image,
        image_size,
        gamma_rad,
        Gamma_rad,
        chi_rad,
        psi_rad,
        zs,
        zb,
        n2,
        n2_array,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        reflection_intensity,
        sigma_rad,
        gamma_pv,
        eta_pv,
        debye_x,
        debye_y,
        center,
        theta_initial_deg,
        cor_angle_deg,
        R_x_detector,
        R_z_detector,
        n_det_rot,
        Detector_Pos,
        e1_det,
        e2_det,
        R_z_R_y,
        R_ZY_n,
        P0,
        unit_x,
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
        nonlocal call_count
        call_count += 1
        return (
            np.array([[1.0, 4.0, 7.0, 0.3, H, K, L]], dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            int(forced_sample_idx),
        )

    monkeypatch.setattr(diffraction, "calculate_phi", fake_calculate_phi)

    miller = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    intensities = np.array([2.5, 2.5], dtype=np.float64)
    single_sample_indices = np.array([0, 1], dtype=np.int64)

    _run_process(
        miller,
        intensities,
        single_sample_indices=single_sample_indices,
    )

    # Identical Gr/Gz/SF should not be reused when forced sample differs.
    assert call_count == 2


def test_process_peaks_parallel_cache_matches_uncached_image(monkeypatch):
    def fake_calculate_phi(
        H,
        K,
        L,
        av,
        cv,
        wavelength_array,
        image,
        image_size,
        gamma_rad,
        Gamma_rad,
        chi_rad,
        psi_rad,
        zs,
        zb,
        n2,
        n2_array,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        reflection_intensity,
        sigma_rad,
        gamma_pv,
        eta_pv,
        debye_x,
        debye_y,
        center,
        theta_initial_deg,
        cor_angle_deg,
        R_x_detector,
        R_z_detector,
        n_det_rot,
        Detector_Pos,
        e1_det,
        e2_det,
        R_z_R_y,
        R_ZY_n,
        P0,
        unit_x,
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
        gr = 4.0 * np.pi / av * np.sqrt((H * H + H * K + K * K) / 3.0)
        gz = 2.0 * np.pi * (L / cv)
        val = reflection_intensity * (gr + gz + 1.0)
        image[3, 4] += val
        return (
            np.array([[val, 4.0, 3.0, 0.2, H, K, L]], dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            0,
        )

    monkeypatch.setattr(diffraction, "calculate_phi", fake_calculate_phi)

    miller = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    intensities = np.array([2.5, 2.5], dtype=np.float64)

    cached = _run_process(miller, intensities, n_samp=1)
    uncached = _run_process(
        miller,
        intensities,
        n_samp=1,
        single_sample_indices=np.array([0, 1], dtype=np.int64),
    )

    image_cached, hits_cached, *_ = cached
    image_uncached, hits_uncached, *_ = uncached
    np.testing.assert_allclose(image_cached, image_uncached)
    np.testing.assert_allclose(np.asarray(hits_cached[0]), np.asarray(hits_uncached[0]))
    np.testing.assert_allclose(np.asarray(hits_cached[1]), np.asarray(hits_uncached[1]))
