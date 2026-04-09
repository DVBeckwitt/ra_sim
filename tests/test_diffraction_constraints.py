import json

import numpy as np

from ra_sim.simulation import diffraction
from ra_sim.utils.calculations import IndexofRefraction, resolve_index_of_refraction


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
        pixel_size_m=100e-6,
        forced_sample_idx=-1,
        sample_qr_ring_once=True,
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
        pixel_size_m=100e-6,
        forced_sample_idx=-1,
        sample_qr_ring_once=True,
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
    monkeypatch.setattr(diffraction, "_calculate_phi_from_precomputed", fake_calculate_phi_precomputed)

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


def test_build_intersection_cache_prefers_per_peak_sample_indices(monkeypatch):
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
        single_sample_indices=np.array([2, 2], dtype=np.int64),
        best_sample_indices_out=np.array([0, -1], dtype=np.int64),
    )

    first = np.asarray(cache[0], dtype=np.float64)
    second = np.asarray(cache[1], dtype=np.float64)

    np.testing.assert_allclose(first[:, 9:], np.array([[-5.0, -10.0, -1.0, -2.0, -0.5]]))
    np.testing.assert_allclose(second[:, 9:], np.array([[5.0, 10.0, 1.0, 2.0, 0.5]]))


def test_build_intersection_cache_keeps_one_specular_representative(monkeypatch):
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

    table = np.asarray(cache[0], dtype=np.float64)
    assert table.shape == (1, 14)
    np.testing.assert_allclose(table[0, 2:4], np.array([20.0, 21.0]))


def test_build_intersection_cache_keeps_two_non_specular_representatives(monkeypatch):
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

    assert len(cache) == 2
    np.testing.assert_allclose(
        np.vstack([np.asarray(table, dtype=np.float64)[:, 2:4] for table in cache]),
        np.array(
            [
                [11.0, 12.0],
                [50.0, 49.0],
            ],
            dtype=np.float64,
        ),
    )


def test_build_intersection_cache_merges_specular_tables_by_nominal_l(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    hit_tables = [
        np.array([[1.0, 10.0, 10.0, 0.0, 0.0, 0.0, 2.99]], dtype=np.float64),
        np.array([[1.0, 20.0, 20.0, 0.0, 0.0, 0.0, 3.00]], dtype=np.float64),
        np.array([[1.0, 21.0, 21.0, 0.0, 0.0, 0.0, 3.01]], dtype=np.float64),
        np.array([[1.0, 40.0, 40.0, 0.0, 0.0, 0.0, 3.02]], dtype=np.float64),
    ]

    cache = diffraction.build_intersection_cache(hit_tables, 4.0, 7.0)

    assert len(cache) == 1
    table = np.asarray(cache[0], dtype=np.float64)
    assert table.shape == (1, 14)
    np.testing.assert_allclose(table[0, 2:4], np.array([21.0, 21.0]))
    assert int(np.rint(float(table[0, 8]))) == 3


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

    assert len(cache) == 2
    for table in cache:
        assert np.asarray(table, dtype=np.float64).shape == (1, 14)
    np.testing.assert_allclose(
        np.vstack([np.asarray(table, dtype=np.float64)[0, 2:4] for table in cache]),
        np.array(
            [
                [11.0, 10.0],
                [50.0, 10.0],
            ],
            dtype=np.float64,
        ),
    )
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
    assert table.shape == (1, 14)
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

    assert len(cache) == 1
    log_paths = list(tmp_path.glob("geometry_fit_log_*.txt"))
    assert len(log_paths) == 1
    log_text = log_paths[0].read_text(encoding="utf-8")
    metadata = json.loads(log_text[log_text.index("{"):])

    assert metadata["reused"] is False
    assert metadata["rebuilt"] is True
    assert metadata["cache_action"] == "rebuilt"
    assert metadata["stale_reason"] is None
    assert metadata["cache_source"] == "build_intersection_cache"
    assert metadata["cache_provenance"]["grouping"] == "nominal_bragg_family"
    assert metadata["group_summary_count"] == 1
    assert metadata["table_count"] == 1
    assert len(metadata["cache_tables"]) == 1

    group_summary = metadata["group_summaries"][0]
    assert group_summary["nominal_hkl"] == [0, 0, 3]
    assert "q_group_key" in group_summary
    assert group_summary["row_count_before_grouping"] == 3
    assert group_summary["row_count_after_grouping"] == 1
    assert isinstance(group_summary["representative_row_indices_kept"], list)

    table_summary = metadata["table_summaries"][0]
    assert table_summary["nominal_hkl"] == [0, 0, 3]
    assert "q_group_key" in table_summary
    assert table_summary["row_count_before_grouping"] == 3
    assert table_summary["row_count_after_grouping"] == 1
    assert isinstance(table_summary["representative_row_indices_kept"], list)


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
        diffraction.OPTICS_MODE_FAST,
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
    monkeypatch.setattr(diffraction, "_build_fast_optics_lut_row", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        diffraction,
        "_lookup_fast_optics_lut_row",
        lambda *_args, **_kwargs: (1.0, 0.0, 1.0, 0.0),
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

    assert hits_100.shape == (1, 7)
    assert hits_200.shape == (1, 7)
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
    monkeypatch.setattr(diffraction, "_build_fast_optics_lut_row", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        diffraction,
        "_lookup_fast_optics_lut_row",
        lambda *_args, **_kwargs: (1.0, 0.0, 1.0, 0.0),
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
        np.array([1.0 + 0.0j], dtype=np.complex128),
        0,
        0,
        np.zeros((1, 1, 5), dtype=np.float64),
        np.zeros(1, dtype=np.int64),
        0,
        sample_qr_ring_once=True,
    )

    assert hits.shape == (1, 7)
    np.testing.assert_allclose(float(hits[0, 0]), 8.0, atol=1.0e-12, rtol=0.0)


def test_resolve_index_of_refraction_uses_cif_when_available():
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
