from __future__ import annotations

import numpy as np

from ra_sim.simulation import diffraction


def _call_safe(intensities, *, save_flag=0, events_per_beam_phase=50, **process_kwargs):
    image_size = 8
    image = np.zeros((image_size, image_size), dtype=np.float64)
    n_samp = 1
    arr1 = np.zeros(n_samp, dtype=np.float64)
    center = np.array([image_size / 2.0, image_size / 2.0], dtype=np.float64)
    unit_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    n_detector = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    miller = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    return diffraction.process_peaks_parallel_safe(
        miller,
        np.asarray(intensities, dtype=np.float64),
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
        arr1.copy(),
        arr1.copy(),
        arr1.copy(),
        arr1.copy(),
        0.5,
        0.5,
        0.0,
        np.ones(n_samp, dtype=np.float64),
        0.0,
        0.0,
        center,
        0.0,
        0.0,
        unit_x,
        n_detector,
        save_flag=save_flag,
        events_per_beam_phase=events_per_beam_phase,
        **process_kwargs,
    )


def _fake_result():
    return (
        np.zeros((8, 8), dtype=np.float64),
        [],
        np.zeros((1, 1, 5), dtype=np.float64),
        np.zeros(1, dtype=np.int64),
        np.zeros((2, 1), dtype=np.int64),
        [],
    )


def test_source_template_cache_is_not_used_when_weighted_events_are_enabled(monkeypatch):
    diffraction._PHASE_SPACE_CACHE.clear()
    diffraction._SOURCE_TEMPLATE_CACHE.clear()
    diffraction._Q_VECTOR_CACHE.clear()
    called = 0

    def fake_build_source(*_args, **_kwargs):
        raise AssertionError("weighted event mode must not build source templates")

    def fake_kernel(*_args, **_kwargs):
        nonlocal called
        called += 1
        return _fake_result()

    monkeypatch.setattr(diffraction, "_build_source_unit_template", fake_build_source)
    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)

    _call_safe([2.0, 3.0], save_flag=0, events_per_beam_phase=50)

    assert called == 1
    stats = diffraction.get_last_process_peaks_safe_stats()
    assert stats["used_safe_cache"] is False


def test_grouped_source_expansion_path_is_not_used(monkeypatch):
    def fake_precompute(*_args, **_kwargs):
        sample_terms = np.zeros((1, diffraction._SAMPLE_COLS), dtype=np.float64)
        sample_terms[:, diffraction._SAMPLE_COL_VALID] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_K_SCAT] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_K0] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_TI2] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_L_IN] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_N2_REAL] = 1.0
        return (
            np.eye(3, dtype=np.float64),
            sample_terms,
            np.ones(1, dtype=np.complex128),
            np.ones(1, dtype=np.complex128),
            0,
        )

    def fake_solve_q(*_args, **_kwargs):
        return np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float64), 0

    def fake_project(**kwargs):
        return True, 0.0, float(kwargs["Qx"]), 0.0, float(kwargs["reflection_intensity"])

    def fail_copy_scaled_hit_table(*_args, **_kwargs):
        raise AssertionError("grouped source expansion must not run")

    monkeypatch.setattr(diffraction, "_precompute_sample_terms", fake_precompute)
    monkeypatch.setattr(diffraction, "solve_q", fake_solve_q)
    monkeypatch.setattr(diffraction, "_project_weighted_candidate", fake_project)
    monkeypatch.setattr(diffraction, "_copy_scaled_hit_table", fail_copy_scaled_hit_table)

    result = diffraction.process_peaks_parallel(
        np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
        16,
        4.0,
        7.0,
        1.54,
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
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        0.5,
        0.5,
        0.0,
        np.ones(1, dtype=np.float64),
        0.0,
        0.0,
        np.array([8.0, 8.0], dtype=np.float64),
        0.0,
        0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=5,
    )
    assert sum(np.asarray(table, dtype=np.float64).shape[0] for table in result[1]) == 5


def test_source_template_and_clustering_disabled(monkeypatch):
    with monkeypatch.context() as scoped:
        test_source_template_cache_is_not_used_when_weighted_events_are_enabled(scoped)

    with monkeypatch.context() as scoped:
        test_grouped_source_expansion_path_is_not_used(scoped)


def test_safe_wrapper_reports_source_template_cache_disabled(monkeypatch):
    def fake_kernel(*_args, **_kwargs):
        return _fake_result()

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)
    _call_safe([1.0, 1.0], save_flag=0, events_per_beam_phase=50)
    stats = diffraction.get_last_process_peaks_safe_stats()
    assert stats["used_safe_cache"] is False
