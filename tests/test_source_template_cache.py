import numpy as np

from ra_sim.simulation import diffraction
from ra_sim.utils import stacking_fault


def _call_safe(intensities, *, save_flag=0, theta_initial=6.0):
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
        float(theta_initial),
        0.0,
        unit_x,
        n_detector,
        save_flag=save_flag,
        sample_qr_ring_once=False,
    )


def test_process_peaks_parallel_safe_reuses_source_template_cache(monkeypatch):
    diffraction._PHASE_SPACE_CACHE.clear()
    diffraction._SOURCE_TEMPLATE_CACHE.clear()
    diffraction._Q_VECTOR_CACHE.clear()
    phase_build_calls = 0
    source_build_calls = 0

    def fake_build_phase(_params):
        nonlocal phase_build_calls
        phase_build_calls += 1
        return {"n_samp": 1}

    def fake_build_source(_params, _phase_entry, H, K, L, forced_idx):
        nonlocal source_build_calls
        source_build_calls += 1
        return {
            "flat_indices": np.array([0], dtype=np.int64),
            "flat_values": np.array([2.0], dtype=np.float64),
            "hit_template": np.array([[1.0, 5.0, 6.0, 0.2, H, K, L]], dtype=np.float64),
            "miss_template": np.empty((0, 3), dtype=np.float64),
            "status_template": np.zeros(1, dtype=np.int64),
            "best_sample_idx": int(forced_idx),
        }

    def fake_kernel(*_args, **_kwargs):
        raise AssertionError("kernel path should not run on cache hit")

    monkeypatch.setattr(diffraction, "_build_phase_space_entry", fake_build_phase)
    monkeypatch.setattr(diffraction, "_build_source_unit_template", fake_build_source)
    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)

    out1 = _call_safe([2.0, 3.0], save_flag=0)
    out2 = _call_safe([4.0, 6.0], save_flag=0)

    # Phase/template state is rebuilt each run; reuse now happens at Q-vector
    # level inside the source builder.
    assert phase_build_calls == 2
    assert source_build_calls == 2
    assert float(out1[0][0, 0]) == 10.0
    assert float(out2[0][0, 0]) == 20.0
    np.testing.assert_allclose(np.asarray(out1[1][0])[:, 0], [2.0])
    np.testing.assert_allclose(np.asarray(out1[1][1])[:, 0], [3.0])
    np.testing.assert_allclose(np.asarray(out2[1][0])[:, 0], [4.0])
    np.testing.assert_allclose(np.asarray(out2[1][1])[:, 0], [6.0])


def test_process_peaks_parallel_safe_bypasses_cache_for_save_flag(monkeypatch):
    diffraction._PHASE_SPACE_CACHE.clear()
    diffraction._SOURCE_TEMPLATE_CACHE.clear()
    diffraction._Q_VECTOR_CACHE.clear()
    called = 0

    def fake_kernel(*_args, **_kwargs):
        nonlocal called
        called += 1
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((2, 1), dtype=np.int64),
            [],
        )

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)

    _call_safe([1.0, 1.0], save_flag=1)

    assert called == 1


def test_process_peaks_parallel_safe_reports_reused_rays(monkeypatch):
    diffraction._PHASE_SPACE_CACHE.clear()
    diffraction._SOURCE_TEMPLATE_CACHE.clear()
    diffraction._Q_VECTOR_CACHE.clear()
    call_count = 0

    def fake_build_phase(_params):
        return {"n_samp": 3}

    def fake_build_source(_params, _phase_entry, H, K, L, forced_idx):
        nonlocal call_count
        call_count += 1
        q_hits = 0 if call_count == 1 else 3
        return {
            "flat_indices": np.array([0], dtype=np.int64),
            "flat_values": np.array([1.0], dtype=np.float64),
            "hit_template": np.array([[1.0, 5.0, 6.0, 0.2, H, K, L]], dtype=np.float64),
            "miss_template": np.empty((0, 3), dtype=np.float64),
            "status_template": np.zeros(3, dtype=np.int64),
            "best_sample_idx": int(forced_idx),
            "q_cache_hits": q_hits,
        }

    def fake_kernel(*_args, **_kwargs):
        raise AssertionError("kernel path should not run when cache path is available")

    monkeypatch.setattr(diffraction, "_build_phase_space_entry", fake_build_phase)
    monkeypatch.setattr(diffraction, "_build_source_unit_template", fake_build_source)
    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)

    _call_safe([2.0, 3.0], save_flag=0)
    stats_first = diffraction.get_last_process_peaks_safe_stats()
    _call_safe([4.0, 6.0], save_flag=0)
    stats_second = diffraction.get_last_process_peaks_safe_stats()

    assert stats_first["used_safe_cache"] is True
    assert stats_first["source_templates_built"] == 1
    assert stats_first["source_templates_reused"] == 0
    assert stats_first["rays_reused"] == 0

    assert stats_second["used_safe_cache"] is True
    assert stats_second["source_templates_built"] == 1
    assert stats_second["source_templates_reused"] == 0
    assert stats_second["rays_reused"] == 3


def test_process_qr_rods_parallel_uses_safe_peak_wrapper(monkeypatch):
    called = 0

    def fake_qr_dict_to_arrays(_qr_dict):
        return (
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64),
            np.array([2.0, 3.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            None,
        )

    def fake_safe(*_args, **_kwargs):
        nonlocal called
        called += 1
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((2, 1), dtype=np.int64),
            [],
        )

    monkeypatch.setattr(stacking_fault, "qr_dict_to_arrays", fake_qr_dict_to_arrays)
    monkeypatch.setattr(diffraction, "process_peaks_parallel_safe", fake_safe)

    out = diffraction.process_qr_rods_parallel(
        qr_dict={},
        image_size=8,
        av=1.0,
        cv=1.0,
        lambda_=1.0,
        image=np.zeros((8, 8), dtype=np.float64),
        Distance_CoR_to_Detector=1.0,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        zs=0.0,
        zb=0.0,
        n2=1.0 + 0.0j,
        beam_x_array=np.zeros(1, dtype=np.float64),
        beam_y_array=np.zeros(1, dtype=np.float64),
        theta_array=np.zeros(1, dtype=np.float64),
        phi_array=np.zeros(1, dtype=np.float64),
        sigma_pv_deg=0.5,
        gamma_pv_deg=0.5,
        eta_pv=0.0,
        wavelength_array=np.ones(1, dtype=np.float64),
        debye_x=0.0,
        debye_y=0.0,
        center=np.array([4.0, 4.0], dtype=np.float64),
        theta_initial_deg=6.0,
        cor_angle_deg=0.0,
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
    )

    assert called == 1
    assert len(out) == 7


def test_q_vector_cache_reuses_when_theta_i_returns(monkeypatch):
    diffraction._Q_VECTOR_CACHE.clear()
    diffraction._PHASE_SPACE_CACHE.clear()
    diffraction._SOURCE_TEMPLATE_CACHE.clear()
    solve_calls = 0

    def fake_solve_q(*_args, **_kwargs):
        nonlocal solve_calls
        solve_calls += 1
        return np.zeros((0, 4), dtype=np.float64), 0

    monkeypatch.setattr(diffraction, "solve_q", fake_solve_q)

    _call_safe([2.0, 3.0], save_flag=0, theta_initial=5.0)
    _call_safe([2.0, 3.0], save_flag=0, theta_initial=10.0)
    _call_safe([2.0, 3.0], save_flag=0, theta_initial=5.0)
    stats_last = diffraction.get_last_process_peaks_safe_stats()

    # First 5deg builds cache, 10deg is distinct, returning to 5deg reuses.
    assert solve_calls == 2
    assert stats_last["rays_reused"] >= 1
