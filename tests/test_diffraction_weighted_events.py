from __future__ import annotations

import numpy as np
import pytest

from ra_sim.simulation import diffraction


def _base_process_kwargs(*, miller=None, intensities=None, n_samp=1, image_size=64):
    if miller is None:
        miller = np.array([[1.0, 0.0, 1.0]], dtype=np.float64)
    if intensities is None:
        intensities = np.ones(int(np.asarray(miller).shape[0]), dtype=np.float64)
    return dict(
        miller=np.asarray(miller, dtype=np.float64),
        intensities=np.asarray(intensities, dtype=np.float64),
        image_size=image_size,
        av=4.0,
        cv=7.0,
        lambda_=1.54,
        image=np.zeros((image_size, image_size), dtype=np.float64),
        Distance_CoR_to_Detector=1.0,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        zs=0.0,
        zb=0.0,
        n2=1.0 + 0.0j,
        beam_x_array=np.zeros(n_samp, dtype=np.float64),
        beam_y_array=np.zeros(n_samp, dtype=np.float64),
        theta_array=np.zeros(n_samp, dtype=np.float64),
        phi_array=np.zeros(n_samp, dtype=np.float64),
        sigma_pv_deg=0.5,
        gamma_pv_deg=0.4,
        eta_pv=0.2,
        wavelength_array=np.ones(n_samp, dtype=np.float64),
        debye_x=0.0,
        debye_y=0.0,
        center=np.array([image_size / 2.0, image_size / 2.0], dtype=np.float64),
        theta_initial_deg=0.0,
        cor_angle_deg=0.0,
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )


def _install_simple_weighted_backend(
    monkeypatch,
    *,
    solution_map: dict[tuple[int, int, int, int], np.ndarray],
    projector=None,
):
    def fake_precompute(
        wavelength_array,
        _n2,
        n2_sample_array,
        beam_x_array,
        _beam_y_array,
        _theta_array,
        _phi_array,
        _zb,
        _thickness,
        _sample_width_m,
        _sample_length_m,
        _optics_mode,
        _theta_initial_deg,
        _cor_angle_deg,
        _psi_z_deg,
        _R_z_R_y,
        _R_ZY_n,
        _P0,
    ):
        n_samp = int(np.asarray(beam_x_array, dtype=np.float64).shape[0])
        sample_terms = np.zeros((n_samp, diffraction._SAMPLE_COLS), dtype=np.float64)
        sample_terms[:, diffraction._SAMPLE_COL_VALID] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_KX_SCAT] = np.arange(n_samp, dtype=np.float64)
        sample_terms[:, diffraction._SAMPLE_COL_K_SCAT] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_K0] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_TI2] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_L_IN] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_N2_REAL] = 1.0
        return (
            np.eye(3, dtype=np.float64),
            sample_terms,
            np.asarray(n2_sample_array, dtype=np.complex128).copy(),
            np.ones(n_samp, dtype=np.complex128),
            0,
        )

    def fake_solve_q(
        k_in_crystal,
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
        sample_idx = int(round(float(np.asarray(k_in_crystal, dtype=np.float64)[0])))
        key = (int(round(H)), int(round(K)), int(round(L)), sample_idx)
        return np.asarray(solution_map.get(key, np.empty((0, 4), dtype=np.float64)), dtype=np.float64), 0

    def default_projector(**kwargs):
        qx = float(kwargs["Qx"])
        qy = float(kwargs["Qy"])
        iq = float(kwargs["I_Q"])
        mass = (
            float(kwargs["reflection_intensity"])
            * float(kwargs["sample_weight"])
            * iq
        )
        if not np.isfinite(mass) or mass <= 0.0:
            return False, 0.0, qx, qy, mass
        return True, 0.0, qx, qy, mass

    monkeypatch.setattr(diffraction, "_precompute_sample_terms", fake_precompute)
    monkeypatch.setattr(diffraction, "solve_q", fake_solve_q)
    monkeypatch.setattr(
        diffraction,
        "_project_weighted_candidate",
        default_projector if projector is None else projector,
    )


def _flatten_hit_tables(hit_tables):
    rows = []
    for table in hit_tables:
        arr = np.asarray(table, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] > 0:
            rows.append(arr)
    if not rows:
        return np.empty((0, diffraction.HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)
    return np.vstack(rows)


def test_weighted_event_targets_are_deterministic_and_bounded():
    targets_a = diffraction._weighted_event_targets(10.0, 50, sample_idx=3)
    targets_b = diffraction._weighted_event_targets(10.0, 50, sample_idx=3)

    np.testing.assert_allclose(targets_a, targets_b)
    assert targets_a.shape == (50,)
    assert np.all(targets_a >= 0.0)
    assert np.all(targets_a < 10.0)


def test_select_weighted_event_indices_from_targets_uses_cumulative_mass():
    masses = np.array([1.0, 3.0], dtype=np.float64)
    targets = np.array([0.0, 0.999, 1.0, 3.999], dtype=np.float64)

    selected = diffraction._select_weighted_event_indices_from_targets(masses, targets)

    np.testing.assert_array_equal(selected, np.array([0, 0, 1, 1]))


def test_weighted_event_sampler_matches_mass_ratio():
    masses = np.array([1.0, 3.0], dtype=np.float64)
    targets = diffraction._weighted_event_targets(
        total_mass=float(np.sum(masses)),
        event_count=400,
        sample_idx=11,
    )

    selected = diffraction._select_weighted_event_indices_from_targets(masses, targets)
    counts = np.bincount(selected, minlength=2)
    observed = counts / counts.sum()
    np.testing.assert_allclose(observed, np.array([0.25, 0.75]), atol=0.04)


def test_weighted_event_deposit_is_total_mass_divided_by_event_count():
    assert diffraction._weighted_event_deposit(20.0, 50) == pytest.approx(0.4)
    assert diffraction._weighted_event_deposit(10.0, 5) == pytest.approx(2.0)


def test_zero_or_invalid_total_mass_emits_no_events():
    masses = np.array([0.0, 0.0], dtype=np.float64)
    targets = diffraction._weighted_event_targets(0.0, 50, sample_idx=0)

    assert targets.size == 0
    assert diffraction._select_weighted_event_indices_from_targets(masses, targets).size == 0


def test_repeated_selected_ordinal_emits_repeated_events():
    masses = np.array([100.0], dtype=np.float64)
    targets = diffraction._weighted_event_targets(100.0, 10, sample_idx=0)
    selected = diffraction._select_weighted_event_indices_from_targets(masses, targets)
    assert selected.shape == (10,)
    assert np.all(selected == 0)


def test_sampled_rows_use_constant_phase_deposit_not_candidate_mass(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 1.0], [20.0, -1.0, 0.0, 9.0]], dtype=np.float64)
        },
    )
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=5,
    )
    rows = np.asarray(result[1][0], dtype=np.float64)
    assert rows.shape[0] == 5
    assert np.allclose(rows[:, 0], 2.0)
    assert float(np.sum(rows[:, 0])) == pytest.approx(10.0)


def test_weighted_event_sampler_uses_raw_candidates_before_collapse(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 1.0], [20.0, 1.0, 0.0, 9.0]], dtype=np.float64)
        },
    )
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=80,
    )
    rows = np.asarray(result[1][0], dtype=np.float64)
    selected_columns = rows[:, 1].astype(int).tolist()
    assert 10 in selected_columns
    assert 20 in selected_columns
    assert selected_columns.count(20) > selected_columns.count(10)


def test_kernel_invalid_masses_are_filtered(monkeypatch):
    def projector(**kwargs):
        qx = float(kwargs["Qx"])
        iq = float(kwargs["I_Q"])
        mass = iq
        if qx == 1.0:
            mass = np.nan
        elif qx == 2.0:
            mass = -1.0
        elif qx == 3.0:
            mass = 0.0
        return bool(np.isfinite(mass) and mass > 0.0), 0.0, qx, 0.0, mass

    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array(
                [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0], [4.0, 0.0, 0.0, 5.0]],
                dtype=np.float64,
            )
        },
        projector=projector,
    )
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=10,
    )
    rows = np.asarray(result[1][0], dtype=np.float64)
    assert rows.shape[0] == 10
    assert set(rows[:, 1].astype(int).tolist()) == {4}


def test_events_per_beam_phase_zero_normalizes_to_one(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={(1, 0, 1, 0): np.array([[10.0, 0.0, 0.0, 5.0]], dtype=np.float64)},
    )
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=0,
    )
    rows = np.asarray(result[1][0], dtype=np.float64)
    assert rows.shape[0] == 1
    assert float(rows[0, 0]) == pytest.approx(5.0)


def test_omitted_events_per_beam_phase_defaults_to_fifty(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={(1, 0, 1, 0): np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float64)},
    )
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
    )
    rows = np.asarray(result[1][0], dtype=np.float64)
    assert rows.shape[0] == 50
    assert np.allclose(rows[:, 0], 1.0 / 50.0)


def test_sample_qr_ring_once_does_not_change_weighted_event_sampling(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={(1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 1.0], [20.0, 1.0, 0.0, 3.0]], dtype=np.float64)},
    )
    result_a = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=25,
        sample_qr_ring_once=False,
    )
    result_b = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=25,
        sample_qr_ring_once=True,
    )
    np.testing.assert_allclose(np.asarray(result_a[1][0], dtype=np.float64), np.asarray(result_b[1][0], dtype=np.float64))


def test_events_per_beam_phase_counts_draws_for_whole_phase_not_per_peak(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 1.0]], dtype=np.float64),
            (2, 0, 1, 0): np.array([[20.0, 1.0, 0.0, 1.0]], dtype=np.float64),
        },
    )
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(
            miller=np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64),
            intensities=np.array([1.0, 1.0], dtype=np.float64),
        ),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=50,
    )
    total_rows = sum(np.asarray(table, dtype=np.float64).shape[0] for table in result[1])
    assert total_rows == 50


def test_image_level_conservation_uses_constant_event_deposit(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[32.0, 0.0, 0.0, 1.0], [32.0, 0.0, 0.0, 3.0]], dtype=np.float64)
        },
    )
    image = np.zeros((64, 64), dtype=np.float64)
    kwargs = _base_process_kwargs()
    kwargs["image"] = image
    result = diffraction.process_peaks_parallel(
        **kwargs,
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=True,
        events_per_beam_phase=20,
    )
    assert result[1] == []
    assert float(np.sum(result[0])) == pytest.approx(4.0)


def test_off_detector_positive_mass_candidates_do_not_enter_weighted_event_pdf(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[500.0, 0.0, 0.0, 5.0]], dtype=np.float64),
        },
    )
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=True,
        events_per_beam_phase=10,
    )
    rows = _flatten_hit_tables(result[1])
    assert rows.shape[0] == 0
    assert float(np.sum(result[0])) == pytest.approx(0.0)


def test_off_detector_candidates_are_excluded_from_hit_row_and_image_mass(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[32.0, 0.0, 0.0, 1.0], [500.0, 0.0, 0.0, 9.0]], dtype=np.float64),
        },
    )
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=True,
        events_per_beam_phase=20,
    )
    rows = _flatten_hit_tables(result[1])
    assert rows.shape[0] == 20
    assert set(rows[:, 1].astype(int).tolist()) == {32}
    assert float(np.sum(rows[:, 0])) == pytest.approx(1.0)
    assert float(np.sum(result[0])) == pytest.approx(1.0)


def test_best_sample_indices_out_tracks_peak_with_most_emitted_events(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 1.0]], dtype=np.float64),
            (2, 0, 1, 0): np.array([[20.0, 1.0, 0.0, 1.0]], dtype=np.float64),
            (1, 0, 1, 1): np.array([[10.0, -1.0, 0.0, 1.0]], dtype=np.float64),
            (2, 0, 1, 1): np.array([[20.0, 1.0, 0.0, 1.0]], dtype=np.float64),
        },
    )
    original_targets = diffraction._weighted_event_targets

    def fake_targets(total_mass, event_count, sample_idx):
        assert total_mass == pytest.approx(2.0)
        assert event_count == 5
        if sample_idx == 0:
            return np.array([0.1, 0.4, 1.2, 1.5, 1.8], dtype=np.float64)
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    monkeypatch.setattr(diffraction, "_weighted_event_targets", fake_targets)
    best_indices = np.full(2, -1, dtype=np.int64)
    diffraction.process_peaks_parallel(
        **_base_process_kwargs(
            miller=np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64),
            intensities=np.array([1.0, 1.0], dtype=np.float64),
            n_samp=2,
        ),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=5,
        best_sample_indices_out=best_indices,
    )
    assert int(best_indices[0]) == 1
    assert int(best_indices[1]) == 0

    monkeypatch.setattr(diffraction, "_weighted_event_targets", original_targets)

    best_indices_tie = np.full(2, -1, dtype=np.int64)

    def tie_targets(_total_mass, _event_count, _sample_idx):
        return np.array([0.1, 0.4, 1.2, 1.5], dtype=np.float64)

    monkeypatch.setattr(diffraction, "_weighted_event_targets", tie_targets)
    diffraction.process_peaks_parallel(
        **_base_process_kwargs(
            miller=np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64),
            intensities=np.array([1.0, 1.0], dtype=np.float64),
            n_samp=2,
        ),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=4,
        best_sample_indices_out=best_indices_tie,
    )
    assert int(best_indices_tie[0]) == 0

    best_indices_empty = np.full(1, 99, dtype=np.int64)
    diffraction.process_peaks_parallel(
        **_base_process_kwargs(intensities=np.array([0.0], dtype=np.float64)),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=4,
        best_sample_indices_out=best_indices_empty,
    )
    assert int(best_indices_empty[0]) == -1


def test_process_peaks_parallel_impl_uses_fast_runtime_not_python_fallback(monkeypatch):
    def fail_python_fallback(*_args, **_kwargs):
        raise AssertionError("python weighted-event fallback should stay off normal runtime path")

    monkeypatch.setattr(
        diffraction,
        "_process_peaks_parallel_weighted_events_python",
        fail_python_fallback,
    )
    result = diffraction._process_peaks_parallel_impl(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=False,
        events_per_beam_phase=0,
    )
    assert len(result) == 6
    stats = diffraction.get_last_process_peaks_weighted_event_stats()
    assert stats["n_solve_q_calls"] >= 0


def test_process_peaks_parallel_uses_fast_runtime_not_python_fallback(monkeypatch):
    def fail_python_fallback(*_args, **_kwargs):
        raise AssertionError("python weighted-event fallback should stay off normal runtime path")

    monkeypatch.setattr(
        diffraction,
        "_process_peaks_parallel_weighted_events_python",
        fail_python_fallback,
    )
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=False,
        events_per_beam_phase=1,
    )
    assert len(result) == 6


def test_process_peaks_parallel_safe_uses_fast_runtime_not_python_fallback(monkeypatch):
    def fail_python_fallback(*_args, **_kwargs):
        raise AssertionError("python weighted-event fallback should stay off normal runtime path")

    monkeypatch.setattr(
        diffraction,
        "_process_peaks_parallel_weighted_events_python",
        fail_python_fallback,
    )
    result = diffraction.process_peaks_parallel_safe(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=False,
        events_per_beam_phase=1,
    )
    assert len(result) == 6


def test_fast_runtime_stats_show_solve_q_reuse_and_pass_mass_consistency():
    diffraction.process_peaks_parallel(
        **_base_process_kwargs(n_samp=2),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=False,
        events_per_beam_phase=5,
    )
    stats = diffraction.get_last_process_peaks_weighted_event_stats()
    assert stats["n_solve_q_calls"] == 1
    assert stats["n_project_candidate_calls"] > 0
    assert stats["pass1_total_mass"] == pytest.approx(stats["pass2_total_mass"])
    assert stats["time_select"] <= stats["time_solve_q"] + stats["time_project"] + 1.0


def test_one_peak_can_receive_all_events_without_truncation(monkeypatch):
    n_samp = 3
    event_count = 7
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            (1, 0, 1, 1): np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            (1, 0, 1, 2): np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        },
    )
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(n_samp=n_samp),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=event_count,
    )
    rows = np.asarray(result[1][0], dtype=np.float64)
    assert rows.shape[0] == n_samp * event_count


def test_duplicate_selection_preserves_hit_rows_cache_rows_image_and_best_sample(monkeypatch):
    diffraction._set_last_intersection_cache([])
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={(1, 0, 1, 0): np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float64)},
    )
    best_indices = np.full(1, -1, dtype=np.int64)
    result = diffraction.process_peaks_parallel_safe(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=True,
        events_per_beam_phase=5,
        best_sample_indices_out=best_indices,
    )
    rows = np.asarray(result[1][0], dtype=np.float64)
    views = diffraction.get_last_intersection_cache_views()
    assert rows.shape[0] == 5
    assert len(views["sampled_event_rows"]) == 5
    assert float(np.sum(result[0])) == pytest.approx(1.0)
    assert int(best_indices[0]) == 0


def test_representative_cache_prefers_closest_branch_ray_even_if_low_mass(monkeypatch):
    diffraction._set_last_intersection_cache([])
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 1.0]], dtype=np.float64),
            (1, 0, 1, 1): np.array([[20.0, -1.0, 0.0, 100.0]], dtype=np.float64),
        },
    )
    kwargs = _base_process_kwargs(n_samp=2)
    kwargs["theta_array"] = np.array([0.01, 1.0], dtype=np.float64)
    kwargs["phi_array"] = np.array([0.01, 1.0], dtype=np.float64)
    diffraction.process_peaks_parallel_safe(
        **kwargs,
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=1,
    )
    views = diffraction.get_last_intersection_cache_views()
    rep_rows = np.vstack([np.asarray(table, dtype=np.float64) for table in views["branch_representative_rows"]])
    sampled_rows = np.vstack([np.asarray(table, dtype=np.float64) for table in views["sampled_event_rows"]])
    assert float(rep_rows[0, 4]) == pytest.approx(1.0)
    assert float(rep_rows[0, 16]) == pytest.approx(0.0)
    assert float(np.max(sampled_rows[:, 4])) == pytest.approx(100.0)


def test_qr_set_detection_uses_representative_rows_not_sampled_rows(monkeypatch):
    diffraction._set_last_intersection_cache([])
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 99.0], [50.0, 1.0, 0.0, 1.0]], dtype=np.float64)
        },
    )
    diffraction.process_peaks_parallel_safe(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=1,
    )
    views = diffraction.get_last_intersection_cache_views()
    assert len(views["sampled_event_rows"]) == 1
    assert len(views["branch_representative_rows"]) == 2


def test_sampled_event_rows_preserve_duplicates_and_representatives_do_not_replace_them(monkeypatch):
    diffraction._set_last_intersection_cache([])
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={(1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 1.0]], dtype=np.float64)},
    )
    diffraction.process_peaks_parallel_safe(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=3,
    )
    views = diffraction.get_last_intersection_cache_views()
    assert len(views["sampled_event_rows"]) == 3
    assert len(views["branch_representative_rows"]) == 1
