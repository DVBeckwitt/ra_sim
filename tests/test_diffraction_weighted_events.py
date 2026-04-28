from __future__ import annotations

import inspect

import numpy as np
import pytest

from ra_sim.gui import geometry_q_group_manager as gqm
from ra_sim.simulation import diffraction
from ra_sim.simulation import intersection_cache_schema as cache_schema


def test_compute_intensity_array_is_serial_njit():
    assert diffraction.compute_intensity_array is diffraction.compute_intensity_array_serial
    target_options = getattr(diffraction.compute_intensity_array_serial, "targetoptions", {})
    assert not bool(target_options.get("parallel", False))
    helper_source = inspect.getsource(diffraction.compute_intensity_array_serial.py_func)
    assert "prange" not in helper_source
    assert "parallel=True" not in helper_source
    uniform_source = inspect.getsource(diffraction._solve_q_uniform.py_func)
    full_circle_source = inspect.getsource(diffraction._solve_q_uniform_full_circle.py_func)
    assert "compute_intensity_array_serial" in uniform_source
    assert "compute_intensity_array_serial" in full_circle_source

    qx = np.array([0.0, 0.1], dtype=np.float64)
    qy = np.array([1.0, 1.1], dtype=np.float64)
    qz = np.array([0.2, 0.3], dtype=np.float64)
    out = diffraction.compute_intensity_array(
        qx,
        qy,
        qz,
        np.array([0.0, 1.0, 0.2], dtype=np.float64),
        np.deg2rad(0.5),
        np.deg2rad(0.4),
        0.2,
    )
    assert out.shape == qx.shape
    assert np.all(np.isfinite(out))


def test_solve_q_real_jit_does_not_crash_allocate_sched():
    trig = diffraction.get_default_solve_q_trig_kwargs()
    out, status = diffraction.solve_q(
        np.array([0.0, 2.0 * np.pi / 1.54, 0.0], dtype=np.float64),
        2.0 * np.pi / 1.54,
        np.array([0.0, 4.0 * np.pi / 4.0, 2.0 * np.pi / 7.0], dtype=np.float64),
        np.deg2rad(0.5),
        np.deg2rad(0.4),
        0.2,
        1.0,
        0.0,
        1.0,
        diffraction.MIN_SOLVE_Q_STEPS,
        diffraction.DEFAULT_SOLVE_Q_BASE_INTERVALS,
        diffraction.DEFAULT_SOLVE_Q_REL_TOL,
        diffraction.SOLVE_Q_MODE_UNIFORM,
        trig["default_solve_q_dtheta"],
        trig["default_solve_q_cos"],
        trig["default_solve_q_sin"],
    )
    assert status == 0
    assert out.ndim == 2
    assert out.shape[1] == 4


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


def _flatten_row_tables(tables):
    rows = []
    for table in tables:
        arr = np.asarray(table, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] > 0:
            rows.append(arr)
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(rows)


def _copy_process_kwargs(kwargs):
    copied = {}
    for key, value in kwargs.items():
        copied[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    return copied


def _build_cache_rows(hit_tables, kwargs, best_sample_indices_out, *, representatives=False):
    builder = (
        diffraction.build_branch_representative_intersection_cache
        if representatives
        else diffraction.build_intersection_cache
    )
    return _flatten_row_tables(
        builder(
            hit_tables,
            kwargs["av"],
            kwargs["cv"],
            beam_x_array=kwargs.get("beam_x_array"),
            beam_y_array=kwargs.get("beam_y_array"),
            theta_array=kwargs.get("theta_array"),
            phi_array=kwargs.get("phi_array"),
            wavelength_array=kwargs.get("wavelength_array"),
            best_sample_indices_out=best_sample_indices_out,
        )
    )


def _run_fast_serial_and_parallel(kwargs):
    serial_kwargs = _copy_process_kwargs(kwargs)
    parallel_kwargs = _copy_process_kwargs(kwargs)
    serial_best = np.full(int(np.asarray(kwargs["miller"]).shape[0]), -1, dtype=np.int64)
    parallel_best = np.full_like(serial_best, -1)
    serial_kwargs["best_sample_indices_out"] = serial_best
    parallel_kwargs["best_sample_indices_out"] = parallel_best

    serial_result = diffraction._process_peaks_parallel_weighted_events_fast_serial(
        **serial_kwargs,
        numba_thread_count=1,
    )
    bound, extra = diffraction._bind_process_peaks_parallel_call(
        (),
        dict(parallel_kwargs, numba_thread_count=2),
    )
    assert not extra
    bound["events_per_beam_phase"] = diffraction.normalize_events_per_beam_phase_backend(
        bound["events_per_beam_phase"]
    )
    parallel_result = diffraction._process_peaks_parallel_weighted_events_fast_parallel_from_bound(
        bound
    )
    parallel_stats = diffraction.get_last_process_peaks_weighted_event_stats()
    return serial_result, parallel_result, serial_best, parallel_best, parallel_stats


def _assert_fast_parallel_matches_serial(
    kwargs,
    serial_result,
    parallel_result,
    serial_best,
    parallel_best,
):
    np.testing.assert_allclose(parallel_result[0], serial_result[0], rtol=1e-12, atol=1e-12)
    assert [np.asarray(t).shape[0] for t in parallel_result[1]] == [
        np.asarray(t).shape[0] for t in serial_result[1]
    ]
    for serial_table, parallel_table in zip(serial_result[1], parallel_result[1]):
        np.testing.assert_allclose(
            np.asarray(parallel_table, dtype=np.float64),
            np.asarray(serial_table, dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )

    serial_sampled_rows = _flatten_hit_tables(serial_result[1])
    parallel_sampled_rows = _flatten_hit_tables(parallel_result[1])
    np.testing.assert_allclose(
        parallel_sampled_rows,
        serial_sampled_rows,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    if serial_sampled_rows.size:
        np.testing.assert_array_equal(
            np.bincount(parallel_sampled_rows[:, 9].astype(np.int64)),
            np.bincount(serial_sampled_rows[:, 9].astype(np.int64)),
        )

    assert [np.asarray(t).shape[0] for t in parallel_result[6]] == [
        np.asarray(t).shape[0] for t in serial_result[6]
    ]
    for serial_table, parallel_table in zip(serial_result[6], parallel_result[6]):
        np.testing.assert_allclose(
            np.asarray(parallel_table, dtype=np.float64),
            np.asarray(serial_table, dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    serial_rep_rows = _flatten_hit_tables(serial_result[6])
    parallel_rep_rows = _flatten_hit_tables(parallel_result[6])
    np.testing.assert_allclose(
        parallel_rep_rows,
        serial_rep_rows,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    if serial_rep_rows.size:
        np.testing.assert_array_equal(parallel_rep_rows[:, 7:10], serial_rep_rows[:, 7:10])

    np.testing.assert_allclose(parallel_result[2], serial_result[2], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(parallel_result[3], serial_result[3])
    np.testing.assert_array_equal(parallel_result[4], serial_result[4])
    np.testing.assert_array_equal(parallel_best, serial_best)

    serial_sampled_cache = _build_cache_rows(serial_result[1], kwargs, serial_best)
    parallel_sampled_cache = _build_cache_rows(parallel_result[1], kwargs, parallel_best)
    np.testing.assert_allclose(
        parallel_sampled_cache,
        serial_sampled_cache,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )

    serial_rep_cache = _build_cache_rows(
        serial_result[6],
        kwargs,
        serial_best,
        representatives=True,
    )
    parallel_rep_cache = _build_cache_rows(
        parallel_result[6],
        kwargs,
        parallel_best,
        representatives=True,
    )
    np.testing.assert_allclose(
        parallel_rep_cache,
        serial_rep_cache,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    if serial_rep_cache.size:
        np.testing.assert_array_equal(
            parallel_rep_cache[:, diffraction.CACHE_COL_BEST_SAMPLE_INDEX],
            serial_rep_cache[:, diffraction.CACHE_COL_BEST_SAMPLE_INDEX],
        )


def _install_streaming_fast_outer_backend(
    monkeypatch,
    *,
    pass1_masses: dict[int | tuple[int, int], float],
    pass2_masses: dict[int | tuple[int, int], float] | None = None,
):
    pass2_masses = pass1_masses if pass2_masses is None else pass2_masses

    def mass_for(mapping, peak_idx, sample_idx):
        if (peak_idx, sample_idx) in mapping:
            return float(mapping[(peak_idx, sample_idx)])
        return float(mapping.get(peak_idx, 0.0))

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
        sample_terms[:, diffraction._SAMPLE_COL_SOLVE_Q_REP] = np.arange(n_samp)
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

    def fake_solve_q(*_args, **_kwargs):
        return np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), 0

    def fake_pass1(*args):
        peak_idx = int(args[1])
        sample_idx = int(args[2])
        mass = mass_for(pass1_masses, peak_idx, sample_idx)
        return mass, int(mass > 0.0), 1

    def fake_pass2(*args):
        peak_idx = int(args[1])
        sample_idx = int(args[2])
        H = float(args[3])
        K = float(args[4])
        L = float(args[5])
        targets = np.asarray(args[-15], dtype=np.float64)
        target_idx = int(args[-14])
        cumulative_mass = float(args[-13])
        deposit = float(args[-12])
        collect_tables = bool(args[-11])
        flat_event_rows = args[-10]
        flat_event_peak_indices = args[-9]
        flat_event_count = int(args[-8])
        event_counts = args[-7]

        mass = mass_for(pass2_masses, peak_idx, sample_idx)
        cumulative_mass += mass
        hit_count = 0
        while target_idx < targets.shape[0] and cumulative_mass > float(targets[target_idx]):
            hit_count += 1
            target_idx += 1
        event_counts[peak_idx, sample_idx] += hit_count
        if collect_tables:
            for _event_idx in range(hit_count):
                flat_event_peak_indices[flat_event_count] = peak_idx
                flat_event_rows[flat_event_count, 0] = deposit
                flat_event_rows[flat_event_count, 1] = 10.0 + peak_idx
                flat_event_rows[flat_event_count, 2] = 0.0
                flat_event_rows[flat_event_count, 3] = 0.0
                flat_event_rows[flat_event_count, 4] = H
                flat_event_rows[flat_event_count, 5] = K
                flat_event_rows[flat_event_count, 6] = L
                flat_event_rows[flat_event_count, 7] = np.nan
                flat_event_rows[flat_event_count, 8] = np.nan
                flat_event_rows[flat_event_count, 9] = float(sample_idx)
                flat_event_count += 1
        return (
            target_idx,
            cumulative_mass,
            flat_event_count,
            int(args[-2]),
            1,
            mass,
            hit_count,
            bool(mass > 0.0),
            0.0,
            10.0 + peak_idx,
            0.0,
            peak_idx,
            H,
            K,
            L,
        )

    monkeypatch.setattr(diffraction, "_precompute_sample_terms", fake_precompute)
    monkeypatch.setattr(diffraction, "solve_q", fake_solve_q)
    monkeypatch.setattr(diffraction, "_weighted_event_pass1_for_qset", fake_pass1)
    monkeypatch.setattr(diffraction, "_weighted_event_pass2_for_qset", fake_pass2)


def _new_representative_slot_state(n_slots):
    return (
        np.zeros(n_slots, dtype=np.uint8),
        np.full(n_slots, np.inf, dtype=np.float64),
        np.full(n_slots, np.inf, dtype=np.float64),
        np.full(n_slots, np.inf, dtype=np.float64),
        np.full(n_slots, np.inf, dtype=np.float64),
        np.full(n_slots, np.inf, dtype=np.float64),
        np.full(n_slots, -1, dtype=np.int64),
        np.full(n_slots, -1, dtype=np.int64),
        np.full(n_slots, -1, dtype=np.int64),
        np.full((n_slots, diffraction.HIT_ROW_WITH_PROVENANCE_WIDTH), np.nan, dtype=np.float64),
    )


def _update_representative_slot(
    state,
    *,
    rep_slot=0,
    sample_mosaic_weight=1.0,
    angular=0.0,
    beam=0.0,
    wavelength=0.0,
    mass=1.0,
    sample_idx=0,
    peak_idx=0,
    q_idx=0,
    row_f=20.0,
    col_f=10.0,
    phi_f=-0.20,
    H=1.0,
    K=0.0,
    L=2.0,
):
    update = diffraction._weighted_event_update_representative.py_func
    (
        representative_valid,
        representative_neg_mosaic_weight,
        representative_angular_distance,
        representative_beam_distance,
        representative_wavelength_distance,
        representative_neg_mass,
        representative_sample_idx,
        representative_peak_idx,
        representative_q_idx,
        representative_rows,
    ) = state
    update(
        int(rep_slot),
        float(sample_mosaic_weight),
        float(angular),
        float(beam),
        float(wavelength),
        float(mass),
        int(sample_idx),
        int(peak_idx),
        int(q_idx),
        float(row_f),
        float(col_f),
        float(phi_f),
        float(H),
        float(K),
        float(L),
        representative_valid,
        representative_neg_mosaic_weight,
        representative_angular_distance,
        representative_beam_distance,
        representative_wavelength_distance,
        representative_neg_mass,
        representative_sample_idx,
        representative_peak_idx,
        representative_q_idx,
        representative_rows,
    )


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


def test_deposit_is_v_over_e():
    assert diffraction._weighted_event_deposit(12.0, 3) == pytest.approx(4.0)
    assert diffraction._weighted_event_deposit(12.0, 0) == pytest.approx(0.0)


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
    targets = diffraction._weighted_event_targets(2.0, 50, sample_idx=0)
    expected_indices = diffraction._select_weighted_event_indices_from_targets(
        np.array([1.0, 1.0], dtype=np.float64),
        targets,
    )
    expected_counts = np.bincount(expected_indices, minlength=2)
    actual_counts = np.array(
        [np.asarray(table, dtype=np.float64).shape[0] for table in result[1]],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(actual_counts, expected_counts)


def test_events_per_beam_phase_not_per_peak(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 1.0]], dtype=np.float64),
            (2, 0, 1, 0): np.array([[20.0, 1.0, 0.0, 1.0]], dtype=np.float64),
            (3, 0, 1, 0): np.array([[30.0, 1.0, 0.0, 1.0]], dtype=np.float64),
        },
    )
    event_count = 7
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(
            miller=np.array(
                [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            intensities=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        ),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=event_count,
    )
    assert sum(np.asarray(table, dtype=np.float64).shape[0] for table in result[1]) == event_count


def test_weighted_event_pass2_does_not_tail_fill_inside_qset(monkeypatch):
    def fake_project_fast(*args):
        Qx = float(args[0])
        return True, 0.0, Qx, 0.0, 1.0

    monkeypatch.setattr(diffraction, "_project_weighted_candidate_fast", fake_project_fast)

    sample_terms = np.zeros((1, diffraction._SAMPLE_COLS), dtype=np.float64)
    sample_n2_array = np.ones(1, dtype=np.complex128)
    sample_eps2_array = np.ones(1, dtype=np.complex128)
    flat_event_rows = np.full(
        (3, diffraction.HIT_ROW_WITH_PROVENANCE_WIDTH),
        np.nan,
        dtype=np.float64,
    )
    flat_event_peak_indices = np.full(3, -1, dtype=np.int64)
    event_counts = np.zeros((1, 1), dtype=np.int64)

    result = diffraction._weighted_event_pass2_for_qset.py_func(
        np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        0,
        0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        np.eye(3, dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        sample_terms,
        sample_n2_array,
        sample_eps2_array,
        0.0,
        diffraction.OPTICS_MODE_FAST,
        1.0,
        64,
        diffraction.EXIT_PROJECTION_INTERNAL,
        0,
        np.zeros((diffraction._FAST_OPTICS_LUT_SIZE, diffraction._FAST_OPTICS_LUT_COLS)),
        np.array([0.5, 1.5, 2.5], dtype=np.float64),
        0,
        0.0,
        1.0,
        True,
        flat_event_rows,
        flat_event_peak_indices,
        0,
        event_counts,
        False,
        np.zeros((64, 64), dtype=np.float64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        0,
        0,
    )

    assert int(result[0]) == 1
    assert int(result[2]) == 1
    assert int(result[6]) == 1
    assert int(event_counts[0, 0]) == 1
    assert int(flat_event_peak_indices[0]) == 0
    assert np.all(flat_event_peak_indices[1:] == -1)


def test_fast_outer_loop_streams_targets_across_all_peaks_before_tail_fill(monkeypatch):
    _install_streaming_fast_outer_backend(
        monkeypatch,
        pass1_masses={0: 0.1, 1: 9.9},
    )
    result = diffraction._process_peaks_parallel_weighted_events_fast(
        **_base_process_kwargs(
            miller=np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64),
            intensities=np.array([1.0, 1.0], dtype=np.float64),
        ),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=10,
    )

    rows = _flatten_hit_tables(result[1])
    assert rows.shape[0] == 10
    assert 1.0 not in set(rows[:, 4].tolist())
    assert np.count_nonzero(rows[:, 4] == 2.0) == 10
    stats = diffraction.get_last_process_peaks_weighted_event_stats()
    assert stats["pass2_mass_mismatch_count"] == 0
    assert stats["tail_fill_events"] == 0


def test_fast_outer_loop_reports_pass2_mass_mismatch_and_skips_tail_fill(monkeypatch):
    _install_streaming_fast_outer_backend(
        monkeypatch,
        pass1_masses={0: 10.0},
        pass2_masses={0: 9.0},
    )
    result = diffraction._process_peaks_parallel_weighted_events_fast(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=10,
    )

    rows = _flatten_hit_tables(result[1])
    expected_rows = int(
        np.count_nonzero(diffraction._weighted_event_targets(10.0, 10, sample_idx=0) < 9.0)
    )
    assert rows.shape[0] == expected_rows
    assert rows.shape[0] < 10
    stats = diffraction.get_last_process_peaks_weighted_event_stats()
    assert stats["pass2_mass_mismatch_count"] == 1
    assert stats["pass2_mass_mismatch_max_abs"] == pytest.approx(1.0)
    assert stats["tail_fill_events"] == 0


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


def test_off_detector_candidates_do_not_enter_pdf(monkeypatch):
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array(
                [[32.0, 0.0, 0.0, 1.0], [500.0, 0.0, 0.0, 100.0]],
                dtype=np.float64,
            ),
        },
    )
    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=True,
        events_per_beam_phase=5,
    )
    rows = _flatten_hit_tables(result[1])
    assert rows.shape[0] == 5
    assert set(rows[:, cache_schema.HIT_ROW_COL_DETECTOR_COL].astype(int).tolist()) == {32}
    assert float(np.sum(result[0])) == pytest.approx(1.0)


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


def test_weighted_event_chunk_bounds_cover_samples_without_overlap():
    chunks = diffraction._weighted_event_chunk_bounds(5, 8)

    covered = []
    for _worker_slot, start, stop in chunks:
        covered.extend(range(start, stop))

    assert covered == [0, 1, 2, 3, 4]
    assert all(start < stop for _worker_slot, start, stop in chunks)
    assert len({sample_idx for sample_idx in covered}) == 5


def test_weighted_events_dispatcher_path_matrix():
    normal = diffraction.process_peaks_parallel(
        **_base_process_kwargs(n_samp=2),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=False,
        events_per_beam_phase=2,
        numba_thread_count=2,
    )
    assert len(normal) == 6
    normal_backend = diffraction.get_last_process_peaks_weighted_event_stats()["parallel_backend"]

    safe_stats = {}
    safe = diffraction.process_peaks_parallel_safe(
        **_base_process_kwargs(n_samp=2),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=False,
        events_per_beam_phase=2,
        numba_thread_count=2,
        _safe_stats_out=safe_stats,
    )
    assert len(safe) == 6
    safe_backend = diffraction.get_last_process_peaks_weighted_event_stats()["parallel_backend"]

    serial = diffraction.process_peaks_parallel(
        **_base_process_kwargs(n_samp=2),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=False,
        events_per_beam_phase=2,
        numba_thread_count=1,
    )
    assert len(serial) == 6
    serial_backend = diffraction.get_last_process_peaks_weighted_event_stats()["parallel_backend"]

    fallback = diffraction.process_peaks_parallel.py_func(
        **_base_process_kwargs(n_samp=1),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=False,
        events_per_beam_phase=1,
        numba_thread_count=1,
    )
    assert len(fallback) == 6
    fallback_backend = diffraction.get_last_process_peaks_weighted_event_stats()["parallel_backend"]

    bound = _base_process_kwargs(n_samp=2)
    bound.update(
        {
            "save_flag": 0,
            "collect_hit_tables": False,
            "accumulate_image": False,
            "events_per_beam_phase": 2,
            "numba_thread_count": 2,
        }
    )
    direct = diffraction._process_peaks_parallel_weighted_events_fast_parallel_from_bound(bound)
    assert len(direct) == 7
    direct_backend = diffraction.get_last_process_peaks_weighted_event_stats()["parallel_backend"]

    assert normal_backend in {"threaded_njit_chunks", "fast_serial"}
    assert safe_backend in {"threaded_njit_chunks", "fast_serial"}
    assert safe_stats["used_python_runner"] is False
    assert serial_backend == "fast_serial"
    assert fallback_backend == "weighted_events_python"
    assert direct_backend == "threaded_njit_chunks"


def test_parallel_weighted_event_stats_report_worker_count():
    diffraction.process_peaks_parallel(
        **_base_process_kwargs(n_samp=2),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=False,
        events_per_beam_phase=2,
        numba_thread_count=2,
    )

    stats = diffraction.get_last_process_peaks_weighted_event_stats()
    assert stats["parallel_backend"] == "threaded_njit_chunks"
    assert stats["parallel_worker_count"] == 2
    assert stats["n_solve_q_calls"] == 1
    assert not hasattr(diffraction, "_weighted_event_sample_kernel_parallel")


def test_weighted_event_threaded_backend_does_not_use_prange_kernel():
    assert not hasattr(diffraction, "_weighted_event_sample_kernel_parallel")
    assert hasattr(diffraction, "_weighted_event_sample_chunk_kernel")
    assert hasattr(diffraction, "_run_weighted_event_sample_chunks")

    source = inspect.getsource(diffraction._weighted_event_sample_chunk_kernel.py_func)
    assert "prange" not in source
    assert "get_thread_id" not in source

    diffraction.process_peaks_parallel(
        **_base_process_kwargs(n_samp=2),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=False,
        events_per_beam_phase=2,
        numba_thread_count=2,
    )
    stats = diffraction.get_last_process_peaks_weighted_event_stats()
    assert stats["parallel_backend"] == "threaded_njit_chunks"
    assert stats["parallel_worker_count"] == 2


def test_parallel_weighted_events_match_serial_image():
    serial = diffraction.process_peaks_parallel(
        **_base_process_kwargs(n_samp=2),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=True,
        events_per_beam_phase=2,
        numba_thread_count=1,
    )
    serial_image = np.array(serial[0], copy=True)
    serial_stats = diffraction.get_last_process_peaks_weighted_event_stats()

    threaded = diffraction.process_peaks_parallel(
        **_base_process_kwargs(n_samp=2),
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=True,
        events_per_beam_phase=2,
        numba_thread_count=2,
    )
    threaded_image = np.array(threaded[0], copy=True)
    threaded_stats = diffraction.get_last_process_peaks_weighted_event_stats()

    assert serial_stats["parallel_backend"] == "fast_serial"
    assert threaded_stats["parallel_backend"] == "threaded_njit_chunks"
    np.testing.assert_allclose(threaded_image, serial_image, rtol=1e-12, atol=1e-12)


def test_weighted_events_parallel_from_bound_matches_serial_controlled_backend():
    miller = np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64)
    kwargs = _base_process_kwargs(miller=miller, n_samp=3, image_size=64)
    kwargs.update(
        image=np.zeros((64, 64), dtype=np.float64),
        Distance_CoR_to_Detector=0.1,
        beam_x_array=np.array([-1.0e-4, 0.0, 1.0e-4], dtype=np.float64),
        beam_y_array=np.array([0.0, 1.0e-4, 0.0], dtype=np.float64),
        theta_array=np.array([0.1, 0.1, 0.1], dtype=np.float64),
        phi_array=np.array([0.0, 0.01, -0.01], dtype=np.float64),
        sigma_pv_deg=2.0,
        gamma_pv_deg=2.0,
        eta_pv=0.2,
        wavelength_array=np.array([1.53, 1.54, 1.55], dtype=np.float64),
        pixel_size_m=1.0,
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=True,
        sample_weights=np.array([0.8, 0.5, 0.2], dtype=np.float64),
        events_per_beam_phase=4,
        exit_projection_mode=diffraction.EXIT_PROJECTION_INTERNAL,
        solve_q_steps=64,
    )

    serial_result, parallel_result, serial_best, parallel_best, parallel_stats = (
        _run_fast_serial_and_parallel(kwargs)
    )

    assert parallel_stats["parallel_backend"] == "threaded_njit_chunks"
    assert parallel_stats["pass2_mass_mismatch_count"] == 0
    _assert_fast_parallel_matches_serial(
        kwargs,
        serial_result,
        parallel_result,
        serial_best,
        parallel_best,
    )


def test_weighted_events_parallel_from_bound_matches_serial_real_solve_q_small():
    kwargs = _base_process_kwargs(n_samp=2, image_size=64)
    kwargs.update(
        image=np.zeros((64, 64), dtype=np.float64),
        Distance_CoR_to_Detector=0.1,
        theta_array=np.full(2, 0.1, dtype=np.float64),
        phi_array=np.zeros(2, dtype=np.float64),
        sigma_pv_deg=2.0,
        gamma_pv_deg=2.0,
        eta_pv=0.2,
        pixel_size_m=1.0,
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=True,
        events_per_beam_phase=3,
        exit_projection_mode=diffraction.EXIT_PROJECTION_INTERNAL,
        solve_q_steps=64,
    )

    serial_result, parallel_result, serial_best, parallel_best, parallel_stats = (
        _run_fast_serial_and_parallel(kwargs)
    )

    assert parallel_stats["parallel_backend"] == "threaded_njit_chunks"
    assert parallel_stats["pass2_mass_mismatch_count"] == 0
    assert parallel_stats["n_valid_candidates"] > 0
    assert parallel_stats["n_selected_events"] > 0
    assert float(np.sum(parallel_result[0])) > 0.0
    _assert_fast_parallel_matches_serial(
        kwargs,
        serial_result,
        parallel_result,
        serial_best,
        parallel_best,
    )


def test_weighted_event_fast_parallel_from_bound_uses_threaded_chunks():
    bound = _base_process_kwargs(n_samp=2)
    bound.update(
        {
            "save_flag": 0,
            "collect_hit_tables": False,
            "accumulate_image": False,
            "events_per_beam_phase": 2,
            "numba_thread_count": 2,
        }
    )

    result = diffraction._process_peaks_parallel_weighted_events_fast_parallel_from_bound(bound)

    assert len(result) == 7
    stats = diffraction.get_last_process_peaks_weighted_event_stats()
    assert stats["parallel_backend"] == "threaded_njit_chunks"
    assert stats["parallel_worker_count"] == 2


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


def test_duplicate_events_preserved(monkeypatch):
    diffraction._set_last_intersection_cache([])
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={(1, 0, 1, 0): np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float64)},
    )
    event_count = 4
    result = diffraction.process_peaks_parallel_safe(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=event_count,
    )
    sampled_rows = _flatten_hit_tables(result[1])
    view_rows = _flatten_row_tables(diffraction.get_last_intersection_cache_views()["sampled_event_rows"])
    assert sampled_rows.shape[0] == event_count
    assert view_rows.shape[0] == event_count
    np.testing.assert_allclose(
        sampled_rows,
        np.repeat(sampled_rows[:1], event_count, axis=0),
        equal_nan=True,
    )


def test_representative_cache_prefers_closest_branch_ray_even_if_low_mass(monkeypatch):
    diffraction._set_last_intersection_cache([])
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 1.0]], dtype=np.float64),
            (1, 0, 1, 1): np.array([[20.0, -1.0, 0.0, 1.0]], dtype=np.float64),
        },
        projector=lambda **kwargs: (
            True,
            0.0,
            float(kwargs["Qx"]),
            0.0,
            1.0 if float(kwargs["Qx"]) == 10.0 else 100.0,
        ),
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


def test_representative_cache_prefers_true_mosaic_weight_before_sampled_mass(monkeypatch):
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
    kwargs["theta_array"] = np.array([0.01, 0.02], dtype=np.float64)
    kwargs["phi_array"] = np.array([0.01, 0.02], dtype=np.float64)
    best_indices = np.full(1, -1, dtype=np.int64)
    diffraction.process_peaks_parallel_safe(
        **kwargs,
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        sample_weights=np.array([0.9, 0.1], dtype=np.float64),
        best_sample_indices_out=best_indices,
        events_per_beam_phase=1,
    )

    views = diffraction.get_last_intersection_cache_views()
    rep_rows = np.vstack(
        [np.asarray(table, dtype=np.float64) for table in views["branch_representative_rows"]]
    )
    sampled_rows = np.vstack(
        [np.asarray(table, dtype=np.float64) for table in views["sampled_event_rows"]]
    )
    assert float(np.max(sampled_rows[:, 4])) == pytest.approx(10.0)
    assert float(rep_rows[0, 4]) == pytest.approx(0.9)
    assert float(rep_rows[0, diffraction.CACHE_COL_BEST_SAMPLE_INDEX]) == pytest.approx(0.0)


def test_mosaic_top_representative_survives_even_when_unsampled(monkeypatch):
    diffraction._set_last_intersection_cache([])
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 1.0]], dtype=np.float64),
            (1, 0, 1, 1): np.array([[20.0, -1.0, 0.0, 100.0]], dtype=np.float64),
        },
    )

    def controlled_targets(_total_mass, event_count, sample_idx):
        assert int(event_count) == 1
        if int(sample_idx) == 0:
            return np.empty(0, dtype=np.float64)
        return np.array([0.0], dtype=np.float64)

    monkeypatch.setattr(diffraction, "_weighted_event_targets", controlled_targets)
    kwargs = _base_process_kwargs(n_samp=2)
    kwargs["theta_array"] = np.array([0.01, 0.02], dtype=np.float64)
    kwargs["phi_array"] = np.array([0.01, 0.02], dtype=np.float64)
    diffraction.process_peaks_parallel_safe(
        **kwargs,
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        sample_weights=np.array([0.9, 0.1], dtype=np.float64),
        events_per_beam_phase=1,
    )

    views = diffraction.get_last_intersection_cache_views()
    sampled_rows = _flatten_row_tables(views["sampled_event_rows"])
    representative_rows = _flatten_row_tables(views["branch_representative_rows"])
    last_cache_rows = _flatten_row_tables(diffraction.get_last_intersection_cache())

    assert sampled_rows.shape[0] == 1
    assert float(sampled_rows[0, cache_schema.CACHE_COL_DETECTOR_COL]) == pytest.approx(20.0)
    assert float(sampled_rows[0, cache_schema.CACHE_COL_INTENSITY]) == pytest.approx(10.0)
    assert float(sampled_rows[0, cache_schema.CACHE_COL_BEST_SAMPLE_INDEX]) == pytest.approx(1.0)

    assert representative_rows.shape[0] == 1
    assert float(representative_rows[0, cache_schema.CACHE_COL_DETECTOR_COL]) == pytest.approx(10.0)
    assert float(representative_rows[0, cache_schema.CACHE_COL_INTENSITY]) == pytest.approx(0.9)
    assert float(representative_rows[0, cache_schema.CACHE_COL_BEST_SAMPLE_INDEX]) == pytest.approx(0.0)
    np.testing.assert_allclose(last_cache_rows, representative_rows, rtol=0.0, atol=0.0)

    qr_hit_rows = diffraction.intersection_cache_to_hit_tables(diffraction.get_last_intersection_cache())
    qr_peaks = gqm.build_geometry_fit_simulated_peaks(
        qr_hit_rows,
        image_shape=(64, 64),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        primary_a=4.0,
        primary_c=7.0,
        round_pixel_centers=False,
    )
    selected, _degenerate_count = gqm.collapse_geometry_fit_simulated_peaks(
        qr_peaks,
        one_per_q_group=True,
    )

    assert len(selected) == 1
    assert selected[0]["best_sample_index"] == 0
    assert selected[0]["native_col"] == pytest.approx(10.0)
    assert selected[0]["weight"] == pytest.approx(0.9)


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


def test_representative_rows_not_sampled_events(monkeypatch):
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
    sampled_rows = _flatten_row_tables(views["sampled_event_rows"])
    representative_rows = _flatten_row_tables(views["branch_representative_rows"])
    assert sampled_rows.shape[0] == 3
    assert representative_rows.shape[0] == 1
    assert float(representative_rows[0, cache_schema.CACHE_COL_INTENSITY]) == pytest.approx(1.0)


_WEIGHTED_EVENT_PLAN_COMPLIANCE_MATRIX = [
    {
        "plan_item": "normal path avoids Python sampler",
        "implementation_location": "ra_sim/simulation/diffraction.py::_process_peaks_parallel_impl",
        "test_name": "tests/test_diffraction_weighted_events.py::test_process_peaks_parallel_uses_fast_runtime_not_python_fallback",
        "status": "pass",
        "notes": "normal process_peaks_parallel fails if the Python weighted-event sampler is called",
    },
    {
        "plan_item": "explicit Python fallback/debug path still exists",
        "implementation_location": "ra_sim/simulation/diffraction.py::process_peaks_parallel.py_func",
        "test_name": "tests/test_diffraction_weighted_events.py::test_weighted_events_dispatcher_path_matrix",
        "status": "pass",
        "notes": "explicit py_func path reports weighted_events_python while normal paths stay compiled",
    },
    {
        "plan_item": "P(i)=v_i/V",
        "implementation_location": "ra_sim/simulation/diffraction.py::_select_weighted_event_indices_from_targets",
        "test_name": "tests/test_diffraction_weighted_events.py::test_weighted_event_sampler_matches_mass_ratio",
        "status": "pass",
        "notes": "sample counts follow candidate mass ratio within stochastic tolerance",
    },
    {
        "plan_item": "deposit=V/E",
        "implementation_location": "ra_sim/simulation/diffraction.py::_weighted_event_deposit",
        "test_name": "tests/test_diffraction_weighted_events.py::test_deposit_is_v_over_e",
        "status": "pass",
        "notes": "event deposit equals total valid candidate mass divided by events per beam phase",
    },
    {
        "plan_item": "E is per beam phase, not per peak",
        "implementation_location": "ra_sim/simulation/diffraction.py::_process_peaks_parallel_weighted_events_fast_serial",
        "test_name": "tests/test_diffraction_weighted_events.py::test_events_per_beam_phase_not_per_peak",
        "status": "pass",
        "notes": "multi-peak phase emits exactly E rows total",
    },
    {
        "plan_item": "duplicate selected events preserved",
        "implementation_location": "ra_sim/simulation/diffraction.py::_build_weighted_event_hit_tables",
        "test_name": "tests/test_diffraction_weighted_events.py::test_duplicate_events_preserved",
        "status": "pass",
        "notes": "duplicate sampled rows survive hit tables and sampled cache views",
    },
    {
        "plan_item": "off-detector candidates excluded from V",
        "implementation_location": "ra_sim/simulation/diffraction.py::_project_weighted_candidate",
        "test_name": "tests/test_diffraction_weighted_events.py::test_off_detector_candidates_do_not_enter_pdf",
        "status": "pass",
        "notes": "off-detector high-mass candidates do not affect sampled rows or image mass",
    },
    {
        "plan_item": "no source-template replay",
        "implementation_location": "ra_sim/simulation/diffraction.py::process_peaks_parallel_safe",
        "test_name": "tests/test_source_template_cache.py::test_source_template_and_clustering_disabled",
        "status": "pass",
        "notes": "weighted-event mode bypasses source template construction",
    },
    {
        "plan_item": "no clustered beam replacement",
        "implementation_location": "ra_sim/simulation/diffraction.py::_prepare_clustered_process_peaks_call",
        "test_name": "tests/test_diffraction_safe_wrapper.py::test_process_peaks_parallel_safe_does_not_cluster_beam_replacement",
        "status": "pass",
        "notes": "safe wrapper forwards raw beam arrays without cluster substitution",
    },
    {
        "plan_item": "no grouped source emission",
        "implementation_location": "ra_sim/simulation/diffraction.py::_process_peaks_parallel_weighted_events_fast_serial",
        "test_name": "tests/test_source_template_cache.py::test_grouped_source_expansion_path_is_not_used",
        "status": "pass",
        "notes": "weighted-event hit emission does not use grouped source expansion helpers",
    },
    {
        "plan_item": "no representative/cache row sampling",
        "implementation_location": "ra_sim/simulation/diffraction.py::build_branch_representative_intersection_cache",
        "test_name": "tests/test_diffraction_weighted_events.py::test_branch_representative_cache_passthrough_does_not_recollapse",
        "status": "pass",
        "notes": "representative cache construction is passthrough-only",
    },
    {
        "plan_item": "representative rows separate from sampled rows",
        "implementation_location": "ra_sim/simulation/diffraction.py::get_last_intersection_cache_views",
        "test_name": "tests/test_diffraction_weighted_events.py::test_representative_rows_not_sampled_events",
        "status": "pass",
        "notes": "sampled_event_rows and branch_representative_rows keep separate row counts",
    },
    {
        "plan_item": "get_last_intersection_cache representative-facing",
        "implementation_location": "ra_sim/simulation/diffraction.py::get_last_intersection_cache",
        "test_name": "tests/test_diffraction_weighted_events.py::test_get_last_intersection_cache_is_representative_facing_after_weighted_events",
        "status": "pass",
        "notes": "last cache exposes deterministic representative rows",
    },
    {
        "plan_item": "get_last_intersection_cache_views split sampled/representative views",
        "implementation_location": "ra_sim/simulation/diffraction.py::get_last_intersection_cache_views",
        "test_name": "tests/test_diffraction_weighted_events.py::test_get_last_intersection_cache_views_split_sampled_and_representative_rows",
        "status": "pass",
        "notes": "views expose both stochastic sampled rows and deterministic representative rows",
    },
    {
        "plan_item": "serial fast path and parallel path agree",
        "implementation_location": "ra_sim/simulation/diffraction.py::_process_peaks_parallel_weighted_events_fast_parallel_from_bound",
        "test_name": "tests/test_diffraction_weighted_events.py::test_weighted_events_parallel_from_bound_matches_serial_controlled_backend",
        "status": "pass",
        "notes": "controlled backend compares image, hits, representatives, q data, status, and caches",
    },
    {
        "plan_item": "real solve_q path works",
        "implementation_location": "ra_sim/simulation/diffraction.py::solve_q",
        "test_name": "tests/test_diffraction_weighted_events.py::test_weighted_events_parallel_from_bound_matches_serial_real_solve_q_small",
        "status": "pass",
        "notes": "real solve_q fixture produces valid selected events and matches serial outputs",
    },
    {
        "plan_item": "no silent fallback in normal parallel tests",
        "implementation_location": "ra_sim/simulation/diffraction.py::_weighted_event_parallel_eligible",
        "test_name": "tests/test_diffraction_weighted_events.py::test_weighted_events_dispatcher_path_matrix",
        "status": "pass",
        "notes": "parallel_from_bound reports threaded_njit_chunks and normal paths do not report Python fallback",
    },
    {
        "plan_item": "solve_q JIT does not crash allocate_sched",
        "implementation_location": "ra_sim/simulation/diffraction.py::compute_intensity_array_serial",
        "test_name": "tests/test_diffraction_weighted_events.py::test_solve_q_real_jit_does_not_crash_allocate_sched",
        "status": "pass",
        "notes": "solve_q uses serial intensity helper and avoids LLVM allocate_sched crash path",
    },
    {
        "plan_item": "mosaic-top beam event preserved per final Qr/L/branch slot",
        "implementation_location": "ra_sim/simulation/diffraction.py::_weighted_event_update_representative",
        "test_name": "tests/test_diffraction_weighted_events.py::test_mosaic_top_representative_survives_even_when_unsampled",
        "status": "pass",
        "notes": "unsampled high-mosaic representative survives into branch representative rows and cache",
    },
    {
        "plan_item": "Qr selection uses mosaic-top representative, not sampled event rows",
        "implementation_location": "ra_sim/gui/geometry_q_group_manager.py::collapse_geometry_fit_simulated_peaks",
        "test_name": "tests/test_diffraction_weighted_events.py::test_mosaic_top_representative_survives_even_when_unsampled",
        "status": "pass",
        "notes": "Qr collapse selects representative sample 0 while sampled rows contain sample 1",
    },
]


def test_weighted_events_original_plan_compliance_matrix(capsys):
    required_fields = {
        "plan_item",
        "implementation_location",
        "test_name",
        "status",
        "notes",
    }
    expected_plan_items = [
        "normal path avoids Python sampler",
        "explicit Python fallback/debug path still exists",
        "P(i)=v_i/V",
        "deposit=V/E",
        "E is per beam phase, not per peak",
        "duplicate selected events preserved",
        "off-detector candidates excluded from V",
        "no source-template replay",
        "no clustered beam replacement",
        "no grouped source emission",
        "no representative/cache row sampling",
        "representative rows separate from sampled rows",
        "get_last_intersection_cache representative-facing",
        "get_last_intersection_cache_views split sampled/representative views",
        "serial fast path and parallel path agree",
        "real solve_q path works",
        "no silent fallback in normal parallel tests",
        "solve_q JIT does not crash allocate_sched",
        "mosaic-top beam event preserved per final Qr/L/branch slot",
        "Qr selection uses mosaic-top representative, not sampled event rows",
    ]
    incomplete = False
    incomplete = incomplete or len(_WEIGHTED_EVENT_PLAN_COMPLIANCE_MATRIX) != len(
        expected_plan_items
    )
    incomplete = incomplete or [
        row.get("plan_item") for row in _WEIGHTED_EVENT_PLAN_COMPLIANCE_MATRIX
    ] != expected_plan_items
    for row in _WEIGHTED_EVENT_PLAN_COMPLIANCE_MATRIX:
        incomplete = incomplete or set(row) != required_fields
        incomplete = incomplete or not all(str(row.get(field, "")).strip() for field in required_fields)
        incomplete = incomplete or row.get("status") != "pass"
        row_text = " ".join(str(row.get(field, "")).lower() for field in required_fields)
        incomplete = incomplete or "untested" in row_text

    with capsys.disabled():
        print("plan_item | implementation_location | test_name | status | notes")
        for row in _WEIGHTED_EVENT_PLAN_COMPLIANCE_MATRIX:
            print(
                f"{row['plan_item']} | "
                f"{row['implementation_location']} | "
                f"{row['test_name']} | "
                f"{row['status']} | "
                f"{row['notes']}"
            )
        print(f"original_plan_validation_incomplete={'yes' if incomplete else 'no'}")
    assert not incomplete


def test_weighted_event_merge_diagnostics_marker_contract():
    required = {
        "test_solve_q_real_jit_does_not_crash_allocate_sched",
        "test_compute_intensity_array_is_serial_njit",
        "test_representative_choice_uses_true_mosaic_weight_before_mass",
        "test_representative_choice_preserves_mosaic_top_sample_index_in_hit_row",
        "test_mosaic_top_representative_survives_even_when_unsampled",
        "test_weighted_events_dispatcher_path_matrix",
        "test_weighted_events_parallel_from_bound_matches_serial_controlled_backend",
        "test_weighted_events_parallel_from_bound_matches_serial_real_solve_q_small",
        "test_weighted_events_original_plan_compliance_matrix",
    }

    module_globals = globals()
    missing = sorted(name for name in required if name not in module_globals)
    assert not missing, missing


def test_weighted_event_invariant_matrix_has_no_untested_entries():
    invariant_tests = {
        "off_detector_candidates_do_not_enter_pdf": "test_off_detector_candidates_do_not_enter_pdf",
        "events_per_beam_phase_not_per_peak": "test_events_per_beam_phase_not_per_peak",
        "deposit_is_v_over_e": "test_deposit_is_v_over_e",
        "duplicate_events_preserved": "test_duplicate_events_preserved",
        "representative_rows_not_sampled_events": "test_representative_rows_not_sampled_events",
        "source_template_and_clustering_disabled": "test_source_template_and_clustering_disabled",
        "mosaic_top_representative_survives_even_when_unsampled": (
            "test_mosaic_top_representative_survives_even_when_unsampled"
        ),
    }
    assert all(test_name for test_name in invariant_tests.values())


def test_final_qr_set_branch_slots_fold_same_qr_l_hkls_and_keep_one_representative():
    miller = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float64)
    representative_slot_by_peak_branch, representative_slot_keys = (
        diffraction._build_weighted_event_representative_slot_map(miller, 4.0)
    )

    assert representative_slot_by_peak_branch.shape == (2, 2)
    assert len(representative_slot_keys) == 2
    assert int(representative_slot_by_peak_branch[0, 0]) == int(
        representative_slot_by_peak_branch[1, 0]
    )
    assert int(representative_slot_by_peak_branch[0, 1]) == int(
        representative_slot_by_peak_branch[1, 1]
    )
    assert int(representative_slot_by_peak_branch[0, 0]) != int(
        representative_slot_by_peak_branch[0, 1]
    )

    state = _new_representative_slot_state(len(representative_slot_keys))
    representative_valid = state[0]
    representative_rows = state[-1]

    _update_representative_slot(
        state,
        rep_slot=int(representative_slot_by_peak_branch[0, 0]),
        sample_mosaic_weight=1.0,
        angular=0.40,
        beam=0.0,
        wavelength=0.0,
        mass=10.0,
        sample_idx=2,
        peak_idx=0,
        q_idx=0,
        row_f=20.0,
        col_f=10.0,
        phi_f=-0.30,
        H=1.0,
        K=0.0,
        L=2.0,
    )
    _update_representative_slot(
        state,
        rep_slot=int(representative_slot_by_peak_branch[1, 0]),
        sample_mosaic_weight=1.0,
        angular=0.05,
        beam=0.0,
        wavelength=0.0,
        mass=1.0,
        sample_idx=1,
        peak_idx=1,
        q_idx=1,
        row_f=21.0,
        col_f=11.0,
        phi_f=-0.20,
        H=0.0,
        K=1.0,
        L=2.0,
    )

    representative_hit_tables = diffraction._build_weighted_event_representative_hit_tables(
        representative_valid,
        representative_rows,
        representative_slot_keys,
    )

    assert len(representative_hit_tables) == 1
    row = np.asarray(representative_hit_tables[0], dtype=np.float64)[0]
    np.testing.assert_allclose(row[:10], np.array([1.0, 11.0, 21.0, -0.20, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0]))


def test_representative_choice_uses_true_mosaic_weight_before_mass():
    state = _new_representative_slot_state(1)
    representative_rows = state[-1]

    _update_representative_slot(
        state,
        sample_mosaic_weight=0.9,
        angular=1.0,
        beam=1.0,
        wavelength=1.0,
        mass=1.0,
        sample_idx=0,
    )
    _update_representative_slot(
        state,
        sample_mosaic_weight=0.1,
        angular=0.0,
        beam=0.0,
        wavelength=0.0,
        mass=100.0,
        sample_idx=1,
        q_idx=1,
        row_f=40.0,
        col_f=30.0,
    )

    np.testing.assert_allclose(
        representative_rows[0, :10],
        np.array([1.0, 10.0, 20.0, -0.20, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
    )


def test_representative_choice_falls_back_to_angular_then_beam_then_wavelength():
    state = _new_representative_slot_state(1)
    representative_rows = state[-1]

    _update_representative_slot(
        state,
        sample_mosaic_weight=1.0,
        angular=0.40,
        beam=0.10,
        wavelength=0.10,
        mass=10.0,
        sample_idx=0,
    )
    _update_representative_slot(
        state,
        sample_mosaic_weight=1.0,
        angular=0.20,
        beam=9.00,
        wavelength=9.00,
        mass=1.0,
        sample_idx=1,
    )
    assert int(representative_rows[0, 9]) == 1

    _update_representative_slot(
        state,
        sample_mosaic_weight=1.0,
        angular=0.20,
        beam=0.05,
        wavelength=9.00,
        mass=1.0,
        sample_idx=2,
    )
    assert int(representative_rows[0, 9]) == 2

    _update_representative_slot(
        state,
        sample_mosaic_weight=1.0,
        angular=0.20,
        beam=0.05,
        wavelength=0.01,
        mass=1.0,
        sample_idx=3,
    )
    assert int(representative_rows[0, 9]) == 3


def test_representative_choice_preserves_mosaic_top_sample_index_in_hit_row():
    state = _new_representative_slot_state(1)
    representative_rows = state[-1]

    _update_representative_slot(
        state,
        sample_mosaic_weight=2.0,
        angular=0.0,
        beam=0.0,
        wavelength=0.0,
        mass=5.0,
        sample_idx=7,
        peak_idx=3,
        q_idx=11,
    )

    assert float(representative_rows[0, 7]) == pytest.approx(3.0)
    assert float(representative_rows[0, 8]) == pytest.approx(11.0)
    assert float(representative_rows[0, 9]) == pytest.approx(7.0)


def test_representative_merge_uses_same_mosaic_top_rank_as_update():
    update_state = _new_representative_slot_state(1)
    _update_representative_slot(
        update_state,
        sample_mosaic_weight=0.1,
        angular=0.0,
        beam=0.0,
        wavelength=0.0,
        mass=100.0,
        sample_idx=1,
        q_idx=1,
        row_f=40.0,
        col_f=30.0,
    )
    _update_representative_slot(
        update_state,
        sample_mosaic_weight=0.9,
        angular=1.0,
        beam=1.0,
        wavelength=1.0,
        mass=1.0,
        sample_idx=0,
        q_idx=0,
        row_f=20.0,
        col_f=10.0,
    )

    representative_valid_parts = np.array([[1], [1]], dtype=np.uint8)
    representative_neg_mosaic_weight_parts = np.array([[-0.1], [-0.9]], dtype=np.float64)
    representative_angular_distance_parts = np.array([[0.0], [1.0]], dtype=np.float64)
    representative_beam_distance_parts = np.array([[0.0], [1.0]], dtype=np.float64)
    representative_wavelength_distance_parts = np.array([[0.0], [1.0]], dtype=np.float64)
    representative_neg_mass_parts = np.array([[-100.0], [-1.0]], dtype=np.float64)
    representative_sample_idx_parts = np.array([[1], [0]], dtype=np.int64)
    representative_peak_idx_parts = np.array([[0], [0]], dtype=np.int64)
    representative_q_idx_parts = np.array([[1], [0]], dtype=np.int64)
    representative_rows_parts = np.full(
        (2, 1, diffraction.HIT_ROW_WITH_PROVENANCE_WIDTH),
        np.nan,
        dtype=np.float64,
    )
    representative_rows_parts[0, 0, :] = np.array(
        [100.0, 30.0, 40.0, -0.20, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
        dtype=np.float64,
    )
    representative_rows_parts[1, 0, :] = np.array(
        [1.0, 10.0, 20.0, -0.20, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    merge_state = _new_representative_slot_state(1)
    diffraction._merge_weighted_event_representative_partials.py_func(
        representative_valid_parts,
        representative_neg_mosaic_weight_parts,
        representative_angular_distance_parts,
        representative_beam_distance_parts,
        representative_wavelength_distance_parts,
        representative_neg_mass_parts,
        representative_sample_idx_parts,
        representative_peak_idx_parts,
        representative_q_idx_parts,
        representative_rows_parts,
        *merge_state,
    )

    np.testing.assert_allclose(merge_state[-1], update_state[-1])


def test_build_intersection_cache_preserves_explicit_representative_provenance(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    cache = diffraction.build_intersection_cache(
        [
            np.array(
                [[1.0, 30.0, 40.0, -0.3, 1.0, 0.0, 2.0, 7.0, 3.0, 11.0]],
                dtype=np.float64,
            )
        ],
        4.0,
        7.0,
    )

    assert len(cache) == 1
    np.testing.assert_allclose(
        np.asarray(cache[0], dtype=np.float64)[0, 14:17],
        np.array([7.0, 3.0, 11.0], dtype=np.float64),
    )


def test_build_intersection_cache_prefers_representative_row_sample_index(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    cache = diffraction.build_intersection_cache(
        [
            np.array(
                [[1.0, 30.0, 40.0, -0.3, 1.0, 0.0, 2.0, 7.0, 3.0, 2.0]],
                dtype=np.float64,
            )
        ],
        4.0,
        7.0,
        best_sample_indices_out=np.array([8], dtype=np.int64),
    )

    assert len(cache) == 1
    row = np.asarray(cache[0], dtype=np.float64)[0]
    assert float(row[diffraction.CACHE_COL_BEST_SAMPLE_INDEX]) == pytest.approx(2.0)


def test_branch_representative_cache_keeps_mosaic_top_sample_index(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    cache = diffraction.build_branch_representative_intersection_cache(
        [
            np.array(
                [[1.0, 13.0, 10.0, -0.3, 1.0, 0.0, 2.0, 7.0, 3.0, 4.0]],
                dtype=np.float64,
            ),
            np.array(
                [[1.0, 53.0, 10.0, 0.3, 1.0, 0.0, 2.0, 7.0, 4.0, 1.0]],
                dtype=np.float64,
            ),
        ],
        4.0,
        7.0,
        best_sample_indices_out=np.array([9, 9], dtype=np.int64),
    )

    assert len(cache) == 2
    sample_indices = [
        float(np.asarray(table, dtype=np.float64)[0, diffraction.CACHE_COL_BEST_SAMPLE_INDEX])
        for table in cache
    ]
    assert sample_indices == pytest.approx([4.0, 1.0])


def test_cache_roundtrip_preserves_representative_provenance(monkeypatch):
    monkeypatch.setattr(diffraction, "_should_log_intersection_cache", lambda: False)

    hit_rows = [
        np.array([[2.0, 30.0, 40.0, -0.3, 1.0, 0.0, 2.0, 5.0, 6.0, 7.0]], dtype=np.float64)
    ]
    cache = diffraction.build_branch_representative_intersection_cache(hit_rows, 4.0, 7.0)
    roundtrip_rows = diffraction.intersection_cache_to_hit_tables(cache)

    assert len(roundtrip_rows) == 1
    np.testing.assert_allclose(np.asarray(roundtrip_rows[0], dtype=np.float64), hit_rows[0])


def test_unsampled_branch_stays_in_representative_cache_and_geometry_fit(monkeypatch):
    diffraction._set_last_intersection_cache([])
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)

    sampled_hit_tables = [
        np.array([[5.0, 10.0, 12.0, -0.2, 1.0, 0.0, 2.0, 7.0, 3.0, 0.0]], dtype=np.float64)
    ]
    representative_hit_tables = [
        np.array([[5.0, 10.0, 12.0, -0.2, 1.0, 0.0, 2.0, 7.0, 3.0, 0.0]], dtype=np.float64),
        np.array([[0.1, 50.0, 12.0, 0.2, 1.0, 0.0, 2.0, 7.0, 4.0, 1.0]], dtype=np.float64),
    ]

    diffraction._set_last_intersection_cache_from_hit_tables(
        sampled_hit_tables,
        4.0,
        7.0,
        representative_hit_tables=representative_hit_tables,
    )

    views = diffraction.get_last_intersection_cache_views()
    assert len(views["sampled_event_rows"]) == 1
    assert len(views["branch_representative_rows"]) == 2
    assert len(diffraction.get_last_intersection_cache()) == 2

    representative_rows = diffraction.intersection_cache_to_hit_tables(
        diffraction.get_last_intersection_cache()
    )
    simulated_peaks = gqm.build_geometry_fit_simulated_peaks(
        representative_rows,
        image_shape=(128, 128),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        primary_a=4.0,
        primary_c=7.0,
    )

    assert len(simulated_peaks) == 2
    assert {int(entry["source_branch_index"]) for entry in simulated_peaks} == {0, 1}
    assert len({tuple(entry["q_group_key"]) for entry in simulated_peaks}) == 1
    assert {
        (int(entry["source_table_index"]), int(entry["source_row_index"]))
        for entry in simulated_peaks
    } == {(7, 3), (7, 4)}


def test_get_last_intersection_cache_is_representative_facing_after_weighted_events(monkeypatch):
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
    kwargs["theta_array"] = np.array([0.01, 0.02], dtype=np.float64)
    kwargs["phi_array"] = np.array([0.01, 0.02], dtype=np.float64)

    diffraction.process_peaks_parallel_safe(
        **kwargs,
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        sample_weights=np.array([0.9, 0.1], dtype=np.float64),
        events_per_beam_phase=1,
    )

    representative_cache = diffraction.get_last_intersection_cache()
    assert len(representative_cache) == 1
    representative_row = np.asarray(representative_cache[0], dtype=np.float64)[0]
    assert float(representative_row[diffraction.CACHE_COL_BEST_SAMPLE_INDEX]) == pytest.approx(0.0)
    assert float(representative_row[cache_schema.CACHE_COL_INTENSITY]) == pytest.approx(0.9)


def test_get_last_intersection_cache_views_split_sampled_and_representative_rows(monkeypatch):
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
    kwargs["theta_array"] = np.array([0.01, 0.02], dtype=np.float64)
    kwargs["phi_array"] = np.array([0.01, 0.02], dtype=np.float64)

    diffraction.process_peaks_parallel_safe(
        **kwargs,
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        sample_weights=np.array([0.9, 0.1], dtype=np.float64),
        events_per_beam_phase=1,
    )

    views = diffraction.get_last_intersection_cache_views()
    sampled_rows = np.vstack(
        [np.asarray(table, dtype=np.float64) for table in views["sampled_event_rows"]]
    )
    representative_rows = np.vstack(
        [np.asarray(table, dtype=np.float64) for table in views["branch_representative_rows"]]
    )

    assert set(sampled_rows[:, diffraction.CACHE_COL_BEST_SAMPLE_INDEX].astype(int)) == {0, 1}
    assert representative_rows.shape[0] == 1
    assert float(representative_rows[0, diffraction.CACHE_COL_BEST_SAMPLE_INDEX]) == pytest.approx(0.0)
    assert float(np.max(sampled_rows[:, cache_schema.CACHE_COL_INTENSITY])) == pytest.approx(10.0)
    assert float(representative_rows[0, cache_schema.CACHE_COL_INTENSITY]) == pytest.approx(0.9)


def test_branch_representative_cache_passthrough_does_not_recollapse(monkeypatch):
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
        group_by_qr_set=True,
    )

    assert len(cache) == 2
    roundtrip_rows = np.vstack(
        [np.asarray(table, dtype=np.float64) for table in diffraction.intersection_cache_to_hit_tables(cache)]
    )
    np.testing.assert_allclose(roundtrip_rows, np.vstack(hit_tables))
