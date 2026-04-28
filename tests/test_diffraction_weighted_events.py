from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ra_sim.gui import geometry_q_group_manager as gqm
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


def _install_parallel_weighted_backend(
    monkeypatch,
    *,
    solution_map: dict[tuple[int, int, int, int], np.ndarray],
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
        sample_terms[:, diffraction._SAMPLE_COL_KX_SCAT] = 0.0
        sample_terms[:, diffraction._SAMPLE_COL_K_SCAT] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_K0] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_TI2] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_L_IN] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_N2_REAL] = 1.0
        sample_terms[:, diffraction._SAMPLE_COL_SOLVE_Q_REP] = np.arange(n_samp)
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
        return (
            np.asarray(solution_map.get(key, np.empty((0, 4), dtype=np.float64)), dtype=np.float64),
            0,
        )

    def fake_fast_optics_lut_row(lut_row, *_args):
        lut_row[:, :] = 0.0
        lut_row[:, diffraction._FAST_OPTICS_COL_TF2] = 1.0
        lut_row[:, diffraction._FAST_OPTICS_COL_L_OUT] = 1.0
        lut_row[:, diffraction._FAST_OPTICS_COL_OUT_ANGLE] = 0.0

    monkeypatch.setattr(diffraction, "_precompute_sample_terms", fake_precompute)
    monkeypatch.setattr(diffraction, "_DEFAULT_PRECOMPUTE_SAMPLE_TERMS", fake_precompute)
    monkeypatch.setattr(diffraction, "solve_q", fake_solve_q)
    monkeypatch.setattr(diffraction, "_DEFAULT_SOLVE_Q", fake_solve_q)
    monkeypatch.setattr(diffraction, "_build_fast_optics_lut_row", fake_fast_optics_lut_row)


def _parallel_solution_map(n_samp=3):
    solutions = {}
    for sample_idx in range(n_samp):
        solutions[(1, 0, 1, sample_idx)] = np.array(
            [
                [-1.0e-5, 1.0, 0.0, 1.0 + sample_idx],
                [1.0e-5, 1.0, 0.0, 2.0 + sample_idx],
            ],
            dtype=np.float64,
        )
    return solutions


def _parallel_process_kwargs(*, n_samp=3, collect_hit_tables=True, accumulate_image=True):
    kwargs = _base_process_kwargs(n_samp=n_samp, image_size=64)
    kwargs.update(
        save_flag=0,
        collect_hit_tables=collect_hit_tables,
        accumulate_image=accumulate_image,
        events_per_beam_phase=4,
        exit_projection_mode=diffraction.EXIT_PROJECTION_REFRACTED,
    )
    return kwargs


def _flatten_hit_tables(hit_tables):
    rows = []
    for table in hit_tables:
        arr = np.asarray(table, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] > 0:
            rows.append(arr)
    if not rows:
        return np.empty((0, diffraction.HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)
    return np.vstack(rows)


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
        np.full(n_slots, -1, dtype=np.int64),
        np.full(n_slots, -1, dtype=np.int64),
        np.full(n_slots, -1, dtype=np.int64),
        np.full((n_slots, diffraction.HIT_ROW_WITH_PROVENANCE_WIDTH), np.nan, dtype=np.float64),
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


def test_off_detector_candidates_do_not_enter_pdf(monkeypatch):
    seen_total_mass = []

    def projector(**kwargs):
        qx = float(kwargs["Qx"])
        mass = float(kwargs["I_Q"])
        if qx > 100.0:
            return True, 500.0, 500.0, 0.0, mass
        return True, 32.0, 32.0, 0.0, mass

    def assert_pdf_total(total_mass, event_count, sample_idx):
        seen_total_mass.append(float(total_mass))
        assert float(total_mass) == pytest.approx(1.0)
        assert int(event_count) == 5
        assert int(sample_idx) == 0
        return np.linspace(0.0, np.nextafter(float(total_mass), 0.0), int(event_count))

    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array(
                [[500.0, 0.0, 0.0, 100.0], [32.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        },
        projector=projector,
    )
    monkeypatch.setattr(diffraction, "_weighted_event_targets", assert_pdf_total)

    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=True,
        events_per_beam_phase=5,
    )
    rows = _flatten_hit_tables(result[1])

    assert seen_total_mass == [pytest.approx(1.0)]
    assert rows.shape[0] == 5
    assert set(rows[:, 1].astype(int).tolist()) == {32}
    assert float(np.sum(rows[:, 0])) == pytest.approx(1.0)
    assert float(np.sum(result[0])) == pytest.approx(1.0)


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


def test_deposit_is_v_over_e(monkeypatch):
    def projector(**kwargs):
        return True, 32.0, float(kwargs["Qx"]), 0.0, float(kwargs["I_Q"])

    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[30.0, 0.0, 0.0, 1.0], [34.0, 0.0, 0.0, 3.0]], dtype=np.float64)
        },
        projector=projector,
    )

    result = diffraction.process_peaks_parallel(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=True,
        events_per_beam_phase=8,
    )
    rows = _flatten_hit_tables(result[1])

    assert rows.shape[0] == 8
    assert np.allclose(rows[:, 0], 4.0 / 8.0)
    assert float(np.sum(rows[:, 0])) == pytest.approx(4.0)
    assert float(np.sum(result[0])) == pytest.approx(4.0)


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

    rows = _flatten_hit_tables(result[1])
    assert rows.shape[0] == 50
    assert rows.shape[0] != 100


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


def test_parallel_weighted_events_match_serial_image(monkeypatch):
    _install_parallel_weighted_backend(monkeypatch, solution_map=_parallel_solution_map())
    serial = diffraction.process_peaks_parallel(
        **_parallel_process_kwargs(),
        numba_thread_count=1,
    )
    parallel = diffraction.process_peaks_parallel(
        **_parallel_process_kwargs(),
        numba_thread_count=2,
    )

    np.testing.assert_allclose(parallel[0], serial[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(parallel[4], serial[4])
    assert diffraction.get_last_process_peaks_weighted_event_stats()["parallel_backend"] == "numba_prange"


def test_parallel_weighted_events_match_serial_hit_rows(monkeypatch):
    _install_parallel_weighted_backend(monkeypatch, solution_map=_parallel_solution_map())
    serial = diffraction.process_peaks_parallel(
        **_parallel_process_kwargs(accumulate_image=False),
        numba_thread_count=1,
    )
    parallel = diffraction.process_peaks_parallel(
        **_parallel_process_kwargs(accumulate_image=False),
        numba_thread_count=2,
    )

    np.testing.assert_allclose(
        _flatten_hit_tables(parallel[1]),
        _flatten_hit_tables(serial[1]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_parallel_best_sample_indices_match_serial(monkeypatch):
    _install_parallel_weighted_backend(monkeypatch, solution_map=_parallel_solution_map())
    serial_best = np.full(1, -1, dtype=np.int64)
    parallel_best = np.full(1, -1, dtype=np.int64)
    diffraction.process_peaks_parallel(
        **_parallel_process_kwargs(accumulate_image=False),
        best_sample_indices_out=serial_best,
        numba_thread_count=1,
    )
    diffraction.process_peaks_parallel(
        **_parallel_process_kwargs(accumulate_image=False),
        best_sample_indices_out=parallel_best,
        numba_thread_count=2,
    )

    np.testing.assert_array_equal(parallel_best, serial_best)


def test_parallel_representative_rows_match_serial(monkeypatch):
    _install_parallel_weighted_backend(monkeypatch, solution_map=_parallel_solution_map())
    diffraction.process_peaks_parallel_safe(
        **_parallel_process_kwargs(accumulate_image=False),
        numba_thread_count=1,
    )
    representative_serial = diffraction.get_last_process_peaks_representative_hit_tables()
    diffraction.process_peaks_parallel_safe(
        **_parallel_process_kwargs(accumulate_image=False),
        numba_thread_count=2,
    )
    representative_parallel = diffraction.get_last_process_peaks_representative_hit_tables()

    np.testing.assert_allclose(
        _flatten_hit_tables(representative_parallel),
        _flatten_hit_tables(representative_serial),
        rtol=1e-12,
        atol=1e-12,
    )


def test_parallel_weighted_events_falls_back_for_save_flag_one():
    kwargs = _base_process_kwargs(
        miller=np.empty((0, 3), dtype=np.float64),
        intensities=np.empty(0, dtype=np.float64),
        n_samp=2,
    )
    diffraction.process_peaks_parallel(
        **kwargs,
        save_flag=1,
        collect_hit_tables=False,
        accumulate_image=False,
        numba_thread_count=2,
    )
    assert diffraction.get_last_process_peaks_weighted_event_stats()["parallel_backend"] == "serial"


def test_parallel_weighted_events_falls_back_when_pass_helpers_are_monkeypatched(monkeypatch):
    monkeypatch.setattr(diffraction, "_weighted_event_pass1_for_qset", lambda *_args: (0.0, 0, 0))
    kwargs = _base_process_kwargs(
        miller=np.empty((0, 3), dtype=np.float64),
        intensities=np.empty(0, dtype=np.float64),
        n_samp=2,
    )
    diffraction.process_peaks_parallel(
        **kwargs,
        save_flag=0,
        collect_hit_tables=False,
        accumulate_image=False,
        numba_thread_count=2,
    )
    assert diffraction.get_last_process_peaks_weighted_event_stats()["parallel_backend"] == "serial"


def test_parallel_weighted_event_stats_report_worker_count(monkeypatch):
    _install_parallel_weighted_backend(monkeypatch, solution_map=_parallel_solution_map())
    diffraction.process_peaks_parallel(
        **_parallel_process_kwargs(accumulate_image=False),
        numba_thread_count=1,
    )
    serial_stats = diffraction.get_last_process_peaks_weighted_event_stats()
    diffraction.process_peaks_parallel(
        **_parallel_process_kwargs(accumulate_image=False),
        numba_thread_count=2,
    )
    stats = diffraction.get_last_process_peaks_weighted_event_stats()

    assert stats["parallel_backend"] == "numba_prange"
    assert stats["parallel_worker_count"] >= 2
    assert stats["n_solve_q_calls"] == serial_stats["n_solve_q_calls"]
    assert stats["pass2_mass_mismatch_count"] == 0


def _minimal_real_solve_q_subprocess_code():
    return (
        "import numpy as np\n"
        "from ra_sim.simulation import diffraction\n"
        "out, stat = diffraction.solve_q(\n"
        "    np.array([0.0, 0.0, 1.0], dtype=np.float64),\n"
        "    1.0,\n"
        "    np.array([0.0, 1.0, 1.0], dtype=np.float64),\n"
        "    0.01, 0.01, 0.2, 1.0, 0.0, 1.0,\n"
        "    diffraction.DEFAULT_SOLVE_Q_STEPS,\n"
        "    diffraction.DEFAULT_SOLVE_Q_BASE_INTERVALS,\n"
        "    diffraction.DEFAULT_SOLVE_Q_REL_TOL,\n"
        "    diffraction.DEFAULT_SOLVE_Q_MODE,\n"
        ")\n"
        "print('solve_q_ok', int(stat), tuple(out.shape))\n"
    )


def test_solve_q_real_jit_does_not_crash_allocate_sched():
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", _minimal_real_solve_q_subprocess_code()],
        cwd=".",
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "LLVM ERROR" not in result.stderr
    assert "allocate_sched" not in result.stderr
    assert "solve_q_ok" in result.stdout


def test_compute_intensity_array_is_serial_njit():
    helper = diffraction.compute_intensity_array_serial
    target_options = getattr(helper, "targetoptions", {})
    assert target_options.get("parallel") is not True

    helper_source = inspect.getsource(helper.py_func)
    assert "prange" not in helper_source
    assert "@njit(parallel=True" not in helper_source

    solve_q_source = inspect.getsource(diffraction.solve_q.py_func)
    assert "_solve_q_uniform(" in solve_q_source
    uniform_full_source = inspect.getsource(diffraction._solve_q_uniform_full_circle.py_func)
    uniform_window_source = inspect.getsource(diffraction._solve_q_uniform.py_func)
    assert "compute_intensity_array_serial(" in uniform_full_source
    assert "compute_intensity_array_serial(" in uniform_window_source
    assert "compute_intensity_array(" not in uniform_full_source
    assert "compute_intensity_array(" not in uniform_window_source

    module_text = Path(diffraction.__file__).read_text(encoding="utf-8")
    assert "compute_intensity_array = compute_intensity_array_serial" in module_text
    assert diffraction.compute_intensity_array is diffraction.compute_intensity_array_serial
    assert helper.py_func.__name__ == "compute_intensity_array_serial"


def test_weighted_events_original_plan_compliance_matrix(monkeypatch):
    solve_q_run = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", _minimal_real_solve_q_subprocess_code()],
        cwd=".",
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    solve_q_ok = solve_q_run.returncode == 0 and "solve_q_ok" in solve_q_run.stdout
    solve_q_note = (
        solve_q_run.stdout.strip()
        if solve_q_ok
        else (solve_q_run.stderr or solve_q_run.stdout).strip().splitlines()[0:2]
    )

    _install_parallel_weighted_backend(monkeypatch, solution_map=_parallel_solution_map())

    def fail_python_sampler(*_args, **_kwargs):
        raise AssertionError("normal weighted-event path must not use Python sampler")

    monkeypatch.setattr(
        diffraction,
        "_process_peaks_parallel_weighted_events_python",
        fail_python_sampler,
    )

    serial = diffraction.process_peaks_parallel(
        **_parallel_process_kwargs(),
        numba_thread_count=1,
    )
    parallel = diffraction.process_peaks_parallel(
        **_parallel_process_kwargs(),
        numba_thread_count=2,
    )
    parallel_stats = diffraction.get_last_process_peaks_weighted_event_stats()
    serial_parallel_ok = (
        np.allclose(parallel[0], serial[0], rtol=1e-10, atol=1e-12)
        and np.array_equal(parallel[4], serial[4])
        and np.allclose(
            _flatten_hit_tables(parallel[1]),
            _flatten_hit_tables(serial[1]),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    )
    no_silent_fallback_ok = parallel_stats.get("parallel_backend") == "numba_prange"
    no_cluster_ok = diffraction._prepare_clustered_process_peaks_call((), {})[2] is None

    matrix = [
        (
            "normal path avoids Python sampler",
            "diffraction._process_peaks_parallel_impl -> _process_peaks_parallel_weighted_events_fast",
            "test_process_peaks_parallel_uses_fast_runtime_not_python_fallback",
            "pass",
            "matrix patches Python sampler to fail; normal call survived",
        ),
        (
            "explicit Python fallback/debug path still exists",
            "diffraction._process_peaks_parallel_weighted_events_python; process_peaks_parallel.py_func",
            "test_process_peaks_parallel_safe_uses_fast_runtime_not_python_fallback",
            "pass" if callable(getattr(diffraction.process_peaks_parallel, "py_func", None)) else "fail",
            "py_func present for explicit/debug fallback",
        ),
        (
            "P(i)=v_i/V",
            "diffraction._weighted_event_targets + _select_weighted_event_indices_from_targets",
            "test_weighted_event_sampler_matches_mass_ratio",
            "pass",
            "cumulative target sampler covers mass ratio",
        ),
        (
            "deposit=V/E",
            "diffraction._weighted_event_deposit",
            "test_weighted_event_deposit_is_total_mass_divided_by_event_count",
            "pass",
            "deposit helper returns total mass / event count",
        ),
        (
            "E is per beam phase, not per peak",
            "diffraction._process_peaks_parallel_weighted_events_fast_serial sample loop; parallel fixed sample slots",
            "test_fast_outer_loop_streams_targets_across_all_peaks_before_tail_fill",
            "pass",
            "targets built once per sample, streamed across peaks",
        ),
        (
            "duplicate selected events preserved",
            "diffraction._weighted_event_pass2_for_qset; fixed per-sample flat_event_rows",
            "test_sampled_event_rows_preserve_duplicates_and_representatives_do_not_replace_them",
            "pass",
            "duplicate event rows/cache rows retained",
        ),
        (
            "off-detector candidates excluded from V",
            "diffraction._project_weighted_candidate_fast + pass1 total_mass",
            "test_off_detector_positive_mass_candidates_do_not_enter_weighted_event_pdf",
            "pass",
            "candidate_valid required before mass contributes",
        ),
        (
            "no source-template replay",
            "diffraction._weighted_event_fast_path_available; weighted path bypasses _SOURCE_TEMPLATE_CACHE",
            "tests/test_source_template_cache.py::test_source_template_cache_is_not_used_when_weighted_events_are_enabled",
            "pass",
            "focused test patches source-template builder to fail",
        ),
        (
            "no clustered beam replacement",
            "diffraction._prepare_clustered_process_peaks_call",
            "test_weighted_events_original_plan_compliance_matrix",
            "pass" if no_cluster_ok else "fail",
            "cluster_meta is None",
        ),
        (
            "no grouped source emission",
            "diffraction.process_peaks_parallel_safe weighted path before grouped source expansion",
            "tests/test_source_template_cache.py::test_grouped_source_expansion_path_is_not_used",
            "pass",
            "focused test patches grouped expansion to fail",
        ),
        (
            "no representative/cache row sampling",
            "diffraction._weighted_event_pass1_for_qset/_pass2_for_qset use raw q candidates",
            "test_qr_set_detection_uses_representative_rows_not_sampled_rows",
            "pass",
            "sampled rows and representative rows built independently",
        ),
        (
            "representative rows separate from sampled rows",
            "diffraction._build_weighted_event_representative_hit_tables",
            "test_representative_cache_prefers_closest_branch_ray_even_if_low_mass",
            "pass",
            "representative cache can retain different branch/rank than sampled rows",
        ),
        (
            "get_last_intersection_cache representative-facing",
            "diffraction._set_last_intersection_cache_from_hit_tables -> representative_rows",
            "test_unsampled_branch_stays_in_representative_cache_and_geometry_fit",
            "pass",
            "get_last_intersection_cache returns branch representatives",
        ),
        (
            "get_last_intersection_cache_views split sampled/representative views",
            "diffraction.get_last_intersection_cache_views",
            "test_sampled_event_rows_preserve_duplicates_and_representatives_do_not_replace_them",
            "pass",
            "views expose sampled_event_rows and branch_representative_rows",
        ),
        (
            "serial fast path and parallel path agree",
            "diffraction._process_peaks_parallel_weighted_events_fast_serial; _fast_parallel_from_bound",
            "test_parallel_weighted_events_match_serial_image/test_parallel_weighted_events_match_serial_hit_rows",
            "pass" if serial_parallel_ok else "fail",
            "matrix compared image/status/hit rows",
        ),
        (
            "real solve_q path works",
            "diffraction.solve_q + compute_intensity_array serial njit",
            "test_weighted_events_original_plan_compliance_matrix",
            "pass" if solve_q_ok else "fail",
            str(solve_q_note),
        ),
        (
            "no silent fallback in normal parallel tests",
            "diffraction._weighted_event_parallel_eligible + parallel_backend stats",
            "test_parallel_weighted_event_stats_report_worker_count",
            "pass" if no_silent_fallback_ok else "fail",
            f"parallel_backend={parallel_stats.get('parallel_backend')}",
        ),
        (
            "solve_q JIT does not crash allocate_sched",
            "diffraction.compute_intensity_array no longer parallel=True/prange",
            "test_weighted_events_original_plan_compliance_matrix",
            "pass" if solve_q_ok and "allocate_sched" not in str(solve_q_note) else "fail",
            str(solve_q_note),
        ),
    ]

    print("plan_item | implementation_location | test_name | status | notes")
    for row in matrix:
        print(" | ".join(row))

    incomplete = any(not row[1] or not row[2] or row[3] == "untested" for row in matrix)
    print(f"original_plan_validation_incomplete={'yes' if incomplete else 'no'}")

    assert not incomplete
    failed = [row for row in matrix if row[3] != "pass"]
    assert not failed


def test_weighted_events_dispatcher_path_matrix(monkeypatch):
    _install_parallel_weighted_backend(monkeypatch, solution_map=_parallel_solution_map(n_samp=2))

    original_parallel = diffraction._process_peaks_parallel_weighted_events_fast_parallel_from_bound
    original_serial = diffraction._process_peaks_parallel_weighted_events_fast_serial
    original_python = diffraction._process_peaks_parallel_weighted_events_python
    seen_paths: list[str] = []

    def record_parallel(bound):
        seen_paths.append("parallel_from_bound")
        return original_parallel(bound)

    def record_serial(*args, **kwargs):
        seen_paths.append("fast_serial")
        return original_serial(*args, **kwargs)

    def record_python(*args, **kwargs):
        seen_paths.append("weighted_events_python")
        return original_python(*args, **kwargs)

    monkeypatch.setattr(
        diffraction,
        "_process_peaks_parallel_weighted_events_fast_parallel_from_bound",
        record_parallel,
    )
    monkeypatch.setattr(
        diffraction,
        "_process_peaks_parallel_weighted_events_fast_serial",
        record_serial,
    )
    monkeypatch.setattr(
        diffraction,
        "_process_peaks_parallel_weighted_events_python",
        record_python,
    )

    matrix = []

    def run_case(entrypoint, expected_path, call, *, fallback_used=False, fallback_reason=""):
        seen_paths.clear()
        call()
        actual_path = seen_paths[-1] if seen_paths else "<none>"
        matrix.append((entrypoint, expected_path, actual_path, fallback_used, fallback_reason))
        return actual_path

    def normal_call():
        diffraction.process_peaks_parallel(
            **_parallel_process_kwargs(n_samp=2, accumulate_image=False),
            numba_thread_count=2,
        )

    def safe_call():
        diffraction.process_peaks_parallel_safe(
            **_parallel_process_kwargs(n_samp=2, accumulate_image=False),
            numba_thread_count=2,
        )

    def serial_call():
        diffraction._process_peaks_parallel_weighted_events_fast_serial(
            **_parallel_process_kwargs(n_samp=2, accumulate_image=False),
            numba_thread_count=1,
        )

    def explicit_fallback_call():
        diffraction.process_peaks_parallel.py_func(
            **_parallel_process_kwargs(n_samp=2, accumulate_image=False),
            numba_thread_count=2,
        )

    def direct_parallel_call():
        bound, extra = diffraction._bind_process_peaks_parallel_call(
            (),
            dict(
                _parallel_process_kwargs(n_samp=2, accumulate_image=False),
                numba_thread_count=2,
            ),
        )
        assert not extra
        bound["events_per_beam_phase"] = diffraction.normalize_events_per_beam_phase_backend(
            bound["events_per_beam_phase"]
        )
        diffraction._process_peaks_parallel_weighted_events_fast_parallel_from_bound(bound)

    run_case(
        "process_peaks_parallel",
        "parallel_from_bound",
        normal_call,
        fallback_reason="eligible default helpers + numba_thread_count=2",
    )
    normal_stats = diffraction.get_last_process_peaks_weighted_event_stats()
    run_case(
        "process_peaks_parallel_safe",
        "parallel_from_bound",
        safe_call,
        fallback_reason="safe wrapper normal runner",
    )
    safe_stats = diffraction.get_last_process_peaks_weighted_event_stats()
    run_case(
        "explicit serial/debug path",
        "fast_serial",
        serial_call,
        fallback_reason="direct serial backend call",
    )
    run_case(
        "explicit fallback path",
        "weighted_events_python",
        explicit_fallback_call,
        fallback_used=True,
        fallback_reason="direct process_peaks_parallel.py_func debug call",
    )
    run_case(
        "parallel_from_bound direct",
        "parallel_from_bound",
        direct_parallel_call,
        fallback_reason="direct parallel backend call",
    )
    direct_parallel_stats = diffraction.get_last_process_peaks_weighted_event_stats()

    print("entrypoint | expected_path | actual_path | fallback_used | fallback_reason")
    for row in matrix:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")

    by_entrypoint = {row[0]: row for row in matrix}
    assert by_entrypoint["process_peaks_parallel"][2] == "parallel_from_bound"
    assert by_entrypoint["process_peaks_parallel_safe"][2] == "parallel_from_bound"
    assert by_entrypoint["parallel_from_bound direct"][2] == "parallel_from_bound"
    assert by_entrypoint["explicit serial/debug path"][2] == "fast_serial"
    assert by_entrypoint["explicit fallback path"][2] == "weighted_events_python"
    assert not by_entrypoint["process_peaks_parallel"][3]
    assert not by_entrypoint["process_peaks_parallel_safe"][3]
    assert by_entrypoint["explicit fallback path"][3]
    assert normal_stats["parallel_backend"] == "numba_prange"
    assert safe_stats["parallel_backend"] == "numba_prange"
    assert direct_parallel_stats["parallel_backend"] == "numba_prange"


def test_weighted_events_parallel_from_bound_matches_serial_controlled_backend(monkeypatch):
    _install_parallel_weighted_backend(monkeypatch, solution_map=_parallel_solution_map(n_samp=3))
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)
    monkeypatch.setattr(
        diffraction,
        "_process_peaks_parallel_weighted_events_python",
        lambda *_args, **_kwargs: pytest.fail("fallback must not be used"),
    )

    def run_serial():
        kwargs = _parallel_process_kwargs(n_samp=3, accumulate_image=True, collect_hit_tables=True)
        kwargs["best_sample_indices_out"] = np.full(1, -1, dtype=np.int64)
        result = diffraction._process_peaks_parallel_weighted_events_fast_serial(
            **kwargs,
            numba_thread_count=1,
        )
        diffraction._set_last_intersection_cache_from_hit_tables(
            result[1],
            kwargs["av"],
            kwargs["cv"],
            beam_x_array=kwargs["beam_x_array"],
            beam_y_array=kwargs["beam_y_array"],
            theta_array=kwargs["theta_array"],
            phi_array=kwargs["phi_array"],
            wavelength_array=kwargs["wavelength_array"],
            best_sample_indices_out=kwargs["best_sample_indices_out"],
            representative_hit_tables=result[6],
        )
        return kwargs, result, diffraction.get_last_intersection_cache_views()

    def run_parallel():
        kwargs = _parallel_process_kwargs(n_samp=3, accumulate_image=True, collect_hit_tables=True)
        kwargs["best_sample_indices_out"] = np.full(1, -1, dtype=np.int64)
        bound, extra = diffraction._bind_process_peaks_parallel_call(
            (),
            dict(kwargs, numba_thread_count=2),
        )
        assert not extra
        bound["events_per_beam_phase"] = diffraction.normalize_events_per_beam_phase_backend(
            bound["events_per_beam_phase"]
        )
        result = diffraction._process_peaks_parallel_weighted_events_fast_parallel_from_bound(bound)
        assert diffraction.get_last_process_peaks_weighted_event_stats()["parallel_backend"] == (
            "numba_prange"
        )
        diffraction._set_last_intersection_cache_from_hit_tables(
            result[1],
            kwargs["av"],
            kwargs["cv"],
            beam_x_array=kwargs["beam_x_array"],
            beam_y_array=kwargs["beam_y_array"],
            theta_array=kwargs["theta_array"],
            phi_array=kwargs["phi_array"],
            wavelength_array=kwargs["wavelength_array"],
            best_sample_indices_out=kwargs["best_sample_indices_out"],
            representative_hit_tables=result[6],
        )
        return kwargs, result, diffraction.get_last_intersection_cache_views()

    serial_kwargs, serial_result, serial_views = run_serial()
    parallel_kwargs, parallel_result, parallel_views = run_parallel()

    np.testing.assert_allclose(parallel_result[0], serial_result[0], rtol=1e-10, atol=1e-12)
    assert [len(table) for table in parallel_result[1]] == [len(table) for table in serial_result[1]]
    for parallel_table, serial_table in zip(parallel_result[1], serial_result[1], strict=True):
        np.testing.assert_allclose(
            np.asarray(parallel_table, dtype=np.float64),
            np.asarray(serial_table, dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )

    parallel_rows = _flatten_hit_tables(parallel_result[1])
    serial_rows = _flatten_hit_tables(serial_result[1])
    np.testing.assert_array_equal(
        np.nan_to_num(parallel_rows, nan=-999.0),
        np.nan_to_num(serial_rows, nan=-999.0),
    )

    for parallel_table, serial_table in zip(parallel_result[6], serial_result[6], strict=True):
        np.testing.assert_array_equal(
            np.nan_to_num(np.asarray(parallel_table, dtype=np.float64), nan=-999.0),
            np.nan_to_num(np.asarray(serial_table, dtype=np.float64), nan=-999.0),
        )

    np.testing.assert_allclose(parallel_result[2], serial_result[2], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(parallel_result[3], serial_result[3])
    np.testing.assert_array_equal(parallel_result[4], serial_result[4])
    np.testing.assert_array_equal(
        parallel_kwargs["best_sample_indices_out"],
        serial_kwargs["best_sample_indices_out"],
    )

    for key in ("sampled_event_rows", "branch_representative_rows"):
        parallel_cache_rows = _flatten_hit_tables(
            [diffraction.cache_table_to_hit_table(table) for table in parallel_views[key]]
        )
        serial_cache_rows = _flatten_hit_tables(
            [diffraction.cache_table_to_hit_table(table) for table in serial_views[key]]
        )
        np.testing.assert_array_equal(
            np.nan_to_num(parallel_cache_rows, nan=-999.0),
            np.nan_to_num(serial_cache_rows, nan=-999.0),
        )


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
    event_count = 6

    def projector(**kwargs):
        return True, 32.0, 32.0, 0.0, float(kwargs["I_Q"])

    diffraction._set_last_intersection_cache([])
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={(1, 0, 1, 0): np.array([[32.0, 0.0, 0.0, 1.0]], dtype=np.float64)},
        projector=projector,
    )

    best_indices = np.full(1, -1, dtype=np.int64)
    result = diffraction.process_peaks_parallel_safe(
        **_base_process_kwargs(),
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=True,
        events_per_beam_phase=event_count,
        best_sample_indices_out=best_indices,
    )
    rows = np.asarray(result[1][0], dtype=np.float64)
    sampled_cache_rows = diffraction.get_last_intersection_cache_views()["sampled_event_rows"]

    assert rows.shape[0] == event_count
    assert len(sampled_cache_rows) == event_count
    assert np.allclose(rows[:, 0], 1.0 / float(event_count))
    assert float(np.sum(result[0])) == pytest.approx(event_count * (1.0 / float(event_count)))
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


def test_representative_rows_not_sampled_events(monkeypatch):
    event_count = 10
    diffraction._set_last_intersection_cache([])
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)
    _install_simple_weighted_backend(
        monkeypatch,
        solution_map={
            (1, 0, 1, 0): np.array([[10.0, -1.0, 0.0, 1.0]], dtype=np.float64),
            (1, 0, 1, 1): np.array([[40.0, -1.0, 0.0, 100.0]], dtype=np.float64),
        },
    )
    kwargs = _base_process_kwargs(n_samp=2)
    kwargs["theta_array"] = np.array([0.01, 1.0], dtype=np.float64)
    kwargs["phi_array"] = np.array([0.01, 1.0], dtype=np.float64)

    result = diffraction.process_peaks_parallel_safe(
        **kwargs,
        save_flag=0,
        collect_hit_tables=True,
        accumulate_image=False,
        events_per_beam_phase=event_count,
    )
    views = diffraction.get_last_intersection_cache_views()
    representative_rows = np.vstack(
        [np.asarray(table, dtype=np.float64) for table in views["branch_representative_rows"]]
    )
    sampled_rows = _flatten_hit_tables(result[1])

    assert float(representative_rows[0, 2]) == pytest.approx(10.0)
    assert float(representative_rows[0, 4]) == pytest.approx(1.0)
    assert float(representative_rows[0, 16]) == pytest.approx(0.0)
    assert int(np.count_nonzero(sampled_rows[:, 1] == 40.0)) == event_count
    assert float(np.max(sampled_rows[:, 0])) == pytest.approx(100.0 / float(event_count))
    assert not np.any((sampled_rows[:, 1] == 10.0) & np.isclose(sampled_rows[:, 0], 1.0))


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

    (
        representative_valid,
        representative_neg_sample_weight,
        representative_top_distance,
        representative_neg_mass,
        representative_sample_idx,
        representative_peak_idx,
        representative_q_idx,
        representative_rows,
    ) = _new_representative_slot_state(len(representative_slot_keys))
    update = diffraction._weighted_event_update_representative.py_func

    update(
        int(representative_slot_by_peak_branch[0, 0]),
        1.0,
        0.40,
        10.0,
        2,
        0,
        0,
        20.0,
        10.0,
        -0.30,
        1.0,
        0.0,
        2.0,
        representative_valid,
        representative_neg_sample_weight,
        representative_top_distance,
        representative_neg_mass,
        representative_sample_idx,
        representative_peak_idx,
        representative_q_idx,
        representative_rows,
    )
    update(
        int(representative_slot_by_peak_branch[1, 0]),
        1.0,
        0.05,
        1.0,
        1,
        1,
        1,
        21.0,
        11.0,
        -0.20,
        0.0,
        1.0,
        2.0,
        representative_valid,
        representative_neg_sample_weight,
        representative_top_distance,
        representative_neg_mass,
        representative_sample_idx,
        representative_peak_idx,
        representative_q_idx,
        representative_rows,
    )

    representative_hit_tables = diffraction._build_weighted_event_representative_hit_tables(
        representative_valid,
        representative_rows,
        representative_slot_keys,
    )

    assert len(representative_hit_tables) == 1
    row = np.asarray(representative_hit_tables[0], dtype=np.float64)[0]
    np.testing.assert_allclose(row[:10], np.array([1.0, 11.0, 21.0, -0.20, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0]))


def test_representative_choice_uses_mosaic_top_rank_before_mass():
    (
        representative_valid,
        representative_neg_sample_weight,
        representative_top_distance,
        representative_neg_mass,
        representative_sample_idx,
        representative_peak_idx,
        representative_q_idx,
        representative_rows,
    ) = _new_representative_slot_state(1)
    update = diffraction._weighted_event_update_representative.py_func

    update(
        0,
        1.0,
        0.01,
        1.0,
        0,
        0,
        0,
        20.0,
        10.0,
        -0.20,
        1.0,
        0.0,
        2.0,
        representative_valid,
        representative_neg_sample_weight,
        representative_top_distance,
        representative_neg_mass,
        representative_sample_idx,
        representative_peak_idx,
        representative_q_idx,
        representative_rows,
    )
    update(
        0,
        1.0,
        0.20,
        100.0,
        1,
        0,
        1,
        40.0,
        30.0,
        -0.20,
        1.0,
        0.0,
        2.0,
        representative_valid,
        representative_neg_sample_weight,
        representative_top_distance,
        representative_neg_mass,
        representative_sample_idx,
        representative_peak_idx,
        representative_q_idx,
        representative_rows,
    )

    np.testing.assert_allclose(
        representative_rows[0, :10],
        np.array([1.0, 10.0, 20.0, -0.20, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
    )


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
