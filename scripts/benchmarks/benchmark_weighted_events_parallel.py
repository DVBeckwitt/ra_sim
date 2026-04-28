"""Benchmark the weighted-event parallel sample kernel.

The default fixture patches only setup/Q solving so the benchmark focuses on
the weighted-event sample kernel and remains stable across machines. It still
uses the production ``process_peaks_parallel`` dispatcher, pass helpers, image
deposition, hit-table building, representative merge, and stats publication.
"""

from __future__ import annotations

import argparse
import statistics
import time
from contextlib import contextmanager

import numpy as np

from ra_sim.simulation import diffraction


def _base_kwargs(*, n_samp: int, events_per_beam_phase: int, collect_hit_tables: bool):
    image_size = 64
    return dict(
        miller=np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
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
        save_flag=0,
        collect_hit_tables=collect_hit_tables,
        accumulate_image=True,
        events_per_beam_phase=events_per_beam_phase,
        exit_projection_mode=diffraction.EXIT_PROJECTION_REFRACTED,
    )


@contextmanager
def _controlled_weighted_backend():
    original_precompute = diffraction._precompute_sample_terms
    original_default_precompute = diffraction._DEFAULT_PRECOMPUTE_SAMPLE_TERMS
    original_solve_q = diffraction.solve_q
    original_default_solve_q = diffraction._DEFAULT_SOLVE_Q
    original_lut_builder = diffraction._build_fast_optics_lut_row

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

    def fake_solve_q(k_in_crystal, *_args, **_kwargs):
        sample_idx = int(round(float(np.asarray(k_in_crystal, dtype=np.float64)[0])))
        intensity_shift = 1.0 + 0.01 * float(sample_idx % 17)
        return (
            np.array(
                [
                    [-1.0e-5, 1.0, 0.0, 1.0 * intensity_shift],
                    [1.0e-5, 1.0, 0.0, 2.0 * intensity_shift],
                    [2.0e-5, 1.0, 0.0, 0.5 * intensity_shift],
                ],
                dtype=np.float64,
            ),
            0,
        )

    def fake_fast_optics_lut_row(lut_row, *_args):
        lut_row[:, :] = 0.0
        lut_row[:, diffraction._FAST_OPTICS_COL_TF2] = 1.0
        lut_row[:, diffraction._FAST_OPTICS_COL_L_OUT] = 1.0
        lut_row[:, diffraction._FAST_OPTICS_COL_OUT_ANGLE] = 0.0

    diffraction._precompute_sample_terms = fake_precompute
    diffraction._DEFAULT_PRECOMPUTE_SAMPLE_TERMS = fake_precompute
    diffraction.solve_q = fake_solve_q
    diffraction._DEFAULT_SOLVE_Q = fake_solve_q
    diffraction._build_fast_optics_lut_row = fake_fast_optics_lut_row
    try:
        yield
    finally:
        diffraction._precompute_sample_terms = original_precompute
        diffraction._DEFAULT_PRECOMPUTE_SAMPLE_TERMS = original_default_precompute
        diffraction.solve_q = original_solve_q
        diffraction._DEFAULT_SOLVE_Q = original_default_solve_q
        diffraction._build_fast_optics_lut_row = original_lut_builder


def _run_once(*, n_samp, events_per_beam_phase, collect_hit_tables, numba_thread_count):
    kwargs = _base_kwargs(
        n_samp=n_samp,
        events_per_beam_phase=events_per_beam_phase,
        collect_hit_tables=collect_hit_tables,
    )
    start = time.perf_counter()
    diffraction.process_peaks_parallel(
        **kwargs,
        numba_thread_count=numba_thread_count,
    )
    runtime_s = time.perf_counter() - start
    return runtime_s, diffraction.get_last_process_peaks_weighted_event_stats()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--n-samp", type=int, nargs="*", default=[64, 256, 1024])
    parser.add_argument("--events", type=int, nargs="*", default=[10, 50])
    args = parser.parse_args()

    print(
        "n_samp,events_per_beam_phase,collect_hit_tables,numba_thread_count,"
        "runtime_s,speedup_vs_1_thread,n_solve_q_calls,n_project_candidate_calls,"
        "n_valid_candidates,n_selected_events,parallel_worker_count,parallel_qset_count"
    )
    with _controlled_weighted_backend():
        for n_samp in args.n_samp:
            for events_per_beam_phase in args.events:
                for collect_hit_tables in (False, True):
                    baselines = {}
                    for numba_thread_count in args.threads:
                        _run_once(
                            n_samp=n_samp,
                            events_per_beam_phase=events_per_beam_phase,
                            collect_hit_tables=collect_hit_tables,
                            numba_thread_count=numba_thread_count,
                        )
                        samples = []
                        last_stats = {}
                        for _ in range(max(int(args.runs), 1)):
                            runtime_s, last_stats = _run_once(
                                n_samp=n_samp,
                                events_per_beam_phase=events_per_beam_phase,
                                collect_hit_tables=collect_hit_tables,
                                numba_thread_count=numba_thread_count,
                            )
                            samples.append(float(runtime_s))
                        median_s = statistics.median(samples)
                        if int(numba_thread_count) == 1:
                            baselines[(n_samp, events_per_beam_phase, collect_hit_tables)] = median_s
                        baseline = baselines.get(
                            (n_samp, events_per_beam_phase, collect_hit_tables),
                            median_s,
                        )
                        speedup = baseline / median_s if median_s > 0.0 else 0.0
                        print(
                            f"{n_samp},{events_per_beam_phase},{collect_hit_tables},"
                            f"{numba_thread_count},{median_s:.6f},{speedup:.3f},"
                            f"{last_stats.get('n_solve_q_calls', 0)},"
                            f"{last_stats.get('n_project_candidate_calls', 0)},"
                            f"{last_stats.get('n_valid_candidates', 0)},"
                            f"{last_stats.get('n_selected_events', 0)},"
                            f"{last_stats.get('parallel_worker_count', 1)},"
                            f"{last_stats.get('parallel_qset_count', 0)}"
                        )


if __name__ == "__main__":
    main()
