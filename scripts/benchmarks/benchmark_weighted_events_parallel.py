"""Benchmark weighted-event threaded chunk execution."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ra_sim
from ra_sim.simulation import diffraction

ra_sim_path = Path(ra_sim.__file__).resolve()
if REPO_ROOT not in ra_sim_path.parents:
    raise RuntimeError(f"benchmark imported stale ra_sim: {ra_sim_path}")


def _base_process_kwargs(*, n_samp: int, image_size: int) -> dict[str, object]:
    miller = np.asarray([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64)
    return {
        "miller": miller,
        "intensities": np.ones(miller.shape[0], dtype=np.float64),
        "image_size": int(image_size),
        "av": 4.0,
        "cv": 7.0,
        "lambda_": 1.54,
        "image": np.zeros((int(image_size), int(image_size)), dtype=np.float64),
        "Distance_CoR_to_Detector": 0.1,
        "gamma_deg": 0.0,
        "Gamma_deg": 0.0,
        "chi_deg": 0.0,
        "psi_deg": 0.0,
        "psi_z_deg": 0.0,
        "zs": 0.0,
        "zb": 0.0,
        "n2": 1.0 + 0.0j,
        "beam_x_array": np.linspace(-1.0e-4, 1.0e-4, int(n_samp), dtype=np.float64),
        "beam_y_array": np.zeros(int(n_samp), dtype=np.float64),
        "theta_array": np.full(int(n_samp), 0.1, dtype=np.float64),
        "phi_array": np.zeros(int(n_samp), dtype=np.float64),
        "sigma_pv_deg": 2.0,
        "gamma_pv_deg": 2.0,
        "eta_pv": 0.2,
        "wavelength_array": np.ones(int(n_samp), dtype=np.float64),
        "debye_x": 0.0,
        "debye_y": 0.0,
        "center": np.asarray([image_size / 2.0, image_size / 2.0], dtype=np.float64),
        "theta_initial_deg": 0.0,
        "cor_angle_deg": 0.0,
        "unit_x": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "n_detector": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        "save_flag": 0,
        "collect_hit_tables": False,
        "accumulate_image": False,
        "pixel_size_m": 1.0,
        "solve_q_steps": 64,
    }


def _representative_row_count() -> int:
    hit_tables = diffraction.get_last_process_peaks_representative_hit_tables()
    if not hit_tables:
        return 0
    row_count = 0
    for table in hit_tables:
        arr = np.asarray(table, dtype=np.float64)
        if arr.ndim >= 2:
            row_count += int(arr.shape[0])
    return int(row_count)


def _time_run(*, n_samp: int, image_size: int, events: int, workers: int) -> dict[str, Any]:
    kwargs = _base_process_kwargs(n_samp=n_samp, image_size=image_size)
    kwargs["events_per_beam_phase"] = int(events)
    kwargs["numba_thread_count"] = int(workers)
    start = time.perf_counter()
    diffraction.process_peaks_parallel(**kwargs)
    elapsed_s = time.perf_counter() - start
    stats = diffraction.get_last_process_peaks_weighted_event_stats()
    return {
        "elapsed_s": float(elapsed_s),
        "parallel_backend": str(stats.get("parallel_backend", "")),
        "parallel_worker_count": int(stats.get("parallel_worker_count", 0)),
        "parallel_requested_worker_count": stats.get("parallel_requested_worker_count"),
        "parallel_effective_worker_count": int(
            stats.get("parallel_effective_worker_count", 0)
        ),
        "parallel_worker_count_source": str(stats.get("parallel_worker_count_source", "")),
        "branch_representative_row_count": _representative_row_count(),
    }


def _validate_backend(*, requested_workers: int, n_samp: int, diagnostics: dict[str, Any]) -> None:
    backend = str(diagnostics.get("parallel_backend", ""))
    effective_workers = int(
        diagnostics.get(
            "parallel_effective_worker_count",
            diagnostics.get("parallel_worker_count", 0),
        )
    )
    if backend == "weighted_events_python":
        raise RuntimeError("weighted-events benchmark used Python fallback")
    if int(requested_workers) <= 1 or effective_workers <= 1:
        if backend not in {"fast_serial", "serial_njit"}:
            raise RuntimeError(f"threads=1 expected serial backend, got {backend!r}")
    elif int(n_samp) > 1 and backend != "threaded_njit_chunks":
        raise RuntimeError(
            f"threads={int(requested_workers)} expected threaded_njit_chunks, got {backend!r}"
        )


def run_benchmark(
    *,
    samples: int = 32,
    image_size: int = 64,
    events: int = 5,
    workers: tuple[int, ...] = (1, 2, 4),
    iterations: int = 3,
) -> dict[str, Any]:
    # Warm compile both serial and threaded paths before timing.
    _time_run(n_samp=max(2, min(samples, 4)), image_size=image_size, events=1, workers=1)
    if max(workers) > 1:
        _time_run(n_samp=max(2, min(samples, 4)), image_size=image_size, events=1, workers=2)

    results: dict[str, Any] = {}
    for worker_count in workers:
        total = 0.0
        last_diagnostics: dict[str, Any] = {}
        for _idx in range(int(iterations)):
            last_diagnostics = _time_run(
                n_samp=int(samples),
                image_size=int(image_size),
                events=int(events),
                workers=int(worker_count),
            )
            _validate_backend(
                requested_workers=int(worker_count),
                n_samp=int(samples),
                diagnostics=last_diagnostics,
            )
            total += float(last_diagnostics["elapsed_s"])
        avg = total / max(int(iterations), 1)
        prefix = f"threads_{int(worker_count)}"
        results[f"{prefix}_avg_s"] = avg
        results[f"{prefix}_parallel_backend"] = str(last_diagnostics["parallel_backend"])
        results[f"{prefix}_parallel_worker_count"] = int(
            last_diagnostics["parallel_worker_count"]
        )
        results[f"{prefix}_parallel_requested_worker_count"] = (
            last_diagnostics["parallel_requested_worker_count"]
        )
        results[f"{prefix}_parallel_effective_worker_count"] = int(
            last_diagnostics["parallel_effective_worker_count"]
        )
        results[f"{prefix}_parallel_worker_count_source"] = str(
            last_diagnostics["parallel_worker_count_source"]
        )
        results[f"{prefix}_branch_representative_row_count"] = int(
            last_diagnostics["branch_representative_row_count"]
        )
    baseline = results.get("workers_1_avg_s", 0.0)
    baseline = results.get("threads_1_avg_s", baseline)
    for worker_count in workers:
        key = f"threads_{int(worker_count)}_avg_s"
        if worker_count == 1 or results.get(key, 0.0) <= 0.0:
            continue
        results[f"threads_{int(worker_count)}_speedup_vs_1"] = float(baseline) / float(
            results[key]
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark weighted-event chunk backend.")
    parser.add_argument("--samples", "--n-samp", dest="samples", type=int, default=32, help="Beam samples per run.")
    parser.add_argument("--image-size", type=int, default=64, help="Square detector image size.")
    parser.add_argument("--events", type=int, default=5, help="Events per beam phase.")
    parser.add_argument("--workers", "--threads", dest="workers", type=int, nargs="+", default=[1, 2, 4], help="Worker counts.")
    parser.add_argument("--iterations", "--runs", dest="iterations", type=int, default=3, help="Timed iterations per worker count.")
    args = parser.parse_args()

    results = run_benchmark(
        samples=args.samples,
        image_size=args.image_size,
        events=args.events,
        workers=tuple(int(value) for value in args.workers),
        iterations=args.iterations,
    )
    print("Benchmark results:")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
