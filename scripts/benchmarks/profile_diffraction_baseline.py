"""Run Phase-0 baseline profiling on a fixed dataset and fixed RNG seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import get_num_threads, set_num_threads

from ra_sim.simulation.diffraction import process_peaks_parallel_safe
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.profiling import (
    profile_process_peaks,
    save_simulation_artifact,
    timed_plot_image,
    write_profile_report,
)


def _scalar(dataset: dict[str, np.ndarray], key: str, *, dtype=float):
    value = np.asarray(dataset[key]).reshape(-1)[0]
    return dtype(value)


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {key: np.array(npz[key]) for key in npz.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="tests/benchmarks/reference_dataset.npz",
        help="Fixed reference dataset NPZ.",
    )
    parser.add_argument(
        "--seed-file",
        default="tests/benchmarks/reference_seed.json",
        help="JSON file containing {'random_seed': <int>}.",
    )
    parser.add_argument(
        "--report-out",
        default="tests/benchmarks/reference_profile_report.json",
        help="Where to write the profiling report JSON.",
    )
    parser.add_argument(
        "--plot-out",
        default="tests/benchmarks/reference_profile_image.png",
        help="Where to save the timed plot image.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warm-up simulation runs before timed profiling.",
    )
    parser.add_argument(
        "--artifact-out",
        default="",
        help=(
            "Optional NPZ output storing simulation image + per-peak centroid/FWHM/"
            "integrated-intensity summaries for validation comparisons."
        ),
    )
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        help="Optional fixed Numba thread count for deterministic runs.",
    )
    args = parser.parse_args()

    if args.numba_threads is not None:
        set_num_threads(max(1, int(args.numba_threads)))
    active_threads = int(get_num_threads())

    dataset_path = Path(args.dataset).resolve()
    seed_path = Path(args.seed_file).resolve()

    dataset = _load_dataset(dataset_path)
    seed_info = json.loads(seed_path.read_text(encoding="utf-8"))
    seed = int(seed_info["random_seed"])

    rng = np.random.default_rng(seed)
    beam_x, beam_y, theta, phi, wavelength = generate_random_profiles(
        _scalar(dataset, "num_samples", dtype=int),
        _scalar(dataset, "divergence_sigma_rad", dtype=float),
        _scalar(dataset, "bw_sigma", dtype=float),
        _scalar(dataset, "lambda_angstrom", dtype=float),
        _scalar(dataset, "bandwidth", dtype=float),
        rng=rng,
    )

    process_args = (
        np.asarray(dataset["miller"], dtype=np.float64),
        np.asarray(dataset["intensities"], dtype=np.float64),
        _scalar(dataset, "image_size", dtype=int),
        _scalar(dataset, "av", dtype=float),
        _scalar(dataset, "cv", dtype=float),
        _scalar(dataset, "lambda_angstrom", dtype=float),
        np.zeros(
            (
                _scalar(dataset, "image_size", dtype=int),
                _scalar(dataset, "image_size", dtype=int),
            ),
            dtype=np.float64,
        ),
        _scalar(dataset, "distance_m", dtype=float),
        _scalar(dataset, "gamma_deg", dtype=float),
        _scalar(dataset, "Gamma_deg", dtype=float),
        _scalar(dataset, "chi_deg", dtype=float),
        _scalar(dataset, "psi_deg", dtype=float),
        _scalar(dataset, "psi_z_deg", dtype=float),
        _scalar(dataset, "zs", dtype=float),
        _scalar(dataset, "zb", dtype=float),
        complex(_scalar(dataset, "n2_real", dtype=float), _scalar(dataset, "n2_imag", dtype=float)),
        np.asarray(beam_x, dtype=np.float64),
        np.asarray(beam_y, dtype=np.float64),
        np.asarray(theta, dtype=np.float64),
        np.asarray(phi, dtype=np.float64),
        _scalar(dataset, "sigma_mosaic_deg", dtype=float),
        _scalar(dataset, "gamma_mosaic_deg", dtype=float),
        _scalar(dataset, "eta", dtype=float),
        np.asarray(wavelength, dtype=np.float64),
        _scalar(dataset, "debye_x", dtype=float),
        _scalar(dataset, "debye_y", dtype=float),
        np.asarray(dataset["center"], dtype=np.float64),
        _scalar(dataset, "theta_initial_deg", dtype=float),
        _scalar(dataset, "cor_angle_deg", dtype=float),
        np.asarray(dataset["unit_x"], dtype=np.float64),
        np.asarray(dataset["n_detector"], dtype=np.float64),
    )
    process_kwargs = {"save_flag": 0, "record_status": False, "thickness": 0.0}

    for _ in range(max(0, int(args.warmup_runs))):
        process_peaks_parallel_safe(*process_args, **process_kwargs)

    result, metrics, timing_out = profile_process_peaks(
        *process_args,
        peak_runner=process_peaks_parallel_safe,
        **process_kwargs,
    )
    sim_image = np.asarray(result[0], dtype=np.float64)
    hit_tables = result[1]
    plotting_s = timed_plot_image(sim_image, output_path=args.plot_out)
    metrics["plotting_s"] = float(plotting_s)

    artifact_path = None
    if str(args.artifact_out).strip():
        artifact_path = save_simulation_artifact(
            args.artifact_out,
            image=sim_image,
            hit_tables=hit_tables,
            metadata={
                "dataset_path": str(dataset_path),
                "seed_path": str(seed_path),
                "random_seed": seed,
                "num_reflections": int(process_args[0].shape[0]),
                "num_samples": int(process_args[16].shape[0]),
            },
        )

    report = {
        "dataset_path": str(dataset_path),
        "seed_path": str(seed_path),
        "random_seed": seed,
        "num_reflections": int(process_args[0].shape[0]),
        "num_samples": int(process_args[16].shape[0]),
        "process_peaks_parallel_s": metrics["process_peaks_parallel_s"],
        "calculate_phi_s": metrics["calculate_phi_s"],
        "solve_q_s": metrics["solve_q_s"],
        "solve_q_calls": metrics["solve_q_calls"],
        "detector_projection_s": metrics["detector_projection_s"],
        "detector_projection_calls": metrics["detector_projection_calls"],
        "pixel_deposition_s": metrics["pixel_deposition_s"],
        "pixel_deposition_calls": metrics["pixel_deposition_calls"],
        "plotting_s": metrics["plotting_s"],
        "cycle_frequency_hz": metrics["cycle_frequency_hz"],
        "numba_threads": active_threads,
    }
    if artifact_path is not None:
        report["artifact_path"] = str(Path(artifact_path).resolve())
    write_profile_report(args.report_out, report)

    print("Phase-0 baseline profiling complete")
    for key in [
        "process_peaks_parallel_s",
        "calculate_phi_s",
        "solve_q_s",
        "detector_projection_s",
        "pixel_deposition_s",
        "plotting_s",
    ]:
        print(f"  {key}: {report[key]:.6f}")
    print(f"  solve_q_calls: {int(report['solve_q_calls'])}")
    print(f"  detector_projection_calls: {int(report['detector_projection_calls'])}")
    print(f"  pixel_deposition_calls: {int(report['pixel_deposition_calls'])}")
    print(f"  timing_out_shape: {tuple(np.asarray(timing_out).shape)}")
    print(f"  numba_threads: {active_threads}")
    if artifact_path is not None:
        print(f"  artifact_path: {Path(artifact_path).resolve()}")


if __name__ == "__main__":
    main()
