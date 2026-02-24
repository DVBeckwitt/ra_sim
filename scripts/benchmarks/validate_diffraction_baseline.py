"""Validate candidate diffraction output against a fixed baseline artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numba import get_num_threads, set_num_threads

from ra_sim.simulation.diffraction import process_peaks_parallel_safe
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.profiling import (
    compute_validation_metrics,
    load_simulation_artifact,
    save_simulation_artifact,
    summarize_peak_statistics,
    write_profile_report,
)


def _scalar(dataset: dict[str, np.ndarray], key: str, *, dtype=float):
    value = np.asarray(dataset[key]).reshape(-1)[0]
    return dtype(value)


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {key: np.array(npz[key]) for key in npz.files}


def _run_candidate_simulation(
    dataset: dict[str, np.ndarray],
    *,
    seed: int,
    warmup_runs: int,
) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    beam_x, beam_y, theta, phi, wavelength = generate_random_profiles(
        _scalar(dataset, "num_samples", dtype=int),
        _scalar(dataset, "divergence_sigma_rad", dtype=float),
        _scalar(dataset, "bw_sigma", dtype=float),
        _scalar(dataset, "lambda_angstrom", dtype=float),
        _scalar(dataset, "bandwidth", dtype=float),
        rng=rng,
    )

    image_size = _scalar(dataset, "image_size", dtype=int)
    process_args = (
        np.asarray(dataset["miller"], dtype=np.float64),
        np.asarray(dataset["intensities"], dtype=np.float64),
        image_size,
        _scalar(dataset, "av", dtype=float),
        _scalar(dataset, "cv", dtype=float),
        _scalar(dataset, "lambda_angstrom", dtype=float),
        np.zeros((image_size, image_size), dtype=np.float64),
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

    for _ in range(max(0, int(warmup_runs))):
        process_peaks_parallel_safe(*process_args, **process_kwargs)

    image, hit_tables, *_ = process_peaks_parallel_safe(*process_args, **process_kwargs)
    meta = {
        "random_seed": int(seed),
        "num_reflections": int(process_args[0].shape[0]),
        "num_samples": int(process_args[16].shape[0]),
    }
    return np.asarray(image, dtype=np.float64), hit_tables, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-artifact",
        default="tests/benchmarks/reference_baseline_artifact.npz",
        help="Baseline simulation artifact NPZ (saved by profile_diffraction_baseline.py --artifact-out).",
    )
    parser.add_argument(
        "--candidate-artifact",
        default="",
        help="Optional candidate artifact NPZ. If omitted, candidate is generated from dataset+seed.",
    )
    parser.add_argument(
        "--dataset",
        default="tests/benchmarks/reference_dataset.npz",
        help="Dataset NPZ used when generating a candidate artifact.",
    )
    parser.add_argument(
        "--seed-file",
        default="tests/benchmarks/reference_seed.json",
        help="Seed JSON used when generating a candidate artifact.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warm-up runs before generating a candidate artifact from dataset+seed.",
    )
    parser.add_argument(
        "--candidate-artifact-out",
        default="",
        help="Optional path to save the generated candidate artifact NPZ.",
    )
    parser.add_argument(
        "--report-out",
        default="tests/benchmarks/reference_validation_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        help="Optional fixed Numba thread count used when generating candidate output.",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_artifact).resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline artifact not found: {baseline_path}\n"
            "Create it with:\n"
            "  python scripts/benchmarks/profile_diffraction_baseline.py "
            "--artifact-out tests/benchmarks/reference_baseline_artifact.npz"
        )

    baseline = load_simulation_artifact(baseline_path)

    candidate_source = ""
    dataset_path = None
    seed_path = None
    seed = None

    if str(args.candidate_artifact).strip():
        candidate_path = Path(args.candidate_artifact).resolve()
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate artifact not found: {candidate_path}")
        candidate = load_simulation_artifact(candidate_path)
        candidate_source = str(candidate_path)
    else:
        if args.numba_threads is not None:
            set_num_threads(max(1, int(args.numba_threads)))
        dataset_path = Path(args.dataset).resolve()
        seed_path = Path(args.seed_file).resolve()
        dataset = _load_dataset(dataset_path)
        seed_info = json.loads(seed_path.read_text(encoding="utf-8"))
        seed = int(seed_info["random_seed"])

        candidate_image, candidate_hits, candidate_meta = _run_candidate_simulation(
            dataset,
            seed=seed,
            warmup_runs=int(args.warmup_runs),
        )
        candidate_path = None
        if str(args.candidate_artifact_out).strip():
            candidate_path = save_simulation_artifact(
                args.candidate_artifact_out,
                image=candidate_image,
                hit_tables=candidate_hits,
                metadata={
                    "dataset_path": str(dataset_path),
                    "seed_path": str(seed_path),
                    **candidate_meta,
                },
            )
        if candidate_path is not None:
            candidate_source = str(Path(candidate_path).resolve())
            candidate = load_simulation_artifact(candidate_path)
        else:
            peak_stats = summarize_peak_statistics(candidate_hits)
            candidate = {
                "image": np.asarray(candidate_image, dtype=np.float32),
                "centroid_col": peak_stats["centroid_col"],
                "centroid_row": peak_stats["centroid_row"],
                "fwhm_px": peak_stats["fwhm_px"],
                "integrated_intensity": peak_stats["integrated_intensity"],
                "num_hits": peak_stats["num_hits"],
                "metadata_json": np.asarray(
                    json.dumps(
                        {
                            "dataset_path": str(dataset_path),
                            "seed_path": str(seed_path),
                            **candidate_meta,
                        },
                        sort_keys=True,
                    )
                ),
            }
            candidate_source = "generated-in-memory"

    metrics = compute_validation_metrics(baseline, candidate)
    report = {
        "baseline_artifact": str(baseline_path),
        "candidate_source": candidate_source,
        "dataset_path": str(dataset_path) if dataset_path is not None else "",
        "seed_path": str(seed_path) if seed_path is not None else "",
        "random_seed": int(seed) if seed is not None else -1,
        "numba_threads": int(get_num_threads()),
        **metrics,
    }
    write_profile_report(args.report_out, report)

    print("Validation complete")
    print(f"  pixel residual max:  {report['pixel_residual_max']:.6f}")
    print(f"  pixel residual mean: {report['pixel_residual_mean']:.6f}")
    print(
        "  peak centroid shift (px): "
        f"mean={report['peak_centroid_shift_px_mean']:.6f}, "
        f"max={report['peak_centroid_shift_px_max']:.6f}"
    )
    print(
        "  FWHM shift (%): "
        f"mean={report['fwhm_shift_pct_mean']:.6f}, "
        f"max_abs={report['fwhm_shift_pct_max_abs']:.6f}"
    )
    print(
        "  integrated intensity shift (%): "
        f"mean={report['integrated_intensity_shift_pct_mean']:.6f}, "
        f"max_abs={report['integrated_intensity_shift_pct_max_abs']:.6f}"
    )


if __name__ == "__main__":
    main()
