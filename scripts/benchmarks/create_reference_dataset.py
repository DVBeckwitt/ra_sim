"""Create the fixed baseline diffraction dataset used for optimization profiling."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ra_sim.utils.calculations import IndexofRefraction


def build_reference_reflections() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic set of HKL reflections and intensities."""

    hkls: list[tuple[int, int, int]] = []
    for h in range(0, 9):
        for k in range(0, 9):
            radial = h * h + h * k + k * k
            if radial > 22:
                continue
            for l in range(1, 12):
                if radial + l * l > 90:
                    continue
                hkls.append((h, k, l))

    miller = np.asarray(hkls, dtype=np.float64)

    h = miller[:, 0]
    k = miller[:, 1]
    l = miller[:, 2]
    radial = h * h + h * k + k * k
    intensities = (
        np.exp(-0.12 * radial - 0.06 * l * l)
        * (1.0 + 0.15 * ((h + 2.0 * k + l) % 5.0))
        * (1.0 + 0.05 * (h == 0.0))
    )
    intensities = 100.0 * intensities / np.max(intensities)
    return miller, intensities.astype(np.float64)


def build_reference_payload() -> dict[str, np.ndarray]:
    """Build the fixed simulation input payload."""

    fwhm_to_sigma = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    miller, intensities = build_reference_reflections()

    image_size = 896
    center = np.array(
        [
            image_size * 0.52,
            image_size * 0.48,
        ],
        dtype=np.float64,
    )

    payload: dict[str, np.ndarray] = {
        "miller": miller,
        "intensities": intensities,
        "image_size": np.array([image_size], dtype=np.int64),
        "av": np.array([4.31], dtype=np.float64),
        "cv": np.array([6.92], dtype=np.float64),
        "lambda_angstrom": np.array([1.5406], dtype=np.float64),
        "distance_m": np.array([0.075], dtype=np.float64),
        "gamma_deg": np.array([0.0], dtype=np.float64),
        "Gamma_deg": np.array([0.0], dtype=np.float64),
        "chi_deg": np.array([0.0], dtype=np.float64),
        "psi_deg": np.array([0.0], dtype=np.float64),
        "psi_z_deg": np.array([0.0], dtype=np.float64),
        "zs": np.array([0.0], dtype=np.float64),
        "zb": np.array([0.0], dtype=np.float64),
        "debye_x": np.array([0.0], dtype=np.float64),
        "debye_y": np.array([0.0], dtype=np.float64),
        "theta_initial_deg": np.array([6.0], dtype=np.float64),
        "cor_angle_deg": np.array([0.0], dtype=np.float64),
        "sigma_mosaic_deg": np.array([0.8 * fwhm_to_sigma], dtype=np.float64),
        "gamma_mosaic_deg": np.array([0.7 * fwhm_to_sigma], dtype=np.float64),
        "eta": np.array([0.05], dtype=np.float64),
        "num_samples": np.array([1200], dtype=np.int64),
        "divergence_sigma_rad": np.array([math.radians(0.05 * fwhm_to_sigma)], dtype=np.float64),
        "bw_sigma": np.array([0.05e-3 * fwhm_to_sigma], dtype=np.float64),
        "bandwidth": np.array([0.7 / 100.0], dtype=np.float64),
        "center": center,
        "unit_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "n_detector": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    }

    n2 = IndexofRefraction()
    payload["n2_real"] = np.array([np.real(n2)], dtype=np.float64)
    payload["n2_imag"] = np.array([np.imag(n2)], dtype=np.float64)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="tests/benchmarks/reference_dataset.npz",
        help="Output NPZ file for the fixed baseline dataset.",
    )
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **build_reference_payload())
    print(f"Wrote reference dataset: {out_path}")


if __name__ == "__main__":
    main()
