"""Benchmark random profile generation.

This benchmark compares the vectorized implementation used by RA-SIM against a
pure-Python baseline implementation. The pure-Python baseline is intentionally
simple and portable so speedup numbers are stable across machines.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np

from ra_sim.simulation.mosaic_profiles import generate_random_profiles


def _baseline_python(
    num_samples: int,
    divergence_sigma: float,
    bw_sigma: float,
    lambda0: float,
    bandwidth: float,
    rng: np.random.Generator,
):
    beam_x = np.empty(num_samples, dtype=np.float64)
    beam_y = np.empty(num_samples, dtype=np.float64)
    theta = np.empty(num_samples, dtype=np.float64)
    phi = np.empty(num_samples, dtype=np.float64)
    wavelength = np.empty(num_samples, dtype=np.float64)

    for i in range(num_samples):
        theta[i] = rng.normal(0.0, divergence_sigma)
        phi[i] = rng.normal(0.0, divergence_sigma)
        beam_x[i] = rng.normal(0.0, bw_sigma)
        beam_y[i] = rng.normal(0.0, bw_sigma)
        wavelength[i] = rng.normal(lambda0, lambda0 * bandwidth)

    return beam_x, beam_y, theta, phi, wavelength


def run_benchmark(
    *,
    num_samples: int = 200_000,
    iterations: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    divergence_sigma = 0.01
    bw_sigma = 0.005
    lambda0 = 1.54
    bandwidth = 0.007

    t_vectorized = 0.0
    t_baseline = 0.0

    for idx in range(iterations):
        rng_vec = np.random.default_rng(seed + idx)
        start = time.perf_counter()
        generate_random_profiles(
            num_samples,
            divergence_sigma,
            bw_sigma,
            lambda0,
            bandwidth,
            rng=rng_vec,
        )
        t_vectorized += time.perf_counter() - start

        rng_base = np.random.default_rng(seed + idx)
        start = time.perf_counter()
        _baseline_python(
            num_samples,
            divergence_sigma,
            bw_sigma,
            lambda0,
            bandwidth,
            rng_base,
        )
        t_baseline += time.perf_counter() - start

    avg_vectorized = t_vectorized / iterations
    avg_baseline = t_baseline / iterations
    speedup = avg_baseline / avg_vectorized if avg_vectorized > 0 else float("inf")
    return {
        "avg_vectorized_s": avg_vectorized,
        "avg_baseline_s": avg_baseline,
        "speedup_vs_baseline": speedup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RA-SIM profile generation.")
    parser.add_argument("--samples", type=int, default=200_000, help="Number of samples per run.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of timed iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    args = parser.parse_args()

    results = run_benchmark(
        num_samples=args.samples,
        iterations=args.iterations,
        seed=args.seed,
    )
    print("Benchmark results:")
    for key, val in results.items():
        print(f"  {key}: {val:.6f}")


if __name__ == "__main__":
    main()
