from __future__ import annotations

import os

import pytest

from scripts.benchmarks.benchmark_mosaic_profiles import run_benchmark


@pytest.mark.skipif(
    os.environ.get("RA_SIM_PERF_GATE", "0") != "1",
    reason="Set RA_SIM_PERF_GATE=1 to enable hardware-dependent performance gates.",
)
def test_generate_profiles_speedup_gate() -> None:
    results = run_benchmark(num_samples=50_000, iterations=3, seed=7)
    assert results["speedup_vs_baseline"] >= 1.2
