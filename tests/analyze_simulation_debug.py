#!/usr/bin/env python3
"""Analyze beam paths from ``simulation.npz``.

This script inspects the ``debug_info`` array stored by
``tests/run_diffraction_test.py``.  Each row in ``debug_info`` contains
``(theta, phi, hit_sample, hit_detector)`` for one beam sample.
The goal is to understand which samples miss the detector to diagnose the
horizontal black band issue.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# location of the NPZ written by run_diffraction_test.py
NPZ_PATH = Path(__file__).resolve().parent / "simulation.npz"

if not NPZ_PATH.exists():
    raise SystemExit(f"File not found: {NPZ_PATH}")

with np.load(NPZ_PATH, allow_pickle=True) as data:
    try:
        dbg = data["debug_info"]
    except KeyError:
        raise SystemExit("debug_info entry not found in npz file")
    solve_status = data["solve_status"] if "solve_status" in data else None

theta = np.rad2deg(dbg[:, 0])
phi = np.rad2deg(dbg[:, 1])
hit_sample = dbg[:, 2] > 0.5
hit_detector = dbg[:, 3] > 0.5

n_total = dbg.shape[0]
n_hit_sample = np.count_nonzero(hit_sample)
n_hit_det = np.count_nonzero(hit_detector)

print(f"total samples: {n_total}")
print(f"hit sample:    {n_hit_sample} ({n_hit_sample/n_total:.1%})")
print(f"hit detector:  {n_hit_det} ({n_hit_det/n_total:.1%})")
print(f"missed sample: {n_total - n_hit_sample}")
print(f"missed detector after sample hit: {n_hit_sample - n_hit_det}")

if solve_status is not None:
    unique, counts = np.unique(solve_status, return_counts=True)
    print("solve_q status codes:")
    for u, c in zip(unique, counts):
        print(f"  {int(u)}: {c}")

# classify for scatter plot
cls = np.full(n_total, 0)
cls[hit_sample] = 1
cls[hit_detector] = 2
colors = np.array(["red", "orange", "green"])[cls]
labels = {0: "missed sample", 1: "missed detector", 2: "hit detector"}

plt.figure(figsize=(6, 5))
for c in np.unique(cls):
    mask = cls == c
    plt.scatter(phi[mask], theta[mask], s=10, c=colors[mask], label=labels[c])

plt.xlabel("phi (deg)")
plt.ylabel("theta (deg)")
plt.title("Beam path classification")
plt.legend()
plt.tight_layout()
plt.show()
