#!/usr/bin/env python3
"""Analyze missed diffracted rays from ``simulation.npz``.

``tests/run_diffraction_test.py`` stores outgoing wavevectors that failed
to intersect the detector plane in ``debug_info``.  Earlier versions only
recorded vectors from a single reflection.  Newer runs may store *all*
missed vectors as an object array of per-reflection tables.  This script
collects all vectors, reports how many were recorded and plots their
directions."""
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

    # ``dbg`` may be a simple ``(N,3)`` array or an object array of
    # per-reflection tables.  Normalize to a single ``(M,3)`` array.
    if dbg.dtype == object:
        dbg_all = np.vstack([arr for arr in dbg if len(arr) > 0])
    else:
        dbg_all = dbg

theta = np.rad2deg(np.arctan2(dbg_all[:, 2], np.sqrt(dbg_all[:, 0]**2 + dbg_all[:, 1]**2)))
phi = np.rad2deg(np.arctan2(dbg_all[:, 0], dbg_all[:, 1]))

print(f"total missed rays: {dbg_all.shape[0]}")

if solve_status is not None:
    unique, counts = np.unique(solve_status, return_counts=True)
    print("solve_q status codes:")
    for u, c in zip(unique, counts):
        print(f"  {int(u)}: {c}")

plt.figure(figsize=(6, 5))
plt.scatter(phi, theta, s=10, color="red")
plt.xlabel("phi (deg)")
plt.ylabel("theta (deg)")
plt.title("Missed ray directions")
plt.tight_layout()
plt.show()
