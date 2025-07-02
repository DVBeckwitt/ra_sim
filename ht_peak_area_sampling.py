#!/usr/bin/env python3
"""
ht_peak_area_sampling.py
────────────────────────
Monte-Carlo validation of the Hendricks–Teller integrated-area formula
using **numerical integration only** and providing four diagnostic plots:

  1. Scatter of ⟨A_num⟩ vs A_HT  (unity line)
  2. Histogram of all sample-wise ratios  A_num / A_HT
  3. Scatter of per-reflection σ (ratio_std) vs r
  4. Histogram of per-reflection mean ratios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ────────────────────────── User Settings ─────────────────────────────
H_MAX      = 4          # reflection range |H|,|K| ≤ H_MAX
K_MAX      = 4
N_SAMPLES  = 50         # Monte-Carlo samples per reflection
p          = 0.20       # stacking-fault probability
F0         = 1.0        # form-factor magnitude

noise_sd_rel = 5e-4     # Gaussian noise SD / I_max
nphi         = 4001     # φ samples over [−π, π]

seed    = 42
out_csv = Path("ht_peak_area_sampling_results.csv")
# ──────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(seed)
phi_axis = np.linspace(-np.pi, np.pi, nphi)

records = []
all_ratios = []  # store every sample’s ratio for histogram

for H in range(-H_MAX, H_MAX + 1):
    for K in range(-K_MAX, K_MAX + 1):
        if (2 * H + K) % 3 == 0:  # basal reflections → infinite A_HT
            continue

        _num = 2 * H + K
        if _num == 0:
            _num = K + 2 * H
        delta_phi = _num / 3.0
        f = (1 - p) + p * np.exp(-1j * delta_phi)
        r = abs(f)

        # Theoretical area (same for every sample)
        A_HT = 2 * np.pi * (F0 ** 2) * r ** 2 / (1 - r ** 2)

        # Exact noiseless profile (re-used across samples)
        I_th = (F0 ** 2) * r ** 2 / (1 + r ** 2 - 2 * r * np.cos(phi_axis))

        ratios = []
        for _ in range(N_SAMPLES):
            I_obs = I_th + noise_sd_rel * I_th.max() * rng.standard_normal(nphi)
            A_num = np.trapezoid(I_obs, phi_axis)
            ratio = A_num / A_HT
            ratios.append(ratio)
            all_ratios.append(ratio)

        ratios = np.asarray(ratios)
        records.append({
            "H": H, "K": K, "r": r,
            "ratio_mean": ratios.mean(),
            "ratio_std": ratios.std(),
            "samples": N_SAMPLES
        })

# ── DataFrame & global summary ──
df = pd.DataFrame(records)
print("=== HT peak-area Monte-Carlo diagnostics ===\n")
print(df.to_string(index=False, float_format="{:.4g}".format))
print("\nOverall mean ratio  = {:.4f}".format(df["ratio_mean"].mean()))
print("Overall std  ratio = {:.4f}".format(df["ratio_mean"].std()))
print("Average per-reflection σ = {:.4f}".format(df["ratio_std"].mean()))

# Save CSV
df.to_csv(out_csv, index=False)
print("\nPer-reflection stats saved →", out_csv.resolve())

# ── Diagnostic Plots ──
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# (1) Scatter mean A_num vs A_HT
ax = axes[0, 0]
A_HT_vals = 2 * np.pi * (F0 ** 2) * df["r"] ** 2 / (1 - df["r"] ** 2)
ax.scatter(A_HT_vals, A_HT_vals * df["ratio_mean"], alpha=0.8)
lims = [0, A_HT_vals.max() * 1.05]
ax.plot(lims, lims, "k--")
ax.set_xlabel(r"Analytical area $A_{HT}$")
ax.set_ylabel(r"Monte-Carlo mean $\langle A_{num}\rangle$")
ax.set_title("Mean numeric area vs theory")
ax.set_aspect("equal", "box")
ax.grid(True, ls=":", alpha=.4)

# (2) Histogram of all sample ratios
ax = axes[0, 1]
ax.hist(all_ratios, bins=30, edgecolor='k')
ax.set_title("Distribution of sample ratios")
ax.set_xlabel(r"Sample ratio $A_{num}/A_{HT}$")
ax.set_ylabel("Frequency")
ax.grid(axis='y', ls=":", alpha=.4)

# (3) Scatter σ vs r
ax = axes[1, 0]
ax.scatter(df["r"], df["ratio_std"], alpha=0.8)
ax.set_xlabel(r"Coherence factor $r$")
ax.set_ylabel(r"Std-dev of ratio $\sigma$")
ax.set_title(r"Per-reflection $\sigma$ vs $r$")
ax.grid(True, ls=":", alpha=.4)

# (4) Histogram of per-reflection mean ratios
ax = axes[1, 1]
ax.hist(df["ratio_mean"], bins=20, edgecolor='k')
ax.set_title("Distribution of reflection-mean ratios")
ax.set_xlabel(r"Reflection mean $\langle A_{num}/A_{HT}\rangle$")
ax.set_ylabel("Count")
ax.grid(axis='y', ls=":", alpha=.4)

plt.tight_layout()
plt.show()
