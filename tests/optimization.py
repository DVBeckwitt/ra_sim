# diffraction_local_lm.py
# =====================================================================
# Local Levenberg–Marquardt refinement of detector / geometry parameters.
# Outputs:
#   • best‑fit parameters with ±1 σ errors
#   • **correlation** matrix (printed + heat‑map)
#   • Two‑panel detector image:
#       – left  : refined parameters vs. blobs
#       – right : initial guess parameters vs. blobs
# ---------------------------------------------------------------------
# No shortcuts – full explicit script, ready to run.
# =====================================================================

from __future__ import annotations

import math, re, io, warnings
from pathlib import Path
from functools import partial
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal   import find_peaks, savgol_filter
from scipy.optimize import least_squares
from numpy.linalg    import inv

# third‑party crystallography utilities (project‑specific)
from CifFile                           import ReadCif
from ra_sim.utils.calculations         import IndexofRefraction
from ra_sim.utils.tools                import miller_generator
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction     import process_peaks_parallel
from ra_sim.io.file_parsing            import parse_poni_file

# ════════════════════════════════════════════════════════════════
# 1. Parsing helpers
# ════════════════════════════════════════════════════════════════

def parse_geometry(poni_path: Path) -> dict:
    p = parse_poni_file(poni_path)
    return {
        "dist":       p.get("Dist", 0.075),  # m
        "rot1":       p.get("Rot1", 0.0),
        "rot2":       p.get("Rot2", 0.0),
        "poni1":      p.get("Poni1", 0.0),  # m
        "poni2":      p.get("Poni2", 0.0),
        "wavelength": p.get("Wavelength", 1e-10),  # m
    }


def parse_cif(cif_path: Path) -> tuple[float, float]:
    cf    = ReadCif(str(cif_path))
    block = cf[next(iter(cf.keys()))]
    def _num(txt: str) -> float:
        m = re.match(r"[-+0-9\.Ee]+", str(txt))
        return float(m.group(0)) if m else math.nan
    return _num(block["_cell_length_a"]), _num(block["_cell_length_c"])

# ════════════════════════════════════════════════════════════════
# 2. Core simulation
# ════════════════════════════════════════════════════════════════

def run_simulation(
    cif_path: Path,
    geometry: dict,
    zb: float,
    zs: float,
    theta_initial: float,
    chi: float,
    *,
    allowed_keys=None,
    num_miller: int = 19,
    image_size: int = 3000,
    num_samples: int = 500,
    int_thresh: float = 1.0,
    two_theta_range=(0, 70),
    beam_div: float = 0.05,
    sigma_mosaic_deg: float = 0.8,
    gamma_mosaic_deg: float = 0.3,
    bandwidth_frac: float = 0.7 / 100,
):
    a_v, c_v = parse_cif(cif_path)
    lam      = geometry["wavelength"] * 1e10  # Å
    energy   = 6.62607015e-34 * 2.99792458e8 / (lam * 1e-10) / 1.602176634e-19

    miller, intens, *_ = miller_generator(
        num_miller, str(cif_path), [1.0, 1.0, 1.0], lam, energy, int_thresh, two_theta_range
    )
    if allowed_keys is not None:
        mask   = [tuple(map(abs, hkl)) in allowed_keys for hkl in miller]
        miller = miller[mask]
        intens = intens[mask]

    fwhm2sigma = 1 / (2 * math.sqrt(2 * math.log(2)))
    div_sigma  = math.radians(beam_div) * fwhm2sigma
    bw_sigma   = 0.05e-3 * fwhm2sigma
    bx, by, thetas, phis, lams = generate_random_profiles(
        num_samples, div_sigma, bw_sigma, lam, bandwidth_frac
    )

    img_buf = np.zeros((image_size, image_size), np.float64)
    n2      = IndexofRefraction()
    center  = (
        geometry["poni2"] / 100e-6,
        image_size - geometry["poni1"] / 100e-6,
    )

    img, hit_tables, *_ = process_peaks_parallel(
        miller, intens, image_size, a_v, c_v, lam, img_buf,
        geometry["dist"], geometry["rot1"], geometry["rot2"],
        chi, 0.0, zs, zb, n2,
        bx, by, thetas, phis,
        sigma_mosaic_deg, gamma_mosaic_deg, 0.05, lams,
        zs, zb, center, theta_initial,
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 0,
    )
    return img, hit_tables

# ════════════════════════════════════════════════════════════════
# 3. Peak detection utilities
# ════════════════════════════════════════════════════════════════

def detect_peaks(hit_tables, smooth_win=11, prominence_frac=0.05):
    rows = []
    for hits in hit_tables:
        arr = np.asarray(hits)
        if arr.size == 0:
            continue
        I, x, y, phi = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        H, K, L      = map(int, arr[0, 4:7])
        phi_deg      = np.rad2deg(phi)
        order        = np.argsort(phi_deg)
        I_s, phi_s   = I[order], phi_deg[order]
        x_s, y_s     = x[order], y[order]
        win          = min(smooth_win, len(I_s) - (len(I_s) + 1) % 2)
        I_sm         = savgol_filter(I_s, win, 2) if win >= 5 else I_s
        peaks, _     = find_peaks(I_sm, prominence=prominence_frac * I_sm.max())
        if peaks.size == 0:
            continue
        df = pd.DataFrame({
            "x_pix": np.round(x_s[peaks]).astype(int),
            "y_pix": np.round(y_s[peaks]).astype(int),
        }).drop_duplicates()
        need = 1 if (H == 0 and K == 0) else 2
        for _, r in df.head(need).iterrows():
            rows.append({"H": H, "K": K, "L": L, "x_pix": r.x_pix, "y_pix": r.y_pix})
    return pd.DataFrame(rows)


def load_blobs(blob_path: Path) -> pd.DataFrame:
    data = np.load(blob_path, allow_pickle=True)
    rows = []
    for e in data:
        h, k, l = map(int, e["label"].split(","))
        rows.append({"H": h, "K": k, "L": l, "x_pix": int(round(e["x"])), "y_pix": int(round(e["y"]))})
    return pd.DataFrame(rows)


def compute_shared(sim_df: pd.DataFrame, blob_df: pd.DataFrame):
    for d in (sim_df, blob_df):
        d[["|H|", "|K|", "|L|"]] = d[["H", "K", "L"]].abs()
    keys = set(zip(sim_df["|H|"], sim_df["|K|"], sim_df["|L|"]))
    mask = lambda df: df[df[["|H|", "|K|", "|L|"]].apply(tuple, axis=1).isin(keys)]
    return mask(sim_df), mask(blob_df)

# ════════════════════════════════════════════════════════════════
# 4. Residuals for LM
# ════════════════════════════════════════════════════════════════

def residuals(params, *, cif_file, blob_df_fixed, geometry, allowed_keys, sigma_px):
    hole = io.StringIO()
    with redirect_stdout(hole), redirect_stderr(hole), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
        try:
            img, hits = run_simulation(
                cif_file, geometry, params[0], params[1], params[3], params[2],
                allowed_keys=allowed_keys
            )
            sim_df      = detect_peaks(hits)
            sim_sh, bl_sh = compute_shared(sim_df, blob_df_fixed)
            if sim_sh.empty or bl_sh.empty:
                return np.full(4, 1e4)
            diffs = (sim_sh[["x_pix", "y_pix"]].values - bl_sh[["x_pix", "y_pix"]].values).ravel()
            return diffs / sigma_px
        except Exception:
            return np.full(4, 1e4)

# ════════════════════════════════════════════════════════════════
# 5. main()
# ════════════════════════════════════════════════════════════════

def main():
    # ---- file paths --------------------------------------------------
    cif_file  = Path(r"C:/Users/Kenpo/OneDrive/Research/Rigaku XRD/ORNL.07.25.2024/Analysis/Bi2Se3/Bi2Se3_test.cif")
    poni_file = Path(r"C:/Users/Kenpo/OneDrive/Research/Rigaku XRD/ORNL.07.25.2024/Analysis/geometry.poni")
    blob_file = Path(r"C:/Users/Kenpo/OneDrive/Documents/GitHub/blobs.npy")

    geometry = parse_geometry(poni_file)

    # blobs and allowed HKL set ---------------------------------------
    blob_df = load_blobs(blob_file)
    blob_df[["|H|", "|K|", "|L|"]] = blob_df[["H", "K", "L"]].abs()
    allowed_keys = set(zip(blob_df["|H|"], blob_df["|K|"], blob_df["|L|"]))

    # initial guess ----------------------------------------------------
    p0 = np.array([0.0, 0.0, 0.0, 7.0])  # zb, zs, chi, θ0

    res_fun = partial(
        residuals,
        cif_file=cif_file,
        blob_df_fixed=blob_df,
        geometry=geometry,
        allowed_keys=allowed_keys,
        sigma_px=1.0,
    )

    print("Starting Levenberg–Marquardt refinement …")
    lsq = least_squares(res_fun, p0, method="lm", jac="3-point", max_nfev=400)

    # best‑fit results --------------------------------------------------
    p_opt = lsq.x
    m, n = lsq.fun.size, p_opt.size
    cov  = inv(lsq.jac.T @ lsq.jac) * (2 * lsq.cost) / (m - n)
    sigma = np.sqrt(np.diag(cov))

    # correlation matrix ----------------------------------------------
    corr = cov / np.outer(sigma, sigma)

    labels = ["zb", "zs", "chi", "theta"]

    print("\n— optimisation finished —")
    print(f"χ²_red : {(2*lsq.cost)/(m-n):.4f}\n")
    for lbl, val, err in zip(labels, p_opt, sigma):
        print(f"{lbl:>6} = {val:+.6f} ± {err:.6f}")

    print("\nCorrelation matrix:")
    with np.printoptions(precision=3, suppress=True):
        print(corr)

    # ---- figure 1: correlation matrix ------------------------------
    plt.figure(figsize=(4, 4))
    im = plt.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="none")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="corr")
    plt.xticks(range(n), labels, rotation=45)
    plt.yticks(range(n), labels)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.show()

    # ---- detector overlays ----------------------------------------
    # refined parameters
    img_ref, hits_ref = run_simulation(
        cif_file, geometry, p_opt[0], p_opt[1], p_opt[3], p_opt[2],
        allowed_keys=allowed_keys
    )
    sim_df_ref = detect_peaks(hits_ref)
    sim_sh_ref, blob_sh = compute_shared(sim_df_ref, blob_df)

    # initial guess parameters
    img_init, hits_init = run_simulation(
        cif_file, geometry, p0[0], p0[1], p0[3], p0[2],
        allowed_keys=allowed_keys
    )
    sim_df_init = detect_peaks(hits_init)
    sim_sh_init, _ = compute_shared(sim_df_init, blob_df)

    # subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # refined overlay -------------------------------------------------
    ax[0].imshow(img_ref, cmap="gray", origin="lower")
    ax[0].scatter(sim_sh_ref["x_pix"], sim_sh_ref["y_pix"],
                  facecolors="none", edgecolors="g", s=80, label="Simulated (refined)")
    ax[0].scatter(blob_sh["x_pix"], blob_sh["y_pix"],
                  marker="+", color="r", s=70, label="Blobs")
    ax[0].set_title("Refined parameters")
    ax[0].set_xlabel("Pixel X"); ax[0].set_ylabel("Pixel Y")
    ax[0].legend(loc="upper right")

    # initial overlay -------------------------------------------------
    ax[1].imshow(img_init, cmap="gray", origin="lower")
    ax[1].scatter(sim_sh_init["x_pix"], sim_sh_init["y_pix"],
                  facecolors="none", edgecolors="b", s=80, label="Simulated (initial)")
    ax[1].scatter(blob_sh["x_pix"], blob_sh["y_pix"],
                  marker="+", color="r", s=70, label="Blobs")
    ax[1].set_title("Initial guess parameters")
    ax[1].set_xlabel("Pixel X")
    ax[1].legend(loc="upper right")

    fig.suptitle("Bragg peak positions versus blob detections")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
