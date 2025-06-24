#!/usr/bin/env python3
# diffraction_global_local.py
# ===============================================================
# 1. Differential Evolution (global)  ➜  2. Levenberg–Marquardt (local)
# Outputs:
#   • DE summary + best parameters
#   • LM best parameters ±1 σ and correlation matrix
#   • Two-panel detector overlay (refined vs initial)
# ===============================================================

from __future__ import annotations
import math, re, io, warnings, sys, time
from pathlib import Path
from functools import partial
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal   import find_peaks, savgol_filter
from scipy.optimize import least_squares, differential_evolution
from numpy.linalg   import inv

# ───────────────── third-party / project modules ────────────────
from CifFile                           import ReadCif
from ra_sim.utils.calculations         import IndexofRefraction
from ra_sim.utils.tools                import miller_generator
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction     import process_peaks_parallel
from ra_sim.io.file_parsing            import parse_poni_file
from ra_sim.path_config                import get_path, get_dir

# ════════════════════════════════════════════════════════════════
# 1. Parsing helpers
# ════════════════════════════════════════════════════════════════
def parse_geometry(poni_path: Path) -> dict:
    p = parse_poni_file(poni_path)
    return {
        "dist":       p.get("Dist", 0.075),      # m
        "rot1":       p.get("Rot1", 0.0),
        "rot2":       p.get("Rot2", 0.0),
        "poni1":      p.get("Poni1", 0.0),       # m
        "poni2":      p.get("Poni2", 0.0),
        "wavelength": p.get("Wavelength", 1e-10) # m
    }

def parse_cif(cif_path: Path) -> tuple[float, float]:
    cf    = ReadCif(str(cif_path))
    block = cf[next(iter(cf.keys()))]
    num   = lambda s: float(re.match(r"[-+0-9\.Ee]+", str(s)).group(0))
    return num(block["_cell_length_a"]), num(block["_cell_length_c"])

# ════════════════════════════════════════════════════════════════
# 2. Core simulation  (unchanged)
# ════════════════════════════════════════════════════════════════
def run_simulation(
    cif_path: Path,
    geometry: dict,
    zb: float, zs: float, theta_initial: float, chi: float,
    *,
    miller_precomputed=None, intens_precomputed=None,
    profiles=None,
    allowed_keys=None, num_miller: int = 19,
    image_size: int = 3000, num_samples: int = 500,
    int_thresh: float = 1.0, two_theta_range=(0, 70),
    beam_div: float = 0.05,
    sigma_mosaic_deg: float = 0.8, gamma_mosaic_deg: float = 0.3,
    bandwidth_frac: float = 0.7/100,
):
    # ── Miller indices (pre-compute friendly) ──
    if miller_precomputed is None:
        a_v, c_v = parse_cif(cif_path)
        lam      = geometry["wavelength"] * 1e10  # Å
        energy   = 6.62607015e-34*2.99792458e8/(lam*1e-10)/1.602176634e-19
        miller, intens, *_ = miller_generator(
            num_miller, str(cif_path), [1.0,1.0,1.0], lam, energy,
            int_thresh, two_theta_range
        )
    else:
        miller, intens = miller_precomputed, intens_precomputed

    if allowed_keys is not None:
        mask = [tuple(map(abs,hkl)) in allowed_keys for hkl in miller]
        miller, intens = miller[mask], intens[mask]

    # ── Beam/divergence profile (optionally cached) ──
    if profiles is None:
        fwhm2sigma = 1/(2*math.sqrt(2*math.log(2)))
        div_sigma  = math.radians(beam_div) * fwhm2sigma
        bw_sigma   = 0.05e-3 * fwhm2sigma
        profiles   = generate_random_profiles(num_samples, div_sigma, bw_sigma,
                                              geometry["wavelength"]*1e10,
                                              bandwidth_frac)
    bx, by, thetas, phis, lams = profiles

    img_buf = np.zeros((image_size, image_size), np.float64)
    n2      = IndexofRefraction()
    center  = np.array([
        geometry["poni2"]/100e-6,
        image_size - geometry["poni1"]/100e-6
    ], dtype=np.float64)

    img, hit_tables, *_ = process_peaks_parallel(
        np.ascontiguousarray(miller, dtype=np.float64),
        np.ascontiguousarray(intens, dtype=np.float64),
        *parse_cif(cif_path), geometry["wavelength"]*1e10, img_buf,
        geometry["dist"], geometry["rot1"], geometry["rot2"],
        chi, 0.0, zs, zb, n2,
        bx, by, thetas, phis,
        sigma_mosaic_deg, gamma_mosaic_deg, 0.05, lams,
        zs, zb, center, theta_initial,
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 0,
    )
    return img, hit_tables, (miller, intens), profiles

# ════════════════════════════════════════════════════════════════
# 3. Peak detection + helpers  (unchanged from your script)
# ════════════════════════════════════════════════════════════════
def detect_peaks(hit_tables, smooth_win=11, prominence_frac=0.05):
    rows=[]
    for hits in hit_tables:
        a=np.asarray(hits)
        if a.size==0: continue
        I,x,y,phi=a[:,0],a[:,1],a[:,2],a[:,3]
        H,K,L=map(int,a[0,4:7]); phi_deg=np.rad2deg(phi)
        order=np.argsort(phi_deg)
        I_s,phi_s,x_s,y_s=I[order],phi_deg[order],x[order],y[order]
        win=min(smooth_win,len(I_s)-(len(I_s)+1)%2)
        I_sm=savgol_filter(I_s,win,2) if win>=5 else I_s
        peaks,_=find_peaks(I_sm,prominence=prominence_frac*I_sm.max())
        if peaks.size==0: continue
        df=pd.DataFrame({"x_pix":np.round(x_s[peaks]).astype(int),
                         "y_pix":np.round(y_s[peaks]).astype(int)}).drop_duplicates()
        need=1 if (H==0 and K==0) else 2
        for _,r in df.head(need).iterrows():
            rows.append({"H":H,"K":K,"L":L,"x_pix":r.x_pix,"y_pix":r.y_pix})
    return pd.DataFrame(rows)

def load_blobs(blob_path:Path)->pd.DataFrame:
    data=np.load(blob_path,allow_pickle=True)
    rows=[]
    for e in data:
        h,k,l=map(int,e["label"].split(","))
        rows.append({"H":h,"K":k,"L":l,
                     "x_pix":int(round(e["x"])),
                     "y_pix":int(round(e["y"]))})
    return pd.DataFrame(rows)

def compute_shared(sim_df,blob_df):
    for d in (sim_df,blob_df):
        d[["|H|","|K|","|L|"]]=d[["H","K","L"]].abs()
    keys=set(zip(sim_df["|H|"],sim_df["|K|"],sim_df["|L|"]))
    mask=lambda df: df[df[["|H|","|K|","|L|"]].apply(tuple,axis=1).isin(keys)]
    return mask(sim_df),mask(blob_df)

# ════════════════════════════════════════════════════════════════
# 4. Objective + wrappers
# ════════════════════════════════════════════════════════════════
def residuals(params,*,
              cif_file,blob_df_fixed,geometry,
              mill_intens,profiles,allowed_keys,sigma_px):
    # params=[zb,zs,chi,theta0]
    hole=io.StringIO()
    with redirect_stdout(hole),redirect_stderr(hole),warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=pd.errors.SettingWithCopyWarning)
        try:
            img,hits,_,_=run_simulation(
                cif_file,geometry,params[0],params[1],params[3],params[2],
                miller_precomputed=mill_intens[0],
                intens_precomputed=mill_intens[1],
                profiles=profiles,
                allowed_keys=allowed_keys
            )
            sim_df=detect_peaks(hits)
            s,b=compute_shared(sim_df,blob_df_fixed)
            if s.empty or b.empty:
                return np.full(4,1e4)
            diffs=(s[["x_pix","y_pix"]].values -
                   b[["x_pix","y_pix"]].values).ravel()
            return diffs/sigma_px
        except Exception:
            return np.full(4,1e4)

def chi2_scalar(params,**kwargs):
    r=residuals(params,**kwargs)
    return float(np.sum(r*r))
def plot_shared_overlay(
    img: np.ndarray,
    hits: list,
    blob_df: pd.DataFrame,
    out_path: Path,
    det_size: int = 3000,
    title: str = "Shared diffraction peaks – simulation vs. blobs"
):
    """
    Given a detector image `img`, the list of `hits` from run_simulation(),
    and a blob_df with columns [H,K,L,x_pix,y_pix],
    overlay *only* those reflections present in both simulation & blobs,
    save to `out_path`, and show the figure.
    """
    # 1) build sim_df and add |H|,|K|,|L|
    sim_df = detect_peaks(hits)
    if sim_df.empty:
        print("⚠️  No simulated peaks detected—skipping overlay.")
        return

    sim_df[["|H|","|K|","|L|"]]  = sim_df[["H","K","L"]].abs()
    blob_df[["|H|","|K|","|L|"]] = blob_df[["H","K","L"]].abs()

    # 2) find shared keys
    sim_keys  = set(zip(sim_df["|H|"], sim_df["|K|"], sim_df["|L|"]))
    blob_keys = set(zip(blob_df["|H|"],blob_df["|K|"],blob_df["|L|"]))
    shared    = sim_keys & blob_keys
    if not shared:
        print("⚠️  No shared HKL between sim & blobs—skipping overlay.")
        return

    sim_shared  = sim_df[sim_df[["|H|","|K|","|L|"]]
                         .apply(tuple, axis=1).isin(shared)]
    blob_shared = blob_df[blob_df[["|H|","|K|","|L|"]]
                          .apply(tuple, axis=1).isin(shared)]

    # 3) plot
    vmin, vmax = 0.0, np.percentile(img, 99.5)
    plt.figure(figsize=(8,8), dpi=150)
    plt.imshow(img, cmap="turbo", vmin=vmin, vmax=vmax, origin="upper")

    # red squares + labels
    plt.scatter(sim_shared.x_pix, sim_shared.y_pix,
                s=36, facecolors="none", edgecolors="red",
                lw=1.2, label="sim (shared)")
    for _, r in sim_shared.iterrows():
        plt.text(r.x_pix + 6, r.y_pix - 6,
                 f"({r.H} {r.K} {r.L})", color="red",
                 fontsize=6, ha="left", va="top")

    # blue pluses
    plt.scatter(blob_shared.x_pix, blob_shared.y_pix,
                marker="+", s=40, lw=1.2,
                color="blue", label="blob (shared)")

    plt.title(title)
    plt.xlim(0, det_size); plt.ylim(det_size, 0)
    plt.gca().set_aspect("equal", "box")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    print(f"[saved overlay] {out_path}")
    plt.show()


# ════════════════════════════════════════════════════════════════
# 5. main()
# ════════════════════════════════════════════════════════════════
def main():
    # ── file paths ───────────────────────────────────────────────
    cif_file  = Path(get_path("test_cif_file"))
    poni_file = Path(get_path("test_poni_file"))
    blob_file = Path(get_path("test_blob_file"))
    DET_SIZE = 3000
    overlay_dir = get_dir("overlay_dir")
    FIG_OUT = overlay_dir / "overlay.png"
    geometry = parse_geometry(poni_file)
    blob_df  = load_blobs(blob_file)
    blob_df[["|H|","|K|","|L|"]]=blob_df[["H","K","L"]].abs()
    allowed_keys=set(zip(blob_df["|H|"],blob_df["|K|"],blob_df["|L|"]))

    # ── precompute simulation once ────────────────────────────────
    print("Pre-computing Miller & profiles …", flush=True)
    dummy_img, dummy_hits, mill_intens, profiles = run_simulation(
        cif_file, geometry, 0.0, 0.0, 7.0, 0.0,
        allowed_keys=set(zip(blob_df["H"].abs(),
                             blob_df["K"].abs(),
                             blob_df["L"].abs()))
    )

    # ── call our new figure-maker ─────────────────────────────────
    plot_shared_overlay(
        img=dummy_img,
        hits=dummy_hits,
        blob_df=blob_df,
        out_path=FIG_OUT,
        det_size=DET_SIZE,
        title="Initial-guess shared peaks"
    )


    # ── objective wrappers ───────────────────────────────────────
    obj_kwargs=dict(
        cif_file=cif_file,
        blob_df_fixed=blob_df,
        geometry=geometry,
        mill_intens=mill_intens,
        profiles=profiles,
        allowed_keys=allowed_keys,
        sigma_px=1.0,
    )
    res_fun = partial(residuals,**obj_kwargs)
    chi2    = partial(chi2_scalar,**obj_kwargs)

    # ════════════════════════════════════════════════════════════
    # 5A.  Differential Evolution (global)
    # ════════════════════════════════════════════════════════════
    bounds=[(-0.005,0.005),   # zb (m)
            (-0.005,0.005),   # zs (m)
            (-2.0,  2.0),     # chi (deg)
            ( 5.0,  9.0)]     # theta0 (deg)
    print("\n=== Global search: Differential Evolution ===")
    t0=time.time()
    de=differential_evolution(
        chi2, bounds=bounds, strategy="best1bin",
        maxiter=40, popsize=12, polish=False, workers=12, disp=True)
    print(f"DE finished in {time.time()-t0:.1f}s  |  χ²={de.fun:.2f}")
    p_de=de.x
    print("DE best parameters:",p_de)

    # ════════════════════════════════════════════════════════════
    # 5B.  Levenberg–Marquardt (local)
    # ════════════════════════════════════════════════════════════
    print("\n=== Local refinement: Levenberg–Marquardt ===")
    scales=np.array([1e-3,1e-3,1.0,1.0])   # simple scaling
    lsq=least_squares(res_fun,p_de/scales,method="lm",
                      jac="3-point",diff_step=1e-5,max_nfev=400,verbose=2)
    p_opt=lsq.x*scales
    m,n=lsq.fun.size,p_opt.size
    cov=inv(lsq.jac.T@lsq.jac) * (2*lsq.cost)/(m-n)
    sigma=np.sqrt(np.diag(cov))
    corr=cov/np.outer(sigma,sigma)
    labels=["zb","zs","chi","theta"]

    print("\n— optimisation finished —")
    print(f"χ²_red : {(2*lsq.cost)/(m-n):.4f}")
    for lbl,val,err in zip(labels,p_opt,sigma):
        print(f"{lbl:>6} = {val:+.6f} ± {err:.6f}")
    print("\nCorrelation matrix:")
    with np.printoptions(precision=3,suppress=True):
        print(corr)
    # ── after the LM call ───────────────────────────────────────────
    p_opt = lsq.x * scales
    m, n   = lsq.fun.size, p_opt.size

    print(f"\nLM finished: residuals m = {m}, parameters n = {n}")

    if m <= n:
        print("⚠️  Not enough residuals to estimate uncertainties "
            f"(m – n = {m-n}).\n"
            "   • Increase the number of matched peaks ↔ set "
            "`num_miller`, widen `two_theta_range`, or\n"
            "   • relax the blob-matching criteria so more HKL pairs survive.")
        cov   = None
        sigma = None
        corr  = None
    else:
        JTJ   = lsq.jac.T @ lsq.jac
        try:
            JTJ_inv = np.linalg.inv(JTJ)
        except np.linalg.LinAlgError:
            print("⚠️  Jacobian is singular; falling back to pseudo-inverse.")
            JTJ_inv = np.linalg.pinv(JTJ, rcond=1e-10)

        cov   = JTJ_inv * (2*lsq.cost)/(m-n)
        sigma = np.sqrt(np.diag(cov))
        corr  = cov / np.outer(sigma, sigma)

    # ════════════════════════════════════════════════════════════
    # 6.  Figures
    # ════════════════════════════════════════════════════════════
    plt.figure(figsize=(4,4))
    im=plt.imshow(corr,vmin=-1.0,vmax=1.0,cmap="coolwarm",interpolation="none")
    plt.colorbar(im,fraction=0.046,pad=0.04,label="corr")
    plt.xticks(range(n),labels,rotation=45); plt.yticks(range(n),labels)
    plt.title("Correlation matrix"); plt.tight_layout()

    # ── detector overlays ───────────────────────────────────────
    img_ref,hits_ref,_,_=run_simulation(
        cif_file,geometry,p_opt[0],p_opt[1],p_opt[3],p_opt[2],
        miller_precomputed=mill_intens[0],intens_precomputed=mill_intens[1],
        profiles=profiles,allowed_keys=allowed_keys)
    sim_df_ref=detect_peaks(hits_ref); sim_sh_ref,blob_sh=compute_shared(sim_df_ref,blob_df)

    img_init,hits_init,_,_=run_simulation(
        cif_file,geometry,p_de[0],p_de[1],p_de[3],p_de[2],
        miller_precomputed=mill_intens[0],intens_precomputed=mill_intens[1],
        profiles=profiles,allowed_keys=allowed_keys)
    sim_df_init=detect_peaks(hits_init); sim_sh_init,_=compute_shared(sim_df_init,blob_df)

    fig,ax=plt.subplots(1,2,figsize=(12,6),sharex=True,sharey=True)
    ax[0].imshow(img_ref,cmap="gray",origin="lower")
    ax[0].scatter(sim_sh_ref["x_pix"],sim_sh_ref["y_pix"],
                  facecolors="none",edgecolors="g",s=80,label="Sim (LM)")
    ax[0].scatter(blob_sh["x_pix"],blob_sh["y_pix"],
                  marker="+",color="r",s=70,label="Blobs")
    ax[0].set_title("Refined parameters"); ax[0].set_xlabel("Pixel X"); ax[0].set_ylabel("Pixel Y"); ax[0].legend()

    ax[1].imshow(img_init,cmap="gray",origin="lower")
    ax[1].scatter(sim_sh_init["x_pix"],sim_sh_init["y_pix"],
                  facecolors="none",edgecolors="b",s=80,label="Sim (DE)")
    ax[1].scatter(blob_sh["x_pix"],blob_sh["y_pix"],
                  marker="+",color="r",s=70,label="Blobs")
    ax[1].set_title("DE best parameters"); ax[1].set_xlabel("Pixel X"); ax[1].legend()

    fig.suptitle("Bragg peak positions – global vs local fits")
    plt.tight_layout(); plt.show()

if __name__=="__main__":
    main()
