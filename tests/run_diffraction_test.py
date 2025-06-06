#!/usr/bin/env python3
"""
Headless one-shot diffraction simulation.
Produces <YYYYMMDD_HHMMSS>_simulation.npy and shows the image.
(No GUI, no background subtraction.)
"""

# ───────────── imports ─────────────
import math, re, numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from datetime import datetime
from pathlib import Path
import CifFile

# ra_sim internals
from ra_sim.utils.calculations   import IndexofRefraction
from ra_sim.utils.tools          import miller_generator
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction     import process_peaks_parallel, debug_detector_paths
from ra_sim.io.file_parsing      import parse_poni_file
from ra_sim.path_config          import get_path
import io, contextlib
from CifFile import ReadCif

def silent_ReadCif(path):
    buf = io.StringIO()
    # divert both stdout and stderr into buf
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        cf = ReadCif(path)
    return cf

# ───────────── file locations ─────────────
CIF_FILE  = get_path("test_cif_file")
PONI_FILE = get_path("test_poni_file")

# ───────────── geometry & wavelength ─────────────
poni  = parse_poni_file(PONI_FILE)
dist  = poni.get("Dist", 0.075)          # m
rot1  = poni.get("Rot1", 0.0)            # γ
rot2  = poni.get("Rot2", 0.0)            # Γ
poni1 = poni.get("Poni1", 0.0)           # m
poni2 = poni.get("Poni2", 0.0)           # m
wave_m = poni.get("Wavelength", 1e-10)   # m
λ = wave_m * 1e10                        # Å

# ───────────── lattice constants from CIF ─────────────
def _num(txt):                           # helper for ‘4.14070(3)’ → 4.14070
    m = re.match(r"[-+0-9\.Ee]+", txt)
    return float(m.group(0)) if m else np.nan

# read with PyCifRW
def parse_cif(cif_path: Path) -> tuple:
    cf = silent_ReadCif(str(cif_path))
    block = cf[next(iter(cf.keys()))]
    return block
blk = parse_cif((CIF_FILE))
a_v = _num(blk["_cell_length_a"])
c_v = _num(blk["_cell_length_c"])

# ───────────── simulation parameters ─────────────
IMAGE_SIZE   = 3000
MX           = 19
NUM_SAMPLES  = 1000
FWHM2SIGMA   = 1 / (2*math.sqrt(2*math.log(2)))
div_sigma    = math.radians(0.05 * FWHM2SIGMA)
sigma_mosaic = math.radians(0.8 * FWHM2SIGMA)
gamma_mosaic = math.radians(0.3 * FWHM2SIGMA)
bandwidth    = 0.7 / 100
bw_sigma     = 0.05e-3 * FWHM2SIGMA
center       = [(poni2/100e-6), IMAGE_SIZE-(poni1/100e-6)]
theta_initial = 6.0
psi          = 0.0
zs = zb      = debye_x = debye_y = 0.0
n2           = IndexofRefraction()
occ          = [1.0, 1.0, 1.0]
int_thresh   = 1.0
two_theta_rng= (0, 70)
chi = 0.0 # tilt of sample 
# ───────────── Miller list & intensities ─────────────
energy_keV = (6.62607015e-34*2.99792458e8) / (λ*1e-10) / 1.602176634e-19
# --------------  key change: str(CIF_FILE) --------------
miller, intens, _, _ = miller_generator(
    MX,
    str(CIF_FILE),          # ← ensure plain string
    occ,
    λ,
    energy_keV,
    int_thresh,
    two_theta_rng
)

# ───────────── random beam / mosaic profile ─────────────
bx, by, θarr, φarr, λarr = generate_random_profiles(
    num_samples     = NUM_SAMPLES,
    divergence_sigma= div_sigma,
    bw_sigma        = bw_sigma,
    lambda0         = λ,
    bandwidth       = bandwidth
)
mosaic_params = dict(
    beam_x_array       = bx,
    beam_y_array       = by,
    theta_array        = θarr,
    phi_array          = φarr,
    wavelength_array   = λarr,
    sigma_mosaic_deg   = np.degrees(sigma_mosaic),
    gamma_mosaic_deg   = np.degrees(gamma_mosaic),
    eta                = 0.05
)

# ───────────── Run the simulation ─────────────
# ───────── Run the simulation ─────────
sim_buffer = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float64)

# grab *all* outputs
image, hit_tables, q_data, q_count, solve_status = process_peaks_parallel(
    miller, intens, IMAGE_SIZE,
    a_v, c_v, λ,
    sim_buffer,
    dist,
    rot1, rot2, chi, psi,
    zs, zb, n2,
    mosaic_params["beam_x_array"],
    mosaic_params["beam_y_array"],
    mosaic_params["theta_array"],
    mosaic_params["phi_array"],
    mosaic_params["sigma_mosaic_deg"],
    mosaic_params["gamma_mosaic_deg"],
    mosaic_params["eta"],
    mosaic_params["wavelength_array"],
    debye_x, debye_y,
    center,
    theta_initial,
    np.array([1.0,0.0,0.0]),
    np.array([0.0,1.0,0.0]),
    save_flag=0,
    record_status=True

)

# ───────────── additional geometry debug info ─────────────
debug_info = debug_detector_paths(
    mosaic_params["beam_x_array"],
    mosaic_params["beam_y_array"],
    mosaic_params["theta_array"],
    mosaic_params["phi_array"],
    theta_initial, chi, psi,
    zb, zs,
    dist, rot1, rot2,
    np.array([0.0,1.0,0.0]),
    np.array([1.0,0.0,0.0])
)


# ───────────── Display ─────────────
turbo = cm.get_cmap('turbo', 256)
turbo_rgba = turbo(np.linspace(0,1,256)); turbo_rgba[0] = [1,1,1,1]
turbo_white0 = ListedColormap(turbo_rgba, name='turbo_white0'); turbo_white0.set_bad('white')

# plt.figure(figsize=(8,8))
# plt.imshow(image, cmap=turbo_white0, vmin=0, origin='upper')
# plt.xlabel("x-pixel"); plt.ylabel("y-pixel")
# plt.colorbar(label="Intensity")
# plt.title("Simulated diffraction pattern")
# plt.tight_layout(); plt.show()

# ───────────── Optional save ─────────────
# ───────────── Save everything together ─────────────

def _detach(x):
    "deep-copy arrays so they are pickle-safe; pack lists as object arrays"
    if x is None:               return None
    if isinstance(x, list):     return np.asarray([_detach(a) for a in x],
                                                  dtype=object)
    if isinstance(x, np.ndarray): return np.ascontiguousarray(x.copy())
    return x                    # fall-back (shouldn’t happen)

# ─── choose a *constant* filename instead of a timestamp ─────────────

script_dir = Path(__file__).resolve().parent
out        = script_dir / "simulation.npz"     # ← no date component
# ---------------------------------------------------------------------

# ------------------------------------------------------------------
#  build the dict that will go into np.savez
# ------------------------------------------------------------------
arrays = {
    "image": _detach(image).astype(np.float32),      # always present
}

# add every hit-table as its own entry
for i, tbl in enumerate(hit_tables):                # ← works with numba List
    arrays[f"hits_peak_{i}"] = _detach(tbl)

# optional extras
if q_data is not None:
    arrays["q_data"]  = _detach(q_data)
    arrays["q_count"] = _detach(q_count)

arrays["debug_info"] = _detach(debug_info)
arrays["solve_status"] = _detach(solve_status)

# finally write the .npz
np.savez(script_dir / "simulation.npz", **arrays)
print("saved →", script_dir / "simulation.npz")
