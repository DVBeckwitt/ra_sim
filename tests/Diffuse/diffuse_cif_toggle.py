#!/usr/bin/env python3
"""
Dual-method Hendricks-Teller viewer for PbI2.
Shows legacy algebraic HT and order-2 Markov totals in one figure,
sharing the same UI, CIF handling, and sliders.
"""

import os, re, sys
from pathlib import Path
from collections import Counter
from time import perf_counter
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.widgets import Slider, RangeSlider, Button, CheckButtons

# External project utilities (as in your originals)
from ra_sim.utils.tools import intensities_for_hkls
from ra_sim.utils.calculations import d_spacing, two_theta
import Dans_Diffraction as dif
from Dans_Diffraction.functions_crystallography import (
    xray_scattering_factor,
    xray_dispersion_corrections,
)

# -----------------------------------------------------------------------------
# Backend safety for interactive UI
if plt.get_backend().lower().endswith("agg") and not os.environ.get("PYTEST_CURRENT_TEST"):
    try:
        plt.switch_backend("TkAgg")
    except Exception:
        pass

# -----------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser(description="Dual HT viewer")
parser.add_argument("--l-max", type=float, default=10.0, help="maximum ℓ range")
_ARGS, _ = parser.parse_known_args()

# -----------------------------------------------------------------------------
# Constants and CIF helpers
COLORBLIND_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#F0E442", "#56B4E9", "#E69F00", "#000000",
]
plt.rc("axes", prop_cycle=cycler("color", COLORBLIND_COLORS))

P_CLAMP = 1e-6
A_HEX = 4.557  # Å
BUNDLE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
CIF_2H = BUNDLE / "PbI2_2H.cif"
CIF_6H = BUNDLE / "PbI2_6H.cif"

def c_from_cif(path: str) -> float:
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        for ln in fp:
            if m := re.match(r"_cell_length_c\s+([\d.]+)", ln):
                return float(m.group(1))
    raise ValueError("_cell_length_c not found in CIF")

C_2H, C_6H = map(c_from_cif, map(str, (CIF_2H, CIF_6H)))

def iodine_z_from_cif(path: Path) -> float:
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        for ln in fp:
            if ln.strip().startswith("I1"):
                parts = ln.split()
                if len(parts) >= 5:
                    return float(parts[4])
    raise ValueError("Iodine position not found")

DEFAULT_I_Z = iodine_z_from_cif(CIF_2H)

# X-ray constants
LAMBDA = 1.5406  # Å (Cu Kα1)
E_CuKa = 12398.4193 / LAMBDA  # eV

ION = {"Pb": "Pb2+", "I": "I1-"}  # ionic for f0
NEUTR = {"Pb": "Pb", "I": "I"}    # neutral for f′, f″

def f_comp(el: str, Q: np.ndarray) -> np.ndarray:
    q = np.asarray(Q, dtype=float).reshape(-1)
    f0 = xray_scattering_factor([ION[el]], q)[:, 0]
    f1, f2 = xray_dispersion_corrections([NEUTR[el]], energy_kev=[E_CuKa / 1000])
    f1 = float(f1[0, 0])
    f2 = float(f2[0, 0])
    out = f0 + f1 + 1j * f2
    return out.reshape(Q.shape)

def _sites_from_cif(path: Path):
    xtl = dif.Crystal(str(path))
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    st = xtl.Structure
    return [(float(st.u[i]), float(st.v[i]), float(st.w[i]), str(st.type[i]))
            for i in range(len(st.u))]

SITES_2H = _sites_from_cif(CIF_2H)

# Geometry and scale
N_P, A_CELL = 3, 17.98e-10
AREA = (2 * np.pi) ** 2 / A_CELL * N_P

# -----------------------------------------------------------------------------
# HK range and ℓ grid
MAX_HK = 10
HK_BY_M = {}
for h in range(-MAX_HK, MAX_HK + 1):
    for k in range(-MAX_HK, MAX_HK + 1):
        m = h * h + h * k + k * k
        HK_BY_M.setdefault(m, []).append((h, k))
ALLOWED_M = sorted(HK_BY_M)

L_MAX = float(_ARGS.l_max)
N_L = 2001
L_GRID = np.linspace(0, L_MAX, N_L)
QZ_GRID = 2 * np.pi / C_2H * L_GRID

# -----------------------------------------------------------------------------
# Global theta / z and Markov cache
THETA_GRID = 2.0 * np.pi * (L_GRID / 3.0)
Z_GRID = (1.0 - P_CLAMP) * np.exp(1j * THETA_GRID)

_MKV_CACHE = {}

# -----------------------------------------------------------------------------
# Precompute F²(h,k,ℓ) for 2H single layer
F2_cache_2H = {}
_FIXED_PART = {}
_IODINE_FACTOR = {}

def _precompute_F_parts():
    for pairs in HK_BY_M.values():
        for h, k in pairs:
            Q_2h = 2*np.pi*np.sqrt((4/3)*(h*h + k*k + h*k)/A_HEX**2 + (L_GRID**2)/C_2H**2)
            fixed = np.zeros_like(L_GRID, dtype=complex)
            factor = np.zeros_like(L_GRID, dtype=complex)
            for x, y, z, sym in SITES_2H:
                f = f_comp(sym, Q_2h)
                phase_xy = np.exp(2j * np.pi * (h*x + k*y))
                if sym.startswith("I"):
                    factor += f * phase_xy
                else:
                    fixed += f * phase_xy * np.exp(2j * np.pi * L_GRID * (z/3))
            _FIXED_PART[(h, k)] = fixed
            _IODINE_FACTOR[(h, k)] = factor

def compute_F2_cache(i_z: float):
    phase_z = np.exp(2j * np.pi * L_GRID * (i_z / 3))
    F2_cache_2H.clear()
    for key in _FIXED_PART:
        total = _FIXED_PART[key] + _IODINE_FACTOR[key] * phase_z
        F2_cache_2H[key] = np.abs(total) ** 2

_precompute_F_parts()
compute_F2_cache(DEFAULT_I_Z)

# -----------------------------------------------------------------------------
# Model A: Legacy algebraic HT (scalar-f)
def _abc(p, h, k):
    δ = 2 * np.pi * ((2*h + k) / 3)
    z = (1 - p) + p * np.exp(1j * δ)
    f = np.minimum(np.abs(z), 1 - P_CLAMP)
    ψ = np.angle(z)
    return f, ψ, δ

def I_inf_alg(p, h, k, F2):
    if state.get("f2_only"):
        return F2

    # flip p → 1-p
    p_flipped = 1.0 - p

    f, ψ, δ = _abc(p_flipped, h, k)
    φ = δ + 2 * np.pi * L_GRID * (1/3)
    return AREA * F2 * (1 - f**2) / (1 + f**2 - 2 * f * np.cos(φ - ψ))

# -----------------------------------------------------------------------------
# Model B: Order-2 Markov (pair-state transfer 3x3)
def _rho_alpha_from_p(p: float):
    p_eff = float(np.clip(p, 0.0, 1.0))
    return 1.0 - p_eff, p_eff  # rho, alpha

def _class_stationary(rho: float, alpha: float):
    denom = 2.0 - alpha - rho
    P_S = (1.0 - alpha) / denom
    P_E = (1.0 - rho)  / (2.0 * denom)
    return P_S, P_E, P_E

def _slip_phase(h: int, k: int) -> float:
    return 2.0 * np.pi * ((2*h + k) / 3.0)

def _R_from_transfer(phi: float, rho: float, alpha: float):
    """Return R(theta) on the full L_GRID via cached eigendecomposition."""
    key = (float(rho), float(alpha), float(phi))
    lam, w = _MKV_CACHE.get(key, (None, None))

    if lam is None:
        eip = np.exp(1j * phi)

        stay = rho
        slip = 0.5 * (1.0 - rho)
        slip_phase = slip * eip

        T = np.array(
            [[stay,      slip_phase,   slip_phase],
             [slip_phase, stay,        slip_phase],
             [slip_phase, slip_phase,  stay      ]],
            dtype=complex,
        )

        lam, R = np.linalg.eig(T)
        R_inv = np.linalg.inv(R)

        P_S, P_Ep, P_Em = _class_stationary(rho, alpha)
        pi = np.array([P_S, P_Ep, P_Em], dtype=complex)
        ones = np.ones(3, dtype=complex)

        piR  = pi @ R
        Rin1 = R_inv @ ones
        w    = piR * Rin1

        _MKV_CACHE[key] = (lam, w)

    # Use precomputed Z_GRID instead of rebuilding theta each time
    z = Z_GRID
    frac = (lam[:, None] * z[None, :]) / (1.0 - lam[:, None] * z[None, :])
    S = (w[:, None] * frac).sum(axis=0)
    Rtheta = 1.0 + 2.0 * np.real(S)
    return np.maximum(Rtheta, 0.0)

def I_inf_mkv(p, h, k, F2):
    if state.get("f2_only"):
        return F2
    rho, alpha = _rho_alpha_from_p(p)
    phi = _slip_phase(h, k)
    Rtheta = _R_from_transfer(phi, rho, alpha)
    return AREA * F2 * Rtheta

# -----------------------------------------------------------------------------
# State and helpers
defaults = {
    "m": ALLOWED_M[0],
    "p0": 0.0, "p1": 1.0, "p3": 0.5,
    "w0": 0.0, "w1": 0.0, "w2": 100.0,
    "L_lo": 1.0, "L_hi": L_MAX,
    "I_z": DEFAULT_I_Z,
    "f2_only": False,
    "qz_axis": False,
    "mode": "m",   # or "hk"
    "h": 0, "k": 0,
    "show_bragg": False,
    "show_alg_comp": False,
    "show_mkv_comp": False,
}
state = defaults.copy()

def active_pairs():
    return HK_BY_M.get(state["m"], []) if state["mode"] == "m" else [(state["h"], state["k"])]

def compute_components_with(I_inf_func):
    pairs = active_pairs()
    counts = {(h, k): 1 for h, k in pairs} if state["mode"] == "hk" else Counter(pairs)

    def comp(p):
        return sum(n * I_inf_func(p, h, k, F2_cache_2H[(h, k)]) for (h, k), n in counts.items())

    return {"I0": comp(state["p0"]), "I1": comp(state["p1"]), "I3": comp(state["p3"])}

curves = {"alg": {}, "mkv": {}}
timings = {"alg": 0.0, "mkv": 0.0}


def recompute_curves():
    start = perf_counter()
    curves["alg"] = compute_components_with(I_inf_alg)
    timings["alg"] = perf_counter() - start

    start = perf_counter()
    curves["mkv"] = compute_components_with(I_inf_mkv)
    timings["mkv"] = perf_counter() - start


recompute_curves()

def _norm_weights():
    w0, w1, w2 = state["w0"], state["w1"], state["w2"]
    s = (w0 + w1 + w2) or 1.0
    return w0/s, w1/s, w2/s

def composite_total(block):
    w0, w1, w2 = _norm_weights()
    return w0*block["I0"] + w1*block["I1"] + w2*block["I3"]

# -----------------------------------------------------------------------------
# Bragg calculation (scales to a chosen model)
BRAGG_SCALE_MODEL = "alg"  # set to "mkv" to scale to Markov totals
_BRAGG_CACHE = {}

def _raw_bragg(h, k, lo, hi, cif_path):
    lo_i = int(np.floor(lo)); hi_i = int(np.ceil(hi))
    key = (cif_path, h, k, lo_i, hi_i)
    if key not in _BRAGG_CACHE:
        hkls = [(h, k, l) for l in range(lo_i, hi_i + 1)]
        intens = intensities_for_hkls(hkls, cif_path, [1.0], LAMBDA, energy=E_CuKa / 1000)
        L_vals = np.arange(lo_i, hi_i + 1)
        _BRAGG_CACHE[key] = (L_vals, np.asarray(intens))
    return _BRAGG_CACHE[key]

def _weight_2h_6h():
    w0, w1 = state["w0"], state["w1"]
    s = (w0 + w1) or 1.0
    return (w1 / s, w0 / s)  # 2H, 6H

def _ht_total_for_pair_model(h, k, model_key):
    w0, w1, w2 = _norm_weights()
    F2 = F2_cache_2H[(h, k)]
    if model_key == "alg":
        return w0*I_inf_alg(state["p0"], h, k, F2) + \
               w1*I_inf_alg(state["p1"], h, k, F2) + \
               w2*I_inf_alg(state["p3"], h, k, F2)
    else:
        return w0*I_inf_mkv(state["p0"], h, k, F2) + \
               w1*I_inf_mkv(state["p1"], h, k, F2) + \
               w2*I_inf_mkv(state["p3"], h, k, F2)

def bragg_intensity_single(h, k):
    lo, hi = state["L_lo"], state["L_hi"]
    L_vals, raw2h = _raw_bragg(h, k, lo, hi, str(CIF_2H))
    _, raw6h = _raw_bragg(h, k, lo, hi, str(CIF_6H))
    ht = _ht_total_for_pair_model(h, k, BRAGG_SCALE_MODEL)
    idx = np.round(L_vals / L_MAX * (len(L_GRID) - 1)).astype(int)
    ht_slice = ht[idx]
    max_ht = float(ht_slice.max()) if np.any(ht_slice) else 1.0
    max_2h = float(raw2h.max()) if np.any(raw2h) else 1.0
    max_6h = float(raw6h.max()) if np.any(raw6h) else 1.0
    scale2 = max_ht / max_2h if max_2h else 1.0
    scale6 = max_ht / max_6h if max_6h else 1.0
    w2h, w6h = _weight_2h_6h()
    return L_vals, raw2h * scale2 * w2h, raw6h * scale6 * w6h

def bragg_intensity_sum(pairs):
    lo, hi = state["L_lo"], state["L_hi"]
    L_vals = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1)
    total_2h = np.zeros_like(L_vals, dtype=float)
    total_6h = np.zeros_like(L_vals, dtype=float)
    total_ht = np.zeros_like(L_vals, dtype=float)
    idx = np.round(L_vals / L_MAX * (len(L_GRID) - 1)).astype(int)
    for h, k in pairs:
        _, b2 = _raw_bragg(h, k, lo, hi, str(CIF_2H))
        _, b6 = _raw_bragg(h, k, lo, hi, str(CIF_6H))
        total_2h += b2
        total_6h += b6
        total_ht += _ht_total_for_pair_model(h, k, BRAGG_SCALE_MODEL)[idx]
    max_ht = float(total_ht.max()) if np.any(total_ht) else 1.0
    max_2h = float(total_2h.max()) if np.any(total_2h) else 1.0
    max_6h = float(total_6h.max()) if np.any(total_6h) else 1.0
    scale2 = max_ht / max_2h if max_2h else 1.0
    scale6 = max_ht / max_6h if max_6h else 1.0
    w2h, w6h = _weight_2h_6h()
    return L_vals, total_2h * scale2 * w2h, total_6h * scale6 * w6h

# -----------------------------------------------------------------------------
# Figure and GUI
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.25, bottom=0.42, top=0.88)
ax.set_xlabel(r"$\ell$")
ax.set_ylabel("I (a.u.)")
ax.set_yscale("log")

# Lines: totals
(line_tot_alg,) = ax.plot([], [], lw=2.2, label="Σ weighted (alg)", color=COLORBLIND_COLORS[0])
(line_tot_mkv,) = ax.plot([], [], lw=2.2, ls=":", label="Σ weighted (Markov)", color=COLORBLIND_COLORS[1])

# Lines: components (initially hidden)
(line0_alg,) = ax.plot([], [], ls="--", label="I_alg(p≈0)", color=COLORBLIND_COLORS[2], visible=False)
(line1_alg,) = ax.plot([], [], ls="--", label="I_alg(p≈1)", color=COLORBLIND_COLORS[3], visible=False)
(line3_alg,) = ax.plot([], [], ls="--", label="I_alg(p)",   color=COLORBLIND_COLORS[4], visible=False)

(line0_mkv,) = ax.plot([], [], ls="-.", label="I_mkv(p≈0)", color=COLORBLIND_COLORS[5], visible=False)
(line1_mkv,) = ax.plot([], [], ls="-.", label="I_mkv(p≈1)", color=COLORBLIND_COLORS[6], visible=False)
(line3_mkv,) = ax.plot([], [], ls="-.", label="I_mkv(p)",   color=COLORBLIND_COLORS[7], visible=False)

title = ax.set_title("")
_bragg_lines = []

def refresh(_=None):
    lo, hi = state["L_lo"], state["L_hi"]
    mask = (L_GRID >= lo) & (L_GRID <= hi)
    x_grid = QZ_GRID if state.get("qz_axis") else L_GRID
    x_lo = 2 * np.pi / C_2H * lo if state.get("qz_axis") else lo
    x_hi = 2 * np.pi / C_2H * hi if state.get("qz_axis") else hi
    ticks = 2 * np.pi / C_2H * np.arange(np.ceil(lo), np.floor(hi) + 1) if state.get("qz_axis") else np.arange(np.ceil(lo), np.floor(hi) + 1)

    # Recompute totals
    tot_alg = composite_total(curves["alg"])
    tot_mkv = composite_total(curves["mkv"])

    # Set totals
    line_tot_alg.set_data(x_grid[mask], tot_alg[mask])
    line_tot_mkv.set_data(x_grid[mask], tot_mkv[mask])

    # Components visibility per toggles and weights
    vis_alg = state["show_alg_comp"]
    vis_mkv = state["show_mkv_comp"]
    w0, w1, w2 = _norm_weights()

    # Algebraic components
    for ln, arr, vis in (
        (line0_alg, curves["alg"]["I0"], vis_alg and w0 > 0),
        (line1_alg, curves["alg"]["I1"], vis_alg and w1 > 0),
        (line3_alg, curves["alg"]["I3"], vis_alg and w2 > 0),
    ):
        ln.set_visible(vis)
        if vis:
            ln.set_data(x_grid[mask], arr[mask])

    # Markov components
    for ln, arr, vis in (
        (line0_mkv, curves["mkv"]["I0"], vis_mkv and w0 > 0),
        (line1_mkv, curves["mkv"]["I1"], vis_mkv and w1 > 0),
        (line3_mkv, curves["mkv"]["I3"], vis_mkv and w2 > 0),
    ):
        ln.set_visible(vis)
        if vis:
            ln.set_data(x_grid[mask], arr[mask])

    # Limits and labels
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks(ticks)
    ax.set_xlabel(r"$q_z$ (Å$^{-1}$)" if state.get("qz_axis") else r"$\ell$")

    # Bragg overlay
    for ln in _bragg_lines:
        ln.remove()
    _bragg_lines.clear()
    if state["show_bragg"]:
        pairs = active_pairs()
        if state["mode"] == "m":
            L_vals, i2h, i6h = bragg_intensity_sum(pairs)
            total = i2h + i6h
            msk = (L_vals >= lo) & (L_vals <= hi)
            x_vals = 2 * np.pi / C_2H * L_vals[msk] if state.get("qz_axis") else L_vals[msk]
            (ln,) = ax.plot(x_vals, total[msk], marker="D", ls="none", label=f'Bragg sum m={state["m"]}', color=COLORBLIND_COLORS[4])
            _bragg_lines.append(ln)
        else:
            for h, k in pairs:
                L_vals, i2h, i6h = bragg_intensity_single(h, k)
                msk = (L_vals >= lo) & (L_vals <= hi)
                x_vals = 2 * np.pi / C_2H * L_vals[msk] if state.get("qz_axis") else L_vals[msk]
                (ln2,) = ax.plot(x_vals, i2h[msk], marker="o", ls="none", label=f"2H({h},{k})", color=COLORBLIND_COLORS[5])
                (ln6,) = ax.plot(x_vals, i6h[msk], marker="s", ls="none", label=f"6H({h},{k})", color=COLORBLIND_COLORS[6])
                _bragg_lines.extend([ln2, ln6])

    # Title
    if state["mode"] == "m":
        m = state["m"]
        r = np.sqrt(m)
        Qr = 2 * np.pi / A_HEX * np.sqrt(4 * m / 3) if m else 0
        base_title = f"r=√m={r:.3f}, Qr={Qr:.3f} Å^-1 | HK set: {HK_BY_M[m]}"
    else:
        h, k = state["h"], state["k"]
        m = h*h + h*k + k*k
        Qr = 2 * np.pi / A_HEX * np.sqrt(4 * m / 3)
        base_title = f"(h,k)=({h},{k}), m={m}, Qr={Qr:.3f} Å^-1"

    timing_text = f"Δt_alg={timings['alg'] * 1e3:.1f} ms | Δt_markov={timings['mkv'] * 1e3:.1f} ms"
    title.set_text(f"{base_title}\n{timing_text}")

    # Legend
    handles = [line_tot_alg, line_tot_mkv] + _bragg_lines
    for ln in (line0_alg, line1_alg, line3_alg, line0_mkv, line1_mkv, line3_mkv):
        if ln.get_visible():
            handles.append(ln)
    ax.legend(handles, [h.get_label() for h in handles], loc="upper right")

    ax.relim(); ax.autoscale_view()
    fig.canvas.draw_idle()

# -----------------------------------------------------------------------------
# Sliders and controls
_sliders = []

def make_slider(rect, label, vmin, vmax, val, valstep, cb):
    axr = plt.axes(rect)
    s = Slider(axr, "", vmin, vmax, valinit=val, valstep=valstep)
    axr.text(0.5, 1.2, label, transform=axr.transAxes, ha="center")
    s.on_changed(cb)
    _sliders.append(s)
    return s

# ℓ range
ax_range = plt.axes([0.25, 0.05, 0.65, 0.03])
rs = RangeSlider(ax_range, "ℓ range", 0, L_MAX, valinit=(state["L_lo"], state["L_hi"]), valstep=0.1)
rs.on_changed(lambda v: (state.update(L_lo=v[0], L_hi=v[1]), refresh()))

# iodine z
ax_z = plt.axes([0.92, 0.05, 0.05, 0.03])
sz = Slider(ax_z, "z(I)", 0, 1, valinit=state["I_z"], valstep=1e-3)
def _on_z(v):
    state["I_z"] = v
    compute_F2_cache(v)
    recompute_curves()
    refresh()
sz.on_changed(_on_z)

# vertical groups
ys = [0.34, 0.28, 0.22, 0.16]

# m index
m_slider = make_slider(
    [0.25, ys[0], 0.65, 0.03],
    "m index",
    min(ALLOWED_M), max(ALLOWED_M), state["m"], ALLOWED_M,
    lambda v: (state.update(m=int(v)),
               recompute_curves(),
               refresh()),
)

# hk sliders (hidden by default)
hk_sliders = []
def make_int_slider(rect, label, vmax, key):
    s = make_slider(rect, label, -MAX_HK, vmax, state[key], 1,
                    lambda v, k=key: (state.update({k: int(v)}),
                                      recompute_curves(),
                                      refresh()))
    hk_sliders.append(s); s.ax.set_visible(False)

make_int_slider([0.25, ys[0], 0.30, 0.03], "H", MAX_HK, "h")
make_int_slider([0.60, ys[0], 0.30, 0.03], "K", MAX_HK, "k")

# p and weights
make_slider([0.25, ys[1], 0.45, 0.03], "p≈0", 0, 0.2, state["p0"], 1e-3,
            lambda v: (state.update(p0=v),
                       recompute_curves(),
                       refresh()))
make_slider([0.72, ys[1], 0.20, 0.03], "w(p≈0)%", 0, 100, state["w0"], 0.1,
            lambda v: (state.update(w0=v), refresh()))

make_slider([0.25, ys[2], 0.45, 0.03], "p≈1", 0.8, 1, state["p1"], 1e-3,
            lambda v: (state.update(p1=v),
                       recompute_curves(),
                       refresh()))
make_slider([0.72, ys[2], 0.20, 0.03], "w(p≈1)%", 0, 100, state["w1"], 0.1,
            lambda v: (state.update(w1=v), refresh()))

make_slider([0.25, ys[3], 0.45, 0.03], "p", 0, 1, state["p3"], 1e-3,
            lambda v: (state.update(p3=v),
                       recompute_curves(),
                       refresh()))
make_slider([0.72, ys[3], 0.20, 0.03], "w(p)%", 0, 100, state["w2"], 0.1,
            lambda v: (state.update(w2=v), refresh()))

# Buttons
btn_scale = Button(plt.axes([0.42, 0.01, 0.16, 0.03]), "Toggle scale")
btn_scale.on_clicked(lambda _: (ax.set_yscale("linear" if ax.get_yscale() == "log" else "log"), refresh()))

btn_bragg = Button(plt.axes([0.25, 0.01, 0.13, 0.03]), "Bragg on/off")
btn_bragg.on_clicked(lambda _: (state.update(show_bragg=not state["show_bragg"]), refresh()))

def toggle_mode(_):
    state["mode"] = "hk" if state["mode"] == "m" else "m"
    m_slider.ax.set_visible(state["mode"] == "m")
    for s in hk_sliders:
        s.ax.set_visible(state["mode"] == "hk")
    recompute_curves()
    refresh()

b_mode = Button(plt.axes([0.60, 0.01, 0.16, 0.03]), "H/K panel")
b_mode.on_clicked(toggle_mode)

# Checkboxes: F² only, qz axis, components
chk_ax = plt.axes([0.05, 0.01, 0.15, 0.14])
checks = CheckButtons(chk_ax, ["F² only", "qz axis", "Components (alg)", "Components (Markov)"],
                      [state["f2_only"], state["qz_axis"], state["show_alg_comp"], state["show_mkv_comp"]])

def _toggle_checks(label):
    if label == "F² only":
        state["f2_only"] = not state["f2_only"]
        recompute_curves()
    elif label == "qz axis":
        state["qz_axis"] = not state["qz_axis"]
    elif label == "Components (alg)":
        state["show_alg_comp"] = not state["show_alg_comp"]
    elif label == "Components (Markov)":
        state["show_mkv_comp"] = not state["show_mkv_comp"]
    refresh()

checks.on_clicked(_toggle_checks)

# Initial draw
refresh()
plt.show()
