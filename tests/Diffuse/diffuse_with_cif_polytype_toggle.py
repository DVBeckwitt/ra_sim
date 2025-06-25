#!/usr/bin/env python3
"""
Fast interactive Hendricks–Teller viewer for PbI₂
Now sums all (h,k) with identical radial index √(h²+hk+k²); displays Qr and the (h,k) pairs.
"""

import re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from ra_sim.utils.tools import intensities_for_hkls

# ───────────────────────── helpers ──────────────────────────
def c_from_cif(f: str) -> float:
    with open(f, "r", encoding="utf-8", errors="ignore") as fp:
        for ln in fp:
            m = re.match(r"_cell_length_c\s+([\d.]+)", ln)
            if m: return float(m.group(1))
    raise ValueError(f"_cell_length_c not found in {f}")

# ────────────────── constants & CIF paths ──────────────────
P_CLAMP, ZERO_THR = 1e-6, 1e-8
a_hex             = 4.557                       # Å
BUNDLE            = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
CIF_2H            = str(BUNDLE / "PbI2_2H.cif")
CIF_6H            = str(BUNDLE / "PbI2_6H.cif")
c_2H, c_6H        = map(c_from_cif, (CIF_2H, CIF_6H))
ratio_26          = c_6H / c_2H

# ─────────────── atomic form factors ───────────────
try:
    import xraydb
    _f0 = lambda s, q: xraydb.f0(s, q)[0]
except ImportError:
    _f0 = lambda s, q: {"Pb": 82, "I": 53}.get(s, 0)

_SITES = [
    (0.0, 0.0,  0.000, "Pb"),
    (1/3, 2/3,  1/4 , "I"),
    (2/3, 1/3, -1/4 , "I"),
]

def _Qmag(h, k, L):
    inv_d2 = (4/3)*(h*h + k*k + h*k)/a_hex**2 + (L**2)/c_2H**2
    return 2*np.pi*np.sqrt(inv_d2)

def _F2(h, k, L_vec):
    Q = _Qmag(h, k, L_vec)
    phase = [np.exp(2j*np.pi*(h*x + k*y + L_vec*z)) for x,y,z,_ in _SITES]
    coeff = [_f0(sym, Q) for *_, sym in _SITES]
    return np.abs(sum(c*ph for c, ph in zip(coeff, phase)))**2

# ───────── Hendricks–Teller formulas ─────────
def _abc(p, h, k):
    δ = 2*np.pi*((2*h + k)/3)
    z = (1 - p) + p*np.exp(1j*δ)
    f_abs = np.minimum(np.abs(z), 1 - P_CLAMP)
    ψ = np.angle(z)
    return f_abs, ψ, δ

def I_inf(L, p, h, k, F2):
    f, ψ, δ = _abc(p, h, k)
    φ = δ + 2*np.pi*L/3
    return AREA * F2 * (1 - f**2)/(1 + f**2 - 2*f*np.cos(φ - ψ))

def I_fin(L, p, h, k, N, F2):
    f, ψ, δ = _abc(p, h, k)
    φ = δ + 2*np.pi*L
    θ = φ - ψ
    rp, rm = f*np.exp(1j*θ), f*np.exp(-1j*θ)
    T1 = rp*(1 - rm**N)/(1 - rm)
    T2 = rm*(1 - rp**N)/(1 - rp)
    return AREA * F2 * (1 + (T1 + T2).real)

# ───────── constants for HT intensity scale ─────────
N_p, A_c = 3, 17.98e-10
AREA = (2*np.pi)**2 / A_c * N_p

# ───────── radial-index machinery ─────────
R_IDX = lambda h,k: h*h + h*k + k*k
H_vals, K_vals = range(-5,6), range(-5,6)
HK_BY_M = {}
for h in H_vals:
    for k in K_vals:
        HK_BY_M.setdefault(R_IDX(h,k), []).append((h,k))

# ───────── caches & state ─────────
L_grid   = np.linspace(0, 10, 500)
F2_cache = {}
state    = {"m": 0, "p": 0.1, "N": 51}

# ───────── figure setup ─────────
fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(left=0.25, bottom=0.25, top=0.85)
secax = ax.secondary_xaxis('top',
    functions=(lambda x: x*ratio_26, lambda x: x/ratio_26))
ax.set_xlabel(r"$\ell^{\,(2H)}$")
secax.set_xlabel(r"$\ell^{\,(6H)}$")
ax.set_xticks(np.arange(11))
secax.set_xticks(np.arange(int(np.floor(10*ratio_26))+1))
ax.set_yscale("log")
ax.set_ylabel(r"$I$")
ht_line, = ax.plot(L_grid, np.zeros_like(L_grid), lw=2, label="HT")
p1_line, = ax.plot(L_grid, np.zeros_like(L_grid), ls="--", label="p→1")
p0_line, = ax.plot(L_grid, np.zeros_like(L_grid), ls="--", label="p→0")
ax.legend(loc="upper right")
title = ax.set_title("")

# ───────── update functions ─────────
def _sum_I(p, N, m, kind):
    y = np.zeros_like(L_grid)
    for h,k in HK_BY_M.get(m, []):
        F2 = F2_cache.setdefault((h,k), _F2(h, k, L_grid))
        if kind == "edge0":
            y += I_inf(L_grid, 1-P_CLAMP, h, k, F2)
        elif kind == "edge1":
            y += I_inf(L_grid, P_CLAMP,   h, k, F2)
        else:
            y += I_inf(L_grid, p, h, k, F2) if N>=51 else I_fin(L_grid, p, h, k, N, F2)
    return y

def refresh_all(event=None):
    m, p, N = state["m"], state["p"], state["N"]
    pairs     = HK_BY_M.get(m, [])
    r_val     = np.sqrt(m)  # √(h²+hk+k²)
    Qr        = 2*np.pi/a_hex*np.sqrt(4*m/3) if m>0 else 0.0
    y_main    = _sum_I(p, N, m, "main")
    ht_line.set_ydata(y_main)
    p1_line.set_ydata(_sum_I(p, N, m, "edge0"))
    p0_line.set_ydata(_sum_I(p, N, m, "edge1"))
    hk_str    = ", ".join(f"({h},{k})" for h,k in pairs) or "none"
    title.set_text(
        f"r=√{{}}={r_val:.3f}, Qr={Qr:.3f} Å⁻¹\n"
        f"HK pairs: {hk_str}    p={p:.3f}    N={'∞' if N>=51 else N}"
    )
    _rescale_y()
    fig.canvas.draw_idle()

def _rescale_y():
    ys = np.concatenate([ht_line.get_ydata(),
                         p1_line.get_ydata(),
                         p0_line.get_ydata()])
    ys = ys[np.isfinite(ys) & (ys>0)]
    if ys.size:
        ax.set_ylim(ys.min()*0.5, ys.max()*2)

# ───────── widgets ─────────
opt = dict(valstep=1)
s_m = Slider(plt.axes([0.25,0.20,0.65,0.03]), "shell idx m", 0, max(H_vals)**2,
             valinit=0, **opt)
s_p = Slider(plt.axes([0.25,0.14,0.65,0.03]), "p", 0,1, valinit=0.1, valstep=0.001)
s_N = Slider(plt.axes([0.25,0.08,0.65,0.03]), "N", 1,51, valinit=51, **opt)

s_m.on_changed(lambda v: (state.update(m=int(v)), refresh_all()))
s_p.on_changed(lambda v: (state.update(p=float(v)), refresh_all()))
s_N.on_changed(lambda v: (state.update(N=int(v)),   refresh_all()))

Button(plt.axes([0.04,0.88,0.12,0.06]), "Toggle scale") \
    .on_clicked(lambda evt: (ax.set_yscale("linear" if ax.get_yscale()=="log" else "log"),
                             _rescale_y(), fig.canvas.draw_idle()))

# ───────── launch ─────────
refresh_all()
plt.show()
