#!/usr/bin/env python3
"""
Fast interactive Hendricks–Teller viewer for PbI₂

Bottom axis : ℓ  (2 H cell)
Top axis    : ℓ  (6 H cell)

Controls
────────
  • h, k sliders      : Miller indices
  • p slider          : probability parameter
  • N slider          : domain length (51 ≡ ∞)
  • Toggle scale      : linear / log y‐axis
  • Checkboxes        : show / hide each curve or marker set
"""

# ───────────────────────── imports ──────────────────────────
import re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from ra_sim.utils.tools import intensities_for_hkls

# ───────────────────────── helpers ─────────────────────────
def c_from_cif(f: str) -> float:
    """Return _cell_length_c (Å) from a CIF file."""
    with open(f, "r", encoding="utf-8", errors="ignore") as fp:
        for ln in fp:
            m = re.match(r"_cell_length_c\s+([\d.]+)", ln)
            if m:
                return float(m.group(1))
    raise ValueError(f"_cell_length_c not found in {f}")

# ────────────────── constants & CIF paths ──────────────────
P_CLAMP, ZERO_THR = 1e-6, 1e-8
a_hex             = 4.557                       # Å (shared)
BUNDLE            = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
CIF_2H            = str(BUNDLE / "PbI2_2H.cif")
CIF_6H            = str(BUNDLE / "PbI2_6H.cif")
c_2H, c_6H        = map(c_from_cif, (CIF_2H, CIF_6H))
ratio_26          = c_6H / c_2H                 # ℓ₂H → ℓ₆H

# ──────────────── atomic form factors (fast) ───────────────
try:
    import xraydb
    _f0 = lambda s, q: xraydb.f0(s, q)[0]
except ImportError:
    _f0 = lambda s, q: {"Pb": 82, "I": 53}.get(s, 0)

_SITES = [(0.0, 0.0,  0.000, "Pb"),
          (1/3, 2/3,  0.25 , "I"),
          (2/3, 1/3, -0.25 , "I")]

def _Qmag(h, k, L):
    inv_d2 = (4/3)*(h*h+k*k+h*k)/a_hex**2 + (L**2)/c_2H**2
    return 2*np.pi*np.sqrt(inv_d2)

def _F2(h, k, L_vec):
    Q = _Qmag(h, k, L_vec)
    phase = [np.exp(2j*np.pi*(h*x + k*y + L_vec*z)) for x,y,z,_ in _SITES]
    coeff = [_f0(sym, Q) for *_, sym in _SITES]
    return np.abs(sum(c*ph for c, ph in zip(coeff, phase)))**2

# ───────────── Hendricks–Teller utilities ────────────────
N_p, A_c = 3, 17.98e-10
AREA = (2*np.pi)**2 / A_c * N_p

def _abc(p, h, k):
    δ = 2*np.pi*((2*h + k)/3)
    # build the full complex f = (1-p) + p·e^{iδ}
    z = (1 - p) + p * np.exp(1j * δ)
    # clamp the magnitude
    f_abs = np.minimum(np.abs(z), 1 - P_CLAMP)
    # get the phase
    ψ = np.angle(z)
    return f_abs, ψ, δ


def I_inf(L, p, h, k, F2):
    f, ψ, δ = _abc(p, h, k)
    φ = δ + 2*np.pi*L
    return AREA * F2 * (1-f**2)/(1 + f**2 - 2*f*np.cos(φ-ψ))

def I_fin(L, p, h, k, N, F2):
    f, ψ, δ = _abc(p, h, k)
    φ = δ + 2*np.pi*L/3
    θ = φ-ψ
    rp, rm = f*np.exp(1j*θ), f*np.exp(-1j*θ)
    T1 = rp*(1-rm**N)/(1-rm)
    T2 = rm*(1-rp**N)/(1-rp)
    return AREA * F2 * (1+(T1+T2).real)

# ─────────────── CIF intensity cache ────────────────
H_vals, K_vals = range(-5,6), range(-5,6)
_all_hkls = [(h,k,L) for h in H_vals for k in K_vals for L in range(11)]
_intensity = {}
for cif in (CIF_2H, CIF_6H):
    raw  = np.asarray(intensities_for_hkls(_all_hkls, cif, [1,1,1], 1.54))
    subc, idx = {}, 0
    for h in H_vals:
        for k in K_vals:
            subc[(h,k)] = raw[idx:idx+11]
            idx += 11
    _intensity[cif] = subc

_peak_pos = {}
def _bottom_pos(cif, h, k):
    key = (cif, h, k)
    if key in _peak_pos:
        return _peak_pos[key]
    L = np.arange(11)
    if cif == CIF_2H:
        x = L.astype(float)
    else:
        x = L / ratio_26
    _peak_pos[key] = x
    return x

# ───────────── grid, cache & state ─────────────
L_grid  = np.linspace(0, 10, 500)
F2_cache = {}
edge_cache = {}
state = {"h":1, "k":0, "p":0.1, "N":51}

# ─────────────── figure & static artists ───────────────
fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(left=0.25, bottom=0.25, top=0.78)

# secondary axis for ℓ₆H
secax = ax.secondary_xaxis('top',
         functions=(lambda x: x*ratio_26, lambda x: x/ratio_26))
ax.set_xlabel(r"$\ell^{\,(2H)}$")
secax.set_xlabel(r"$\ell^{\,(6H)}$")

# integer ticks
max_l2 = 10
max_l6 = int(np.floor(max_l2*ratio_26))
ax.set_xticks(np.arange(max_l2+1))
secax.set_xticks(np.arange(max_l6+1))

# dashed guides (draw once)
for l2 in range(max_l2+1):
    ax.axvline(l2, ls="--", color="lightgray", lw=0.8, zorder=0)
for l6 in range(max_l6+1):
    ax.axvline(l6/ratio_26, ls="--", color="lightgray", lw=0.8, zorder=0)

# initial HT and edge curves
ht_line, = ax.plot(L_grid, np.empty_like(L_grid), lw=2, label="HT")
p1_line, = ax.plot(L_grid, np.empty_like(L_grid), ls="--", label="p→1")
p0_line, = ax.plot(L_grid, np.empty_like(L_grid), ls="--", label="p→0")

# initial CIF markers
overlay_lines = {}
for cif, mk, lab in [(CIF_2H,"o","2H"), (CIF_6H,"s","6H")]:
    ln, = ax.plot([], [], mk, ms=5, mfc="none", ls="None", label=f"CIF {lab}")
    overlay_lines[cif] = ln

ax.set_xlim(0, 10)
ax.set_yscale("log")
ax.set_ylabel(r"$I / |F_0|^2$")
title = ax.set_title("")
ax.legend(loc="upper right")

# ─────────────── checkbox for visibility ───────────────
rax = plt.axes([0.02, 0.45, 0.15, 0.20])  # x, y, width, height
labels = ["HT", "p→1", "p→0", "CIF 2H", "CIF 6H"]
visibility = [True]*5
check = CheckButtons(rax, labels, visibility)

def _toggle(label):
    if label == "HT":
        ht_line.set_visible(not ht_line.get_visible())
    elif label == "p→1":
        p1_line.set_visible(not p1_line.get_visible())
    elif label == "p→0":
        p0_line.set_visible(not p0_line.get_visible())
    elif label == "CIF 2H":
        overlay_lines[CIF_2H].set_visible(not overlay_lines[CIF_2H].get_visible())
    elif label == "CIF 6H":
        overlay_lines[CIF_6H].set_visible(not overlay_lines[CIF_6H].get_visible())
    fig.canvas.draw_idle()

check.on_clicked(_toggle)

# ─────────────── low-level update helpers ───────────────
def _update_ht(p, N, h, k):
    F2 = F2_cache[(h,k)]
    if N >= 51:
        y = I_inf(L_grid, p, h, k, F2)
    else:
        y = I_fin(L_grid, p, h, k, N, F2)
    ht_line.set_ydata(y)
    return y

def _update_fixed_curves(h,k):
    F2 = F2_cache[(h,k)]
    y1 = I_inf(L_grid, 1-P_CLAMP, h, k, F2)
    y0 = I_inf(L_grid, P_CLAMP,   h, k, F2)
    p1_line.set_ydata(y1)
    p0_line.set_ydata(y0)

def _update_peaks(h,k, ymax):
    for cif, ln in overlay_lines.items():
        x = _bottom_pos(cif, h, k)
        I = _intensity[cif].get((h,k), np.zeros(11))
        mask = (I/I.max(initial=1.0)) > ZERO_THR
        ln.set_data(x[mask], I[mask]/I.max(initial=1.0)*ymax)

# ─────────────── high-level redraw handlers ───────────────
def refresh_from_hk(event=None):
    h, k = state["h"], state["k"]
    if (h,k) not in F2_cache:
        F2_cache[(h,k)] = _F2(h, k, L_grid)
    if (h,k) not in edge_cache:
        F2 = F2_cache[(h,k)]
        edge_cache[(h,k)] = (I_inf(L_grid,1-P_CLAMP,h,k,F2),
                             I_inf(L_grid,P_CLAMP  ,h,k,F2))
    p1_line.set_ydata(edge_cache[(h,k)][0])
    p0_line.set_ydata(edge_cache[(h,k)][1])
    y = _update_ht(state["p"], state["N"], h, k)
    _update_peaks(h, k, y.max())
    title.set_text(f"h={h} k={k} p={state['p']:.4f} N={'∞' if state['N']>=51 else state['N']}")
    _rescale_y()
    fig.canvas.draw_idle()

def refresh_p(event=None):
    y = _update_ht(state["p"], state["N"], state["h"], state["k"])
    _update_peaks(state["h"], state["k"], y.max())
    title.set_text(f"h={state['h']} k={state['k']} p={state['p']:.4f} N={'∞' if state['N']>=51 else state['N']}")
    _rescale_y()
    fig.canvas.draw_idle()

def refresh_N(event=None):
    refresh_p()

# ─────────────── y-axis autoscale ───────────────
def _rescale_y():
    ys = np.concatenate([
        ht_line.get_ydata(),
        p1_line.get_ydata(),
        p0_line.get_ydata(),
        *[ln.get_ydata() for ln in overlay_lines.values()]
    ])
    ys = ys[np.isfinite(ys)]
    if ys.size == 0:
        return
    ymin, ymax = ys.min(), ys.max()
    if ax.get_yscale() == "log":
        ax.set_ylim(max(ymin*0.5, 1e-8), ymax*2)
    else:
        pad = 0.1*(ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

# ───────────── widgets & callbacks ─────────────
opt = dict(valstep=1)
s_h = Slider(plt.axes([0.25,0.20,0.65,0.03]), "h", -5,5, valinit=1, **opt)
s_k = Slider(plt.axes([0.25,0.15,0.65,0.03]), "k", -5,5, valinit=0, **opt)
s_p = Slider(plt.axes([0.25,0.10,0.65,0.03]), "p", 0,1, valinit=0.1, valstep=0.001)
s_N = Slider(plt.axes([0.25,0.05,0.65,0.03]), "N", 1,51,valinit=51, **opt)

s_h.on_changed(lambda v: (state.update(h=int(v)), refresh_from_hk()))
s_k.on_changed(lambda v: (state.update(k=int(v)), refresh_from_hk()))
s_p.on_changed(lambda v: (state.update(p=float(v)), refresh_p()))
s_N.on_changed(lambda v: (state.update(N=int(v)), refresh_N()))

def toggle_scale(evt):
    ax.set_yscale("linear" if ax.get_yscale()=="log" else "log")
    _rescale_y()
    fig.canvas.draw_idle()

Button(plt.axes([0.04,0.83,0.12,0.06]), "Toggle scale").on_clicked(toggle_scale)

# ───────────── launch viewer ─────────────
refresh_from_hk()
plt.show()
