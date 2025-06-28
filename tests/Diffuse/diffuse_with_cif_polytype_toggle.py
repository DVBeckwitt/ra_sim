#!/usr/bin/env python3
"""
Optimised Hendricks–Teller viewer for PbI₂ (modified)
──────────────────────────────────────────
* Pre-computes F²(h,k,ℓ) once for extended HK range.
* Caches component intensities for three p-values.
* ℓ-range RangeSlider up to `L_MAX` (default 20).
* All sliders placed below the figure area; no overlap.
* Top secondary axis removed.
* Uses ionic form factors: Pb²⁺ and I⁻.
* Deduplicates symmetry-equivalent HK pairs with degeneracy.
"""
import re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider, Button
from collections import Counter

from ra_sim.utils.tools import intensities_for_hkls
import pandas as pd
import tkinter as tk
from tkinter.filedialog import asksaveasfilename

# preserve slider objects so they aren’t garbage-collected
_sliders = []

def c_from_cif(path: str) -> float:
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        for ln in fp:
            if (m := re.match(r"_cell_length_c\s+([\d.]+)", ln)):
                return float(m.group(1))
    raise ValueError("_cell_length_c not found in CIF")

# constants
P_CLAMP = 1e-6

A_HEX   = 4.557  # Å
BUNDLE  = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
CIF_2H  = BUNDLE / "PbI2_2H.cif"
CIF_6H  = BUNDLE / "PbI2_6H.cif"
C_2H, C_6H = map(c_from_cif, map(str, (CIF_2H, CIF_6H)))

# ───────── constants ─────────
LAMBDA = 1.5406                          # Å   (Cu Kα1)
E_CuKa = 12398.4193 / LAMBDA             # eV  (≈ 8040 eV)

ION   = {'Pb': 'Pb2+', 'I': 'I1-'}       # ionic labels for f₀
NEUTR = {'Pb': 'Pb',  'I': 'I'}          # neutral symbols for f′/f″

def f_comp(el: str, Q: np.ndarray) -> np.ndarray:
    """Total atomic scattering factor  f = f₀(Q)+f′(E)+i f″(E)."""
    q   = Q / (4*np.pi)                                      # Å⁻¹
    f0  = xraydb.f0(ION[el], q)
    f1  = xraydb.f1_chantler(NEUTR[el], E_CuKa)              # scalar
    f2  = xraydb.f2_chantler(NEUTR[el], E_CuKa)              # scalar
    return f0 + f1 + 1j*f2                                   # broadcasts


# ionic form factors
try:
    import xraydb
    def f0(el: str, Q: np.ndarray) -> np.ndarray:
        q = Q / (4*np.pi)
        ion = {'Pb': 'Pb2+', 'I': 'I1-'}.get(el, el)
        return xraydb.f0(ion, q)
except ImportError:
    _fallback = {'Pb2+': 82.0, 'I1-': 53.0}
    def f0(el: str, Q: np.ndarray) -> np.ndarray:
        ion = {'Pb': 'Pb2+', 'I': 'Pb2+' if el=='Pb' else 'I1-'}.get(el, el)
        print(f"Warning: xraydb not found, using fallback for {ion}", file=sys.stderr)
        return np.full_like(Q, _fallback[ion])

# atomic sites
SITES    = [(0,0,0.0,'Pb'), (1/3,2/3,0.25,'I'), (2/3,1/3,-0.25,'I')]
N_P, A_CELL = 3, 17.98e-10
AREA     = (2*np.pi)**2 / A_CELL * N_P

# ─── Recommendation 1: extend HK range
MAX_HK = 10  # increase as needed
HK_BY_M = {}
for h in range(-MAX_HK, MAX_HK+1):
    for k in range(-MAX_HK, MAX_HK+1):
        m = h*h + h*k + k*k
        HK_BY_M.setdefault(m, []).append((h,k))
ALLOWED_M = sorted(HK_BY_M)   # only these m values are valid

# ℓ grid
L_MAX  = 20
N_L    = 2001
L_GRID = np.linspace(0, L_MAX, N_L)

# precompute |F|² (2H lattice)
F2_cache = {}
for pairs in HK_BY_M.values():
    for h,k in pairs:
        Q = 2*np.pi * np.sqrt((4/3)*(h*h+k*k+h*k)/A_HEX**2 + (L_GRID**2)/C_2H**2)
        phases = np.array([np.exp(2j*np.pi*(h*x + k*y + L_GRID*z)) for x,y,z,_ in SITES])
        coeffs = np.array([f_comp(sym, Q) for *_, sym in SITES])
        F2_cache[(h,k)] = np.abs((coeffs * phases).sum(axis=0))**2

# Hendricks–Teller infinite stacking

def _abc(p,h,k):
    δ = 2*np.pi*((2*h+k)/3)
    z = (1-p) + p*np.exp(1j*δ)
    f = np.minimum(np.abs(z), 1-P_CLAMP)
    ψ = np.angle(z)
    return f, ψ, δ

def I_inf(p,h,k,F2):
    f,ψ,δ = _abc(p,h,k)
    φ = δ + 2*np.pi * L_GRID/3
    return AREA * F2 * (1 - f**2) / (1 + f**2 - 2*f*np.cos(φ - ψ))

# GUI state
defaults = {
    'm': ALLOWED_M[0],
    'p0': 0.0, 'p1': 1.0, 'p3': 0.5,
    'w0': 33.3, 'w1': 33.3, 'w2': 0,
    'I0': None, 'I1': None, 'I3': None,
    'L_lo': 1.0, 'L_hi': L_MAX
}
state = defaults.copy()

# ─── after 'state = defaults.copy()' ───────────────────────────────────────────
state.update({'mode': 'm',      # 'm' or 'hk'
              'h': 0, 'k': 0,
              'show_bragg': False})  # current HK in hk‑mode

# cache Bragg intensities per CIF and (h,k)
_BRAGG_CACHE = {}
# ─── utilities ‑‑‑ choose active (h,k) list based on state['mode'] ────────────
def active_pairs():
    if state['mode'] == 'm':
        return HK_BY_M.get(state['m'], [])
    else:
        return [(state['h'], state['k'])]

# 1. keep only ONE definition of compute_components --------------------------
def compute_components():
    pairs = active_pairs()                       # picks either HK or m list

    if state['mode'] == 'hk':                    # no symmetry weight
        counts = {(h, k): 1 for h, k in pairs}
    else:                                        # hexagonal degeneracy
        counts = Counter((abs(h), abs(k)) for h, k in pairs)

    def comp(p):
        return sum(n * I_inf(p, h, k, F2_cache[(h, k)])
                   for (h, k), n in counts.items())

    state['I0'], state['I1'], state['I3'] = comp(state['p0']), comp(state['p1']), comp(state['p3'])

compute_components()

def composite_tot():
    w0,w1,w2 = state['w0'], state['w1'], state['w2']
    s = (w0 + w1 + w2) or 1
    w0, w1, w2 = w0/s, w1/s, w2/s
    return w0*state['I0'] + w1*state['I1'] + w2*state['I3']


def ht_total_for_pair(h, k):
    """Return total HT intensity for a single (h,k) pair."""
    w0, w1, w2 = state['w0'], state['w1'], state['w2']
    s = (w0 + w1 + w2) or 1
    w0, w1, w2 = w0/s, w1/s, w2/s
    F2 = F2_cache[(h, k)]
    return (
        w0 * I_inf(state['p0'], h, k, F2)
        + w1 * I_inf(state['p1'], h, k, F2)
        + w2 * I_inf(state['p3'], h, k, F2)
    )

def _raw_bragg(h, k, lo, hi, cif_path):
    """Return ``(L_vals, intensities)`` for integer L within ``[lo, hi]``."""
    lo_i = int(np.floor(lo))
    hi_i = int(np.ceil(hi))
    key = (cif_path, h, k, lo_i, hi_i)
    if key not in _BRAGG_CACHE:
        hkls = [(h, k, l) for l in range(lo_i, hi_i + 1)]
        intens = intensities_for_hkls(hkls, cif_path, [1.0], LAMBDA)
        L_vals = np.arange(lo_i, hi_i + 1)
        _BRAGG_CACHE[key] = (L_vals, np.asarray(intens))
    return _BRAGG_CACHE[key]


def _weight_2h_6h():
    """Return normalized weights for 2H and 6H contributions."""
    w0, w1 = state['w0'], state['w1']
    s = (w0 + w1) or 1.0
    return (w1 / s, w0 / s)  # 2H weight, 6H weight


def bragg_intensity_single(h, k):
    """Return ``(L_vals, b2h, b6h)`` scaled and weighted for one (h,k) pair."""
    lo, hi = state['L_lo'], state['L_hi']
    L_vals, raw2h = _raw_bragg(h, k, lo, hi, str(CIF_2H))
    _, raw6h = _raw_bragg(h, k, lo, hi, str(CIF_6H))
    ht = ht_total_for_pair(h, k)
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
    """Return weighted 2H and 6H intensities for all ``pairs``.

    The returned arrays correspond to integer ``L`` values within the
    current slider range. Intensities from each polytype are scaled to
    the Hendricks–Teller data and multiplied by the user weights before
    summing or further processing.
    """
    lo, hi = state['L_lo'], state['L_hi']
    L_vals = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1)
    total_2h = np.zeros_like(L_vals, dtype=float)
    total_6h = np.zeros_like(L_vals, dtype=float)
    total_ht = np.zeros_like(L_vals, dtype=float)
    idx = np.round(L_vals / L_MAX * (len(L_GRID) - 1)).astype(int)
    for h, k in pairs:
        L_tmp, b2 = _raw_bragg(h, k, lo, hi, str(CIF_2H))
        _, b6 = _raw_bragg(h, k, lo, hi, str(CIF_6H))
        total_2h += b2
        total_6h += b6
        total_ht += ht_total_for_pair(h, k)[idx]
    max_ht = float(total_ht.max()) if np.any(total_ht) else 1.0
    max_2h = float(total_2h.max()) if np.any(total_2h) else 1.0
    max_6h = float(total_6h.max()) if np.any(total_6h) else 1.0
    scale2 = max_ht / max_2h if max_2h else 1.0
    scale6 = max_ht / max_6h if max_6h else 1.0
    w2h, w6h = _weight_2h_6h()
    return L_vals, total_2h * scale2 * w2h, total_6h * scale6 * w6h

def _is_hk_mode() -> bool:                  ### ← NEW
    return state['mode'] == 'hk'
# ──────────────────────────────────────────────────────────────
# 2)  SIMPLE closed-form integrated area for **any** p
def ht_integrated_area(p, h, k, ell):
    """
    Analytic Hendricks–Teller area:
        A = 2π |F|² r² / (1 − r²)

    • p = 0  → p_eff = +P_EPS
    • p = 1  → p_eff = 1 − P_EPS
    """
    P_EPS = 1.0e-6           # 1 ppm away from the exact pole

    # ------- effective p (keep inside (0,1)) ------------------------
    if p <= 1e-15:
        p_eff = P_EPS
    elif p >= 1 - 1e-15:
        p_eff = 1.0 - P_EPS
    else:
        p_eff = p

    # ------- lattice slice -----------------------------------------
    idx   = int(round(ell / L_MAX * (N_L - 1)))
    F2    = F2_cache[(h, k)][idx]

    delta = 2 * np.pi * (2*h + k) / 3
    z     = (1 - p_eff) + p_eff * np.exp(-1j * delta)     # ← use np.exp
    r2    = abs(z)**2

    return 2 * np.pi * F2 * r2 / (1.0 - r2)


def ht_numeric_area(p, h, k, ell):
    """Numeric Hendricks–Teller peak area for a given integer ``ell``."""
    F2 = F2_cache[(h, k)]
    I_vals = I_inf(p, h, k, F2)
    mask = (L_GRID >= ell - 0.5) & (L_GRID <= ell + 0.5)
    return float(np.trapz(I_vals[mask], L_GRID[mask]))

# composite (still honours weights, but p-values are 0 now)
def analytic_area_weighted(h, k, ell):
    w0, w1, w2 = state['w0'], state['w1'], state['w2']
    s          = (w0 + w1 + w2) or 1
    w0, w1, w2 = w0/s, w1/s, w2/s
    return (
        w0 * ht_integrated_area(state['p0'], h, k, ell) +
        w1 * ht_integrated_area(state['p1'], h, k, ell) +
        w2 * ht_integrated_area(state['p3'], h, k, ell)
    )

# Numeric integration counterpart
def numeric_area_weighted(h, k, ell):
    w0, w1, w2 = _norm_weights()
    F2 = F2_cache[(h, k)]
    I_total = (
        w0 * I_inf(state['p0'], h, k, F2)
        + w1 * I_inf(state['p1'], h, k, F2)
        + w2 * I_inf(state['p3'], h, k, F2)
    )
    mask = (L_GRID >= ell - 0.5) & (L_GRID <= ell + 0.5)
    return float(np.trapz(I_total[mask], L_GRID[mask]))

# ──────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# helper – return normalised slider weights  w0 w1 w2  (sum = 1)
def _norm_weights():
    w0, w1, w2 = state['w0'], state['w1'], state['w2']
    s = (w0 + w1 + w2) or 1.0          # avoid division by zero
    return w0/s, w1/s, w2/s
# ------------------------------------------------------------------
# ────────── XLSX EXPORT  (changed block) ────────────────────────────
def export_bragg_data(_):
    """Save Bragg info to XLSX with numerically integrated HT areas."""
    root = tk.Tk(); root.withdraw()
    fname = tk.filedialog.asksaveasfilename(
        defaultextension='.xlsx',
        filetypes=[('Excel', '*.xlsx')]
    )
    if not fname:
        return

    rows, w2h, w6h = [], *_norm_weights()[:2]

    pairs = ([(state['h'], state['k'])] if _is_hk_mode()
             else HK_BY_M[state['m']])

    for h, k in pairs:
        L, raw2 = _raw_bragg(h, k, 0, L_MAX, str(CIF_2H))
        _, raw6 = _raw_bragg(h, k, 0, L_MAX, str(CIF_6H))

        # scale factors (unchanged logic)
        ht_curve = composite_tot()
        s_ht     = ht_curve.max() or 1.0
        sc2      = s_ht / (raw2.max() or 1.0)
        sc6      = s_ht / (raw6.max() or 1.0)

        for l, r2, r6 in zip(L, raw2, raw6):
            row = dict(h=h, k=k, l=int(l),
                       Dans2H_scaled=r2*sc2*w2h,
                       Dans2H_raw=r2,
                       Dans6H_scaled=r6*sc6*w6h,
                       Dans6H_raw=r6)

            if _is_hk_mode():                           ### ← NEW
                row['Numeric_2H_area'] = ht_numeric_area(state['p0'], h, k, l)
                row['Numeric_6H_area'] = ht_numeric_area(state['p1'], h, k, l)
            else:
                row['Numeric_area'] = numeric_area_weighted(h, k, l)

            rows.append(row)

    pd.DataFrame(rows).to_excel(fname, index=False)
    print("Saved →", fname)

# set up figure
fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(left=0.25, bottom=0.40, top=0.88)
ax.set_xlabel(r"$\ell$")
ax.set_ylabel("I (a.u.)")
ax.set_yscale('log')

line_tot, = ax.plot([], [], lw=2, label='Σ weighted')
line0,   = ax.plot([], [], ls='--', label='I(p≈0)')
line1,   = ax.plot([], [], ls='--', label='I(p≈1)')
line3,   = ax.plot([], [], ls='--', label='I(p)')
title = ax.set_title("")
_bragg_lines = []

# refresh function
def refresh(_=None):
    lo, hi = state['L_lo'], state['L_hi']
    mask = (L_GRID>=lo) & (L_GRID<=hi)
    tot = composite_tot()
    for line, I in [(line_tot, tot), (line0, state['I0']),
                    (line1, state['I1']), (line3, state['I3'])]:
        line.set_data(L_GRID[mask], I[mask])
    ax.set_xlim(lo, hi)
    ax.set_xticks(np.arange(np.ceil(lo), np.floor(hi)+1))
    vis = [state['w0']>0, state['w1']>0, state['w2']>0]
    for ln, v in zip((line0, line1, line3), vis):
        ln.set_visible(v)
    handles = [line_tot] + [ln for ln,v in zip((line0,line1,line3), vis) if v]
    for ln in _bragg_lines:
        ln.remove()
    _bragg_lines.clear()
    if state['show_bragg']:
        pairs = active_pairs()
        if state['mode'] == 'm':
            L_vals, i2h, i6h = bragg_intensity_sum(pairs)
            total = i2h + i6h
            msk = (L_vals >= lo) & (L_vals <= hi)
            ln, = ax.plot(
                L_vals[msk], total[msk], marker='D', ls='none',
                label=f'Bragg sum m={state["m"]}'
            )
            _bragg_lines.append(ln)
        else:
            for h, k in pairs:
                L_vals, i2h, i6h = bragg_intensity_single(h, k)
                msk = (L_vals >= lo) & (L_vals <= hi)
                ln2, = ax.plot(L_vals[msk], i2h[msk], marker='o', ls='none', label=f'2H({h},{k})')
                ln6, = ax.plot(L_vals[msk], i6h[msk], marker='s', ls='none', label=f'6H({h},{k})')
                _bragg_lines.extend([ln2, ln6])
        handles += _bragg_lines
    ax.legend(handles, [h.get_label() for h in handles], loc='upper right')
    m = state['m']; r = np.sqrt(m)
    Qr = 2*np.pi/A_HEX * np.sqrt(4*m/3) if m else 0
# ─── refresh(): tweak title string ────────────────────────────────────────────
    if state['mode'] == 'm':
        m = state['m']; r = np.sqrt(m)
        Qr = 2*np.pi/A_HEX * np.sqrt(4*m/3) if m else 0
        title.set_text(f"r=√m={r:.3f}, Qᵣ={Qr:.3f} Å⁻¹ | HK set: {HK_BY_M[m]}")
    else:
        h,k = state['h'], state['k']
        m   = h*h + h*k + k*k
        Qr  = 2*np.pi/A_HEX * np.sqrt(4*m/3)
        title.set_text(f"(h,k)=({h},{k}), m={m}, Qᵣ={Qr:.3f} Å⁻¹")

    ax.relim(); ax.autoscale_view()
    fig.canvas.draw_idle()

# slider factory
def make_slider(rect, label, vmin, vmax, val, valstep, cb):
    axr = plt.axes(rect)
    s   = Slider(axr, '', vmin, vmax, valinit=val, valstep=valstep)
    axr.text(0.5, 1.2, label, transform=axr.transAxes, ha='center')
    s.on_changed(cb)
    _sliders.append(s)
    return s

# ℓ-range slider
ax_range = plt.axes([0.25, 0.05, 0.65, 0.03])
rs = RangeSlider(ax_range, 'ℓ range', 0, L_MAX,
                 valinit=(state['L_lo'], state['L_hi']), valstep=0.1)
rs.on_changed(lambda v: (state.update(L_lo=v[0], L_hi=v[1]), refresh()))

# m-index slider
ys = [0.32, 0.26, 0.20, 0.14]
# new: capture the slider in a variable
m_slider = make_slider(
    [0.25, ys[0], 0.65, 0.03],
    'm index',
    min(ALLOWED_M), max(ALLOWED_M),
    state['m'],
    ALLOWED_M,
    lambda v: (state.update(m=int(v)), compute_components(), refresh())
)

# ─── build HK sliders (hidden by default) ‑ insert after m‑slider creation ────
hk_sliders = []
def make_int_slider(rect,label,vmax,key):
    s = make_slider(rect,label,-MAX_HK,vmax,state[key],1,
                    lambda v:(state.update({key:int(v)}),
                              compute_components(),refresh()))
    hk_sliders.append(s); s.ax.set_visible(False)

make_int_slider([0.25, ys[0], 0.30, 0.03],'H', MAX_HK,'h')
make_int_slider([0.60, ys[0], 0.30, 0.03],'K', MAX_HK,'k')

# p0 & w0
make_slider([0.25, ys[1], 0.45, 0.03],
            'p≈0', 0, 0.2, state['p0'], 1e-3,
            lambda v: (state.update(p0=v), compute_components(), refresh()))
make_slider([0.72, ys[1], 0.20, 0.03],
            'w(p≈0)%', 0,100,state['w0'],0.1,
            lambda v: (state.update(w0=v), refresh()))

# p1 & w1
make_slider([0.25, ys[2], 0.45, 0.03],
            'p≈1', 0.8,1, state['p1'],1e-3,
            lambda v: (state.update(p1=v), compute_components(), refresh()))
make_slider([0.72, ys[2], 0.20, 0.03],
            'w(p≈1)%', 0,100,state['w1'],0.1,
            lambda v: (state.update(w1=v), refresh()))

# p3 & w2
make_slider([0.25, ys[3], 0.45, 0.03],
            'p', 0,1, state['p3'],1e-3,
            lambda v: (state.update(p3=v), compute_components(), refresh()))
make_slider([0.72, ys[3], 0.20, 0.03],
            'w(p)%', 0,100,state['w2'],0.1,
            lambda v: (state.update(w2=v), refresh()))

# toggle scale button
btn_ax = plt.axes([0.42, 0.01, 0.16, 0.03])
b = Button(btn_ax, 'Toggle scale')
b.on_clicked(lambda _: (
    ax.set_yscale('linear' if ax.get_yscale()=='log' else 'log'),
    refresh()
))

# toggle Bragg peaks
btn_bragg = Button(plt.axes([0.25, 0.01, 0.13, 0.03]), 'Bragg on/off')
def _toggle_bragg(_):
    state['show_bragg'] = not state['show_bragg']
    refresh()
btn_bragg.on_clicked(_toggle_bragg)

# export button
btn_export = Button(plt.axes([0.78, 0.01, 0.16, 0.03]), 'Save Bragg XLSX')
btn_export.on_clicked(export_bragg_data)


def toggle_mode(_):
    state['mode'] = 'hk' if state['mode']=='m' else 'm'
    m_slider.ax.set_visible(state['mode']=='m')
    for s in hk_sliders:
        s.ax.set_visible(state['mode']=='hk')
    compute_components(); refresh()


b_mode = Button(plt.axes([0.60,0.01,0.16,0.03]), 'H/K panel')
b_mode.on_clicked(toggle_mode)

refresh()
plt.show()
