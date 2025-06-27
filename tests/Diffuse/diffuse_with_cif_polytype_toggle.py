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
import Dans_Diffraction as dd

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

# constants for Hendricks–Teller model
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
L_INT  = np.arange(0, L_MAX+1)

XTALS = {
    '2H': dd.Crystal(str(CIF_2H)),
    '6H': dd.Crystal(str(CIF_6H)),
}

def make_caches(xtl):
    grid_cache = {}
    disc_cache = {}
    for pairs in HK_BY_M.values():
        for h, k in pairs:
            hk_grid = np.column_stack([
                np.full_like(L_GRID, h),
                np.full_like(L_GRID, k),
                L_GRID,
            ])
            grid_cache[(h, k)] = xtl.Scatter.intensity(hk_grid, 'xray', energy_kev=E_CuKa/1000)

            hk_disc = np.array([[h, k, l] for l in L_INT])
            disc_cache[(h, k)] = xtl.Scatter.intensity(hk_disc, 'xray', energy_kev=E_CuKa/1000)
    return grid_cache, disc_cache

F2_cache = {}
BRAGG_cache = {}
for key, xtl in XTALS.items():
    F2_cache[key], BRAGG_cache[key] = make_caches(xtl)
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
    'L_lo': 1.0, 'L_hi': L_MAX,
    'poly': '2H',
    'show_bragg': True,
}
state = defaults.copy()

# ─── after 'state = defaults.copy()' ───────────────────────────────────────────
state.update({'mode': 'm',      # 'm' or 'hk'
              'h': 0, 'k': 0})  # current HK in hk‑mode
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

    fc = F2_cache[state['poly']]
    def comp(p):
        return sum(n * I_inf(p, h, k, fc[(h, k)])
                   for (h, k), n in counts.items())

    state['I0'], state['I1'], state['I3'] = comp(state['p0']), comp(state['p1']), comp(state['p3'])

compute_components()

def composite_tot():
    w0,w1,w2 = state['w0'], state['w1'], state['w2']
    s = (w0 + w1 + w2) or 1
    w0, w1, w2 = w0/s, w1/s, w2/s
    return w0*state['I0'] + w1*state['I1'] + w2*state['I3']

def bragg_tot():
    pairs = active_pairs()
    if state['mode'] == 'hk':
        counts = {(h, k): 1 for h, k in pairs}
    else:
        counts = Counter((abs(h), abs(k)) for h, k in pairs)
    bc = BRAGG_cache[state['poly']]
    tot = np.zeros(len(L_INT))
    for (h, k), n in counts.items():
        tot += n * bc[(h, k)]
    return tot

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
bragg_scat = ax.scatter([], [], c='C4', marker='o', label='Bragg', zorder=3)
title = ax.set_title("")

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
    if state.get('show_bragg', True):
        b_tot = bragg_tot()
        mask_b = (L_INT>=lo) & (L_INT<=hi)
        bragg_scat.set_offsets(np.column_stack([L_INT[mask_b], b_tot[mask_b]]))
        bragg_scat.set_visible(True)
        handles.append(bragg_scat)
    else:
        bragg_scat.set_visible(False)
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


def toggle_mode(_):
    state['mode'] = 'hk' if state['mode']=='m' else 'm'
    m_slider.ax.set_visible(state['mode']=='m')
    for s in hk_sliders:
        s.ax.set_visible(state['mode']=='hk')
    compute_components(); refresh()


def toggle_poly(_):
    state['poly'] = '6H' if state['poly']=='2H' else '2H'
    compute_components(); refresh()


def toggle_bragg(_):
    state['show_bragg'] = not state['show_bragg']
    refresh()


b_mode = Button(plt.axes([0.60,0.01,0.16,0.03]), 'H/K panel')
b_mode.on_clicked(toggle_mode)

b_poly = Button(plt.axes([0.24,0.01,0.16,0.03]), 'Polytype')
b_poly.on_clicked(toggle_poly)

b_bragg = Button(plt.axes([0.78,0.01,0.16,0.03]), 'Bragg on/off')
b_bragg.on_clicked(toggle_bragg)

refresh()
plt.show()
