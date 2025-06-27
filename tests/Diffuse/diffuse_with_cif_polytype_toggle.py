#!/usr/bin/env python3
"""
Optimised Hendricks–Teller viewer for PbI₂
──────────────────────────────────────────
* Pre-computes F²(h,k,ℓ) once.
* Caches component intensities for three p-values.
* ℓ-range RangeSlider up to `L_MAX` (default 50).
* All sliders placed below the figure area; no overlap.
* Top secondary axis removed.
"""
import re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider, Button

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

try:
    import xraydb
    f0 = lambda el, q: xraydb.f0(el, q)[0]
except ImportError:
    f0 = lambda el, q: {"Pb":82, "I":53}.get(el, 0)

SITES    = [(0,0,0.0,'Pb'), (1/3,2/3,0.25,'I'), (2/3,1/3,-0.25,'I')]
N_P, A_CELL = 3, 17.98e-10
AREA     = (2*np.pi)**2/A_CELL * N_P

# group (h,k) by radial index m
HK_BY_M = {}
for h in range(-5,6):
    for k in range(-5,6):
        HK_BY_M.setdefault(h*h+h*k+k*k, []).append((h,k))

# ℓ grid
L_MAX  = 50
N_L    = 2001
L_GRID = np.linspace(0, L_MAX, N_L)

# precompute |F|²
F2_cache = {}
for m in HK_BY_M:
    for h,k in HK_BY_M[m]:
        Q = 2*np.pi * np.sqrt((4/3)*(h*h+k*k+h*k)/A_HEX**2 + (L_GRID**2)/C_2H**2)
        phases = np.array([np.exp(2j*np.pi*(h*x + k*y + L_GRID*z))
                           for x,y,z,_ in SITES])
        coeffs = np.array([f0(sym, Q) for *_, sym in SITES])
        F2_cache[(h,k)] = np.abs((coeffs[:,None]*phases).sum(axis=0))**2

def _abc(p,h,k):
    δ = 2*np.pi*((2*h+k)/3)
    z = (1-p) + p*np.exp(1j*δ)
    f = np.minimum(np.abs(z), 1-P_CLAMP)
    ψ = np.angle(z)
    return f,ψ,δ

def I_inf(p,h,k,F2):
    f,ψ,δ = _abc(p,h,k)
    φ = δ + 2*np.pi * L_GRID/3
    return AREA * F2 * (1-f**2)/(1 + f**2 - 2*f*np.cos(φ-ψ))

state = {
    'm':0, 'p0':0.05, 'p1':0.95, 'p3':0.5,
    'w0':33.3,'w1':33.3,'w2':33.4,
    'I0':None,'I1':None,'I3':None,
    'L_lo':1.0,'L_hi':20.0
}

def compute_components():
    m = state['m']
    state['I0'] = sum(I_inf(state['p0'],h,k,F2_cache[(h,k)]) for h,k in HK_BY_M[m])
    state['I1'] = sum(I_inf(state['p1'],h,k,F2_cache[(h,k)]) for h,k in HK_BY_M[m])
    state['I3'] = sum(I_inf(state['p3'],h,k,F2_cache[(h,k)]) for h,k in HK_BY_M[m])
compute_components()

def composite_tot():
    w0,w1,w2 = state['w0'], state['w1'], state['w2']
    s = (w0 + w1 + w2) or 1
    w0, w1, w2 = w0/s, w1/s, w2/s
    return w0*state['I0'] + w1*state['I1'] + w2*state['I3']

# figure
fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(left=0.25, bottom=0.40, top=0.88)

ax.set_xlabel(r"$\ell$")
ax.set_ylabel('I (a.u.)')
ax.set_yscale('log')

line_tot, = ax.plot([], [], lw=2, label='Σ weighted')
line0,   = ax.plot([], [], ls='--', label='I(p≈0)')
line1,   = ax.plot([], [], ls='--', label='I(p≈1)')
line3,   = ax.plot([], [], ls='--', label='I(p)')

title = ax.set_title('')

def refresh(_=None):
    lo, hi = state['L_lo'], state['L_hi']
    mask = (L_GRID>=lo)&(L_GRID<=hi)
    tot = composite_tot()

    # update curve data
    for line, I in [
        (line_tot, tot),
        (line0, state['I0']),
        (line1, state['I1']),
        (line3, state['I3'])
    ]:
        line.set_data(L_GRID[mask], I[mask])

    # axis limits & ticks
    ax.set_xlim(lo, hi)
    ax.set_xticks(np.arange(np.ceil(lo), np.floor(hi)+1))

    # hide zero-weight components
    vis0 = state['w0'] > 0
    vis1 = state['w1'] > 0
    vis2 = state['w2'] > 0
    line0.set_visible(vis0)
    line1.set_visible(vis1)
    line3.set_visible(vis2)

    # rebuild legend
    handles = [line_tot]
    labels  = [line_tot.get_label()]
    if vis0:
        handles.append(line0); labels.append(line0.get_label())
    if vis1:
        handles.append(line1); labels.append(line1.get_label())
    if vis2:
        handles.append(line3); labels.append(line3.get_label())
    ax.legend(handles, labels, loc='upper right')

    # update title
    m = state['m']; r = np.sqrt(m)
    Qr = 2*np.pi/A_HEX*np.sqrt(4*m/3) if m else 0
    title.set_text(f"r=√m={r:.3f}, Qᵣ={Qr:.3f} Å⁻¹ | HK: {HK_BY_M[m]}")

    ax.relim(); ax.autoscale_view()
    fig.canvas.draw_idle()

def make_slider(rect, label, vmin, vmax, val, step, cb):
    axr = plt.axes(rect)
    s   = Slider(axr, '', vmin, vmax, valinit=val, valstep=step)
    axr.text(0.5, 1.2, label, transform=axr.transAxes, ha='center')
    s.on_changed(cb)
    _sliders.append(s)
    return s

# ℓ-range slider
ax_range = plt.axes([0.25, 0.05, 0.65, 0.03])
rs = RangeSlider(ax_range, 'ℓ range', 0, L_MAX,
                 valinit=(state['L_lo'], state['L_hi']), valstep=0.1)
rs.on_changed(lambda v: (state.update(L_lo=v[0], L_hi=v[1]), refresh()))

# other sliders
ys = [0.32, 0.26, 0.20, 0.14]
make_slider([0.25, ys[0], 0.65, 0.03],
            'm index', 0, max(HK_BY_M), state['m'], 1,
            lambda v: (state.update(m=int(v)), compute_components(), refresh()))
make_slider([0.25, ys[1], 0.45, 0.03],
            'p≈0', 0, 0.2, state['p0'], 1e-3,
            lambda v: (state.update(p0=v), compute_components(), refresh()))
make_slider([0.72, ys[1], 0.20, 0.03],
            'w(p≈0)%', 0,100,state['w0'],0.1,
            lambda v: (state.update(w0=v), refresh()))
make_slider([0.25, ys[2], 0.45, 0.03],
            'p≈1', 0.8,1, state['p1'], 1e-3,
            lambda v: (state.update(p1=v), compute_components(), refresh()))
make_slider([0.72, ys[2], 0.20, 0.03],
            'w(p≈1)%', 0,100,state['w1'],0.1,
            lambda v: (state.update(w1=v), refresh()))
make_slider([0.25, ys[3], 0.45, 0.03],
            'p', 0,1, state['p3'], 1e-3,
            lambda v: (state.update(p3=v), compute_components(), refresh()))
make_slider([0.72, ys[3], 0.20, 0.03],
            'w(p)%', 0,100,state['w2'],0.1,
            lambda v: (state.update(w2=v), refresh()))

# toggle scale button
btn_ax = plt.axes([0.42, 0.01, 0.16, 0.03])
b = Button(btn_ax, 'Toggle scale')
b.on_clicked(lambda _: (ax.set_yscale('linear' if ax.get_yscale()=='log' else 'log'), refresh()))

refresh()
plt.show()
