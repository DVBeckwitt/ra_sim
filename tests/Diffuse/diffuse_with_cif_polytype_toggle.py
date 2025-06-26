#!/usr/bin/env python3
"""
Fast interactive Hendricks–Teller viewer for PbI₂.
Three-component HT intensity (p≈0, p≈1, free-p) shown as weighted sum.
Sliders have labels above and min/max below tracks; paired sliders are side by side.
Toggle button sits below all sliders.
"""

import re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ───────── helpers ─────────

def c_from_cif(path: str) -> float:
    with open(path, encoding="utf-8", errors="ignore") as fp:
        for ln in fp:
            if m := re.match(r"_cell_length_c\s+([\d.]+)", ln):
                return float(m.group(1))
    raise ValueError("_cell_length_c not found in CIF")

# ───────── constants & CIF paths ─────────
P_CLAMP = 1e-6
A_HEX   = 4.557  # Å
BUNDLE  = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
C2H = BUNDLE / "PbI2_2H.cif"
C6H = BUNDLE / "PbI2_6H.cif"
C_2H, C_6H = map(c_from_cif, map(str, (C2H, C6H)))
RATIO_26  = C_6H / C_2H

try:
    import xraydb
    _f0 = lambda el, q: xraydb.f0(el, q)[0]
except ImportError:
    _f0 = lambda el, q: {"Pb":82, "I":53}.get(el, 0)

SITES = [(0,0,0.0,"Pb"),(1/3,2/3,0.25,"I"),(2/3,1/3,-0.25,"I")]
N_P, A_CELL = 3, 17.98e-10
AREA = (2*np.pi)**2 / A_CELL * N_P

# group hk by radial index
HK_BY_M = {}
for h in range(-5,6):
    for k in range(-5,6):
        HK_BY_M.setdefault(h*h + h*k + k*k, []).append((h,k))

# ───────── scattering & HT formulas ─────────

def _Qmag(h,k,L):
    return 2*np.pi*np.sqrt((4/3)*(h*h+k*k+h*k)/A_HEX**2 + (L**2)/C_2H**2)

def _F2(h,k,L):
    Q = _Qmag(h,k,L)
    ph = [np.exp(2j*np.pi*(h*x+k*y+L*z)) for x,y,z,_ in SITES]
    cf = [_f0(sym,Q) for *_,sym in SITES]
    return np.abs(sum(c*p for c,p in zip(cf,ph)))**2


def _abc(p,h,k):
    δ = 2*np.pi*((2*h+k)/3)
    z = (1-p) + p*np.exp(1j*δ)
    f = np.minimum(np.abs(z),1-P_CLAMP)
    ψ = np.angle(z)
    return f,ψ,δ


def I_inf(L,p,h,k,F2):
    f,ψ,δ = _abc(p,h,k)
    φ = δ + 2*np.pi*L/3
    return AREA * F2 * (1 - f**2) / (1 + f**2 - 2*f*np.cos(φ-ψ))

# ───────── state & grid ─────────
L_GRID = np.linspace(0,10,500)
state = {"m":0, "p0":0.05, "p1":0.95, "p3":0.5, "w0":33.3, "w1":33.3, "w2":33.4}

# ───────── plot setup ─────────
fig, ax = plt.subplots(figsize=(8,6))
# reserve space below for controls
plt.subplots_adjust(left=0.25, bottom=0.38, top=0.88)

sec = ax.secondary_xaxis('top', functions=(lambda x:x*RATIO_26, lambda x:x/RATIO_26))
ax.set_xlabel(r"$\ell^{(2H)}$"); sec.set_xlabel(r"$\ell^{(6H)}$")
ax.set_xticks(range(11)); sec.set_xticks(range(int(np.floor(10*RATIO_26))+1))
ax.set_yscale('log'); ax.set_ylabel('I')

line_tot, = ax.plot(L_GRID, np.zeros_like(L_GRID), lw=2, label='Σ weighted')
line0,    = ax.plot(L_GRID, np.zeros_like(L_GRID), ls='--', label='I(p≈0)')
line1,    = ax.plot(L_GRID, np.zeros_like(L_GRID), ls='--', label='I(p≈1)')
line3,    = ax.plot(L_GRID, np.zeros_like(L_GRID), ls='--', label='I(p)')
ax.legend(loc='upper right')
title = ax.set_title('')

# ───────── compute intensities ─────────

def _I_single(p,m):
    y = np.zeros_like(L_GRID)
    for h,k in HK_BY_M.get(m,[]): y += I_inf(L_GRID,p,h,k,_F2(h,k,L_GRID))
    return y


def _composite(m,p0,p1,p3,w0,w1,w2):
    I0, I1, I3 = _I_single(p0,m), _I_single(p1,m), _I_single(p3,m)
    tot = (w0+w1+w2) or 1.0
    w0, w1, w2 = w0/tot, w1/tot, w2/tot
    return w0*I0 + w1*I1 + w2*I3, (I0,I1,I3)

# ───────── refresh plot ─────────

def refresh(_=None):
    m = state['m']; tot, (I0,I1,I3) = _composite(m, state['p0'], state['p1'], state['p3'], state['w0'], state['w1'], state['w2'])
    line_tot.set_ydata(tot); line0.set_ydata(I0); line1.set_ydata(I1); line3.set_ydata(I3)
    r = np.sqrt(m); Qr = 2*np.pi/A_HEX * np.sqrt(4*m/3) if m else 0.0
    title.set_text(f"r=√{{}}={r:.3f}, Qr={Qr:.3f} Å⁻¹ | HK: {HK_BY_M.get(m,[])}")
    ax.relim(); ax.autoscale_view(); fig.canvas.draw_idle()

# ───────── slider factory ─────────

def make_slider(rect, label, vmin, vmax, val, step):
    ax_s = plt.axes(rect)
    s = Slider(ax_s, '', vmin, vmax, valinit=val, valstep=step)
    # label above
    ax_s.text(0.5, 1.2, label, transform=ax_s.transAxes, ha='center', va='bottom')
    # min/max below
    ax_s.text(0, -0.5, f"{vmin:g}", transform=ax_s.transAxes, ha='left',   va='top')
    ax_s.text(1, -0.5, f"{vmax:g}", transform=ax_s.transAxes, ha='right',  va='top')
    return s

# rows for sliders
ys = [0.30, 0.24, 0.18, 0.12]
# shell idx (full width)
s_m  = make_slider([0.25, ys[0], 0.65, 0.03], 'm index',    0, max(HK_BY_M), state['m'], 1)
# p≈0 + w(p≈0)
s_p0 = make_slider([0.25, ys[1], 0.45, 0.03], 'p≈0',      0.0, 0.2, state['p0'], 1e-3)
s_w0 = make_slider([0.72, ys[1], 0.20, 0.03], 'w(p≈0)%',   0, 100,  state['w0'], 0.1)
# p≈1 + w(p≈1)
s_p1 = make_slider([0.25, ys[2], 0.45, 0.03], 'p≈1',      0.8, 1.0, state['p1'], 1e-3)
s_w1 = make_slider([0.72, ys[2], 0.20, 0.03], 'w(p≈1)%',   0, 100,  state['w1'], 0.1)
# free p + w(p)
s_p3 = make_slider([0.25, ys[3], 0.45, 0.03], 'p',        0.0, 1.0, state['p3'], 1e-3)
s_w2 = make_slider([0.72, ys[3], 0.20, 0.03], 'w(p)%',     0, 100,  state['w2'], 0.1)

# callbacks
for name, sl in zip(['m','p0','w0','p1','w1','p3','w2'], [s_m,s_p0,s_w0,s_p1,s_w1,s_p3,s_w2]):
    sl.on_changed(lambda v, n=name: (state.update({n:int(v) if n=='m' else float(v)}), refresh()))

# toggle button below sliders
btn_ax = plt.axes([0.42, 0.04, 0.16, 0.05])
btn = Button(btn_ax, 'Toggle scale')
btn.on_clicked(lambda _: (ax.set_yscale('linear' if ax.get_yscale()=='log' else 'log'), refresh()))

refresh()
plt.show()
