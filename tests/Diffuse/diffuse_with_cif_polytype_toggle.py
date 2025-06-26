#!/usr/bin/env python3
"""
Optimized Hendricks–Teller viewer for PbI₂:
- Precomputes F²(h,k,ℓ) once.
- Caches component intensities I0, I1, I3 and updates only when p-sliders or m change.
- Keeps weights and total recomputation minimal.
"""

import re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ───────────────────────── CIF helper ─────────────────────────

def c_from_cif(path: str) -> float:
    with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
        for ln in fp:
            if (m := re.match(r"_cell_length_c\s+([\d.]+)", ln)):
                return float(m.group(1))
    raise ValueError("_cell_length_c not found in CIF")

# ───────── constants & CIF paths ─────────
P_CLAMP = 1e-6
A_HEX   = 4.557  # Å
BUNDLE  = Path(getattr(sys, '_MEIPASS', Path(__file__).parent))
CIF_2H  = BUNDLE / 'PbI2_2H.cif'
CIF_6H  = BUNDLE / 'PbI2_6H.cif'
C_2H, C_6H = map(c_from_cif, map(str, (CIF_2H, CIF_6H)))
RATIO_26 = C_6H / C_2H

# atomic form factors
try:
    import xraydb
    f0 = lambda el, q: xraydb.f0(el, q)[0]
except ImportError:
    f0 = lambda el, q: {'Pb':82, 'I':53}.get(el, 0)

SITES = [(0,0,0.0,'Pb'),(1/3,2/3,0.25,'I'),(2/3,1/3,-0.25,'I')]
N_P, A_CELL = 3, 17.98e-10
AREA = (2*np.pi)**2/A_CELL * N_P

# group hk by radial index m
HK_BY_M = {}
for h in range(-5,6):
    for k in range(-5,6):
        HK_BY_M.setdefault(h*h + h*k + k*k, []).append((h,k))

# ℓ-grid
L_GRID = np.linspace(0,10,500)

# precompute F2_cache[(h,k)] = array over L_GRID
F2_cache = { (h,k): None for m in HK_BY_M for (h,k) in HK_BY_M[m] }
for (h,k) in F2_cache:
    Q = 2*np.pi*np.sqrt((4/3)*(h*h + k*k + h*k)/A_HEX**2 + (L_GRID**2)/C_2H**2)
    phases = np.array([np.exp(2j*np.pi*(h*x + k*y + L_GRID*z)) for x,y,z,_ in SITES])
    coeffs = np.array([f0(sym, Q) for *_, sym in SITES])
    F2_cache[(h,k)] = np.abs((coeffs[:,None] * phases).sum(axis=0))**2

# abc helper
def _abc(p,h,k):
    δ = 2*np.pi*((2*h+k)/3)
    z = (1-p) + p*np.exp(1j*δ)
    f = np.minimum(np.abs(z),1-P_CLAMP)
    ψ = np.angle(z)
    return f, ψ, δ

# intensity for infinite N
def I_inf(p, h, k, F2):
    f, ψ, δ = _abc(p,h,k)
    φ = δ + 2*np.pi * L_GRID / 3
    return AREA * F2 * (1 - f**2) / (1 + f**2 - 2*f*np.cos(φ-ψ))

# compute component arrays on demand
state = {
    'm': 0,
    'p0': 0.05, 'p1': 0.95, 'p3': 0.5,
    'w0':33.3, 'w1':33.3, 'w2':33.4,
    'I0': None, 'I1': None, 'I3': None
}

# initial comp on startup
def compute_components():
    m = state['m']
    state['I0'] = sum(I_inf(state['p0'],h,k,F2_cache[(h,k)]) for h,k in HK_BY_M[m])
    state['I1'] = sum(I_inf(state['p1'],h,k,F2_cache[(h,k)]) for h,k in HK_BY_M[m])
    state['I3'] = sum(I_inf(state['p3'],h,k,F2_cache[(h,k)]) for h,k in HK_BY_M[m])
compute_components()

# composite using cached arrays
def composite_tot():
    w0,w1,w2 = state['w0'],state['w1'],state['w2']
    s = (w0+w1+w2) or 1.0
    w0,w1,w2 = w0/s, w1/s, w2/s
    return w0*state['I0'] + w1*state['I1'] + w2*state['I3']

# ───────── plot setup ─────────
fig,ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(left=0.25,bottom=0.38,top=0.88)
sec = ax.secondary_xaxis('top', functions=(lambda x:x*RATIO_26, lambda x:x/RATIO_26))
ax.set_xlabel(r"$\ell^{(2H)}$"); sec.set_xlabel(r"$\ell^{(6H)}$")
ax.set_xticks(range(11)); sec.set_xticks(range(int(np.floor(10*RATIO_26))+1))
ax.set_yscale('log'); ax.set_ylabel('I')
line_tot,=ax.plot(L_GRID,np.zeros_like(L_GRID),lw=2,label='Σ weighted')
line0,  =ax.plot(L_GRID,np.zeros_like(L_GRID),ls='--',label='I(p≈0)')
line1,  =ax.plot(L_GRID,np.zeros_like(L_GRID),ls='--',label='I(p≈1)')
line3,  =ax.plot(L_GRID,np.zeros_like(L_GRID),ls='--',label='I(p)')
ax.legend(loc='upper right'); title=ax.set_title('')

# ───────── refresh ─────────
def refresh(_=None):
    # update total; components already cached
    tot = composite_tot()
    line_tot.set_ydata(tot)
    line0.set_ydata(state['I0']); line1.set_ydata(state['I1']); line3.set_ydata(state['I3'])
    m=state['m']; r=np.sqrt(m); Qr=2*np.pi/A_HEX*np.sqrt(4*m/3) if m else 0.0
    title.set_text(f"r=√{{}}={r:.3f}, Qr={Qr:.3f} Å⁻¹ | HK: {HK_BY_M.get(m,[])}")
    ax.relim(); ax.autoscale_view(); fig.canvas.draw_idle()

# ───────── slider + button setup ─────────
from matplotlib.widgets import Slider, Button

def slider(rect,label,vmin,vmax,val,step,callback):
    ax_s = plt.axes(rect)
    s = Slider(ax_s,'',vmin,vmax,valinit=val,valstep=step)
    ax_s.text(0.5,1.2,label,transform=ax_s.transAxes,ha='center',va='bottom')
    s.on_changed(callback)
    return s

ys=[0.30,0.24,0.18,0.12]
s_m = slider([0.25,ys[0],0.65,0.03],'m index',0,max(HK_BY_M),state['m'],1,
             lambda v: (state.update(m=int(v)), compute_components(), refresh()))
# p0
s_p0=slider([0.25,ys[1],0.45,0.03],'p≈0',0,0.2,state['p0'],1e-3,
             lambda v: (state.update(p0=v), compute_components(), refresh()))
# w0
def cb_w0(v): state.update(w0=v); refresh()
s_w0=slider([0.72,ys[1],0.20,0.03],'w(p≈0)%',0,100,state['w0'],0.1,cb_w0)
# p1
s_p1=slider([0.25,ys[2],0.45,0.03],'p≈1',0.8,1,state['p1'],1e-3,
             lambda v: (state.update(p1=v), compute_components(), refresh()))
# w1
def cb_w1(v): state.update(w1=v); refresh()
s_w1=slider([0.72,ys[2],0.20,0.03],'w(p≈1)%',0,100,state['w1'],0.1,cb_w1)
# p3
s_p3=slider([0.25,ys[3],0.45,0.03],'p',0,1,state['p3'],1e-3,
             lambda v: (state.update(p3=v), compute_components(), refresh()))
# w2
def cb_w2(v): state.update(w2=v); refresh()
s_w2=slider([0.72,ys[3],0.20,0.03],'w(p)%',0,100,state['w2'],0.1,cb_w2)

# toggle scale button
btn = Button(plt.axes([0.42,0.04,0.16,0.05]),'Toggle scale')
def toggle(_): ax.set_yscale('linear' if ax.get_yscale()=='log' else 'log'); refresh()
btn.on_clicked(toggle)

# initial draw
refresh()
plt.show()
