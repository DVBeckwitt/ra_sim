# ────────────────────────── global constants (unchanged) ──────────────────────────
import numpy as np
import os
A_HEX   = 4.557
P_CLAMP = 1e-6
N_P, A_C = 3, 17.98e-10
AREA    = (2*np.pi)**2 / A_C * N_P
_FALLBACK_Z = {"Pb": 82, "I": 53}
_SITES  = [(0.0, 0.0,  0.000, "Pb"),
           (1/3, 2/3,  0.25 , "I"),
           (2/3, 1/3, -0.25 , "I")]

# Pre-compute site coordinates and atomic numbers for vectorised operations
_SITES_POS = np.array([(x, y, z) for x, y, z, _ in _SITES], dtype=float)
_SITES_Z   = np.array([_FALLBACK_Z.get(sym, 0.) for *_, sym in _SITES],
                      dtype=float)
_TWO_PI = 2 * np.pi

# Cache of base L grids and F2 values keyed by parameters that do not depend
# on occupancy or stacking probability.  Each entry is a mapping
# ``(h,k) -> {"L": array, "F2": array}``.
_HT_BASE_CACHE: dict[tuple, dict] = {}

# ───────────────────────── occupancy helper ─────────────────────────
def _temp_cif_with_occ(cif_in: str, occ):
    """
    Return path of a temporary CIF in which every `_atom_site_occupancy`
    is multiplied by `occ`.  Accepts:
      • scalar      → global factor
      • list/tuple  → per-site factors; if length ≠ n_sites, occ[0] used.
    """
    import tempfile, os, CifFile
    cf          = CifFile.ReadCif(cif_in)
    block_name  = list(cf.keys())[0]
    block       = cf[block_name]

    occ_field = block.get('_atom_site_occupancy')
    if occ_field is None:                     # CIF had no occupancies
        labels = block.get('_atom_site_label')
        occ_field = ['1.0'] * (len(labels) if isinstance(labels, list) else 1)
        block['_atom_site_occupancy'] = occ_field

    # harmonise occ → list of factors
    if isinstance(occ, (list, tuple)):
        if len(occ) != len(occ_field):
            occ = [occ[0]] * len(occ_field)
    else:
        occ = [occ] * len(occ_field)

    for i, fac in enumerate(occ):
        occ_field[i] = str(float(occ_field[i]) * float(fac))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.cif')
    tmp.close()
    try:             # PyCifRW ≥ 5
        CifFile.WriteCif(cf, tmp.name)
    except AttributeError:
        with open(tmp.name, 'w') as f:
            f.write(cf.WriteOut())
    return tmp.name, os.unlink          # unlink-fn for later cleanup


# ───────────────────────── low-level physics helpers (unchanged) ─────────────────────────
def _cell_c_from_cif(cif_path: str) -> float:
    """Return the c lattice parameter from ``cif_path`` multiplied by three."""
    import re

    pat = re.compile(r"_cell_length_c\s+([\d.]+)")
    with open(cif_path, "r", encoding="utf-8", errors="ignore") as fp:
        for ln in fp:
            m = pat.match(ln)
            if m:
                return float(m.group(1)) * 3.0
    raise ValueError(f"_cell_length_c not found in {cif_path}")

from .factors import F_comp

def _Qmag(h, k, L, c_2h):
    import numpy as np
    inv_d2 = (4/3)*(h*h+k*k+h*k)/A_HEX**2 + (L**2)/c_2h**2
    return 2*np.pi*np.sqrt(inv_d2)

def _F2(h, k, L, c_2h, energy_kev):
    """Return |F|² using tabulated form factors."""
    import numpy as np

    Q = _Qmag(h, k, L, c_2h)
    F = np.zeros_like(Q, dtype=complex)
    for x, y, z, sym in _SITES:
        ff = F_comp(sym, Q, energy_kev)
        phase = np.exp(1j * _TWO_PI * (h * x + k * y + L * z))
        F += ff * phase
    return np.abs(F) ** 2

def _abc(p, h, k):
    """Compute amplitude factor ``f`` and phase ``ψ`` without complex math."""
    import numpy as np

    δ = _TWO_PI * ((2 * h + k) / 3)
    real = (1 - p) + p * np.cos(δ)
    imag = p * np.sin(δ)
    abs_z = np.hypot(real, imag)
    f = np.minimum(abs_z, 1 - P_CLAMP)
    ψ = np.arctan2(imag, real)
    return f, ψ, δ

def _I_inf(L, p, h, k, F2):
    import numpy as np
    f, ψ, δ = _abc(p, h, k)
    φ = δ + _TWO_PI * L/3
    return AREA * F2 * (1-f**2) / (1 + f**2 - 2*f*np.cos(φ-ψ))



def _get_base_curves(
    cif_path: str,
    hk_list=None,
    mx: int | None = None,
    L_step: float = 0.01,
    L_max: float = 10.0,
    two_theta_max: float | None = None,
    lambda_: float = 1.54,
):
    """Return cached ``{(h,k): {"L": ..., "F2": ...}}`` independent of ``occ`` and ``p``."""
    import numpy as np, itertools, math

    if hk_list is None:
        if mx is None:
            raise ValueError("Specify hk_list or mx")
        hk_list = [
            (h, k)
            for h, k in itertools.product(range(-mx + 1, mx), repeat=2)
            if not (h == 0 and k == 0)
        ]

    key = (
        os.path.abspath(cif_path),
        tuple(hk_list),
        float(L_step),
        float(L_max),
        two_theta_max,
        float(lambda_),
    )

    cached = _HT_BASE_CACHE.get(key)
    if cached is not None:
        return cached

    c_2h = _cell_c_from_cif(cif_path)

    if L_step <= 0.0:
        raise ValueError("L_step must be > 0")
    if L_step < 1e-4:
        L_step = 1e-4

    energy_kev = 12398.4193 / lambda_
    out: dict[tuple, dict] = {}
    if two_theta_max is None:
        base_L = np.arange(0.0, L_max + L_step / 2, L_step)
        for h, k in hk_list:
            F2 = _F2(h, k, base_L, c_2h, energy_kev)
            out[(h, k)] = {"L": base_L.copy(), "F2": F2}
    else:
        q_max = (4 * math.pi / lambda_) * math.sin(math.radians(two_theta_max / 2))
        for h, k in hk_list:
            const = (4 / 3) * (h * h + k * k + h * k) / (A_HEX**2)
            l_sq = (q_max / (2 * math.pi)) ** 2 - const
            if l_sq <= 0:
                L_vals = np.array([], dtype=float)
                out[(h, k)] = {"L": L_vals, "F2": L_vals}
                continue
            L_max_local = c_2h * math.sqrt(l_sq)
            L_vals = np.arange(0.0, L_max_local + L_step / 2, L_step)
            F2 = _F2(h, k, L_vals, c_2h, energy_kev)
            out[(h, k)] = {"L": L_vals.copy(), "F2": F2}

    _HT_BASE_CACHE[key] = out
    return out


# ───────────────────────── revitalised public routine ─────────────────────────
def ht_Iinf_dict(
        cif_path: str,
        hk_list=None,                  # explicit list or None
        mx: int|None = None,           # generate –mx+1…mx-1 if hk_list is None
        occ=1.0,                       # occupancy scaling, same rules as miller_generator
        p: float = 0.1,
        L_step: float = 0.01,
        L_max: float = 10.0,
        two_theta_max: float | None = None,
        lambda_: float = 1.54,
    ):
    """
    Infinite-domain Hendricks–Teller intensities, now with:
      • on-the-fly occupancy scaling (scalar / list)
      • optional automatic HK grid via `mx` (drop h=k=0)
      • limit by ``two_theta_max`` instead of ``L_max`` if desired
    Returns { (h,k): {'L':…, 'I':…} }
    """
    base = _get_base_curves(
        cif_path=cif_path,
        hk_list=hk_list,
        mx=mx,
        L_step=L_step,
        L_max=L_max,
        two_theta_max=two_theta_max,
        lambda_=lambda_,
    )

    out = {}
    for (h, k), data in base.items():
        L_vals = data["L"]
        F2 = data["F2"]
        I = _I_inf(L_vals, p, h, k, F2)
        out[(h, k)] = {"L": L_vals.copy(), "I": I}

    return out


# ---------------------------------------------------------------------------
# Helper: convert ``ht_Iinf_dict`` output to ``(miller, intens, degeneracy, details)``
# ---------------------------------------------------------------------------
def ht_dict_to_arrays(ht_curves):
    """Return arrays in the same format as :func:`miller_generator`.

    Parameters
    ----------
    ht_curves : dict
        Mapping ``(h, k)`` to ``{"L": array, "I": array}`` as returned by
        :func:`ht_Iinf_dict`.

    Returns
    -------
    miller : ndarray
        ``(N, 3)`` array of Miller indices including fractional ``L`` values.
    intensities : ndarray
        ``(N,)`` array of intensities corresponding to ``miller``.
    degeneracy : ndarray
        ``(N,)`` array with all ones, matching :func:`miller_generator` output.
    details : list
        List of length ``N`` with ``[((h, k, L), intensity)]`` records.
    """
    import numpy as np

    total = sum(len(c["L"]) for c in ht_curves.values())

    miller = np.empty((total, 3), dtype=np.float64)
    intens = np.empty(total, dtype=np.float64)
    degeneracy = np.ones(total, dtype=np.int32)
    details = []

    idx = 0
    for (h, k), curve in ht_curves.items():
        L_vals = curve["L"]
        I_vals = curve["I"]
        n = len(L_vals)

        miller[idx:idx+n, 0] = h
        miller[idx:idx+n, 1] = k
        miller[idx:idx+n, 2] = L_vals
        intens[idx:idx+n] = I_vals

        for L_val, inten in zip(L_vals, I_vals):
            details.append([((h, k, float(L_val)), float(inten))])

        idx += n

    return miller, intens, degeneracy, details


# ---------------------------------------------------------------------------
# New helpers: group HT curves by radial index (Qr rods)
# ---------------------------------------------------------------------------
def ht_dict_to_qr_dict(ht_curves):
    """Combine Hendricks–Teller curves with identical radial index.

    Parameters
    ----------
    ht_curves : dict
        Mapping ``(h, k)`` to ``{"L": array, "I": array}``.

    Returns
    -------
    dict
        ``{m: {"L": array, "I": array, "hk": (h,k)}}`` keyed by the radial index
        ``m = h*h + h*k + k*k``. Intensities for curves with the same ``m`` are
        summed and one representative ``(h,k)`` pair is stored. If the ``L`` grids
        for reflections with the same radial index differ, the intensities are
        interpolated onto the union of both grids before summation.
    """
    import numpy as np

    rods = {}
    for (h, k), curve in ht_curves.items():
        L_vals = np.asarray(curve["L"], dtype=float)
        I_vals = np.asarray(curve["I"], dtype=float)
        m = h * h + h * k + k * k
        if m not in rods:
            rods[m] = {"L": L_vals.copy(), "I": I_vals.copy(), "hk": (h, k)}
            continue

        entry = rods[m]
        if entry["L"].shape != L_vals.shape or not np.allclose(entry["L"], L_vals):
            union_L = np.union1d(entry["L"], L_vals)
            entry_I = np.interp(union_L, entry["L"], entry["I"], left=0.0, right=0.0)
            add_I = np.interp(union_L, L_vals, I_vals, left=0.0, right=0.0)
            entry["L"] = union_L
            entry["I"] = entry_I + add_I
        else:
            entry["I"] += I_vals

    return rods


def qr_dict_to_arrays(qr_dict):
    """Convert a ``qr_dict`` from :func:`ht_dict_to_qr_dict` into arrays."""
    import numpy as np

    total = sum(len(v["L"]) for v in qr_dict.values())
    miller = np.empty((total, 3), dtype=np.float64)
    intens = np.empty(total, dtype=np.float64)
    degeneracy = np.ones(total, dtype=np.int32)
    details = []

    idx = 0
    for m, data in sorted(qr_dict.items()):
        h, k = data["hk"]
        L_vals = data["L"]
        I_vals = data["I"]
        n = len(L_vals)

        miller[idx:idx+n, 0] = h
        miller[idx:idx+n, 1] = k
        miller[idx:idx+n, 2] = L_vals
        intens[idx:idx+n] = I_vals

        for L_val, inten in zip(L_vals, I_vals):
            details.append([((h, k, float(L_val)), float(inten))])

        idx += n

    return miller, intens, degeneracy, details

