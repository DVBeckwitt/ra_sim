# ────────────────────────── global constants (unchanged) ──────────────────────────
import numpy as np
A_HEX   = 4.557
P_CLAMP = 1e-6
N_P, A_C = 3, 17.98e-10
AREA    = (2*np.pi)**2 / A_C * N_P
_FALLBACK_Z = {"Pb": 82, "I": 53}
_SITES  = [(0.0, 0.0,  0.000, "Pb"),
           (1/3, 2/3,  0.25 , "I"),
           (2/3, 1/3, -0.25 , "I")]

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
    import re
    pat = re.compile(r"_cell_length_c\s+([\d.]+)")
    with open(cif_path, "r", encoding="utf-8", errors="ignore") as fp:
        for ln in fp:
            m = pat.match(ln)
            if m:
                return float(m.group(1))
    raise ValueError(f"_cell_length_c not found in {cif_path}")

def _f0(symbol: str, q):               # Z-approx
    import numpy as np
    return np.full_like(q, _FALLBACK_Z.get(symbol, 0.), dtype=float)

def _Qmag(h, k, L, c_2h):
    import numpy as np
    inv_d2 = (4/3)*(h*h+k*k+h*k)/A_HEX**2 + (L**2)/c_2h**2
    return 2*np.pi*np.sqrt(inv_d2)

def _F2(h, k, L, c_2h):
    import numpy as np
    Q  = _Qmag(h, k, L, c_2h)
    ph = [np.exp(2j*np.pi*(h*x + k*y + L*z)) for x, y, z, _ in _SITES]
    cf = [_f0(sym, Q) for *_, sym in _SITES]
    return np.abs(sum(c*p for c, p in zip(cf, ph)))**2

def _abc(p, h, k):
    import numpy as np
    δ = 2*np.pi*((2*h + k)/3)
    z = (1-p) + p*np.exp(1j*δ)
    f = np.minimum(np.abs(z), 1-P_CLAMP)
    ψ = np.angle(z)
    return f, ψ, δ

def _I_inf(L, p, h, k, F2):
    import numpy as np
    f, ψ, δ = _abc(p, h, k)
    φ = δ + 2*np.pi*L
    return AREA * F2 * (1-f**2) / (1 + f**2 - 2*f*np.cos(φ-ψ))


# ───────────────────────── revitalised public routine ─────────────────────────
def ht_Iinf_dict(
        cif_path: str,
        hk_list=None,                  # explicit list or None
        mx: int|None = None,           # generate –mx+1…mx-1 if hk_list is None
        occ=1.0,                       # occupancy scaling, same rules as miller_generator
        p: float = 0.1,
        L_step: float = 0.01,
        L_max: float = 10.0,
    ):
    """
    Infinite-domain Hendricks–Teller intensities, now with:
      • on-the-fly occupancy scaling (scalar / list)
      • optional automatic HK grid via `mx` (drop h=k=0)
    Returns { (h,k): {'L':…, 'I':…} }
    """
    import numpy as np, itertools

    if hk_list is None:
        if mx is None:
            raise ValueError("Specify hk_list or mx")
        hk_list = [(h, k) for h, k
                   in itertools.product(range(-mx+1, mx), repeat=2)
                   if not (h == 0 and k == 0)]

    # create occupancy-modified CIF
    tmp_cif, _cleanup = _temp_cif_with_occ(cif_path, occ)
    try:
        c_2h   = _cell_c_from_cif(tmp_cif)
        L_vals = np.arange(0.0, L_max + L_step/2, L_step)

        out = {}
        for h, k in hk_list:
            F2 = _F2(h, k, L_vals, c_2h)
            I  = _I_inf(L_vals, p, h, k, F2)
            out[(h, k)] = {"L": L_vals.copy(), "I": I}
        return out
    finally:
        _cleanup(tmp_cif)          # remove temp file


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

    miller_list = []
    intens_list = []
    degeneracy_list = []
    details = []

    for (h, k), curve in ht_curves.items():
        # Preserve the ordering of L values exactly as generated in ``ht_Iinf_dict``
        # so that callers obtain the same sequence as the previous inline loop
        # implementation.

        for L_val, inten in zip(curve["L"], curve["I"]):
            miller_list.append((h, k, float(L_val)))
            intens_list.append(float(inten))
            degeneracy_list.append(1)
            details.append([((h, k, float(L_val)), float(inten))])

    if miller_list:
        miller = np.asarray(miller_list, dtype=np.float64)

        intens = np.asarray(intens_list, dtype=np.float64)
        degeneracy = np.asarray(degeneracy_list, dtype=np.int32)
    else:
        miller = np.empty((0, 3), dtype=float)
        intens = np.empty((0,), dtype=np.float64)
        degeneracy = np.empty((0,), dtype=np.int32)

    return miller, intens, degeneracy, details

