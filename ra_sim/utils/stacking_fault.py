"""Utilities for stacking fault simulations using Hendricks–Teller models."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import CifFile
import Dans_Diffraction as dif

from .tools import d_spacing, two_theta  # re-use existing helpers
from ra_sim.path_config import get_temp_dir


def _temp_cif_with_occ(cif_path: str, occ) -> Tuple[str, callable]:
    """Return path to a temporary CIF with updated occupancies.

    Parameters
    ----------
    cif_path : str
        Path to the input CIF file.
    occ : sequence[float] or float
        Occupancy multipliers. If a sequence is supplied and has the same length
        as the number of sites, each value is multiplied elementwise.
        Otherwise a single multiplier is applied to all occupancies.
    """
    cf = CifFile.ReadCif(cif_path)
    block = cf[list(cf.keys())[0]]

    occ_field = block.get("_atom_site_occupancy")
    if occ_field is None:
        labels = block.get("_atom_site_label")
        if isinstance(labels, list):
            occ_field = ["1.0"] * len(labels)
        elif labels is not None:
            occ_field = ["1.0"]
        else:
            occ_field = ["1.0"]
        block["_atom_site_occupancy"] = occ_field

    if isinstance(occ, (list, tuple)):
        if len(occ) == len(occ_field):
            for i in range(len(occ_field)):
                try:
                    occ_field[i] = str(float(occ_field[i]) * float(occ[i]))
                except Exception:
                    occ_field[i] = str(float(occ_field[i]))
        else:
            factor = float(occ[0]) if occ else 1.0
            for i in range(len(occ_field)):
                occ_field[i] = str(float(occ_field[i]) * factor)
    else:
        factor = float(occ)
        for i in range(len(occ_field)):
            occ_field[i] = str(float(occ_field[i]) * factor)

    tmp_dir = get_temp_dir()
    tmp = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, dir=str(tmp_dir))
    tmp.close()
    try:
        CifFile.WriteCif(cf, tmp.name)
    except AttributeError:
        with open(tmp.name, "w") as f:
            f.write(cf.WriteOut())

    def _cleanup() -> None:
        try:
            Path(tmp.name).unlink()
        except FileNotFoundError:
            pass

    return tmp.name, _cleanup


def ht_Iinf_dict(
    *,
    cif_path: str,
    occ,
    h_range: Tuple[int, int] = (-5, 5),
    k_range: Tuple[int, int] = (-5, 5),
    L_step: float = 0.1,
    L_max: float = 5.0,
    energy: float = 8.047,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Return Hendricks–Teller intensities on a dense L grid.

    The returned dictionary maps ``(h, k)`` pairs to intensity arrays evaluated
    on ``np.arange(0, L_max + L_step/2, L_step)``.  The grid itself is available
    under the ``"L"`` key.
    """
    tmp_cif, _cleanup = _temp_cif_with_occ(cif_path, occ)
    try:
        xtl = dif.Crystal(tmp_cif)
        xtl.Symmetry.generate_matrices()
        xtl.generate_structure()
        xtl.Scatter.setup_scatter(scattering_type="xray", energy_kev=energy)
        xtl.Scatter.integer_hkl = False

        L_vals = np.arange(0.0, L_max + L_step / 2.0, L_step)
        result: Dict[Tuple[int, int], np.ndarray] = {"L": L_vals}
        for h in range(h_range[0], h_range[1] + 1):
            for k in range(k_range[0], k_range[1] + 1):
                intens = []
                for L in L_vals:
                    val = xtl.Scatter.intensity([h, k, L])
                    try:
                        val = float(val)
                    except Exception:
                        val = 0.0
                    intens.append(val)
                result[(h, k)] = np.asarray(intens, dtype=float)
    finally:
        _cleanup()

    return result
