"""Diffraction and CIF helpers extracted from ``ra_sim.utils.tools``."""

from __future__ import annotations

import io as pyio
import json
import math
import re
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
except Exception:  # pragma: no cover - optional dependency may be absent
    class AzimuthalIntegrator:  # minimal stub for type hints
        pass

from ra_sim.utils.calculations import d_spacing, two_theta


DEFAULT_PIXEL_SIZE_M = 100e-6


def detector_two_theta_max(
    image_size: int,
    center,
    detector_distance: float,
    pixel_size: float = DEFAULT_PIXEL_SIZE_M,
) -> float:
    """Estimate the largest 2θ captured by the detector plane."""

    if image_size is None or image_size <= 0:
        return 180.0
    if not math.isfinite(detector_distance) or detector_distance <= 0:
        return 180.0
    if not math.isfinite(pixel_size) or pixel_size <= 0:
        pixel_size = DEFAULT_PIXEL_SIZE_M

    try:
        centre_row = float(center[0])
        centre_col = float(center[1])
    except (TypeError, ValueError, IndexError):
        centre_row = (image_size - 1) / 2.0
        centre_col = (image_size - 1) / 2.0

    if not math.isfinite(centre_row) or not math.isfinite(centre_col):
        centre_row = (image_size - 1) / 2.0
        centre_col = (image_size - 1) / 2.0

    rows = (0.0, image_size - 1.0)
    cols = (0.0, image_size - 1.0)
    max_radius = 0.0
    for row in rows:
        for col in cols:
            dx = (col - centre_col) * pixel_size
            dy = (centre_row - row) * pixel_size
            radius = math.hypot(dx, dy)
            if radius > max_radius:
                max_radius = radius

    return math.degrees(math.atan2(max_radius, detector_distance))


# Cache CIF objects and temporary files so repeated updates only modify
# in-memory data instead of reading/writing new files each time. Keys are
# absolute CIF paths.
_CIF_CACHE: dict[str, tuple] = {}
_CIF_ORIG_OCC: dict[str, list[str]] = {}
_TMP_CIF_PATH: dict[str, str] = {}


def _prepare_temp_cif(cif_path: str, occ) -> str:
    """Return path to a temporary CIF with updated occupancies."""
    import os
    import tempfile

    import CifFile

    abs_path = os.path.abspath(cif_path)
    if abs_path not in _CIF_CACHE:
        with redirect_stdout(pyio.StringIO()):
            cf = CifFile.ReadCif(abs_path)
        block_name = list(cf.keys())[0]
        block = cf[block_name]
        occ_field = block.get("_atom_site_occupancy")
        if occ_field is None:
            labels = block.get("_atom_site_label")
            occ_field = ["1.0"] * (len(labels) if isinstance(labels, list) else 1)
            block["_atom_site_occupancy"] = occ_field
        _CIF_CACHE[abs_path] = (cf, block_name)
        _CIF_ORIG_OCC[abs_path] = [str(v) for v in occ_field]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".cif")
        tmp.close()
        _TMP_CIF_PATH[abs_path] = tmp.name
    cf, block_name = _CIF_CACHE[abs_path]
    occ_field = cf[block_name]["_atom_site_occupancy"]
    orig_vals = _CIF_ORIG_OCC[abs_path]
    for i, val in enumerate(orig_vals):
        occ_field[i] = val

    def _parse_cif_float(raw):
        txt = str(raw).strip()
        try:
            return float(txt)
        except (TypeError, ValueError):
            match = re.match(r"[-+0-9.eE]+", txt)
            if match is None:
                return 1.0
            return float(match.group(0))

    n_sites = len(occ_field)
    if isinstance(occ, (list, tuple, np.ndarray)):
        factors = [float(v) for v in occ]
        if not factors:
            factors = [1.0]
        if len(factors) < n_sites:
            factors.extend([factors[-1]] * (n_sites - len(factors)))
        else:
            factors = factors[:n_sites]
    else:
        factors = [float(occ)] * n_sites

    for i in range(n_sites):
        occ_field[i] = str(_parse_cif_float(orig_vals[i]) * factors[i])

    tmp_path = _TMP_CIF_PATH[abs_path]
    try:
        with redirect_stdout(pyio.StringIO()):
            CifFile.WriteCif(cf, tmp_path)
    except AttributeError:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            with redirect_stdout(pyio.StringIO()):
                handle.write(cf.WriteOut())
    return tmp_path


def setup_azimuthal_integrator(parameters):
    """Build a pyFAI azimuthal integrator from detector parameters."""

    detector_config = json.loads(parameters["Detector_config"])
    pixel1 = float(detector_config["pixel1"])
    pixel2 = float(detector_config["pixel2"])

    return AzimuthalIntegrator(
        dist=parameters["Distance"],
        poni1=parameters["Poni1"],
        poni2=parameters["Poni2"],
        pixel1=pixel1,
        pixel2=pixel2,
        rot1=-parameters["Rot1"] * 180 / np.pi,
        rot2=-parameters["Rot2"] * 180 / np.pi,
        rot3=parameters["Rot3"],
        wavelength=parameters["Wavelength"],
    )


def miller_generator(
    mx,
    cif_file,
    occ,
    lambda_,
    energy=8.047,
    intensity_threshold=1.0,
    two_theta_range=(0, 70),
):
    """Generate filtered Miller indices and normalized intensities."""
    import Dans_Diffraction as dif

    raw_miller = [
        (h, k, l)
        for h in range(-mx + 1, mx)
        for k in range(-mx + 1, mx)
        for l in range(1, mx)
    ]

    tmp_cif = _prepare_temp_cif(cif_file, occ)
    xtl = dif.Crystal(tmp_cif)
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    xtl.Scatter.setup_scatter(scattering_type="xray", energy_kev=energy)
    xtl.Scatter.integer_hkl = True

    kept = []
    for h, k, l in raw_miller:
        d = d_spacing(h, k, l, xtl.Cell.a, xtl.Cell.c)
        tth = two_theta(d, lambda_)
        if tth is None or not (two_theta_range[0] <= tth <= two_theta_range[1]):
            continue
        intensity_val = xtl.Scatter.intensity([h, k, l])
        try:
            intensity_val = float(
                np.asarray(intensity_val, dtype=np.float64).reshape(-1)[0]
            )
        except Exception:
            continue
        if intensity_val < intensity_threshold:
            continue
        kept.append(((h, k, l), float(intensity_val)))

    if not kept:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int32),
            [],
        )

    max_intensity = max(item[1] for item in kept)
    scale = 100.0 / max_intensity if max_intensity > 0 else 0.0

    miller_arr = np.array([item[0] for item in kept], dtype=np.int32)
    intensities_arr = np.array(
        [round(item[1] * scale, 2) for item in kept],
        dtype=np.float64,
    )
    degeneracy_arr = np.ones(len(kept), dtype=np.int32)
    normalized_details = [
        [(item[0], round(item[1] * scale, 2))]
        for item in kept
    ]
    return miller_arr, intensities_arr, degeneracy_arr, normalized_details


def inject_fractional_reflections(miller, intensities, mx, step=0.5, value=0.1):
    """Add fractional Miller indices with constant intensity."""

    offsets = np.array([-step, step])
    candidates = []
    for h, k, l in miller:
        for dl in offsets:
            nl = l + dl
            if (
                -mx + 1 <= h < mx
                and -mx + 1 <= k < mx
                and 1 <= nl < mx
                and not abs(nl - round(nl)) < 1e-8
            ):
                candidates.append((h, k, nl))

    if not candidates:
        return miller.astype(float), intensities

    uniq = np.unique(np.array(candidates, dtype=float), axis=0)
    frac_intens = np.full(len(uniq), value, dtype=float)
    miller_new = np.vstack((miller.astype(float), uniq))
    intensities_new = np.concatenate((intensities, frac_intens))
    return miller_new, intensities_new


def intensities_for_hkls(
    hkls,
    cif_file,
    occ,
    lambda_,
    energy=8.047,
    intensity_threshold=0,
    two_theta_range=None,
):
    """Return scattering intensities for specific Miller indices."""
    import Dans_Diffraction as dif

    tmp_cif = _prepare_temp_cif(cif_file, occ)
    xtl = dif.Crystal(tmp_cif)
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    xtl.Scatter.setup_scatter(scattering_type="xray", energy_kev=energy)
    xtl.Scatter.integer_hkl = True

    intensities = []
    for h, k, l in hkls:
        d = d_spacing(h, k, l, xtl.Cell.a, xtl.Cell.c)
        tth = two_theta(d, lambda_)
        if tth is None:
            intensities.append(0.0)
            continue
        if two_theta_range is not None and not (
            two_theta_range[0] <= tth <= two_theta_range[1]
        ):
            intensities.append(0.0)
            continue
        intensity_val = xtl.Scatter.intensity([h, k, l])
        try:
            intensity_val = float(
                np.asarray(intensity_val, dtype=np.float64).reshape(-1)[0]
            )
        except Exception:
            intensity_val = 0.0
        if intensity_val < intensity_threshold:
            intensity_val = 0.0
        intensities.append(intensity_val)

    return np.asarray(intensities, dtype=float)


def view_azimuthal_radial(simulated_image, center, detector_params):
    """Display the azimuthal vs radial intensity map for a simulated image."""

    _ = center
    pixel_size = detector_params["pixel_size"]
    poni1 = detector_params["poni1"]
    poni2 = detector_params["poni2"]
    dist = detector_params["dist"]
    rot1 = detector_params["rot1"]
    rot2 = detector_params["rot2"]
    rot3 = detector_params["rot3"]
    wavelength = detector_params["wavelength"]

    ai = AzimuthalIntegrator(
        dist=dist,
        poni1=poni1,
        poni2=poni2,
        pixel1=pixel_size,
        pixel2=pixel_size,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        wavelength=wavelength,
    )

    res2 = ai.integrate2d(
        simulated_image,
        npt_rad=2000,
        npt_azim=1000,
        unit="2th_deg",
    )
    intensity = res2.intensity
    radial = res2.radial
    azimuthal = res2.azimuthal

    azimuthal_adjusted = np.where(
        azimuthal < 0,
        azimuthal + 180,
        azimuthal - 180,
    )
    sort_indices = np.argsort(azimuthal_adjusted)
    azimuthal_adjusted_sorted = azimuthal_adjusted[sort_indices]
    intensity_sorted = intensity[sort_indices, :]

    mask = (azimuthal_adjusted_sorted > -90) & (azimuthal_adjusted_sorted < 90)
    azimuthal_adjusted_sorted = azimuthal_adjusted_sorted[mask]
    intensity_sorted = intensity_sorted[mask, :]

    extent = [
        radial.min(),
        radial.max(),
        azimuthal_adjusted_sorted.min(),
        azimuthal_adjusted_sorted.max(),
    ]

    plt.figure(figsize=(10, 8))
    plt.imshow(
        intensity_sorted,
        extent=extent,
        cmap="turbo",
        vmin=0,
        vmax=5e6,
        aspect="auto",
        origin="lower",
    )
    plt.title("Azimuthal vs Radial View")
    plt.xlabel("2θ (degrees)")
    plt.ylabel("Azimuthal angle φ (degrees)")
    plt.colorbar(label="Intensity")
    plt.show()


def build_intensity_dataframes(miller, intensities, degeneracy, details):
    """Return summary and details DataFrames for diffraction results."""

    df_summary = pd.DataFrame(miller, columns=["h", "k", "l"])
    df_summary["Intensity"] = intensities
    df_summary["Degeneracy"] = degeneracy
    df_summary["Details"] = [
        f"See Details Sheet - Group {i + 1}" for i in range(len(df_summary))
    ]

    details_list = []
    for i, group_details in enumerate(details):
        for hkl, indiv_intensity in group_details:
            details_list.append(
                {
                    "Group": i + 1,
                    "h": hkl[0],
                    "k": hkl[1],
                    "l": hkl[2],
                    "Individual Intensity": indiv_intensity,
                }
            )

    df_details = pd.DataFrame(details_list)
    return df_summary, df_details


__all__ = [
    "DEFAULT_PIXEL_SIZE_M",
    "build_intensity_dataframes",
    "detector_two_theta_max",
    "inject_fractional_reflections",
    "intensities_for_hkls",
    "miller_generator",
    "setup_azimuthal_integrator",
    "view_azimuthal_radial",
]
