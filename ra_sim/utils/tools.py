"""Utility helpers for blob detection and analysis."""

import itertools
import json
import math
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import pyFAI
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
except Exception:  # pragma: no cover - optional dependency may be absent
    pyFAI = None
    class AzimuthalIntegrator:  # minimal stub for type hints
        pass
from skimage import color, exposure, feature, io

from ra_sim.StructureFactor.StructureFactor import calculate_structure_factor
from ra_sim.utils.calculations import d_spacing, two_theta
from ra_sim.path_config import get_temp_dir

# Cache CIF objects and temporary files so repeated updates only modify
# in-memory data instead of reading/writing new files each time.  Keys are
# absolute CIF paths.
_CIF_CACHE: dict[str, tuple] = {}
_CIF_ORIG_OCC: dict[str, list[str]] = {}
_TMP_CIF_PATH: dict[str, str] = {}


def _prepare_temp_cif(cif_path: str, occ) -> str:
    """Return path to a temporary CIF with updated occupancies."""
    import os, tempfile, CifFile

    abs_path = os.path.abspath(cif_path)
    if abs_path not in _CIF_CACHE:
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
    # reset to original before applying new factors
    orig_vals = _CIF_ORIG_OCC[abs_path]
    for i, val in enumerate(orig_vals):
        occ_field[i] = val

    if isinstance(occ, (list, tuple)):
        if len(occ) == len(occ_field):
            for i in range(len(occ_field)):
                occ_field[i] = str(float(orig_vals[i]) * float(occ[i]))
        else:
            fac = float(occ[0])
            for i in range(len(occ_field)):
                occ_field[i] = str(float(orig_vals[i]) * fac)
    else:
        fac = float(occ)
        for i in range(len(occ_field)):
            occ_field[i] = str(float(orig_vals[i]) * fac)

    tmp_path = _TMP_CIF_PATH[abs_path]
    try:
        CifFile.WriteCif(cf, tmp_path)
    except AttributeError:
        with open(tmp_path, "w") as f:
            f.write(cf.WriteOut())
    return tmp_path

def detect_blobs(
    source,
    notblob=None,
    groups=None,
    min_sigma=10,
    max_sigma=20,
    num_sigma=5,
    threshold=0.1,
    rotate_times=3,
    plot=False
):
    """
    Detect blobs from either a filename (.asc) or a numpy array.

    Args:
        source (str or np.ndarray): 
            - If str and ends with '.asc': path to an .asc file containing pixel data.
            - If np.ndarray: a 2D array representing the image.
        notblob (list[int] or None):
            Indices of blobs to remove after detection.
        groups (list[tuple[list[int], str]] or None):
            A list of (blob_indices, label_str) pairs, e.g. [([0,1], "0,1,5"), ([2,3], "1,0,10"), ...].
            If provided, only those blob indices are kept, and each index is labeled with its associated label_str.
        min_sigma (float):
            Minimum sigma for the Gaussian kernel for blob detection.
        max_sigma (float):
            Maximum sigma for the Gaussian kernel for blob detection.
        num_sigma (int):
            Number of intermediate sigma values for multiscale detection.
        threshold (float):
            Lower bound for scale-space maxima.
        rotate_times (int):
            Number of 90-degree counterclockwise rotations to apply to the image.
        plot (bool):
            If True, displays the detected blobs.

    Returns:
        list of dict:
            Each element is a dictionary with keys {'label', 'x', 'y', 'sigma', 'confidence'}.
    """
    # --------------------------------------------------------------------------
    # 1. Flatten groups into isblob, labels if provided
    # --------------------------------------------------------------------------
    isblob, labels = None, None
    if groups:
        isblob, labels = [], []
        for blob_indices, label_str in groups:
            for idx in blob_indices:
                isblob.append(idx)
                labels.append(label_str)

    # --------------------------------------------------------------------------
    # 2. Load the image (from .asc or other format)
    # --------------------------------------------------------------------------
    if isinstance(source, str) and source.lower().endswith('.asc'):
        with open(source, 'r') as file:
            lines = file.readlines()
        pixel_lines = lines[6:]  # Lines after the 6th line contain pixel data
        pixels = [list(map(int, line.split())) for line in pixel_lines]
        flattened_pixels = np.array(pixels).flatten()
        background_image = flattened_pixels.reshape((3000, 3000))
        image = np.array(background_image, dtype=np.int32)
    else:
        # Otherwise assume 'source' is a valid image path or a PIL-supported format
        image = Image.open(source)
        image = np.array(image)

    # --------------------------------------------------------------------------
    # 3. Preprocess: rotate + log transform + grayscale (if needed)
    # --------------------------------------------------------------------------
    image = np.rot90(image, k=rotate_times)  # Rotate the image
    
    # Ensure no zero or negative values before log
    image[image < 1] = 1
    image = np.log(image)  # simple log transform

    # If the image is color, convert to grayscale
    if image.ndim == 3:
        processed_image = color.rgb2gray(image)
    else:
        processed_image = image

    # Rescale to use the full data type range
    processed_image = exposure.rescale_intensity(
        processed_image, in_range='image', out_range='dtype'
    )

    # --------------------------------------------------------------------------
    # 4. Detect blobs with Laplacian of Gaussian
    # --------------------------------------------------------------------------
    blobs = feature.blob_log(
        processed_image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold
    )
    # blobs has shape (n_blobs, 3) where each row is (y, x, sigma)

    # --------------------------------------------------------------------------
    # 5. Filter out or keep only certain blobs
    # --------------------------------------------------------------------------
    if notblob is not None:
        # Remove the unwanted blob indices
        blobs = np.delete(blobs, notblob, axis=0)
    elif isblob is not None:
        # Keep only the specified blob indices
        isblob = np.array(isblob, dtype=int)
        blobs = blobs[isblob]

    # --------------------------------------------------------------------------
    # 5b. Compute "confidence" for each blob
    # --------------------------------------------------------------------------
    # We'll define "confidence" = pixel intensity at (y, x) in processed_image
    # Round the blob coordinates to nearest int in case they are floats.
    blob_conf_list = []
    for (y, x, sigma) in blobs:
        yy = int(round(y))
        xx = int(round(x))
        # Make sure we're within bounds
        if 0 <= yy < processed_image.shape[0] and 0 <= xx < processed_image.shape[1]:
            confidence = processed_image[yy, xx]
        else:
            confidence = 0
        blob_conf_list.append([y, x, sigma, confidence])
    blob_conf_array = np.array(blob_conf_list)  # shape (n, 4)

    # --------------------------------------------------------------------------
    # 5c. Sort by confidence DESCENDING, keep top N
    # --------------------------------------------------------------------------
    N = 30
    if len(blob_conf_array) > N:
        sorted_indices = np.argsort(blob_conf_array[:, 3])[::-1]  # sort by 4th col (confidence)
        top20 = sorted_indices[:N]
        blob_conf_array = blob_conf_array[top20]

    # --------------------------------------------------------------------------
    # 6. Label the detected blobs
    # --------------------------------------------------------------------------
    labeled_blobs = []
    for i, row in enumerate(blob_conf_array):
        y, x, sigma, confidence = row
        if labels is not None and i < len(labels):
            label_str = labels[i]
        else:
            label_str = i  # fallback to an integer label

        labeled_blobs.append({
            'label': label_str,
            'x': x,
            'y': y,
            'sigma': sigma,
            'confidence': confidence,
        })

    # --------------------------------------------------------------------------
    # 7. Plot the results if requested
    # --------------------------------------------------------------------------
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(processed_image, cmap='gray')

        for blob_data in labeled_blobs:
            y = blob_data['y']
            x = blob_data['x']
            sigma = blob_data['sigma']
            label_str = blob_data['label']

            r = sigma * math.sqrt(2)
            circ = plt.Circle((x, y), r, color='red', linewidth=0.5, fill=False)
            ax.add_patch(circ)

            ax.text(
                x + 5, y + 5, str(label_str), 
                color='yellow', fontsize=8
            )

        plt.show()

    return labeled_blobs


def setup_azimuthal_integrator(parameters):
    # Parse the detector configuration from the JSON-like structure
    detector_config = json.loads(parameters['Detector_config'])
    pixel1 = float(detector_config['pixel1'])
    pixel2 = float(detector_config['pixel2'])
    max_shape = list(map(int, detector_config['max_shape']))  # Convert max_shape elements to integers

    # Initialize the AzimuthalIntegrator
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(
        dist=parameters['Distance'],
        poni1=parameters['Poni1'],
        poni2=  parameters['Poni2'],
        pixel1= pixel1,
        pixel2= pixel2,
        rot1= -parameters['Rot1']* 180/np.pi,
        rot2= -parameters['Rot2']*180/np.pi,
        rot3= parameters['Rot3'],
        wavelength=parameters['Wavelength']
    )
    return ai

#!/usr/bin/env python
"""
Generate Miller indices from 0 to mx and filter them by twoθ and intensity,
using the scattering calculations from Dans_Diffraction.

For each (h, k, l) in the set defined by 0 ≤ h,k,l ≤ mx, this code:
  - Computes d-spacing and twoθ using Bragg’s law.
  - Retrieves the intensity via xtl.Scatter.intensity.
  - Keeps only those reflections that lie within the twoθ_range
    and whose intensity is above intensity_threshold.

Finally, the filtered Miller indices and intensities are printed.
  
Usage:
  python miller_generator.py
"""

import itertools
import numpy as np
import math
import Dans_Diffraction as dif


def miller_generator(mx, cif_file, occ, lambda_, energy=8.047,
                     intensity_threshold=1.0, two_theta_range=(0,70)):
    """
    Generate Miller indices, compute 2θ and intensities, group reflections,
    and normalize intensities. This version updates the occupancy values in the
    atom-site loop by writing a temporary CIF file with the new occupancies.
    
    The occupancy modification uses the given occupancy array: if occ is a list 
    (or tuple) and its length matches the number of occupancy entries in the CIF, 
    each occupancy is multiplied by the corresponding occ value; otherwise, occ[0]
    (or occ if a single number) is used uniformly.
    
    Parameters:
      mx                 : maximum index bound (h,k: -mx+1...mx-1, l: 1...mx-1)
      cif_file           : path to the CIF file
      occ                : occupancy multiplier(s); if list and its length equals that of the occupancy
                           entries in the CIF, each is applied individually, otherwise the first element is used uniformly.
      lambda_            : X-ray wavelength in Å.
      energy             : energy in keV (default 8.047)
      intensity_threshold: minimum intensity to keep a reflection.
      two_theta_range    : allowed 2θ range (degrees)
    
    Returns:
      miller:       (N, 3) array of representative Miller indices.
      intensities:  (N,) array of normalized intensities.
      degeneracy:   (N,) array of multiplicities.
      details:      list of length N; each element is a list of tuples ((h,k,l), normalized intensity)
                    for the reflections in that group.
    """
    from collections import defaultdict
    import numpy as np, math
    import Dans_Diffraction as dif

    # Generate candidate Miller indices.
    raw_miller = [
        (h, k, l)
        for h in range(-mx+1, mx)
        for k in range(-mx+1, mx)
        for l in range(1, mx)
    ]

    # --------------- prepare/update temporary CIF -----------------------
    tmp_cif = _prepare_temp_cif(cif_file, occ)

    # Load the crystal using the temporary CIF file.
    xtl = dif.Crystal(tmp_cif)
    # Optionally, remove the temporary file later if desired:
    # os.remove(tmp.name)
    # ---------------------------------------------------------------------

    #print("Space group:", xtl.Symmetry.spacegroup)
    #print("Sym ops:", xtl.Symmetry.symmetry_operations)
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    xtl.Scatter.setup_scatter(scattering_type='xray', energy_kev=energy)
    xtl.Scatter.integer_hkl = True

    groups = defaultdict(list)  # For reflections with (h,k) ≠ (0,0)
    zeros = []                  # For (0,0,l) reflections

    for (h, k, l) in raw_miller:
        d = d_spacing(h, k, l, xtl.Cell.a, xtl.Cell.c)
        tth = two_theta(d, lambda_)
        if tth is None or not (two_theta_range[0] <= tth <= two_theta_range[1]):
            continue
        intensity_val = xtl.Scatter.intensity([h, k, l])
        try:
            intensity_val = float(intensity_val)
        except Exception:
            continue
        if intensity_val < intensity_threshold:
            continue
        if h == 0 and k == 0:
            zeros.append(((h, k, l), intensity_val))
        else:
            key = (h*h + k*k, l)
            groups[key].append(((h, k, l), intensity_val))
    
    grouped_results = []
    for key, items in groups.items():
        rep_miller = items[0][0]
        multiplicity = len(items)
        total_intensity = sum(item[1] for item in items)
        details_list = [(hk, intensity) for (hk, intensity) in items]
        grouped_results.append((rep_miller, total_intensity, multiplicity, details_list))
    zeros_group = [
        (((h, k, l)), intensity, 1, [((h, k, l), intensity)])
        for ((h, k, l), intensity) in zeros
    ]
    combined = grouped_results + zeros_group
    if not combined:
        return (np.empty((0, 3), dtype=np.int32),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.int32),
                [])
    max_total_intensity = max(item[1] for item in combined)
    normalized_combined = []
    for rep_miller, total_intensity, multiplicity, details_list in combined:
        normalized_total = round(total_intensity * 100 / max_total_intensity, 2)
        normalized_details = [
            (hk, round(indiv_intensity * 100 / max_total_intensity, 2))
            for hk, indiv_intensity in details_list
        ]
        normalized_combined.append((rep_miller, normalized_total, multiplicity, normalized_details))
    
    miller_arr = np.array([item[0] for item in normalized_combined], dtype=np.int32)
    intensities_arr = np.array([item[1] for item in normalized_combined], dtype=np.float64)
    degeneracy_arr = np.array([item[2] for item in normalized_combined], dtype=np.int32)
    normalized_details = [item[3] for item in normalized_combined]
    
    return miller_arr, intensities_arr, degeneracy_arr, normalized_details


def inject_fractional_reflections(miller, intensities, mx, step=0.5, value=0.1):
    """Add fractional Miller indices with constant intensity.

    Parameters
    ----------
    miller : ndarray
        Existing ``(N,3)`` array of integer Miller indices.
    intensities : ndarray
        Intensities corresponding to ``miller``.
    mx : int
        Maximum index bound used for the original Miller generation.
    step : float, optional
        Spacing for fractional indices along ``l``. The default ``0.5``
        inserts half steps between integer peaks.

    value : float, optional
        Intensity assigned to each injected reflection. Default ``0.1``.

    Returns
    -------
    miller_new : ndarray
        Array containing both the original and fractional Miller indices.
    intensities_new : ndarray
        Intensities for ``miller_new``.
    """

    offsets = np.array([-step, step])
    candidates = []
    for h, k, l in miller:
        # Only offset the L value while keeping H and K fixed
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



def intensities_for_hkls(hkls, cif_file, occ, lambda_, energy=8.047,
                         intensity_threshold=0, two_theta_range=None):
    """Return scattering intensities for specific Miller indices.

    This helper performs the same CIF and :mod:`Dans_Diffraction` setup as
    :func:`miller_generator` but takes an explicit list of ``(h, k, l)``
    tuples.  Each reflection is evaluated individually and the raw intensity
    from ``xtl.Scatter.intensity`` is returned.

    Parameters
    ----------
    hkls : iterable of tuple[int, int, int]
        Miller indices to evaluate.
    cif_file : str
        Path to the CIF file describing the crystal structure.
    occ : sequence[float] or float
        Occupancy multipliers applied in the same manner as in
        :func:`miller_generator`.
    lambda_ : float
        Wavelength in Å used to compute 2θ for optional filtering.
    energy : float, optional
        Energy in keV passed to ``xtl.Scatter.setup_scatter``.
    intensity_threshold : float, optional
        Intensities below this value are returned as ``0``.
    two_theta_range : tuple[float, float] or None, optional
        If given, reflections whose 2θ lies outside the range are assigned
        zero intensity.

    Returns
    -------
    numpy.ndarray
        Array of intensities ordered as ``hkls``.

    Examples
    --------
    >>> hkls = [(0, 0, 1), (1, 0, 1)]
    >>> intensities_for_hkls(hkls, "tests/local_test.cif", [1.0], 1.0)
    array([...])
    """
    import numpy as np
    import Dans_Diffraction as dif

    tmp_cif = _prepare_temp_cif(cif_file, occ)

    xtl = dif.Crystal(tmp_cif)
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    xtl.Scatter.setup_scatter(scattering_type='xray', energy_kev=energy)
    xtl.Scatter.integer_hkl = True

    intensities = []
    for h, k, l in hkls:
        d = d_spacing(h, k, l, xtl.Cell.a, xtl.Cell.c)
        tth = two_theta(d, lambda_)
        if tth is None:
            intensities.append(0.0)
            continue
        if two_theta_range is not None and not (two_theta_range[0] <= tth <= two_theta_range[1]):
            intensities.append(0.0)
            continue
        intensity_val = xtl.Scatter.intensity([h, k, l])
        try:
            intensity_val = float(intensity_val)
        except Exception:
            intensity_val = 0.0
        if intensity_val < intensity_threshold:
            intensity_val = 0.0
        intensities.append(intensity_val)

    return np.asarray(intensities, dtype=float)



import matplotlib.pyplot as plt
import numpy as np

def view_azimuthal_radial(simulated_image, center, detector_params):
    """
    Displays the azimuthal vs radial intensity map based on a simulated image.

    Args:
        simulated_image (numpy.ndarray): The 2D diffraction pattern image.
        center (tuple): The beam center given as (row_pixels, col_pixels).
        detector_params (dict): Contains detector geometry and wavelength.
    """
    # Retrieve detector parameters
    pixel_size = detector_params['pixel_size']
    dist = detector_params['dist']
    rot1 = detector_params['rot1']
    rot2 = detector_params['rot2']
    rot3 = detector_params['rot3']
    wavelength = detector_params['wavelength']

    if simulated_image is None:
        raise ValueError("simulated_image must be provided for integration")

    image_rows = simulated_image.shape[0]
    center_row, center_col = center
    poni1 = (image_rows - center_row) * pixel_size
    poni2 = center_col * pixel_size

    # Set up the AzimuthalIntegrator
    ai = AzimuthalIntegrator(
        dist=dist,
        poni1=poni1,
        poni2=poni2,
        pixel1=pixel_size,
        pixel2=pixel_size,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        wavelength=wavelength
    )

    # Perform azimuthal integration
    res2 = ai.integrate2d(
        simulated_image,
        npt_rad=2000,
        npt_azim=1000,
        unit="2th_deg"
    )

    # Extract intensity, radial, and azimuthal arrays
    intensity = res2.intensity
    radial = res2.radial
    azimuthal = res2.azimuthal

    # Adjust azimuthal values and sort
    azimuthal_adjusted = np.where(azimuthal < 0, azimuthal + 180, azimuthal - 180)
    sort_indices = np.argsort(azimuthal_adjusted)
    azimuthal_adjusted_sorted = azimuthal_adjusted[sort_indices]
    intensity_sorted = intensity[sort_indices, :]

    mask = (azimuthal_adjusted_sorted > -90) & (azimuthal_adjusted_sorted < 90)
    azimuthal_adjusted_sorted = azimuthal_adjusted_sorted[mask]
    intensity_sorted = intensity_sorted[mask, :]

    extent = [radial.min(), radial.max(), azimuthal_adjusted_sorted.min(), azimuthal_adjusted_sorted.max()]

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(
        intensity_sorted,
        extent=extent,
        cmap='turbo',
        vmin=0,
        vmax=5e6,
        aspect='auto',
        origin='lower'
    )
    plt.title('Azimuthal vs Radial View')
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Azimuthal angle φ (degrees)')
    plt.colorbar(label='Intensity')
    plt.show()


def build_intensity_dataframes(miller, intensities, degeneracy, details):
    """Return summary and details DataFrames for diffraction results."""

    df_summary = pd.DataFrame(miller, columns=["h", "k", "l"])
    df_summary["Intensity"] = intensities
    df_summary["Degeneracy"] = degeneracy
    df_summary["Details"] = [
        f"See Details Sheet - Group {i+1}" for i in range(len(df_summary))
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
