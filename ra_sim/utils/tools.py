import numpy as np
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator   
import json
from ra_sim.utils.calculations import d_spacing, two_theta
from ra_sim.StructureFactor.StructureFactor  import calculate_structure_factor
from skimage import exposure

import numpy as np
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, color
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, color
from PIL import Image

import numpy as np
import math
from skimage import feature, color
import matplotlib.pyplot as plt

import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import feature, color

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


def miller_generator(max_miller, av, cv, lambda_, atoms, data, occ):
    # Generate Miller Indices
    Raw_Miller = [p for p in itertools.product(range(max_miller + 1), repeat=3)]  # Now includes 0 to mx, inclusive
    # Filter Miller indices based on 2theta condition and structure factor intensity
    miller = []
    intensities = []
    for h, k, l in Raw_Miller:
        D = d_spacing(h, k, l, av, cv)
        two_theta_value = two_theta(D, lambda_)
        if two_theta_value is not None and 0 <= two_theta_value <= 70:
            intensity = calculate_structure_factor(h, k, l, atoms, data, occ)
            if intensity > 1:  # Arbitrary threshold for small intensity
                miller.append((h, k, l))
                intensities.append(intensity)
    miller = np.array(miller, dtype = np.int32)
    intensities = np.array(intensities, dtype = np.float64)

    return miller, intensities

import matplotlib.pyplot as plt
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import numpy as np

def view_azimuthal_radial(simulated_image, center, detector_params):
    """
    Displays the azimuthal vs radial intensity map based on a simulated image.

    Args:
        simulated_image (numpy.ndarray): The 2D diffraction pattern image.
        center (tuple): The beam center in the format (center_x, center_y).
        detector_params (dict): Contains detector geometry and wavelength.
    """
    # Retrieve detector parameters
    pixel_size = detector_params['pixel_size']
    poni1 = detector_params['poni1']
    poni2 = detector_params['poni2']
    dist = detector_params['dist']
    rot1 = detector_params['rot1']
    rot2 = detector_params['rot2']
    rot3 = detector_params['rot3']
    wavelength = detector_params['wavelength']

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
