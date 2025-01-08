import numpy as np
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator   
import json
from ra_sim.utils.calculations import d_spacing, two_theta
from ra_sim.StructureFactor.StructureFactor  import calculate_structure_factor

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

def detect_blobs(source,notblob=None, isblob = None, labels = None, min_sigma=10, max_sigma=20, num_sigma=5, threshold=0.1, rotate_times=3, plot=False):
    """
    Detect blobs from either a filename (.asc) or a numpy array.
    
    If `source` is a string that ends with '.asc', it will be treated as a file path.
    If `source` is a numpy array, it will be treated as image data directly.

    Args:
        source (str or np.ndarray): 
            - If str and ends with .asc: path to an .asc file containing pixel data.
            - If np.ndarray: a 2D array representing the image.
        min_sigma (float): Minimum sigma for Gaussian kernel for blob detection.
        max_sigma (float): Maximum sigma for Gaussian kernel for blob detection.
        num_sigma (int): Number of intermediate sigma values.
        threshold (float): Lower bound for scale space maxima.
        rotate_times (int): Number of times to rotate the image by 90 degrees counterclockwise.
        plot (bool): If True, displays the detected blobs.

    Returns:
        np.ndarray: An array of detected blobs with shape (n_blobs, 3), each row:
                    (y, x, sigma).
    """

    if isinstance(source, str) and source.lower().endswith('.asc'):
        with open(source, 'r') as file:
            lines = file.readlines()

        # Lines after the 6th line contain pixel values
        pixel_lines = lines[6:]
        # Convert each line to a list of integers
        pixels = [list(map(int, line.split())) for line in pixel_lines]
        flattened_pixels = np.array(pixels).flatten()
        background_image = flattened_pixels.reshape((3000, 3000))
        image = np.array(background_image, dtype=np.int32)

    else:
        # its a .jpg so import it 
        image = Image.open(source)
    # Rotate the image rotate_times * 90 degrees
    # Note: np.rot90 rotates counterclockwise``, if you need clockwise rotation
    # use rotate_times = -<number> or adjust accordingly.
    image = np.rot90(image, rotate_times)
    # take the log of all the pixel values
    image = np.log(image)
    # If the image is 2D, it's already grayscale. If it is somehow 3D (RGB),
    # convert to grayscale. Usually, .asc will produce a 2D array.
    if image.ndim == 3:
        processed_image = color.rgb2gray(image)
    else:
        processed_image = image

    # Perform blob detection on the processed_image
    blobs = feature.blob_log(
        processed_image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold
    )

    # If notblob is not None, remove the unwanted blobs
    if notblob is not None:

        blobs = np.delete(blobs, notblob, axis=0)
        
   # If isblob is provided, keep only those indices
    elif isblob is not None:
        # Make sure isblob is an integer array/list
        isblob = np.array(isblob, dtype=int)
        blobs = blobs[isblob]

    if labels is not None and isblob is not None:
        # Now associate each detected blob with its corresponding label
        labeled_blobs = []
        for blob, label in zip(blobs, labels):
            y, x, sigma = blob
            labeled_blobs.append({
                'label': label,
                'x': x,
                'y': y,
                'sigma': sigma
            })
    else: 
        labeled_blobs = []
        for i, blob in enumerate(blobs):
            y, x, sigma = blob
            labeled_blobs.append({
                'label': i,
                'x': x,
                'y': y,
                'sigma': sigma
            })

    # If plot is True, visualize the blobs
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(processed_image, cmap='gray')
        for i, blob in enumerate(blobs):
            y, x, sigma = blob
            r = sigma * math.sqrt(2)
            c = plt.Circle((x, y), r, color='red', linewidth=0.5, fill=False)
            ax.add_patch(c)
            if labels is not None:
                ax.text(x + 5, y + 5, f"{labels[i]}", color='yellow', fontsize=8)
            else:   
                ax.text(x + 5, y + 5, f"{i}", color='yellow', fontsize=8)

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
