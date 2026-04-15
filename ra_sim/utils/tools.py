"""Blob detection helpers plus compatibility re-exports."""

from __future__ import annotations

import math

import numpy as np

from ra_sim.utils.diffraction_tools import (
    DEFAULT_PIXEL_SIZE_M,
    build_intensity_dataframes,
    detector_two_theta_max,
    inject_fractional_reflections,
    intensities_for_hkls,
    miller_generator,
    setup_azimuthal_integrator,
)


def detect_blobs(
    source,
    notblob=None,
    groups=None,
    min_sigma=10,
    max_sigma=20,
    num_sigma=5,
    threshold=0.1,
    rotate_times=3,
    plot=False,
):
    """Detect blobs from either a filename (.asc) or a numpy array."""

    from PIL import Image
    from skimage import color, exposure, feature

    isblob, labels = None, None
    if groups:
        isblob, labels = [], []
        for blob_indices, label_str in groups:
            for idx in blob_indices:
                isblob.append(idx)
                labels.append(label_str)

    if isinstance(source, str) and source.lower().endswith(".asc"):
        with open(source, encoding="utf-8") as file:
            lines = file.readlines()
        pixel_lines = lines[6:]
        pixels = [list(map(int, line.split())) for line in pixel_lines]
        flattened_pixels = np.array(pixels).flatten()
        background_image = flattened_pixels.reshape((3000, 3000))
        image = np.array(background_image, dtype=np.int32)
    else:
        image = Image.open(source)
        image = np.array(image)

    image = np.rot90(image, k=rotate_times)
    image[image < 1] = 1
    image = np.log(image)

    if image.ndim == 3:
        processed_image = color.rgb2gray(image)
    else:
        processed_image = image

    processed_image = exposure.rescale_intensity(
        processed_image,
        in_range="image",
        out_range="dtype",
    )

    blobs = feature.blob_log(
        processed_image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
    )

    if notblob is not None:
        blobs = np.delete(blobs, notblob, axis=0)
    elif isblob is not None:
        isblob = np.array(isblob, dtype=int)
        blobs = blobs[isblob]

    blob_conf_list = []
    for y, x, sigma in blobs:
        yy = int(round(y))
        xx = int(round(x))
        if 0 <= yy < processed_image.shape[0] and 0 <= xx < processed_image.shape[1]:
            confidence = processed_image[yy, xx]
        else:
            confidence = 0
        blob_conf_list.append([y, x, sigma, confidence])
    blob_conf_array = np.array(blob_conf_list)

    max_results = 30
    if len(blob_conf_array) > max_results:
        sorted_indices = np.argsort(blob_conf_array[:, 3])[::-1]
        blob_conf_array = blob_conf_array[sorted_indices[:max_results]]

    labeled_blobs = []
    for i, row in enumerate(blob_conf_array):
        y, x, sigma, confidence = row
        if labels is not None and i < len(labels):
            label_str = labels[i]
        else:
            label_str = i

        labeled_blobs.append(
            {
                "label": label_str,
                "x": x,
                "y": y,
                "sigma": sigma,
                "confidence": confidence,
            }
        )

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(processed_image, cmap="gray")

        for blob_data in labeled_blobs:
            y = blob_data["y"]
            x = blob_data["x"]
            sigma = blob_data["sigma"]
            label_str = blob_data["label"]

            radius = sigma * math.sqrt(2)
            circ = plt.Circle((x, y), radius, color="red", linewidth=0.5, fill=False)
            ax.add_patch(circ)
            ax.text(
                x + 5,
                y + 5,
                str(label_str),
                color="yellow",
                fontsize=8,
            )

        plt.show()

    return labeled_blobs


def view_azimuthal_radial(*args, **kwargs):
    from ra_sim.utils.diffraction_tools import (
        view_azimuthal_radial as _view_azimuthal_radial,
    )

    return _view_azimuthal_radial(*args, **kwargs)


__all__ = [
    "DEFAULT_PIXEL_SIZE_M",
    "build_intensity_dataframes",
    "detect_blobs",
    "detector_two_theta_max",
    "inject_fractional_reflections",
    "intensities_for_hkls",
    "miller_generator",
    "setup_azimuthal_integrator",
    "view_azimuthal_radial",
]
