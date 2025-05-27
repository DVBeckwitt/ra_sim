import numpy as np
import yaml
import os

from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import process_peaks_parallel
from ra_sim.utils.calculations import IndexofRefraction

here = os.path.dirname(__file__)


def main():
    with open(os.path.join(here, "defaults.yaml"), "r") as f:
        params = yaml.safe_load(f)

    n2 = IndexofRefraction()

    beam_x, beam_y, theta_arr, phi_arr, wavelength_array = generate_random_profiles(
        params["num_samples"],
        params["divergence_sigma"],
        params["bw_sigma"],
        params["lambda"],
        params["wavelength_bandwidth"],
    )

    image = np.zeros((params["image_size"], params["image_size"]), dtype=np.float64)

    image_out, max_positions, q_data, q_count = process_peaks_parallel(
        np.array(params["miller"], dtype=np.int64),
        np.array(params["intensities"], dtype=np.float64),
        params["image_size"],
        params["av"], params["cv"], params["lambda"], image,
        params["distance"], params["gamma_deg"], params["Gamma_deg"],
        params["chi_deg"], params["psi_deg"],
        params["zs"], params["zb"], n2,
        beam_x, beam_y,
        theta_arr, phi_arr,
        params["sigma_mosaic_deg"], params["gamma_mosaic_deg"], params["eta"],
        wavelength_array,
        params["debye_x"], params["debye_y"], np.array(params["center"], dtype=np.float64),
        params["theta_initial_deg"],
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=1,
    )

    np.save(os.path.join(here, "results.npy"), {
        "image": image_out,
        "max_positions": max_positions,
        "q_data": q_data,
        "q_count": q_count,
    }, allow_pickle=True)


if __name__ == "__main__":
    main()
