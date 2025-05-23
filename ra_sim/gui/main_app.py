import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Import from ra_sim package modules
from ra_sim.gui.plotting import setup_figure
from ra_sim.gui.sliders import create_slider
from ra_sim.io.file_parsing import parse_poni_file
from ra_sim.io.data_loading import load_background_image
from ra_sim.simulation.geometry import setup_azimuthal_integrator
from ra_sim.simulation.diffraction import simulate_diffraction_pattern
from ra_sim.simulation.mosaic_profiles import sample_pseudo_voigt_2d
from ra_sim.utils.constants import av, cv

def main():
    root = tk.Tk()
    root.title("XRD Analysis")

    # Load parameters
    poni_path = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Analysis\Bi2Se3\geometry.poni"
    params = parse_poni_file(poni_path)
    ai = setup_azimuthal_integrator(params)

    # Load a background image
    bg_image_path = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Analysis\Bi2Se3\In-Plane\3\Bi2Se3_6d_5m.asc"
    bg_image = load_background_image(bg_image_path)

    # Setup figure
    fig, ax = setup_figure()
    ax.imshow(bg_image, cmap='turbo', vmin=0, vmax=1e3)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()

    # Add sliders
    slider_frame = ttk.Frame(root)
    slider_frame.pack(side=tk.LEFT, fill=tk.Y)

    # Initialize variables for sliders
    theta_initial_var, _ = create_slider(slider_frame, "Theta Initial", 5.0, 20.0, 6.0, 0.01)
    gamma_var, _ = create_slider(slider_frame, "Gamma", -5, 5, ai.rot2, 0.001)
    Gamma_var, _ = create_slider(slider_frame, "Detector Rotation Gamma", -5, 5, ai.rot1, 0.001)
    chi_var, _ = create_slider(slider_frame, "Chi", -1, 1, 0.0, 0.001)
    zs_var, _ = create_slider(slider_frame, "Zs", 0.0, 5e-3, 0.0, 0.0001)
    zb_var, _ = create_slider(slider_frame, "Zb", 0.0, 5e-3, -0.1e-3, 0.0001)
    eta_var, _ = create_slider(slider_frame, "Eta (fraction)", 0.0, 1.0, 0.08, 0.001)
    sigma_mosaic_var, _ = create_slider(slider_frame, "Sigma Mosaic (deg)", 0.0, 5.0, 3.0, 0.01)
    gamma_mosaic_var, _ = create_slider(slider_frame, "Gamma Mosaic (deg)", 0.0, 5.0, 0.7, 0.01)

    # Generate random profiles
    def generate_random_profiles():
        num_samples = 1000
        eta = eta_var.get()
        sigma_mosaic = np.radians(sigma_mosaic_var.get())
        gamma_mosaic = np.radians(gamma_mosaic_var.get())

        # Mosaic profile
        mosaic_profiles = sample_pseudo_voigt_2d(num_samples, eta, sigma_mosaic, gamma_mosaic)
        beta_array = mosaic_profiles[:, 0]
        kappa_array = mosaic_profiles[:, 1]

        # Divergence profile
        divergence_profiles = sample_pseudo_voigt_2d(num_samples, 0.0, np.radians(0.1), np.radians(0.0))
        theta_array = divergence_profiles[:, 0]
        phi_array = divergence_profiles[:, 1]

        # Beam profile
        beam_profiles = sample_pseudo_voigt_2d(num_samples, 0.0, 0.25e-3, 0.0)
        beam_x_array = beam_profiles[:, 0]
        beam_y_array = beam_profiles[:, 1]

        return (beam_x_array, beam_y_array, np.ones(num_samples)), \
               (beta_array, kappa_array, np.ones(num_samples)), \
               (theta_array, phi_array, np.ones(num_samples))

    def update_plot(val=None):
        """
        Update the plot dynamically based on slider values.
        """
        # Retrieve slider values
        theta_initial = theta_initial_var.get()
        gamma = gamma_var.get()
        Gamma = Gamma_var.get()
        chi = chi_var.get()
        zs = zs_var.get()
        zb = zb_var.get()

        # Generate random profiles
        beam_arrays, mosaic_arrays, divergence_arrays = generate_random_profiles()

        # Simulate diffraction pattern
        simulated_image = simulate_diffraction_pattern(
            miller=[(0, 0, 3), (0, 0, 6)],  # Replace with actual data
            intensities=[100, 200],         # Replace with actual data
            parameters=params,
            beam_arrays=beam_arrays,
            mosaic_arrays=mosaic_arrays,
            divergence_arrays=divergence_arrays,
            av=av,
            cv=cv,
            lambda_=ai.wavelength * 1e10,
            center=[3000 - ai.poni2 * 1e4, 3000 - ai.poni1 * 1e4],
            debye_x=0.4,
            debye_y=0.5,
            theta_initial=theta_initial,
            theta_range=0.1,
            step=0.1,
            geometry_params=(
                ai.dist, gamma, Gamma, chi, 0, zs, zb, 1
            )
        )

        # Update the plot
        ax.clear()
        ax.imshow(simulated_image, cmap='turbo', vmin=0, vmax=1e5)
        canvas.draw_idle()

    # Link sliders to the update function
    for var in [theta_initial_var, gamma_var, Gamma_var, chi_var, zs_var, zb_var, eta_var, sigma_mosaic_var, gamma_mosaic_var]:
        var.trace_add("write", update_plot)

    # Launch the main GUI loop
    root.mainloop()

if __name__ == "__main__":
    main()
