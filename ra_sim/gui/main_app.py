"""Light-weight GUI components built with Tkinter."""

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import threading

# Import from ra_sim package modules
from ra_sim.gui.plotting import setup_figure
from ra_sim.gui.sliders import create_slider
from ra_sim.io.file_parsing import parse_poni_file
from ra_sim.io.data_loading import load_background_image
from ra_sim.simulation.geometry import setup_azimuthal_integrator
from ra_sim.simulation.diffraction import simulate_diffraction_pattern
from ra_sim.simulation.mosaic_profiles import sample_pseudo_voigt_2d
from ra_sim.utils.constants import av, cv, q_c
from ra_sim.path_config import get_path

def main():
    root = tk.Tk()
    root.title("XRD Analysis")

    # Load parameters
    poni_path = get_path("gui_geometry_poni")
    params = parse_poni_file(poni_path)
    ai = setup_azimuthal_integrator(params)

    # Load a background image
    bg_image_path = get_path("gui_background_image")
    bg_image = load_background_image(bg_image_path)

    # Setup figure
    fig, ax = setup_figure()
    img = ax.imshow(bg_image, cmap='turbo', vmin=0, vmax=1e3)

    # Status indicator square
    status_canvas = tk.Canvas(root, width=40, height=40, highlightthickness=0)
    status_rect = status_canvas.create_rectangle(0, 0, 40, 40, fill="green", outline="")
    status_canvas.pack(side=tk.RIGHT, padx=10, pady=10)

    # Add sliders
    slider_frame = ttk.Frame(root)
    slider_frame.pack(side=tk.LEFT, fill=tk.Y)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()

    # Loading indicator shown during long computations
    loading_label = ttk.Label(root, text="Loading...", font=("Helvetica", 12))

    # Initialize variables for sliders
    theta_initial_var, _ = create_slider(
        "Theta Initial", 5.0, 20.0, 6.0, 0.01, parent=slider_frame
    )
    gamma_var, _ = create_slider(
        "Gamma", -5, 5, ai.rot2, 0.001, parent=slider_frame
    )
    Gamma_var, _ = create_slider(
        "Detector Rotation Gamma", -5, 5, ai.rot1, 0.001, parent=slider_frame
    )
    chi_var, _ = create_slider(
        "Chi", -1, 1, 0.0, 0.001, parent=slider_frame
    )
    zs_var, _ = create_slider(
        "Zs", 0.0, 5e-3, 0.0, 0.0001, parent=slider_frame
    )
    zb_var, _ = create_slider(
        "Zb", 0.0, 5e-3, -0.1e-3, 0.0001, parent=slider_frame
    )
    eta_var, _ = create_slider(
        "Eta (fraction)", 0.0, 1.0, 0.08, 0.001, parent=slider_frame
    )
    sigma_mosaic_var, _ = create_slider(
        "Sigma Mosaic (deg)", 0.0, 5.0, 3.0, 0.01, parent=slider_frame
    )
    gamma_mosaic_var, _ = create_slider(
        "Gamma Mosaic (deg)", 0.0, 5.0, 0.7, 0.01, parent=slider_frame
    )

    # Flags to manage asynchronous updates
    is_computing = False
    pending_update = False

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

    def update_plot():
        """Start an asynchronous update of the plot."""
        nonlocal is_computing, pending_update
        if is_computing:
            pending_update = True
            return

        def worker(theta_initial, gamma, Gamma, chi, zs, zb):
            beam_arrays, mosaic_arrays, divergence_arrays = generate_random_profiles()
            result = simulate_diffraction_pattern(
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
                geometry_params=(ai.dist, gamma, Gamma, chi, 0, zs, zb, 1),
            )

            def finish(image):
                nonlocal is_computing, pending_update
                ax.clear()
                ax.imshow(image, cmap='turbo', vmin=0, vmax=1e5)
                canvas.draw_idle()
                loading_label.pack_forget()
                status_canvas.itemconfig(status_rect, fill="green")
                root.update_idletasks()
                is_computing = False
                if pending_update:
                    pending_update = False
                    update_plot()

            root.after(0, lambda: finish(result))

        is_computing = True
        loading_label.pack(side=tk.BOTTOM, pady=5)
        status_canvas.itemconfig(status_rect, fill="red")
        root.update_idletasks()

        theta_initial = theta_initial_var.get()
        gamma = gamma_var.get()
        Gamma = Gamma_var.get()
        chi = chi_var.get()
        zs = zs_var.get()
        zb = zb_var.get()

        threading.Thread(
            target=worker,
            args=(theta_initial, gamma, Gamma, chi, zs, zb),
            daemon=True,
        ).start()

    # Throttle updates to avoid excessive computation when the user is
    # dragging sliders. Each change schedules an update after a short
    # delay and cancels any previously scheduled update.
    update_job = None

    def schedule_update(*args):
        nonlocal update_job
        if update_job is not None:
            root.after_cancel(update_job)
        update_job = root.after(200, update_plot)

    # Link sliders to the throttled update function
    for var in [theta_initial_var, gamma_var, Gamma_var, chi_var,
                zs_var, zb_var, eta_var, sigma_mosaic_var, gamma_mosaic_var]:
        var.trace_add("write", schedule_update)

    # Launch the main GUI loop
    root.mainloop()

if __name__ == "__main__":
    main()
