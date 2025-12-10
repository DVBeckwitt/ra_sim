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

    background_vmax_default = float(np.nanpercentile(bg_image, 99))
    if not np.isfinite(background_vmax_default) or background_vmax_default <= 0:
        background_vmax_default = 1e3

    background_min_candidate = float(np.nanpercentile(bg_image, 1))
    if not np.isfinite(background_min_candidate):
        background_min_candidate = 0.0

    background_vmin_default = 0.0

    background_slider_min_value = min(background_min_candidate, background_vmin_default)
    background_slider_max_value = max(
        background_vmax_default * 5.0, background_slider_min_value + 1.0
    )
    background_slider_step = max(
        (background_slider_max_value - background_slider_min_value) / 500.0, 0.01
    )

    bg_artist = ax.imshow(bg_image, cmap='turbo')
    bg_artist.set_clim(background_vmin_default, background_vmax_default)
    sim_artist = None

    # Status indicator square
    status_canvas = tk.Canvas(root, width=40, height=40, highlightthickness=0)
    status_rect = status_canvas.create_rectangle(0, 0, 40, 40, fill="green", outline="")
    status_canvas.pack(side=tk.RIGHT, padx=10, pady=10)

    # Add sliders
    slider_frame = ttk.Frame(root)
    slider_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()

    # Status indicator square
    status_canvas = tk.Canvas(root, width=40, height=40, highlightthickness=0)
    status_rect = status_canvas.create_rectangle(0, 0, 40, 40, fill="green", outline="")
    status_canvas.pack(side=tk.RIGHT, padx=10, pady=10)

    # Loading indicator shown during long computations
    loading_label = ttk.Label(root, text="Loading...", font=("Helvetica", 12))

    background_min_var = None
    background_max_var = None
    background_transparency_var = None
    simulation_min_var = None
    simulation_max_var = None
    scale_factor_var = None

    def update_background_norm():
        if (
            background_min_var is None
            or background_max_var is None
            or background_transparency_var is None
        ):
            return

        min_val = background_min_var.get()
        max_val = background_max_var.get()

        if min_val >= max_val:
            adjustment = max(abs(max_val) * 1e-6, 1e-6)
            background_min_var.set(max_val - adjustment)
            return

        bg_artist.set_clim(min_val, max_val)
        transparency = background_transparency_var.get()
        alpha = max(0.0, min(1.0, 1.0 - transparency))
        bg_artist.set_alpha(alpha)
        canvas.draw_idle()

    def update_simulation_norm():
        if simulation_min_var is None or simulation_max_var is None:
            return

        min_val = simulation_min_var.get()
        max_val = simulation_max_var.get()

        if min_val >= max_val:
            adjustment = max(abs(max_val) * 1e-6, 1e-6)
            simulation_min_var.set(max_val - adjustment)
            return

        if sim_artist is not None:
            sim_artist.set_clim(min_val, max_val)
            canvas.draw_idle()

    simulation_vmax_default = background_vmax_default
    simulation_vmin_default = 0.0
    simulation_slider_min_value = 0.0
    simulation_slider_max_value = max(
        background_slider_max_value, simulation_vmax_default * 10.0
    )
    simulation_slider_step = max(
        (simulation_slider_max_value - simulation_slider_min_value) / 500.0, 0.01
    )

    simulation_limits_initialized = False
    scale_factor_slider_min = 0.0
    scale_factor_slider_max = 2.0
    scale_factor_step = 0.0001
    latest_result = None
    suppress_scale_update = False

    def apply_scaled_simulation():
        if (
            scale_factor_var is None
            or simulation_min_var is None
            or simulation_max_var is None
            or sim_artist is None
            or latest_result is None
        ):
            return

        scaled_image = latest_result * scale_factor_var.get()
        sim_artist.set_data(scaled_image)
        sim_artist.set_clim(
            simulation_min_var.get(), simulation_max_var.get()
        )
        canvas.draw_idle()

    # Initialize variables for sliders
    # Background display controls
    background_section = ttk.LabelFrame(slider_frame, text="Background Display")
    background_section.pack(fill=tk.X, padx=5, pady=5)

    background_min_var, _ = create_slider(
        "Background Min Intensity",
        background_slider_min_value,
        background_slider_max_value,
        background_vmin_default,
        background_slider_step,
        parent=background_section,
        update_callback=update_background_norm,
    )
    background_max_var, _ = create_slider(
        "Background Max Intensity",
        background_slider_min_value,
        background_slider_max_value,
        background_vmax_default,
        background_slider_step,
        parent=background_section,
        update_callback=update_background_norm,
    )
    background_transparency_var, _ = create_slider(
        "Background Transparency",
        0.0,
        1.0,
        0.0,
        0.01,
        parent=background_section,
        update_callback=update_background_norm,
    )
    update_background_norm()
    # Simulation display controls
    simulation_section = ttk.LabelFrame(slider_frame, text="Simulation Display")
    simulation_section.pack(fill=tk.X, padx=5, pady=5)

    simulation_min_var, simulation_min_slider = create_slider(
        "Simulation Min Intensity",
        simulation_slider_min_value,
        simulation_slider_max_value,
        simulation_vmin_default,
        simulation_slider_step,
        parent=simulation_section,
        update_callback=update_simulation_norm,
    )
    simulation_max_var, simulation_max_slider = create_slider(
        "Simulation Max Intensity",
        simulation_slider_min_value,
        simulation_slider_max_value,
        simulation_vmax_default,
        simulation_slider_step,
        parent=simulation_section,
        update_callback=update_simulation_norm,
    )
    scale_section = ttk.LabelFrame(slider_frame, text="Simulation Scaling")
    scale_section.pack(fill=tk.X, padx=5, pady=5)

    scale_factor_var, scale_factor_slider = create_slider(
        "Simulation Scale Factor",
        scale_factor_slider_min,
        scale_factor_slider_max,
        1.0,
        scale_factor_step,
        parent=scale_section,
    )

    def handle_scale_factor_change(*args):
        if suppress_scale_update:
            return
        apply_scaled_simulation()

    scale_factor_var.trace_add("write", handle_scale_factor_change)

    update_background_norm()

    geometry_section = ttk.LabelFrame(slider_frame, text="Geometry Parameters")
    geometry_section.pack(fill=tk.X, padx=5, pady=5)

    theta_initial_var, _ = create_slider(
        "Theta Initial", 5.0, 20.0, 6.0, 0.01, parent=geometry_section
    )
    cor_angle_var, _ = create_slider(
        "CoR Axis Angle", -45.0, 45.0, 0.0, 0.01, parent=geometry_section
    )
    gamma_var, _ = create_slider(
        "Gamma", -5, 5, ai.rot2, 0.001, parent=geometry_section
    )
    Gamma_var, _ = create_slider(
        "Detector Rotation Gamma", -5, 5, ai.rot1, 0.001, parent=geometry_section
    )
    chi_var, _ = create_slider(
        "Chi", -1, 1, 0.0, 0.001, parent=geometry_section
    )
    zs_var, _ = create_slider(
        "Zs", 0.0, 5e-3, 0.0, 0.0001, parent=geometry_section
    )
    zb_var, _ = create_slider(
        "Zb", 0.0, 5e-3, -0.1e-3, 0.0001, parent=geometry_section
    )
    eta_var, _ = create_slider(
        "Eta (fraction)", 0.0, 1.0, 0.08, 0.001, parent=geometry_section
    )
    sigma_mosaic_var, _ = create_slider(
        "Sigma Mosaic (deg)", 0.0, 5.0, 3.0, 0.01, parent=geometry_section
    )
    gamma_mosaic_var, _ = create_slider(
        "Gamma Mosaic (deg)", 0.0, 5.0, 0.7, 0.01, parent=geometry_section
    )

    # Flags to manage asynchronous updates
    is_computing = False
    pending_update = False

    # Resolution selector for profile sampling
    resolution_frame = ttk.Frame(slider_frame)
    resolution_frame.pack(fill=tk.X, pady=(10, 0))
    ttk.Label(resolution_frame, text="Sampling Resolution").pack(anchor=tk.W)

    resolution_var = tk.StringVar(value="High")
    resolution_options = {
        "Low": 25,
        "Medium": 250,
        "High": 500,
    }

    resolution_menu = ttk.OptionMenu(
        resolution_frame,
        resolution_var,
        resolution_var.get(),
        *resolution_options.keys(),
    )
    resolution_menu.pack(fill=tk.X, pady=(2, 5))

    # Generate random profiles
    def generate_random_profiles():
        num_samples = resolution_options.get(resolution_var.get(), 500)
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

        def worker(theta_initial, cor_angle, gamma, Gamma, chi, zs, zb):
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
                geometry_params=(ai.dist, gamma, Gamma, chi, cor_angle, zs, zb, 1),
            )

            try:
                raw_max = float(np.nanmax(result))
            except ValueError:
                raw_max = float("nan")

            def finish(image, image_max):
                nonlocal is_computing, pending_update, sim_artist
                nonlocal simulation_limits_initialized
                nonlocal latest_result, suppress_scale_update

                bg_reference = (
                    background_vmax_default if background_vmax_default > 0 else None
                )
                normalization_scale = 1.0
                finite_pixels = image[np.isfinite(image)]
                if finite_pixels.size and bg_reference is not None:
                    positive_pixels = finite_pixels[finite_pixels > 0]
                    reference_pixels = (
                        positive_pixels if positive_pixels.size else np.abs(finite_pixels)
                    )
                    sim_reference = float(np.nanpercentile(reference_pixels, 99))
                    if np.isfinite(sim_reference) and sim_reference > 0:
                        normalization_scale = bg_reference / sim_reference

                if (
                    (not np.isfinite(normalization_scale) or normalization_scale <= 0.0)
                    and np.isfinite(image_max)
                    and image_max > 0
                    and bg_reference is not None
                ):
                    normalization_scale = bg_reference / image_max

                if not np.isfinite(normalization_scale) or normalization_scale <= 0.0:
                    normalization_scale = 1.0

                normalized_image = image * normalization_scale
                latest_result = normalized_image

                suppress_scale_update = True
                scale_factor_var.set(1.0)
                suppress_scale_update = False

                scale_factor = scale_factor_var.get()
                scaled_image = normalized_image * scale_factor

                if not simulation_limits_initialized:
                    finite_pixels = scaled_image[np.isfinite(scaled_image)]
                    if finite_pixels.size:
                        sim_min = float(np.min(finite_pixels))
                        sim_max = float(np.max(finite_pixels))
                        if not np.isfinite(sim_min):
                            sim_min = simulation_vmin_default
                        if not np.isfinite(sim_max):
                            sim_max = simulation_vmax_default
                        if sim_max <= sim_min:
                            sim_max = sim_min + 1.0
                        margin = 0.05 * max(abs(sim_max), 1.0)
                        lower_bound = simulation_vmin_default
                        upper_bound = max(
                            sim_max + margin, simulation_vmin_default + 1.0
                        )
                        simulation_min_slider.configure(
                            from_=lower_bound, to=upper_bound
                        )
                        simulation_max_slider.configure(
                            from_=lower_bound, to=upper_bound
                        )
                        simulation_min_var.set(simulation_vmin_default)
                        simulation_max_var.set(upper_bound)
                    simulation_limits_initialized = True

                if sim_artist is None:
                    sim_artist = ax.imshow(scaled_image, cmap='turbo')
                else:
                    sim_artist.set_data(scaled_image)

                sim_artist.set_clim(
                    simulation_min_var.get(), simulation_max_var.get()
                )

                canvas.draw_idle()
                loading_label.pack_forget()
                status_canvas.itemconfig(status_rect, fill="green")
                root.update_idletasks()

                is_computing = False
                if pending_update:
                    pending_update = False
                    update_plot()

            root.after(0, lambda: finish(result, raw_max))

        is_computing = True
        loading_label.pack(side=tk.BOTTOM, pady=5)
        status_canvas.itemconfig(status_rect, fill="red")
        root.update_idletasks()


        theta_initial = theta_initial_var.get()
        cor_angle = cor_angle_var.get()
        gamma = gamma_var.get()
        Gamma = Gamma_var.get()
        chi = chi_var.get()
        zs = zs_var.get()
        zb = zb_var.get()

        threading.Thread(
            target=worker,
            args=(theta_initial, cor_angle, gamma, Gamma, chi, zs, zb),
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
    for var in [theta_initial_var, cor_angle_var, gamma_var, Gamma_var, chi_var,
                zs_var, zb_var, eta_var, sigma_mosaic_var, gamma_mosaic_var]:
        var.trace_add("write", schedule_update)

    resolution_var.trace_add("write", lambda *args: schedule_update())

    # Launch the main GUI loop
    root.mainloop()

if __name__ == "__main__":
    main()
