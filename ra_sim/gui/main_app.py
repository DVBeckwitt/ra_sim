import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..io.file_parsing import parse_poni_file
from ..io.data_loading import load_background_image
from ..simulation.geometry import setup_azimuthal_integrator
from ..gui.plotting import setup_figure
from ..gui.sliders import create_slider

def main():
    root = tk.Tk()
    root.title("XRD Analysis")

    # Example usage:
    # Load parameters
    params = parse_poni_file(r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Analysis\Bi2Se3\geometry.poni")

    ai = setup_azimuthal_integrator(params)

    # Load a background image
    bg_image = load_background_image(r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Analysis\Bi2Se3\In-Plane\3\Bi2Se3_6d_5m.asc")

    # Setup figure
    fig, ax = setup_figure()
    ax.imshow(bg_image, cmap='turbo', vmin=0, vmax=1e3)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()

    # Add sliders
    slider_frame = ttk.Frame(root)
    slider_frame.pack(side=tk.LEFT, fill=tk.Y)
    gamma_var, gamma_slider = create_slider(slider_frame, "Gamma", -5, 5, 0.0, 0.001, command=canvas.draw_idle)

    # Additional UI elements as needed
    root.mainloop()
