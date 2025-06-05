"""Small GUI helpers for update dialogs."""

import tkinter as tk
from tkinter import ttk

def create_slider(parent, label, min_val, max_val, initial_val, step_size):
    frame = ttk.Frame(parent)
    frame.pack(pady=5, fill=tk.X)
    label_widget = ttk.Label(frame, text=label)
    label_widget.pack(anchor=tk.W)
    slider_var = tk.DoubleVar(value=initial_val)

    def slider_command(val):
        slider_var.set(round(float(val) / step_size) * step_size)

    slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=slider_var, orient=tk.HORIZONTAL, command=slider_command)
    slider.pack(fill=tk.X, padx=5)
    return slider_var, slider

