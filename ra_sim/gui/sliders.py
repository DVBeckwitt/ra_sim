import tkinter as tk
from tkinter import ttk

def create_slider(parent, label, min_val, max_val, initial_val, step_size, command=None):
    frame = ttk.Frame(parent)
    frame.pack(pady=5, fill=tk.X)
    label_widget = ttk.Label(frame, text=label)
    label_widget.pack(anchor=tk.W)

    slider_var = tk.DoubleVar(value=initial_val)

    def slider_command(val):
        precise_value = round(float(val)/step_size)*step_size
        slider_var.set(precise_value)
        if command:
            command()

    slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=slider_var, command=slider_command)
    slider.pack(fill=tk.X, padx=5)

    entry = ttk.Entry(frame, textvariable=slider_var, width=10)
    entry.pack(side=tk.RIGHT, padx=5)

    return slider_var, slider
