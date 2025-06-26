"""Reusable Tkinter slider widgets."""

import tkinter as tk
from tkinter import ttk

def create_slider(label, min_val, max_val, initial_val, step_size, parent, update_callback=None):
    frame = ttk.Frame(parent)
    frame.pack(pady=5, fill=tk.X)
    label_widget = ttk.Label(frame, text=label, font=("Helvetica", 10))
    label_widget.pack(anchor=tk.W)
    slider_var = tk.DoubleVar(value=initial_val)

    def slider_command(val):
        precise_value = round(float(val) / step_size) * step_size
        slider_var.set(precise_value)
        if update_callback is not None:
            update_callback()

    slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                       variable=slider_var, command=slider_command)
    slider.pack(fill=tk.X, padx=5)

    entry = ttk.Entry(frame, textvariable=slider_var, width=10)
    entry.pack(side=tk.RIGHT, padx=5)

    def on_key(event):
        if event.keysym == 'Left':
            new_val = slider_var.get() - step_size + 1.0
            new_val = max(new_val, min_val)
            slider_var.set(round(new_val / step_size) * step_size)
            if update_callback is not None:
                update_callback()
        elif event.keysym == 'Right':
            new_val = slider_var.get() + step_size - 1.0
            new_val = min(new_val, max_val)
            slider_var.set(round(new_val / step_size) * step_size)
            if update_callback is not None:
                update_callback()

    def on_click(event):
        slider.focus_set()
        slider.bind('<KeyPress>', on_key)

    slider.bind('<Button-1>', on_click)

    return slider_var, slider
