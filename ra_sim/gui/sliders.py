"""Reusable Tkinter slider widgets."""

import tkinter as tk
from tkinter import ttk

def create_slider(
    label,
    min_val,
    max_val,
    initial_val,
    step_size,
    parent,
    update_callback=None,
    *,
    visible=True,
):
    frame = ttk.Frame(parent)
    if visible:
        frame.pack(pady=5, fill=tk.X)
    label_widget = ttk.Label(frame, text=label, font=("Helvetica", 10))
    label_widget.pack(anchor=tk.W)
    slider_var = tk.DoubleVar(value=initial_val)

    def slider_command(val):
        precise_value = round(float(val) / step_size) * step_size
        slider_var.set(precise_value)
        if update_callback is not None:
            update_callback()

    slider_row = ttk.Frame(frame)
    slider_row.pack(fill=tk.X, expand=True, padx=5)
    slider_row.columnconfigure(0, weight=1)

    slider = ttk.Scale(
        slider_row,
        from_=min_val,
        to=max_val,
        orient=tk.HORIZONTAL,
        variable=slider_var,
        command=slider_command,
    )
    slider.grid(row=0, column=0, sticky=tk.EW)

    entry = ttk.Entry(slider_row, textvariable=slider_var, width=10)
    entry.grid(row=0, column=1, sticky=tk.E, padx=(5, 0))

    def apply_entry_value(event=None):
        try:
            value = float(entry.get())
        except (tk.TclError, ValueError):
            return

        value = max(min_val, min(max_val, value))
        precise_value = round(value / step_size) * step_size
        slider.set(precise_value)

    entry.bind("<FocusOut>", apply_entry_value)
    entry.bind("<Return>", apply_entry_value)

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
