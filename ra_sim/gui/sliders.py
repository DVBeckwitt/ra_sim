import tkinter as tk
from tkinter import ttk

def create_slider(parent, label, min_val, max_val, initial_val, step_size, command=None):
    """
    Creates a labeled slider with an entry box.

    Parameters:
        parent (tk.Widget): The parent widget to contain the slider.
        label (str): The label for the slider.
        min_val (float): The minimum value of the slider.
        max_val (float): The maximum value of the slider.
        initial_val (float): The initial value of the slider.
        step_size (float): The step size for the slider.
        command (callable, optional): Function to call when the slider value changes.

    Returns:
        tuple: A tkinter DoubleVar for the slider value and the slider widget itself.
    """
    # Create a frame to contain the slider and its label
    frame = ttk.Frame(parent)
    frame.pack(pady=5, fill=tk.X)

    # Add a label
    label_widget = ttk.Label(frame, text=label)
    label_widget.pack(anchor=tk.W)

    # Create a variable to store the slider's value
    slider_var = tk.DoubleVar(value=initial_val)

    def slider_command(val):
        """
        Updates the slider value to match the closest step and calls the optional command.
        """
        precise_value = round(float(val) / step_size) * step_size
        slider_var.set(precise_value)
        if command:
            command()

    # Add the slider
    slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                       variable=slider_var, command=slider_command)
    slider.pack(fill=tk.X, padx=5)

    # Add an entry box linked to the slider
    entry = ttk.Entry(frame, textvariable=slider_var, width=10)
    entry.pack(side=tk.RIGHT, padx=5)

    # Enable keyboard input for fine adjustments
    def on_key(event):
        if event.keysym in ['Left', 'Right']:
            adjustment = step_size if event.keysym == 'Right' else -step_size
            new_val = slider_var.get() + adjustment
            slider_var.set(min(max(new_val, min_val), max_val))
            if command:
                command()

    slider.bind('<KeyPress>', on_key)

    return slider_var, slider
