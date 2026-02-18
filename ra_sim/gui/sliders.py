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
    allow_range_expand=False,
    range_expand_pad=None,
):
    frame = ttk.Frame(parent)
    if visible:
        frame.pack(pady=5, fill=tk.X)
    label_widget = ttk.Label(frame, text=label, font=("Helvetica", 10))
    label_widget.pack(anchor=tk.W)
    slider_var = tk.DoubleVar(value=float(initial_val))
    entry_var = tk.StringVar()
    step = abs(float(step_size))
    if step <= 0:
        step = 0.0

    def _decimal_places(step_value):
        if step_value <= 0:
            return 6
        text = f"{step_value:.12f}".rstrip("0").rstrip(".")
        if "." in text:
            return len(text.split(".", 1)[1])
        return 0

    decimals = _decimal_places(step)

    def _current_limits():
        try:
            lo = float(slider.cget("from"))
            hi = float(slider.cget("to"))
        except Exception:
            lo = float(min_val)
            hi = float(max_val)
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    def _clamp(value):
        lo, hi = _current_limits()
        return max(lo, min(hi, float(value)))

    def _expand_range_to_include(value):
        if not allow_range_expand:
            return
        lo, hi = _current_limits()
        val = float(value)
        new_lo = lo
        new_hi = hi
        if range_expand_pad is None:
            pad = max(step, 0.1)
        else:
            pad = max(0.0, float(range_expand_pad))
        if val < lo:
            new_lo = val - pad
        if val > hi:
            new_hi = val + pad
        if new_lo != lo or new_hi != hi:
            slider.configure(from_=new_lo, to=new_hi)

    def _snap(value):
        if allow_range_expand:
            _expand_range_to_include(value)
        value = _clamp(value)
        if step <= 0:
            return value
        lo, hi = _current_limits()
        snapped = lo + round((value - lo) / step) * step
        return max(lo, min(hi, snapped))

    def _format_value(value):
        if decimals <= 0:
            return f"{int(round(value))}"
        return f"{value:.{decimals}f}"

    def slider_command(val):
        precise_value = _snap(float(val))
        slider_var.set(precise_value)
        entry_var.set(_format_value(precise_value))
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

    entry = ttk.Entry(slider_row, textvariable=entry_var, width=10)
    entry.grid(row=0, column=1, sticky=tk.E, padx=(5, 0))

    entry_var.set(_format_value(_snap(slider_var.get())))

    def _sync_entry_from_var(*_):
        try:
            current_value = float(slider_var.get())
            if allow_range_expand:
                _expand_range_to_include(current_value)
            precise_value = _snap(current_value)
        except (tk.TclError, ValueError):
            return
        entry_var.set(_format_value(precise_value))

    slider_var.trace_add("write", _sync_entry_from_var)

    def apply_entry_value(event=None):
        raw = entry_var.get().strip().lower()
        if not raw:
            entry_var.set(_format_value(_snap(slider_var.get())))
            return
        if raw == "max":
            value = _current_limits()[1]
        elif raw == "min":
            value = _current_limits()[0]
        else:
            try:
                value = float(raw)
            except (tk.TclError, ValueError):
                entry_var.set(_format_value(_snap(slider_var.get())))
                return

        if allow_range_expand:
            _expand_range_to_include(value)
        precise_value = _snap(value)
        try:
            slider.set(precise_value)
        except (tk.TclError, ValueError):
            entry_var.set(_format_value(_snap(slider_var.get())))
            return
        entry_var.set(_format_value(precise_value))

    entry.bind("<FocusOut>", apply_entry_value)
    entry.bind("<Return>", apply_entry_value)

    def on_key(event):
        if event.keysym == 'Left':
            new_val = _snap(slider_var.get() - (step if step > 0 else 1.0))
            slider_var.set(new_val)
            if update_callback is not None:
                update_callback()
            return "break"
        elif event.keysym == 'Right':
            new_val = _snap(slider_var.get() + (step if step > 0 else 1.0))
            slider_var.set(new_val)
            if update_callback is not None:
                update_callback()
            return "break"

    def on_click(event):
        slider.focus_set()
        slider.bind('<KeyPress>', on_key)

    slider.bind('<Button-1>', on_click)

    # Ensure initial value is clamped/snapped and reflected in the entry.
    initial_precise = _snap(slider_var.get())
    slider_var.set(initial_precise)
    entry_var.set(_format_value(initial_precise))

    return slider_var, slider
