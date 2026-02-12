"""GUI view helpers used by RA-SIM Tk applications."""

from __future__ import annotations

import tkinter as tk


def create_root_window(title: str = "RA Simulation") -> tk.Tk:
    """Create and return a Tk root window with the provided title."""

    root = tk.Tk()
    root.title(title)
    return root
