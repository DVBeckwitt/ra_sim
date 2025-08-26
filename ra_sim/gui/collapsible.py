import tkinter as tk
from tkinter import ttk

class CollapsibleFrame(ttk.Frame):
    """A Tkinter collapsible frame with a toggle button."""

    def __init__(self, parent, text="", expanded=False):
        super().__init__(parent)
        self._text = text
        self._variable = tk.IntVar(value=1 if expanded else 0)
        self._button = ttk.Checkbutton(
            self,
            text=self._label_text(),
            variable=self._variable,
            command=self._toggle,
            style="Toolbutton",
        )
        self._button.pack(fill=tk.X)
        self.frame = ttk.Frame(self)
        if expanded:
            self.frame.pack(fill=tk.X)

    def _label_text(self):
        return ("\u25BC " if self._variable.get() else "\u25B6 ") + self._text

    def _toggle(self):
        if self._variable.get():
            self.frame.pack(fill=tk.X)
        else:
            self.frame.forget()
        self._button.configure(text=self._label_text())
