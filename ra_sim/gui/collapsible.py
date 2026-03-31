import tkinter as tk
from tkinter import ttk

class CollapsibleFrame(ttk.Frame):
    """A Tkinter collapsible frame with a toggle button."""

    def __init__(self, parent, text="", expanded=False):
        super().__init__(parent)
        self._text = text
        self._summary_text = ""
        self._variable = tk.IntVar(value=1 if expanded else 0)
        self._button = ttk.Checkbutton(
            self,
            text=self._label_text(),
            variable=self._variable,
            command=self._toggle,
            style="SectionHeader.Toolbutton",
        )
        self._button.pack(fill=tk.X, padx=2, pady=(2, 0))
        self._separator = ttk.Separator(self, orient=tk.HORIZONTAL)
        self.frame = ttk.Frame(self, padding=(8, 4, 8, 8))
        if expanded:
            self._separator.pack(fill=tk.X, padx=2, pady=(2, 0))
            self.frame.pack(fill=tk.X, padx=2, pady=(0, 2))

    def _label_text(self):
        prefix = ("\u25BC " if self._variable.get() else "\u25B6 ") + self._text
        if not self._variable.get() and self._summary_text:
            return f"{prefix} | {self._summary_text}"
        return prefix

    def set_header_summary(self, text=""):
        self._summary_text = " ".join(str(text or "").split())
        self._button.configure(text=self._label_text())

    def _toggle(self):
        if self._variable.get():
            self._separator.pack(fill=tk.X, padx=2, pady=(2, 0))
            self.frame.pack(fill=tk.X, padx=2, pady=(0, 2))
        else:
            self.frame.forget()
            self._separator.forget()
        self._button.configure(text=self._label_text())
