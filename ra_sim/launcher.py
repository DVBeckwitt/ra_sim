"""Package-owned launcher helpers for GUI entrypoints.

This keeps ``ra_sim.cli`` and the repository-root ``main.py`` off the legacy
root-level GUI module while preserving the historical launch flows.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


_LAUNCHER_COMMANDS = {"gui", "calibrant"}


def _quick_startup_mode_dialog() -> str | None:
    """Ask which GUI mode to launch before importing the heavy simulation app."""

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        # Headless or Tk unavailable: default to simulation behavior.
        return "simulation"

    choice = {"mode": None}

    launcher = tk.Tk()
    launcher.title("RA-SIM Launcher")
    launcher.configure(bg="#e8eef5")
    launcher.resizable(False, False)
    launcher.attributes("-topmost", True)

    try:
        style = ttk.Style(launcher)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    panel = tk.Frame(
        launcher,
        bg="#ffffff",
        highlightbackground="#d4deeb",
        highlightthickness=1,
    )
    panel.pack(fill="both", expand=True, padx=18, pady=18)

    tk.Label(
        panel,
        text="Choose Startup Mode",
        font=("Segoe UI", 16, "bold"),
        fg="#0f172a",
        bg="#ffffff",
    ).pack(anchor="w", padx=18, pady=(18, 4))
    tk.Label(
        panel,
        text="Select what you want to run right now.",
        font=("Segoe UI", 10),
        fg="#475569",
        bg="#ffffff",
    ).pack(anchor="w", padx=18, pady=(0, 14))

    def _set_mode(mode_name: str | None) -> None:
        choice["mode"] = mode_name
        launcher.destroy()

    sim_btn = tk.Button(
        panel,
        text="Run Simulation GUI",
        font=("Segoe UI", 11, "bold"),
        bg="#2563eb",
        fg="#ffffff",
        activebackground="#1d4ed8",
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        padx=14,
        pady=10,
        cursor="hand2",
        command=lambda: _set_mode("simulation"),
    )
    sim_btn.pack(fill="x", padx=18, pady=(0, 6))
    tk.Label(
        panel,
        text="Full RA-SIM simulation workspace and controls.",
        font=("Segoe UI", 9),
        fg="#64748b",
        bg="#ffffff",
    ).pack(anchor="w", padx=20, pady=(0, 12))

    cal_btn = tk.Button(
        panel,
        text="Fit Calibrant (hBN Fitter)",
        font=("Segoe UI", 11, "bold"),
        bg="#0f766e",
        fg="#ffffff",
        activebackground="#0f5f59",
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        padx=14,
        pady=10,
        cursor="hand2",
        command=lambda: _set_mode("calibrant"),
    )
    cal_btn.pack(fill="x", padx=18, pady=(0, 6))
    tk.Label(
        panel,
        text="Open the calibrant fitting tool directly.",
        font=("Segoe UI", 9),
        fg="#64748b",
        bg="#ffffff",
    ).pack(anchor="w", padx=20, pady=(0, 16))

    footer = tk.Frame(panel, bg="#ffffff")
    footer.pack(fill="x", padx=18, pady=(0, 16))
    tk.Label(
        footer,
        text="Keyboard: 1 = Simulation, 2 = Calibrant, Esc = Cancel",
        font=("Segoe UI", 8),
        fg="#64748b",
        bg="#ffffff",
    ).pack(side="left")
    tk.Button(
        footer,
        text="Cancel",
        font=("Segoe UI", 9),
        bg="#e2e8f0",
        fg="#1e293b",
        activebackground="#cbd5e1",
        activeforeground="#1e293b",
        relief="flat",
        bd=0,
        padx=12,
        pady=6,
        cursor="hand2",
        command=lambda: _set_mode(None),
    ).pack(side="right")

    launcher.bind("<Escape>", lambda _event: _set_mode(None))
    launcher.bind("<Return>", lambda _event: _set_mode("simulation"))
    launcher.bind("1", lambda _event: _set_mode("simulation"))
    launcher.bind("2", lambda _event: _set_mode("calibrant"))
    launcher.protocol("WM_DELETE_WINDOW", lambda: _set_mode(None))

    launcher.update_idletasks()
    width = max(launcher.winfo_width(), 480)
    height = max(launcher.winfo_height(), 330)
    x = launcher.winfo_screenwidth() // 2 - width // 2
    y = launcher.winfo_screenheight() // 2 - height // 2
    launcher.geometry(f"{width}x{height}+{x}+{y}")
    sim_btn.focus_set()
    launcher.mainloop()

    mode = choice["mode"]
    if mode in {"simulation", "calibrant"}:
        return mode
    return None


def launch_simulation_gui(*, write_excel_flag: bool | None = None) -> None:
    """Launch the packaged simulation GUI."""

    from ra_sim.gui.app import main as gui_main

    gui_main(
        write_excel_flag=write_excel_flag,
        startup_mode="simulation",
    )


def launch_calibrant_gui(*, bundle: str | None = None) -> None:
    """Launch the calibrant fitter GUI."""

    import tkinter as tk

    try:
        from hbn_fitter.fitter import HBNFitterGUI
    except Exception as exc:
        raise RuntimeError(
            "Unable to import hbn_fitter GUI. Ensure `hbn_fitter/fitter.py` exists."
        ) from exc

    root = tk.Tk()
    _ = HBNFitterGUI(root, startup_bundle=bundle)
    root.mainloop()


def launch_startup_mode(
    startup_mode: str | None,
    *,
    write_excel_flag: bool | None = None,
    calibrant_bundle: str | None = None,
) -> None:
    """Launch the selected startup mode."""

    if startup_mode is None:
        return
    if startup_mode == "simulation":
        launch_simulation_gui(write_excel_flag=write_excel_flag)
        return
    if startup_mode == "calibrant":
        launch_calibrant_gui(bundle=calibrant_bundle)
        return
    raise ValueError("startup_mode must resolve to one of: simulation, calibrant")


def compatibility_main(
    write_excel_flag: bool | None = None,
    startup_mode: str = "prompt",
    calibrant_bundle: str | None = None,
) -> None:
    """Compatibility entrypoint matching the legacy ``main.main`` signature."""

    if startup_mode not in {"prompt", "simulation", "calibrant"}:
        raise ValueError("startup_mode must be one of: prompt, simulation, calibrant")

    resolved_mode = startup_mode
    if resolved_mode == "prompt":
        resolved_mode = _quick_startup_mode_dialog()

    launch_startup_mode(
        resolved_mode,
        write_excel_flag=write_excel_flag,
        calibrant_bundle=calibrant_bundle,
    )


def _should_forward_to_cli(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    first_arg = argv[0]
    return (
        first_arg not in _LAUNCHER_COMMANDS.union({"-h", "--help"})
        and not first_arg.startswith("--")
    )


def main(argv: list[str] | None = None) -> None:
    """Run the lightweight launcher used by the repository-root wrapper."""

    cli_argv = list(sys.argv[1:] if argv is None else argv)
    if _should_forward_to_cli(cli_argv):
        from ra_sim.cli import main as cli_main

        cli_main(cli_argv)
        return

    parser = argparse.ArgumentParser(description="RA Simulation launcher")
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(_LAUNCHER_COMMANDS),
        help=(
            "Optional startup mode: 'gui' runs simulation directly; "
            "'calibrant' launches the hBN fitter directly."
        ),
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Do not write the initial intensity Excel file",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Optional NPZ bundle path to preload in calibrant mode.",
    )
    args = parser.parse_args(cli_argv)

    startup_mode = "prompt"
    if args.command == "gui":
        startup_mode = "simulation"
    elif args.command == "calibrant":
        startup_mode = "calibrant"

    compatibility_main(
        write_excel_flag=False if args.no_excel else None,
        startup_mode=startup_mode,
        calibrant_bundle=args.bundle,
    )
