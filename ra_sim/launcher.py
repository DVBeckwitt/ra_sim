"""Package-owned launcher helpers for GUI entrypoints.

This keeps ``ra_sim.cli`` and the repository-root ``main.py`` off the legacy
root-level GUI module while preserving the historical launch flows.
"""

from __future__ import annotations

import sys
from ra_sim.gui import bootstrap as gui_bootstrap


def launch_simulation_gui(*, write_excel_flag: bool | None = None) -> None:
    """Launch the packaged simulation GUI."""

    from ra_sim.gui.app import main as gui_main

    gui_main(
        write_excel_flag=write_excel_flag,
        startup_mode="simulation",
    )


def launch_calibrant_gui(*, bundle: str | None = None) -> None:
    """Launch the calibrant fitter GUI."""

    gui_bootstrap.launch_calibrant_gui(bundle=bundle)


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
        resolved_mode = gui_bootstrap.quick_startup_mode_dialog()

    launch_startup_mode(
        resolved_mode,
        write_excel_flag=write_excel_flag,
        calibrant_bundle=calibrant_bundle,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the lightweight launcher used by the repository-root wrapper."""

    cli_argv = list(sys.argv[1:] if argv is None else argv)
    if gui_bootstrap.should_forward_to_cli(cli_argv):
        from ra_sim.cli import main as cli_main

        cli_main(cli_argv)
        return

    args = gui_bootstrap.parse_launch_args(cli_argv)
    startup_mode = gui_bootstrap.resolve_startup_mode(args.command)

    compatibility_main(
        write_excel_flag=False if args.no_excel else None,
        startup_mode=startup_mode,
        calibrant_bundle=args.bundle,
    )
