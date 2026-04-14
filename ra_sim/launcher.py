"""Package-owned launcher helpers for GUI entrypoints."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path
from ra_sim import install_prereqs
from ra_sim.config import get_config_dir
from ra_sim.debug_controls import start_run_bundle, temporary_startup_debug_override
from ra_sim.gui import bootstrap as gui_bootstrap

_MOSAIC_MODULE = "mosaic_sim.unified_app"


def _simulation_gui_startup_error_message(exc: FileNotFoundError) -> str:
    """Build one concise startup hint for missing local GUI inputs."""

    config_dir = get_config_dir().resolve()
    file_paths_path = config_dir / "file_paths.yaml"
    missing_path = getattr(exc, "filename", None) or str(exc)
    config_hint = (
        f"No local config at '{file_paths_path}'. "
        "Copy `config/file_paths.example.yaml` to `config/file_paths.yaml` "
        "and point it at local `.osc`, `.poni`, `.cif`, and measured-peak files."
        if not file_paths_path.exists()
        else f"Check configured paths in '{file_paths_path}'."
    )
    return (
        "RA-SIM simulation GUI startup failed. "
        f"Missing file: {missing_path}. "
        f"{config_hint} "
        "Or set `RA_SIM_CONFIG_DIR` to your machine-specific config folder."
    )


def _show_simulation_gui_startup_error(message: str) -> None:
    """Best-effort GUI error dialog for startup failures."""

    try:
        tk_modules = install_prereqs.require_tkinter_modules(
            "The RA-SIM simulation GUI (`python -m ra_sim gui` or `ra-sim gui`)"
        )
        root = tk_modules.tk.Tk()
        try:
            root.withdraw()
            root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            tk_modules.tk.messagebox.showerror("RA-SIM Startup Error", str(message))
        finally:
            try:
                root.destroy()
            except Exception:
                pass
    except Exception:
        pass


def _pick_available_local_port() -> int:
    """Return an ephemeral localhost TCP port for a new Dash subprocess."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_simulation_gui(*, write_excel_flag: bool | None = None) -> None:
    """Launch the canonical packaged simulation GUI runtime."""

    install_prereqs.require_tkinter(
        "The RA-SIM simulation GUI (`python -m ra_sim gui` or `ra-sim gui`)"
    )

    debug_override = gui_bootstrap.quick_simulation_debug_override_dialog()
    if debug_override is None:
        return

    from ra_sim.gui.runtime import main as gui_main

    try:
        with temporary_startup_debug_override(debug_override):
            gui_main(
                write_excel_flag=write_excel_flag,
                startup_mode="simulation",
            )
    except FileNotFoundError as exc:
        error_message = _simulation_gui_startup_error_message(exc)
        _show_simulation_gui_startup_error(error_message)
        raise SystemExit(error_message) from exc


def launch_calibrant_gui(*, bundle: str | None = None) -> None:
    """Launch the calibrant fitter GUI."""

    gui_bootstrap.launch_calibrant_gui(bundle=bundle)


def _mosaic_command(*args: str) -> list[str]:
    """Build the module execution command for the installed mosaic visualizer."""

    return [sys.executable, "-m", _MOSAIC_MODULE, *args]


def launch_mosaic_visualizer() -> None:
    """Launch the installed mosaic_sim visualization tool."""

    try:
        subprocess.run(
            _mosaic_command(),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"mosaic_sim exited with status {exc.returncode}."
        ) from exc


def launch_mosaic_specular_visualizer(initial_state: object) -> None:
    """Launch the installed mosaic_sim app directly in seeded specular mode."""

    try:
        state_json = json.dumps(initial_state)
    except TypeError as exc:
        raise RuntimeError(f"Unable to serialize mosaic_sim startup state: {exc}") from exc

    port = _pick_available_local_port()
    try:
        subprocess.Popen(
            _mosaic_command(
                "--mode",
                "specular-view",
                "--port",
                str(port),
                "--state-json",
                state_json,
            ),
        )
    except OSError as exc:
        raise RuntimeError(f"Unable to launch mosaic_sim: {exc}") from exc


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
    if startup_mode == "mosaic":
        launch_mosaic_visualizer()
        return
    raise ValueError(
        "startup_mode must resolve to one of: simulation, calibrant, mosaic"
    )


def main(argv: list[str] | None = None) -> None:
    """Run the lightweight packaged GUI launcher."""

    cli_argv = list(sys.argv[1:] if argv is None else argv)
    if gui_bootstrap.should_forward_to_cli(cli_argv):
        from ra_sim.cli import main as cli_main

        cli_main(cli_argv)
        return

    args = gui_bootstrap.parse_launch_args(cli_argv)
    startup_mode = gui_bootstrap.resolve_startup_mode(args.command)
    if startup_mode == "prompt":
        startup_mode = gui_bootstrap.quick_startup_mode_dialog()

    if startup_mode is not None:
        start_run_bundle(entrypoint=f"launcher:{startup_mode}")
    launch_startup_mode(
        startup_mode,
        write_excel_flag=False if args.no_excel else None,
        calibrant_bundle=args.bundle,
    )
