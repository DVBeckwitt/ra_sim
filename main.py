#!/usr/bin/env python3

"""Compatibility wrapper for the packaged RA-SIM GUI launcher."""

from __future__ import annotations

from ra_sim.launcher import compatibility_main as _compatibility_main
from ra_sim.launcher import main as _launcher_main

write_excel = False


def main(
    write_excel_flag: bool | None = None,
    startup_mode: str = "prompt",
    calibrant_bundle: str | None = None,
) -> None:
    """Preserve the historical ``main.main(...)`` entrypoint."""

    global write_excel
    if write_excel_flag is not None:
        write_excel = bool(write_excel_flag)

    _compatibility_main(
        write_excel_flag=write_excel_flag,
        startup_mode=startup_mode,
        calibrant_bundle=calibrant_bundle,
    )


if __name__ == "__main__":
    _launcher_main()
