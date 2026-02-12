"""GUI controller helpers."""

from __future__ import annotations

from typing import Any

from ra_sim.path_config import get_instrument_config

from .state import AppState


def build_initial_state() -> AppState:
    """Build the initial GUI state snapshot from current configuration."""

    instrument_cfg = get_instrument_config()
    detector_cfg = instrument_cfg.get("instrument", {}).get("detector", {})
    image_size = int(detector_cfg.get("image_size", 3000))
    return AppState(
        instrument_config=instrument_cfg,
        image_size=image_size,
    )


def launch_gui(*, write_excel_flag: bool | None = None) -> Any:
    """Launch the full GUI application."""

    from . import app

    return app.main(write_excel_flag=write_excel_flag)
