"""Compatibility wrapper for older ``ra_sim.gui.main_app`` imports."""

from __future__ import annotations


def main(*args, **kwargs):
    from ra_sim.gui.app import main as app_main

    return app_main(*args, **kwargs)


if __name__ == "__main__":
    main()
