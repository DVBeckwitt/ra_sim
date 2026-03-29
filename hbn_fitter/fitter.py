"""Compatibility wrapper for the packaged hBN calibrant fitter."""

from __future__ import annotations

from ra_sim.hbn_fitter.fitter import *  # noqa: F401,F403


if __name__ == "__main__":
    from ra_sim.hbn_fitter.fitter import main

    main()
