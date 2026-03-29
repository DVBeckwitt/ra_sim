"""Compatibility wrapper for the packaged hBN tilt optimizer."""

from __future__ import annotations

from ra_sim.hbn_fitter.optimizer import *  # noqa: F401,F403


if __name__ == "__main__":
    from ra_sim.hbn_fitter.optimizer import main

    main()
