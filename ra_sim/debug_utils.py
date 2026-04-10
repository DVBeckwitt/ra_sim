"""Helper utilities for debugging RA_SIM execution."""

import os
import logging
from collections.abc import Mapping

import numpy as np
from ra_sim.debug_controls import (
    console_debug_enabled,
    env_flag_enabled as _env_flag_enabled,
    is_logging_disabled as _is_logging_disabled,
)


def env_flag_enabled(
    name: str,
    env: Mapping[str, object] | None = None,
) -> bool:
    """Return whether one environment-style flag is set to a truthy value."""

    return _env_flag_enabled(name, env)


def is_logging_disabled(
    env: Mapping[str, object] | None = None,
) -> bool:
    """Return whether the global logging kill-switch is enabled.

    ``RA_SIM_DISABLE_ALL_LOGGING`` is the preferred switch. The older
    ``RA_SIM_DISABLE_LOGGING`` name is still honored for compatibility.
    """

    return _is_logging_disabled(env)


def is_debug_enabled() -> bool:
    """Return ``True`` if ``RA_SIM_DEBUG`` is set to a truthy value."""
    return console_debug_enabled()


DEBUG_ENABLED = is_debug_enabled()


def debug_print(*args, **kwargs) -> None:
    """Print only when ``RA_SIM_DEBUG`` is enabled."""
    if is_debug_enabled():
        print(*args, **kwargs)


def enable_numba_logging(default_level: str = "WARNING") -> None:
    """Configure the ``numba`` logger when debug mode is active.

    If ``RA_SIM_DEBUG`` is enabled this sets up the ``numba`` logger to emit
    messages to ``stdout`` using the log level from ``NUMBA_LOG_LEVEL`` if
    defined or ``default_level`` otherwise. The default stays at
    ``WARNING`` so general RA_SIM debug sessions do not dump Numba's
    internal compiler trace unless the user explicitly opts in.
    """
    if not is_debug_enabled():
        return

    level_name = os.environ.get("NUMBA_LOG_LEVEL", default_level).upper()
    os.environ["NUMBA_LOG_LEVEL"] = level_name

    try:
        level = getattr(logging, level_name)
    except AttributeError:
        level = logging.DEBUG

    logger = logging.getLogger("numba")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s numba: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)


def check_ht_arrays(miller1: np.ndarray, intens1: np.ndarray) -> None:
    """Print diagnostics for Hendricks–Teller arrays when debugging is enabled."""
    debug_print("miller1 dtype:", miller1.dtype, "shape:", miller1.shape)
    if miller1.size:
        l_min = float(miller1[:, 2].min())
        l_max = float(miller1[:, 2].max())
        debug_print("L range:", l_min, l_max)
        if np.allclose(miller1[:, 2], np.round(miller1[:, 2])):
            debug_print("L values are integer-only")
        else:
            frac_count = np.count_nonzero(~np.isclose(miller1[:, 2], np.round(miller1[:, 2])))
            debug_print(f"L values contain {frac_count} fractional entries")
    else:
        debug_print("L range: array empty")

    if intens1.size:
        i_min = float(intens1.min())
        i_max = float(intens1.max())
    else:
        i_min = i_max = float('nan')
    debug_print("intens1 dtype:", intens1.dtype, "min:", i_min, "max:", i_max)
    debug_print("miller1 contiguous:", miller1.flags['C_CONTIGUOUS'])
    debug_print("intens1 contiguous:", intens1.flags['C_CONTIGUOUS'])
