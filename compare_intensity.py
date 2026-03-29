"""Compatibility wrapper for ``ra_sim.tools.compare_intensity``."""

from __future__ import annotations

from ra_sim.tools.compare_intensity import (
    _numeric_column,
    compute_metrics,
    main,
    plot_comparison,
)

__all__ = [
    "_numeric_column",
    "compute_metrics",
    "plot_comparison",
    "main",
]


if __name__ == "__main__":
    main()
