"""Compatibility wrapper for ``ra_sim.tools.plot_excel_scatter``."""

from __future__ import annotations

from ra_sim.tools.plot_excel_scatter import (
    _find_column,
    _find_intensity_columns,
    _normalize_columns,
    main,
)

__all__ = [
    "_find_column",
    "_find_intensity_columns",
    "_normalize_columns",
    "main",
]


if __name__ == "__main__":
    main()
