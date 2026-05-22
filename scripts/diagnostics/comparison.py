"""Compatibility entry point for the background peak-fit diagnostic.

The old generated copy in this file drifted from the maintained diagnostic and
could overwrite manuscript figures with stale Qr-rod plots. Keep one source of
truth by delegating to the canonical script next to this file.
"""

from __future__ import annotations

from pathlib import Path
import runpy


CANONICAL_DIAGNOSTIC = (
    Path(__file__).resolve().with_name(
        "all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py"
    )
)


if __name__ == "__main__":
    runpy.run_path(str(CANONICAL_DIAGNOSTIC), run_name="__main__")
