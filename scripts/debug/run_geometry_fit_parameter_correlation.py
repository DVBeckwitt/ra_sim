#!/usr/bin/env python
"""Export a headless geometry-fit parameter correlation map."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_sim.gui.geometry_fit_correlation import (  # noqa: E402
    GeometryFitCorrelationError,
    run_geometry_fit_correlation,
    write_correlation_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute a headless geometry-fit parameter correlation matrix.",
    )
    parser.add_argument("--state", required=True, help="Saved GUI-state JSON path.")
    parser.add_argument(
        "--background-index",
        type=int,
        default=None,
        help="Optional background index. Defaults to the saved state's current selection.",
    )
    parser.add_argument(
        "--params",
        default="all",
        help="Comma-separated parameters, 'all', or 'active'. Default: all.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.90,
        help="Absolute correlation threshold used to flag high-correlation pairs.",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=1,
        help="Baseline no-solve probe max_nfev placed into the solver config.",
    )
    parser.add_argument("--outdir", required=True, help="Output artifact directory.")
    args = parser.parse_args(argv)

    try:
        result = run_geometry_fit_correlation(
            state_path=Path(args.state),
            background_index=args.background_index,
            parameter_names=str(args.params),
            max_nfev=int(args.max_nfev),
            correlation_threshold=float(args.correlation_threshold),
            outdir=None,
        )
    except GeometryFitCorrelationError as exc:
        print(json.dumps({"status": "eval_error", "error": str(exc)}, sort_keys=True))
        return 2
    paths = write_correlation_artifacts(result, args.outdir)
    print(
        json.dumps(
            {
                "status": "ok",
                "outdir": str(Path(args.outdir).expanduser().resolve()),
                "files": {name: str(path) for name, path in paths.items()},
                "parameter_count": len(result.parameters),
                "pair_count": int(result.metadata.get("pair_count", 0)),
                "high_correlation_pair_count": int(
                    result.metadata.get("high_correlation_pair_count", 0)
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
