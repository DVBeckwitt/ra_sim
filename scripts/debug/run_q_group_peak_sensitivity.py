#!/usr/bin/env python
"""Export a central finite-difference Qr/Qz peak sensitivity matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_sim.gui.peak_sensitivity import (  # noqa: E402
    DEFAULT_PARAMETER_NAMES,
    PeakSensitivityEvaluationError,
    run_peak_sensitivity,
    write_sensitivity_artifacts,
)


def _parse_params(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return list(DEFAULT_PARAMETER_NAMES)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute a headless Qr/Qz caked peak sensitivity Jacobian.",
    )
    parser.add_argument("--state", required=True, help="Saved GUI-state JSON path.")
    parser.add_argument("--group-key", required=True, help="Q-group key, e.g. q_group,primary,3,4.")
    parser.add_argument(
        "--params",
        default=",".join(DEFAULT_PARAMETER_NAMES),
        help="Comma-separated sensitivity parameters.",
    )
    parser.add_argument("--relative-step", type=float, default=1.0e-4)
    parser.add_argument(
        "--step-mode",
        choices=("default", "relative-all"),
        default="default",
        help="default keeps fixed angular/pixel steps; relative-all scales every parameter.",
    )
    parser.add_argument("--outdir", required=True, help="Output artifact directory.")
    args = parser.parse_args(argv)

    try:
        result = run_peak_sensitivity(
            state_path=Path(args.state),
            group_key=str(args.group_key),
            parameter_names=_parse_params(args.params),
            relative_step=float(args.relative_step),
            step_mode=str(args.step_mode),
            outdir=None,
        )
    except PeakSensitivityEvaluationError as exc:
        print(json.dumps({"status": "eval_error", "error": str(exc)}, sort_keys=True))
        return 2
    paths = write_sensitivity_artifacts(result, args.outdir, metadata=result.metadata)
    status = str(result.metadata.get("status") or "ok")
    print(
        json.dumps(
            {
                "status": status,
                "outdir": str(Path(args.outdir).expanduser().resolve()),
                "files": {name: str(path) for name, path in paths.items()},
                "baseline_peak_count": len(result.baseline_observations),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
