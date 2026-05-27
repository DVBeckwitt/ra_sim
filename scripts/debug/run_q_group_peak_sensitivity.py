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

from ra_sim.gui.peak_sensitivity import (
    DEFAULT_PARAMETER_NAMES,
    DEFAULT_METRIC,
    METRIC_CHOICES,
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
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default=DEFAULT_METRIC,
        help=(
            "Sensitivity metric. Default preserves refined-max tracking; 'all' writes shape metrics."
        ),
    )
    parser.add_argument("--roi-two-theta-half-width", type=float, default=0.5)
    parser.add_argument("--roi-phi-half-width", type=float, default=0.5)
    parser.add_argument("--background-percentile", type=float, default=50.0)
    parser.add_argument("--min-total-weight", type=float, default=0.0)
    parser.add_argument("--min-cloud-points", type=int, default=3)
    parser.add_argument("--outdir", required=True, help="Output artifact directory.")
    args = parser.parse_args(argv)

    try:
        result = run_peak_sensitivity(
            state_path=Path(args.state),
            group_key=str(args.group_key),
            parameter_names=_parse_params(args.params),
            relative_step=float(args.relative_step),
            step_mode=str(args.step_mode),
            metric=str(args.metric),
            roi_two_theta_half_width=float(args.roi_two_theta_half_width),
            roi_phi_half_width=float(args.roi_phi_half_width),
            background_percentile=float(args.background_percentile),
            min_total_weight=float(args.min_total_weight),
            min_cloud_points=int(args.min_cloud_points),
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
                "selected_metric": result.selected_metric,
                "shape_baseline": result.metadata.get("baseline_shape_metrics", {}),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
