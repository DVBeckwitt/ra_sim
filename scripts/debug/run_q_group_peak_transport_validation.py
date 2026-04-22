#!/usr/bin/env python
"""Run headless Q-group peak transport validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_sim.gui.peak_sensitivity import PeakSensitivityEvaluationError  # noqa: E402
from ra_sim.gui.peak_transport_validation import (  # noqa: E402
    METRIC_ALL,
    METRIC_CHOICES,
    TransportValidationOptions,
    _json_safe,
    run_peak_transport_validation,
    write_transport_validation_artifacts,
)


def _parse_params(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return ["theta_initial", "corto_detector"]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _status_counts(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _decision_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for row in rows:
        parameter = str(row.get("parameter"))
        metric = str(row.get("metric"))
        comparison = str(row.get("comparison"))
        branch = str(row.get("branch_id"))
        decision_scope = str(row.get("decision_scope"))
        summary.setdefault(parameter, {})
        param_payload = summary[parameter]
        if isinstance(param_payload, dict):
            param_payload.setdefault(metric, {})
            metric_payload = param_payload[metric]
            if isinstance(metric_payload, dict):
                metric_payload.setdefault(comparison, {})
                comparison_payload = metric_payload[comparison]
                if isinstance(comparison_payload, dict):
                    comparison_payload.setdefault(branch, {})
                    branch_payload = comparison_payload[branch]
                    if not isinstance(branch_payload, dict):
                        continue
                    branch_payload.setdefault(
                        "overall_recommendation",
                        row.get("overall_recommendation"),
                    )
                    branch_payload[decision_scope] = {
                        "recommendation": row.get("recommendation"),
                        "overall_recommendation": row.get("overall_recommendation"),
                        "can_transport": row.get("can_transport"),
                        "plus_pass": row.get("plus_pass"),
                        "minus_pass": row.get("minus_pass"),
                        "max_abs_error": row.get("max_abs_error"),
                    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Q-group peak point transport against full recompute.",
    )
    parser.add_argument("--state", required=True, help="Saved GUI-state JSON path.")
    parser.add_argument("--group-key", required=True, help="Q-group key, e.g. q_group,primary,1,5.")
    parser.add_argument(
        "--params",
        default="theta_initial,corto_detector",
        help="Comma-separated parameters.",
    )
    parser.add_argument("--metric", choices=METRIC_CHOICES, default=METRIC_ALL)
    parser.add_argument("--outdir", required=True, help="Output artifact directory.")
    parser.add_argument("--roi-two-theta-half-width", type=float, default=0.5)
    parser.add_argument("--roi-phi-half-width", type=float, default=0.5)
    parser.add_argument("--background-percentile", type=float, default=50.0)
    parser.add_argument("--min-total-weight", type=float, default=0.0)
    parser.add_argument("--min-cloud-points", type=int, default=3)
    parser.add_argument("--tol-two-theta-deg", type=float, default=0.002)
    parser.add_argument("--tol-phi-deg", type=float, default=0.002)
    parser.add_argument("--tol-shape", type=float, default=0.005)
    parser.add_argument(
        "--step-mode",
        choices=("default", "relative-all"),
        default="default",
    )
    parser.add_argument("--relative-step", type=float, default=1.0e-4)
    args = parser.parse_args(argv)

    options = TransportValidationOptions(
        roi_two_theta_half_width=float(args.roi_two_theta_half_width),
        roi_phi_half_width=float(args.roi_phi_half_width),
        background_percentile=float(args.background_percentile),
        min_total_weight=float(args.min_total_weight),
        min_cloud_points=int(args.min_cloud_points),
        tol_two_theta_deg=float(args.tol_two_theta_deg),
        tol_phi_deg=float(args.tol_phi_deg),
        tol_shape=float(args.tol_shape),
        relative_step=float(args.relative_step),
        step_mode=str(args.step_mode),
    )
    try:
        result = run_peak_transport_validation(
            state_path=Path(args.state),
            group_key=str(args.group_key),
            parameter_names=_parse_params(args.params),
            metric=str(args.metric),
            options=options,
        )
    except (PeakSensitivityEvaluationError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "eval_error", "error": str(exc)}, sort_keys=True))
        return 2

    paths = write_transport_validation_artifacts(result, args.outdir)
    payload = {
        "status": result.metadata.get("status", "ok"),
        "outdir": str(Path(args.outdir).expanduser().resolve()),
        "files": {name: str(path) for name, path in paths.items()},
        "baseline_branch_count": len(
            {
                str(row.get("branch_id"))
                for row in result.baseline_point_rows
                if row.get("branch_id") is not None
            }
        ),
        "metric_status_counts": _status_counts(result.comparison_rows),
        "decisions": _decision_summary(result.decision_rows),
        "transport_used_integrate2d": result.metadata.get("transport_used_integrate2d"),
        "transport_used_refinement": result.metadata.get("transport_used_refinement"),
        "transport_used_full_recompute": result.metadata.get("transport_used_full_recompute"),
    }
    print(json.dumps(_json_safe(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
