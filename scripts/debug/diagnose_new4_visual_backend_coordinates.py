"""Diagnose New4 visual-vs-backend geometry-fit coordinate parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_sim.gui.geometry_fit_coordinate_diagnostics import (  # noqa: E402
    run_new4_visual_backend_coordinate_diagnostic,
)


def _latest_rung_one(state_path: Path) -> Path | None:
    ladder_root = state_path.parents[1] / "geometry_fit_ladder" / "new4"
    if not ladder_root.exists():
        return None
    candidates = list(ladder_root.glob("*/rung_01_objective_dry_run.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose New4 visual/backend coordinate parity without optimizer calls.",
    )
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--provider-report", required=False, type=Path, default=None)
    parser.add_argument("--background-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--include-optimizer-request",
        action="store_true",
        help="Build optimizer request rows without running the solver.",
    )
    args = parser.parse_args(argv)

    state_path = args.state.resolve()
    provider_report_path = (
        args.provider_report.resolve() if args.provider_report is not None else None
    )
    rung_one = _latest_rung_one(state_path)
    report = run_new4_visual_backend_coordinate_diagnostic(
        state_path=state_path,
        provider_report_path=provider_report_path,
        background_index=int(args.background_index),
        output_dir=args.output_dir.resolve(),
        include_optimizer_request=bool(args.include_optimizer_request),
        rung_report_path=rung_one,
    )
    report["auto_detected_rung_01_objective_dry_run"] = bool(rung_one is not None)
    (args.output_dir.resolve() / "coordinate_transform_diagnosis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Keep stdout compact; full deterministic report is written by the runner.
    print(
        json.dumps(
            {
                "ok": bool(report.get("ok", False)),
                "classification": report.get("classification"),
                "first_mismatching_surface": report.get("first_mismatching_surface"),
                "recommended_fix_location": report.get("recommended_fix_location"),
                "optimizer_request_compared": bool(
                    report.get("optimizer_request_compared", False)
                ),
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(report.get("visual_truth_available", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
