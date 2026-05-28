"""Import-safe helpers for assembling runtime geometry-fit workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeGeometryFitWorkflow:
    """Bundled geometry-fit runtime assembly results."""

    value_callbacks: object
    var_map: dict[str, object]
    manual_dataset_bindings_factory: object
    runtime_config_factory: object
    action_workflow: object
    action_runtime: object
    action_bindings_factory: object
    on_fit_geometry_click: object


def persist_geometry_fit_preflight_failure_log(
    *,
    geometry_fit_module,
    stamp: str,
    log_path: Path | str,
    error_text: str,
    failure_log_sections: Sequence[tuple[str, Sequence[str]]] | None,
) -> Path | None:
    """Persist one geometry-fit preflight failure log when logging is enabled."""

    if geometry_fit_module.geometry_fit_all_logging_disabled():
        return None
    try:
        return geometry_fit_module.write_geometry_fit_preflight_failure_log(
            stamp=str(stamp),
            error_text=str(error_text),
            log_path=Path(log_path),
            log_sections=failure_log_sections,
        )
    except Exception:
        return None


def build_geometry_fit_worker_action_result(
    *,
    geometry_fit_module,
    job: Mapping[str, object],
    prepare_result: object = None,
    execution_result: object = None,
    error_text: str | None = None,
):
    """Build one geometry-fit action result from async runtime job metadata."""

    resolved_error_text = (
        str(error_text)
        if error_text
        else (
            str(getattr(execution_result, "error_text", "") or "")
            if execution_result is not None
            else None
        )
    )
    return geometry_fit_module.GeometryFitRuntimeActionResult(
        params=dict(job.get("params", {}) or {}),
        var_names=[str(name) for name in (job.get("var_names", ()) or ())],
        preserve_live_theta=bool(job.get("preserve_live_theta", False)),
        prepare_result=prepare_result,
        execution_result=execution_result,
        error_text=resolved_error_text or None,
    )


def build_runtime_geometry_fit_workflow(
    *,
    geometry_fit_module,
    runtime_fit_analysis_module,
    bootstrap_module,
    value_bindings,
    manual_dataset_bindings_factory_kwargs: dict[str, Any],
    runtime_config_factory_kwargs: dict[str, Any],
    action_bootstrap_kwargs: dict[str, Any],
) -> RuntimeGeometryFitWorkflow:
    """Assemble the live geometry-fit runtime wiring and aliases."""

    value_callbacks = geometry_fit_module.build_runtime_geometry_fit_value_callbacks(
        value_bindings
    )
    var_map = dict(getattr(value_callbacks, "var_map", {}) or {})
    manual_dataset_bindings_factory = (
        geometry_fit_module.make_runtime_geometry_fit_manual_dataset_bindings_factory(
            **dict(manual_dataset_bindings_factory_kwargs)
        )
    )
    runtime_config_factory = geometry_fit_module.build_runtime_geometry_fit_config_factory(
        **dict(runtime_config_factory_kwargs)
    )
    merged_action_kwargs = dict(action_bootstrap_kwargs)
    merged_action_kwargs["manual_dataset_bindings_factory"] = (
        manual_dataset_bindings_factory
    )
    merged_action_kwargs["build_runtime_config_factory"] = runtime_config_factory
    merged_action_kwargs["var_map"] = var_map
    action_workflow = runtime_fit_analysis_module.build_runtime_geometry_fit_action_workflow(
        bootstrap_module=bootstrap_module,
        geometry_fit_module=geometry_fit_module,
        geometry_fit_action_bootstrap_kwargs=merged_action_kwargs,
    )
    return RuntimeGeometryFitWorkflow(
        value_callbacks=value_callbacks,
        var_map=var_map,
        manual_dataset_bindings_factory=manual_dataset_bindings_factory,
        runtime_config_factory=runtime_config_factory,
        action_workflow=action_workflow,
        action_runtime=action_workflow.runtime,
        action_bindings_factory=action_workflow.bindings_factory,
        on_fit_geometry_click=action_workflow.callback,
    )
