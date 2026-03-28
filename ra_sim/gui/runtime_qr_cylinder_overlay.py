"""Import-safe helpers for assembling runtime Qr-cylinder overlay workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeQrCylinderOverlayWorkflow:
    """Bundled Qr-cylinder overlay runtime assembly results."""

    active_entries_factory: object
    render_config_factory: object
    runtime: object
    bindings_factory: object
    refresh: object
    toggle: object


def build_runtime_qr_cylinder_overlay_workflow(
    *,
    bragg_qr_manager_module,
    qr_cylinder_overlay_module,
    bootstrap_module,
    active_entry_factory_kwargs: dict[str, Any],
    render_config_factory_kwargs: dict[str, Any],
    overlay_bootstrap_kwargs: dict[str, Any],
) -> RuntimeQrCylinderOverlayWorkflow:
    """Assemble the live Qr-cylinder overlay factories and runtime bundle."""

    active_entries_factory = (
        bragg_qr_manager_module.make_runtime_active_qr_cylinder_overlay_entries_factory(
            **dict(active_entry_factory_kwargs)
        )
    )
    render_config_factory = (
        qr_cylinder_overlay_module.make_runtime_qr_cylinder_overlay_render_config_factory(
            **dict(render_config_factory_kwargs)
        )
    )
    merged_bootstrap_kwargs = dict(overlay_bootstrap_kwargs)
    merged_bootstrap_kwargs["get_active_entries"] = active_entries_factory
    merged_bootstrap_kwargs["render_config_factory"] = render_config_factory
    runtime = bootstrap_module.build_runtime_qr_cylinder_overlay_bootstrap(
        qr_cylinder_overlay_module=qr_cylinder_overlay_module,
        **merged_bootstrap_kwargs,
    )
    return RuntimeQrCylinderOverlayWorkflow(
        active_entries_factory=active_entries_factory,
        render_config_factory=render_config_factory,
        runtime=runtime,
        bindings_factory=runtime.bindings_factory,
        refresh=runtime.refresh,
        toggle=runtime.toggle,
    )
