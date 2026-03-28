from __future__ import annotations

from types import SimpleNamespace

from ra_sim.gui import runtime_qr_cylinder_overlay


def test_build_runtime_qr_cylinder_overlay_workflow_composes_factories_and_runtime() -> None:
    calls: list[tuple[str, object]] = []
    runtime_obj = SimpleNamespace(
        bindings_factory="overlay-bindings-factory",
        refresh="overlay-refresh",
        toggle="overlay-toggle",
    )

    bragg_qr_manager_module = SimpleNamespace(
        make_runtime_active_qr_cylinder_overlay_entries_factory=(
            lambda **kwargs: calls.append(("active-entries", kwargs))
            or "active-entries-factory"
        )
    )
    qr_cylinder_overlay_module = SimpleNamespace(
        make_runtime_qr_cylinder_overlay_render_config_factory=(
            lambda **kwargs: calls.append(("render-config", kwargs))
            or "render-config-factory"
        )
    )
    bootstrap_module = SimpleNamespace(
        build_runtime_qr_cylinder_overlay_bootstrap=lambda **kwargs: (
            calls.append(("runtime", kwargs)) or runtime_obj
        )
    )

    workflow = runtime_qr_cylinder_overlay.build_runtime_qr_cylinder_overlay_workflow(
        bragg_qr_manager_module=bragg_qr_manager_module,
        qr_cylinder_overlay_module=qr_cylinder_overlay_module,
        bootstrap_module=bootstrap_module,
        active_entry_factory_kwargs={
            "simulation_runtime_state": "simulation-runtime-state",
            "primary_fallback": 4.0,
        },
        render_config_factory_kwargs={
            "image_size": 2048,
            "display_rotate_k": 3,
        },
        overlay_bootstrap_kwargs={
            "ax": "axis",
            "overlay_cache": "overlay-cache",
        },
    )

    assert workflow.active_entries_factory == "active-entries-factory"
    assert workflow.render_config_factory == "render-config-factory"
    assert workflow.runtime is runtime_obj
    assert workflow.bindings_factory == "overlay-bindings-factory"
    assert workflow.refresh == "overlay-refresh"
    assert workflow.toggle == "overlay-toggle"
    assert calls == [
        (
            "active-entries",
            {
                "simulation_runtime_state": "simulation-runtime-state",
                "primary_fallback": 4.0,
            },
        ),
        (
            "render-config",
            {
                "image_size": 2048,
                "display_rotate_k": 3,
            },
        ),
        (
            "runtime",
            {
                "qr_cylinder_overlay_module": qr_cylinder_overlay_module,
                "ax": "axis",
                "overlay_cache": "overlay-cache",
                "get_active_entries": "active-entries-factory",
                "render_config_factory": "render-config-factory",
            },
        ),
    ]
