from __future__ import annotations

from types import SimpleNamespace

from ra_sim.gui import runtime_geometry_fit


def test_persist_geometry_fit_preflight_failure_log_delegates_when_logging_enabled(tmp_path) -> None:
    calls: list[dict[str, object]] = []
    expected_path = tmp_path / "geometry-fit-preflight.log"

    geometry_fit_module = SimpleNamespace(
        geometry_fit_all_logging_disabled=lambda: False,
        write_geometry_fit_preflight_failure_log=lambda **kwargs: (
            calls.append(dict(kwargs)) or expected_path
        ),
    )

    result = runtime_geometry_fit.persist_geometry_fit_preflight_failure_log(
        geometry_fit_module=geometry_fit_module,
        stamp="20260528_120000",
        log_path=tmp_path / "fit.log",
        error_text="preflight failed",
        failure_log_sections=[("details", ["one", "two"])],
    )

    assert result == expected_path
    assert calls == [
        {
            "stamp": "20260528_120000",
            "error_text": "preflight failed",
            "log_path": tmp_path / "fit.log",
            "log_sections": [("details", ["one", "two"])],
        }
    ]


def test_persist_geometry_fit_preflight_failure_log_suppresses_disabled_or_failed_writes(
    tmp_path,
) -> None:
    disabled_module = SimpleNamespace(
        geometry_fit_all_logging_disabled=lambda: True,
        write_geometry_fit_preflight_failure_log=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled logging must not write")
        ),
    )
    failing_module = SimpleNamespace(
        geometry_fit_all_logging_disabled=lambda: False,
        write_geometry_fit_preflight_failure_log=lambda **_kwargs: (_ for _ in ()).throw(
            OSError("disk full")
        ),
    )

    assert (
        runtime_geometry_fit.persist_geometry_fit_preflight_failure_log(
            geometry_fit_module=disabled_module,
            stamp="stamp",
            log_path=tmp_path / "fit.log",
            error_text="error",
            failure_log_sections=None,
        )
        is None
    )
    assert (
        runtime_geometry_fit.persist_geometry_fit_preflight_failure_log(
            geometry_fit_module=failing_module,
            stamp="stamp",
            log_path=tmp_path / "fit.log",
            error_text="error",
            failure_log_sections=None,
        )
        is None
    )


def test_build_geometry_fit_worker_action_result_normalizes_job_payload() -> None:
    created: list[dict[str, object]] = []
    geometry_fit_module = SimpleNamespace(
        GeometryFitRuntimeActionResult=lambda **kwargs: (
            created.append(dict(kwargs)) or SimpleNamespace(**kwargs)
        )
    )
    execution_result = SimpleNamespace(error_text="execution failed")
    prepare_result = SimpleNamespace(status="prepared")

    result = runtime_geometry_fit.build_geometry_fit_worker_action_result(
        geometry_fit_module=geometry_fit_module,
        job={
            "params": {"gamma": 1.25},
            "var_names": ["gamma", 2],
            "preserve_live_theta": 1,
        },
        prepare_result=prepare_result,
        execution_result=execution_result,
    )

    assert result.params == {"gamma": 1.25}
    assert result.var_names == ["gamma", "2"]
    assert result.preserve_live_theta is True
    assert result.prepare_result is prepare_result
    assert result.execution_result is execution_result
    assert result.error_text == "execution failed"
    assert created == [
        {
            "params": {"gamma": 1.25},
            "var_names": ["gamma", "2"],
            "preserve_live_theta": True,
            "prepare_result": prepare_result,
            "execution_result": execution_result,
            "error_text": "execution failed",
        }
    ]


def test_build_geometry_fit_worker_action_result_explicit_error_text_wins() -> None:
    geometry_fit_module = SimpleNamespace(
        GeometryFitRuntimeActionResult=lambda **kwargs: SimpleNamespace(**kwargs)
    )

    result = runtime_geometry_fit.build_geometry_fit_worker_action_result(
        geometry_fit_module=geometry_fit_module,
        job={},
        execution_result=SimpleNamespace(error_text="execution failed"),
        error_text="preflight failed",
    )

    assert result.params == {}
    assert result.var_names == []
    assert result.preserve_live_theta is False
    assert result.error_text == "preflight failed"


def test_build_runtime_geometry_fit_workflow_composes_value_dataset_config_and_action_workflow() -> None:
    calls: list[tuple[str, object]] = []
    value_callbacks = SimpleNamespace(var_map={"theta_initial": "theta-var"})
    action_workflow = SimpleNamespace(
        runtime="action-runtime",
        bindings_factory="action-bindings-factory",
        callback="on-fit-geometry-click",
    )

    geometry_fit_module = SimpleNamespace(
        build_runtime_geometry_fit_value_callbacks=(
            lambda bindings: calls.append(("value", bindings)) or value_callbacks
        ),
        make_runtime_geometry_fit_manual_dataset_bindings_factory=(
            lambda **kwargs: calls.append(("manual-dataset", kwargs))
            or "manual-dataset-factory"
        ),
        build_runtime_geometry_fit_config_factory=(
            lambda **kwargs: calls.append(("runtime-config", kwargs))
            or "runtime-config-factory"
        ),
    )
    runtime_fit_analysis_module = SimpleNamespace(
        build_runtime_geometry_fit_action_workflow=lambda **kwargs: (
            calls.append(("action-workflow", kwargs)) or action_workflow
        )
    )

    workflow = runtime_geometry_fit.build_runtime_geometry_fit_workflow(
        geometry_fit_module=geometry_fit_module,
        runtime_fit_analysis_module=runtime_fit_analysis_module,
        bootstrap_module="bootstrap-module",
        value_bindings="value-bindings",
        manual_dataset_bindings_factory_kwargs={
            "osc_files_factory": "osc-files-factory",
            "image_size": 2048,
        },
        runtime_config_factory_kwargs={
            "base_config": {"solver": {"loss": "soft_l1"}},
            "current_constraint_state": "constraint-state",
        },
        action_bootstrap_kwargs={
            "value_callbacks_factory": "value-callbacks-factory",
            "schedule_update": "schedule-update",
        },
    )

    assert workflow.value_callbacks is value_callbacks
    assert workflow.var_map == {"theta_initial": "theta-var"}
    assert workflow.manual_dataset_bindings_factory == "manual-dataset-factory"
    assert workflow.runtime_config_factory == "runtime-config-factory"
    assert workflow.action_workflow is action_workflow
    assert workflow.action_runtime == "action-runtime"
    assert workflow.action_bindings_factory == "action-bindings-factory"
    assert workflow.on_fit_geometry_click == "on-fit-geometry-click"
    assert calls == [
        ("value", "value-bindings"),
        (
            "manual-dataset",
            {
                "osc_files_factory": "osc-files-factory",
                "image_size": 2048,
            },
        ),
        (
            "runtime-config",
            {
                "base_config": {"solver": {"loss": "soft_l1"}},
                "current_constraint_state": "constraint-state",
            },
        ),
        (
            "action-workflow",
            {
                "bootstrap_module": "bootstrap-module",
                "geometry_fit_module": geometry_fit_module,
                "geometry_fit_action_bootstrap_kwargs": {
                    "value_callbacks_factory": "value-callbacks-factory",
                    "schedule_update": "schedule-update",
                    "manual_dataset_bindings_factory": "manual-dataset-factory",
                    "build_runtime_config_factory": "runtime-config-factory",
                    "var_map": {"theta_initial": "theta-var"},
                },
            },
        ),
    ]
