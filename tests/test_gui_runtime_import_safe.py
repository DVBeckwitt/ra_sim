import ast
import concurrent.futures.thread as futures_thread
import contextlib
import importlib
import json
import os
import py_compile
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


RUNTIME_MODULE_NAME = "ra_sim.gui.runtime"
RUNTIME_IMPL_MODULE_NAME = "ra_sim.gui._runtime_impl"
PACKAGE_RUNTIME_IMPL_MODULE_NAME = "ra_sim.gui._runtime.runtime_impl"
RUNTIME_IMPL_WRAPPER_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "ra_sim" / "gui" / "_runtime" / "runtime_impl.py"
)
RUNTIME_SESSION_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "ra_sim" / "gui" / "_runtime" / "runtime_session.py"
)
PRIMARY_CACHE_HELPERS_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent
    / "ra_sim"
    / "gui"
    / "_runtime"
    / "primary_cache_helpers.py"
)
STRUCTURE_MODEL_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "ra_sim" / "gui" / "structure_model.py"
)
FILE_PARSING_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "ra_sim" / "io" / "file_parsing.py"
)
CLI_SOURCE_PATH = Path(__file__).resolve().parent.parent / "ra_sim" / "cli.py"
HEADLESS_GEOMETRY_FIT_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "ra_sim" / "headless_geometry_fit.py"
)
REPO_ROOT = Path(__file__).resolve().parent.parent
GUI_SOURCE_ROOT = REPO_ROOT / "ra_sim" / "gui"
RA_SIM_SOURCE_ROOT = REPO_ROOT / "ra_sim"
OPTIMIZATION_MOSAIC_PROFILES_SOURCE_PATH = (
    REPO_ROOT / "ra_sim" / "fitting" / "optimization_mosaic_profiles.py"
)
RAW_SOURCE_PEAK_READ_ALLOWLIST = {
    GUI_SOURCE_ROOT / "_runtime" / "runtime_session.py",
    GUI_SOURCE_ROOT / "analysis_peak_tools.py",
    GUI_SOURCE_ROOT / "geometry_fit.py",
    GUI_SOURCE_ROOT / "geometry_q_group_manager.py",
    GUI_SOURCE_ROOT / "manual_geometry.py",
}
TRUST_FIELD_ASSIGNMENT_ALLOWLIST = {
    GUI_SOURCE_ROOT / "manual_geometry.py",
}


def _geometry_fit_worker_live_row() -> dict[str, object]:
    return {
        "hkl": (1, 0, 0),
        "q_group_key": ("q", 1),
        "source_reflection_index": 7,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 10.0,
        "sim_row": 20.0,
    }


def _geometry_fit_worker_required_pair() -> dict[str, object]:
    return {
        "pair_id": "bg0:pair0",
        "overlay_match_index": 0,
        "hkl": (1, 0, 0),
        "q_group_key": ("q", 1),
        "source_reflection_index": 7,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }


def _make_geometry_fit_worker_job(runtime_session) -> dict[str, object]:
    event_queue = runtime_session.queue.Queue()
    return {
        "job_id": 99,
        "event_queue": event_queue,
        "params": {
            "center": [1.0, 1.0],
            "corto_detector": 0.5,
            "pixel_size_m": 1.0e-4,
            "lambda": 1.54e-10,
            "theta_initial": 0.0,
        },
        "var_names": [],
        "preserve_live_theta": False,
        "source_snapshots": {},
        "source_snapshot_diagnostics": {},
        "simulation_diagnostics": {},
        "background_images": {
            0: {
                "native": np.ones((4, 4), dtype=np.float64),
                "display": np.ones((4, 4), dtype=np.float64),
            }
        },
        "requested_signatures": {0: ("sig", 0)},
        "requested_signature_summaries": {0: "sig-summary"},
        "background_labels": {0: "bg0.osc"},
        "live_rows_signature": ("sig", 0),
        "live_rows_by_background": {0: [_geometry_fit_worker_live_row()]},
        "live_rows_cache_metadata_by_background": {0: {}},
        "memory_intersection_cache": [],
        "memory_intersection_cache_signature": ("sig", 0),
        "manual_pairs_by_background": {0: [_geometry_fit_worker_required_pair()]},
        "required_indices": [0],
        "current_background_index": 0,
        "image_size": 4,
        "theta_initial": 0.0,
        "theta_initial_by_background": {0: 0.0},
        "theta_base_by_background": {0: 0.0},
        "geometry_runtime_cfg": {},
        "fit_config": {},
        "live_cache_inventory": {"source_snapshot_count": 1},
        "solver_inputs": SimpleNamespace(miller=[], intensities=[], image_size=4),
        "selected_background_indices": [0],
        "joint_background_mode": False,
        "osc_files": ["bg0.osc"],
        "selection_applied": True,
        "theta_metadata_applied": True,
        "background_theta_values": [0.0],
        "theta_offset": 0.0,
        "uses_shared_theta": False,
        "projection_view_mode": "detector",
        "projection_view_signature": {"mode": "detector", "detector_shape": [4, 4]},
        "projection_view_signature_by_background": {
            0: {"mode": "detector", "detector_shape": [4, 4]}
        },
        "projection_payload_by_background": {},
        "stamp": "20260420_000000",
        "log_path": REPO_ROOT / "artifacts" / "geometry_fit_worker_test.log",
        "enable_live_update_events": False,
    }


def _geometry_fit_worker_caked_payload(
    runtime_session,
    *,
    background_value: float,
    radial_values: Sequence[float],
    azimuth_values: Sequence[float],
) -> dict[str, object]:
    radial_axis = np.asarray(radial_values, dtype=np.float64)
    azimuth_axis = np.asarray(azimuth_values, dtype=np.float64)
    raw_azimuth_axis = np.asarray(
        runtime_session.gui_geometry_fit.gui_phi_to_raw_phi(azimuth_axis),
        dtype=np.float64,
    )
    return {
        "background": np.full(
            (int(azimuth_axis.size), int(radial_axis.size)),
            float(background_value),
            dtype=np.float64,
        ),
        "radial_axis": radial_axis,
        "azimuth_axis": azimuth_axis,
        "raw_azimuth_axis": raw_azimuth_axis,
        "raw_to_gui_row_permutation": np.arange(int(azimuth_axis.size), dtype=np.int32),
        "transform_bundle": runtime_session.CakeTransformBundle(
            detector_shape=(4, 4),
            radial_deg=radial_axis,
            raw_azimuth_deg=raw_azimuth_axis,
            gui_azimuth_deg=azimuth_axis,
            lut=object(),
        ),
        "detector_shape": (4, 4),
    }


def _drain_geometry_fit_worker_events(event_queue) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    while True:
        try:
            events.append(event_queue.get_nowait())
        except Exception:
            return events


def _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session) -> SimpleNamespace:
    simulation_state = SimpleNamespace(
        geometry_fit_targeted_projected_cache_by_background={},
        geometry_fit_caking_ai_cache={},
        analysis_preview_bins=(4, 4),
        source_row_snapshots={},
        last_simulation_signature=None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_source_snapshot_diagnostics_state",
        {},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=1, osc_files=["bg0.osc"]),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {
            "a": 4.143,
            "c": 28.64,
            "lambda": 1.54,
            "theta_initial": 0.0,
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_for_background",
        lambda index, params: (
            "sig",
            int(index),
            float((params or {}).get("a", 0.0)),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_signature_summary",
        lambda signature: f"sig-summary:{signature!r}",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_inventory_snapshot",
        lambda: {"source_snapshot_count": 0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_background_label",
        lambda index: f"bg{int(index)}.osc",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_build_live_preview_simulated_peaks_from_cache",
        lambda: [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_last_live_preview_cache_metadata",
        lambda: {},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "get_last_intersection_cache",
        lambda: [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "is_intersection_cache_table",
        lambda _table: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_copy_hit_tables",
        lambda tables: list(tables or ()),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_filter_hit_tables_for_required_branch_groups",
        lambda hit_tables, required_branch_group_keys=None: (
            list(hit_tables or ()),
            {
                "total_hit_tables_available": int(len(hit_tables or ())),
                "hit_tables_considered_for_rebinding": int(len(hit_tables or ())),
                "hit_tables_expanded_for_rebinding": int(len(hit_tables or ())),
            },
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        lambda rows: [dict(entry) for entry in rows or () if isinstance(entry, dict)],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_trace_live_cache_event",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_last_simulation_diagnostics",
        lambda: {"status": "success"},
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "miller", [], raising=False)
    monkeypatch.setattr(runtime_session, "intensities", [], raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    return simulation_state


def _top_level_import_targets(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            targets.extend(str(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "" if node.module is None else str(node.module)
            targets.extend(f"{module}:{alias.name}" for alias in node.names)
    return targets


def test_runtime_import_is_lazy() -> None:
    previous_runtime = sys.modules.pop(RUNTIME_MODULE_NAME, None)
    previous_impl = sys.modules.pop(RUNTIME_IMPL_MODULE_NAME, None)

    try:
        runtime = importlib.import_module(RUNTIME_MODULE_NAME)

        assert runtime.__name__ == RUNTIME_MODULE_NAME
        assert RUNTIME_IMPL_MODULE_NAME not in sys.modules
        assert callable(runtime.main)
    finally:
        sys.modules.pop(RUNTIME_MODULE_NAME, None)
        sys.modules.pop(RUNTIME_IMPL_MODULE_NAME, None)
        if previous_runtime is not None:
            sys.modules[RUNTIME_MODULE_NAME] = previous_runtime
        if previous_impl is not None:
            sys.modules[RUNTIME_IMPL_MODULE_NAME] = previous_impl


def test_runtime_main_loads_impl_and_forwards_arguments(monkeypatch) -> None:
    runtime = importlib.import_module(RUNTIME_MODULE_NAME)
    calls: list[dict[str, object]] = []

    fake_impl = SimpleNamespace(
        write_excel=None,
        main=lambda **kwargs: calls.append(dict(kwargs)),
    )

    monkeypatch.setattr(runtime, "_load_runtime_module", lambda: fake_impl)
    runtime.write_excel = False

    runtime.main(
        write_excel_flag=True,
        startup_mode="simulation",
        calibrant_bundle="bundle.npz",
    )

    assert fake_impl.write_excel is True
    assert calls == [
        {
            "write_excel_flag": True,
            "startup_mode": "simulation",
            "calibrant_bundle": "bundle.npz",
        }
    ]


def test_runtime_dunder_attribute_raises_without_loading(monkeypatch) -> None:
    runtime = importlib.import_module(RUNTIME_MODULE_NAME)

    monkeypatch.setattr(
        runtime,
        "_load_runtime_module",
        lambda: (_ for _ in ()).throw(AssertionError("impl should not be imported")),
    )

    with pytest.raises(AttributeError):
        _ = runtime.__runtime_test_guard__


def test_runtime_dir_is_lazy() -> None:
    runtime = importlib.import_module(RUNTIME_MODULE_NAME)

    available = dir(runtime)

    assert "main" in available
    assert "write_excel" in available


def test_runtime_unknown_attr_forwards_to_impl(monkeypatch) -> None:
    runtime = importlib.import_module(RUNTIME_MODULE_NAME)
    fake_impl = SimpleNamespace(test_value=123)

    monkeypatch.setattr(runtime, "_load_runtime_module", lambda: fake_impl)

    assert runtime.test_value == 123


def test_runtime_impl_source_compiles() -> None:
    py_compile.compile(str(RUNTIME_IMPL_WRAPPER_SOURCE_PATH), doraise=True)
    py_compile.compile(str(RUNTIME_SESSION_SOURCE_PATH), doraise=True)


def test_runtime_impl_wrapper_import_is_lazy() -> None:
    previous_wrapper = sys.modules.pop(PACKAGE_RUNTIME_IMPL_MODULE_NAME, None)
    previous_session = sys.modules.pop("ra_sim.gui._runtime.runtime_session", None)

    try:
        wrapper = importlib.import_module(PACKAGE_RUNTIME_IMPL_MODULE_NAME)

        assert wrapper.__name__ == PACKAGE_RUNTIME_IMPL_MODULE_NAME
        assert "ra_sim.gui._runtime.runtime_session" not in sys.modules
        assert callable(wrapper.main)
    finally:
        sys.modules.pop(PACKAGE_RUNTIME_IMPL_MODULE_NAME, None)
        sys.modules.pop("ra_sim.gui._runtime.runtime_session", None)
        if previous_wrapper is not None:
            sys.modules[PACKAGE_RUNTIME_IMPL_MODULE_NAME] = previous_wrapper
        if previous_session is not None:
            sys.modules["ra_sim.gui._runtime.runtime_session"] = previous_session


def test_runtime_session_import_stays_headless_until_tk_backend_is_needed() -> None:
    script = """
import importlib
import json
import sys

sys.modules.pop("matplotlib.backends.backend_tkagg", None)
sys.modules.pop("ra_sim.gui._runtime.runtime_session", None)

runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

import matplotlib

print(
    json.dumps(
        {
            "backend": matplotlib.get_backend(),
            "tk_backend_loaded": "matplotlib.backends.backend_tkagg" in sys.modules,
            "tk_canvas_cls_cached": runtime_session._TK_FIGURE_CANVAS_CLS is not None,
        }
    )
)
"""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout.strip())

    assert payload["backend"].lower() == "agg"
    assert payload["tk_backend_loaded"] is False
    assert payload["tk_canvas_cls_cached"] is False


def test_runtime_session_defers_scipy_gui_modules_until_first_use() -> None:
    script = """
import importlib
import json
import sys
from types import SimpleNamespace

lazy_modules = [
    "ra_sim.gui.analysis_peak_tools",
    "ra_sim.gui.ordered_structure_fit",
]
for name in lazy_modules:
    sys.modules.pop(name, None)
sys.modules.pop("ra_sim.gui._runtime.runtime_session", None)

runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
after_import = {name: (name in sys.modules) for name in lazy_modules}

runtime_session.analysis_peak_selection_state = SimpleNamespace(
    radial_fit_results=[],
    azimuth_fit_results=[],
)
runtime_session._ANALYSIS_PEAK_EMPTY_RESULTS_TEXT = "Fit results will appear here."
original_peak_getter = runtime_session._get_analysis_peak_tools_module
runtime_session._get_analysis_peak_tools_module = (
    lambda: (_ for _ in ()).throw(AssertionError("peak tools imported too early"))
)
empty_text = runtime_session._analysis_peak_fit_results_text()
runtime_session._get_analysis_peak_tools_module = original_peak_getter
after_empty_text = {name: (name in sys.modules) for name in lazy_modules}

peak_module = runtime_session._get_analysis_peak_tools_module()
ordered_module = runtime_session._get_ordered_structure_fit_module()
after_getters = {name: (name in sys.modules) for name in lazy_modules}

print(
    json.dumps(
        {
            "after_import": after_import,
            "empty_text": empty_text,
            "after_empty_text": after_empty_text,
            "peak_module": getattr(peak_module, "__name__", ""),
            "ordered_module": getattr(ordered_module, "__name__", ""),
            "after_getters": after_getters,
        }
    )
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )
    payload = json.loads(result.stdout.strip())

    assert payload["after_import"] == {
        "ra_sim.gui.analysis_peak_tools": False,
        "ra_sim.gui.ordered_structure_fit": False,
    }
    assert payload["empty_text"] == "Fit results will appear here."
    assert payload["after_empty_text"] == {
        "ra_sim.gui.analysis_peak_tools": False,
        "ra_sim.gui.ordered_structure_fit": False,
    }
    assert payload["peak_module"] == "ra_sim.gui.analysis_peak_tools"
    assert payload["ordered_module"] == "ra_sim.gui.ordered_structure_fit"
    assert payload["after_getters"] == {
        "ra_sim.gui.analysis_peak_tools": True,
        "ra_sim.gui.ordered_structure_fit": True,
    }


def test_shutdown_gui_clears_geometry_fit_workers_and_pending_tokens(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    cleared_tokens: list[object] = []
    closed_popouts: list[object] = []

    class _FakeFuture:
        def __init__(self) -> None:
            self.cancel_calls = 0

        def cancel(self) -> bool:
            self.cancel_calls += 1
            return True

    class _FakeExecutor:
        def __init__(self) -> None:
            self.shutdown_calls: list[tuple[bool, bool]] = []

        def shutdown(
            self,
            *,
            wait: bool = False,
            cancel_futures: bool = False,
        ) -> None:
            self.shutdown_calls.append((bool(wait), bool(cancel_futures)))

    class _FakeRoot:
        def __init__(self) -> None:
            self.destroy_calls = 0
            self.quit_calls = 0

        def winfo_exists(self) -> bool:
            return True

        def destroy(self) -> None:
            self.destroy_calls += 1

        def quit(self) -> None:
            self.quit_calls += 1

    worker_future = _FakeFuture()
    analysis_future = _FakeFuture()
    geometry_fit_future = _FakeFuture()
    worker_executor = _FakeExecutor()
    analysis_executor = _FakeExecutor()
    geometry_fit_executor = _FakeExecutor()
    fake_root = _FakeRoot()

    monkeypatch.setattr(
        runtime_session, "_append_runtime_update_trace", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(runtime_session, "root", fake_root, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "gui_controllers",
        SimpleNamespace(clear_tk_after_token=lambda _root, token: cleared_tokens.append(token)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(close_analysis_popout_window=lambda state: closed_popouts.append(state)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "tk",
        SimpleNamespace(TclError=RuntimeError),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_popout_view_state",
        "analysis-popout",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            update_phase="running",
            update_running=True,
            integration_update_pending="integration-token",
            update_pending="update-token",
            worker_poll_token="worker-poll-token",
            analysis_poll_token="analysis-poll-token",
            interaction_settle_token="settle-token",
            first_visible_simulation_settle_token="first-visible-settle-token",
            geometry_fit_poll_token="geometry-fit-poll-token",
            worker_executor=worker_executor,
            worker_future=worker_future,
            worker_active_job={"job_id": 1},
            worker_queued_job={"job_id": 2},
            worker_ready_result={"status": "ready"},
            analysis_executor=analysis_executor,
            analysis_future=analysis_future,
            analysis_active_job={"job_id": 3},
            analysis_queued_job={"job_id": 4},
            analysis_ready_result={"status": "ready"},
            geometry_fit_executor=geometry_fit_executor,
            geometry_fit_future=geometry_fit_future,
            geometry_fit_active_job={"job_id": 5},
            geometry_fit_ready_result={"status": "ready"},
            geometry_fit_error_text="stale-error",
        ),
        raising=False,
    )

    runtime_session._shutdown_gui()

    assert cleared_tokens == [
        "integration-token",
        "update-token",
        "worker-poll-token",
        "analysis-poll-token",
        "settle-token",
        "first-visible-settle-token",
        "geometry-fit-poll-token",
    ]
    assert worker_future.cancel_calls == 1
    assert analysis_future.cancel_calls == 1
    assert geometry_fit_future.cancel_calls == 1
    assert worker_executor.shutdown_calls == [(False, True)]
    assert analysis_executor.shutdown_calls == [(False, True)]
    assert geometry_fit_executor.shutdown_calls == [(False, True)]
    assert closed_popouts == ["analysis-popout"]
    assert fake_root.destroy_calls == 1
    assert fake_root.quit_calls == 1
    assert runtime_session.simulation_runtime_state.integration_update_pending is None
    assert runtime_session.simulation_runtime_state.update_pending is None
    assert runtime_session.simulation_runtime_state.worker_poll_token is None
    assert runtime_session.simulation_runtime_state.analysis_poll_token is None
    assert runtime_session.simulation_runtime_state.interaction_settle_token is None
    assert runtime_session.simulation_runtime_state.geometry_fit_poll_token is None
    assert runtime_session.simulation_runtime_state.worker_executor is None
    assert runtime_session.simulation_runtime_state.analysis_executor is None
    assert runtime_session.simulation_runtime_state.geometry_fit_executor is None
    assert runtime_session.simulation_runtime_state.worker_future is None
    assert runtime_session.simulation_runtime_state.analysis_future is None
    assert runtime_session.simulation_runtime_state.geometry_fit_future is None
    assert runtime_session.simulation_runtime_state.geometry_fit_active_job is None
    assert runtime_session.simulation_runtime_state.geometry_fit_ready_result is None
    assert runtime_session.simulation_runtime_state.geometry_fit_error_text is None


def test_gui_worker_executors_use_detached_daemon_threads(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    executors: list[object] = []
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            worker_executor=None,
            analysis_executor=None,
            geometry_fit_executor=None,
        ),
        raising=False,
    )

    try:
        for helper_name, state_attr in (
            ("_ensure_simulation_worker_executor", "worker_executor"),
            ("_ensure_analysis_worker_executor", "analysis_executor"),
            ("_ensure_geometry_fit_worker_executor", "geometry_fit_executor"),
        ):
            executor = getattr(runtime_session, helper_name)()
            executors.append(executor)
            daemon, registered = executor.submit(
                lambda: (
                    bool(threading.current_thread().daemon),
                    bool(threading.current_thread() in futures_thread._threads_queues),
                )
            ).result(timeout=5.0)
            assert daemon is True
            assert registered is False
            assert getattr(runtime_session.simulation_runtime_state, state_attr) is executor
    finally:
        for executor in executors:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=True)


def test_shutdown_gui_does_not_wait_on_nested_simulation_and_fit_thread_pools(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    exact_cake = importlib.import_module("ra_sim.simulation.exact_cake")
    optimization_runtime = importlib.import_module("ra_sim.fitting.optimization_runtime")
    cleared_tokens: list[object] = []
    simulation_inner_threads: list[tuple[bool, bool]] = []
    geometry_fit_inner_threads: list[tuple[bool, bool]] = []
    simulation_started = threading.Event()
    geometry_fit_started = threading.Event()
    release_inner_workers = threading.Event()

    class _FakeRoot:
        def __init__(self) -> None:
            self.destroy_calls = 0
            self.quit_calls = 0

        def winfo_exists(self) -> bool:
            return True

        def destroy(self) -> None:
            self.destroy_calls += 1

        def quit(self) -> None:
            self.quit_calls += 1

    def _thread_flags() -> tuple[bool, bool]:
        current = threading.current_thread()
        return bool(current.daemon), bool(current in futures_thread._threads_queues)

    def _wait_for_release(started: threading.Event) -> None:
        started.set()
        if not release_inner_workers.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to release nested GUI worker")

    def _fake_run_chunk_numba(
        signal,
        normalization,
        mask,
        has_mask,
        row_edges,
        col_edges,
        distance,
        radial,
        azimuthal,
        rows,
        cols,
        start,
        stop,
        use_selection,
    ):
        out_shape = (int(np.asarray(azimuthal).size), int(np.asarray(radial).size))
        if int(start) != int(stop):
            simulation_inner_threads.append(_thread_flags())
            _wait_for_release(simulation_started)
        zeros = np.zeros(out_shape, dtype=np.float64)
        return zeros, zeros.copy(), zeros.copy()

    def _geometry_fit_inner(item: object) -> object:
        geometry_fit_inner_threads.append(_thread_flags())
        _wait_for_release(geometry_fit_started)
        return item

    fake_root = _FakeRoot()
    state = SimpleNamespace(
        update_phase="running",
        update_running=True,
        integration_update_pending="integration-token",
        update_pending="update-token",
        worker_poll_token="worker-poll-token",
        analysis_poll_token=None,
        interaction_settle_token="settle-token",
        geometry_fit_poll_token="geometry-fit-poll-token",
        worker_executor=None,
        worker_future=None,
        worker_active_job={"job_id": 1},
        worker_queued_job=None,
        worker_ready_result=None,
        analysis_executor=None,
        analysis_future=None,
        analysis_active_job=None,
        analysis_queued_job=None,
        analysis_ready_result=None,
        geometry_fit_executor=None,
        geometry_fit_future=None,
        geometry_fit_active_job={"job_id": 2},
        geometry_fit_ready_result=None,
        geometry_fit_error_text=None,
    )

    monkeypatch.setattr(
        runtime_session, "_append_runtime_update_trace", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(runtime_session, "root", fake_root, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "gui_controllers",
        SimpleNamespace(clear_tk_after_token=lambda _root, token: cleared_tokens.append(token)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(close_analysis_popout_window=lambda _state: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "tk",
        SimpleNamespace(TclError=RuntimeError),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "analysis_popout_view_state", None, raising=False)
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", state, raising=False)
    monkeypatch.setattr(exact_cake, "_run_chunk_numba", _fake_run_chunk_numba)

    worker_executor = runtime_session._ensure_simulation_worker_executor()
    geometry_fit_executor = runtime_session._ensure_geometry_fit_worker_executor()
    worker_future = worker_executor.submit(
        exact_cake._run_numba,
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.zeros((1, 1), dtype=np.int8),
        False,
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        1.0,
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        False,
        2,
    )
    geometry_fit_future = geometry_fit_executor.submit(
        optimization_runtime.threaded_map,
        _geometry_fit_inner,
        [0, 1],
        max_workers=2,
    )
    state.worker_future = worker_future
    state.geometry_fit_future = geometry_fit_future

    try:
        assert simulation_started.wait(timeout=5.0)
        assert geometry_fit_started.wait(timeout=5.0)

        runtime_session._shutdown_gui()

        assert simulation_inner_threads
        assert geometry_fit_inner_threads
        assert all(daemon and not registered for daemon, registered in simulation_inner_threads)
        assert all(daemon and not registered for daemon, registered in geometry_fit_inner_threads)
        assert fake_root.destroy_calls == 1
        assert fake_root.quit_calls == 1
        assert state.worker_executor is None
        assert state.worker_future is None
        assert state.geometry_fit_executor is None
        assert state.geometry_fit_future is None
    finally:
        release_inner_workers.set()
        worker_future.result(timeout=5.0)
        geometry_fit_future.result(timeout=5.0)


def test_runtime_session_defers_top_level_pycifrw_and_structure_model_imports() -> None:
    import_targets = _top_level_import_targets(RUNTIME_SESSION_SOURCE_PATH)

    assert "CifFile" not in import_targets
    assert "ra_sim.gui:structure_model" not in import_targets


def test_structure_model_defers_top_level_optional_dependency_imports() -> None:
    import_targets = _top_level_import_targets(STRUCTURE_MODEL_SOURCE_PATH)

    assert "CifFile" not in import_targets
    assert "Dans_Diffraction" not in import_targets


def test_runtime_session_uses_lazy_structure_model_and_cif_reader_helpers() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "_STRUCTURE_MODEL_MODULE = None" in source
    assert "def _get_structure_model_module():" in source
    assert "gui_structure_model = _LazyModuleProxy(_get_structure_model_module)" in source

    cif_helper_start = source.index("def _read_cif_block(cif_path):")
    cif_helper_end = source.index("def _normalize_ordered_structure_scale(", cif_helper_start)
    cif_helper_block = source[cif_helper_start:cif_helper_end]
    assert "import CifFile" in cif_helper_block
    assert "CifFile.ReadCif" in cif_helper_block

    block_01_start = source.index("def _initialize_runtime_state_block_01() -> None:")
    block_01_end = source.index("def _normalize_occupancy_label(", block_01_start)
    block_01 = source[block_01_start:block_01_end]
    assert "cf, blk = _read_cif_block(cif_file)" in block_01
    assert "CifFile.ReadCif" not in block_01

    block_05_start = source.index("def _initialize_runtime_state_block_05() -> None:")
    block_05_end = source.index("def _initialize_runtime_state_block_06() -> None:", block_05_start)
    block_05 = source[block_05_start:block_05_end]
    assert "cf2, blk2 = _read_cif_block(cif_file2)" in block_05
    assert "CifFile.ReadCif" not in block_05


def test_primary_cif_import_forces_full_runtime_invalidation() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    block_start = source.index("def _apply_primary_cif_path(raw_path):")
    block_end = source.index("def _apply_secondary_cif_path(raw_path):", block_start)
    block = source[block_start:block_end]

    rebuild_index = block.index("_rebuild_diffraction_inputs(")
    full_invalidation_index = block.index("_invalidate_for_update_action(")
    schedule_index = block.index("_invalidate_and_schedule_update()")

    assert "trigger_update=False" in block
    assert "UpdateAction.FULL_SIMULATION" in block
    assert "physics_signature_changed=True" in block
    assert "hit_table_signature_changed=True" in block
    assert "q_group_content_signature_changed=True" in block
    assert rebuild_index < full_invalidation_index < schedule_index


def test_runtime_impl_prompts_from_root_only_before_full_runtime_bootstrap() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    main_start = source.index(
        'def main(write_excel_flag=None, startup_mode="prompt", calibrant_bundle=None):'
    )
    prompt_line = "resolved_mode = gui_bootstrap.choose_startup_mode_dialog(root)"
    prompt_index = source.index(prompt_line, main_start)
    ensure_root_index = source.index("ensure_runtime_root_initialized()", main_start)

    assert ensure_root_index < prompt_index


def test_runtime_impl_runtime_context_builders_gate_staged_initializers() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    state_start = source.index("def build_runtime_state_context() -> RuntimeContext:")
    window_start = source.index(
        "def build_runtime_window_context(context: RuntimeContext) -> RuntimeContext:"
    )
    plot_start = source.index(
        "def build_runtime_plot_context(context: RuntimeContext) -> RuntimeContext:"
    )
    controls_start = source.index(
        "def build_runtime_controls_context(context: RuntimeContext) -> RuntimeContext:"
    )
    benchmark_start = source.index("_STARTUP_BENCHMARK_ENABLED = str(")

    state_block = source[state_start:window_start]
    window_block = source[window_start:plot_start]
    plot_block = source[plot_start:controls_start]
    controls_block = source[controls_start:benchmark_start]

    assert "ensure_runtime_state_initialized()" in state_block
    assert "ensure_runtime_shell_initialized()" in window_block
    assert "ensure_runtime_plot_initialized()" in plot_block
    assert "ensure_runtime_controls_initialized()" in controls_block


def test_runtime_impl_keeps_default_bound_constants_eager() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert source.index('QR_CYLINDER_DISPLAY_MODE_OFF = "Off"') < source.index(
        "def _qr_cylinder_display_mode("
    )
    assert source.index("DEFAULT_ANALYSIS_RADIAL_BINS = 1000") < source.index("def caking(")
    assert source.index("INITIAL_PREVIEW_MAX_SAMPLES = 24") < source.index(
        "def _build_preview_simulation_job("
    )


def test_runtime_impl_gates_raster_projection_helpers_until_controls_stage() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _sync_primary_raster_geometry(")
    helper_end = source.index("def _maybe_refresh_run_status_bar()", helper_start)
    helper_block = source[helper_start:helper_end]

    assert '_RUNTIME_CONTROLS_INITIALIZED", False' in helper_block
    assert "if controls_ready" in helper_block
    assert "apply_projection(artist)" in helper_block


def test_runtime_impl_removes_fast_viewer_and_viewport_selector_symbols() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "RA_SIM_FAST_VIEWER" not in source
    assert "RA_SIM_PRIMARY_VIEWPORT" not in source
    assert "fast_viewer_workflow" not in source
    assert "_fast_viewer_active" not in source
    assert "runtime_display_acceleration" not in source
    assert "fast_plot_viewer" not in source
    assert "runtime_primary_viewport" not in source
    assert "tk_canvas" not in source


def test_runtime_impl_uses_tiny_startup_placeholder_rasters() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "global_image_buffer = np.zeros((1, 1), dtype=np.float32)" in source
    assert "np.zeros_like(global_image_buffer)" in source
    assert "_ensure_global_image_buffer_shape(simulation_runtime_state.unscaled_image)" in source


def test_runtime_impl_expands_global_image_buffer_for_first_real_frame() -> None:
    resize_buffer = _load_runtime_session_function("_ensure_global_image_buffer_shape")
    resize_buffer.__globals__["global_image_buffer"] = np.zeros((1, 1), dtype=np.float32)
    source = np.arange(12, dtype=np.float64).reshape(3, 4)

    resized = resize_buffer(source)

    assert resized.shape == source.shape
    assert resized.dtype == np.float64
    assert resize_buffer.__globals__["global_image_buffer"] is resized
    assert resize_buffer(source) is resized


def test_runtime_impl_lazy_allocates_job_result_images() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    block_start = source.index("def _run_simulation_generation_job(")
    block_end = source.index("def _submit_async_simulation_job(", block_start)
    block = source[block_start:block_end]

    primary_run_index = block.index('if bool(job["run_primary"]):')
    secondary_run_index = block.index('if bool(job["run_secondary"]):')
    img1_fallback_index = block.index("if img1 is None:")
    img2_fallback_index = block.index("if img2 is None:")

    assert block.index("img1 = None") < primary_run_index < img1_fallback_index
    assert block.index("img2 = None") < secondary_run_index < img2_fallback_index
    assert "if img1 is None:" in block
    assert "if img2 is None:" in block


def test_runtime_impl_routes_legacy_fit_logs_through_debug_controls() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert 'os.environ.setdefault("RA_SIM_DEBUG", "0")' not in source
    assert "if gui_geometry_fit.geometry_fit_log_files_enabled():" in source
    assert "if mosaic_fit_log_files_enabled():" in source
    assert "log_file={log_path if log_path is not None else 'disabled'}" in source


def test_runtime_impl_uses_fast_exact_cake_integrator_for_analysis() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert (
        "from ra_sim.simulation.exact_cake import start_exact_cake_numba_warmup_in_background"
        in source
    )
    assert "start_exact_cake_geometry_warmup_in_background" in source
    assert "from ra_sim.simulation.exact_cake_portable import (" in source
    assert "FastAzimuthalIntegrator," in source
    assert "start_exact_cake_geometry_warmup_in_background," in source
    assert "_AZIMUTHAL_INTEGRATOR_CLS = FastAzimuthalIntegrator" in source
    assert "start_forward_simulation_numba_warmup_in_background" in source
    assert "start_qr_rod_simulation_numba_warmup_in_background" in source
    assert "process_peaks_parallel_safe," in source
    assert "def _process_peaks_parallel_safe_prefer_python_runner(*args, **kwargs):" in source
    assert 'kwargs.setdefault("prefer_python_runner", True)' in source
    assert "def _schedule_exact_cake_numba_warmup_once() -> None:" in source
    assert "def _schedule_forward_simulation_numba_warmup_once() -> None:" in source
    assert "def _schedule_qr_rod_simulation_numba_warmup_once() -> None:" in source
    assert "simulation_runtime_state.exact_cake_numba_warmup_scheduled" in source
    assert "simulation_runtime_state.forward_simulation_numba_warmup_scheduled" in source
    assert "simulation_runtime_state.qr_rod_simulation_numba_warmup_scheduled" in source
    assert "process_peaks_parallel=_process_peaks_parallel_safe_prefer_python_runner," in source
    assert source.count("process_peaks_parallel=process_peaks_parallel_safe,") == 1
    assert "source_entry=resolved_source_entry," in source
    assert "source_entry=source_entry," in source
    assert "source_entry=raw_record," in source
    assert "root.after_idle(start_exact_cake_numba_warmup_in_background)" in source
    assert "root.after_idle(start_forward_simulation_numba_warmup_in_background)" in source
    assert "root.after_idle(start_qr_rod_simulation_numba_warmup_in_background)" in source


def test_runtime_session_selected_peak_probe_runner_prefers_python(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    captured = {}

    monkeypatch.setattr(
        runtime_session,
        "process_peaks_parallel_safe",
        lambda *args, **kwargs: (
            captured.update({"args": args, "kwargs": dict(kwargs)}) or "runner-result"
        ),
        raising=False,
    )

    result = runtime_session._process_peaks_parallel_safe_prefer_python_runner(
        "sentinel",
        sample="value",
    )

    assert result == "runner-result"
    assert captured["args"] == ("sentinel",)
    assert captured["kwargs"]["sample"] == "value"
    assert captured["kwargs"]["prefer_python_runner"] is True


def test_runtime_impl_exact_cake_cache_signature_uses_distance_center_pixel_and_wavelength() -> (
    None
):
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    signature_block = """sig = _caked_geometry_cache_signature(
        corto_det_up,
        center_x_up,
        center_y_up,
        pixel_size_m,
        wave_m,
    )"""

    assert "distance, detector center, pixel size, or the fundamental wavelength" in source
    assert "updates intentionally do not flush the detector-map or LUT" in source
    assert "caches so the live detector<->angle transform stays stable" in source
    assert signature_block in source
    assert "Gamma_updated" not in signature_block
    assert "gamma_updated" not in signature_block
    assert "center_x_up" in signature_block
    assert "center_y_up" in signature_block
    assert "pixel_size_m" in signature_block
    assert "wave_m" in signature_block
    assert "wavelength=wave_m," in source


def test_runtime_exact_cake_cache_signature_tracks_pixel_size() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    first = runtime_session._caked_geometry_cache_signature(0.5, 1.0, 2.0, 1.0e-4, 1.54e-10)
    second = runtime_session._caked_geometry_cache_signature(0.5, 1.0, 2.0, 2.0e-4, 1.54e-10)

    assert first != second


def test_runtime_impl_warms_live_exact_cake_geometry_cache() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "start_exact_cake_geometry_warmup_in_background(" in source


def test_runtime_impl_async_caked_hkl_cache_uses_live_detector_cache_signature() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '"intersection_cache_source_signature": (' in source
    assert "_current_combined_detector_intersection_cache_signature()" in source
    assert '"sim_caked_intersection_cache_source_signature": job.get(' in source
    assert '"intersection_cache_source_signature",' in source
    assert "_detector_intersection_cache_signature(intersection_cache)," in source


def test_runtime_impl_caked_hkl_cache_tracks_exact_cake_transform_pixels() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    start = source.index("def _prepare_caked_intersection_cache(")
    stop = source.index("def _detector_intersection_cache_signature", start)
    block = source[start:stop]

    assert "native_detector_coords_to_bundle_detector_coords(" in block
    assert "detector_pixel_to_caked_bin(" in block
    assert "detector_points_to_angles(" not in block
    assert "out[int(row_index), caked_two_theta_col] = two_theta_float" in block
    assert "out[int(row_index), caked_phi_col] = phi_float" in block


def test_runtime_impl_geometry_fit_caking_reuses_signature_by_distance_center_and_wavelength() -> (
    None
):
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "simulation_runtime_state.geometry_fit_caking_ai_cache = {}" in source
    assert "worker_geometry_fit_caking_sig" in source
    assert "requested_sig = _caked_geometry_cache_signature(" in source
    assert 'params_local.get("corto_detector", np.nan),' in source
    assert "center_value[0] if center_value.size > 0 else np.nan," in source
    assert "center_value[1] if center_value.size > 1 else np.nan," in source
    assert "pixel_size_value," in source
    assert 'params_local.get("lambda", np.nan),' in source
    assert "persistent_cache = getattr(" in source
    assert '"geometry_fit_caking_ai_cache",' in source
    assert 'persistent_cache.get("sig") == requested_sig' in source
    assert "simulation_runtime_state.geometry_fit_caking_ai_cache = {" in source
    assert "and worker_geometry_fit_caking_sig == requested_sig" in source
    assert "np.asarray(simulation_runtime_state.unscaled_image).shape" in source
    assert "npt_rad=DEFAULT_ANALYSIS_RADIAL_BINS" in source
    assert "npt_azim=DEFAULT_ANALYSIS_AZIMUTH_BINS" in source


def test_runtime_caking_does_not_call_exact_integrator(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    exact_cake = importlib.import_module("ra_sim.simulation.exact_cake")
    exact_cake_portable = importlib.import_module("ra_sim.simulation.exact_cake_portable")

    def _fail(*_args, **_kwargs):
        raise AssertionError("live caking should not call exact integration")

    monkeypatch.setattr(
        runtime_session,
        "integrate_detector_to_cake_exact",
        _fail,
        raising=False,
    )
    monkeypatch.setattr(exact_cake, "integrate_detector_to_cake_exact", _fail)
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_exact", _fail)

    calls: list[dict[str, object]] = []

    class _FakeAI:
        def integrate2d(self, data, **kwargs):
            calls.append(
                {
                    "data_shape": tuple(np.asarray(data).shape),
                    "kwargs": dict(kwargs),
                }
            )
            return "lut-result"

    result = runtime_session.caking(
        np.ones((3, 4), dtype=float),
        _FakeAI(),
        npt_rad=32,
        npt_azim=64,
        rows=np.array([0, 1], dtype=np.int32),
        cols=np.array([2, 3], dtype=np.int32),
    )

    assert result == "lut-result"
    assert len(calls) == 1
    assert calls[0]["data_shape"] == (3, 4)
    assert calls[0]["kwargs"]["npt_rad"] == 32
    assert calls[0]["kwargs"]["npt_azim"] == 64
    assert calls[0]["kwargs"]["correctSolidAngle"] is True
    assert calls[0]["kwargs"]["method"] == "lut"
    assert calls[0]["kwargs"]["unit"] == "2th_deg"
    np.testing.assert_array_equal(
        calls[0]["kwargs"]["rows"],
        np.array([0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        calls[0]["kwargs"]["cols"],
        np.array([2, 3], dtype=np.int32),
    )


def test_runtime_session_live_manual_peak_cache_update_uses_branch_aware_keys(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    left_entry = {
        "label": "left",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "display_col": 181.0,
        "display_row": 95.0,
    }
    right_entry = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    legacy_saved_right_entry = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    saved_pairs = [
        {
            **left_entry,
            "x": 181.0,
            "y": 95.0,
        },
        {
            **legacy_saved_right_entry,
            "x": 190.0,
            "y": 96.0,
        },
    ]
    peak_records = [dict(left_entry), dict(right_entry)]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(left_entry), dict(right_entry)],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }
    captured_saved_pairs: dict[int, list[dict[str, object]]] = {}
    side_effects: list[tuple[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            stored_sim_image=np.zeros((8, 8), dtype=float),
            peak_records=peak_records,
            peak_positions=peak_positions,
            peak_overlay_cache=peak_overlay_cache,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 8, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pairs_for_index",
        lambda index: [dict(entry) for entry in saved_pairs] if int(index) == 0 else [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_pairs_for_index",
        lambda index, entries: captured_saved_pairs.__setitem__(
            int(index),
            [dict(entry) for entry in entries or ()],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: side_effects.append(("invalidate", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_render_current_geometry_manual_pairs",
        lambda **kwargs: side_effects.append(("render", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "schedule_update",
        lambda: side_effects.append(("schedule", None)),
        raising=False,
    )

    runtime_session._geometry_fit_live_update_manual_peak_cache(
        {
            "live_cache_records": [
                {
                    "dataset_index": 0,
                    "source_table_index": 9,
                    "source_row_index": 0,
                    "source_reflection_index": 203,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "simulated_detector_x": 191.0,
                    "simulated_detector_y": 97.0,
                    "simulated_two_theta_deg": 30.25,
                    "simulated_phi_deg": -57.5,
                }
            ]
        }
    )

    assert peak_records[0]["display_col"] == 181.0
    assert peak_records[0]["display_row"] == 95.0
    assert peak_records[1]["display_col"] == 191.0
    assert peak_records[1]["display_row"] == 97.0
    assert peak_positions == [(181.0, 95.0), (191.0, 97.0)]
    assert peak_overlay_cache["records"][0]["display_col"] == 181.0
    assert peak_overlay_cache["records"][1]["display_col"] == 191.0
    assert peak_overlay_cache["positions"] == [(181.0, 95.0), (191.0, 97.0)]
    assert captured_saved_pairs[0][0]["source_branch_index"] == 0
    assert "refined_sim_x" not in captured_saved_pairs[0][0]
    assert captured_saved_pairs[0][1]["source_branch_index"] == 1
    assert captured_saved_pairs[0][1]["refined_sim_x"] == 191.0
    assert captured_saved_pairs[0][1]["refined_sim_y"] == 97.0
    assert captured_saved_pairs[0][1]["refined_sim_caked_x"] == 30.25
    assert captured_saved_pairs[0][1]["refined_sim_caked_y"] == -57.5
    assert side_effects == [
        ("invalidate", None),
        ("render", {"update_status": False}),
        ("schedule", None),
    ]


def test_runtime_session_live_manual_peak_cache_update_matches_branchless_saved_entry_by_hkl(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    left_entry = {
        "label": "1,0,5",
        "hkl": (1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "display_col": 181.0,
        "display_row": 95.0,
    }
    right_entry = {
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    saved_pairs = [
        {
            "label": "1,0,5",
            "hkl": (1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "x": 181.0,
            "y": 95.0,
        },
        {
            "label": "-1,0,5",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "x": 190.0,
            "y": 96.0,
        },
    ]
    peak_records = [dict(left_entry), dict(right_entry)]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(left_entry), dict(right_entry)],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }
    captured_saved_pairs: dict[int, list[dict[str, object]]] = {}

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            stored_sim_image=np.zeros((8, 8), dtype=float),
            peak_records=peak_records,
            peak_positions=peak_positions,
            peak_overlay_cache=peak_overlay_cache,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 8, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pairs_for_index",
        lambda index: [dict(entry) for entry in saved_pairs] if int(index) == 0 else [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_pairs_for_index",
        lambda index, entries: captured_saved_pairs.__setitem__(
            int(index),
            [dict(entry) for entry in entries or ()],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_render_current_geometry_manual_pairs",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "schedule_update",
        lambda: None,
        raising=False,
    )

    runtime_session._geometry_fit_live_update_manual_peak_cache(
        {
            "live_cache_records": [
                {
                    "dataset_index": 0,
                    "label": "-1,0,5",
                    "hkl": (-1, 0, 5),
                    "source_table_index": 9,
                    "source_row_index": 0,
                    "source_reflection_index": 203,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "simulated_detector_x": 191.0,
                    "simulated_detector_y": 97.0,
                    "simulated_two_theta_deg": 30.25,
                    "simulated_phi_deg": -57.5,
                }
            ]
        }
    )

    assert "refined_sim_x" not in captured_saved_pairs[0][0]
    assert captured_saved_pairs[0][1]["refined_sim_x"] == 191.0
    assert captured_saved_pairs[0][1]["refined_sim_y"] == 97.0
    assert captured_saved_pairs[0][1]["refined_sim_caked_x"] == 30.25
    assert captured_saved_pairs[0][1]["refined_sim_caked_y"] == -57.5


def test_runtime_session_replace_gui_state_peak_cache_reprojects_detector_view_from_native_fields(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    simulation_runtime_state = SimpleNamespace(
        peak_records=[],
        peak_positions=[],
        peak_millers=[],
        peak_intensities=[],
        selected_peak_record={"stale": True},
        peak_overlay_cache={},
    )
    projection_calls: list[list[dict[str, object]]] = []
    invalidated: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.append(True),
        raising=False,
    )

    def _project_to_detector(records):
        projection_calls.append([dict(record) for record in records or ()])
        projected: list[dict[str, object]] = []
        for record in records or ():
            if not isinstance(record, dict):
                continue
            native_col = float(record["native_col"])
            native_row = float(record["native_row"])
            projected.append(
                {
                    **dict(record),
                    "sim_col": native_col,
                    "sim_row": native_row,
                    "display_col": native_col,
                    "display_row": native_row,
                }
            )
        return projected

    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        _project_to_detector,
        raising=False,
    )

    runtime_session._replace_gui_state_peak_cache(
        [
            {
                "display_col": 30.25,
                "display_row": -57.5,
                "native_col": 190.0,
                "native_row": 96.0,
                "hkl": [-1, 0, 5],
                "intensity": 3.0,
                "q_group_key": ["q_group", "primary", 1.0, 5],
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
            }
        ]
    )

    assert projection_calls[0][0]["display_col"] == 30.25
    assert projection_calls[0][0]["display_row"] == -57.5
    assert projection_calls[0][0]["native_col"] == 190.0
    assert projection_calls[0][0]["native_row"] == 96.0
    assert simulation_runtime_state.peak_positions == [(190.0, 96.0)]
    assert simulation_runtime_state.peak_positions_filtered is False
    assert simulation_runtime_state.peak_millers == [(-1, 0, 5)]
    assert simulation_runtime_state.peak_intensities == [3.0]
    assert simulation_runtime_state.selected_peak_record is None
    assert simulation_runtime_state.peak_records[0]["sim_col"] == 190.0
    assert simulation_runtime_state.peak_records[0]["sim_row"] == 96.0
    assert simulation_runtime_state.peak_records[0]["sim_col_raw"] == 190.0
    assert simulation_runtime_state.peak_records[0]["sim_row_raw"] == 96.0
    assert simulation_runtime_state.peak_records[0]["display_col"] == 190.0
    assert simulation_runtime_state.peak_records[0]["display_row"] == 96.0
    assert simulation_runtime_state.peak_records[0]["native_col"] == 190.0
    assert simulation_runtime_state.peak_records[0]["native_row"] == 96.0
    assert simulation_runtime_state.peak_overlay_cache["positions"] == [(190.0, 96.0)]
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["sim_col_raw"] == 190.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["sim_row_raw"] == 96.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["display_col"] == 190.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["display_row"] == 96.0
    assert simulation_runtime_state.peak_overlay_cache["peak_positions_filtered"] is False
    assert simulation_runtime_state.peak_overlay_cache["restored_from_gui_state"] is True
    assert invalidated == [True]


def test_runtime_session_replace_gui_state_peak_cache_skips_bad_rows_when_projection_raises(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    simulation_runtime_state = SimpleNamespace(
        peak_records=[{"stale": True}],
        peak_positions=[(1.0, 2.0)],
        peak_millers=[(9, 9, 9)],
        peak_intensities=[4.0],
        selected_peak_record={"stale": True},
        peak_overlay_cache={"records": [{"stale": True}], "positions": [(1.0, 2.0)]},
    )
    invalidated: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.append(True),
        raising=False,
    )

    projection_calls: list[int] = []

    def _project_per_row(records):
        record = dict(records[0])
        projection_calls.append(int(record["source_row_index"]))
        if int(record["source_row_index"]) == 1:
            raise RuntimeError("boom")
        native_col = float(record["native_col"])
        native_row = float(record["native_row"])
        return [
            {
                **record,
                "sim_col": native_col,
                "sim_row": native_row,
                "display_col": native_col,
                "display_row": native_row,
            }
        ]

    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        _project_per_row,
        raising=False,
    )

    runtime_session._replace_gui_state_peak_cache(
        [
            {
                "display_col": 30.25,
                "display_row": -57.5,
                "native_col": 190.0,
                "native_row": 96.0,
                "hkl": [-1, 0, 5],
                "intensity": 3.0,
                "q_group_key": ["q_group", "primary", 1.0, 5],
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
            },
            {
                "display_col": 31.25,
                "display_row": -56.5,
                "native_col": 191.0,
                "native_row": 97.0,
                "hkl": [-2, 0, 5],
                "intensity": 4.0,
                "q_group_key": ["q_group", "primary", 2.0, 5],
                "source_table_index": 9,
                "source_row_index": 1,
                "source_branch_index": 0,
            },
        ]
    )

    assert projection_calls == [0, 1]
    assert len(simulation_runtime_state.peak_records) == 1
    assert simulation_runtime_state.peak_positions == [(190.0, 96.0)]
    assert simulation_runtime_state.peak_positions_filtered is False
    assert simulation_runtime_state.peak_millers == [(-1, 0, 5)]
    assert simulation_runtime_state.peak_intensities == [3.0]
    assert simulation_runtime_state.selected_peak_record is None
    assert simulation_runtime_state.peak_records[0]["sim_col"] == 190.0
    assert simulation_runtime_state.peak_records[0]["sim_row"] == 96.0
    assert simulation_runtime_state.peak_records[0]["display_col"] == 190.0
    assert simulation_runtime_state.peak_records[0]["display_row"] == 96.0
    assert len(simulation_runtime_state.peak_overlay_cache["records"]) == 1
    assert simulation_runtime_state.peak_overlay_cache["positions"] == [(190.0, 96.0)]
    assert simulation_runtime_state.peak_overlay_cache["peak_positions_filtered"] is False
    assert simulation_runtime_state.peak_overlay_cache["restored_from_gui_state"] is True
    assert invalidated == [True]


def test_runtime_session_replace_gui_state_peak_cache_marks_empty_import_as_restored(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    simulation_runtime_state = SimpleNamespace(
        peak_records=[{"stale": True}],
        peak_positions=[(1.0, 2.0)],
        peak_millers=[(9, 9, 9)],
        peak_intensities=[4.0],
        selected_peak_record={"stale": True},
        peak_overlay_cache={"records": [{"stale": True}], "positions": [(1.0, 2.0)]},
    )
    invalidated: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.append(True),
        raising=False,
    )

    runtime_session._replace_gui_state_peak_cache([])

    expected_sig = runtime_session.gui_peak_selection._peak_overlay_restored_view_signature(
        simulation_runtime_state,
        show_caked=False,
    )
    assert simulation_runtime_state.peak_records == []
    assert simulation_runtime_state.peak_positions == []
    assert simulation_runtime_state.peak_positions_filtered is False
    assert simulation_runtime_state.peak_millers == []
    assert simulation_runtime_state.peak_intensities == []
    assert simulation_runtime_state.selected_peak_record is None
    assert simulation_runtime_state.peak_overlay_cache["positions"] == []
    assert simulation_runtime_state.peak_overlay_cache["records"] == []
    assert simulation_runtime_state.peak_overlay_cache["peak_positions_filtered"] is False
    assert simulation_runtime_state.peak_overlay_cache["restored_from_gui_state"] is True
    assert simulation_runtime_state.peak_overlay_cache["restored_view_sig"] == expected_sig
    assert invalidated == [True]


def test_runtime_session_replace_gui_state_peak_cache_uses_caked_active_coords(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    simulation_runtime_state = SimpleNamespace(
        peak_records=[],
        peak_positions=[],
        peak_millers=[],
        peak_intensities=[],
        selected_peak_record={"stale": True},
        peak_overlay_cache={},
    )
    invalidated: list[bool] = []
    projection_inputs: list[list[dict[str, object]]] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.append(True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "filter_enabled_q_group_rows",
        lambda rows, _state: [dict(entry) for entry in rows or () if isinstance(entry, dict)],
    )
    monkeypatch.setattr(
        runtime_session,
        "_active_caked_primary_view",
        lambda: True,
        raising=False,
    )

    def _project_to_caked(records):
        projection_inputs.append(
            [dict(record) for record in records or () if isinstance(record, dict)]
        )
        projected: list[dict[str, object]] = []
        for record in records or ():
            if not isinstance(record, dict):
                continue
            projected.append(
                {
                    **dict(record),
                    "sim_col_raw": 100.0,
                    "sim_row_raw": 200.0,
                    "sim_col": 100.0,
                    "sim_row": 200.0,
                    "display_col": 5.5,
                    "display_row": 12.25,
                    "caked_x": 5.5,
                    "caked_y": 12.25,
                }
            )
        return projected

    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        _project_to_caked,
        raising=False,
    )

    runtime_session._replace_gui_state_peak_cache(
        [
            {
                "sim_col": 100.0,
                "sim_row": 200.0,
                "display_col": 5.5,
                "display_row": 12.25,
                "caked_x": 5.5,
                "caked_y": 12.25,
                "native_col": 100.0,
                "native_row": 200.0,
                "hkl": [-1, 0, 5],
                "intensity": 7.0,
                "q_group_key": ["q_group", "primary", 1.0, 5],
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
                "source_peak_index": 1,
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
            }
        ]
    )

    expected_sig = runtime_session.gui_peak_selection._peak_overlay_restored_view_signature(
        simulation_runtime_state,
        show_caked=True,
    )
    assert simulation_runtime_state.peak_positions == [(5.5, 12.25)]
    assert simulation_runtime_state.peak_positions_filtered is False
    assert simulation_runtime_state.peak_millers == [(-1, 0, 5)]
    assert simulation_runtime_state.peak_intensities == [7.0]
    assert simulation_runtime_state.selected_peak_record is None
    assert projection_inputs == [
        [
            {
                "sim_col": 100.0,
                "sim_row": 200.0,
                "display_col": 5.5,
                "display_row": 12.25,
                "caked_x": 5.5,
                "caked_y": 12.25,
                "native_col": 100.0,
                "native_row": 200.0,
                "hkl": (-1, 0, 5),
                "intensity": 7.0,
                "weight": 7.0,
                "q_group_key": ("q_group", "primary", 1.0, 5),
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
                "source_peak_index": 1,
            }
        ]
    ]
    assert simulation_runtime_state.peak_records[0]["sim_col_raw"] == 100.0
    assert simulation_runtime_state.peak_records[0]["sim_row_raw"] == 200.0
    assert simulation_runtime_state.peak_records[0]["sim_col"] == 100.0
    assert simulation_runtime_state.peak_records[0]["sim_row"] == 200.0
    assert simulation_runtime_state.peak_records[0]["display_col"] == 5.5
    assert simulation_runtime_state.peak_records[0]["display_row"] == 12.25
    assert simulation_runtime_state.peak_records[0]["native_col"] == 100.0
    assert simulation_runtime_state.peak_records[0]["native_row"] == 200.0
    assert "source_reflection_index" not in simulation_runtime_state.peak_records[0]
    assert "source_reflection_namespace" not in simulation_runtime_state.peak_records[0]
    assert "source_reflection_is_full" not in simulation_runtime_state.peak_records[0]
    assert simulation_runtime_state.peak_overlay_cache["positions"] == [(5.5, 12.25)]
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["sim_col_raw"] == 100.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["sim_row_raw"] == 200.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["sim_col"] == 100.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["sim_row"] == 200.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["display_col"] == 5.5
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["display_row"] == 12.25
    assert (
        "source_reflection_index" not in simulation_runtime_state.peak_overlay_cache["records"][0]
    )
    assert (
        "source_reflection_namespace"
        not in simulation_runtime_state.peak_overlay_cache["records"][0]
    )
    assert (
        "source_reflection_is_full" not in simulation_runtime_state.peak_overlay_cache["records"][0]
    )
    assert simulation_runtime_state.peak_overlay_cache["peak_positions_filtered"] is False
    assert simulation_runtime_state.peak_overlay_cache["restored_from_gui_state"] is True
    assert simulation_runtime_state.peak_overlay_cache["restored_view_sig"] == expected_sig
    assert invalidated == [True]


def test_runtime_session_replace_gui_state_peak_cache_does_not_write_caked_display_into_sim_coords(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    simulation_runtime_state = SimpleNamespace(
        peak_records=[],
        peak_positions=[],
        peak_millers=[],
        peak_intensities=[],
        selected_peak_record={"stale": True},
        peak_overlay_cache={},
    )
    invalidated: list[bool] = []
    projection_inputs: list[list[dict[str, object]]] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.append(True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_active_caked_primary_view",
        lambda: True,
        raising=False,
    )

    def _project_to_caked_display_only(records):
        projection_inputs.append(
            [dict(record) for record in records or () if isinstance(record, dict)]
        )
        projected: list[dict[str, object]] = []
        for record in records or ():
            if not isinstance(record, dict):
                continue
            projected.append(
                {
                    **dict(record),
                    "display_col": 5.5,
                    "display_row": 12.25,
                    "caked_x": 5.5,
                    "caked_y": 12.25,
                }
            )
        return projected

    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        _project_to_caked_display_only,
        raising=False,
    )

    runtime_session._replace_gui_state_peak_cache(
        [
            {
                "sim_col": 88.0,
                "sim_row": 144.0,
                "display_col": 5.5,
                "display_row": 12.25,
                "caked_x": 5.5,
                "caked_y": 12.25,
                "hkl": [-1, 0, 5],
                "intensity": 7.0,
                "q_group_key": ["q_group", "primary", 1.0, 5],
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
                "source_peak_index": 1,
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
            }
        ]
    )

    expected_sig = runtime_session.gui_peak_selection._peak_overlay_restored_view_signature(
        simulation_runtime_state,
        show_caked=True,
    )
    assert simulation_runtime_state.peak_positions == [(5.5, 12.25)]
    assert simulation_runtime_state.peak_positions_filtered is False
    assert simulation_runtime_state.peak_millers == [(-1, 0, 5)]
    assert simulation_runtime_state.peak_intensities == [7.0]
    assert simulation_runtime_state.selected_peak_record is None
    assert projection_inputs == [
        [
            {
                "sim_col": 88.0,
                "sim_row": 144.0,
                "display_col": 5.5,
                "display_row": 12.25,
                "caked_x": 5.5,
                "caked_y": 12.25,
                "hkl": (-1, 0, 5),
                "intensity": 7.0,
                "weight": 7.0,
                "q_group_key": ("q_group", "primary", 1.0, 5),
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
                "source_peak_index": 1,
            }
        ]
    ]
    assert simulation_runtime_state.peak_records[0]["display_col"] == 5.5
    assert simulation_runtime_state.peak_records[0]["display_row"] == 12.25
    assert "sim_col" not in simulation_runtime_state.peak_records[0]
    assert "sim_row" not in simulation_runtime_state.peak_records[0]
    assert "sim_col_raw" not in simulation_runtime_state.peak_records[0]
    assert "sim_row_raw" not in simulation_runtime_state.peak_records[0]
    assert "source_reflection_index" not in simulation_runtime_state.peak_records[0]
    assert "source_reflection_namespace" not in simulation_runtime_state.peak_records[0]
    assert "source_reflection_is_full" not in simulation_runtime_state.peak_records[0]
    assert simulation_runtime_state.peak_overlay_cache["positions"] == [(5.5, 12.25)]
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["display_col"] == 5.5
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["display_row"] == 12.25
    assert "sim_col" not in simulation_runtime_state.peak_overlay_cache["records"][0]
    assert "sim_row" not in simulation_runtime_state.peak_overlay_cache["records"][0]
    assert "sim_col_raw" not in simulation_runtime_state.peak_overlay_cache["records"][0]
    assert "sim_row_raw" not in simulation_runtime_state.peak_overlay_cache["records"][0]
    assert (
        "source_reflection_index" not in simulation_runtime_state.peak_overlay_cache["records"][0]
    )
    assert (
        "source_reflection_namespace"
        not in simulation_runtime_state.peak_overlay_cache["records"][0]
    )
    assert (
        "source_reflection_is_full" not in simulation_runtime_state.peak_overlay_cache["records"][0]
    )
    assert simulation_runtime_state.peak_overlay_cache["peak_positions_filtered"] is False
    assert simulation_runtime_state.peak_overlay_cache["restored_from_gui_state"] is True
    assert simulation_runtime_state.peak_overlay_cache["restored_view_sig"] == expected_sig
    assert invalidated == [True]


def test_runtime_session_replace_gui_state_peak_cache_rehydrates_legacy_detector_only_raw_aliases(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    simulation_runtime_state = SimpleNamespace(
        peak_records=[],
        peak_positions=[],
        peak_millers=[],
        peak_intensities=[],
        selected_peak_record={"stale": True},
        peak_overlay_cache={},
    )
    invalidated: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.append(True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_active_caked_primary_view",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        lambda records: [dict(record) for record in records or () if isinstance(record, dict)],
        raising=False,
    )

    runtime_session._replace_gui_state_peak_cache(
        [
            {
                "sim_col": 12.0,
                "sim_row": 34.0,
                "display_col": 12.0,
                "display_row": 34.0,
                "hkl": [-1, 0, 5],
                "intensity": 2.5,
            }
        ]
    )

    assert simulation_runtime_state.peak_positions == [(12.0, 34.0)]
    assert simulation_runtime_state.peak_positions_filtered is False
    assert simulation_runtime_state.peak_records[0]["sim_col"] == 12.0
    assert simulation_runtime_state.peak_records[0]["sim_row"] == 34.0
    assert simulation_runtime_state.peak_records[0]["sim_col_raw"] == 12.0
    assert simulation_runtime_state.peak_records[0]["sim_row_raw"] == 34.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["sim_col"] == 12.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["sim_row"] == 34.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["sim_col_raw"] == 12.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["sim_row_raw"] == 34.0
    assert simulation_runtime_state.peak_overlay_cache["peak_positions_filtered"] is False
    assert simulation_runtime_state.peak_overlay_cache["restored_from_gui_state"] is True
    assert invalidated == [True]


def test_runtime_session_replace_gui_state_peak_cache_skips_rows_without_safe_active_view_point(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    simulation_runtime_state = SimpleNamespace(
        peak_records=[{"stale": True}],
        peak_positions=[(1.0, 2.0)],
        peak_millers=[(9, 9, 9)],
        peak_intensities=[4.0],
        selected_peak_record={"stale": True},
        peak_overlay_cache={"records": [{"stale": True}], "positions": [(1.0, 2.0)]},
    )
    invalidated: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.append(True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_active_caked_primary_view",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        lambda records: [
            {
                **dict(record),
                "display_col": 5.5,
                "display_row": 12.25,
                "caked_x": 5.5,
                "caked_y": 12.25,
                "stale_caked_fields": True,
            }
            for record in records or ()
            if isinstance(record, dict)
        ],
        raising=False,
    )

    runtime_session._replace_gui_state_peak_cache(
        [
            {
                "sim_col": 88.0,
                "sim_row": 144.0,
                "display_col": 5.5,
                "display_row": 12.25,
                "caked_x": 5.5,
                "caked_y": 12.25,
                "hkl": [-1, 0, 5],
                "intensity": 7.0,
            }
        ]
    )

    expected_sig = runtime_session.gui_peak_selection._peak_overlay_restored_view_signature(
        simulation_runtime_state,
        show_caked=True,
    )
    assert simulation_runtime_state.peak_records == []
    assert simulation_runtime_state.peak_positions == []
    assert simulation_runtime_state.peak_millers == []
    assert simulation_runtime_state.peak_intensities == []
    assert simulation_runtime_state.selected_peak_record is None
    assert simulation_runtime_state.peak_overlay_cache["records"] == []
    assert simulation_runtime_state.peak_overlay_cache["positions"] == []
    assert simulation_runtime_state.peak_overlay_cache["restored_from_gui_state"] is True
    assert simulation_runtime_state.peak_overlay_cache["restored_view_sig"] == expected_sig
    assert invalidated == [True]


def test_runtime_session_collect_full_gui_state_snapshot_strips_peak_record_trust_fields(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    peak_records = [
        {
            "display_col": 10.0,
            "display_row": 20.0,
            "sim_col": 30.0,
            "sim_row": 40.0,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
        }
    ]

    monkeypatch.setattr(runtime_session, "_gui_state_variable_items", lambda: {}, raising=False)
    monkeypatch.setattr(runtime_session, "_occupancy_control_vars", lambda: [], raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_atom_site_fractional_control_vars",
        lambda: [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_q_group_export_rows",
        lambda: [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pairs_export_rows",
        lambda: [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_state",
        SimpleNamespace(disabled_qr_sets=set(), disabled_qz_sections=set()),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(peak_records=peak_records),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "peak_selection_state",
        SimpleNamespace(selected_hkl_target=None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_primary_cif_path",
        lambda: "primary.cif",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "structure_model_state",
        SimpleNamespace(cif_file2=None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(
            osc_files=[],
            current_background_index=0,
            visible=True,
            backend_rotation_k=0,
            backend_flip_x=False,
            backend_flip_y=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(
            background_limits_user_override=False,
            simulation_limits_user_override=False,
            scale_factor_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_roi_values",
        lambda: {"integrate_selected_qr_rod": False},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_state_io,
        "collect_full_gui_state_snapshot",
        lambda **kwargs: dict(kwargs),
    )

    snapshot = runtime_session._collect_full_gui_state_snapshot()

    assert snapshot["geometry_peak_records"] == [
        {
            "display_col": 10.0,
            "display_row": 20.0,
            "sim_col": 30.0,
            "sim_row": 40.0,
        }
    ]
    assert snapshot["analysis_range"] == {"integrate_selected_qr_rod": False}
    assert peak_records[0]["source_reflection_index"] == 203
    assert peak_records[0]["source_reflection_namespace"] == "full_reflection"
    assert peak_records[0]["source_reflection_is_full"] is True


def test_runtime_session_set_runtime_peak_cache_from_source_rows_uses_active_view_coords(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    projection_calls: list[list[dict[str, object]]] = []
    invalidated: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.append(True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "filter_enabled_q_group_rows",
        lambda rows, _state: [dict(entry) for entry in rows or () if isinstance(entry, dict)],
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_state",
        SimpleNamespace(disabled_qr_sets=set(), disabled_qz_sections=set()),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_canonicalize_live_source_entry",
        lambda entry, **_kwargs: (
            {
                **dict(entry),
                "hkl": tuple(entry.get("hkl", ())),
            }
            if isinstance(entry, dict)
            else None
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "_geometry_manual_entry_active_view_point",
        lambda entry, *, use_caked_display: (
            (
                float(entry["caked_x"]),
                float(entry["caked_y"]),
            )
            if use_caked_display and "caked_x" in entry and "caked_y" in entry
            else (
                (float(entry["raw_caked_x"]), float(entry["raw_caked_y"]))
                if use_caked_display and "raw_caked_x" in entry and "raw_caked_y" in entry
                else (
                    (float(entry["two_theta_deg"]), float(entry["phi_deg"]))
                    if use_caked_display and "two_theta_deg" in entry and "phi_deg" in entry
                    else (float(entry["sim_col"]), float(entry["sim_row"]))
                )
            )
        ),
    )

    def _project_preserving_coords(records):
        projection_calls.append([dict(record) for record in records or ()])
        return [dict(record) for record in records or () if isinstance(record, dict)]

    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        _project_preserving_coords,
        raising=False,
    )

    base_source_row = {
        "display_col": 100.0,
        "display_row": 200.0,
        "sim_col": 100.0,
        "sim_row": 200.0,
        "sim_col_raw": 100.0,
        "sim_row_raw": 200.0,
        "native_col": 100.0,
        "native_row": 200.0,
        "hkl": [-1, 0, 5],
        "q_group_key": ("q_group", "primary", 1, 5),
        "weight": 7.0,
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }

    def _make_state() -> SimpleNamespace:
        return SimpleNamespace(
            peak_records=[{"stale": True}],
            peak_positions=[(1.0, 2.0)],
            peak_millers=[(9, 9, 9)],
            peak_intensities=[4.0],
            selected_peak_record={"stale": True},
            peak_overlay_cache={"records": [{"stale": True}], "positions": [(1.0, 2.0)]},
        )

    def _restore_state(
        source_row: dict[str, object],
        *,
        use_caked_display: bool,
    ) -> SimpleNamespace:
        monkeypatch.setattr(
            runtime_session,
            "_active_caked_primary_view",
            lambda: use_caked_display,
        )
        restored_state = _make_state()
        monkeypatch.setattr(
            runtime_session,
            "simulation_runtime_state",
            restored_state,
            raising=False,
        )
        runtime_session._geometry_manual_set_runtime_peak_cache_from_source_rows([dict(source_row)])
        return restored_state

    caked_cases = (
        (
            {
                **base_source_row,
                "caked_x": 5.5,
                "caked_y": 12.25,
            },
            (5.5, 12.25),
        ),
        (
            {
                **base_source_row,
                "raw_caked_x": 6.5,
                "raw_caked_y": 14.25,
            },
            (6.5, 14.25),
        ),
        (
            {
                **base_source_row,
                "two_theta_deg": 7.5,
                "phi_deg": 16.25,
            },
            (7.5, 16.25),
        ),
    )

    for source_row, expected_position in caked_cases:
        caked_state = _restore_state(source_row, use_caked_display=True)

        assert caked_state.peak_positions == [expected_position]
        assert caked_state.peak_positions_filtered is True
        assert caked_state.peak_millers == [(-1, 0, 5)]
        assert caked_state.peak_intensities == [7.0]
        assert caked_state.selected_peak_record is None
        assert caked_state.peak_records[0]["display_col"] == 100.0
        assert caked_state.peak_records[0]["display_row"] == 200.0
        assert caked_state.peak_records[0]["sim_col"] == 100.0
        assert caked_state.peak_records[0]["sim_row"] == 200.0
        assert caked_state.peak_records[0]["sim_col_raw"] == 100.0
        assert caked_state.peak_records[0]["sim_row_raw"] == 200.0
        assert caked_state.peak_records[0]["native_col"] == 100.0
        assert caked_state.peak_records[0]["native_row"] == 200.0
        assert caked_state.peak_overlay_cache["positions"] == [expected_position]
        assert caked_state.peak_overlay_cache["records"][0]["display_col"] == 100.0
        assert caked_state.peak_overlay_cache["records"][0]["display_row"] == 200.0
        assert caked_state.peak_overlay_cache["peak_positions_filtered"] is True

    detector_state = _restore_state(
        {
            **base_source_row,
            "caked_x": 5.5,
            "caked_y": 12.25,
            "raw_caked_x": 6.5,
            "raw_caked_y": 14.25,
            "two_theta_deg": 7.5,
            "phi_deg": 16.25,
        },
        use_caked_display=False,
    )

    assert detector_state.peak_positions == [(100.0, 200.0)]
    assert detector_state.peak_positions_filtered is True
    assert detector_state.peak_millers == [(-1, 0, 5)]
    assert detector_state.peak_intensities == [7.0]
    assert detector_state.selected_peak_record is None
    assert detector_state.peak_records[0]["display_col"] == 100.0
    assert detector_state.peak_records[0]["display_row"] == 200.0
    assert detector_state.peak_records[0]["sim_col"] == 100.0
    assert detector_state.peak_records[0]["sim_row"] == 200.0
    assert detector_state.peak_records[0]["sim_col_raw"] == 100.0
    assert detector_state.peak_records[0]["sim_row_raw"] == 200.0
    assert detector_state.peak_records[0]["native_col"] == 100.0
    assert detector_state.peak_records[0]["native_row"] == 200.0
    assert detector_state.peak_overlay_cache["positions"] == [(100.0, 200.0)]
    assert detector_state.peak_overlay_cache["records"][0]["display_col"] == 100.0
    assert detector_state.peak_overlay_cache["records"][0]["display_row"] == 200.0
    assert detector_state.peak_overlay_cache["peak_positions_filtered"] is True

    assert len(projection_calls) == 4
    assert invalidated == [True, True, True, True]


def test_runtime_session_set_runtime_peak_cache_from_source_rows_falls_back_when_q_group_filter_raises(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    simulation_runtime_state = SimpleNamespace(
        peak_records=[{"stale": True}],
        peak_positions=[(1.0, 2.0)],
        peak_millers=[(9, 9, 9)],
        peak_intensities=[4.0],
        selected_peak_record={"stale": True},
        peak_overlay_cache={"records": [{"stale": True}], "positions": [(1.0, 2.0)]},
    )
    invalidated: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.append(True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "filter_enabled_q_group_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mask boom")),
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_state",
        SimpleNamespace(disabled_qr_sets=set(), disabled_qz_sections=set()),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_active_caked_primary_view",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_canonicalize_live_source_entry",
        lambda entry, **_kwargs: (
            {
                **dict(entry),
                "hkl": tuple(entry.get("hkl", ())),
            }
            if isinstance(entry, dict)
            else None
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "_geometry_manual_entry_active_view_point",
        lambda entry, *, use_caked_display: (
            float(entry["sim_col"]),
            float(entry["sim_row"]),
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        lambda records: [dict(record) for record in records or () if isinstance(record, dict)],
        raising=False,
    )

    runtime_session._geometry_manual_set_runtime_peak_cache_from_source_rows(
        [
            {
                "display_col": 100.0,
                "display_row": 200.0,
                "sim_col": 100.0,
                "sim_row": 200.0,
                "sim_col_raw": 100.0,
                "sim_row_raw": 200.0,
                "native_col": 100.0,
                "native_row": 200.0,
                "hkl": [-1, 0, 5],
                "q_group_key": ("q_group", "primary", 1, 5),
                "weight": 7.0,
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
                "source_peak_index": 1,
            }
        ]
    )

    assert simulation_runtime_state.peak_positions == [(100.0, 200.0)]
    assert simulation_runtime_state.peak_positions_filtered is True
    assert simulation_runtime_state.peak_millers == [(-1, 0, 5)]
    assert simulation_runtime_state.peak_intensities == [7.0]
    assert simulation_runtime_state.selected_peak_record is None
    assert simulation_runtime_state.peak_records[0]["sim_col"] == 100.0
    assert simulation_runtime_state.peak_records[0]["sim_row"] == 200.0
    assert simulation_runtime_state.peak_overlay_cache["positions"] == [(100.0, 200.0)]
    assert simulation_runtime_state.peak_overlay_cache["peak_positions_filtered"] is True
    assert simulation_runtime_state.peak_overlay_cache["restored_from_gui_state"] is False
    assert invalidated == [True]


def test_runtime_session_refine_manual_pair_entry_from_cache_uses_branch_aware_lookup(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    left_source = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "caked_x": 29.0,
        "caked_y": -58.5,
    }
    right_source = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "caked_x": 30.0,
        "caked_y": -57.0,
    }
    right_entry = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "caked_x": 29.5,
        "caked_y": -58.0,
    }
    peak_records = [
        {
            **left_source,
            "display_col": 181.0,
            "display_row": 95.0,
        },
        {
            **right_source,
            "display_col": 190.0,
            "display_row": 96.0,
        },
    ]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(peak_records[0]), dict(peak_records[1])],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }
    seen_source_branches: list[int] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.zeros((8, 8), dtype=float),
            last_caked_radial_values=np.linspace(10.0, 17.0, 8, dtype=float),
            last_caked_azimuth_values=np.linspace(-4.0, 3.0, 8, dtype=float),
            stored_sim_image=np.zeros((8, 8), dtype=float),
            peak_records=peak_records,
            peak_positions=peak_positions,
            peak_overlay_cache=peak_overlay_cache,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        SimpleNamespace(
            manual_pick_cache_signature=("sig",),
            manual_pick_cache_data={"old": True},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 8, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_get_geometry_manual_pick_cache",
        lambda **_kwargs: {
            "simulated_lookup": {
                runtime_session._geometry_manual_candidate_source_key(left_source): dict(
                    left_source
                ),
                runtime_session._geometry_manual_candidate_source_key(right_source): dict(
                    right_source
                ),
            },
            "match_config": {"search_radius_px": 6.0},
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 1.0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_manual_pick_background_image",
        lambda: np.zeros((8, 8), dtype=float),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_auto_match_background_context",
        lambda image, cfg: (dict(cfg), {"image_shape": tuple(np.asarray(image).shape)}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_refine_preview_point",
        lambda source_entry, *_args, **_kwargs: (
            seen_source_branches.append(int(source_entry["source_branch_index"])) or (30.25, -57.5)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_caked_angles_to_background_display_coords",
        lambda two_theta, phi: (float(two_theta), float(phi)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_background_display_to_native_detector_coords",
        lambda col, row: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )

    refined_entry = runtime_session._refine_geometry_manual_pair_entry_from_cache(dict(right_entry))

    assert seen_source_branches == [1]
    assert refined_entry is not None
    assert refined_entry["refined_sim_caked_x"] == 30.25
    assert refined_entry["refined_sim_caked_y"] == -57.5
    assert refined_entry["refined_sim_native_x"] == 30.25
    assert refined_entry["refined_sim_native_y"] == -57.5
    assert refined_entry["refined_sim_x"] == 30.25
    assert refined_entry["refined_sim_y"] == -57.5
    assert peak_records[0]["display_col"] == 181.0
    assert peak_records[0]["display_row"] == 95.0
    assert peak_records[1]["display_col"] == 190.0
    assert peak_records[1]["display_row"] == 96.0
    assert peak_positions == [(181.0, 95.0), (190.0, 96.0)]
    assert peak_overlay_cache["records"][0]["display_col"] == 181.0
    assert peak_overlay_cache["records"][1]["display_col"] == 190.0
    assert peak_overlay_cache["positions"] == [(181.0, 95.0), (190.0, 96.0)]
    assert peak_overlay_cache["click_spatial_index"] == {"position_count": 2}
    assert runtime_session.geometry_runtime_state.manual_pick_cache_signature is None
    assert runtime_session.geometry_runtime_state.manual_pick_cache_data == {}


def test_runtime_session_refine_current_manual_pairs_uses_branch_aware_lookup(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    left_entry = {
        "label": "left",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "caked_x": 29.0,
        "caked_y": -58.5,
        "x": 181.0,
        "y": 95.0,
    }
    right_entry = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "caked_x": 29.5,
        "caked_y": -58.0,
        "x": 190.0,
        "y": 96.0,
    }
    left_source_entry = {
        **left_entry,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
    }
    right_source_entry = {
        **right_entry,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
    }
    source_lookup = {
        runtime_session._geometry_manual_candidate_source_key(left_source_entry): {
            **left_source_entry,
            "caked_x": 29.0,
            "caked_y": -58.5,
        },
        runtime_session._geometry_manual_candidate_source_key(right_source_entry): {
            **right_source_entry,
            "caked_x": 30.0,
            "caked_y": -57.0,
        },
    }
    peak_records = [
        {
            **left_source_entry,
            "display_col": 181.0,
            "display_row": 95.0,
        },
        {
            **right_source_entry,
            "display_col": 190.0,
            "display_row": 96.0,
        },
    ]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(peak_records[0]), dict(peak_records[1])],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }
    saved_after_refine: dict[int, list[dict[str, object]]] = {}
    progress_messages: list[str] = []
    side_effects: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_manual_state",
        SimpleNamespace(pick_session={}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label_geometry",
        SimpleNamespace(
            config=lambda **kwargs: progress_messages.append(str(kwargs.get("text", "")))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.zeros((8, 8), dtype=float),
            last_caked_radial_values=np.linspace(10.0, 17.0, 8, dtype=float),
            last_caked_azimuth_values=np.linspace(-4.0, 3.0, 8, dtype=float),
            stored_sim_image=np.zeros((8, 8), dtype=float),
            peak_records=peak_records,
            peak_positions=peak_positions,
            peak_overlay_cache=peak_overlay_cache,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        SimpleNamespace(
            manual_pick_cache_signature=("sig",),
            manual_pick_cache_data={"old": True},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 8, raising=False)
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_pick_session_active",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pairs_for_index",
        lambda index: [dict(left_entry), dict(right_entry)] if int(index) == 0 else [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_geometry_manual_pick_cache",
        lambda **_kwargs: {
            "simulated_lookup": {
                key: ([dict(entry) for entry in value] if isinstance(value, list) else dict(value))
                for key, value in source_lookup.items()
            },
            "match_config": {"search_radius_px": 6.0},
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 1.0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_manual_pick_background_image",
        lambda: np.zeros((8, 8), dtype=float),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_auto_match_background_context",
        lambda image, cfg: (dict(cfg), {"image_shape": tuple(np.asarray(image).shape)}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_refine_preview_point",
        lambda source_entry, *_args, **_kwargs: (
            (29.0, -58.5) if int(source_entry["source_branch_index"]) == 0 else (30.25, -57.5)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_caked_angles_to_background_display_coords",
        lambda two_theta, phi: (float(two_theta), float(phi)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_background_display_to_native_detector_coords",
        lambda col, row: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_push_geometry_manual_undo_state",
        lambda: side_effects.append("undo"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_pairs_for_index",
        lambda index, entries: saved_after_refine.__setitem__(
            int(index),
            [dict(entry) for entry in entries or ()],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_geometry_fit_dataset_cache",
        lambda: side_effects.append("clear-fit-cache"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_geometry_manual_preview_artists",
        lambda **kwargs: side_effects.append(f"clear-preview:{kwargs.get('redraw')}"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_render_current_geometry_manual_pairs",
        lambda **kwargs: side_effects.append(f"render:{kwargs.get('update_status')}"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_geometry_manual_pick_button_label",
        lambda: side_effects.append("button"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_background_status",
        lambda: side_effects.append("status"),
        raising=False,
    )

    runtime_session._refine_current_geometry_manual_pairs()

    assert saved_after_refine[0][0]["source_branch_index"] == 0
    assert saved_after_refine[0][0]["refined_sim_caked_x"] == 29.0
    assert saved_after_refine[0][0]["refined_sim_caked_y"] == -58.5
    assert saved_after_refine[0][1]["source_branch_index"] == 1
    assert saved_after_refine[0][1]["refined_sim_caked_x"] == 30.25
    assert saved_after_refine[0][1]["refined_sim_caked_y"] == -57.5
    assert peak_records[0]["display_col"] == 181.0
    assert peak_records[0]["display_row"] == 95.0
    assert peak_records[1]["display_col"] == 190.0
    assert peak_records[1]["display_row"] == 96.0
    assert peak_positions == [(181.0, 95.0), (190.0, 96.0)]
    assert peak_overlay_cache["records"][0]["display_col"] == 181.0
    assert peak_overlay_cache["records"][1]["display_col"] == 190.0
    assert peak_overlay_cache["positions"] == [(181.0, 95.0), (190.0, 96.0)]
    assert peak_overlay_cache["click_spatial_index"] == {"position_count": 2}
    assert runtime_session.geometry_runtime_state.manual_pick_cache_signature is None
    assert runtime_session.geometry_runtime_state.manual_pick_cache_data == {}
    assert side_effects == [
        "undo",
        "clear-fit-cache",
        "clear-preview:False",
        "render:False",
        "button",
        "status",
    ]
    assert progress_messages[-1] == (
        "Refined 2 Qr/Qz simulation points on background 1 (1 moved, 0 skipped)."
    )


def test_runtime_session_refine_current_manual_pairs_ignores_stale_caked_coords_for_branchless_entry(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    saved_entry = {
        "label": "",
        "source_table_index": 9,
        "source_row_index": 0,
        "caked_x": 29.0,
        "caked_y": -58.5,
        "stale_caked_fields": True,
        "x": 190.0,
        "y": 96.0,
    }
    left_source_entry = {
        "label": "left",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "caked_x": 29.0,
        "caked_y": -58.5,
        "x": 181.0,
        "y": 95.0,
    }
    right_source_entry = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "caked_x": 30.0,
        "caked_y": -57.0,
        "x": 190.0,
        "y": 96.0,
    }
    source_lookup = {
        runtime_session._geometry_manual_candidate_source_key(left_source_entry): dict(
            left_source_entry
        ),
        runtime_session._geometry_manual_candidate_source_key(right_source_entry): dict(
            right_source_entry
        ),
        ("source", 9, 0): [dict(left_source_entry), dict(right_source_entry)],
        (9, 0): [dict(left_source_entry), dict(right_source_entry)],
    }
    peak_records = [
        {
            **left_source_entry,
            "display_col": 181.0,
            "display_row": 95.0,
        },
        {
            **right_source_entry,
            "display_col": 190.0,
            "display_row": 96.0,
        },
    ]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(peak_records[0]), dict(peak_records[1])],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }
    saved_after_refine: dict[int, list[dict[str, object]]] = {}
    progress_messages: list[str] = []
    side_effects: list[str] = []
    seen_source_branches: list[int] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_manual_state",
        SimpleNamespace(pick_session={}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label_geometry",
        SimpleNamespace(
            config=lambda **kwargs: progress_messages.append(str(kwargs.get("text", "")))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.zeros((8, 8), dtype=float),
            last_caked_radial_values=np.linspace(10.0, 17.0, 8, dtype=float),
            last_caked_azimuth_values=np.linspace(-4.0, 3.0, 8, dtype=float),
            stored_sim_image=np.zeros((8, 8), dtype=float),
            peak_records=peak_records,
            peak_positions=peak_positions,
            peak_overlay_cache=peak_overlay_cache,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        SimpleNamespace(
            manual_pick_cache_signature=("sig",),
            manual_pick_cache_data={"old": True},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 8, raising=False)
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_pick_session_active",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_lookup_source_entry",
        lambda _lookup, entry, **_kwargs: (
            dict(right_source_entry)
            if int(entry.get("source_table_index", -1)) == 9
            and int(entry.get("source_row_index", -1)) == 0
            else None
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pairs_for_index",
        lambda index: [dict(saved_entry)] if int(index) == 0 else [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_geometry_manual_pick_cache",
        lambda **_kwargs: {
            "simulated_lookup": {
                key: ([dict(entry) for entry in value] if isinstance(value, list) else dict(value))
                for key, value in source_lookup.items()
            },
            "match_config": {"search_radius_px": 6.0},
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 1.0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_manual_pick_background_image",
        lambda: np.zeros((8, 8), dtype=float),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_auto_match_background_context",
        lambda image, cfg: (dict(cfg), {"image_shape": tuple(np.asarray(image).shape)}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_refine_preview_point",
        lambda source_entry, *_args, **_kwargs: (
            seen_source_branches.append(int(source_entry["source_branch_index"]))
            or ((29.0, -58.5) if int(source_entry["source_branch_index"]) == 0 else (30.25, -57.5))
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_caked_angles_to_background_display_coords",
        lambda two_theta, phi: (float(two_theta), float(phi)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_background_display_to_native_detector_coords",
        lambda col, row: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_push_geometry_manual_undo_state",
        lambda: side_effects.append("undo"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_pairs_for_index",
        lambda index, entries: saved_after_refine.__setitem__(
            int(index),
            [dict(entry) for entry in entries or ()],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_geometry_fit_dataset_cache",
        lambda: side_effects.append("clear-fit-cache"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_geometry_manual_preview_artists",
        lambda **kwargs: side_effects.append(f"clear-preview:{kwargs.get('redraw')}"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_render_current_geometry_manual_pairs",
        lambda **kwargs: side_effects.append(f"render:{kwargs.get('update_status')}"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_geometry_manual_pick_button_label",
        lambda: side_effects.append("button"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_background_status",
        lambda: side_effects.append("status"),
        raising=False,
    )

    runtime_session._refine_current_geometry_manual_pairs()

    peak_records_by_branch = {
        int(record["source_branch_index"]): dict(record) for record in peak_records
    }

    assert seen_source_branches == [1]
    assert saved_after_refine[0][0]["stale_caked_fields"] is True
    assert saved_after_refine[0][0]["refined_sim_caked_x"] == 30.25
    assert saved_after_refine[0][0]["refined_sim_caked_y"] == -57.5
    assert peak_records_by_branch[0]["display_col"] == 181.0
    assert peak_records_by_branch[0]["display_row"] == 95.0
    assert peak_records_by_branch[1]["display_col"] == 190.0
    assert peak_records_by_branch[1]["display_row"] == 96.0
    assert peak_overlay_cache["records"][0]["display_col"] == 181.0
    assert peak_overlay_cache["records"][1]["display_col"] == 190.0
    assert peak_overlay_cache["positions"] == [(181.0, 95.0), (190.0, 96.0)]
    assert peak_overlay_cache["click_spatial_index"] == {"position_count": 2}
    assert runtime_session.geometry_runtime_state.manual_pick_cache_signature is None
    assert runtime_session.geometry_runtime_state.manual_pick_cache_data == {}
    assert side_effects == [
        "undo",
        "clear-fit-cache",
        "clear-preview:False",
        "render:False",
        "button",
        "status",
    ]
    assert progress_messages[-1] == (
        "Refined 1 Qr/Qz simulation points on background 1 (1 moved, 0 skipped)."
    )


def test_cli_routes_mosaic_fit_logs_through_debug_controls() -> None:
    source = CLI_SOURCE_PATH.read_text(encoding="utf-8")

    assert "from ra_sim.debug_controls import mosaic_fit_log_files_enabled" in source
    assert "if mosaic_fit_log_files_enabled():" in source
    assert '"log_path": str(mosaic_log_path) if mosaic_log_path is not None else None' in source


def test_runtime_impl_attaches_background_theta_trace_after_theta_var_assignment() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    theta_assignment = (
        "theta_initial_var = beam_mosaic_parameter_sliders_view_state.theta_initial_var"
    )
    trace_call = "_attach_live_theta_background_theta_trace(theta_initial_var)"

    assert theta_assignment in source
    assert trace_call in source
    assert source.index(theta_assignment) < source.index(trace_call)


def test_runtime_impl_uses_cached_caking_results_for_range_refreshes() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "range_refresh_requires_pending_analysis_result(" in source


def test_runtime_impl_disables_live_drag_preview_degradation() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "LIVE_DRAG_PREVIEW_ENABLED = False" in source
    assert "if _live_interaction_active() and LIVE_DRAG_PREVIEW_ENABLED:" in source
    assert "LIVE_DRAG_PREVIEW_ENABLED\n        and PREVIEW_CALCULATIONS_ENABLED" in source


def test_runtime_impl_preserves_wrapped_phi_ranges_for_detector_drags() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "draw_idle_factory=lambda: (" in source
    assert "_request_main_canvas_redraw" in source
    assert "phi_min = float(phi_min_var.get())" in source
    assert "phi_max = float(phi_max_var.get())" in source
    assert "gui_integration_range_drag.detector_phi_mask(" in source
    assert "if phi_max < phi_min and azimuth_sub.size:" in source


def test_runtime_impl_tracks_detector_source_geometry_for_projected_rasters() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '_MAIN_RASTER_SOURCE_EXTENT_ATTR = "_ra_sim_source_extent"' in source
    assert '_MAIN_RASTER_SOURCE_ORIGIN_ATTR = "_ra_sim_source_origin"' in source
    assert 'image_geometry = ("upper", detector_extent)' in source
    assert 'overlay_geometry = ("upper", detector_extent)' in source
    assert "setattr(artist, _MAIN_RASTER_SOURCE_EXTENT_ATTR, normalized_extent)" in source
    assert "setattr(artist, _MAIN_RASTER_SOURCE_ORIGIN_ATTR, str(origin))" in source
    assert "store_geometry(artist, origin=origin, extent=extent)" in source


def test_runtime_impl_overlay_refresh_copies_image_source_geometry() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "_primary_raster_source_payload(" in source
    assert "_image_source" in source
    assert "image_source_signature" in source
    assert "if image_extent is not None:" in source
    assert "origin=image_origin," in source
    assert "extent=image_extent," in source
    assert "_store_primary_raster_source(" in source
    assert "_integration_overlay_raster_source_signature(" in source
    assert "parent_source_signature=image_source_signature," in source
    assert "source_signature=overlay_source_signature," in source
    assert "_apply_projected_primary_raster_to_artist(overlay_artist)" in source


def test_runtime_impl_wires_caked_custom_mask_signature_factory() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "caked_custom_mask_signature_factory=lambda: (" in source
    assert '(_current_selected_qr_rod_caked_mask_payload() or {}).get("signature")' in source


def test_runtime_impl_wires_detector_geometry_signature_factory() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert (
        "detector_geometry_signature_factory=lambda: simulation_runtime_state.ai_cache.get("
        in source
    )


def test_runtime_impl_routes_main_figure_chrome_through_shared_helper() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "from ra_sim.gui import main_figure_chrome as gui_main_figure_chrome" in source
    assert "gui_main_figure_chrome.configure_matplotlib_canvas_widget(" in source
    assert "gui_main_figure_chrome.configure_main_figure_layout(" in source
    assert "gui_main_figure_chrome.apply_main_figure_axes_chrome(" in source
    assert "axes_visible=bool(analysis_space_display_available)" in source
    assert "axes_visible=bool(show_caked_image)" not in source
    assert "gui_main_figure_chrome.set_main_figure_axes_axis_visibility(ax, visible=True)" in source
    assert (
        "gui_main_figure_chrome.set_main_figure_axes_axis_visibility(ax, visible=False)" in source
    )
    assert "ax.set_title('Simulated Diffraction Pattern')" not in source


def test_runtime_impl_falls_back_to_detector_image_when_caked_cache_is_missing() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "active_view_mode = _resolved_primary_analysis_display_mode()" in source
    assert "show_analysis_image = show_caked_image or show_q_space_image" in source
    assert "if not show_analysis_image:" in source
    assert "_store_primary_raster_source(image_display, global_image_buffer)" in source
    assert '_sync_primary_raster_geometry(view_mode="detector")' in source


def test_runtime_impl_restores_caked_payload_when_view_returns_to_caked() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _analysis_display_payload_ready(" in source
    assert "def _analysis_payload_ready_after_restore_attempt(" in source
    assert "def _restore_caked_display_payload_from_cached_results(" in source
    assert "missing_analysis_payload = bool(" in source
    assert "and not analysis_payload_ready" in source
    assert "analysis_payload_ready = _analysis_payload_ready_after_restore_attempt(" in source
    assert "_restore_caked_display_payload_from_cached_results(" in source
    assert "caked_background_image is not None" in source
    assert "q_space_background_image is not None" in source
    assert "_get_current_background_native()" in source
    assert source.index(
        "analysis_payload_ready = _analysis_payload_ready_after_restore_attempt("
    ) < source.index("analysis_result_matches_target = bool(")
    assert source.index("analysis_result_matches_target = bool(") < source.index(
        "_request_async_analysis_job(",
        source.index("analysis_result_matches_target = bool("),
    )


def test_runtime_impl_preserves_primary_axis_limits_across_same_mode_redraws() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "previous_primary_view_mode = _current_primary_figure_mode()" in source
    assert "preserved_primary_limits = gui_canvas_interactions.capture_axis_limits(ax)" in source
    assert "gui_canvas_interactions.restore_axis_view(" in source
    assert "preserve=(previous_primary_view_mode == analysis_space_display_mode)" in source
    assert 'preserve=(previous_primary_view_mode == "detector")' in source


def test_runtime_impl_reuses_cached_hit_tables_for_optics_and_mosaic_only_updates() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _hit_table_state_present_for_run_sides(" in source
    assert "def _cached_hit_tables_reusable(" in source
    assert "include_mosaic_shape: bool = True" in source
    assert "if include_mosaic_shape:" in source
    assert "requested_hit_table_sig = _simulation_signature_base(" in source
    assert "optics_mode_component=0," in source
    assert "include_mosaic_shape=False," in source
    assert "selection_peak_cache_needed = bool(" in source
    assert "collect_hit_tables_for_job = bool(" in source
    assert "hit_tables_reusable = bool(" in source
    assert '"collected_hit_tables": bool(job["collect_hit_tables"]),' in source
    assert "request_build_intersection_cache = bool(build_intersection_cache)" in source
    assert "collect_hit_tables or request_build_intersection_cache" in source
    assert "build_intersection_cache=request_build_intersection_cache," in source
    assert '"build_primary_intersection_cache": build_primary_intersection_cache,' in source
    assert '"primary_intersection_cache_built": bool(primary_intersection_cache_built),' in source
    assert '"secondary_intersection_cache_built": bool(' in source
    assert "return _hit_table_state_present_for_run_sides(" in source
    assert "intersection_cache_to_hit_tables," in source
    assert "def _resolved_peak_table_payload(" in source
    assert "primary_peak_tables = _resolved_peak_table_payload(cache1, raw_hit_tables1)" in source
    assert '"primary_max_positions": list(primary_peak_tables),' in source
    assert "if primary_intersection_cache_built:" in source
    assert "if secondary_intersection_cache_built:" in source
    assert "stored_primary_peak_table_lattice = list(" in source
    assert "stored_secondary_peak_table_lattice = list(" in source
    assert 'or "primary_intersection_cache" in result' not in source
    assert "simulation_runtime_state.last_sim_signature = new_sim_image_sig" in source
    assert "need_hit_table_refresh = bool(" in source
    assert "do_update_refresh_hit_tables_in_background" in source
    assert 'progress_label.config(text="Refreshing peak tables in background...")' in source


def test_runtime_impl_supports_incremental_sf_prune_primary_cache_updates() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "primary_contribution_cache_signature" in source
    assert "primary_requested_contribution_keys" in source
    assert "resolve_incremental_sf_prune_action(" in source
    assert 'job_kind="primary_fill"' in source
    assert source.count("if _hit_table_state_present_for_run_sides(") >= 2
    assert "_rematerialize_primary_cache_artifacts(" in source
    assert "do_update_extend_primary_cache_in_background" in source
    assert 'text="Extending primary cache in background..."' in source


def test_runtime_impl_routes_optional_runtime_caches_through_retention_policy() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "from ra_sim.debug_controls import (" in source
    assert "retain_optional_cache," in source
    assert '"primary_contribution"' in source
    assert '"source_snapshots"' in source
    assert '"caking"' in source
    assert '"manual_pick"' in source
    assert '"geometry_fit_dataset"' in source
    assert '"peak_overlay"' in source


def test_runtime_impl_uses_geometry_manual_state_name_for_manual_pairs() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "manual_geometry_state" not in source
    assert 'globals().get("geometry_manual_state")' in source


def test_runtime_impl_keeps_manual_pick_cache_restores_cache_only() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert (
        'lookup_context == "geometry_fit_dataset" or allow_manual_pick_rebuild' in source
    )
    assert "if allow_source_snapshot_rebuild:" in source


def test_runtime_impl_worker_geometry_fit_rebuilds_source_rows_on_demand() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _prebuild_required_background_caches() -> None:")
    helper_end = source.index("worker_manual_dataset_bindings =", helper_start)
    helper_source = source[helper_start:helper_end]

    assert "def _rebuild_source_rows_for_background_worker(" in source
    assert "geometry_manual_rebuild_source_rows_for_background=(" in source
    assert "_rebuild_source_rows_for_background_worker" in source
    assert helper_source.count("raise RuntimeError(") == 2
    assert "exact caked projector unavailable" in helper_source
    assert "mixed detector/caked manual fit spaces are not supported" in helper_source


def test_runtime_impl_keeps_qr_overlay_live_during_background_updates() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _clear_deferred_overlays(*, clear_qr_overlay: bool = True) -> None:" in source
    assert "if _live_interaction_active():" in source
    assert "qr_cylinder_overlay_runtime_refresh(redraw=True, update_status=False)" in source
    assert "_clear_deferred_overlays(clear_qr_overlay=False)" in source


def test_runtime_impl_uses_roi_preview_display_sources_for_main_redraw() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "display_primary_source, display_secondary_source = (" in source
    assert "_geometry_fit_caked_roi_preview_display_sources(" in source
    assert "simulation_image=scaled_analysis_for_limits," in source
    assert "simulation_image=global_image_buffer," in source


def test_runtime_impl_hides_aux_artists_while_roi_preview_is_enabled() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "if _current_geometry_fit_caked_roi_preview_enabled():" in source
    assert "_hide_geometry_fit_caked_roi_preview_aux_artists()" in source
    assert "_request_overlay_canvas_redraw(force=True)" in source
    assert "visible and _current_geometry_fit_caked_roi_preview_enabled()" in source


def test_runtime_impl_invalidates_qr_overlay_before_view_mode_toggles() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert (
        "def _invalidate_qr_cylinder_overlay_view_state(*, clear_artists: bool) -> None:" in source
    )
    assert "gui_qr_cylinder_overlay.invalidate_runtime_qr_cylinder_overlay_cache(" in source
    assert "def toggle_caked_2d(requested_mode: str | None = None) -> None:" in source
    assert "_invalidate_qr_cylinder_overlay_view_state(clear_artists=True)" in source
    assert "def _apply_main_caked_view_toggle() -> None:" in source
    assert "_apply_main_caked_view_toggle()" in source


def test_runtime_impl_keeps_detector_and_caked_intersection_caches_separate() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    restore_start = source.index("def _restore_caked_display_payload_from_cached_results(")
    restore_end = source.index("def _run_analysis_job(", restore_start)
    restore_source = source[restore_start:restore_end]
    apply_start = source.index(
        "def _apply_ready_analysis_result(result: dict[str, object]) -> None:"
    )
    apply_end = source.index("def schedule_update():", apply_start)
    apply_source = source[apply_start:apply_end]

    assert (
        "simulation_runtime_state.last_caked_intersection_cache = caked_intersection_cache"
        in restore_source
    )
    assert (
        "simulation_runtime_state.last_caked_intersection_cache_transform_bundle = "
        in restore_source
    )
    assert (
        "simulation_runtime_state.last_caked_intersection_cache_source_signature = "
        in restore_source
    )
    assert (
        "simulation_runtime_state.stored_intersection_cache = caked_intersection_cache"
        not in restore_source
    )
    assert (
        "simulation_runtime_state.last_caked_intersection_cache = caked_intersection_cache"
        in apply_source
    )
    assert (
        "simulation_runtime_state.last_caked_intersection_cache_transform_bundle = " in apply_source
    )
    assert (
        "simulation_runtime_state.last_caked_intersection_cache_source_signature = " in apply_source
    )
    assert (
        "simulation_runtime_state.stored_intersection_cache = caked_intersection_cache"
        not in apply_source
    )
    assert "simulation_runtime_state.last_caked_transform_bundle = (" in source


def test_runtime_impl_threads_primary_raster_source_signature_into_projection() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '_MAIN_RASTER_SOURCE_SIGNATURE_ATTR = "_ra_sim_source_signature"' in source
    assert "def _detector_display_raster_source_signature() -> object | None:" in source
    assert "def _resolved_primary_raster_source_signature(" in source
    assert "_MAIN_RASTER_SOURCE_SIGNATURE_ATTR," in source
    assert "source, extent, origin, source_signature" in source
    assert "_primary_raster_source_payload(artist)" in source
    assert "source_signature=source_signature," in source


def test_runtime_session_toggle_simulation_overlay_updates_artist(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Artist:
        def __init__(self) -> None:
            self.visible = True

        def set_visible(self, value: object) -> None:
            self.visible = bool(value)

    progress_messages: list[str] = []
    redraw_calls: list[dict[str, object]] = []
    artist = _Artist()
    runtime_state = SimpleNamespace(
        simulation_overlay_visible=True,
        unscaled_image=np.ones((2, 2), dtype=float),
        stored_sim_image=None,
    )

    monkeypatch.setattr(runtime_session, "image_display", artist, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label",
        SimpleNamespace(config=lambda **kwargs: progress_messages.append(kwargs["text"])),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **kwargs: redraw_calls.append(dict(kwargs)),
        raising=False,
    )

    assert runtime_session.toggle_simulation_overlay_visibility() is False
    assert runtime_state.simulation_overlay_visible is False
    assert artist.visible is False
    assert runtime_session._timing_display_has_simulation_image() is False

    assert runtime_session.toggle_simulation_overlay_visibility() is True
    assert runtime_state.simulation_overlay_visible is True
    assert artist.visible is True
    assert runtime_session._timing_display_has_simulation_image() is True
    assert progress_messages == [
        "Simulation overlay hidden.",
        "Simulation overlay shown.",
    ]
    assert redraw_calls == [
        {"force_matplotlib": False},
        {"force_matplotlib": False},
    ]


def test_runtime_impl_preserves_no_redraw_preview_cancel_contract_for_overlay_restore() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    restore_start = source.index("def _restore_legacy_main_matplotlib_overlays(")
    restore_end = source.index("def _preview_legacy_main_matplotlib_view_limits(", restore_start)
    restore_source = source[restore_start:restore_end]
    reset_start = source.index("def _reset_main_figure_live_interaction_state(")
    configure_start = source.index("def _configure_primary_viewport_redraw_helpers()", reset_start)
    reset_source = source[reset_start:configure_start]

    assert (
        "def _restore_legacy_main_matplotlib_overlays(*, redraw: bool = True) -> bool:"
        in restore_source
    )
    assert "if bool(redraw)" in restore_source
    assert "else None" in restore_source
    assert "_request_legacy_main_matplotlib_redraw(force=True)" in restore_source
    assert (
        "def _reset_main_figure_live_interaction_state(*, redraw: bool = True) -> None:"
        in reset_source
    )
    assert "_restore_legacy_main_matplotlib_overlays(redraw=bool(redraw))" in reset_source
    assert (
        "preview_cleared = bool(_clear_legacy_main_matplotlib_preview_view(redraw=False))"
        in reset_source
    )
    assert "if dropped_preview_state and not preview_cleared:" in reset_source
    assert "_reset_main_figure_live_interaction_state(redraw=bool(redraw))" in reset_source
    assert "_reset_main_figure_live_interaction_state(redraw=False)" in reset_source


def test_apply_projected_primary_raster_to_artist_passes_stored_source_signature(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    recorded_signatures: list[object] = []
    set_data_calls: list[np.ndarray] = []

    class _Artist:
        def __init__(self) -> None:
            self._extent = None

        def set_extent(self, extent):
            self._extent = list(extent)

        def set_data(self, image):
            set_data_calls.append(np.asarray(image, dtype=float).copy())

    monkeypatch.setattr(
        runtime_session,
        "_legacy_main_matplotlib_interaction_active",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_main_display_raster_size_limit",
        lambda: 100,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_axes_image_origin",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_display_projection,
        "project_raster_to_view",
        lambda image, *, source_signature=None, **_kwargs: (
            recorded_signatures.append(source_signature)
            or runtime_session.gui_display_projection.RasterProjection(
                image=np.asarray(image, dtype=float).copy(),
                extent=(0.0, 2.0, 2.0, 0.0),
            )
        ),
        raising=False,
    )

    artist = _Artist()
    source_image = np.arange(4, dtype=float).reshape(2, 2)
    runtime_session._store_primary_raster_source(
        artist,
        source_image,
        source_signature=("sig", 7),
    )
    runtime_session._store_primary_raster_geometry(
        artist,
        origin="upper",
        extent=(0.0, 2.0, 2.0, 0.0),
    )

    assert runtime_session._apply_projected_primary_raster_to_artist(artist) is True
    assert recorded_signatures == [("sig", 7)]
    assert len(set_data_calls) == 1
    np.testing.assert_array_equal(set_data_calls[0], source_image)


def test_apply_projected_primary_raster_to_artist_detector_view_ignores_live_window_projection(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    recorded_kwargs: list[dict[str, object]] = []

    class _Artist:
        def set_extent(self, _extent):
            return None

        def set_data(self, _image):
            return None

    monkeypatch.setattr(
        runtime_session,
        "_legacy_main_matplotlib_interaction_active",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_main_display_raster_size_limit",
        lambda: 128,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_axes_image_origin",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            get_xlim=lambda: (10.0, 20.0),
            get_ylim=lambda: (30.0, 5.0),
            bbox=SimpleNamespace(width=640.0, height=480.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_display_projection,
        "project_raster_to_view",
        lambda image, **kwargs: (
            recorded_kwargs.append(dict(kwargs))
            or runtime_session.gui_display_projection.RasterProjection(
                image=np.asarray(image, dtype=float).copy(),
                extent=(0.0, 4.0, 4.0, 0.0),
            )
        ),
        raising=False,
    )

    artist = _Artist()
    runtime_session._store_primary_raster_source(artist, np.arange(16, dtype=float).reshape(4, 4))
    runtime_session._store_primary_raster_geometry(
        artist,
        origin="upper",
        extent=(0.0, 4.0, 4.0, 0.0),
    )

    assert runtime_session._apply_projected_primary_raster_to_artist(artist) is True
    assert recorded_kwargs == [
        {
            "source_signature": getattr(
                artist,
                runtime_session._MAIN_RASTER_SOURCE_SIGNATURE_ATTR,
            ),
            "extent": (0.0, 4.0, 4.0, 0.0),
            "axis_xlim": None,
            "axis_ylim": None,
            "max_size": 128,
            "bbox_width_px": None,
            "bbox_height_px": None,
            "preserve_bright_features": False,
        }
    ]


@pytest.mark.parametrize("view_mode", ["caked", "q_space"])
def test_apply_projected_primary_raster_to_artist_analysis_view_uses_live_window_projection(
    monkeypatch,
    view_mode: str,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    recorded_kwargs: list[dict[str, object]] = []

    class _Artist:
        def set_extent(self, _extent):
            return None

        def set_data(self, _image):
            return None

    monkeypatch.setattr(
        runtime_session,
        "_legacy_main_matplotlib_interaction_active",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: view_mode,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_main_display_raster_size_limit",
        lambda: 256,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_axes_image_origin",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            get_xlim=lambda: (1.0, 2.0),
            get_ylim=lambda: (-3.0, 7.0),
            bbox=SimpleNamespace(width=320.0, height=240.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_display_projection,
        "project_raster_to_view",
        lambda image, **kwargs: (
            recorded_kwargs.append(dict(kwargs))
            or runtime_session.gui_display_projection.RasterProjection(
                image=np.asarray(image, dtype=float).copy(),
                extent=(0.0, 4.0, 4.0, 0.0),
            )
        ),
        raising=False,
    )

    artist = _Artist()
    runtime_session._store_primary_raster_source(artist, np.arange(16, dtype=float).reshape(4, 4))
    runtime_session._store_primary_raster_geometry(
        artist,
        origin="lower",
        extent=(0.0, 4.0, 4.0, 0.0),
    )

    assert runtime_session._apply_projected_primary_raster_to_artist(artist) is True
    assert recorded_kwargs == [
        {
            "source_signature": getattr(
                artist,
                runtime_session._MAIN_RASTER_SOURCE_SIGNATURE_ATTR,
            ),
            "extent": (0.0, 4.0, 4.0, 0.0),
            "axis_xlim": (1.0, 2.0),
            "axis_ylim": (-3.0, 7.0),
            "max_size": 256,
            "bbox_width_px": 320.0,
            "bbox_height_px": 240.0,
            "preserve_bright_features": False,
        }
    ]


def test_store_primary_raster_source_uses_detector_buffer_signature_for_global_image_buffer(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    artist = SimpleNamespace()
    detector_buffer = np.arange(4, dtype=float).reshape(2, 2)
    detector_signature = ("detector-cache", 3)

    monkeypatch.setattr(runtime_session, "global_image_buffer", detector_buffer, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(last_unscaled_image_signature=detector_signature),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: 2.5,
        raising=False,
    )

    runtime_session._store_primary_raster_source(artist, detector_buffer)
    source, _extent, _origin, source_signature = runtime_session._primary_raster_source_payload(
        artist
    )

    assert source is detector_buffer
    assert source_signature == (detector_signature, 2.5)


def test_get_scale_factor_value_returns_default_when_scale_var_missing(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(),
        raising=False,
    )

    assert runtime_session._get_scale_factor_value(default=1.25) == 1.25


def test_set_primary_integration_overlay_image_uses_distinct_overlay_source_signature(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    image_artist = SimpleNamespace()
    overlay_artist = SimpleNamespace()
    parent_signature = ("image", 5)
    explicit_overlay_signature = ("raw_drag_preview", (2, 2), 3)
    overlay_image = np.full((2, 2), 7.0, dtype=float)
    applied_artists: list[object] = []

    monkeypatch.setattr(runtime_session, "image_display", image_artist, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "integration_region_overlay",
        overlay_artist,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_projected_primary_raster_to_artist",
        lambda artist: applied_artists.append(artist) or True,
        raising=False,
    )

    runtime_session._store_primary_raster_source(
        image_artist,
        np.arange(4, dtype=float).reshape(2, 2),
        source_signature=parent_signature,
    )
    runtime_session._store_primary_raster_geometry(
        image_artist,
        origin="upper",
        extent=(0.0, 1.0, 1.0, 0.0),
    )

    runtime_session._set_primary_integration_overlay_image(
        overlay_image,
        source_signature=explicit_overlay_signature,
    )
    stored_source, extent, origin, overlay_signature = (
        runtime_session._primary_raster_source_payload(overlay_artist)
    )

    np.testing.assert_array_equal(stored_source, overlay_image)
    assert tuple(extent) == (0.0, 1.0, 1.0, 0.0)
    assert origin == "upper"
    assert overlay_signature != parent_signature
    assert overlay_signature[0] == "integration_region_overlay"
    assert overlay_signature[1] == parent_signature
    assert overlay_signature[2] == explicit_overlay_signature
    assert applied_artists == [overlay_artist]


def test_integration_overlay_raster_source_signature_prefers_explicit_overlay_signature() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    parent_signature = ("image", 5)
    explicit_overlay_signature = ("raw_drag_preview", (2, 2), 4)
    overlay_image = np.full((2, 2), 7.0, dtype=float)

    overlay_signature = runtime_session._integration_overlay_raster_source_signature(
        parent_source_signature=parent_signature,
        overlay_source=overlay_image,
        overlay_source_signature=explicit_overlay_signature,
    )

    assert overlay_signature == (
        "integration_region_overlay",
        parent_signature,
        explicit_overlay_signature,
    )


def test_runtime_live_caked_projection_helper_uses_live_bundle_directly(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    bundle = object()

    def _fail(*_args, **_kwargs):
        raise AssertionError("live caked helper should not use bound callback")

    projection_calls: list[tuple[object, float, float]] = []

    def _record(bundle_arg: object, col: float, row: float):
        projection_calls.append((bundle_arg, float(col), float(row)))
        return (12.0, 34.0)

    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_caked_display_coords",
        _fail,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_live_caked_transform_bundle",
        lambda: bundle,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        _record,
    )

    assert runtime_session._native_detector_coords_to_live_caked_coords(1.25, 2.5) == (
        12.0,
        34.0,
    )
    assert projection_calls == [(bundle, 1.25, 2.5)]


def _install_preview_mask_legacy_helper_guards(monkeypatch, runtime_session) -> None:
    def _fail(*_args, **_kwargs):
        raise AssertionError("preview mask should not call legacy analytic helpers")

    monkeypatch.setattr(runtime_session, "_detector_angular_maps_for_shape", _fail)
    monkeypatch.setattr(runtime_session, "_get_detector_angular_maps", _fail)
    monkeypatch.setattr(
        runtime_session,
        "_detector_pixel_to_scattering_angles",
        _fail,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_scattering_angles_to_detector_pixel",
        _fail,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_angular_maps",
        _fail,
        raising=False,
    )


def _preview_mask_regression_inputs() -> dict[str, object]:
    return {
        "selection": {
            "valid": True,
            "rows": np.array([0, 2, 9], dtype=np.int64),
            "cols": np.array([1, 1, 0], dtype=np.int64),
        },
        "native_shape": (3, 3),
        "radial_axis": np.array([10.0, 20.0], dtype=float),
        "azimuth_axis": np.array([-5.0, 5.0], dtype=float),
        "expected_mask": np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=bool,
        ),
    }


def test_runtime_geometry_fit_roi_projector_does_not_call_legacy_projection_paths(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    bundle = runtime_session.CakeTransformBundle(
        detector_shape=(6, 6),
        radial_deg=np.array([1.0, 2.0], dtype=float),
        raw_azimuth_deg=np.array([-100.0, -80.0], dtype=float),
        gui_azimuth_deg=np.array([-10.0, 10.0], dtype=float),
        lut=SimpleNamespace(),
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("ROI projector should not call legacy projection paths")

    bundle_calls: list[object] = []

    def _record_caked_point_to_detector_pixel(
        ai,
        detector_shape,
        radial_deg,
        azimuth_deg,
        two_theta_deg,
        phi_deg,
        *,
        transform_bundle=None,
    ):
        del ai, detector_shape, radial_deg, azimuth_deg, two_theta_deg, phi_deg
        bundle_calls.append(transform_bundle)
        return (12.0, 34.0)

    monkeypatch.setattr(
        runtime_session,
        "caked_point_to_detector_pixel",
        _record_caked_point_to_detector_pixel,
    )
    monkeypatch.setattr(
        runtime_session,
        "_scattering_angles_to_detector_pixel",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "_detector_pixel_to_scattering_angles",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_detector_angular_maps",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_angular_maps",
        _fail,
        raising=False,
    )

    projector = runtime_session._geometry_fit_caked_roi_fit_space_to_detector_point(
        detector_shape=(6, 6),
        radial_axis=None,
        azimuth_axis=None,
        ai=None,
        transform_bundle=bundle,
    )

    assert projector is not None
    assert projector(1.5, -10.0) == (12.0, 34.0)
    assert bundle_calls == [bundle]


def test_runtime_current_geometry_fit_roi_selection_does_not_call_legacy_projection_paths(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    bundle = runtime_session.CakeTransformBundle(
        detector_shape=(6, 6),
        radial_deg=np.array([1.0, 2.0], dtype=float),
        raw_azimuth_deg=np.array([-100.0, -80.0], dtype=float),
        gui_azimuth_deg=np.array([-10.0, 10.0], dtype=float),
        lut=SimpleNamespace(),
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("ROI selection should not call legacy projection paths")

    bundle_calls: list[tuple[float, float, object]] = []

    def _record_caked_point_to_detector_pixel(
        ai,
        detector_shape,
        radial_deg,
        azimuth_deg,
        two_theta_deg,
        phi_deg,
        *,
        transform_bundle=None,
    ):
        del ai, detector_shape, radial_deg, azimuth_deg
        bundle_calls.append(
            (
                float(two_theta_deg),
                float(phi_deg),
                transform_bundle,
            )
        )
        return (12.0, 34.0)

    monkeypatch.setattr(
        runtime_session,
        "caked_point_to_detector_pixel",
        _record_caked_point_to_detector_pixel,
    )
    monkeypatch.setattr(
        runtime_session,
        "_scattering_angles_to_detector_pixel",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "_detector_pixel_to_scattering_angles",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_detector_angular_maps",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_angular_maps",
        _fail,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            source_row_snapshots={
                0: {
                    "rows": [
                        {
                            "label": "1,0,0",
                            "q_group_key": ("q_group", "primary", 1, 0),
                            "source_table_index": 1,
                            "source_row_index": 2,
                            "sim_col": 3.0,
                            "sim_row": 4.0,
                        }
                    ]
                }
            },
            last_caked_radial_values=np.array([1.0, 2.0], dtype=float),
            last_caked_azimuth_values=np.array([-10.0, 10.0], dtype=float),
            ai_cache={"ai": None},
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pairs_for_index",
        lambda index: (
            [
                {
                    "label": "1,0,0",
                    "background_two_theta_deg": 1.5,
                    "background_phi_deg": -10.0,
                }
            ]
            if int(index) == 0
            else []
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_current_background_native",
        lambda: np.ones((6, 6), dtype=float),
    )
    monkeypatch.setattr(
        runtime_session,
        "_build_live_preview_simulated_peaks_from_cache",
        lambda: [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_live_caked_transform_bundle",
        lambda: bundle,
    )
    monkeypatch.setattr(runtime_session, "fit_config", {"geometry": {"roi": True}})

    def _fake_build_geometry_fit_caked_roi_selection(
        source_rows,
        *,
        required_pairs,
        image_shape,
        fit_config,
        enabled_override,
        fit_space_to_detector_point,
    ):
        del fit_config
        return {
            "enabled": bool(enabled_override),
            "image_shape": tuple(int(v) for v in image_shape),
            "source_row_count": len(source_rows),
            "required_pair_count": len(required_pairs),
            "projected_point": fit_space_to_detector_point(1.5, -10.0),
        }

    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_fit_caked_roi_selection",
        _fake_build_geometry_fit_caked_roi_selection,
    )

    selection = runtime_session._current_geometry_fit_caked_roi_selection(force_enabled=True)

    assert selection == {
        "enabled": True,
        "image_shape": (6, 6),
        "source_row_count": 1,
        "required_pair_count": 1,
        "projected_point": (12.0, 34.0),
    }
    assert bundle_calls == [(1.5, -10.0, bundle)]


def test_runtime_geometry_fit_caked_roi_defaults_to_full_image_when_toggle_missing(
    monkeypatch,
) -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    assert "geometry_fit_caked_roi_enabled_var = tk.BooleanVar(value=False)" in source

    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    monkeypatch.delattr(
        runtime_session,
        "geometry_fit_caked_roi_enabled_var",
        raising=False,
    )

    assert runtime_session._current_geometry_fit_caked_roi_enabled() is False


def test_runtime_geometry_fit_caked_roi_preview_mask_uses_live_exact_bundle_only(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_preview_mask_legacy_helper_guards(monkeypatch, runtime_session)
    inputs = _preview_mask_regression_inputs()
    live_bundle = runtime_session.CakeTransformBundle(
        detector_shape=inputs["native_shape"],
        radial_deg=np.array([10.0, 20.0], dtype=float),
        raw_azimuth_deg=np.array([-95.0, -85.0], dtype=float),
        gui_azimuth_deg=np.array([-5.0, 5.0], dtype=float),
        lut=SimpleNamespace(),
    )
    projection_calls: list[tuple[object, float, float]] = []

    def _record_detector_pixel_to_caked_bin(bundle, col, row):
        projection_calls.append((bundle, float(col), float(row)))
        if (float(col), float(row)) == (1.0, 0.0):
            return (10.2, -4.0)
        if (float(col), float(row)) == (1.0, 2.0):
            return (19.6, 4.0)
        return (None, None)

    def _fail_integrator():
        raise AssertionError("preview mask should reuse matching live bundle")

    monkeypatch.setattr(
        runtime_session,
        "_current_live_caked_transform_bundle",
        lambda: live_bundle,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_preview_integrator",
        _fail_integrator,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        _record_detector_pixel_to_caked_bin,
    )

    mask = runtime_session._geometry_fit_caked_roi_preview_mask(
        selection=inputs["selection"],
        native_shape=inputs["native_shape"],
        radial_axis=inputs["radial_axis"],
        azimuth_axis=inputs["azimuth_axis"],
    )

    np.testing.assert_array_equal(mask, inputs["expected_mask"])
    assert projection_calls == [
        (live_bundle, 1.0, 0.0),
        (live_bundle, 1.0, 2.0),
    ]


def test_runtime_geometry_fit_caked_roi_preview_mask_rebuilds_exact_bundle_without_legacy_helpers(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_preview_mask_legacy_helper_guards(monkeypatch, runtime_session)
    inputs = _preview_mask_regression_inputs()
    ai = runtime_session.FastAzimuthalIntegrator(
        dist=0.25,
        poni1=0.0015,
        poni2=0.0020,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    expected_raw_azimuth = np.asarray(
        runtime_session.gui_phi_to_raw_phi(inputs["azimuth_axis"]),
        dtype=float,
    )
    expected_raw_azimuth = np.sort(expected_raw_azimuth, kind="stable")
    original_resolve_bundle = runtime_session.gui_geometry_fit.resolve_cake_transform_bundle
    resolve_calls: list[
        tuple[object, tuple[int, int], tuple[float, ...], tuple[float, ...] | None, bool]
    ] = []
    projection_calls: list[tuple[object, float, float]] = []

    def _record_resolve_bundle(
        ai_arg,
        detector_shape,
        radial_deg,
        *,
        gui_azimuth_deg=None,
        raw_azimuth_deg=None,
        transform_bundle=None,
        require_gui_display_match=False,
        engine="auto",
        workers="auto",
    ):
        del transform_bundle
        resolve_calls.append(
            (
                ai_arg,
                tuple(detector_shape),
                tuple(np.asarray(radial_deg, dtype=float)),
                (
                    tuple(np.asarray(raw_azimuth_deg, dtype=float))
                    if raw_azimuth_deg is not None
                    else None
                ),
                bool(require_gui_display_match),
            )
        )
        return original_resolve_bundle(
            ai_arg,
            detector_shape,
            radial_deg,
            gui_azimuth_deg=gui_azimuth_deg,
            raw_azimuth_deg=raw_azimuth_deg,
            require_gui_display_match=require_gui_display_match,
            engine=engine,
            workers=workers,
        )

    def _record_detector_pixel_to_caked_bin(bundle, col, row):
        projection_calls.append((bundle, float(col), float(row)))
        if (float(col), float(row)) == (1.0, 0.0):
            return (10.2, -4.0)
        if (float(col), float(row)) == (1.0, 2.0):
            return (19.6, 4.0)
        return (None, None)

    monkeypatch.setattr(
        runtime_session,
        "_current_live_caked_transform_bundle",
        lambda: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_preview_integrator",
        lambda: ai,
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "resolve_cake_transform_bundle",
        _record_resolve_bundle,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        _record_detector_pixel_to_caked_bin,
    )

    mask = runtime_session._geometry_fit_caked_roi_preview_mask(
        selection=inputs["selection"],
        native_shape=inputs["native_shape"],
        radial_axis=inputs["radial_axis"],
        azimuth_axis=inputs["azimuth_axis"],
    )

    np.testing.assert_array_equal(mask, inputs["expected_mask"])
    assert resolve_calls[0] == (
        None,
        inputs["native_shape"],
        (10.0, 20.0),
        tuple(expected_raw_azimuth),
        True,
    )
    assert resolve_calls[1] == (
        ai,
        inputs["native_shape"],
        (10.0, 20.0),
        tuple(expected_raw_azimuth),
        True,
    )
    rebuilt_bundle = projection_calls[0][0]
    assert isinstance(rebuilt_bundle, runtime_session.CakeTransformBundle)
    assert projection_calls == [
        (rebuilt_bundle, 1.0, 0.0),
        (rebuilt_bundle, 1.0, 2.0),
    ]


def test_runtime_geometry_fit_caked_roi_preview_mask_rebuilds_when_live_bundle_raw_axis_is_stale(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_preview_mask_legacy_helper_guards(monkeypatch, runtime_session)
    inputs = _preview_mask_regression_inputs()
    ai = runtime_session.FastAzimuthalIntegrator(
        dist=0.25,
        poni1=0.0015,
        poni2=0.0020,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    stale_bundle = runtime_session.CakeTransformBundle(
        detector_shape=inputs["native_shape"],
        radial_deg=np.array([10.0, 20.0], dtype=float),
        raw_azimuth_deg=np.array([70.0, 110.0], dtype=float),
        gui_azimuth_deg=np.array([-20.0, 20.0], dtype=float),
        lut=SimpleNamespace(),
    )
    expected_raw_azimuth = np.asarray(
        runtime_session.gui_phi_to_raw_phi(inputs["azimuth_axis"]),
        dtype=float,
    )
    expected_raw_azimuth = np.sort(expected_raw_azimuth, kind="stable")
    projection_calls: list[tuple[object, float, float]] = []

    def _record_detector_pixel_to_caked_bin(bundle, col, row):
        projection_calls.append((bundle, float(col), float(row)))
        np.testing.assert_array_equal(
            np.asarray(bundle.raw_azimuth_deg, dtype=float),
            expected_raw_azimuth,
        )
        if (float(col), float(row)) == (1.0, 0.0):
            return (10.2, -4.0)
        if (float(col), float(row)) == (1.0, 2.0):
            return (19.6, 4.0)
        return (None, None)

    monkeypatch.setattr(
        runtime_session,
        "_current_live_caked_transform_bundle",
        lambda: stale_bundle,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_preview_integrator",
        lambda: ai,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        _record_detector_pixel_to_caked_bin,
    )

    mask = runtime_session._geometry_fit_caked_roi_preview_mask(
        selection=inputs["selection"],
        native_shape=inputs["native_shape"],
        radial_axis=inputs["radial_axis"],
        azimuth_axis=inputs["azimuth_axis"],
    )

    np.testing.assert_array_equal(mask, inputs["expected_mask"])
    rebuilt_bundle = projection_calls[0][0]
    assert isinstance(rebuilt_bundle, runtime_session.CakeTransformBundle)
    assert rebuilt_bundle is not stale_bundle
    np.testing.assert_array_equal(
        np.asarray(rebuilt_bundle.raw_azimuth_deg, dtype=float),
        expected_raw_azimuth,
    )
    assert projection_calls == [
        (rebuilt_bundle, 1.0, 0.0),
        (rebuilt_bundle, 1.0, 2.0),
    ]


def test_runtime_geometry_fit_caked_roi_preview_mask_returns_none_without_valid_bundle(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_preview_mask_legacy_helper_guards(monkeypatch, runtime_session)
    inputs = _preview_mask_regression_inputs()
    ai = runtime_session.FastAzimuthalIntegrator(
        dist=0.25,
        poni1=0.0015,
        poni2=0.0020,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )

    def _fail_projection(*_args, **_kwargs):
        raise AssertionError("preview mask should return None before pixel projection")

    monkeypatch.setattr(
        runtime_session,
        "_current_live_caked_transform_bundle",
        lambda: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_preview_integrator",
        lambda: ai,
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "resolve_cake_transform_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        _fail_projection,
    )

    assert (
        runtime_session._geometry_fit_caked_roi_preview_mask(
            selection=inputs["selection"],
            native_shape=inputs["native_shape"],
            radial_axis=inputs["radial_axis"],
            azimuth_axis=inputs["azimuth_axis"],
        )
        is None
    )


def test_runtime_impl_worker_roi_precomputes_projection_context_before_selection() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    store_start = source.index("def _store_worker_caked_view_for_background(")
    store_end = source.index("def _build_source_rows_for_rebuild(", store_start)
    store_source = source[store_start:store_end]

    helper_idx = store_source.index("_geometry_fit_worker_caked_projection_view(")
    selection_idx = store_source.index(
        "roi_selection = gui_geometry_fit.build_geometry_fit_caked_roi_selection("
    )

    assert helper_idx < selection_idx
    assert "background_caked_view.update(precomputed_caked_view)" in store_source
    assert "_worker_projection_analysis_bins()" in store_source
    assert "npt_rad=precompute_npt_rad" in store_source
    assert "npt_azim=precompute_npt_azim" in store_source


def test_runtime_worker_caked_projection_view_builds_same_axes_metadata(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    ai = runtime_session.FastAzimuthalIntegrator(
        dist=0.25,
        poni1=0.0015,
        poni2=0.0020,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    bundle = runtime_session.CakeTransformBundle(
        detector_shape=(6, 8),
        radial_deg=np.array([0.5, 1.5], dtype=float),
        raw_azimuth_deg=np.array([-135.0, 45.0], dtype=float),
        gui_azimuth_deg=np.array([-135.0, 45.0], dtype=float),
        lut=SimpleNamespace(),
    )
    calls: list[tuple[object, ...]] = []

    def _fake_two_theta_max(detector_shape, geometry):
        calls.append(("two_theta_max", tuple(detector_shape), geometry))
        return 12.0

    def _fake_build_angle_axes(
        *,
        npt_rad,
        npt_azim,
        tth_min_deg,
        tth_max_deg,
        azimuth_min_deg,
        azimuth_max_deg,
    ):
        calls.append(
            (
                "axes",
                npt_rad,
                npt_azim,
                tth_min_deg,
                tth_max_deg,
                azimuth_min_deg,
                azimuth_max_deg,
            )
        )
        return (
            np.array([0.5, 1.5], dtype=float),
            np.array([-135.0, 45.0], dtype=float),
        )

    def _fake_build_bundle(ai_arg, detector_shape, radial_deg, raw_azimuth_deg):
        calls.append(
            (
                "bundle",
                ai_arg,
                tuple(detector_shape),
                tuple(np.asarray(radial_deg, dtype=float)),
                tuple(np.asarray(raw_azimuth_deg, dtype=float)),
            )
        )
        return bundle

    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "detector_two_theta_max_deg",
        _fake_two_theta_max,
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit, "build_angle_axes", _fake_build_angle_axes
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_cake_transform_bundle",
        _fake_build_bundle,
    )

    projection_view = runtime_session._geometry_fit_worker_caked_projection_view(
        detector_shape=(6, 8),
        ai=ai,
        npt_rad=17,
        npt_azim=19,
    )

    assert projection_view is not None
    np.testing.assert_array_equal(
        projection_view["radial_axis"],
        np.array([0.5, 1.5], dtype=float),
    )
    np.testing.assert_array_equal(
        projection_view["raw_azimuth_axis"],
        np.array([-135.0, 45.0], dtype=float),
    )
    expected_gui = np.asarray(
        runtime_session.raw_phi_to_gui_phi(np.array([-135.0, 45.0], dtype=float)),
        dtype=float,
    )
    expected_order = np.asarray(np.argsort(expected_gui, kind="stable"), dtype=np.int32)
    np.testing.assert_array_equal(
        projection_view["raw_to_gui_row_permutation"],
        expected_order,
    )
    np.testing.assert_allclose(
        projection_view["azimuth_axis"],
        expected_gui[expected_order],
    )
    assert projection_view["transform_bundle"] is bundle
    assert projection_view["detector_shape"] == (6, 8)
    assert calls[0][:2] == ("two_theta_max", (6, 8))
    assert calls[1] == (
        "axes",
        17,
        19,
        0.0,
        12.0,
        -180.0,
        180.0,
    )
    assert calls[2] == (
        "bundle",
        ai,
        (6, 8),
        (0.5, 1.5),
        (-135.0, 45.0),
    )


def test_runtime_impl_geometry_fit_async_job_captures_analysis_bins() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _build_geometry_fit_async_job(")
    helper_end = source.index("def _handle_geometry_fit_worker_event(", helper_start)
    helper_source = source[helper_start:helper_end]

    assert '"analysis_bins": analysis_bins' in helper_source
    assert '"npt_rad": int(analysis_bins[0])' in helper_source
    assert '"npt_azim": int(analysis_bins[1])' in helper_source


def test_runtime_impl_prepare_caked_payload_keeps_canonical_transform_metadata() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _prepare_caked_display_payload(")
    helper_end = source.index("def _prepare_q_space_display_payload(", helper_start)
    helper_source = source[helper_start:helper_end]

    assert "gui_geometry_fit.build_geometry_fit_caked_view_payload_from_result(" in helper_source
    assert "detector_shape=detector_shape" in helper_source


def test_prepare_caked_intersection_cache_uses_exact_cake_bundle_projector(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        detector_shape = (64, 96)

    bundle = FakeBundle()
    remap_calls: list[tuple[float, float]] = []
    projector_calls: list[tuple[object, float, float]] = []

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session,
        "build_geometry",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("caked HKL cache should not use analytic geometry")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_points_to_angles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("caked HKL cache should not use analytic detector angles")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        lambda bundle_arg, col, row: (
            projector_calls.append((bundle_arg, float(col), float(row)))
            or (
                float(col) + 0.5,
                float(row) - 0.25,
            )
        ),
    )

    transformed = runtime_session._prepare_caked_intersection_cache(
        [
            np.asarray(
                [[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0]],
                dtype=float,
            )
        ],
        transform_bundle=bundle,
        pixel_size_m=1.0e-4,
        distance_m=0.5,
        center_row_px=1.5,
        center_col_px=2.5,
        native_detector_coords_to_bundle_detector_coords=lambda col, row: (
            remap_calls.append((float(col), float(row))) or (float(col) + 2.0, float(row) + 3.0)
        ),
    )

    out = np.asarray(transformed[0], dtype=float)
    assert remap_calls == [(40.0, 50.0)]
    assert projector_calls == [(bundle, 42.0, 53.0)]
    assert out.shape == (1, 16)
    np.testing.assert_allclose(out[0, :9], [1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0])
    assert out[0, 14] == 42.5
    assert out[0, 15] == 52.75


def test_prepare_caked_intersection_cache_blanks_prefilled_caked_cols_without_valid_bundle() -> (
    None
):
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    table = np.asarray(
        [[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.5, -32.0]],
        dtype=float,
    )

    transformed = runtime_session._prepare_caked_intersection_cache(
        [table],
        transform_bundle=None,
        pixel_size_m=1.0e-4,
        distance_m=0.5,
        center_row_px=1.5,
        center_col_px=2.5,
    )

    out = np.asarray(transformed[0], dtype=float)
    assert out.shape == (1, 16)
    np.testing.assert_allclose(out[0, :14], table[0, :14])
    assert np.isnan(out[0, 14])
    assert np.isnan(out[0, 15])


def test_prepare_caked_intersection_cache_blanks_prefilled_caked_cols_when_exact_angles_are_nan(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        detector_shape = (64, 96)

    bundle = FakeBundle()
    projector_calls: list[tuple[object, float, float]] = []
    table = np.asarray(
        [[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.5, -32.0]],
        dtype=float,
    )

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session,
        "build_geometry",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("caked HKL cache should not use analytic geometry")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_points_to_angles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("caked HKL cache should not use analytic detector angles")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        lambda bundle_arg, col, row: (
            projector_calls.append((bundle_arg, float(col), float(row)))
            or (float("nan"), float("nan"))
        ),
    )

    transformed = runtime_session._prepare_caked_intersection_cache(
        [table],
        transform_bundle=bundle,
        pixel_size_m=1.0e-4,
        distance_m=0.5,
        center_row_px=1.5,
        center_col_px=2.5,
    )

    out = np.asarray(transformed[0], dtype=float)
    assert projector_calls == [(bundle, 40.0, 50.0)]
    assert out.shape == (1, 16)
    np.testing.assert_allclose(out[0, :14], table[0, :14])
    assert np.isnan(out[0, 14])
    assert np.isnan(out[0, 15])


def test_analysis_cache_overlay_coords_reuses_current_prepared_caked_cache_columns(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        detector_shape = (32, 48)

    bundle = FakeBundle()
    detector_cache = [np.asarray([[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0]], dtype=float)]
    projector_calls: list[tuple[object, float, float]] = []

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_bundle_detector_coords",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bundle-coordinate remap should not run")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "build_geometry",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("caked cache should not use analytic geometry")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_points_to_angles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("caked cache should not use analytic detector angles")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        lambda bundle_arg, col, row: (
            projector_calls.append((bundle_arg, float(col), float(row))) or (12.25, -33.5)
        ),
    )

    prepared_cache = runtime_session._prepare_caked_intersection_cache(
        detector_cache,
        transform_bundle=bundle,
        pixel_size_m=1.0e-4,
        distance_m=0.5,
        center_row_px=1.5,
        center_col_px=2.5,
    )

    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_transform_bundle",
        bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_transform_bundle",
        bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache",
        prepared_cache,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "stored_intersection_cache",
        detector_cache,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_source_signature",
        (id(detector_cache), len(detector_cache)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_caked_coords",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("current caked cache should reuse prepared caked columns")
        ),
    )

    x_vals, y_vals = runtime_session._analysis_cache_overlay_coords(
        prepared_cache[0],
        show_caked=True,
    )

    assert projector_calls == [(bundle, 40.0, 50.0)]
    np.testing.assert_allclose(x_vals, [12.25])
    np.testing.assert_allclose(y_vals, [-33.5])


def test_run_analysis_job_uses_job_geometry_for_caked_intersection_cache(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        detector_shape = (32, 48)

    captured: dict[str, object] = {}
    bundle = FakeBundle()

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(runtime_session, "corto_detector_var", _RuntimeVar(99.0), raising=False)
    monkeypatch.setattr(runtime_session, "center_x_var", _RuntimeVar(98.0), raising=False)
    monkeypatch.setattr(runtime_session, "center_y_var", _RuntimeVar(97.0), raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 9.9e-4, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_build_analysis_integrator", lambda _job: object())
    monkeypatch.setattr(runtime_session, "caking", lambda image, *_args, **_kwargs: image)
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: {
            "image": np.zeros((2, 2), dtype=float),
            "radial": np.zeros(2, dtype=float),
            "azimuth": np.zeros(2, dtype=float),
            "extent": [0.0, 1.0, 0.0, 1.0],
            "transform_bundle": bundle,
            "detector_shape": bundle.detector_shape,
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda cache, **kwargs: captured.update(kwargs) or list(cache),
    )

    result = runtime_session._run_analysis_job(
        {
            "job_id": 1,
            "signature": "sig",
            "epoch": 2,
            "image": np.zeros((2, 2), dtype=float),
            "background_image": None,
            "intersection_cache": [
                np.asarray([[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0]], dtype=float)
            ],
            "center": [1.25, 2.75],
            "distance_m": 0.625,
            "pixel_size_m": 1.5e-4,
            "q_space_requested": False,
            "is_preview": False,
        }
    )

    assert captured["transform_bundle"] is bundle
    assert captured["pixel_size_m"] == 1.5e-4
    assert captured["distance_m"] == 0.625
    assert captured["center_row_px"] == 1.25
    assert captured["center_col_px"] == 2.75
    assert len(result["sim_caked_intersection_cache"]) == 1
    np.testing.assert_allclose(
        np.asarray(result["sim_caked_intersection_cache"][0], dtype=float),
        [[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0]],
    )


def test_analysis_cache_overlay_coords_ignores_stale_cached_caked_columns(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        pass

    live_bundle = FakeBundle()
    stale_cache_bundle = FakeBundle()
    projector_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_transform_bundle",
        live_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_transform_bundle",
        stale_cache_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_caked_coords",
        lambda col, row: projector_calls.append((float(col), float(row))) or (91.0, -44.0),
    )

    x_vals, y_vals = runtime_session._analysis_cache_overlay_coords(
        np.asarray(
            [
                [
                    1.5,
                    2.5,
                    40.0,
                    50.0,
                    8.0,
                    0.375,
                    1.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    17.5,
                    -32.0,
                ]
            ],
            dtype=float,
        ),
        show_caked=True,
    )

    assert projector_calls == [(40.0, 50.0)]
    np.testing.assert_allclose(x_vals, [91.0])
    np.testing.assert_allclose(y_vals, [-44.0])


def test_analysis_cache_overlay_coords_ignores_stale_cached_caked_columns_when_source_sig_mismatches(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        pass

    live_bundle = FakeBundle()
    caked_cache_table = np.asarray(
        [[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.5, -32.0]],
        dtype=float,
    )
    detector_cache = [np.asarray([[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0]], dtype=float)]
    projector_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_transform_bundle",
        live_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_transform_bundle",
        live_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache",
        [caked_cache_table],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "stored_intersection_cache",
        detector_cache,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_source_signature",
        (999, 1),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_caked_coords",
        lambda col, row: projector_calls.append((float(col), float(row))) or (93.0, -46.0),
    )

    x_vals, y_vals = runtime_session._analysis_cache_overlay_coords(
        caked_cache_table,
        show_caked=True,
    )

    assert projector_calls == [(40.0, 50.0)]
    np.testing.assert_allclose(x_vals, [93.0])
    np.testing.assert_allclose(y_vals, [-46.0])


def test_analysis_cache_overlay_coords_ignores_detector_cache_caked_columns_even_when_bundle_matches(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        pass

    live_bundle = FakeBundle()
    detector_cache_table = np.asarray(
        [[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.5, -32.0]],
        dtype=float,
    )
    projector_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_transform_bundle",
        live_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_transform_bundle",
        live_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache",
        [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_caked_coords",
        lambda col, row: projector_calls.append((float(col), float(row))) or (82.0, -41.0),
    )

    x_vals, y_vals = runtime_session._analysis_cache_overlay_coords(
        detector_cache_table,
        show_caked=True,
    )

    assert projector_calls == [(40.0, 50.0)]
    np.testing.assert_allclose(x_vals, [82.0])
    np.testing.assert_allclose(y_vals, [-41.0])


def test_invalidate_cached_analysis_space_payloads_clears_caked_cache_provenance(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    live_bundle_calls: list[object] = []
    q_space_payload_calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_extent=[0.0, 1.0, -1.0, 1.0],
            last_caked_background_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_radial_values=np.array([1.0, 2.0], dtype=np.float64),
            last_caked_azimuth_values=np.array([-1.0, 1.0], dtype=np.float64),
            last_caked_intersection_cache=("cache",),
            last_caked_intersection_cache_transform_bundle="bundle",
            last_caked_intersection_cache_source_signature=(1, 1),
            last_q_space_payload_signature="q-space-sig",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_live_caked_transform_bundle",
        lambda bundle: live_bundle_calls.append(bundle),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_q_space_display_payload",
        lambda *, sim_payload=None, bg_payload=None: q_space_payload_calls.append(
            (sim_payload, bg_payload)
        ),
        raising=False,
    )

    runtime_session._invalidate_cached_analysis_space_payloads(
        clear_caked=True,
        clear_q_space=False,
    )

    assert runtime_session.simulation_runtime_state.last_caked_image_unscaled is None
    assert runtime_session.simulation_runtime_state.last_caked_extent is None
    assert runtime_session.simulation_runtime_state.last_caked_background_image_unscaled is None
    assert runtime_session.simulation_runtime_state.last_caked_radial_values is None
    assert runtime_session.simulation_runtime_state.last_caked_azimuth_values is None
    assert runtime_session.simulation_runtime_state.last_caked_intersection_cache is None
    assert (
        runtime_session.simulation_runtime_state.last_caked_intersection_cache_transform_bundle
        is None
    )
    assert (
        runtime_session.simulation_runtime_state.last_caked_intersection_cache_source_signature
        is None
    )
    assert live_bundle_calls == [None]
    assert q_space_payload_calls == []


def test_runtime_impl_combined_detector_cache_recomposition_invalidates_caked_intersection_cache() -> (
    None
):
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    update_start = source.index("def _publish_combined_simulation_state(")
    update_end = source.index(
        "simulation_runtime_state.stored_sim_image = updated_image",
        update_start,
    )
    update_source = source[update_start:update_end]

    assert (
        "simulation_runtime_state.stored_intersection_cache = list(intersection_cache_local)"
        in update_source
    )
    assert "_clear_caked_intersection_cache()" in update_source


def test_runtime_impl_full_reset_invalidates_caked_intersection_cache() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    reset_start = source.index("def _initialize_runtime_controls_block_28() -> None:")
    reset_end = source.index(
        "###############################################################################",
        reset_start,
    )
    reset_source = source[reset_start:reset_end]

    assert "simulation_runtime_state.stored_q_group_content_signature = None" in reset_source

    clear_start = reset_source.index("simulation_runtime_state.stored_intersection_cache = None")
    clear_end = reset_source.index(
        "simulation_runtime_state.last_unscaled_image_signature = None",
        clear_start,
    )
    clear_source = reset_source[clear_start:clear_end]

    assert "_clear_caked_intersection_cache()" in clear_source


def test_runtime_impl_combined_reset_helper_clears_q_group_content_signature() -> None:
    source = PRIMARY_CACHE_HELPERS_SOURCE_PATH.read_text(encoding="utf-8")
    reset_start = source.index("def reset_combined_simulation_artifacts(")
    reset_end = source.index("def store_primary_cache_payload(", reset_start)
    reset_source = source[reset_start:reset_end]

    assert "simulation_runtime_state.stored_q_group_content_signature = None" in reset_source


def test_runtime_impl_manual_rebuild_invalidates_caked_intersection_cache() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    clear_start = source.index(
        "simulation_runtime_state.stored_intersection_cache = _copy_intersection_cache_tables("
    )
    rebuild_end = source.index(
        "simulation_runtime_state.last_simulation_signature = rebuild_result.requested_signature",
        clear_start,
    )
    clear_source = source[clear_start:rebuild_end]
    function_start = source.index("def _commit_geometry_manual_source_row_rebuild_result(")
    function_end = source.index(
        "def _geometry_manual_rebuild_source_rows_for_background(",
        function_start,
    )
    function_source = source[function_start:function_end]

    assert "_clear_caked_intersection_cache()" in clear_source
    assert "simulation_runtime_state.stored_q_group_content_signature =" in function_source
    assert "simulation_runtime_state.stored_hit_table_signature =" not in function_source


def test_runtime_session_manual_rebuild_rows_only_clears_stale_q_group_hit_tables(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")
    state_module = importlib.import_module("ra_sim.gui.state")

    stale_hit_tables = [
        np.asarray(
            [[12.0, 10.0, 20.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    ]
    fresh_rows = [
        {
            "display_col": 30.0,
            "display_row": 40.0,
            "native_col": 30.0,
            "native_row": 40.0,
            "sim_col": 30.0,
            "sim_row": 40.0,
            "sim_col_raw": 30.0,
            "sim_row_raw": 40.0,
            "hkl": (1, 0, 1),
            "hkl_raw": (1.0, 0.0, 1.0),
            "intensity": 7.0,
            "weight": 7.0,
            "source_label": "primary",
            "source_table_index": 0,
            "source_row_index": 1,
            "q_group_key": ("q_group", "primary", 1, 1),
        }
    ]
    runtime_state = state_module.SimulationRuntimeState(
        stored_max_positions_local=list(stale_hit_tables),
        stored_hit_table_signature=("sig", 0),
        stored_q_group_content_signature=(
            runtime_session.gui_geometry_q_group_manager._geometry_q_group_content_signature_from_hit_tables(
                stale_hit_tables
            )
        ),
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        runtime_state,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 64, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_set_runtime_peak_cache_from_projected_rows",
        lambda rows: setattr(
            runtime_state,
            "peak_records",
            [dict(entry) for entry in rows],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_source_snapshot_diagnostics",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_trace_live_cache_event",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    rebuild_result = geometry_fit.GeometryFitSourceRowRebuildResult(
        background_index=0,
        requested_signature=("sig", 1),
        requested_signature_summary="sig-1",
        projected_rows=[dict(entry) for entry in fresh_rows],
        stored_rows=[dict(entry) for entry in fresh_rows],
        rebuild_source="live_runtime_cache",
        rebuild_attempts=["live_runtime_cache"],
        diagnostics={},
        peak_table_lattice=[(3.0, 5.0, "primary")],
        hit_tables=None,
        source_reflection_indices=[7],
        metadata={},
    )

    committed_rows = runtime_session._commit_geometry_manual_source_row_rebuild_result(
        rebuild_result
    )

    assert runtime_state.stored_max_positions_local is None
    assert committed_rows == [dict(entry) for entry in fresh_rows]

    q_group_bundle = (
        runtime_session.gui_geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
            simulation_runtime_state=runtime_state,
            preview_state=state_module.GeometryPreviewState(),
            q_group_state=state_module.GeometryQGroupState(),
            fit_config=None,
            current_geometry_fit_var_names_factory=lambda: [],
            primary_a_factory=lambda: 3.0,
            primary_c_factory=lambda: 5.0,
            image_size_factory=lambda: 64,
            native_sim_to_display_coords=lambda col, row, _shape: (
                float(col),
                float(row),
            ),
        )
    )
    entries = q_group_bundle.build_entries_snapshot()

    assert [entry["key"] for entry in entries] == [("q_group", "primary", 1, 1)]
    assert entries[0]["peak_count"] == 1
    assert entries[0]["total_intensity"] == 7.0


def test_manual_rebuild_source_snapshot_is_row_content_aware_and_reusable(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")
    state_module = importlib.import_module("ra_sim.gui.state")

    fresh_rows = [
        {
            "display_col": 30.0,
            "display_row": 40.0,
            "native_col": 30.0,
            "native_row": 40.0,
            "sim_col": 30.0,
            "sim_row": 40.0,
            "sim_col_raw": 30.0,
            "sim_row_raw": 40.0,
            "hkl": (1, 0, 1),
            "hkl_raw": (1.0, 0.0, 1.0),
            "intensity": 7.0,
            "weight": 7.0,
            "source_label": "primary",
            "source_table_index": 0,
            "source_row_index": 1,
            "q_group_key": ("q_group", "primary", 1, 1),
        }
    ]
    runtime_state = state_module.SimulationRuntimeState(
        last_simulation_signature=("base", 0),
    )
    background_state = state_module.BackgroundRuntimeState()
    background_state.current_background_index = 0

    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(
        runtime_session, "background_runtime_state", background_state, raising=False
    )
    monkeypatch.setattr(runtime_session, "image_size", 64, raising=False)
    monkeypatch.setattr(
        runtime_session, "_retain_runtime_optional_cache", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_set_runtime_peak_cache_from_projected_rows",
        lambda _rows: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_source_snapshot_diagnostics",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_trace_live_cache_event",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_signature_summary",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_inventory_snapshot",
        lambda: {"source_snapshot_count": len(runtime_state.source_row_snapshots)},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 3.0, "c": 5.0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_for_background",
        lambda _idx, _params=None: ("sig", 1),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda _idx, **_kwargs: {"mode": "detector", "detector_shape": [64, 64]},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_project_peaks_for_background",
        lambda _idx, rows, **_kwargs: [dict(entry) for entry in rows],
        raising=False,
    )

    rebuild_result = geometry_fit.GeometryFitSourceRowRebuildResult(
        background_index=0,
        requested_signature=("sig", 1),
        requested_signature_summary="sig-1",
        projected_rows=[dict(entry) for entry in fresh_rows],
        stored_rows=[dict(entry) for entry in fresh_rows],
        rebuild_source="live_runtime_cache",
        rebuild_attempts=["live_runtime_cache"],
        diagnostics={},
        peak_table_lattice=[(3.0, 5.0, "primary")],
        hit_tables=None,
        source_reflection_indices=[7],
        metadata={},
    )

    committed_rows = runtime_session._commit_geometry_manual_source_row_rebuild_result(
        rebuild_result
    )

    assert committed_rows == [dict(entry) for entry in fresh_rows]
    snapshot = runtime_state.source_row_snapshots[0]
    assert snapshot["simulation_signature"] == ("sig", 1)
    assert snapshot["base_simulation_signature"] == runtime_state.last_simulation_signature
    assert snapshot["row_content_signature"] == runtime_state.stored_q_group_content_signature
    assert snapshot["row_count"] == 1
    assert snapshot["valid_for_picker"] is True
    assert snapshot["empty_reason"] is None

    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_rebuild_source_rows_for_background",
        lambda *_args, **_kwargs: pytest.fail("row-content-aware snapshot should be reused"),
        raising=False,
    )

    rows = runtime_session._geometry_manual_source_rows_for_background(
        0,
        {"a": 3.0, "c": 5.0},
        consumer="manual_picker",
    )

    assert rows == [dict(entry, background_index=0) for entry in fresh_rows]
    assert snapshot["row_content_signature"] == runtime_state.stored_q_group_content_signature
    assert snapshot["valid_for_picker"] is True


def _patch_apply_ready_simulation_result_dependencies(
    monkeypatch,
    runtime_session,
    runtime_state,
) -> None:
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_copy_intersection_cache_tables",
        lambda tables: list(tables) if isinstance(tables, (list, tuple)) else [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_copy_hit_tables",
        lambda tables: list(tables) if isinstance(tables, (list, tuple)) else [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_peak_table_payload",
        lambda intersection_cache, legacy_hit_tables: (
            list(intersection_cache)
            if isinstance(intersection_cache, (list, tuple)) and len(intersection_cache) > 0
            else (list(legacy_hit_tables) if isinstance(legacy_hit_tables, (list, tuple)) else [])
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_q_group_manager,
        "audited_full_order_source_reflection_index_groups",
        lambda groups, owner: [list(range(len(group or []))) for group in groups],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_cache_payload",
        lambda **kwargs: setattr(runtime_state, "store_primary_cache_payload_kwargs", kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_reset_combined_simulation_artifacts",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_trace_live_cache_event",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_count",
        lambda value: len(value) if isinstance(value, (list, tuple)) else 0,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_shape",
        lambda value: None if value is None else tuple(np.asarray(value).shape),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_signature_summary",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_mark_numba_cache_compiled_artifacts_available",
        lambda: None,
        raising=False,
    )


def _make_runtime_simulation_generation_job(
    *,
    sample_count: int = 1,
    job_kind: str = "full",
    run_primary: bool = True,
    run_secondary: bool = True,
    collect_hit_tables: bool = False,
    collect_primary_hit_tables: bool = False,
    collect_secondary_hit_tables: bool = False,
    build_primary_intersection_cache: bool = False,
    build_secondary_intersection_cache: bool = False,
    capture_primary_hit_tables_raw: bool = False,
    capture_secondary_hit_tables_raw: bool = False,
    accumulate_image: bool = True,
) -> dict[str, object]:
    sample_axis = np.arange(max(int(sample_count), 1), dtype=np.float64)
    secondary_available = bool(run_secondary)
    if str(job_kind) == "preview":
        active_peak_row_sides: tuple[str, ...] = ()
    elif str(job_kind) == "primary_fill":
        active_peak_row_sides = ("primary",) if bool(run_primary) else ()
    else:
        active_peak_row_sides = tuple(
            side
            for side, enabled in (
                ("primary", bool(run_primary)),
                ("secondary", bool(run_secondary and secondary_available)),
            )
            if enabled
        )
    return {
        "job_id": 17,
        "job_kind": str(job_kind),
        "signature": ("runtime-job", job_kind, int(sample_count)),
        "epoch": 3,
        "image_size": 4,
        "pixel_size_m": 1.0e-4,
        "center": np.asarray([2.0, 2.0], dtype=np.float64),
        "mosaic_params": {
            "beam_x_array": sample_axis.copy(),
            "beam_y_array": sample_axis.copy(),
            "theta_array": sample_axis.copy(),
            "phi_array": sample_axis.copy(),
            "wavelength_array": np.full(sample_axis.shape, 1.54e-10, dtype=np.float64),
            "sample_weights": np.ones(sample_axis.shape, dtype=np.float64),
            "n2_sample_array": None,
            "sigma_mosaic_deg": 0.0,
            "gamma_mosaic_deg": 0.0,
            "eta": 0.0,
            "solve_q_steps": 1,
            "solve_q_rel_tol": 1.0e-6,
            "solve_q_mode": 0,
        },
        "lambda_value": 1.54,
        "distance_m": 0.5,
        "gamma_deg": 0.0,
        "Gamma_deg": 0.0,
        "chi_deg": 0.0,
        "psi_deg": 0.0,
        "psi_z_deg": 0.0,
        "zs": 0.0,
        "zb": 0.0,
        "theta_initial_deg": 0.0,
        "cor_angle_deg": 0.0,
        "sample_width_m": 1.0,
        "sample_length_m": 1.0,
        "sample_depth_m": 1.0,
        "debye_x": 0.0,
        "debye_y": 0.0,
        "optics_mode": 0,
        "collect_hit_tables": bool(collect_hit_tables),
        "collect_primary_hit_tables": bool(collect_primary_hit_tables),
        "collect_secondary_hit_tables": bool(collect_secondary_hit_tables),
        "build_primary_intersection_cache": bool(build_primary_intersection_cache),
        "build_secondary_intersection_cache": bool(build_secondary_intersection_cache),
        "capture_primary_hit_tables_raw": bool(capture_primary_hit_tables_raw),
        "capture_secondary_hit_tables_raw": bool(capture_secondary_hit_tables_raw),
        "secondary_available": bool(secondary_available),
        "active_peak_row_sides": tuple(active_peak_row_sides),
        "hit_table_signature": ("hit-signature", int(sample_count)),
        "primary_contribution_cache_signature": None,
        "primary_source_mode": "source-mode",
        "primary_contribution_keys": [],
        "active_primary_contribution_keys": [],
        "accumulate_image": bool(accumulate_image),
        "qr_cylinder_replace_simulation": False,
        "n2_value": 1.0 + 0.0j,
        "primary_data": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        "primary_intensities": np.asarray([10.0], dtype=np.float64),
        "secondary_data": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64),
        "secondary_intensities": np.asarray([5.0], dtype=np.float64),
        "run_primary": bool(run_primary),
        "run_secondary": bool(run_secondary),
        "a_primary": 5.0,
        "c_primary": 6.0,
        "a_secondary": 7.0,
        "c_secondary": 8.0,
        "is_preview": bool(str(job_kind) == "preview"),
        "preview_sample_count": None,
        "exit_projection_mode": "internal",
    }


def test_primary_cache_helpers_copy_non_list_hit_table_payloads() -> None:
    helpers = importlib.import_module("ra_sim.gui._runtime.primary_cache_helpers")

    class TableIterable:
        def __init__(self, tables):
            self._tables = list(tables)

        def __iter__(self):
            return iter(self._tables)

        def __len__(self):
            return len(self._tables)

    hit_table = np.asarray(
        [[1000.0, 42.5, 55.5, 0.0, 1.0, 0.0, 2.0]],
        dtype=np.float64,
    )

    copied = helpers.copy_hit_tables(TableIterable([hit_table]))

    assert len(copied) == 1
    assert copied[0].shape == (1, 7)
    np.testing.assert_allclose(copied[0], hit_table)


def test_primary_cache_helpers_malformed_detector_cache_copy_stays_detector_shaped() -> None:
    helpers = importlib.import_module("ra_sim.gui._runtime.primary_cache_helpers")
    schema = importlib.import_module("ra_sim.simulation.intersection_cache_schema")

    copied = helpers.copy_intersection_cache_tables(
        [
            object(),
            np.ones((1, schema.BASE_HIT_ROW_WIDTH), dtype=np.float64),
        ]
    )

    assert len(copied) == 2
    for table in copied:
        assert table.ndim == 2
        assert table.shape == (0, schema.CURRENT_DETECTOR_CACHE_WIDTH)


def test_apply_ready_simulation_result_preserves_existing_cache_when_cache_was_not_built(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    runtime_state = SimpleNamespace(
        stored_primary_intersection_cache=["old-primary-cache"],
        stored_secondary_intersection_cache=["old-secondary-cache"],
        stored_primary_max_positions=["old-primary-peak"],
        stored_secondary_max_positions=["old-secondary-peak"],
        stored_primary_peak_table_lattice=[("old-primary",)],
        stored_secondary_peak_table_lattice=[("old-secondary",)],
        stored_primary_source_reflection_indices=[10],
        stored_secondary_source_reflection_indices=[20],
        stored_hit_table_signature=("old-signature", 1),
        primary_source_mode="source-mode",
        primary_active_contribution_keys=["key"],
    )
    _patch_apply_ready_simulation_result_dependencies(
        monkeypatch,
        runtime_session,
        runtime_state,
    )

    runtime_session._apply_ready_simulation_result(
        {
            "primary_image": np.ones((2, 2), dtype=np.float64),
            "secondary_image": np.zeros((2, 2), dtype=np.float64),
            "run_primary": True,
            "run_secondary": True,
            "primary_hit_table_state_refreshed": False,
            "secondary_hit_table_state_refreshed": False,
            "primary_intersection_cache_built": False,
            "secondary_intersection_cache_built": False,
            "primary_max_positions": [],
            "secondary_max_positions": [],
            "primary_peak_table_lattice": [],
            "secondary_peak_table_lattice": [],
            "primary_intersection_cache": [],
            "secondary_intersection_cache": [],
            "hit_table_signature": ("new-signature", 2),
            "primary_contribution_cache_signature": None,
            "primary_source_mode": "source-mode",
            "primary_contribution_keys": [],
            "active_primary_contribution_keys": [],
            "primary_hit_tables_raw": [],
            "primary_best_sample_indices": [],
            "image_generation_elapsed_ms": 1.0,
            "is_preview": False,
        }
    )

    assert runtime_state.stored_primary_intersection_cache == ["old-primary-cache"]
    assert runtime_state.stored_secondary_intersection_cache == ["old-secondary-cache"]
    assert runtime_state.stored_primary_max_positions == ["old-primary-peak"]
    assert runtime_state.stored_secondary_max_positions == ["old-secondary-peak"]
    assert runtime_state.stored_hit_table_signature == ("old-signature", 1)


def test_apply_ready_simulation_result_keeps_signature_stale_until_all_run_sides_refresh(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    runtime_state = SimpleNamespace(
        stored_primary_intersection_cache=["old-primary-cache"],
        stored_secondary_intersection_cache=["old-secondary-cache"],
        stored_primary_max_positions=["old-primary-peak"],
        stored_secondary_max_positions=["old-secondary-peak"],
        stored_primary_peak_table_lattice=[("old-primary",)],
        stored_secondary_peak_table_lattice=[("old-secondary",)],
        stored_primary_source_reflection_indices=[10],
        stored_secondary_source_reflection_indices=[20],
        stored_hit_table_signature=("old-signature", 1),
        primary_source_mode="source-mode",
        primary_active_contribution_keys=["key"],
    )
    _patch_apply_ready_simulation_result_dependencies(
        monkeypatch,
        runtime_session,
        runtime_state,
    )

    runtime_session._apply_ready_simulation_result(
        {
            "primary_image": np.ones((2, 2), dtype=np.float64),
            "secondary_image": np.zeros((2, 2), dtype=np.float64),
            "run_primary": True,
            "run_secondary": True,
            "primary_hit_table_state_refreshed": True,
            "secondary_hit_table_state_refreshed": False,
            "primary_intersection_cache_built": False,
            "secondary_intersection_cache_built": False,
            "primary_max_positions": ["new-primary-peak"],
            "secondary_max_positions": [],
            "primary_peak_table_lattice": [("new-primary",)],
            "secondary_peak_table_lattice": [],
            "primary_intersection_cache": [],
            "secondary_intersection_cache": [],
            "hit_table_signature": ("new-signature", 2),
            "primary_contribution_cache_signature": None,
            "primary_source_mode": "source-mode",
            "primary_contribution_keys": [],
            "active_primary_contribution_keys": [],
            "primary_hit_tables_raw": ["new-primary-peak"],
            "primary_best_sample_indices": [],
            "image_generation_elapsed_ms": 1.0,
            "is_preview": False,
        }
    )

    assert runtime_state.stored_primary_intersection_cache == ["old-primary-cache"]
    assert runtime_state.stored_secondary_intersection_cache == ["old-secondary-cache"]
    assert runtime_state.stored_primary_max_positions == ["new-primary-peak"]
    assert runtime_state.stored_secondary_max_positions == ["old-secondary-peak"]
    assert runtime_state.stored_hit_table_signature == ("old-signature", 1)
    assert not runtime_session._cached_hit_tables_reusable(
        ("new-signature", 2),
        run_primary=True,
        run_secondary=True,
    )


def test_apply_ready_simulation_result_advances_signature_when_raw_hit_tables_refresh_all_run_sides(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    runtime_state = SimpleNamespace(
        stored_primary_intersection_cache=["old-primary-cache"],
        stored_secondary_intersection_cache=["old-secondary-cache"],
        stored_primary_max_positions=["old-primary-peak"],
        stored_secondary_max_positions=["old-secondary-peak"],
        stored_primary_peak_table_lattice=[("old-primary",)],
        stored_secondary_peak_table_lattice=[("old-secondary",)],
        stored_primary_source_reflection_indices=[10],
        stored_secondary_source_reflection_indices=[20],
        stored_hit_table_signature=("old-signature", 1),
        primary_source_mode="source-mode",
        primary_active_contribution_keys=["key"],
    )
    _patch_apply_ready_simulation_result_dependencies(
        monkeypatch,
        runtime_session,
        runtime_state,
    )

    runtime_session._apply_ready_simulation_result(
        {
            "primary_image": np.ones((2, 2), dtype=np.float64),
            "secondary_image": np.zeros((2, 2), dtype=np.float64),
            "run_primary": True,
            "run_secondary": True,
            "primary_hit_table_state_refreshed": True,
            "secondary_hit_table_state_refreshed": True,
            "primary_intersection_cache_built": False,
            "secondary_intersection_cache_built": False,
            "primary_max_positions": ["new-primary-peak"],
            "secondary_max_positions": ["new-secondary-peak"],
            "primary_peak_table_lattice": [("new-primary",)],
            "secondary_peak_table_lattice": [("new-secondary",)],
            "primary_intersection_cache": [],
            "secondary_intersection_cache": [],
            "hit_table_signature": ("new-signature", 2),
            "primary_contribution_cache_signature": None,
            "primary_source_mode": "source-mode",
            "primary_contribution_keys": [],
            "active_primary_contribution_keys": [],
            "primary_hit_tables_raw": ["new-primary-peak"],
            "primary_best_sample_indices": [],
            "image_generation_elapsed_ms": 1.0,
            "is_preview": False,
        }
    )

    assert runtime_state.stored_primary_intersection_cache == ["old-primary-cache"]
    assert runtime_state.stored_secondary_intersection_cache == ["old-secondary-cache"]
    assert runtime_state.stored_primary_max_positions == ["new-primary-peak"]
    assert runtime_state.stored_secondary_max_positions == ["new-secondary-peak"]
    assert runtime_state.stored_hit_table_signature == ("new-signature", 2)
    assert runtime_session._cached_hit_tables_reusable(
        ("new-signature", 2),
        run_primary=True,
        run_secondary=True,
    )


def test_hit_table_state_present_for_run_sides_requires_all_requested_run_side_rows(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    runtime_state = SimpleNamespace(
        stored_primary_max_positions=["primary-peak"],
        stored_secondary_max_positions=None,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        runtime_state,
        raising=False,
    )

    assert runtime_session._hit_table_state_present_for_run_sides(
        run_primary=True,
        run_secondary=False,
    )
    assert not runtime_session._hit_table_state_present_for_run_sides(
        run_primary=True,
        run_secondary=True,
    )


def test_preview_job_disables_raw_capture_and_picker_restore(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    monkeypatch.setattr(
        runtime_session,
        "PREVIEW_CALCULATIONS_ENABLED",
        True,
        raising=False,
    )
    preview_job = runtime_session._build_preview_simulation_job(
        _make_runtime_simulation_generation_job(
            sample_count=4,
            job_kind="full",
            run_primary=True,
            run_secondary=True,
            collect_hit_tables=False,
            accumulate_image=True,
        ),
        max_samples=2,
    )

    assert isinstance(preview_job, dict)
    assert preview_job["job_kind"] == "preview"
    assert preview_job["capture_primary_hit_tables_raw"] is False
    assert preview_job["capture_secondary_hit_tables_raw"] is False
    assert preview_job["build_primary_intersection_cache"] is False
    assert preview_job["build_secondary_intersection_cache"] is False
    assert preview_job["active_peak_row_sides"] == ()


def test_copy_hit_tables_accepts_custom_iterable() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class TableIterable:
        def __iter__(self):
            return iter([np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)])

    copied = runtime_session._copy_hit_tables(TableIterable())

    assert len(copied) == 1
    assert copied[0].shape == (1, 3)


def test_copy_intersection_cache_tables_keeps_detector_shape_for_bad_payload() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    copied = runtime_session._copy_intersection_cache_tables([object()])

    assert len(copied) == 1
    assert copied[0].shape == (0, 17)


def test_build_preview_simulation_job_preserves_n2_metadata_and_slices_snapshot(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    monkeypatch.setattr(
        runtime_session,
        "PREVIEW_CALCULATIONS_ENABLED",
        True,
        raising=False,
    )
    source_meta = ("cif_path", "C:/optics/runtime-preview.cif")
    n2_sample_array = np.array(
        [0.91 + 0.01j, 0.92 + 0.02j, 0.93 + 0.03j, 0.94 + 0.04j],
        dtype=np.complex128,
    )
    wavelength_snapshot = np.array([1.11, 1.22, 1.33, 1.44], dtype=np.float64)
    job = _make_runtime_simulation_generation_job(
        sample_count=4,
        job_kind="full",
        run_primary=True,
        run_secondary=True,
        collect_hit_tables=False,
        accumulate_image=True,
    )
    job["mosaic_params"]["n2_sample_array"] = n2_sample_array.copy()
    job["mosaic_params"]["_n2_sample_array_source"] = source_meta
    job["mosaic_params"]["_n2_sample_array_wavelength_snapshot"] = wavelength_snapshot.copy()

    preview_job = runtime_session._build_preview_simulation_job(
        job,
        max_samples=2,
    )

    assert isinstance(preview_job, dict)
    assert preview_job["job_kind"] == "preview"
    assert preview_job["mosaic_params"]["_n2_sample_array_source"] == source_meta
    np.testing.assert_array_equal(
        preview_job["mosaic_params"]["n2_sample_array"],
        n2_sample_array[[0, 3]],
    )
    np.testing.assert_array_equal(
        preview_job["mosaic_params"]["_n2_sample_array_wavelength_snapshot"],
        wavelength_snapshot[[0, 3]],
    )


def test_run_debug_simulation_forwards_cached_n2_and_sample_weights(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    captured: dict[str, object] = {}
    status_texts: list[str] = []
    sample_weights = np.array([0.75, 0.25], dtype=np.float64)
    n2_sample_array = np.array([0.95 + 0.01j, 0.96 + 0.02j], dtype=np.complex128)

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            profile_cache={
                "beam_x_array": np.array([0.0, 1.0], dtype=np.float64),
                "beam_y_array": np.array([0.0, 1.0], dtype=np.float64),
                "theta_array": np.array([0.0, 0.1], dtype=np.float64),
                "phi_array": np.array([0.0, 0.2], dtype=np.float64),
                "wavelength_array": np.array([1.54, 1.55], dtype=np.float64),
                "sample_weights": sample_weights.copy(),
                "n2_sample_array": n2_sample_array.copy(),
                "sigma_mosaic_deg": 0.3,
                "gamma_mosaic_deg": 0.4,
                "eta": 0.5,
                "solve_q_steps": 7,
                "solve_q_rel_tol": 1.0e-6,
                "solve_q_mode": 2,
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "gamma_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "Gamma_var", _RuntimeVar(2.0), raising=False)
    monkeypatch.setattr(runtime_session, "chi_var", _RuntimeVar(3.0), raising=False)
    monkeypatch.setattr(runtime_session, "psi_z_var", _RuntimeVar(4.0), raising=False)
    monkeypatch.setattr(runtime_session, "zs_var", _RuntimeVar(5.0), raising=False)
    monkeypatch.setattr(runtime_session, "zb_var", _RuntimeVar(6.0), raising=False)
    monkeypatch.setattr(runtime_session, "a_var", _RuntimeVar(7.0), raising=False)
    monkeypatch.setattr(runtime_session, "c_var", _RuntimeVar(8.0), raising=False)
    monkeypatch.setattr(runtime_session, "debye_x_var", _RuntimeVar(0.1), raising=False)
    monkeypatch.setattr(runtime_session, "debye_y_var", _RuntimeVar(0.2), raising=False)
    monkeypatch.setattr(runtime_session, "corto_detector_var", _RuntimeVar(0.3), raising=False)
    monkeypatch.setattr(runtime_session, "center_x_var", _RuntimeVar(10.0), raising=False)
    monkeypatch.setattr(runtime_session, "center_y_var", _RuntimeVar(11.0), raising=False)
    monkeypatch.setattr(runtime_session, "cor_angle_var", _RuntimeVar(12.0), raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 16, raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 2.0e-4, raising=False)
    monkeypatch.setattr(runtime_session, "lambda_", 1.54, raising=False)
    monkeypatch.setattr(runtime_session, "psi", 0.6, raising=False)
    monkeypatch.setattr(runtime_session, "n2", 1.0 + 0.0j, raising=False)
    monkeypatch.setattr(
        runtime_session, "miller", np.array([[1.0, 0.0, 0.0]], dtype=np.float64), raising=False
    )
    monkeypatch.setattr(
        runtime_session, "intensities", np.array([5.0], dtype=np.float64), raising=False
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_effective_theta_initial",
        lambda strict_count=False: 13.0,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "current_solve_q_values",
        lambda: SimpleNamespace(steps=7, rel_tol=1.0e-6, mode_flag=2),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "process_peaks_parallel_debug",
        lambda *args, **kwargs: (
            captured.update({"args": args, "kwargs": kwargs})
            or (
                np.zeros((16, 16), dtype=np.float64),
                np.zeros((1, 6), dtype=np.float64),
                np.zeros((1, 1, 5), dtype=np.float64),
                np.zeros((1,), dtype=np.int64),
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "dump_debug_log", lambda: None, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "progress_label",
        SimpleNamespace(config=lambda **kwargs: status_texts.append(str(kwargs.get("text")))),
        raising=False,
    )

    runtime_session.run_debug_simulation()

    np.testing.assert_array_equal(captured["kwargs"]["sample_weights"], sample_weights)
    assert captured["kwargs"]["pixel_size_m"] == pytest.approx(2.0e-4)
    np.testing.assert_array_equal(
        captured["kwargs"]["n2_sample_array_override"],
        n2_sample_array,
    )
    assert status_texts[-1] == "Debug simulation complete. Log saved."


def test_run_simulation_generation_job_collects_raw_peak_rows_for_both_run_sides(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    projection_debug = importlib.import_module("ra_sim.simulation.projection_debug")
    simulate_calls: list[tuple[bool, bool, float]] = []

    monkeypatch.setattr(
        projection_debug,
        "start_projection_debug_session",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        projection_debug,
        "finalize_projection_debug_session",
        lambda *_args, **_kwargs: None,
    )

    def _fake_simulate_request(request, **_kwargs):
        a_value = float(request.geometry.av)
        simulate_calls.append(
            (
                bool(request.collect_hit_tables),
                bool(request.build_intersection_cache),
                a_value,
            )
        )
        side = "primary" if a_value < 6.0 else "secondary"
        return SimpleNamespace(
            image=np.full((4, 4), a_value, dtype=np.float64),
            hit_tables=[
                {
                    "side": side,
                    "hkl": (1, 0, 0) if side == "primary" else (0, 1, 0),
                    "q_group_key": (side, 1, 0, 0),
                    "display_col": a_value,
                    "display_row": a_value + 1.0,
                }
            ],
            intersection_cache=[],
            used_python_runner=False,
        )

    monkeypatch.setattr(
        runtime_session,
        "simulate_request",
        _fake_simulate_request,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulate_qr_rods_request",
        lambda *_args, **_kwargs: pytest.fail("qr rod runner should stay unused"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_copy_hit_tables",
        lambda tables: [dict(entry) for entry in tables],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_copy_intersection_cache_tables",
        lambda tables: list(tables),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_peak_table_payload",
        lambda intersection_cache, legacy_hit_tables: (
            list(intersection_cache) if intersection_cache else list(legacy_hit_tables)
        ),
        raising=False,
    )

    result = runtime_session._run_simulation_generation_job(
        _make_runtime_simulation_generation_job(
            run_primary=True,
            run_secondary=True,
            collect_hit_tables=False,
            collect_primary_hit_tables=False,
            collect_secondary_hit_tables=False,
            capture_primary_hit_tables_raw=True,
            capture_secondary_hit_tables_raw=True,
        )
    )

    assert simulate_calls == [(True, False, 5.0), (True, False, 7.0)]
    assert result["active_peak_row_sides"] == ("primary", "secondary")
    assert result["primary_hit_table_state_refreshed"] is True
    assert result["secondary_hit_table_state_refreshed"] is True
    assert result["primary_raw_rows_fresh"] is True
    assert result["secondary_raw_rows_fresh"] is True
    assert result["primary_intersection_cache_built"] is False
    assert result["secondary_intersection_cache_built"] is False
    assert np.asarray(result["primary_best_sample_indices"]).size == 0
    assert np.asarray(result["secondary_best_sample_indices"]).size == 0
    assert result["primary_max_positions"] == [
        {
            "side": "primary",
            "hkl": (1, 0, 0),
            "q_group_key": ("primary", 1, 0, 0),
            "display_col": 5.0,
            "display_row": 6.0,
        }
    ]
    assert result["secondary_max_positions"] == [
        {
            "side": "secondary",
            "hkl": (0, 1, 0),
            "q_group_key": ("secondary", 1, 0, 0),
            "display_col": 7.0,
            "display_row": 8.0,
        }
    ]


def test_run_simulation_generation_job_collects_qr_raw_rows_without_best_sample_buffer(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    projection_debug = importlib.import_module("ra_sim.simulation.projection_debug")
    simulate_calls: list[tuple[bool, bool]] = []

    monkeypatch.setattr(
        projection_debug,
        "start_projection_debug_session",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        projection_debug,
        "finalize_projection_debug_session",
        lambda *_args, **_kwargs: None,
    )

    def _fake_simulate_qr_rods_request(_data, request, **_kwargs):
        simulate_calls.append(
            (
                bool(request.collect_hit_tables),
                bool(request.build_intersection_cache),
            )
        )
        return SimpleNamespace(
            image=np.full((4, 4), 5.0, dtype=np.float64),
            hit_tables=[
                {
                    "side": "primary",
                    "hkl": (1, 0, 0),
                    "q_group_key": ("primary", 1, 0, 0),
                    "display_col": 5.0,
                    "display_row": 6.0,
                }
            ],
            intersection_cache=[],
            used_python_runner=False,
        )

    monkeypatch.setattr(
        runtime_session,
        "simulate_request",
        lambda *_args, **_kwargs: pytest.fail("miller runner should stay unused"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulate_qr_rods_request",
        _fake_simulate_qr_rods_request,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_copy_hit_tables",
        lambda tables: [dict(entry) for entry in tables],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_copy_intersection_cache_tables",
        lambda tables: list(tables),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_peak_table_payload",
        lambda intersection_cache, legacy_hit_tables: (
            list(intersection_cache) if intersection_cache else list(legacy_hit_tables)
        ),
        raising=False,
    )

    job = _make_runtime_simulation_generation_job(
        run_primary=True,
        run_secondary=False,
        collect_hit_tables=False,
        collect_primary_hit_tables=False,
        capture_primary_hit_tables_raw=True,
    )
    job["primary_data"] = {
        1: {
            "hk": (1, 0),
            "L": np.asarray([0.0], dtype=np.float64),
            "I": np.asarray([10.0], dtype=np.float64),
            "deg": 1,
        }
    }
    job["primary_source_mode"] = "qr"

    result = runtime_session._run_simulation_generation_job(job)

    assert simulate_calls == [(True, False)]
    assert result["primary_raw_rows_fresh"] is True
    assert result["primary_intersection_cache_built"] is False
    assert np.asarray(result["primary_best_sample_indices"]).size == 0
    assert result["primary_max_positions"] == [
        {
            "side": "primary",
            "hkl": (1, 0, 0),
            "q_group_key": ("primary", 1, 0, 0),
            "display_col": 5.0,
            "display_row": 6.0,
        }
    ]


def test_run_simulation_generation_job_supports_legacy_miller_fake_without_timing_fields(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    projection_debug = importlib.import_module("ra_sim.simulation.projection_debug")
    simulate_calls: list[bool] = []

    monkeypatch.setattr(runtime_session, "timing_enabled", lambda: True, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "timing_span",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runtime_session, "timing_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        projection_debug,
        "start_projection_debug_session",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        projection_debug,
        "finalize_projection_debug_session",
        lambda *_args, **_kwargs: None,
    )

    def _fake_simulate_request(request):
        simulate_calls.append(bool(request.collect_hit_tables))
        return SimpleNamespace(
            image=np.full((4, 4), 5.0, dtype=np.float64),
            hit_tables=[],
            intersection_cache=[],
            used_python_runner=False,
        )

    monkeypatch.setattr(runtime_session, "simulate_request", _fake_simulate_request, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "simulate_qr_rods_request",
        lambda *_args, **_kwargs: pytest.fail("qr runner should stay unused"),
        raising=False,
    )

    result = runtime_session._run_simulation_generation_job(
        _make_runtime_simulation_generation_job(run_primary=True, run_secondary=False)
    )

    assert simulate_calls == [False]
    assert np.asarray(result["primary_image"]).shape == (4, 4)


def test_run_simulation_generation_job_supports_legacy_qr_fake_without_timing_fields(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    projection_debug = importlib.import_module("ra_sim.simulation.projection_debug")
    simulate_calls: list[bool] = []

    monkeypatch.setattr(runtime_session, "timing_enabled", lambda: True, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "timing_span",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runtime_session, "timing_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        projection_debug,
        "start_projection_debug_session",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        projection_debug,
        "finalize_projection_debug_session",
        lambda *_args, **_kwargs: None,
    )

    def _fake_simulate_qr_rods_request(_data, request):
        simulate_calls.append(bool(request.collect_hit_tables))
        return SimpleNamespace(
            image=np.full((4, 4), 5.0, dtype=np.float64),
            hit_tables=[],
            intersection_cache=[],
            used_python_runner=False,
        )

    monkeypatch.setattr(
        runtime_session,
        "simulate_request",
        lambda *_args, **_kwargs: pytest.fail("miller runner should stay unused"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulate_qr_rods_request",
        _fake_simulate_qr_rods_request,
        raising=False,
    )
    job = _make_runtime_simulation_generation_job(run_primary=True, run_secondary=False)
    job["primary_data"] = {
        1: {
            "hk": (1, 0),
            "L": np.asarray([0.0], dtype=np.float64),
            "I": np.asarray([10.0], dtype=np.float64),
            "deg": 1,
        }
    }
    job["primary_source_mode"] = "qr"

    result = runtime_session._run_simulation_generation_job(job)

    assert simulate_calls == [False]
    assert np.asarray(result["primary_image"]).shape == (4, 4)


def test_raw_only_full_update_restores_qr_and_hkl_picker_rows(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_q_group_manager = importlib.import_module("ra_sim.gui.geometry_q_group_manager")
    manual_geometry = importlib.import_module("ra_sim.gui.manual_geometry")
    peak_selection = importlib.import_module("ra_sim.gui.peak_selection")
    runtime_primary_cache = importlib.import_module("ra_sim.gui.runtime_primary_cache")
    simulation_engine = importlib.import_module("ra_sim.simulation.engine")
    state_module = importlib.import_module("ra_sim.gui.state")
    projection_debug = importlib.import_module("ra_sim.simulation.projection_debug")

    def hit_row_count(tables):
        return sum(np.asarray(table).shape[0] for table in tables if np.asarray(table).ndim == 2)

    class TableIterable:
        def __init__(self, tables):
            self._tables = list(tables)

        def __iter__(self):
            return iter(self._tables)

        def __len__(self):
            return len(self._tables)

    primary_hit_table = np.asarray(
        [
            [1000.0, 42.5, 55.5, -8.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [900.0, 52.5, 65.5, 8.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    secondary_hit_table = np.asarray(
        [[800.0, 70.5, 80.5, 4.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    sanity_records = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        [primary_hit_table, secondary_hit_table],
        image_shape=(128, 128),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        peak_table_lattice=[(5.0, 6.0, "primary"), (7.0, 8.0, "secondary")],
        source_reflection_indices=[0, 1],
        primary_a=5.0,
        primary_c=6.0,
        default_source_label="primary",
        allow_nominal_hkl_indices=True,
    )
    assert sanity_records

    monkeypatch.setattr(
        projection_debug,
        "start_projection_debug_session",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        projection_debug,
        "finalize_projection_debug_session",
        lambda *_args, **_kwargs: None,
    )
    detector_cache_build_calls: list[str] = []

    def _fail_detector_cache_build(*_args, **_kwargs):
        detector_cache_build_calls.append("called")
        raise AssertionError("detector cache build helper should stay unused")

    monkeypatch.setattr(
        simulation_engine,
        "build_intersection_cache",
        _fail_detector_cache_build,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_primary_cache,
        "build_intersection_cache",
        _fail_detector_cache_build,
        raising=False,
    )
    simulate_calls: list[tuple[bool, bool, float]] = []

    def _fake_peak_runner(*args, **kwargs):
        a_value = float(args[3])
        image = np.array(args[6], copy=True)
        image[...] = a_value
        best_sample_indices_out = kwargs.get("best_sample_indices_out")
        if best_sample_indices_out is not None:
            best_sample_indices_out[:] = 0
        hit_tables = []
        if kwargs.get("collect_hit_tables"):
            hit_tables = [primary_hit_table if a_value < 6.0 else secondary_hit_table]
        return (
            image,
            hit_tables,
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            [],
        )

    def _simulate_request_from_main_run(request, **_kwargs):
        a_value = float(request.geometry.av)
        simulate_calls.append(
            (
                bool(request.collect_hit_tables),
                bool(request.build_intersection_cache),
                a_value,
            )
        )
        assert request.collect_hit_tables is True
        assert request.build_intersection_cache is False
        return simulation_engine.simulate(request, peak_runner=_fake_peak_runner)

    monkeypatch.setattr(
        runtime_session,
        "simulate_request",
        _simulate_request_from_main_run,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulate_qr_rods_request",
        lambda *_args, **_kwargs: pytest.fail("hidden QR rerun should stay unused"),
        raising=False,
    )

    result = runtime_session._run_simulation_generation_job(
        _make_runtime_simulation_generation_job(
            run_primary=True,
            run_secondary=True,
            collect_hit_tables=False,
            collect_primary_hit_tables=False,
            collect_secondary_hit_tables=False,
            build_primary_intersection_cache=False,
            build_secondary_intersection_cache=False,
            capture_primary_hit_tables_raw=True,
            capture_secondary_hit_tables_raw=True,
        )
    )

    assert simulate_calls == [(True, False, 5.0), (True, False, 7.0)]
    assert detector_cache_build_calls == []
    assert result["primary_intersection_cache_built"] is False
    assert result["secondary_intersection_cache_built"] is False
    assert result["primary_intersection_cache"] == []
    assert result["secondary_intersection_cache"] == []
    assert hit_row_count(result["primary_hit_tables_raw"]) > 0
    assert hit_row_count(result["secondary_hit_tables_raw"]) > 0
    assert hit_row_count(result["primary_max_positions"]) > 0
    assert hit_row_count(result["secondary_max_positions"]) > 0

    runtime_state = state_module.SimulationRuntimeState()
    geometry_state = state_module.GeometryRuntimeState()
    background_state = state_module.BackgroundRuntimeState()
    background_state.current_background_index = 0
    background_state.current_background_image = np.ones((128, 128), dtype=np.float64)
    runtime_state.stored_primary_intersection_cache = [np.zeros((1, 17), dtype=np.float64)]
    runtime_state.stored_secondary_intersection_cache = [np.zeros((1, 17), dtype=np.float64)]
    runtime_state.stored_primary_intersection_cache_signature = ("old-cache",)
    runtime_state.stored_secondary_intersection_cache_signature = ("old-cache",)
    runtime_state.source_row_snapshots[0] = {
        "simulation_signature": ("snapshot", "old"),
        "row_content_signature": ("old",),
        "rows": [],
        "row_count": 0,
        "valid_for_picker": False,
    }
    geometry_state.manual_pick_cache_signature = ("old-manual-cache",)
    geometry_state.manual_pick_cache_data = {
        "grouped_candidates": {},
        "simulated_lookup": {},
    }

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        geometry_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        background_state,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 128, raising=False)
    monkeypatch.setattr(runtime_session, "a_var", _RuntimeVar(5.0), raising=False)
    monkeypatch.setattr(runtime_session, "c_var", _RuntimeVar(6.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_bundle_detector_coords",
        lambda col, row: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_caked_coords",
        lambda col, row: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        lambda rows: [dict(entry) for entry in rows],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_active_caked_primary_view",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_caked_intersection_cache",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_trace_live_cache_event",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_signature_summary",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 5.0, "c": 6.0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_for_background",
        lambda _idx, _params=None: (
            "snapshot",
            runtime_state.stored_q_group_content_signature,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "filter_enabled_q_group_rows",
        lambda rows, _state: [dict(entry) for entry in rows],
        raising=False,
    )

    runtime_session._apply_ready_simulation_result(result)
    assert hit_row_count(runtime_state.stored_primary_max_positions) > 0
    assert hit_row_count(runtime_state.stored_secondary_max_positions) > 0
    assert runtime_state.stored_hit_table_signature == result["hit_table_signature"]
    runtime_session.weight2_var.set(1.0)

    previous_row_signature = runtime_state.stored_q_group_content_signature
    combined_diagnostics = runtime_session._publish_combined_simulation_state(
        image_size_value=128,
        primary_a_value=5.0,
        primary_c_value=6.0,
        secondary_a_value=7.0,
        secondary_c_value=8.0,
        active_peak_row_sides=result["active_peak_row_sides"],
    )
    assert combined_diagnostics["active_sides"] == ("primary", "secondary")
    assert combined_diagnostics["combined_row_count"] == 3
    assert combined_diagnostics["published_intersection_cache_row_count"] == 0
    assert runtime_state.stored_intersection_cache == []
    runtime_session._invalidate_peak_picker_caches(clear_source_snapshot=True)
    assert 0 not in runtime_state.source_row_snapshots

    restore_diagnostics = runtime_session._restore_live_peak_rows_from_combined_hit_tables(
        active_peak_row_sides=result["active_peak_row_sides"],
        primary_raw_rows_fresh=result["primary_raw_rows_fresh"],
        secondary_raw_rows_fresh=result["secondary_raw_rows_fresh"],
        previous_row_content_signature=previous_row_signature,
    )
    assert restore_diagnostics["input_row_count"] > 0
    assert restore_diagnostics["projected_row_count"] > 0
    assert restore_diagnostics["published_peak_record_count"] > 0
    assert restore_diagnostics["skipped_reason"] is None
    assert hit_row_count(runtime_state.stored_max_positions_local) > 0
    assert len(runtime_state.peak_records) > 0

    monkeypatch.setattr(
        runtime_session,
        "_build_live_preview_simulated_peaks_from_cache",
        lambda: [dict(entry) for entry in runtime_state.peak_records],
        raising=False,
    )
    runtime_session._capture_geometry_source_snapshot()
    snapshot = runtime_state.source_row_snapshots[0]
    assert snapshot["row_count"] > 0
    assert snapshot["valid_for_picker"] is True
    assert snapshot["row_content_signature"] == runtime_state.stored_q_group_content_signature

    cache_state = {"signature": None, "data": {}}
    projection_callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.ones((128, 128), dtype=float),
        current_background_native=lambda: np.ones((128, 128), dtype=float),
        image_size=lambda: 128,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        simulation_native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 0.25,
            float(row) + 0.5,
        ),
        filter_simulated_peaks=lambda rows: (list(rows or []), None, None),
        collapse_simulated_peaks=lambda rows, merge_radius_px=6.0: (
            list(rows or []),
            None,
        ),
        build_live_preview_simulated_peaks_from_cache=lambda: [
            dict(entry) for entry in runtime_state.peak_records
        ],
    )
    cache_callbacks = manual_geometry.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 1),
        current_background_index=lambda: 0,
        current_background_image=lambda: np.ones((128, 128), dtype=float),
        use_caked_space=projection_callbacks.pick_uses_caked_space,
        replace_cache_state=lambda signature, data: cache_state.update(
            {"signature": signature, "data": dict(data)}
        ),
        current_geometry_fit_params=lambda: {"a": 5.0, "c": 6.0},
        pairs_for_index=lambda _idx: [],
        source_rows_for_background=lambda *_args, **_kwargs: [
            dict(entry) for entry in runtime_state.source_row_snapshots[0]["rows"]
        ],
        simulated_peaks_for_params=projection_callbacks.simulated_peaks_for_params,
        build_grouped_candidates=projection_callbacks.pick_candidates,
        build_simulated_lookup=projection_callbacks.simulated_lookup,
        project_peaks_to_current_view=projection_callbacks.project_peaks_to_current_view,
        entry_display_coords=projection_callbacks.entry_display_coords,
        peak_records=lambda: [dict(entry) for entry in runtime_state.peak_records],
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_geometry_manual_pick_cache",
        cache_callbacks.get_pick_cache,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_hkl_pick_simulation_points_payload_cache",
        {},
        raising=False,
    )

    assert runtime_state.stored_intersection_cache == []
    cache_data = runtime_session._get_geometry_manual_pick_cache(
        param_set={"a": 5.0, "c": 6.0},
        prefer_cache=True,
    )
    grouped_candidates = cache_data["grouped_candidates"]
    simulated_lookup = cache_data["simulated_lookup"]
    assert sum(len(entries) for entries in grouped_candidates.values()) > 0
    assert simulated_lookup
    qr_group_key, qr_entries = next(iter(grouped_candidates.items()))
    qr_candidate = dict(qr_entries[0])

    def _candidate_pixel(candidate: Mapping[str, object]) -> tuple[float, float]:
        col = candidate.get("display_col", candidate.get("sim_col"))
        row = candidate.get("display_row", candidate.get("sim_row"))
        return float(col), float(row)

    def _assert_detector_pixels(candidate: Mapping[str, object]) -> None:
        assert candidate.get("q_group_key") is not None
        assert candidate.get("hkl") is not None
        assert np.isfinite(float(candidate["native_col"]))
        assert np.isfinite(float(candidate["native_row"]))
        assert np.isfinite(float(candidate["sim_col_raw"]))
        assert np.isfinite(float(candidate["sim_row_raw"]))
        has_sim_pixel = np.isfinite(float(candidate.get("sim_col", np.nan))) and np.isfinite(
            float(candidate.get("sim_row", np.nan))
        )
        has_display_pixel = np.isfinite(
            float(candidate.get("display_col", np.nan))
        ) and np.isfinite(float(candidate.get("display_row", np.nan)))
        assert has_sim_pixel or has_display_pixel

    _assert_detector_pixels(qr_candidate)
    assert qr_candidate.get("qr") is not None
    assert qr_candidate.get("qz") is not None
    branch_values = {
        int(entry["source_branch_index"]) for entry in qr_entries if "source_branch_index" in entry
    }
    assert branch_values == {0, 1}
    qr_col, qr_row = _candidate_pixel(qr_candidate)
    selected_group, selected_entries, selected_dist = (
        manual_geometry.geometry_manual_choose_group_at(
            grouped_candidates,
            qr_col,
            qr_row,
            window_size_px=20.0,
            use_caked_display=False,
        )
    )
    assert selected_group == qr_group_key
    assert selected_entries
    assert np.isfinite(float(selected_dist))

    payload = runtime_session._hkl_pick_simulation_points_from_qr_picker_cache()
    assert payload["source_signature"][0] == "grouped"
    assert payload["candidates"]
    hkl_candidate = dict(payload["candidates"][0])
    _assert_detector_pixels(hkl_candidate)
    hkl_col, hkl_row = _candidate_pixel(hkl_candidate)
    (
        nearest_idx,
        nearest_candidate,
        nearest_dist,
        within_window,
    ) = peak_selection._nearest_simulation_point_for_click(
        runtime_state,
        hkl_col,
        hkl_row,
        candidate_records=payload,
        max_axis_distance_px=20.0,
        use_caked_display=False,
    )
    assert nearest_idx >= -1
    assert nearest_candidate is not None
    assert tuple(nearest_candidate["hkl"]) == tuple(hkl_candidate["hkl"])
    assert np.isfinite(float(nearest_dist))
    assert within_window is True
    assert runtime_state.stored_intersection_cache == []


def test_selection_cache_update_builds_reduced_qr_hkl_picker_cache(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_q_group_manager = importlib.import_module("ra_sim.gui.geometry_q_group_manager")
    manual_geometry = importlib.import_module("ra_sim.gui.manual_geometry")
    peak_selection = importlib.import_module("ra_sim.gui.peak_selection")
    state_module = importlib.import_module("ra_sim.gui.state")
    projection_debug = importlib.import_module("ra_sim.simulation.projection_debug")

    primary_hit_table = np.asarray(
        [
            [1000.0, 42.5, 55.5, -8.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [900.0, 52.5, 65.5, 8.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    secondary_hit_table = np.asarray(
        [[800.0, 70.5, 80.5, 4.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    primary_intersection_cache = np.asarray(
        [
            [
                2.0,
                4.0,
                42.5,
                55.5,
                1000.0,
                -8.0,
                1.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                2.0,
                4.0,
                52.5,
                65.5,
                900.0,
                8.0,
                1.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )
    secondary_intersection_cache = np.asarray(
        [[1.0, 2.0, 70.5, 80.5, 800.0, 4.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    simulate_calls: list[tuple[bool, bool, float]] = []

    monkeypatch.setattr(projection_debug, "start_projection_debug_session", lambda *_a, **_k: None)
    monkeypatch.setattr(
        projection_debug, "finalize_projection_debug_session", lambda *_a, **_k: None
    )

    def _simulate_request_from_main_run(request, **_kwargs):
        a_value = float(request.geometry.av)
        simulate_calls.append(
            (
                bool(request.collect_hit_tables),
                bool(request.build_intersection_cache),
                a_value,
            )
        )
        assert request.collect_hit_tables is True
        assert request.build_intersection_cache is True
        request.best_sample_indices_out = np.asarray([0, 1], dtype=np.int64)
        if a_value < 6.0:
            hit_tables = [primary_hit_table]
            cache_tables = [primary_intersection_cache]
        else:
            hit_tables = [secondary_hit_table]
            cache_tables = [secondary_intersection_cache]
        return SimpleNamespace(
            image=np.full((4, 4), a_value, dtype=np.float64),
            hit_tables=hit_tables,
            intersection_cache=cache_tables,
            used_python_runner=False,
        )

    monkeypatch.setattr(
        runtime_session,
        "simulate_request",
        _simulate_request_from_main_run,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulate_qr_rods_request",
        lambda *_args, **_kwargs: pytest.fail("hidden simulation rerun should stay unused"),
        raising=False,
    )

    result = runtime_session._run_simulation_generation_job(
        _make_runtime_simulation_generation_job(
            run_primary=True,
            run_secondary=True,
            collect_hit_tables=True,
            collect_primary_hit_tables=True,
            collect_secondary_hit_tables=True,
            build_primary_intersection_cache=True,
            build_secondary_intersection_cache=True,
            capture_primary_hit_tables_raw=True,
            capture_secondary_hit_tables_raw=True,
        )
    )

    assert simulate_calls == [(True, True, 5.0), (True, True, 7.0)]
    assert result["primary_intersection_cache_built"] is True
    assert result["secondary_intersection_cache_built"] is True
    assert runtime_session._table_row_count(result["primary_intersection_cache"]) > 0
    assert runtime_session._table_row_count(result["secondary_intersection_cache"]) > 0

    runtime_state = state_module.SimulationRuntimeState()
    geometry_state = state_module.GeometryRuntimeState()
    background_state = state_module.BackgroundRuntimeState()
    background_state.current_background_index = 0
    background_state.current_background_image = np.ones((128, 128), dtype=np.float64)
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(runtime_session, "geometry_runtime_state", geometry_state, raising=False)
    monkeypatch.setattr(
        runtime_session, "background_runtime_state", background_state, raising=False
    )
    monkeypatch.setattr(runtime_session, "image_size", 128, raising=False)
    monkeypatch.setattr(runtime_session, "a_var", _RuntimeVar(5.0), raising=False)
    monkeypatch.setattr(runtime_session, "c_var", _RuntimeVar(6.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_clear_caked_intersection_cache",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_trace_live_cache_event",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_signature_summary",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 5.0, "c": 6.0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_for_background",
        lambda _idx, _params=None: (
            "snapshot",
            runtime_state.stored_q_group_content_signature,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "filter_enabled_q_group_rows",
        lambda rows, _state: [dict(entry) for entry in rows],
        raising=False,
    )

    runtime_session._apply_ready_simulation_result(result)
    assert (
        runtime_state.stored_primary_intersection_cache_signature == result["hit_table_signature"]
    )
    assert (
        runtime_state.stored_secondary_intersection_cache_signature == result["hit_table_signature"]
    )
    combined_diagnostics = runtime_session._publish_combined_simulation_state(
        image_size_value=128,
        primary_a_value=5.0,
        primary_c_value=6.0,
        secondary_a_value=7.0,
        secondary_c_value=8.0,
        active_peak_row_sides=result["active_peak_row_sides"],
    )
    assert combined_diagnostics["active_sides"] == ("primary", "secondary")
    assert combined_diagnostics["published_intersection_cache_row_count"] > 0
    assert runtime_session._table_row_count(runtime_state.stored_intersection_cache) > 0
    runtime_state.peak_records = [
        {
            "hkl": (9, 9, 9),
            "q_group_key": ("stale", 9, 9, 9),
            "native_col": 1.0,
            "native_row": 1.0,
            "display_col": 1.0,
            "display_row": 1.0,
        }
    ]

    q_group_state = SimpleNamespace(disabled_qr_sets=set(), disabled_qz_sections=set())
    value_callbacks = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=SimpleNamespace(excluded_keys=set(), overlay=SimpleNamespace(pairs=[])),
        q_group_state=q_group_state,
        fit_config={},
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 5.0,
        primary_c_factory=lambda: 6.0,
        image_size_factory=lambda: 128,
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        caked_view_enabled_factory=lambda: True,
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 0.25,
            float(row) + 0.5,
        ),
        project_peaks_to_current_view=lambda rows: [dict(entry) for entry in rows or ()],
    )
    projection_callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.ones((128, 128), dtype=float),
        last_caked_radial_values=lambda: np.linspace(0.0, 127.0, 128),
        last_caked_azimuth_values=lambda: np.linspace(0.0, 127.0, 128),
        current_background_display=lambda: np.ones((128, 128), dtype=float),
        current_background_native=lambda: np.ones((128, 128), dtype=float),
        image_size=lambda: 128,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        simulation_native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 0.25,
            float(row) + 0.5,
        ),
        filter_simulated_peaks=lambda rows: (list(rows or []), None, None),
        collapse_simulated_peaks=lambda rows, merge_radius_px=6.0: (
            list(rows or []),
            None,
        ),
        build_live_preview_simulated_peaks_from_cache=(
            value_callbacks.build_live_preview_simulated_peaks_from_cache
        ),
    )
    cache_state = {"signature": None, "data": {}}
    cache_callbacks = manual_geometry.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: result["signature"],
        current_background_index=lambda: 0,
        current_background_image=lambda: np.ones((128, 128), dtype=float),
        use_caked_space=projection_callbacks.pick_uses_caked_space,
        replace_cache_state=lambda signature, data: cache_state.update(
            {"signature": signature, "data": dict(data)}
        ),
        current_geometry_fit_params=lambda: {"a": 5.0, "c": 6.0},
        pairs_for_index=lambda _idx: [],
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=projection_callbacks.simulated_peaks_for_params,
        build_grouped_candidates=projection_callbacks.pick_candidates,
        build_simulated_lookup=projection_callbacks.simulated_lookup,
        project_peaks_to_current_view=projection_callbacks.project_peaks_to_current_view,
        entry_display_coords=projection_callbacks.entry_display_coords,
        peak_records=lambda: [],
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_geometry_manual_pick_cache",
        cache_callbacks.get_pick_cache,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_hkl_pick_simulation_points_payload_cache",
        {},
        raising=False,
    )

    cache_data = runtime_session._get_geometry_manual_pick_cache(
        param_set={"a": 5.0, "c": 6.0},
        prefer_cache=True,
    )
    assert value_callbacks.last_live_preview_cache_metadata()["cache_source"] == (
        "stored_intersection_cache"
    )
    grouped_candidates = cache_data["grouped_candidates"]
    assert sum(len(entries) for entries in grouped_candidates.values()) > 0
    primary_entries = [
        dict(entry)
        for entries in grouped_candidates.values()
        for entry in entries
        if tuple(entry.get("hkl", ())) == (1, 0, 2)
    ]
    assert len(primary_entries) >= 2
    branch_values = {
        int(entry["source_branch_index"])
        for entry in primary_entries
        if "source_branch_index" in entry
    }
    assert branch_values == {0, 1}

    qr_candidate = primary_entries[0]
    for key in ("qr", "qz", "hkl", "source_branch_index"):
        assert qr_candidate.get(key) is not None
    for key in (
        "native_col",
        "native_row",
        "sim_col",
        "sim_row",
        "sim_col_raw",
        "sim_row_raw",
        "display_col",
        "display_row",
    ):
        assert np.isfinite(float(qr_candidate[key]))
    caked_x = qr_candidate.get("caked_x", qr_candidate.get("raw_caked_x"))
    caked_y = qr_candidate.get("caked_y", qr_candidate.get("raw_caked_y"))
    assert np.isfinite(float(caked_x))
    assert np.isfinite(float(caked_y))

    qr_group_key = qr_candidate["q_group_key"]
    qr_col = float(qr_candidate["caked_x"])
    qr_row = float(qr_candidate["caked_y"])
    selected_group, selected_entries, selected_dist = (
        manual_geometry.geometry_manual_choose_group_at(
            grouped_candidates,
            qr_col,
            qr_row,
            window_size_px=20.0,
            use_caked_display=True,
        )
    )
    assert selected_group == qr_group_key
    assert selected_entries
    assert np.isfinite(float(selected_dist))

    payload = runtime_session._hkl_pick_simulation_points_from_qr_picker_cache()
    assert payload["candidates"]
    hkl_candidates = [
        dict(candidate)
        for candidate in payload["candidates"]
        if tuple(candidate.get("hkl", ())) == (1, 0, 2)
    ]
    assert hkl_candidates
    hkl_candidate = hkl_candidates[0]
    hkl_col = float(hkl_candidate["caked_x"])
    hkl_row = float(hkl_candidate["caked_y"])
    nearest_idx, nearest_candidate, nearest_dist, within_window = (
        peak_selection._nearest_simulation_point_for_click(
            runtime_state,
            hkl_col,
            hkl_row,
            candidate_records=payload,
            max_axis_distance_px=20.0,
            use_caked_display=True,
        )
    )
    assert nearest_idx >= -1
    assert nearest_candidate is not None
    assert tuple(nearest_candidate["hkl"]) == (1, 0, 2)
    assert np.isfinite(float(nearest_dist))
    assert within_window is True
    assert simulate_calls == [(True, True, 5.0), (True, True, 7.0)]


def test_restore_combined_detector_cache_refuses_stale_cache_after_raw_only_run(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_q_group_manager = importlib.import_module("ra_sim.gui.geometry_q_group_manager")
    manual_geometry = importlib.import_module("ra_sim.gui.manual_geometry")
    state_module = importlib.import_module("ra_sim.gui.state")

    primary_hit_table = np.asarray(
        [[1000.0, 42.5, 55.5, 0.0, 1.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    secondary_hit_table = np.asarray(
        [[800.0, 70.5, 80.5, 0.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    restored_rows = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        [primary_hit_table, secondary_hit_table],
        image_shape=(128, 128),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        peak_table_lattice=[(5.0, 6.0, "primary"), (7.0, 8.0, "secondary")],
        source_reflection_indices=[0, 1],
        primary_a=5.0,
        primary_c=6.0,
        default_source_label="primary",
        allow_nominal_hkl_indices=True,
    )
    assert restored_rows

    runtime_state = state_module.SimulationRuntimeState(
        stored_hit_table_signature=("fresh-hit-tables",),
        stored_primary_sim_image=np.ones((128, 128), dtype=np.float64),
        stored_secondary_sim_image=np.ones((128, 128), dtype=np.float64),
        stored_primary_intersection_cache=[np.ones((1, 17), dtype=np.float64)],
        stored_secondary_intersection_cache=[np.ones((1, 17), dtype=np.float64)],
        stored_primary_intersection_cache_signature=("fresh-hit-tables",),
        stored_secondary_intersection_cache_signature=("stale-detector-cache", "secondary"),
        stored_intersection_cache=[np.ones((1, 17), dtype=np.float64)],
        stored_max_positions_local=[primary_hit_table.copy(), secondary_hit_table.copy()],
        stored_q_group_content_signature=(
            geometry_q_group_manager._geometry_q_group_content_signature_from_hit_tables(
                [primary_hit_table, secondary_hit_table]
            )
        ),
        peak_records=[dict(entry) for entry in restored_rows],
    )
    runtime_state.source_row_snapshots[0] = {
        "simulation_signature": ("snapshot", runtime_state.stored_q_group_content_signature),
        "base_simulation_signature": ("sim", 1),
        "row_content_signature": runtime_state.stored_q_group_content_signature,
        "rows": [dict(entry) for entry in restored_rows],
        "row_count": len(restored_rows),
        "valid_for_picker": True,
        "empty_reason": None,
        "created_from": "raw_hit_tables",
    }
    geometry_state = state_module.GeometryRuntimeState()
    background_state = state_module.BackgroundRuntimeState()
    background_state.current_background_index = 0
    background_state.current_background_image = np.ones((128, 128), dtype=np.float64)

    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(runtime_session, "geometry_runtime_state", geometry_state, raising=False)
    monkeypatch.setattr(
        runtime_session, "background_runtime_state", background_state, raising=False
    )
    monkeypatch.setattr(runtime_session, "weight1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 128, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 5.0, "c": 6.0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_caked_intersection_cache",
        lambda: None,
        raising=False,
    )

    runtime_session._restore_combined_detector_intersection_cache(
        active_peak_row_sides=("primary", "secondary")
    )

    assert runtime_state.stored_intersection_cache == []

    projection_callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.ones((128, 128), dtype=float),
        current_background_native=lambda: np.ones((128, 128), dtype=float),
        image_size=lambda: 128,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        filter_simulated_peaks=lambda rows: (list(rows or []), None, None),
        collapse_simulated_peaks=lambda rows, merge_radius_px=6.0: (
            list(rows or []),
            None,
        ),
        build_live_preview_simulated_peaks_from_cache=lambda: [
            dict(entry) for entry in runtime_state.peak_records
        ],
    )
    cache_callbacks = manual_geometry.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 1),
        current_background_index=lambda: 0,
        current_background_image=lambda: np.ones((128, 128), dtype=float),
        use_caked_space=projection_callbacks.pick_uses_caked_space,
        replace_cache_state=lambda _signature, _data: None,
        current_geometry_fit_params=lambda: {"a": 5.0, "c": 6.0},
        pairs_for_index=lambda _idx: [],
        source_rows_for_background=lambda *_args, **_kwargs: [
            dict(entry) for entry in runtime_state.source_row_snapshots[0]["rows"]
        ],
        simulated_peaks_for_params=projection_callbacks.simulated_peaks_for_params,
        build_grouped_candidates=projection_callbacks.pick_candidates,
        build_simulated_lookup=projection_callbacks.simulated_lookup,
        project_peaks_to_current_view=projection_callbacks.project_peaks_to_current_view,
        entry_display_coords=projection_callbacks.entry_display_coords,
        peak_records=lambda: [dict(entry) for entry in runtime_state.peak_records],
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_geometry_manual_pick_cache",
        cache_callbacks.get_pick_cache,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_hkl_pick_simulation_points_payload_cache",
        {},
        raising=False,
    )

    cache_data = runtime_session._get_geometry_manual_pick_cache(
        param_set={"a": 5.0, "c": 6.0},
        prefer_cache=True,
    )
    assert sum(len(entries) for entries in cache_data["grouped_candidates"].values()) > 0

    payload = runtime_session._hkl_pick_simulation_points_from_qr_picker_cache()
    assert payload["candidates"]
    assert runtime_state.stored_intersection_cache == []


def test_partial_full_raw_refresh_does_not_publish_combined_picker_state(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    state_module = importlib.import_module("ra_sim.gui.state")

    primary_hit_table = np.asarray(
        [[1000.0, 42.5, 55.5, 0.0, 1.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    secondary_hit_table = np.asarray(
        [[800.0, 70.5, 80.5, 0.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    stale_peak_record = {
        "side": "secondary",
        "hkl": (0, 1, 0),
        "q_group_key": ("secondary", 1, 0, 0),
        "native_col": 70.5,
        "native_row": 80.5,
        "sim_col_raw": 70.5,
        "sim_row_raw": 80.5,
        "display_col": 70.5,
        "display_row": 80.5,
    }
    runtime_state = state_module.SimulationRuntimeState(
        stored_hit_table_signature=("old-signature",),
        stored_secondary_max_positions=[secondary_hit_table.copy()],
        stored_secondary_peak_table_lattice=[(7.0, 8.0, "secondary")],
        stored_secondary_source_reflection_indices=[1],
        stored_primary_intersection_cache=[np.zeros((1, 7), dtype=np.float64)],
        stored_secondary_intersection_cache=[np.zeros((1, 7), dtype=np.float64)],
        stored_primary_intersection_cache_signature=("old-signature",),
        stored_secondary_intersection_cache_signature=("old-signature",),
    )
    runtime_state.peak_records = [dict(stale_peak_record)]
    runtime_state.source_row_snapshots[0] = {
        "simulation_signature": ("old-snapshot",),
        "base_simulation_signature": ("old-base",),
        "row_content_signature": ("old-rows",),
        "rows": [dict(stale_peak_record)],
        "row_count": 1,
        "valid_for_picker": True,
        "empty_reason": None,
    }
    geometry_state = state_module.GeometryRuntimeState()
    geometry_state.manual_pick_cache_signature = ("old-picker",)
    geometry_state.manual_pick_cache_data = {
        "grouped_candidates": {("secondary", 1, 0, 0): [dict(stale_peak_record)]},
        "simulated_lookup": {("secondary", 1, 0, 0): dict(stale_peak_record)},
    }
    background_state = state_module.BackgroundRuntimeState()
    background_state.current_background_index = 0
    hkl_payload_cache = {"old": {"candidates": [dict(stale_peak_record)]}}
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        geometry_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        background_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_hkl_pick_simulation_points_payload_cache",
        hkl_payload_cache,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 128, raising=False)
    monkeypatch.setattr(runtime_session, "a_var", _RuntimeVar(5.0), raising=False)
    monkeypatch.setattr(runtime_session, "c_var", _RuntimeVar(6.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_cache_payload",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_trace_live_cache_event",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_count",
        lambda value: len(value) if isinstance(value, (list, tuple)) else 0,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_shape",
        lambda value: None if value is None else tuple(np.asarray(value).shape),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_signature_summary",
        lambda value: value,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_mark_numba_cache_compiled_artifacts_available",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_caked_intersection_cache",
        lambda: None,
        raising=False,
    )

    result = {
        "primary_image": np.ones((4, 4), dtype=np.float64),
        "secondary_image": np.ones((4, 4), dtype=np.float64),
        "run_primary": True,
        "run_secondary": True,
        "secondary_available": True,
        "active_peak_row_sides": ("primary", "secondary"),
        "primary_hit_table_state_refreshed": True,
        "secondary_hit_table_state_refreshed": False,
        "primary_raw_rows_fresh": True,
        "secondary_raw_rows_fresh": False,
        "primary_intersection_cache_built": False,
        "secondary_intersection_cache_built": False,
        "primary_max_positions": [primary_hit_table.copy()],
        "secondary_max_positions": [],
        "primary_peak_table_lattice": [(5.0, 6.0, "primary")],
        "secondary_peak_table_lattice": [],
        "primary_intersection_cache": [],
        "secondary_intersection_cache": [],
        "hit_table_signature": ("new-signature",),
        "is_preview": False,
        "primary_contribution_cache_signature": None,
        "primary_source_mode": "miller",
        "primary_contribution_keys": [],
        "active_primary_contribution_keys": [],
        "primary_hit_tables_raw": [],
        "primary_best_sample_indices": [],
        "image_generation_elapsed_ms": 1.0,
    }

    runtime_session._apply_ready_simulation_result(result)

    assert runtime_state.stored_hit_table_signature == ("old-signature",)
    assert runtime_state.stored_primary_intersection_cache is not None
    assert runtime_state.stored_primary_intersection_cache_signature is None
    assert runtime_state.stored_secondary_intersection_cache_signature == ("old-signature",)
    assert runtime_state.peak_records == []
    assert geometry_state.manual_pick_cache_signature is None
    assert geometry_state.manual_pick_cache_data == {}
    assert hkl_payload_cache == {}
    assert 0 not in runtime_state.source_row_snapshots
    runtime_state.stored_hit_table_signature = ("current-detector-cache",)
    runtime_state.stored_primary_intersection_cache_signature = ("current-detector-cache",)
    runtime_state.stored_secondary_intersection_cache_signature = ("old-signature",)
    runtime_session.weight2_var.set(1.0)
    mixed_cache_diagnostics = runtime_session._publish_combined_simulation_state(
        image_size_value=128,
        primary_a_value=5.0,
        primary_c_value=6.0,
        secondary_a_value=7.0,
        secondary_c_value=8.0,
        active_peak_row_sides=("primary", "secondary"),
    )
    assert mixed_cache_diagnostics["published_intersection_cache_row_count"] == 0
    assert runtime_state.stored_intersection_cache == []
    runtime_state.stored_hit_table_signature = ("old-signature",)
    runtime_state.stored_primary_intersection_cache_signature = None
    runtime_state.stored_secondary_intersection_cache_signature = ("old-signature",)
    runtime_session.weight2_var.set(1.0)

    diagnostics = runtime_session._publish_combined_simulation_state(
        image_size_value=128,
        primary_a_value=5.0,
        primary_c_value=6.0,
        secondary_a_value=7.0,
        secondary_c_value=8.0,
        active_peak_row_sides=(),
    )
    assert diagnostics["active_sides"] == ()
    assert diagnostics["combined_row_count"] == 0
    assert diagnostics["published_intersection_cache_row_count"] == 0
    assert runtime_state.stored_max_positions_local == []
    assert runtime_state.stored_intersection_cache == []

    restore_diagnostics = runtime_session._restore_live_peak_rows_from_combined_hit_tables(
        active_peak_row_sides=("primary", "secondary"),
        primary_raw_rows_fresh=True,
        secondary_raw_rows_fresh=False,
        previous_row_content_signature=None,
    )
    assert restore_diagnostics["skipped_reason"] == "stale_active_sides"
    assert runtime_state.peak_records == []
    assert geometry_state.manual_pick_cache_signature is None
    assert geometry_state.manual_pick_cache_data == {}


def test_combined_state_publisher_uses_serialized_active_peak_row_sides(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    state_module = importlib.import_module("ra_sim.gui.state")

    runtime_state = state_module.SimulationRuntimeState(
        stored_hit_table_signature=("current-hit-tables",),
        stored_primary_sim_image=np.ones((128, 128), dtype=np.float64),
        stored_secondary_sim_image=np.ones((128, 128), dtype=np.float64),
        stored_primary_max_positions=[
            np.asarray([[100.0, 10.0, 11.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        ],
        stored_secondary_max_positions=[
            np.asarray([[200.0, 20.0, 21.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
        ],
        stored_primary_peak_table_lattice=[(5.0, 6.0, "primary")],
        stored_secondary_peak_table_lattice=[(7.0, 8.0, "secondary")],
        stored_primary_intersection_cache=[np.ones((1, 17), dtype=np.float64)],
        stored_secondary_intersection_cache=[np.ones((1, 17), dtype=np.float64)],
        stored_primary_intersection_cache_signature=("current-hit-tables",),
        stored_secondary_intersection_cache_signature=("current-hit-tables",),
    )
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_clear_caked_intersection_cache",
        lambda: None,
        raising=False,
    )

    serialized_diagnostics = runtime_session._publish_combined_simulation_state(
        image_size_value=128,
        primary_a_value=5.0,
        primary_c_value=6.0,
        secondary_a_value=7.0,
        secondary_c_value=8.0,
        active_peak_row_sides=("primary", "secondary"),
    )
    assert serialized_diagnostics["active_sides"] == ("primary", "secondary")
    assert serialized_diagnostics["combined_row_count"] == 2
    assert serialized_diagnostics["published_intersection_cache_row_count"] == 2

    fallback_diagnostics = runtime_session._publish_combined_simulation_state(
        image_size_value=128,
        primary_a_value=5.0,
        primary_c_value=6.0,
        secondary_a_value=7.0,
        secondary_c_value=8.0,
    )
    assert fallback_diagnostics["active_sides"] == ("primary",)
    assert fallback_diagnostics["combined_row_count"] == 1


def test_combined_state_publisher_does_not_filter_serialized_sides_by_active_weight(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    state_module = importlib.import_module("ra_sim.gui.state")

    runtime_state = state_module.SimulationRuntimeState(
        stored_hit_table_signature=("current-hit-tables",),
        stored_primary_sim_image=np.ones((128, 128), dtype=np.float64),
        stored_secondary_sim_image=np.ones((128, 128), dtype=np.float64),
        stored_primary_max_positions=[
            np.asarray([[100.0, 10.0, 11.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        ],
        stored_secondary_max_positions=[
            np.asarray([[200.0, 20.0, 21.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
        ],
        stored_primary_peak_table_lattice=[(5.0, 6.0, "primary")],
        stored_secondary_peak_table_lattice=[(7.0, 8.0, "secondary")],
        stored_primary_intersection_cache=[np.ones((1, 17), dtype=np.float64)],
        stored_secondary_intersection_cache=[np.ones((1, 17), dtype=np.float64)],
        stored_primary_intersection_cache_signature=("current-hit-tables",),
        stored_secondary_intersection_cache_signature=("current-hit-tables",),
    )
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_clear_caked_intersection_cache",
        lambda: None,
        raising=False,
    )

    diagnostics = runtime_session._publish_combined_simulation_state(
        image_size_value=128,
        primary_a_value=5.0,
        primary_c_value=6.0,
        secondary_a_value=7.0,
        secondary_c_value=8.0,
        active_peak_row_sides=("primary", "secondary"),
    )

    assert diagnostics["active_sides"] == ("primary", "secondary")
    assert diagnostics["combined_row_count"] == 2
    assert diagnostics["published_intersection_cache_row_count"] == 2
    assert runtime_session._table_row_count(runtime_state.stored_intersection_cache) == 2
    assert runtime_state.stored_peak_table_lattice == [
        (5.0, 6.0, "primary"),
        (7.0, 8.0, "secondary"),
    ]


def test_combined_state_publisher_respects_empty_active_sides_and_none_signatures(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    state_module = importlib.import_module("ra_sim.gui.state")

    runtime_state = state_module.SimulationRuntimeState(
        stored_primary_sim_image=np.ones((128, 128), dtype=np.float64),
        stored_primary_max_positions=[
            np.asarray([[100.0, 10.0, 11.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        ],
        stored_primary_intersection_cache=[np.ones((1, 17), dtype=np.float64)],
        stored_primary_intersection_cache_signature=None,
        stored_hit_table_signature=None,
    )
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_clear_caked_intersection_cache",
        lambda: None,
        raising=False,
    )

    none_signature_diagnostics = runtime_session._publish_combined_simulation_state(
        image_size_value=128,
        primary_a_value=5.0,
        primary_c_value=6.0,
        secondary_a_value=7.0,
        secondary_c_value=8.0,
        active_peak_row_sides=("primary",),
    )
    assert none_signature_diagnostics["published_intersection_cache_row_count"] == 0
    assert runtime_state.stored_intersection_cache == []

    runtime_state.stored_hit_table_signature = ("current",)
    runtime_state.stored_primary_intersection_cache_signature = ("current",)
    empty_active_diagnostics = runtime_session._publish_combined_simulation_state(
        image_size_value=128,
        primary_a_value=5.0,
        primary_c_value=6.0,
        secondary_a_value=7.0,
        secondary_c_value=8.0,
        active_peak_row_sides=(),
    )
    assert empty_active_diagnostics["active_sides"] == ()
    assert empty_active_diagnostics["published_intersection_cache_row_count"] == 0
    assert runtime_state.stored_intersection_cache == []

    no_payload_diagnostics = runtime_session._publish_combined_simulation_state(
        image_size_value=128,
        primary_a_value=5.0,
        primary_c_value=6.0,
        secondary_a_value=7.0,
        secondary_c_value=8.0,
    )
    assert no_payload_diagnostics["active_sides"] == ("primary",)
    assert no_payload_diagnostics["published_intersection_cache_row_count"] == 1
    assert runtime_session._table_row_count(runtime_state.stored_intersection_cache) == 1


def test_raw_row_restore_clears_overlay_cache_when_projection_drops_all_rows(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    state_module = importlib.import_module("ra_sim.gui.state")

    runtime_state = state_module.SimulationRuntimeState(
        stored_max_positions_local=[
            np.asarray([[100.0, 10.0, 11.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        ],
        stored_q_group_content_signature=("rows",),
        peak_records=[{"stale": True}],
        peak_positions=[(1.0, 2.0)],
        peak_millers=[(9, 9, 9)],
        peak_intensities=[123.0],
        selected_peak_record={"stale": True},
        peak_overlay_cache={
            "records": [{"stale": True}],
            "positions": [(1.0, 2.0)],
            "click_spatial_index": {"stale": True},
        },
    )
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(runtime_session, "a_var", _RuntimeVar(5.0), raising=False)
    monkeypatch.setattr(runtime_session, "c_var", _RuntimeVar(6.0), raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 128, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_bundle_detector_coords",
        lambda col, row: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_caked_coords",
        lambda col, row: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        lambda rows: [dict(row) for row in rows or ()],
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_active_caked_primary_view", lambda: False, raising=False)
    monkeypatch.setattr(
        runtime_session.gui_geometry_q_group_manager,
        "build_projected_geometry_fit_simulated_peaks",
        lambda *_args, **_kwargs: [],
        raising=False,
    )

    diagnostics = runtime_session._restore_live_peak_rows_from_combined_hit_tables(
        active_peak_row_sides=("primary",),
        primary_raw_rows_fresh=True,
        secondary_raw_rows_fresh=False,
        previous_row_content_signature=("old",),
    )

    assert diagnostics["skipped_reason"] == "no_projected_rows"
    assert diagnostics["input_row_count"] > 0
    assert diagnostics["projected_row_count"] == 0
    assert runtime_state.peak_records == []
    assert runtime_state.peak_positions == []
    assert runtime_state.peak_millers == []
    assert runtime_state.peak_intensities == []
    assert runtime_state.selected_peak_record is None
    assert runtime_state.peak_overlay_cache["records"] == []
    assert runtime_state.peak_overlay_cache["positions"] == []
    assert runtime_state.peak_overlay_cache["click_spatial_index"] is None


def test_raw_row_restore_clears_overlay_cache_when_fresh_rows_are_empty(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    state_module = importlib.import_module("ra_sim.gui.state")

    runtime_state = state_module.SimulationRuntimeState(
        stored_max_positions_local=[],
        peak_records=[{"stale": True}],
        peak_positions=[(1.0, 2.0)],
        peak_millers=[(9, 9, 9)],
        peak_intensities=[123.0],
        selected_peak_record={"stale": True},
        peak_overlay_cache={
            "records": [{"stale": True}],
            "positions": [(1.0, 2.0)],
            "click_spatial_index": {"stale": True},
        },
    )
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)

    diagnostics = runtime_session._restore_live_peak_rows_from_combined_hit_tables(
        active_peak_row_sides=("primary",),
        primary_raw_rows_fresh=True,
        secondary_raw_rows_fresh=False,
        previous_row_content_signature=("old",),
    )

    assert diagnostics["skipped_reason"] == "no_combined_rows"
    assert diagnostics["input_row_count"] == 0
    assert runtime_state.peak_records == []
    assert runtime_state.peak_positions == []
    assert runtime_state.peak_millers == []
    assert runtime_state.peak_intensities == []
    assert runtime_state.selected_peak_record is None
    assert runtime_state.peak_overlay_cache["records"] == []
    assert runtime_state.peak_overlay_cache["positions"] == []
    assert runtime_state.peak_overlay_cache["click_spatial_index"] is None


def test_do_update_primary_fill_keeps_hit_table_signature_stale_until_all_run_sides_present(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=[],
        scheduled_settle_calls=[],
        apply_scale_factor_calls=[],
    )
    runtime_session.simulation_runtime_state.sim_miller1 = np.asarray(
        [[1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    runtime_session.simulation_runtime_state.sim_intens1 = np.asarray([10.0], dtype=np.float64)
    runtime_session.simulation_runtime_state.sim_miller2 = np.asarray(
        [[0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    runtime_session.simulation_runtime_state.sim_intens2 = np.asarray([5.0], dtype=np.float64)
    runtime_session.weight2_var.set(1.0)
    runtime_session.simulation_runtime_state.stored_primary_max_positions = None
    runtime_session.simulation_runtime_state.stored_secondary_max_positions = None
    runtime_session.simulation_runtime_state.stored_hit_table_signature = None
    stored_primary_cache_payload_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "_store_primary_cache_payload",
        lambda **kwargs: stored_primary_cache_payload_calls.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_rematerialize_primary_cache_artifacts",
        lambda **_kwargs: {
            "image": np.full((2, 2), 9.0, dtype=np.float64),
            "intersection_cache": [],
            "peak_tables": ["primary-peak"],
            "peak_table_lattice": [("primary",)],
        },
        raising=False,
    )

    def _apply_primary_cache_artifacts(payload: dict[str, object]) -> None:
        runtime_session.simulation_runtime_state.stored_primary_sim_image = np.asarray(
            payload["image"],
            dtype=np.float64,
        )
        runtime_session.simulation_runtime_state.stored_primary_max_positions = list(
            payload["peak_tables"]
        )
        runtime_session.simulation_runtime_state.stored_primary_peak_table_lattice = list(
            payload["peak_table_lattice"]
        )
        runtime_session.simulation_runtime_state.stored_primary_intersection_cache = list(
            payload["intersection_cache"]
        )

    monkeypatch.setattr(
        runtime_session,
        "_apply_primary_cache_artifacts",
        _apply_primary_cache_artifacts,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_consume_ready_simulation_result",
        lambda _sig: {
            "job_kind": "primary_fill",
            "primary_contribution_cache_signature": ("primary-cache", 1),
            "primary_source_mode": "miller",
            "active_primary_contribution_keys": [0],
            "primary_contribution_keys": [0],
            "primary_hit_tables_raw": [np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)],
            "primary_best_sample_indices": [0],
            "image_generation_elapsed_ms": 1.0,
        },
        raising=False,
    )

    runtime_session.do_update()

    assert stored_primary_cache_payload_calls
    assert runtime_session.simulation_runtime_state.stored_primary_max_positions == ["primary-peak"]
    assert runtime_session.simulation_runtime_state.stored_secondary_max_positions is None
    assert runtime_session.simulation_runtime_state.stored_hit_table_signature is None


def test_runtime_trace_records_classifier_display_decision(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=[],
        scheduled_settle_calls=[],
        apply_scale_factor_calls=[],
    )
    state = runtime_session.simulation_runtime_state
    state.last_simulation_signature = fixture["sim_signature"] + (0, 0)
    state.stored_primary_sim_image = np.ones((2, 2), dtype=np.float64)
    state.stored_secondary_sim_image = None

    trace_events: list[dict[str, object]] = []
    requested_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.do_update()

    assert requested_jobs == [], [
        event for event in trace_events if event["event"] == "do_update_signature"
    ]
    assert state.last_dependency_signatures is not None
    trace_events.clear()

    runtime_session.do_update()

    assert requested_jobs == [], [
        event for event in trace_events if event["event"] == "do_update_signature"
    ]
    complete_trace = next(
        event for event in reversed(trace_events) if event["event"] == "do_update_complete"
    )
    assert complete_trace["classifier_update_action"] == "display_only"
    assert complete_trace["classifier_update_reason"] == "no_dependency_change"
    assert complete_trace["classifier_requires_worker"] is False
    assert complete_trace["effective_update_action"] == "display_only"
    assert complete_trace["dependency_signatures_applied"] is True


def _patch_do_update_prune_fast_path_state(runtime_session) -> None:
    state = runtime_session.simulation_runtime_state
    state.sim_miller1_all = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    state.sim_intens1_all = np.asarray([10.0, 5.0, 2.0], dtype=np.float64)
    state.sim_miller2_all = np.empty((0, 3), dtype=np.float64)
    state.sim_intens2_all = np.empty((0,), dtype=np.float64)
    state.sim_primary_qr_all = {}
    state.sim_miller1 = state.sim_miller1_all[:2].copy()
    state.sim_intens1 = state.sim_intens1_all[:2].copy()
    state.sim_miller2 = np.empty((0, 3), dtype=np.float64)
    state.sim_intens2 = np.empty((0,), dtype=np.float64)
    state.primary_requested_source_mode = "miller"
    state.primary_requested_filter_signature = ("stable-source-filter",)
    state.primary_requested_contribution_keys = [0, 1]
    state.primary_active_contribution_keys = [0, 1]
    state.primary_contribution_cache_signature = ("primary-cache", 0.5)
    state.primary_source_mode = "miller"
    state.primary_hit_table_cache = {
        0: np.asarray([[10.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64),
        1: np.asarray([[5.0, 1.5, 0.5, 0.0, 1.0, 0.0, 1.0]], dtype=np.float64),
    }
    state.primary_best_sample_index_cache = {0: 0, 1: 0}
    state.stored_primary_sim_image = np.ones((2, 2), dtype=np.float64)
    state.stored_secondary_sim_image = None
    state.stored_sim_image = np.ones((2, 2), dtype=np.float64)
    state.peak_positions = []
    state.peak_millers = []
    state.peak_intensities = []
    state.peak_records = []
    state.selected_peak_record = None
    state.simulation_epoch = 0
    runtime_session.geometry_q_group_state = SimpleNamespace(
        refresh_requested=False,
        disabled_qr_sets=set(),
        disabled_qz_sections=set(),
    )
    runtime_session.geometry_runtime_state.manual_pick_armed = False
    runtime_session.peak_selection_state = SimpleNamespace(
        hkl_pick_armed=False,
        selected_hkl_target=None,
    )
    runtime_session._geometry_manual_pick_session_active = lambda: False


def _patch_prune_signature_function(monkeypatch, runtime_session) -> None:
    def _signature(
        _param_set,
        *,
        primary_source_signature,
        sf_prune_bias,
        **_kwargs,
    ):
        if (
            isinstance(primary_source_signature, tuple)
            and len(primary_source_signature) == 2
            and primary_source_signature[1] == ("stable-source-filter",)
        ):
            return ("primary-cache",)
        return (
            "broad-sim",
            primary_source_signature,
            round(float(sf_prune_bias), 3),
        )

    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_from_params",
        _signature,
        raising=False,
    )


def _patch_rich_mosaic_params(monkeypatch, runtime_session) -> None:
    monkeypatch.setattr(runtime_session, "lambda_", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "n2", 1.0 + 0.0j, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "build_mosaic_params",
        lambda: {
            "beam_x_array": np.asarray([0.0], dtype=np.float64),
            "beam_y_array": np.asarray([0.0], dtype=np.float64),
            "theta_array": np.asarray([0.0], dtype=np.float64),
            "phi_array": np.asarray([0.0], dtype=np.float64),
            "wavelength_array": np.asarray([1.0], dtype=np.float64),
            "sample_weights": None,
            "n2_sample_array": None,
            "_n2_sample_array_source": None,
            "_n2_sample_array_wavelength_snapshot": None,
            "sigma_mosaic_deg": 0.0,
            "gamma_mosaic_deg": 0.0,
            "eta": 0.0,
            "events_per_beam_phase": 50,
            "solve_q_steps": 7,
            "solve_q_rel_tol": 1.0e-6,
            "solve_q_mode": 2,
            "_sampling_signature": (),
        },
        raising=False,
    )


def test_do_update_prune_reuse_does_not_request_full_simulation_when_active_keys_cached(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(monkeypatch, runtime_session, fixture)
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=[],
        scheduled_settle_calls=[],
        apply_scale_factor_calls=[],
    )
    _patch_do_update_prune_fast_path_state(runtime_session)
    _patch_rich_mosaic_params(monkeypatch, runtime_session)
    state = runtime_session.simulation_runtime_state
    state.last_sim_signature = fixture["sim_signature"]
    state.last_simulation_signature = fixture["sim_signature"] + (0, 0)
    requested_jobs: list[dict[str, object]] = []
    rematerialize_calls: list[dict[str, object]] = []
    trace_events: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: trace_events.append({"event": event, **fields}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.do_update()
    assert state.last_dependency_signatures is not None
    trace_events.clear()

    _patch_prune_signature_function(monkeypatch, runtime_session)
    _patch_rich_mosaic_params(monkeypatch, runtime_session)
    state.sim_miller1 = state.sim_miller1_all[:1].copy()
    state.sim_intens1 = state.sim_intens1_all[:1].copy()
    state.primary_requested_contribution_keys = [0]
    state.last_sim_signature = ("seed-sim",)
    state.last_simulation_signature = ("seed-sim", 0, 0)

    def _rematerialize(**kwargs):
        rematerialize_calls.append(dict(kwargs))
        return {
            "image": np.full((2, 2), 4.0, dtype=np.float64),
            "intersection_cache": [],
            "peak_tables": [],
            "peak_table_lattice": [],
        }

    monkeypatch.setattr(
        runtime_session,
        "_rematerialize_primary_cache_artifacts",
        _rematerialize,
        raising=False,
    )

    runtime_session.do_update()

    assert requested_jobs == [], [
        event for event in trace_events if event["event"] == "do_update_signature"
    ]
    assert len(rematerialize_calls) == 1
    assert state.primary_active_contribution_keys == [0]
    complete_trace = next(
        event for event in reversed(trace_events) if event["event"] == "do_update_complete"
    )
    assert complete_trace["update_action"] == "primary_prune_reuse"
    assert complete_trace["requires_worker"] is False
    assert complete_trace["primary_prune_cache_mode"] == "reuse"


def test_do_update_prune_fill_requests_only_missing_contribution_keys(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(monkeypatch, runtime_session, fixture)
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=[],
        scheduled_settle_calls=[],
        apply_scale_factor_calls=[],
    )
    _patch_do_update_prune_fast_path_state(runtime_session)
    _patch_rich_mosaic_params(monkeypatch, runtime_session)
    state = runtime_session.simulation_runtime_state
    state.last_sim_signature = fixture["sim_signature"]
    state.last_simulation_signature = fixture["sim_signature"] + (0, 0)
    requested_jobs: list[dict[str, object]] = []
    requested_subset_keys: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.do_update()
    assert state.last_dependency_signatures is not None

    _patch_prune_signature_function(monkeypatch, runtime_session)
    _patch_rich_mosaic_params(monkeypatch, runtime_session)
    state.sim_miller1 = state.sim_miller1_all.copy()
    state.sim_intens1 = state.sim_intens1_all.copy()
    state.primary_requested_contribution_keys = [0, 1, 2]
    state.primary_hit_table_cache = {
        0: state.primary_hit_table_cache[0],
        2: np.asarray([[2.0, 1.0, 1.5, 0.0, 1.0, 0.0, 2.0]], dtype=np.float64),
    }
    state.primary_best_sample_index_cache = {0: 0, 2: 0}
    state.last_sim_signature = ("seed-sim",)
    state.last_simulation_signature = ("seed-sim", 0, 0)

    def _subset_payload(**kwargs):
        requested_subset_keys.append(tuple(kwargs["requested_keys"]))
        return {
            "primary_data": np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64),
            "primary_intensities": np.asarray([5.0], dtype=np.float64),
            "primary_contribution_keys": list(kwargs["requested_keys"]),
        }

    monkeypatch.setattr(
        runtime_session.gui_runtime_primary_cache,
        "build_primary_subset_payload",
        _subset_payload,
        raising=False,
    )

    runtime_session.do_update()

    assert requested_subset_keys == [(1,)]
    assert len(requested_jobs) == 1
    job = requested_jobs[0]
    assert job["job_kind"] == "primary_fill"
    assert job["primary_contribution_keys"] == [1]
    assert job["run_primary"] is True
    assert job["run_secondary"] is False


def test_runtime_session_manual_rebuild_failure_preserves_runtime_cache_state(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")
    state_module = importlib.import_module("ra_sim.gui.state")

    stale_rows = [
        {
            "display_col": 11.0,
            "display_row": 12.0,
            "sim_col": 11.0,
            "sim_row": 12.0,
            "hkl": (1, 0, 1),
            "q_group_key": ("q_group", "primary", 1, 1),
        }
    ]
    stale_hit_tables = [
        np.asarray(
            [[12.0, 10.0, 20.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    ]
    runtime_state = state_module.SimulationRuntimeState(
        stored_max_positions_local=list(stale_hit_tables),
        stored_intersection_cache=["old-cache"],
        stored_q_group_content_signature=("old-q-group-sig",),
        stored_source_reflection_indices_local=[99],
        stored_peak_table_lattice=[("old",)],
        stored_sim_image=np.full((4, 4), 5.0, dtype=np.float64),
        last_simulation_signature=("old-sig", 0),
        source_row_snapshots={
            0: {
                "rows": [dict(entry) for entry in stale_rows],
                "simulation_signature": ("old-sig", 0),
                "created_from": "old-cache",
            }
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        runtime_state,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 64, raising=False)

    peak_cache_calls: list[list[dict[str, object]]] = []
    diagnostics_calls: list[dict[str, object]] = []
    trace_calls: list[dict[str, object]] = []
    targeted_cache_store_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_set_runtime_peak_cache_from_projected_rows",
        lambda rows: peak_cache_calls.append([dict(entry) for entry in rows]),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_store_targeted_projected_cache_entry",
        lambda **kwargs: targeted_cache_store_calls.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_source_snapshot_diagnostics",
        lambda **kwargs: diagnostics_calls.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_trace_live_cache_event",
        lambda *_args, **kwargs: trace_calls.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )

    rebuild_result = geometry_fit.GeometryFitSourceRowRebuildResult(
        background_index=0,
        requested_signature=("new-sig", 1),
        requested_signature_summary="new-sig-1",
        projected_rows=[],
        stored_rows=[],
        rebuild_source=None,
        rebuild_attempts=["fresh_simulation"],
        diagnostics={
            "status": "snapshot_rebuild_failed",
            "consumer": "geometry_fit_dataset",
            "targeted_preflight_enabled": True,
            "required_hkl_branch_keys_digest": "targeted-digest",
            "requested_signature_summary": "new-sig-1",
        },
        peak_table_lattice=[("new",)],
        hit_tables=[np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)],
        source_reflection_indices=[1, 2],
        intersection_cache=["new-cache"],
        metadata={},
    )

    committed_rows = runtime_session._commit_geometry_manual_source_row_rebuild_result(
        rebuild_result
    )

    assert committed_rows == []
    assert peak_cache_calls == []
    assert targeted_cache_store_calls == []
    assert diagnostics_calls == [
        {
            "status": "snapshot_rebuild_failed",
            "consumer": "geometry_fit_dataset",
            "targeted_preflight_enabled": True,
            "required_hkl_branch_keys_digest": "targeted-digest",
            "requested_signature_summary": "new-sig-1",
        }
    ]
    assert trace_calls == [
        {
            "background_index": 0,
            "outcome": "failed",
            "consumer": "geometry_fit_dataset",
            "status": "snapshot_rebuild_failed",
            "requested_signature_summary": "new-sig-1",
            "raw_peak_count": 0,
            "projected_peak_count": 0,
            "rebuild_source": "unknown",
        }
    ]
    assert runtime_state.stored_max_positions_local == stale_hit_tables
    assert runtime_state.stored_intersection_cache == ["old-cache"]
    assert runtime_state.stored_q_group_content_signature == ("old-q-group-sig",)
    assert runtime_state.stored_source_reflection_indices_local == [99]
    assert runtime_state.stored_peak_table_lattice == [("old",)]
    assert runtime_state.last_simulation_signature == ("old-sig", 0)
    assert np.array_equal(runtime_state.stored_sim_image, np.full((4, 4), 5.0))
    assert runtime_state.source_row_snapshots[0]["created_from"] == "old-cache"


class _RuntimeVar:
    def __init__(self, value: float | bool) -> None:
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


def _install_idle_main_figure_preview_state(monkeypatch, runtime_session) -> None:
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        SimpleNamespace(
            _canvas_preview_limits=None,
            _canvas_pan_session=None,
        ),
        raising=False,
    )


def _install_matching_hidden_analysis_payload_state(
    monkeypatch,
    runtime_session,
    *,
    include_do_update_state: bool,
) -> dict[str, object]:
    sim_signature = ("sim-sig",)
    caked_geometry_sig = ("caked-geom",)
    q_space_payload_geometry_sig = ("q-geom",)
    sim_caking_sig = (sim_signature, caked_geometry_sig, 1.0)
    current_analysis_cache_sig = (
        (
            sim_caking_sig,
            int(runtime_session.DEFAULT_ANALYSIS_RADIAL_BINS),
            int(runtime_session.DEFAULT_ANALYSIS_AZIMUTH_BINS),
        ),
        None,
    )
    caked_image = np.ones((2, 2), dtype=np.float64)
    q_space_image = np.full((2, 2), 3.0, dtype=np.float64)
    state_fields = {
        "last_analysis_cache_sig": current_analysis_cache_sig,
        "last_q_space_payload_signature": (
            current_analysis_cache_sig,
            q_space_payload_geometry_sig,
        ),
        "last_caked_image_unscaled": caked_image.copy(),
        "last_caked_extent": [0.0, 1.0, -1.0, 1.0],
        "last_caked_background_image_unscaled": np.full((2, 2), 2.0, dtype=np.float64),
        "last_caked_radial_values": np.array([1.0, 2.0], dtype=np.float64),
        "last_caked_azimuth_values": np.array([-1.0, 1.0], dtype=np.float64),
        "last_caked_intersection_cache": ("cache",),
        "last_caked_intersection_cache_transform_bundle": "bundle",
        "last_q_space_image_unscaled": q_space_image.copy(),
        "last_q_space_qr_values": np.array([0.1, 0.2], dtype=np.float64),
        "last_q_space_qz_values": np.array([0.3, 0.4], dtype=np.float64),
        "last_q_space_extent": [-1.0, 1.0, -2.0, 2.0],
        "last_q_space_background_image_unscaled": np.full((2, 2), 4.0, dtype=np.float64),
        "caked_limits_user_override": False,
    }
    if include_do_update_state:
        state_fields.update(
            update_trace_counter=0,
            update_pending=None,
            update_running=False,
            update_phase="idle",
            worker_active_job=None,
            worker_queued_job=None,
            analysis_active_job=None,
            analysis_queued_job=None,
            preview_active=False,
            num_samples=0,
            sim_primary_qr={},
            sim_miller1=np.zeros((0, 3), dtype=np.float64),
            sim_intens1=np.zeros((0,), dtype=np.float64),
            sim_miller2=np.zeros((0, 3), dtype=np.float64),
            sim_intens2=np.zeros((0,), dtype=np.float64),
            primary_requested_source_mode="",
            primary_requested_contribution_keys=(),
            primary_requested_filter_signature=None,
            sf_prune_stats={},
            unscaled_image=np.ones((2, 2), dtype=np.float64),
            stored_primary_sim_image=np.ones((2, 2), dtype=np.float64),
            stored_secondary_sim_image=None,
            last_sim_signature=sim_signature,
            last_simulation_signature=sim_signature + (0,),
            stored_primary_max_positions=None,
            stored_secondary_max_positions=None,
            stored_primary_source_reflection_indices=None,
            stored_secondary_source_reflection_indices=None,
            stored_primary_peak_table_lattice=None,
            stored_secondary_peak_table_lattice=None,
            stored_primary_intersection_cache=None,
            stored_secondary_intersection_cache=None,
            normalization_scale_cache={"sig": None, "value": 1.0},
            peak_intensities=[],
            peak_records=[],
            last_1d_integration_data={},
            ai_cache={"sig": caked_geometry_sig, "ai": object()},
            stored_hit_table_signature=None,
            stored_sim_image=None,
            stored_max_positions_local=None,
            stored_source_reflection_indices_local=None,
            stored_peak_table_lattice=None,
            stored_intersection_cache=None,
            last_unscaled_image_signature=None,
        )
    state = SimpleNamespace(**state_fields)
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", state, raising=False)
    return {
        "simulation_runtime_state": state,
        "sim_signature": sim_signature,
        "caked_geometry_sig": caked_geometry_sig,
        "q_space_payload_geometry_sig": q_space_payload_geometry_sig,
        "current_analysis_cache_sig": current_analysis_cache_sig,
        "caked_image": caked_image,
        "q_space_image": q_space_image,
    }


def _patch_do_update_detector_cache_prereqs(monkeypatch, runtime_session, fixture) -> tuple:
    _install_idle_main_figure_preview_state(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(
            current_background_index=0,
            current_background_image=None,
            backend_rotation_k=0,
            backend_flip_x=False,
            backend_flip_y=False,
            visible=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(scale_factor_user_override=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            background_min_var=_RuntimeVar(0.0),
            background_max_var=_RuntimeVar(1.0),
            simulation_min_var=_RuntimeVar(0.0),
            simulation_max_var=_RuntimeVar(1.0),
            simulation_scale_factor_var=_RuntimeVar(1.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_caked_2d_var=_RuntimeVar(False), show_1d_var=_RuntimeVar(False)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "gamma_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "Gamma_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "chi_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "psi_z_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "zs_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "zb_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "sample_width_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "sample_length_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "sample_depth_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "cor_angle_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "a_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "c_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "debye_x_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "debye_y_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "corto_detector_var", _RuntimeVar(0.5), raising=False)
    monkeypatch.setattr(runtime_session, "center_x_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "center_y_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 2, raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0e-4, raising=False)
    monkeypatch.setattr(runtime_session, "wave_m", 1.24e-10, raising=False)
    monkeypatch.setattr(runtime_session, "psi", 0.0, raising=False)
    monkeypatch.setattr(runtime_session, "av2", None, raising=False)
    monkeypatch.setattr(runtime_session, "cv2", None, raising=False)
    monkeypatch.setattr(runtime_session, "geometry_q_group_state", object(), raising=False)
    monkeypatch.setattr(
        runtime_session, "peak_selection_runtime_maintenance", object(), raising=False
    )
    monkeypatch.setattr(runtime_session, "two_theta_range", (0.0, 2.0), raising=False)
    monkeypatch.setattr(runtime_session, "_last_a_for_ht", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "_last_c_for_ht", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "_last_iodine_z_for_ht", 0.0, raising=False)
    monkeypatch.setattr(runtime_session, "_last_phi_l_divisor", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "_last_phase_delta_expression", "", raising=False)
    monkeypatch.setattr(runtime_session, "_last_atom_site_fractional_signature", (), raising=False)
    monkeypatch.setattr(runtime_session, "_ensure_runtime_update_trace_hooks", lambda: None)
    monkeypatch.setattr(
        runtime_session, "_append_runtime_update_trace", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(runtime_session, "_refresh_run_status_bar", lambda: None)
    monkeypatch.setattr(
        runtime_session,
        "_clear_initial_simulation_loading_overlay",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "detector_two_theta_max", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(runtime_session, "_current_phase_delta_expression", lambda: "")
    monkeypatch.setattr(runtime_session, "_current_phi_l_divisor", lambda: 1.0)
    monkeypatch.setattr(runtime_session, "_current_iodine_z", lambda: 0.0)
    monkeypatch.setattr(runtime_session, "_current_atom_site_fractional_values", lambda: ())
    monkeypatch.setattr(runtime_session, "_atom_site_fractional_signature", lambda _values: ())
    monkeypatch.setattr(
        runtime_session, "_current_effective_theta_initial", lambda strict_count=False: 0.0
    )
    monkeypatch.setattr(runtime_session, "_current_ordered_structure_scale", lambda: 1.0)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "build_mosaic_params",
        lambda: {
            "sigma_mosaic_deg": 0.0,
            "gamma_mosaic_deg": 0.0,
            "eta": 0.0,
        },
    )
    monkeypatch.setattr(runtime_session, "_current_optics_mode_flag", lambda: 0)
    monkeypatch.setattr(
        runtime_session,
        "_qr_cylinder_replace_simulation_enabled",
        lambda: False,
    )
    monkeypatch.setattr(runtime_session, "_should_collect_hit_tables_for_update", lambda: False)
    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_from_params",
        lambda *_args, **_kwargs: fixture["sim_signature"],
    )
    monkeypatch.setattr(runtime_session, "current_sf_prune_bias", lambda: 0.0, raising=False)
    monkeypatch.setattr(
        runtime_session, "_cached_hit_tables_reusable", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(runtime_session, "_consume_ready_simulation_result", lambda _sig: None)
    monkeypatch.setattr(
        runtime_session, "_copy_intersection_cache_tables", lambda tables: list(tables)
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "consume_geometry_q_group_refresh_request",
        lambda _state: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_runtime_geometry_interaction,
        "refresh_runtime_peak_selection_after_update",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_geometry_preview_enabled",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_get_current_background_backend", lambda: None)
    monkeypatch.setattr(runtime_session, "_trace_live_cache_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_live_cache_signature_summary", lambda sig: sig)
    monkeypatch.setattr(runtime_session, "_set_scale_factor_value", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "_caked_geometry_cache_signature",
        lambda *_args, **_kwargs: fixture["caked_geometry_sig"],
    )
    monkeypatch.setattr(
        runtime_session,
        "_q_space_geometry_cache_signature",
        lambda **_kwargs: fixture["q_space_payload_geometry_sig"],
    )
    monkeypatch.setattr(
        runtime_session,
        "start_exact_cake_geometry_warmup_in_background",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "detector")
    monkeypatch.setattr(
        runtime_session,
        "_current_primary_figure_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "LIVE_DRAG_PREVIEW_ENABLED", False, raising=False)
    monkeypatch.setattr(runtime_session, "PREVIEW_CALCULATIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(runtime_session, "_analysis_integration_outputs_visible", lambda: False)
    monkeypatch.setattr(runtime_session, "_live_interaction_active", lambda: False)
    monkeypatch.setattr(
        runtime_session,
        "_clear_pending_main_figure_preview_interaction",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_defer_nonessential_redraw",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "ax", object(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_canvas_interactions,
        "capture_axis_limits",
        lambda *_args, **_kwargs: ((0.0, 2.0), (2.0, 0.0)),
        raising=False,
    )
    return ((0.0, 2.0), (2.0, 0.0))


def _patch_do_update_first_visible_simulation_finish_prereqs(
    monkeypatch,
    runtime_session,
    *,
    scheduled_post_idle_redraw_calls: list[str],
    scheduled_settle_calls: list[str],
    apply_scale_factor_calls: list[dict[str, object]],
    initial_detector_artist_signature: object | None = None,
    move_detector_artist_to_current_signature: bool = True,
) -> None:
    detector_artist_signature_state = {"current": initial_detector_artist_signature}

    monkeypatch.setattr(
        runtime_session,
        "_capture_geometry_source_snapshot",
        lambda: None,
        raising=False,
    )

    def _apply_primary_display(*_args, **_kwargs) -> str:
        if move_detector_artist_to_current_signature:
            detector_artist_signature_state["current"] = (
                runtime_session._detector_display_raster_source_signature()
            )
        return "detector"

    monkeypatch.setattr(
        runtime_session,
        "_apply_primary_figure_display_from_cached_results",
        _apply_primary_display,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_detector_artist_source_signature",
        lambda: detector_artist_signature_state["current"],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "apply_scale_factor_to_existing_results",
        lambda **kwargs: apply_scale_factor_calls.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_1d_plot_cache_and_lines",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_deferred_overlays",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "qr_cylinder_overlay_runtime_refresh",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_schedule_post_idle_main_canvas_redraw",
        lambda: scheduled_post_idle_redraw_calls.append("scheduled"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_schedule_first_visible_simulation_settle_pass",
        lambda: scheduled_settle_calls.append("scheduled"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label",
        SimpleNamespace(config=lambda **_kwargs: None, cget=lambda _name: ""),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "update_timing_label",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_schedule_exact_cake_numba_warmup_once",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_schedule_forward_simulation_numba_warmup_once",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_schedule_qr_rod_simulation_numba_warmup_once",
        lambda: None,
        raising=False,
    )


def _patch_apply_ready_simulation_result_to_store_first_visible_simulation(
    monkeypatch,
    runtime_session,
) -> list[dict[str, object]]:
    apply_ready_calls: list[dict[str, object]] = []

    def _apply_ready(result: dict[str, object]) -> None:
        apply_ready_calls.append(dict(result))
        runtime_session.simulation_runtime_state.stored_primary_sim_image = np.asarray(
            result["primary_image"],
            dtype=np.float64,
        )
        runtime_session.simulation_runtime_state.stored_secondary_sim_image = None
        runtime_session.simulation_runtime_state.stored_primary_max_positions = None
        runtime_session.simulation_runtime_state.stored_secondary_max_positions = None
        runtime_session.simulation_runtime_state.stored_primary_source_reflection_indices = None
        runtime_session.simulation_runtime_state.stored_secondary_source_reflection_indices = None
        runtime_session.simulation_runtime_state.stored_primary_peak_table_lattice = None
        runtime_session.simulation_runtime_state.stored_secondary_peak_table_lattice = None
        runtime_session.simulation_runtime_state.stored_primary_intersection_cache = None
        runtime_session.simulation_runtime_state.stored_secondary_intersection_cache = None

    monkeypatch.setattr(
        runtime_session,
        "_apply_ready_simulation_result",
        _apply_ready,
        raising=False,
    )
    return apply_ready_calls


def test_raw_only_full_update_restores_qr_and_hkl_picker_rows_job_builds_raw_only_requests(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(monkeypatch, runtime_session, fixture)
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=[],
        scheduled_settle_calls=[],
        apply_scale_factor_calls=[],
    )
    state = runtime_session.simulation_runtime_state
    state.sim_miller1 = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    state.sim_intens1 = np.asarray([10.0], dtype=np.float64)
    state.sim_miller2 = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64)
    state.sim_intens2 = np.asarray([5.0], dtype=np.float64)
    state.stored_primary_max_positions = None
    state.stored_secondary_max_positions = None
    state.stored_intersection_cache = []
    state.stored_hit_table_signature = ("stale-hit-tables",)
    state.last_sim_signature = fixture["sim_signature"]
    state.last_simulation_signature = fixture["sim_signature"] + (0, 0)
    state.peak_positions = []
    state.peak_millers = []
    state.selected_peak_record = None
    state.primary_contribution_cache_signature = None
    state.primary_source_mode = "miller"
    state.primary_active_contribution_keys = []
    state.primary_hit_table_cache = {}
    state.simulation_epoch = 1
    state.worker_ready_result = None
    state.worker_future = None
    state.worker_job_counter = 0
    state.worker_error_text = None
    runtime_session.weight2_var.set(1.0)
    monkeypatch.setattr(runtime_session, "_should_collect_hit_tables_for_update", lambda: True)
    monkeypatch.setattr(
        runtime_session,
        "_cached_hit_tables_reusable",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(runtime_session, "lambda_", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "n2", 1.0 + 0.0j, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "build_mosaic_params",
        lambda: {
            "beam_x_array": np.asarray([0.0], dtype=np.float64),
            "beam_y_array": np.asarray([0.0], dtype=np.float64),
            "theta_array": np.asarray([0.0], dtype=np.float64),
            "phi_array": np.asarray([0.0], dtype=np.float64),
            "wavelength_array": np.asarray([1.0], dtype=np.float64),
            "sample_weights": None,
            "n2_sample_array": None,
            "_n2_sample_array_source": None,
            "_n2_sample_array_wavelength_snapshot": None,
            "sigma_mosaic_deg": 0.0,
            "gamma_mosaic_deg": 0.0,
            "eta": 0.0,
            "solve_q_steps": 7,
            "solve_q_rel_tol": 1.0e-6,
            "solve_q_mode": 2,
            "_sampling_signature": (),
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_state",
        SimpleNamespace(
            refresh_requested=False,
            disabled_qr_sets=set(),
            disabled_qz_sections=set(),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.geometry_runtime_state,
        "manual_pick_armed",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "peak_selection_state",
        SimpleNamespace(hkl_pick_armed=False, selected_hkl_target=None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pick_session_active",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_runtime_primary_cache,
        "resolve_incremental_sf_prune_action",
        lambda **_kwargs: SimpleNamespace(
            mode="full",
            added_keys=(),
            removed_keys=(),
            missing_keys=(),
            reason="raw_only_picker_refresh",
        ),
        raising=False,
    )
    captured_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: captured_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.do_update()

    assert len(captured_jobs) == 1
    job = captured_jobs[0]
    assert job["job_kind"] == "full"
    assert job["collect_primary_hit_tables"] is True
    assert job["collect_secondary_hit_tables"] is True
    assert job["capture_primary_hit_tables_raw"] is True
    assert job["capture_secondary_hit_tables_raw"] is True
    assert job["build_primary_intersection_cache"] is False
    assert job["build_secondary_intersection_cache"] is False
    assert job["active_peak_row_sides"] == ("primary", "secondary")


def test_selection_cache_refresh_not_blocked_by_stored_raw_rows_or_peak_records(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(monkeypatch, runtime_session, fixture)
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=[],
        scheduled_settle_calls=[],
        apply_scale_factor_calls=[],
    )
    state = runtime_session.simulation_runtime_state
    primary_rows = np.asarray(
        [[1000.0, 42.5, 55.5, -8.0, 1.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    secondary_rows = np.asarray(
        [[800.0, 70.5, 80.5, 8.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    state.sim_miller1 = np.asarray([[1.0, 0.0, 2.0]], dtype=np.float64)
    state.sim_intens1 = np.asarray([10.0], dtype=np.float64)
    state.sim_miller2 = np.asarray([[0.0, 1.0, 1.0]], dtype=np.float64)
    state.sim_intens2 = np.asarray([5.0], dtype=np.float64)
    state.stored_primary_sim_image = np.ones((2, 2), dtype=np.float64)
    state.stored_secondary_sim_image = np.full((2, 2), 2.0, dtype=np.float64)
    state.stored_primary_max_positions = [primary_rows.copy()]
    state.stored_secondary_max_positions = [secondary_rows.copy()]
    state.stored_max_positions_local = [primary_rows.copy(), secondary_rows.copy()]
    state.stored_intersection_cache = []
    state.stored_primary_intersection_cache = []
    state.stored_secondary_intersection_cache = []
    state.stored_primary_intersection_cache_signature = ("stale-cache", "primary")
    state.stored_secondary_intersection_cache_signature = ("stale-cache", "secondary")
    state.stored_hit_table_signature = ("stale-hit-tables",)
    state.last_sim_signature = ("stale-image",)
    state.last_simulation_signature = ("stale-full",)
    state.peak_positions = [(42.5, 55.5), (70.5, 80.5)]
    state.peak_millers = [(1, 0, 2), (0, 1, 1)]
    state.peak_intensities = [1000.0, 800.0]
    state.peak_records = [
        {
            "hkl": (1, 0, 2),
            "qr": 2.0,
            "qz": 4.0,
            "q_group_key": ("primary", 1, 0, 2),
            "native_col": 42.5,
            "native_row": 55.5,
            "display_col": 42.5,
            "display_row": 55.5,
        }
    ]
    state.selected_peak_record = None
    state.primary_contribution_cache_signature = None
    state.primary_source_mode = "miller"
    state.primary_active_contribution_keys = []
    state.primary_hit_table_cache = {}
    state.simulation_epoch = 1
    state.worker_ready_result = None
    state.worker_future = None
    state.worker_job_counter = 0
    state.worker_error_text = None
    runtime_session.weight2_var.set(1.0)
    monkeypatch.setattr(
        runtime_session,
        "_cached_hit_tables_reusable",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(runtime_session, "_should_collect_hit_tables_for_update", lambda: False)
    monkeypatch.setattr(runtime_session, "lambda_", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "n2", 1.0 + 0.0j, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "build_mosaic_params",
        lambda: {
            "beam_x_array": np.asarray([0.0], dtype=np.float64),
            "beam_y_array": np.asarray([0.0], dtype=np.float64),
            "theta_array": np.asarray([0.0], dtype=np.float64),
            "phi_array": np.asarray([0.0], dtype=np.float64),
            "wavelength_array": np.asarray([1.0], dtype=np.float64),
            "sample_weights": None,
            "n2_sample_array": None,
            "_n2_sample_array_source": None,
            "_n2_sample_array_wavelength_snapshot": None,
            "sigma_mosaic_deg": 0.0,
            "gamma_mosaic_deg": 0.0,
            "eta": 0.0,
            "solve_q_steps": 7,
            "solve_q_rel_tol": 1.0e-6,
            "solve_q_mode": 2,
            "_sampling_signature": (),
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_state",
        SimpleNamespace(
            refresh_requested=True,
            disabled_qr_sets=set(),
            disabled_qz_sections=set(),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.geometry_runtime_state,
        "manual_pick_armed",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "peak_selection_state",
        SimpleNamespace(hkl_pick_armed=True, selected_hkl_target=None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_runtime_primary_cache,
        "resolve_incremental_sf_prune_action",
        lambda **_kwargs: SimpleNamespace(
            mode="full",
            added_keys=(),
            removed_keys=(),
            missing_keys=(),
            reason="selection_cache_refresh",
        ),
        raising=False,
    )
    submitted_jobs: list[dict[str, object]] = []

    def _submit_async(job: dict[str, object]) -> None:
        submitted_jobs.append(dict(job))
        state.worker_active_job = dict(job)
        state.worker_future = object()

    monkeypatch.setattr(
        runtime_session,
        "_submit_async_simulation_job",
        _submit_async,
        raising=False,
    )

    assert state.stored_max_positions_local
    assert state.peak_records
    assert state.stored_intersection_cache == []
    runtime_session.do_update()
    runtime_session.do_update()

    assert len(submitted_jobs) == 1
    job = submitted_jobs[0]
    assert job["job_kind"] == "full"
    assert job["active_peak_row_sides"] == ("primary", "secondary")
    assert job["collect_primary_hit_tables"] is True
    assert job["collect_secondary_hit_tables"] is True
    assert job["capture_primary_hit_tables_raw"] is True
    assert job["capture_secondary_hit_tables_raw"] is True
    assert job["build_primary_intersection_cache"] is True
    assert job["build_secondary_intersection_cache"] is True
    assert not (job["build_primary_intersection_cache"] and not job["collect_primary_hit_tables"])
    assert not (
        job["build_secondary_intersection_cache"] and not job["collect_secondary_hit_tables"]
    )


def test_do_update_selection_cache_refresh_omits_zero_weight_secondary(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(monkeypatch, runtime_session, fixture)
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=[],
        scheduled_settle_calls=[],
        apply_scale_factor_calls=[],
    )
    state = runtime_session.simulation_runtime_state
    state.sim_miller1 = np.asarray([[1.0, 0.0, 2.0]], dtype=np.float64)
    state.sim_intens1 = np.asarray([10.0], dtype=np.float64)
    state.sim_miller2 = np.asarray([[0.0, 1.0, 1.0]], dtype=np.float64)
    state.sim_intens2 = np.asarray([5.0], dtype=np.float64)
    state.stored_primary_sim_image = np.ones((2, 2), dtype=np.float64)
    state.stored_secondary_sim_image = np.full((2, 2), 2.0, dtype=np.float64)
    state.stored_primary_max_positions = [
        np.asarray([[1000.0, 42.5, 55.5, -8.0, 1.0, 0.0, 2.0]], dtype=np.float64)
    ]
    state.stored_secondary_max_positions = [
        np.asarray([[800.0, 70.5, 80.5, 8.0, 0.0, 1.0, 1.0]], dtype=np.float64)
    ]
    state.stored_intersection_cache = []
    state.stored_primary_intersection_cache = []
    state.stored_secondary_intersection_cache = []
    state.stored_hit_table_signature = ("stale-hit-tables",)
    state.last_sim_signature = ("stale-image",)
    state.last_simulation_signature = ("stale-full",)
    state.peak_positions = []
    state.peak_millers = []
    state.peak_intensities = []
    state.peak_records = []
    state.selected_peak_record = None
    state.primary_contribution_cache_signature = None
    state.primary_source_mode = "miller"
    state.primary_active_contribution_keys = []
    state.primary_hit_table_cache = {}
    state.simulation_epoch = 1
    state.worker_ready_result = None
    state.worker_future = None
    state.worker_job_counter = 0
    state.worker_error_text = None
    runtime_session.weight2_var.set(0.0)
    monkeypatch.setattr(
        runtime_session,
        "_cached_hit_tables_reusable",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(runtime_session, "_should_collect_hit_tables_for_update", lambda: False)
    monkeypatch.setattr(runtime_session, "lambda_", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "n2", 1.0 + 0.0j, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "build_mosaic_params",
        lambda: {
            "beam_x_array": np.asarray([0.0], dtype=np.float64),
            "beam_y_array": np.asarray([0.0], dtype=np.float64),
            "theta_array": np.asarray([0.0], dtype=np.float64),
            "phi_array": np.asarray([0.0], dtype=np.float64),
            "wavelength_array": np.asarray([1.0], dtype=np.float64),
            "sample_weights": None,
            "n2_sample_array": None,
            "_n2_sample_array_source": None,
            "_n2_sample_array_wavelength_snapshot": None,
            "sigma_mosaic_deg": 0.0,
            "gamma_mosaic_deg": 0.0,
            "eta": 0.0,
            "solve_q_steps": 7,
            "solve_q_rel_tol": 1.0e-6,
            "solve_q_mode": 2,
            "_sampling_signature": (),
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_state",
        SimpleNamespace(
            refresh_requested=True,
            disabled_qr_sets=set(),
            disabled_qz_sections=set(),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.geometry_runtime_state,
        "manual_pick_armed",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "peak_selection_state",
        SimpleNamespace(hkl_pick_armed=True, selected_hkl_target=None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_runtime_primary_cache,
        "resolve_incremental_sf_prune_action",
        lambda **_kwargs: SimpleNamespace(
            mode="full",
            added_keys=(),
            removed_keys=(),
            missing_keys=(),
            reason="selection_cache_refresh",
        ),
        raising=False,
    )
    requested_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.do_update()

    assert len(requested_jobs) == 1
    job = requested_jobs[0]
    assert job["active_peak_row_sides"] == ("primary",)
    assert job["run_primary"] is True
    assert job["run_secondary"] is False
    assert job["secondary_available"] is False
    assert job["collect_primary_hit_tables"] is True
    assert job["build_primary_intersection_cache"] is True
    assert job["capture_primary_hit_tables_raw"] is True
    assert job["collect_secondary_hit_tables"] is False
    assert job["build_secondary_intersection_cache"] is False
    assert job["capture_secondary_hit_tables_raw"] is False


def test_do_update_picker_refresh_reuses_current_raw_rows_without_hidden_rerun(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(monkeypatch, runtime_session, fixture)
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=[],
        scheduled_settle_calls=[],
        apply_scale_factor_calls=[],
    )
    state = runtime_session.simulation_runtime_state
    primary_rows = np.asarray(
        [[1000.0, 42.5, 55.5, -8.0, 1.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    secondary_rows = np.asarray(
        [[800.0, 70.5, 80.5, 8.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    state.sim_miller1 = np.asarray([[1.0, 0.0, 2.0]], dtype=np.float64)
    state.sim_intens1 = np.asarray([10.0], dtype=np.float64)
    state.sim_miller2 = np.asarray([[0.0, 1.0, 1.0]], dtype=np.float64)
    state.sim_intens2 = np.asarray([5.0], dtype=np.float64)
    state.stored_primary_sim_image = np.ones((2, 2), dtype=np.float64)
    state.stored_secondary_sim_image = np.full((2, 2), 2.0, dtype=np.float64)
    state.stored_primary_max_positions = [primary_rows.copy()]
    state.stored_secondary_max_positions = [secondary_rows.copy()]
    state.stored_max_positions_local = [primary_rows.copy(), secondary_rows.copy()]
    state.stored_intersection_cache = []
    state.stored_primary_intersection_cache = []
    state.stored_secondary_intersection_cache = []
    state.stored_primary_intersection_cache_signature = ("stale-cache", "primary")
    state.stored_secondary_intersection_cache_signature = ("stale-cache", "secondary")
    state.stored_hit_table_signature = ("current-hit-tables",)
    state.last_sim_signature = fixture["sim_signature"]
    state.last_simulation_signature = fixture["sim_signature"] + (0, 0)
    state.peak_positions = [(42.5, 55.5), (70.5, 80.5)]
    state.peak_millers = [(1, 0, 2), (0, 1, 1)]
    state.peak_records = [
        {
            "hkl": (1, 0, 2),
            "qr": 2.0,
            "qz": 4.0,
            "q_group_key": ("primary", 1, 0, 2),
            "native_col": 42.5,
            "native_row": 55.5,
            "display_col": 42.5,
            "display_row": 55.5,
        }
    ]
    state.selected_peak_record = None
    state.primary_contribution_cache_signature = None
    state.primary_source_mode = "miller"
    state.primary_active_contribution_keys = []
    state.primary_hit_table_cache = {}
    state.simulation_epoch = 1
    state.worker_ready_result = None
    state.worker_future = None
    state.worker_job_counter = 0
    state.worker_error_text = None
    runtime_session.weight2_var.set(1.0)
    monkeypatch.setattr(runtime_session, "_should_collect_hit_tables_for_update", lambda: False)
    monkeypatch.setattr(
        runtime_session,
        "_cached_hit_tables_reusable",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(runtime_session, "lambda_", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "n2", 1.0 + 0.0j, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "build_mosaic_params",
        lambda: {
            "beam_x_array": np.asarray([0.0], dtype=np.float64),
            "beam_y_array": np.asarray([0.0], dtype=np.float64),
            "theta_array": np.asarray([0.0], dtype=np.float64),
            "phi_array": np.asarray([0.0], dtype=np.float64),
            "wavelength_array": np.asarray([1.0], dtype=np.float64),
            "sample_weights": None,
            "n2_sample_array": None,
            "_n2_sample_array_source": None,
            "_n2_sample_array_wavelength_snapshot": None,
            "sigma_mosaic_deg": 0.0,
            "gamma_mosaic_deg": 0.0,
            "eta": 0.0,
            "solve_q_steps": 7,
            "solve_q_rel_tol": 1.0e-6,
            "solve_q_mode": 2,
            "_sampling_signature": (),
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_state",
        SimpleNamespace(
            refresh_requested=True,
            disabled_qr_sets=set(),
            disabled_qz_sections=set(),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.geometry_runtime_state,
        "manual_pick_armed",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "peak_selection_state",
        SimpleNamespace(hkl_pick_armed=True, selected_hkl_target=None),
        raising=False,
    )
    requested_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.do_update()

    assert len(requested_jobs) == 1
    job = requested_jobs[0]
    assert job["job_kind"] == "full"
    assert job["active_peak_row_sides"] == ("primary", "secondary")
    assert job["collect_primary_hit_tables"] is True
    assert job["collect_secondary_hit_tables"] is True
    assert job["build_primary_intersection_cache"] is True
    assert job["build_secondary_intersection_cache"] is True
    assert not (job["build_primary_intersection_cache"] and not job["collect_primary_hit_tables"])
    assert not (
        job["build_secondary_intersection_cache"] and not job["collect_secondary_hit_tables"]
    )
    assert state.stored_intersection_cache == []


def test_do_update_routes_hidden_payload_guard_into_primary_display_helper(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _StopAtPrimaryDisplayHelper(RuntimeError):
        pass

    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    preserved_primary_limits = _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    helper_calls: list[dict[str, object]] = []
    display_helper_calls: list[dict[str, object]] = []
    real_invalidate_hidden_payloads = (
        runtime_session._invalidate_hidden_analysis_space_payloads_if_stale
    )

    def _record_hidden_payload_guard(**kwargs) -> None:
        helper_calls.append(dict(kwargs))
        real_invalidate_hidden_payloads(**kwargs)

    def _stop_at_primary_display_helper(
        previous_primary_view_mode: str | None,
        preserved_primary_limits_arg: object,
    ) -> str:
        display_helper_calls.append(
            {
                "previous_primary_view_mode": previous_primary_view_mode,
                "preserved_primary_limits": preserved_primary_limits_arg,
                "caked_image": np.asarray(
                    runtime_session.simulation_runtime_state.last_caked_image_unscaled,
                    dtype=np.float64,
                ).copy(),
                "q_space_image": np.asarray(
                    runtime_session.simulation_runtime_state.last_q_space_image_unscaled,
                    dtype=np.float64,
                ).copy(),
            }
        )
        raise _StopAtPrimaryDisplayHelper()

    monkeypatch.setattr(
        runtime_session,
        "_invalidate_hidden_analysis_space_payloads_if_stale",
        _record_hidden_payload_guard,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_primary_figure_display_from_cached_results",
        _stop_at_primary_display_helper,
        raising=False,
    )

    with pytest.raises(_StopAtPrimaryDisplayHelper):
        runtime_session.do_update()

    assert helper_calls == [
        {
            "caked_analysis_requested": False,
            "q_space_requested": False,
            "current_analysis_cache_sig": fixture["current_analysis_cache_sig"],
            "q_space_payload_geometry_sig": fixture["q_space_payload_geometry_sig"],
        }
    ]
    assert len(display_helper_calls) == 1
    assert display_helper_calls[0]["previous_primary_view_mode"] == "detector"
    assert display_helper_calls[0]["preserved_primary_limits"] == preserved_primary_limits
    np.testing.assert_array_equal(
        display_helper_calls[0]["caked_image"],
        fixture["caked_image"],
    )
    np.testing.assert_array_equal(
        display_helper_calls[0]["q_space_image"],
        fixture["q_space_image"],
    )


def test_do_update_repairs_combined_peak_table_lattice_before_q_group_capture(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _StopAtQGroupCapture(RuntimeError):
        pass

    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )

    primary_table = np.asarray(
        [[12.0, 10.0, 20.0, 0.0, 1.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    secondary_table = np.asarray(
        [[9.0, 12.0, 24.0, 0.0, 1.0, 0.0, 3.0]],
        dtype=np.float64,
    )
    simulation_runtime_state = runtime_session.simulation_runtime_state
    simulation_runtime_state.stored_primary_sim_image = np.ones((2, 2), dtype=np.float64)
    simulation_runtime_state.stored_secondary_sim_image = np.full(
        (2, 2),
        2.0,
        dtype=np.float64,
    )
    simulation_runtime_state.stored_primary_max_positions = [primary_table]
    simulation_runtime_state.stored_secondary_max_positions = [secondary_table]
    simulation_runtime_state.stored_primary_source_reflection_indices = [101]
    simulation_runtime_state.stored_secondary_source_reflection_indices = [202]
    simulation_runtime_state.stored_primary_peak_table_lattice = [(4.0, 6.0, "primary")]
    simulation_runtime_state.stored_secondary_peak_table_lattice = []

    runtime_session.a_var.set(4.0)
    runtime_session.c_var.set(6.0)
    runtime_session.weight2_var.set(1.0)
    monkeypatch.setattr(runtime_session, "_last_a_for_ht", 4.0, raising=False)
    monkeypatch.setattr(runtime_session, "_last_c_for_ht", 6.0, raising=False)
    monkeypatch.setattr(runtime_session, "av2", 8.0, raising=False)
    monkeypatch.setattr(runtime_session, "cv2", 10.0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "progress_label_geometry",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "consume_geometry_q_group_refresh_request",
        lambda _state: True,
        raising=False,
    )
    state_module = importlib.import_module("ra_sim.gui.state")
    q_group_bundle = (
        runtime_session.gui_geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
            simulation_runtime_state=simulation_runtime_state,
            preview_state=state_module.GeometryPreviewState(),
            q_group_state=state_module.GeometryQGroupState(),
            fit_config=None,
            current_geometry_fit_var_names_factory=lambda: [],
            primary_a_factory=lambda: float(runtime_session.a_var.get()),
            primary_c_factory=lambda: float(runtime_session.c_var.get()),
            image_size_factory=lambda: int(runtime_session.image_size),
            native_sim_to_display_coords=lambda col, row, _shape: (
                float(col),
                float(row),
            ),
        )
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_runtime_bindings_factory",
        lambda: SimpleNamespace(build_entries_snapshot=q_group_bundle.build_entries_snapshot),
        raising=False,
    )

    captured: dict[str, object] = {}

    def _capture_and_stop(bindings) -> list[dict[str, object]]:
        captured["stored_max_positions_local"] = list(
            runtime_session.simulation_runtime_state.stored_max_positions_local or []
        )
        captured["stored_source_reflection_indices_local"] = list(
            runtime_session.simulation_runtime_state.stored_source_reflection_indices_local or []
        )
        captured["stored_peak_table_lattice"] = list(
            runtime_session.simulation_runtime_state.stored_peak_table_lattice or []
        )
        captured["entries"] = [dict(entry) for entry in bindings.build_entries_snapshot()]
        raise _StopAtQGroupCapture()

    monkeypatch.setattr(
        runtime_session.gui_geometry_q_group_manager,
        "capture_runtime_geometry_q_group_entries_snapshot",
        _capture_and_stop,
        raising=False,
    )

    with pytest.raises(_StopAtQGroupCapture):
        runtime_session.do_update()

    assert len(captured["stored_max_positions_local"]) == 2
    assert captured["stored_source_reflection_indices_local"] == [101, 202]
    assert captured["stored_peak_table_lattice"] == [
        (4.0, 6.0, "primary"),
        (8.0, 10.0, "secondary"),
    ]

    primary_key, primary_qr, primary_qz = (
        runtime_session.gui_geometry_q_group_manager.reflection_q_group_metadata(
            (1.0, 0.0, 2.0),
            source_label="primary",
            a_value=4.0,
            c_value=6.0,
            allow_nominal_hkl_indices=True,
        )
    )
    secondary_key, secondary_qr, secondary_qz = (
        runtime_session.gui_geometry_q_group_manager.reflection_q_group_metadata(
            (1.0, 0.0, 3.0),
            source_label="secondary",
            a_value=8.0,
            c_value=10.0,
            allow_nominal_hkl_indices=True,
        )
    )
    entries_by_source = {
        str(entry.get("source_label")): entry for entry in captured["entries"] or []
    }

    assert set(entries_by_source) == {"primary", "secondary"}
    assert entries_by_source["primary"]["key"] == primary_key
    assert entries_by_source["primary"]["peak_count"] == 1
    assert entries_by_source["primary"]["qr"] == pytest.approx(primary_qr)
    assert entries_by_source["primary"]["qz"] == pytest.approx(primary_qz)
    assert entries_by_source["secondary"]["key"] == secondary_key
    assert entries_by_source["secondary"]["peak_count"] == 1
    assert entries_by_source["secondary"]["qr"] == pytest.approx(secondary_qr)
    assert entries_by_source["secondary"]["qz"] == pytest.approx(secondary_qz)


def test_do_update_schedules_post_idle_redraw_when_worker_result_creates_first_visible_simulation(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    scheduled_post_idle_redraw_calls: list[str] = []
    scheduled_settle_calls: list[str] = []
    apply_scale_factor_calls: list[dict[str, object]] = []
    sync_selected_qr_rod_calls: list[str] = []
    ready_result = {
        "primary_image": np.ones((2, 2), dtype=np.float64),
        "secondary_image": np.zeros((2, 2), dtype=np.float64),
        "job_kind": "full",
        "image_generation_elapsed_ms": 1.25,
    }

    runtime_session.simulation_runtime_state.stored_primary_sim_image = None
    runtime_session.simulation_runtime_state.stored_secondary_sim_image = None
    runtime_session.simulation_runtime_state.stored_sim_image = None
    runtime_session.simulation_runtime_state.unscaled_image = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=scheduled_post_idle_redraw_calls,
        scheduled_settle_calls=scheduled_settle_calls,
        apply_scale_factor_calls=apply_scale_factor_calls,
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_selected_qr_rod_controls_state",
        lambda: sync_selected_qr_rod_calls.append("sync"),
        raising=False,
    )
    apply_ready_calls = _patch_apply_ready_simulation_result_to_store_first_visible_simulation(
        monkeypatch,
        runtime_session,
    )
    monkeypatch.setattr(
        runtime_session,
        "_consume_ready_simulation_result",
        lambda _sig: dict(ready_result),
        raising=False,
    )

    runtime_session.do_update()

    assert len(apply_ready_calls) == 1
    np.testing.assert_array_equal(
        runtime_session.simulation_runtime_state.stored_sim_image,
        ready_result["primary_image"],
    )
    assert apply_scale_factor_calls == [
        {
            "update_limits": False,
            "update_1d": False,
            "force_canvas_redraw": False,
            "update_chi_square": True,
        }
    ]
    assert sync_selected_qr_rod_calls == ["sync"]
    assert scheduled_post_idle_redraw_calls == ["scheduled"]
    assert scheduled_settle_calls == ["scheduled"]


def test_do_update_placeholder_simulation_does_not_block_first_real_detector_redraw(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    scheduled_post_idle_redraw_calls: list[str] = []
    scheduled_settle_calls: list[str] = []
    apply_scale_factor_calls: list[dict[str, object]] = []
    ready_result = {
        "primary_image": np.ones((2, 2), dtype=np.float64),
        "secondary_image": np.zeros((2, 2), dtype=np.float64),
        "job_kind": "full",
        "image_generation_elapsed_ms": 1.25,
    }

    runtime_session.simulation_runtime_state.stored_primary_sim_image = None
    runtime_session.simulation_runtime_state.stored_secondary_sim_image = None
    runtime_session.simulation_runtime_state.stored_sim_image = np.zeros(
        (2, 2),
        dtype=np.float64,
    )
    runtime_session.simulation_runtime_state.unscaled_image = None
    runtime_session.simulation_runtime_state.last_unscaled_image_signature = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=scheduled_post_idle_redraw_calls,
        scheduled_settle_calls=scheduled_settle_calls,
        apply_scale_factor_calls=apply_scale_factor_calls,
    )
    apply_ready_calls = _patch_apply_ready_simulation_result_to_store_first_visible_simulation(
        monkeypatch,
        runtime_session,
    )
    monkeypatch.setattr(
        runtime_session,
        "_consume_ready_simulation_result",
        lambda _sig: dict(ready_result),
        raising=False,
    )

    runtime_session.do_update()

    assert len(apply_ready_calls) == 1
    assert apply_scale_factor_calls == [
        {
            "update_limits": False,
            "update_1d": False,
            "force_canvas_redraw": False,
            "update_chi_square": True,
        }
    ]
    assert scheduled_post_idle_redraw_calls == ["scheduled"]
    assert scheduled_settle_calls == ["scheduled"]


def test_do_update_schedules_post_idle_redraw_when_detector_artist_first_shows_current_startup_simulation(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    scheduled_post_idle_redraw_calls: list[str] = []
    scheduled_settle_calls: list[str] = []
    apply_scale_factor_calls: list[dict[str, object]] = []

    runtime_session.simulation_runtime_state.unscaled_image = np.ones((2, 2), dtype=np.float64)
    runtime_session.simulation_runtime_state.last_unscaled_image_signature = ("startup-real",)
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=scheduled_post_idle_redraw_calls,
        scheduled_settle_calls=scheduled_settle_calls,
        apply_scale_factor_calls=apply_scale_factor_calls,
        initial_detector_artist_signature=("startup-placeholder",),
    )

    runtime_session.do_update()

    assert apply_scale_factor_calls == [
        {
            "update_limits": False,
            "update_1d": False,
            "force_canvas_redraw": False,
            "update_chi_square": True,
        }
    ]
    assert scheduled_post_idle_redraw_calls == ["scheduled"]
    assert scheduled_settle_calls == ["scheduled"]


def test_do_update_does_not_reschedule_post_idle_redraw_when_detector_artist_already_shows_current_simulation(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    scheduled_post_idle_redraw_calls: list[str] = []
    scheduled_settle_calls: list[str] = []
    apply_scale_factor_calls: list[dict[str, object]] = []

    runtime_session.simulation_runtime_state.unscaled_image = np.ones((2, 2), dtype=np.float64)
    runtime_session.simulation_runtime_state.last_unscaled_image_signature = ("startup-real",)
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None
    current_detector_signature = ("detector-current",)
    monkeypatch.setattr(
        runtime_session,
        "_detector_display_raster_source_signature",
        lambda: current_detector_signature,
        raising=False,
    )
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=scheduled_post_idle_redraw_calls,
        scheduled_settle_calls=scheduled_settle_calls,
        apply_scale_factor_calls=apply_scale_factor_calls,
        initial_detector_artist_signature=current_detector_signature,
    )

    runtime_session.do_update()

    assert apply_scale_factor_calls == [
        {
            "update_limits": False,
            "update_1d": False,
            "force_canvas_redraw": False,
            "update_chi_square": True,
        }
    ]
    assert scheduled_post_idle_redraw_calls == []
    assert scheduled_settle_calls == ["scheduled"]


def test_do_update_schedules_post_idle_redraw_for_first_sync_simulation(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    scheduled_post_idle_redraw_calls: list[str] = []
    scheduled_settle_calls: list[str] = []
    apply_scale_factor_calls: list[dict[str, object]] = []
    run_simulation_calls: list[dict[str, object]] = []
    sync_result = {
        "primary_image": np.ones((2, 2), dtype=np.float64),
        "secondary_image": np.zeros((2, 2), dtype=np.float64),
        "job_kind": "full",
        "image_generation_elapsed_ms": 2.5,
    }

    runtime_session.simulation_runtime_state.stored_primary_sim_image = None
    runtime_session.simulation_runtime_state.stored_secondary_sim_image = None
    runtime_session.simulation_runtime_state.stored_sim_image = None
    runtime_session.simulation_runtime_state.unscaled_image = None
    runtime_session.simulation_runtime_state.last_sim_signature = ("stale-sim",)
    runtime_session.simulation_runtime_state.last_simulation_signature = ("stale-sim", 0)
    runtime_session.simulation_runtime_state.simulation_epoch = 0
    runtime_session.simulation_runtime_state.peak_positions = []
    runtime_session.simulation_runtime_state.peak_millers = []
    runtime_session.simulation_runtime_state.peak_records = []
    runtime_session.simulation_runtime_state.selected_peak_record = None
    runtime_session.simulation_runtime_state.primary_contribution_cache_signature = None
    runtime_session.simulation_runtime_state.primary_source_mode = ""
    runtime_session.simulation_runtime_state.primary_active_contribution_keys = []
    runtime_session.simulation_runtime_state.primary_hit_table_cache = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None
    expected_n2 = np.array([0.95 + 0.01j], dtype=np.complex128)
    expected_source_meta = ("cif_path", "C:/optics/do-update-sync.cif")
    expected_wavelength_snapshot = np.array([runtime_session.wave_m], dtype=np.float64)
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=scheduled_post_idle_redraw_calls,
        scheduled_settle_calls=scheduled_settle_calls,
        apply_scale_factor_calls=apply_scale_factor_calls,
    )
    apply_ready_calls = _patch_apply_ready_simulation_result_to_store_first_visible_simulation(
        monkeypatch,
        runtime_session,
    )
    monkeypatch.setattr(
        runtime_session,
        "root",
        SimpleNamespace(update_idletasks=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "lambda_", runtime_session.wave_m, raising=False)
    monkeypatch.setattr(runtime_session, "n2", 1.0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_runtime_primary_cache,
        "resolve_incremental_sf_prune_action",
        lambda **_kwargs: runtime_session.gui_runtime_primary_cache.IncrementalSfPruneAction(
            mode="full",
            added_keys=(),
            removed_keys=(),
            missing_keys=(),
            reason="test",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "build_mosaic_params",
        lambda: {
            "beam_x_array": np.zeros((1,), dtype=np.float64),
            "beam_y_array": np.zeros((1,), dtype=np.float64),
            "theta_array": np.zeros((1,), dtype=np.float64),
            "phi_array": np.zeros((1,), dtype=np.float64),
            "wavelength_array": np.full((1,), runtime_session.wave_m, dtype=np.float64),
            "sample_weights": np.ones((1,), dtype=np.float64),
            "n2_sample_array": expected_n2.copy(),
            "_n2_sample_array_source": expected_source_meta,
            "_n2_sample_array_wavelength_snapshot": expected_wavelength_snapshot.copy(),
            "sigma_mosaic_deg": 0.0,
            "gamma_mosaic_deg": 0.0,
            "eta": 0.0,
            "solve_q_steps": 1,
            "solve_q_rel_tol": 1.0e-6,
            "solve_q_mode": 0,
            "_sampling_signature": ("sync-test",),
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_run_simulation_generation_job",
        lambda job: run_simulation_calls.append(dict(job)) or dict(sync_result),
        raising=False,
    )

    runtime_session.do_update()

    assert len(run_simulation_calls) == 1
    assert (
        run_simulation_calls[0]["mosaic_params"]["_n2_sample_array_source"] == expected_source_meta
    )
    np.testing.assert_array_equal(
        run_simulation_calls[0]["mosaic_params"]["n2_sample_array"],
        expected_n2,
    )
    np.testing.assert_array_equal(
        run_simulation_calls[0]["mosaic_params"]["_n2_sample_array_wavelength_snapshot"],
        expected_wavelength_snapshot,
    )
    assert len(apply_ready_calls) == 1
    np.testing.assert_array_equal(
        runtime_session.simulation_runtime_state.stored_sim_image,
        sync_result["primary_image"],
    )
    assert apply_scale_factor_calls == [
        {
            "update_limits": False,
            "update_1d": False,
            "force_canvas_redraw": False,
            "update_chi_square": True,
        }
    ]
    assert scheduled_post_idle_redraw_calls == ["scheduled"]
    assert scheduled_settle_calls == ["scheduled"]


def test_build_mosaic_params_preserves_profile_cache_n2_metadata(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    source_meta = ("cif_path", "C:/optics/profile-cache.cif")
    n2_sample_array = np.array([0.9 + 0.01j, 0.8 + 0.02j], dtype=np.complex128)
    wavelength_snapshot = np.array([1.1, 1.2], dtype=np.float64)
    monkeypatch.setattr(runtime_session, "update_mosaic_cache", lambda: None, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "current_solve_q_values",
        lambda: SimpleNamespace(steps=5, rel_tol=1.0e-6, mode_flag=2),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            profile_cache={
                "beam_x_array": np.array([0.0, 1.0], dtype=np.float64),
                "beam_y_array": np.array([0.5, 1.5], dtype=np.float64),
                "theta_array": np.array([0.1, 0.2], dtype=np.float64),
                "phi_array": np.array([0.3, 0.4], dtype=np.float64),
                "wavelength_array": wavelength_snapshot.copy(),
                "sample_weights": np.array([0.6, 0.4], dtype=np.float64),
                "n2_sample_array": n2_sample_array.copy(),
                "_n2_sample_array_source": source_meta,
                "_n2_sample_array_wavelength_snapshot": wavelength_snapshot.copy(),
                "_sampling_signature": ("cached", 2),
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "sigma_mosaic_var", _RuntimeVar(0.7), raising=False)
    monkeypatch.setattr(runtime_session, "gamma_mosaic_var", _RuntimeVar(0.8), raising=False)
    monkeypatch.setattr(runtime_session, "eta_var", _RuntimeVar(0.9), raising=False)

    mosaic = runtime_session.build_mosaic_params()

    assert mosaic["_n2_sample_array_source"] == source_meta
    np.testing.assert_array_equal(mosaic["n2_sample_array"], n2_sample_array)
    np.testing.assert_array_equal(
        mosaic["_n2_sample_array_wavelength_snapshot"],
        wavelength_snapshot,
    )


def test_runtime_session_source_keeps_n2_metadata_in_ordered_structure_job_clones() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    ordered_fit_start = source.index("def on_fit_ordered_structure_click(")
    ordered_fit_block = source[ordered_fit_start:]

    assert '"_n2_sample_array_source": mask_mosaic_params.get(' in ordered_fit_block
    assert 'mask_mosaic_params["_n2_sample_array_wavelength_snapshot"]' in ordered_fit_block
    assert '"_n2_sample_array_source": base_mosaic_params.get(' in ordered_fit_block
    assert 'base_mosaic_params["_n2_sample_array_wavelength_snapshot"]' in ordered_fit_block


def test_do_update_defers_first_visible_simulation_settle_while_preview_active_then_schedules_when_preview_finishes(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    scheduled_post_idle_redraw_calls: list[str] = []
    scheduled_settle_calls: list[str] = []
    apply_scale_factor_calls: list[dict[str, object]] = []
    ready_result = {
        "primary_image": np.ones((2, 2), dtype=np.float64),
        "secondary_image": np.zeros((2, 2), dtype=np.float64),
        "job_kind": "full",
        "image_generation_elapsed_ms": 1.25,
    }

    runtime_session.simulation_runtime_state.stored_primary_sim_image = None
    runtime_session.simulation_runtime_state.stored_secondary_sim_image = None
    runtime_session.simulation_runtime_state.stored_sim_image = None
    runtime_session.simulation_runtime_state.unscaled_image = None
    runtime_session.simulation_runtime_state.preview_active = True
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=scheduled_post_idle_redraw_calls,
        scheduled_settle_calls=scheduled_settle_calls,
        apply_scale_factor_calls=apply_scale_factor_calls,
    )
    _patch_apply_ready_simulation_result_to_store_first_visible_simulation(
        monkeypatch,
        runtime_session,
    )
    monkeypatch.setattr(
        runtime_session,
        "_consume_ready_simulation_result",
        lambda _sig: dict(ready_result),
        raising=False,
    )

    runtime_session.do_update()

    assert scheduled_post_idle_redraw_calls == ["scheduled"]
    assert scheduled_settle_calls == []

    runtime_session.simulation_runtime_state.preview_active = False
    monkeypatch.setattr(
        runtime_session,
        "_consume_ready_simulation_result",
        lambda _sig: None,
        raising=False,
    )

    runtime_session.do_update()

    assert scheduled_post_idle_redraw_calls == ["scheduled"]
    assert scheduled_settle_calls == ["scheduled"]


def test_do_update_still_schedules_first_visible_simulation_settle_while_analysis_preview_active(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    scheduled_post_idle_redraw_calls: list[str] = []
    scheduled_settle_calls: list[str] = []
    apply_scale_factor_calls: list[dict[str, object]] = []
    ready_result = {
        "primary_image": np.ones((2, 2), dtype=np.float64),
        "secondary_image": np.zeros((2, 2), dtype=np.float64),
        "job_kind": "full",
        "image_generation_elapsed_ms": 1.25,
    }

    runtime_session.simulation_runtime_state.stored_primary_sim_image = None
    runtime_session.simulation_runtime_state.stored_secondary_sim_image = None
    runtime_session.simulation_runtime_state.stored_sim_image = None
    runtime_session.simulation_runtime_state.unscaled_image = None
    runtime_session.simulation_runtime_state.preview_active = False
    runtime_session.simulation_runtime_state.analysis_preview_active = True
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=scheduled_post_idle_redraw_calls,
        scheduled_settle_calls=scheduled_settle_calls,
        apply_scale_factor_calls=apply_scale_factor_calls,
    )
    _patch_apply_ready_simulation_result_to_store_first_visible_simulation(
        monkeypatch,
        runtime_session,
    )
    monkeypatch.setattr(
        runtime_session,
        "_consume_ready_simulation_result",
        lambda _sig: dict(ready_result),
        raising=False,
    )
    runtime_session.do_update()

    assert scheduled_post_idle_redraw_calls == ["scheduled"]
    assert scheduled_settle_calls == ["scheduled"]


def test_do_update_defers_first_visible_simulation_settle_while_worker_job_pending_then_schedules_when_stable(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    scheduled_post_idle_redraw_calls: list[str] = []
    scheduled_settle_calls: list[str] = []
    apply_scale_factor_calls: list[dict[str, object]] = []
    ready_result = {
        "primary_image": np.ones((2, 2), dtype=np.float64),
        "secondary_image": np.zeros((2, 2), dtype=np.float64),
        "job_kind": "full",
        "image_generation_elapsed_ms": 1.25,
    }

    runtime_session.simulation_runtime_state.stored_primary_sim_image = None
    runtime_session.simulation_runtime_state.stored_secondary_sim_image = None
    runtime_session.simulation_runtime_state.stored_sim_image = None
    runtime_session.simulation_runtime_state.unscaled_image = None
    runtime_session.simulation_runtime_state.worker_queued_job = {"job_id": 17}
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=scheduled_post_idle_redraw_calls,
        scheduled_settle_calls=scheduled_settle_calls,
        apply_scale_factor_calls=apply_scale_factor_calls,
    )
    _patch_apply_ready_simulation_result_to_store_first_visible_simulation(
        monkeypatch,
        runtime_session,
    )
    monkeypatch.setattr(
        runtime_session,
        "_consume_ready_simulation_result",
        lambda _sig: dict(ready_result),
        raising=False,
    )

    runtime_session.do_update()

    assert scheduled_post_idle_redraw_calls == ["scheduled"]
    assert scheduled_settle_calls == []

    runtime_session.simulation_runtime_state.worker_queued_job = None
    monkeypatch.setattr(
        runtime_session,
        "_consume_ready_simulation_result",
        lambda _sig: None,
        raising=False,
    )

    runtime_session.do_update()

    assert scheduled_post_idle_redraw_calls == ["scheduled"]
    assert scheduled_settle_calls == ["scheduled"]


def test_do_update_still_schedules_first_visible_simulation_settle_while_analysis_job_pending(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    scheduled_post_idle_redraw_calls: list[str] = []
    scheduled_settle_calls: list[str] = []
    apply_scale_factor_calls: list[dict[str, object]] = []
    ready_result = {
        "primary_image": np.ones((2, 2), dtype=np.float64),
        "secondary_image": np.zeros((2, 2), dtype=np.float64),
        "job_kind": "full",
        "image_generation_elapsed_ms": 1.25,
    }

    runtime_session.simulation_runtime_state.stored_primary_sim_image = None
    runtime_session.simulation_runtime_state.stored_secondary_sim_image = None
    runtime_session.simulation_runtime_state.stored_sim_image = None
    runtime_session.simulation_runtime_state.unscaled_image = None
    runtime_session.simulation_runtime_state.analysis_active_job = {"job_id": 23}
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=scheduled_post_idle_redraw_calls,
        scheduled_settle_calls=scheduled_settle_calls,
        apply_scale_factor_calls=apply_scale_factor_calls,
    )
    _patch_apply_ready_simulation_result_to_store_first_visible_simulation(
        monkeypatch,
        runtime_session,
    )
    monkeypatch.setattr(
        runtime_session,
        "_consume_ready_simulation_result",
        lambda _sig: dict(ready_result),
        raising=False,
    )
    runtime_session.do_update()

    assert scheduled_post_idle_redraw_calls == ["scheduled"]
    assert scheduled_settle_calls == ["scheduled"]


def test_schedule_first_visible_simulation_settle_pass_reapplies_display_application_after_delay(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    scheduled_callbacks: list[object] = []
    cleared_tokens: list[object] = []
    events: list[tuple[str, object]] = []

    class _Root:
        def after(self, delay_ms, callback) -> str:
            scheduled_callbacks.append((delay_ms, callback))
            return f"after-token-{len(scheduled_callbacks)}"

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda _root, token: cleared_tokens.append(token),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "apply_scale_factor_to_existing_results",
        lambda **kwargs: events.append(("apply", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: events.append(("refresh", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda **kwargs: events.append(("redraw", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_flush_main_canvas_tk_present",
        lambda: events.append(("flush", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_interaction_active",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_defer_nonessential_redraw",
        lambda: False,
        raising=False,
    )
    runtime_session.simulation_runtime_state.unscaled_image = np.ones((2, 2), dtype=np.float64)
    runtime_session.simulation_runtime_state.last_unscaled_image_signature = ("sig-a", 1.0)
    runtime_session.simulation_runtime_state.preview_active = False
    runtime_session.simulation_runtime_state.worker_active_job = None
    runtime_session.simulation_runtime_state.worker_queued_job = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settle_token = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None

    runtime_session._schedule_first_visible_simulation_settle_pass()

    assert cleared_tokens == [None]
    assert len(scheduled_callbacks) == 1
    assert scheduled_callbacks[0][0] == getattr(runtime_session, "LIVE_DRAG_SETTLE_MS", 80)
    assert (
        runtime_session.simulation_runtime_state.first_visible_simulation_settle_token
        == "after-token-1"
    )
    assert events == []

    scheduled_callbacks[0][1]()

    assert events == [
        (
            "apply",
            {
                "update_limits": False,
                "update_1d": False,
                "update_canvas": False,
                "update_chi_square": False,
            },
        ),
        ("refresh", None),
        ("redraw", {"force": True}),
        ("flush", None),
    ]
    assert runtime_session.simulation_runtime_state.first_visible_simulation_settle_token is None
    assert runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature == (
        "sig-a",
        1.0,
    )


def test_schedule_first_visible_simulation_settle_pass_runs_immediately_when_after_missing(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    events: list[tuple[str, object]] = []

    class _Root:
        after = None

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "apply_scale_factor_to_existing_results",
        lambda **kwargs: events.append(("apply", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: events.append(("refresh", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda **kwargs: events.append(("redraw", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_flush_main_canvas_tk_present",
        lambda: events.append(("flush", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_defer_nonessential_redraw",
        lambda: False,
        raising=False,
    )
    runtime_session.simulation_runtime_state.unscaled_image = np.ones((2, 2), dtype=np.float64)
    runtime_session.simulation_runtime_state.last_unscaled_image_signature = ("sig-a", 1.0)
    runtime_session.simulation_runtime_state.first_visible_simulation_settle_token = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None

    runtime_session._schedule_first_visible_simulation_settle_pass()

    assert events == [
        (
            "apply",
            {
                "update_limits": False,
                "update_1d": False,
                "update_canvas": False,
                "update_chi_square": False,
            },
        ),
        ("refresh", None),
        ("redraw", {"force": True}),
        ("flush", None),
    ]
    assert runtime_session.simulation_runtime_state.first_visible_simulation_settle_token is None
    assert runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature == (
        "sig-a",
        1.0,
    )


def test_schedule_first_visible_simulation_settle_pass_runs_immediately_when_after_raises(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    events: list[tuple[str, object]] = []

    class _Root:
        def after(self, _delay_ms, _callback) -> None:
            raise RuntimeError("after-unavailable")

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "apply_scale_factor_to_existing_results",
        lambda **kwargs: events.append(("apply", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: events.append(("refresh", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda **kwargs: events.append(("redraw", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_flush_main_canvas_tk_present",
        lambda: events.append(("flush", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_defer_nonessential_redraw",
        lambda: False,
        raising=False,
    )
    runtime_session.simulation_runtime_state.unscaled_image = np.ones((2, 2), dtype=np.float64)
    runtime_session.simulation_runtime_state.last_unscaled_image_signature = ("sig-a", 1.0)
    runtime_session.simulation_runtime_state.first_visible_simulation_settle_token = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None

    runtime_session._schedule_first_visible_simulation_settle_pass()

    assert events == [
        (
            "apply",
            {
                "update_limits": False,
                "update_1d": False,
                "update_canvas": False,
                "update_chi_square": False,
            },
        ),
        ("refresh", None),
        ("redraw", {"force": True}),
        ("flush", None),
    ]
    assert runtime_session.simulation_runtime_state.first_visible_simulation_settle_token is None
    assert runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature == (
        "sig-a",
        1.0,
    )


def test_schedule_first_visible_simulation_settle_pass_noops_for_stale_signature(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    scheduled_callbacks: list[object] = []
    events: list[tuple[str, object]] = []

    class _Root:
        def after(self, delay_ms, callback) -> str:
            scheduled_callbacks.append((delay_ms, callback))
            return "after-token-1"

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "apply_scale_factor_to_existing_results",
        lambda **kwargs: events.append(("apply", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: events.append(("refresh", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda **kwargs: events.append(("redraw", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_flush_main_canvas_tk_present",
        lambda: events.append(("flush", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_interaction_active",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_defer_nonessential_redraw",
        lambda: False,
        raising=False,
    )
    runtime_session.simulation_runtime_state.unscaled_image = np.ones((2, 2), dtype=np.float64)
    runtime_session.simulation_runtime_state.last_unscaled_image_signature = ("sig-a", 1.0)
    runtime_session.simulation_runtime_state.preview_active = False
    runtime_session.simulation_runtime_state.worker_active_job = None
    runtime_session.simulation_runtime_state.worker_queued_job = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settle_token = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None

    runtime_session._schedule_first_visible_simulation_settle_pass()
    runtime_session.simulation_runtime_state.last_unscaled_image_signature = ("sig-b", 1.0)

    scheduled_callbacks[0][1]()

    assert events == []
    assert runtime_session.simulation_runtime_state.first_visible_simulation_settle_token is None
    assert (
        runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature is None
    )


def test_schedule_first_visible_simulation_settle_pass_replaces_pending_token_for_new_signature(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    scheduled_callbacks: list[object] = []
    cleared_tokens: list[object] = []
    events: list[tuple[str, object]] = []

    class _Root:
        def after(self, delay_ms, callback) -> str:
            scheduled_callbacks.append((delay_ms, callback))
            return f"after-token-{len(scheduled_callbacks)}"

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda _root, token: cleared_tokens.append(token),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "apply_scale_factor_to_existing_results",
        lambda **kwargs: events.append(("apply", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: events.append(("refresh", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda **kwargs: events.append(("redraw", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_flush_main_canvas_tk_present",
        lambda: events.append(("flush", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_interaction_active",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_defer_nonessential_redraw",
        lambda: False,
        raising=False,
    )
    runtime_session.simulation_runtime_state.unscaled_image = np.ones((2, 2), dtype=np.float64)
    runtime_session.simulation_runtime_state.preview_active = False
    runtime_session.simulation_runtime_state.worker_active_job = None
    runtime_session.simulation_runtime_state.worker_queued_job = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settle_token = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None

    runtime_session.simulation_runtime_state.last_unscaled_image_signature = ("sig-a", 1.0)
    runtime_session._schedule_first_visible_simulation_settle_pass()
    runtime_session.simulation_runtime_state.last_unscaled_image_signature = ("sig-b", 1.0)
    runtime_session._schedule_first_visible_simulation_settle_pass()

    assert cleared_tokens == [None, "after-token-1"]
    assert (
        runtime_session.simulation_runtime_state.first_visible_simulation_settle_token
        == "after-token-2"
    )

    scheduled_callbacks[0][1]()
    assert events == []

    scheduled_callbacks[1][1]()
    assert events == [
        (
            "apply",
            {
                "update_limits": False,
                "update_1d": False,
                "update_canvas": False,
                "update_chi_square": False,
            },
        ),
        ("refresh", None),
        ("redraw", {"force": True}),
        ("flush", None),
    ]
    assert runtime_session.simulation_runtime_state.first_visible_simulation_settle_token is None
    assert runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature == (
        "sig-b",
        1.0,
    )


def test_schedule_first_visible_simulation_settle_pass_retries_once_before_force_redraw_fallback(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    scheduled_callbacks: list[object] = []
    events: list[tuple[str, object]] = []
    exception_trace_calls: list[dict[str, object]] = []

    class _Root:
        def after(self, delay_ms, callback) -> str:
            scheduled_callbacks.append((delay_ms, callback))
            return f"after-token-{len(scheduled_callbacks)}"

    def _fail_apply(**kwargs) -> None:
        events.append(("apply", dict(kwargs)))
        raise RuntimeError("settle-failed")

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "apply_scale_factor_to_existing_results",
        _fail_apply,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_settled_overlays",
        lambda: events.append(("refresh", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_legacy_main_matplotlib_redraw",
        lambda **kwargs: events.append(("redraw", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_flush_main_canvas_tk_present",
        lambda: events.append(("flush", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_defer_nonessential_redraw",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_exception_trace",
        lambda event, exc_type, exc_value, exc_tb, **fields: exception_trace_calls.append(
            {
                "event": event,
                "exc_type": getattr(exc_type, "__name__", str(exc_type)),
                "error": str(exc_value),
                **dict(fields),
            }
        ),
        raising=False,
    )
    runtime_session.simulation_runtime_state.unscaled_image = np.ones((2, 2), dtype=np.float64)
    runtime_session.simulation_runtime_state.last_unscaled_image_signature = ("sig-a", 1.0)
    runtime_session.simulation_runtime_state.first_visible_simulation_settle_token = None
    runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature = None

    runtime_session._schedule_first_visible_simulation_settle_pass()

    assert len(scheduled_callbacks) == 1
    assert events == []
    assert exception_trace_calls == []

    scheduled_callbacks[0][1]()

    assert len(scheduled_callbacks) == 2
    assert events == [
        (
            "apply",
            {
                "update_limits": False,
                "update_1d": False,
                "update_canvas": False,
                "update_chi_square": False,
            },
        )
    ]
    assert exception_trace_calls == [
        {
            "event": "first_visible_simulation_settle_failed",
            "exc_type": "RuntimeError",
            "error": "settle-failed",
            "attempt": 1,
            "retry_allowed": True,
        }
    ]

    scheduled_callbacks[1][1]()

    assert events == [
        (
            "apply",
            {
                "update_limits": False,
                "update_1d": False,
                "update_canvas": False,
                "update_chi_square": False,
            },
        ),
        (
            "apply",
            {
                "update_limits": False,
                "update_1d": False,
                "update_canvas": False,
                "update_chi_square": False,
            },
        ),
        ("redraw", {"force": True}),
        ("flush", None),
    ]
    assert exception_trace_calls == [
        {
            "event": "first_visible_simulation_settle_failed",
            "exc_type": "RuntimeError",
            "error": "settle-failed",
            "attempt": 1,
            "retry_allowed": True,
        },
        {
            "event": "first_visible_simulation_settle_failed",
            "exc_type": "RuntimeError",
            "error": "settle-failed",
            "attempt": 2,
            "retry_allowed": False,
        },
    ]
    assert runtime_session.simulation_runtime_state.first_visible_simulation_settle_token is None
    assert (
        runtime_session.simulation_runtime_state.first_visible_simulation_settled_signature is None
    )


def test_apply_primary_figure_display_from_cached_results_preserves_hidden_analysis_payloads_during_detector_redraw(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _StopAtDetectorRedrawBoundary(RuntimeError):
        pass

    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=False,
    )
    detector_redraw_states: list[dict[str, np.ndarray | str]] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(visible=False, current_background_display=None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            background_min_var=_RuntimeVar(0.0),
            background_max_var=_RuntimeVar(1.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 2, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            set_aspect=lambda *_args, **_kwargs: None,
            set_xlabel=lambda *_args, **_kwargs: None,
            set_ylabel=lambda *_args, **_kwargs: None,
            set_title=lambda *_args, **_kwargs: None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "image_display",
        SimpleNamespace(set_visible=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        SimpleNamespace(
            set_visible=lambda *_args, **_kwargs: None,
            set_clim=lambda *_args, **_kwargs: None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "global_image_buffer",
        np.zeros((2, 2), dtype=np.float64),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_canvas_interactions,
        "restore_axis_view",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_roi_preview_display_sources",
        lambda **kwargs: (kwargs["simulation_image"], kwargs["background_image"]),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "set_main_figure_axes_axis_visibility",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "apply_main_figure_axes_chrome",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    def _stop_at_detector_redraw(**kwargs) -> None:
        if kwargs.get("view_mode") == "detector":
            detector_redraw_states.append(
                {
                    "view_mode": str(kwargs["view_mode"]),
                    "caked_image": np.asarray(
                        runtime_session.simulation_runtime_state.last_caked_image_unscaled,
                        dtype=np.float64,
                    ).copy(),
                    "q_space_image": np.asarray(
                        runtime_session.simulation_runtime_state.last_q_space_image_unscaled,
                        dtype=np.float64,
                    ).copy(),
                }
            )
            raise _StopAtDetectorRedrawBoundary()

    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        _stop_at_detector_redraw,
        raising=False,
    )

    with pytest.raises(_StopAtDetectorRedrawBoundary):
        runtime_session._apply_primary_figure_display_from_cached_results(
            "detector",
            ((0.0, 2.0), (2.0, 0.0)),
        )

    assert len(detector_redraw_states) == 1
    assert detector_redraw_states[0]["view_mode"] == "detector"
    np.testing.assert_array_equal(
        detector_redraw_states[0]["caked_image"],
        fixture["caked_image"],
    )
    np.testing.assert_array_equal(
        detector_redraw_states[0]["q_space_image"],
        fixture["q_space_image"],
    )


def test_apply_primary_detector_display_refreshes_buffer_before_projection_cache(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _RasterArtist:
        def __init__(self) -> None:
            self.data = None
            self.extent = None

        def set_data(self, data) -> None:
            self.data = np.asarray(data, dtype=float).copy()

        def set_extent(self, extent) -> None:
            self.extent = tuple(float(value) for value in extent)

        def set_visible(self, *_args, **_kwargs) -> None:
            return None

    image_artist = _RasterArtist()
    background_artist = _RasterArtist()
    source_image = np.arange(1.0, 17.0, dtype=np.float64).reshape(4, 4)
    stale_detector_buffer = np.zeros((4, 4), dtype=np.float64)
    runtime_session.gui_display_projection._PROJECTION_CACHE.clear()

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            unscaled_image=source_image,
            last_unscaled_image_signature=("startup-default",),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(visible=False, current_background_display=None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            background_min_var=_RuntimeVar(0.0),
            background_max_var=_RuntimeVar(1.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "global_image_buffer",
        stale_detector_buffer,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_display", image_artist, raising=False)
    monkeypatch.setattr(runtime_session, "background_display", background_artist, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            set_aspect=lambda *_args, **_kwargs: None,
            set_xlabel=lambda *_args, **_kwargs: None,
            set_ylabel=lambda *_args, **_kwargs: None,
            set_title=lambda *_args, **_kwargs: None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_legacy_main_matplotlib_interaction_active",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_main_display_raster_size_limit",
        lambda: 2,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: 1.0,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_roi_preview_display_sources",
        lambda **kwargs: (kwargs["simulation_image"], kwargs["background_image"]),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_canvas_interactions,
        "restore_axis_view",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "set_main_figure_axes_axis_visibility",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "apply_main_figure_axes_chrome",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    def _sync_detector_projection(**kwargs) -> None:
        assert kwargs.get("view_mode") == "detector"
        runtime_session._store_primary_raster_geometry(
            image_artist,
            origin="upper",
            extent=(0.0, 4.0, 4.0, 0.0),
        )
        runtime_session._apply_projected_primary_raster_to_artist(image_artist)

    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        _sync_detector_projection,
        raising=False,
    )

    runtime_session._apply_primary_figure_display_from_cached_results(
        "detector",
        ((0.0, 4.0), (4.0, 0.0)),
    )

    np.testing.assert_array_equal(runtime_session.global_image_buffer, source_image)
    assert image_artist.data is not None
    assert float(np.max(image_artist.data)) == 16.0


@pytest.mark.parametrize(
    ("view_mode", "payload_signature"),
    [
        ("caked", ("analysis", 3)),
        ("q_space", ("q-space", 7)),
    ],
)
def test_apply_primary_figure_display_from_cached_results_scales_cached_background_payload(
    monkeypatch,
    view_mode: str,
    payload_signature: tuple[str, int],
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    stored_sources: list[tuple[str, np.ndarray, object | None]] = []
    preview_calls: list[dict[str, np.ndarray | None]] = []
    scale = 2.5
    q_space_image = np.full((2, 2), 3.0, dtype=np.float64)
    q_space_background = np.full((2, 2), 4.0, dtype=np.float64)
    caked_image = np.full((2, 2), 2.0, dtype=np.float64)
    caked_background = np.full((2, 2), 5.0, dtype=np.float64)
    image_artist = SimpleNamespace()
    background_artist = SimpleNamespace(
        set_visible=lambda *_args, **_kwargs: None,
        set_clim=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_q_space_image_unscaled=q_space_image,
            last_q_space_background_image_unscaled=q_space_background,
            last_q_space_extent=[-1.0, 1.0, -2.0, 2.0],
            last_q_space_payload_signature=(
                payload_signature if view_mode == "q_space" else ("q-space", 99)
            ),
            last_caked_image_unscaled=caked_image,
            last_caked_background_image_unscaled=caked_background,
            last_caked_extent=[0.0, 90.0, -180.0, 180.0],
            last_analysis_cache_sig=(
                payload_signature if view_mode == "caked" else ("analysis", 99)
            ),
            caked_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(visible=True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(simulation_limits_user_override=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            simulation_min_var=_RuntimeVar(0.0),
            simulation_max_var=_RuntimeVar(20.0),
            background_min_var=_RuntimeVar(0.0),
            background_max_var=_RuntimeVar(20.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "vmin_caked_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "vmax_caked_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "image_display", image_artist, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        background_artist,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            set_aspect=lambda *_args, **_kwargs: None,
            set_xlabel=lambda *_args, **_kwargs: None,
            set_ylabel=lambda *_args, **_kwargs: None,
            set_title=lambda *_args, **_kwargs: None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: view_mode,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: scale,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_auto_caked_limits",
        lambda image: (float(np.min(image)), float(np.max(image))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_simulation_sliders_from_image",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_intensity_display_range",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda artist, source, **kwargs: stored_sources.append(
            (
                "primary" if artist is image_artist else "background",
                np.asarray(source, dtype=float).copy(),
                kwargs.get("source_signature"),
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_canvas_interactions,
        "restore_axis_view",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "set_main_figure_axes_axis_visibility",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "apply_main_figure_axes_chrome",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    def _record_preview_sources(**kwargs):
        preview_calls.append(
            {
                "simulation_image": np.asarray(kwargs["simulation_image"], dtype=float).copy(),
                "background_image": (
                    None
                    if kwargs["background_image"] is None
                    else np.asarray(kwargs["background_image"], dtype=float).copy()
                ),
            }
        )
        return (kwargs["simulation_image"], kwargs["background_image"])

    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_roi_preview_display_sources",
        _record_preview_sources,
        raising=False,
    )

    target_mode = runtime_session._apply_primary_figure_display_from_cached_results(
        "detector",
        ((0.0, 2.0), (2.0, 0.0)),
    )

    expected_primary = caked_image * scale if view_mode == "caked" else q_space_image * scale
    expected_background = (
        caked_background * scale if view_mode == "caked" else q_space_background * scale
    )

    assert target_mode == view_mode
    assert len(stored_sources) == 2
    assert stored_sources[0][0] == "primary"
    np.testing.assert_array_equal(stored_sources[0][1], expected_primary)
    assert stored_sources[0][2] == (view_mode, "primary", payload_signature, scale)
    assert stored_sources[1][0] == "background"
    np.testing.assert_array_equal(stored_sources[1][1], expected_background)
    assert stored_sources[1][2] == (view_mode, "background", payload_signature, scale)
    if view_mode == "caked":
        assert len(preview_calls) == 1
        np.testing.assert_array_equal(
            preview_calls[0]["simulation_image"],
            expected_primary,
        )
        np.testing.assert_array_equal(
            preview_calls[0]["background_image"],
            expected_background,
        )
    else:
        assert preview_calls == []


@pytest.mark.parametrize("stale_target", ["caked", "q_space"])
def test_invalidate_hidden_analysis_space_payloads_if_stale_clears_only_stale_hidden_payload(
    monkeypatch,
    stale_target: str,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    current_analysis_cache_sig = (("sim-cache", 1), ("bg-cache", 1))
    q_space_payload_geometry_sig = ("geom", 1)
    matching_q_space_signature = (current_analysis_cache_sig, q_space_payload_geometry_sig)
    caked_image = np.ones((2, 2), dtype=np.float64)
    q_space_image = np.full((2, 2), 3.0, dtype=np.float64)

    if stale_target == "caked":
        last_analysis_cache_sig = (("stale-sim", 0), ("stale-bg", 0))
        last_q_space_payload_signature = matching_q_space_signature
    else:
        last_analysis_cache_sig = current_analysis_cache_sig
        last_q_space_payload_signature = (current_analysis_cache_sig, ("old-geom", 0))

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_analysis_cache_sig=last_analysis_cache_sig,
            last_q_space_payload_signature=last_q_space_payload_signature,
            last_caked_image_unscaled=caked_image.copy(),
            last_caked_extent=[0.0, 1.0, -1.0, 1.0],
            last_caked_background_image_unscaled=np.full((2, 2), 2.0, dtype=np.float64),
            last_caked_radial_values=np.array([1.0, 2.0], dtype=np.float64),
            last_caked_azimuth_values=np.array([-1.0, 1.0], dtype=np.float64),
            last_caked_intersection_cache=("cache",),
            last_caked_intersection_cache_transform_bundle="bundle",
            last_q_space_image_unscaled=q_space_image.copy(),
            last_q_space_qr_values=np.array([0.1, 0.2], dtype=np.float64),
            last_q_space_qz_values=np.array([0.3, 0.4], dtype=np.float64),
            last_q_space_extent=[-1.0, 1.0, -2.0, 2.0],
            last_q_space_background_image_unscaled=np.full((2, 2), 4.0, dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_live_caked_transform_bundle",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    runtime_session._invalidate_hidden_analysis_space_payloads_if_stale(
        caked_analysis_requested=False,
        q_space_requested=False,
        current_analysis_cache_sig=current_analysis_cache_sig,
        q_space_payload_geometry_sig=q_space_payload_geometry_sig,
    )

    if stale_target == "caked":
        assert runtime_session.simulation_runtime_state.last_caked_image_unscaled is None
        assert runtime_session.simulation_runtime_state.last_caked_extent is None
        assert runtime_session.simulation_runtime_state.last_caked_background_image_unscaled is None
        assert runtime_session.simulation_runtime_state.last_caked_radial_values is None
        assert runtime_session.simulation_runtime_state.last_caked_azimuth_values is None
        assert runtime_session.simulation_runtime_state.last_caked_intersection_cache is None
        assert (
            runtime_session.simulation_runtime_state.last_caked_intersection_cache_transform_bundle
            is None
        )
        np.testing.assert_array_equal(
            runtime_session.simulation_runtime_state.last_q_space_image_unscaled,
            q_space_image,
        )
        assert runtime_session.simulation_runtime_state.last_q_space_payload_signature == (
            current_analysis_cache_sig,
            q_space_payload_geometry_sig,
        )
    else:
        np.testing.assert_array_equal(
            runtime_session.simulation_runtime_state.last_caked_image_unscaled,
            caked_image,
        )
        assert runtime_session.simulation_runtime_state.last_q_space_image_unscaled is None
        assert runtime_session.simulation_runtime_state.last_q_space_qr_values is None
        assert runtime_session.simulation_runtime_state.last_q_space_qz_values is None
        assert runtime_session.simulation_runtime_state.last_q_space_extent is None
        assert (
            runtime_session.simulation_runtime_state.last_q_space_background_image_unscaled is None
        )
        assert runtime_session.simulation_runtime_state.last_q_space_payload_signature is None


@pytest.mark.parametrize(
    ("stale_target", "expected_calls"),
    [
        ("caked", [(True, False)]),
        ("q_space", [(False, True)]),
    ],
)
def test_invalidate_hidden_analysis_space_payloads_if_stale_clears_each_payload_once(
    monkeypatch,
    stale_target: str,
    expected_calls: list[tuple[bool, bool]],
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    current_analysis_cache_sig = (("sim-cache", 1), ("bg-cache", 1))
    q_space_payload_geometry_sig = ("geom", 1)
    matching_q_space_signature = (current_analysis_cache_sig, q_space_payload_geometry_sig)
    invalidation_calls: list[tuple[bool, bool]] = []

    if stale_target == "caked":
        last_analysis_cache_sig = (("stale-sim", 0), ("stale-bg", 0))
        last_q_space_payload_signature = matching_q_space_signature
    else:
        last_analysis_cache_sig = current_analysis_cache_sig
        last_q_space_payload_signature = (current_analysis_cache_sig, ("old-geom", 0))

    state = SimpleNamespace(
        last_analysis_cache_sig=last_analysis_cache_sig,
        last_q_space_payload_signature=last_q_space_payload_signature,
        last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
        last_caked_extent=[0.0, 1.0, -1.0, 1.0],
        last_caked_background_image_unscaled=np.full((2, 2), 2.0, dtype=np.float64),
        last_caked_radial_values=np.array([1.0, 2.0], dtype=np.float64),
        last_caked_azimuth_values=np.array([-1.0, 1.0], dtype=np.float64),
        last_caked_intersection_cache=("cache",),
        last_caked_intersection_cache_transform_bundle="bundle",
        last_q_space_image_unscaled=np.full((2, 2), 3.0, dtype=np.float64),
        last_q_space_qr_values=np.array([0.1, 0.2], dtype=np.float64),
        last_q_space_qz_values=np.array([0.3, 0.4], dtype=np.float64),
        last_q_space_extent=[-1.0, 1.0, -2.0, 2.0],
        last_q_space_background_image_unscaled=np.full((2, 2), 4.0, dtype=np.float64),
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        state,
        raising=False,
    )

    def _record_invalidation(*, clear_caked=False, clear_q_space=False) -> None:
        invalidation_calls.append((bool(clear_caked), bool(clear_q_space)))
        if clear_caked:
            state.last_caked_image_unscaled = None
            state.last_caked_extent = None
            state.last_caked_background_image_unscaled = None
            state.last_caked_radial_values = None
            state.last_caked_azimuth_values = None
            state.last_caked_intersection_cache = None
            state.last_caked_intersection_cache_transform_bundle = None
        if clear_q_space:
            state.last_q_space_image_unscaled = None
            state.last_q_space_qr_values = None
            state.last_q_space_qz_values = None
            state.last_q_space_extent = None
            state.last_q_space_background_image_unscaled = None
            state.last_q_space_payload_signature = None

    monkeypatch.setattr(
        runtime_session,
        "_invalidate_cached_analysis_space_payloads",
        _record_invalidation,
        raising=False,
    )

    runtime_session._invalidate_hidden_analysis_space_payloads_if_stale(
        caked_analysis_requested=False,
        q_space_requested=False,
        current_analysis_cache_sig=current_analysis_cache_sig,
        q_space_payload_geometry_sig=q_space_payload_geometry_sig,
    )
    runtime_session._invalidate_hidden_analysis_space_payloads_if_stale(
        caked_analysis_requested=False,
        q_space_requested=False,
        current_analysis_cache_sig=current_analysis_cache_sig,
        q_space_payload_geometry_sig=q_space_payload_geometry_sig,
    )

    assert invalidation_calls == expected_calls


def test_runtime_impl_forces_canvas_redraw_when_primary_view_mode_changes() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "force_canvas_redraw=False," in source
    assert "_request_main_canvas_redraw(force_matplotlib=bool(force_canvas_redraw))" in source
    assert "target_primary_view_mode = (" in source
    assert "force_canvas_redraw=(previous_primary_view_mode != target_primary_view_mode)," in source


def test_runtime_impl_prepare_q_space_payload_uses_direct_detector_remap() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _prepare_q_space_display_payload(")
    helper_end = source.index("def _store_q_space_display_payload(", helper_start)
    helper_source = source[helper_start:helper_end]

    assert "detector_image: np.ndarray | None" in helper_source
    assert "convert_image_to_q_space(" in helper_source
    assert "caked_image_to_q_space_payload" not in helper_source
    assert "center_row_px=float(center_array[0])" in helper_source
    assert "center_col_px=float(center_array[1])" in helper_source


def test_runtime_impl_run_analysis_job_builds_q_space_from_detector_images() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _run_analysis_job(")
    helper_end = source.index("def _submit_async_analysis_job(", helper_start)
    helper_source = source[helper_start:helper_end]

    assert 'q_space_requested = bool(job.get("q_space_requested", False))' in helper_source
    assert "if q_space_requested:" in helper_source
    assert "sim_q_space = _prepare_q_space_display_payload_with_geometry(" in helper_source
    assert "bg_q_space = _prepare_q_space_display_payload_with_geometry(" in helper_source
    assert (
        "sim_q_space = _prepare_q_space_display_payload(\n        sim_caked," not in helper_source
    )
    assert "bg_q_space = _prepare_q_space_display_payload(\n        bg_caked," not in helper_source


def test_runtime_impl_analysis_job_payload_tracks_q_space_intent() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '"q_space_requested": q_space_requested,' in source


def test_runtime_impl_resolves_live_view_mode_before_q_space_restore() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "requested_view_mode = _current_app_shell_view_mode()" in source
    assert 'and requested_view_mode == "q_space"' in source
    assert "q_space_requested=q_space_requested," in source
    assert 'live_q_space_requested = bool(_current_app_shell_view_mode() == "q_space")' in source


def test_runtime_impl_q_space_view_mode_is_not_gated_on_caked_toggle() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _current_app_shell_view_mode() -> str:")
    helper_end = source.index("def _active_caked_primary_view()", helper_start)
    helper_source = source[helper_start:helper_end]

    assert 'if selected_mode == "q_space":\n        return "q_space"' in helper_source
    assert helper_source.index('if selected_mode == "q_space":') < helper_source.index(
        "if show_caked:"
    )


def test_runtime_impl_single_viewport_startup_uses_embedded_matplotlib_tk() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    assert "figure_canvas_cls = _get_tk_figure_canvas_cls()" in source
    assert "matplotlib_canvas = figure_canvas_cls(" in source
    assert "master=app_shell_view_state.canvas_frame" in source
    assert "_set_runtime_canvas(matplotlib_canvas)" in source
    assert "_initialize_runtime_plot_block_03()" not in source


def test_initialize_runtime_plot_block_02_projects_startup_rasters_without_viewport_workflow(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _FakeArtist:
        def __init__(self, array: np.ndarray, extent: tuple[float, float, float, float]) -> None:
            self._array = np.asarray(array, dtype=float)
            self._extent = tuple(float(value) for value in extent)
            self.origin = "upper"
            self.data = None

        def get_array(self):
            return self._array

        def get_extent(self):
            return self._extent

        def set_extent(self, extent):
            self._extent = tuple(float(value) for value in extent)

        def set_data(self, data):
            self.data = np.asarray(data, dtype=float)

    projection_calls: list[dict[str, object]] = []

    def _project_raster_to_view(source, **kwargs):
        projection_calls.append(dict(kwargs))
        return SimpleNamespace(
            extent=tuple(float(value) for value in kwargs["extent"]),
            image=np.asarray(source, dtype=float),
        )

    image_display = _FakeArtist(np.arange(4, dtype=float).reshape(2, 2), (0.0, 1.0, 1.0, 0.0))
    background_display = _FakeArtist(np.ones((2, 2), dtype=float), (0.0, 1.0, 1.0, 0.0))
    overlay_display = _FakeArtist(np.zeros((2, 2), dtype=float), (0.0, 1.0, 1.0, 0.0))

    monkeypatch.setattr(runtime_session, "_RUNTIME_CONTROLS_INITIALIZED", True, raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 2, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(last_caked_extent=None, last_q_space_extent=None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            get_xlim=lambda: (0.0, 2.0),
            get_ylim=lambda: (2.0, 0.0),
            bbox=SimpleNamespace(width=200.0, height=200.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_display_projection",
        SimpleNamespace(project_raster_to_view=_project_raster_to_view),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_main_display_raster_size_limit",
        lambda: 64,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_display", image_display, raising=False)
    monkeypatch.setattr(runtime_session, "background_display", background_display, raising=False)
    monkeypatch.setattr(
        runtime_session, "integration_region_overlay", overlay_display, raising=False
    )

    runtime_session._initialize_runtime_plot_block_02()

    assert len(projection_calls) == 3
    assert all(call["axis_xlim"] is None for call in projection_calls)
    assert all(call["axis_ylim"] is None for call in projection_calls)
    assert all(call["bbox_width_px"] is None for call in projection_calls)
    assert all(call["bbox_height_px"] is None for call in projection_calls)
    assert image_display.get_extent() == (0.0, 2.0, 2.0, 0.0)
    assert background_display.get_extent() == (0.0, 2.0, 2.0, 0.0)
    assert overlay_display.get_extent() == (0.0, 2.0, 2.0, 0.0)
    assert image_display.data is not None
    assert background_display.data is not None
    assert overlay_display.data is not None


def test_run_analysis_job_can_emit_sim_q_space_without_caked_payload(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(runtime_session, "_build_analysis_integrator", lambda _job: object())
    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda image, _ai, **_kwargs: {
            "shape": tuple(np.asarray(image).shape),
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runtime_session, "default_reserved_cpu_worker_count", lambda: 1)

    result = runtime_session._run_analysis_job(
        {
            "job_id": 7,
            "signature": ("analysis", 1),
            "epoch": 3,
            "image": np.eye(8, dtype=np.float64),
            "background_image": None,
            "npt_rad": 48,
            "npt_azim": 40,
            "is_preview": False,
            "cached_bg_res2": None,
            "cached_bg_caked": None,
            "intersection_cache": None,
            "sim_cache_sig": ("sim", 1),
            "bg_cache_sig": None,
            "sim_caking_sig": ("cake", 1),
            "bg_caking_sig": None,
            "distance_m": 0.5,
            "center": np.array([4.0, 4.0], dtype=np.float64),
            "pixel_size_m": 1.0e-4,
            "wavelength_m": 1.24e-10,
            "gamma_deg": 0.0,
            "Gamma_deg": 0.0,
            "chi_deg": 0.0,
            "psi_deg": 0.0,
            "psi_z_deg": 0.0,
            "theta_initial_deg": 0.0,
            "cor_angle_deg": 0.0,
            "zs": 0.0,
            "zb": 0.0,
            "q_space_requested": True,
        }
    )

    assert result["sim_caked"] is None
    assert result["bg_caked"] is None
    assert isinstance(result["sim_q_space"], dict)
    assert result["bg_q_space"] is None
    assert np.asarray(result["sim_q_space"]["image"], dtype=float).size > 0
    assert np.asarray(result["sim_q_space"]["qr"], dtype=float).size == 48
    assert np.asarray(result["sim_q_space"]["qz"], dtype=float).size == 40


def test_toggle_caked_2d_legacy_path_overrides_stale_q_space_shell_mode(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

        def set(self, value: object) -> None:
            self._value = value

    shell_mode_var = _Var("q_space")
    show_caked_var = _Var(True)
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_caked_2d_var=show_caked_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "app_shell_view_state",
        SimpleNamespace(view_mode_var=shell_mode_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(pick_armed=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(
            set_app_shell_view_mode=lambda view_state, mode: view_state.view_mode_var.set(mode)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_qr_cylinder_overlay_view_state",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_apply_main_caked_view_toggle", lambda: None)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)

    runtime_session.toggle_caked_2d()

    assert shell_mode_var.get() == "caked"


def test_toggle_caked_2d_skips_qr_overlay_invalidation_when_overlays_are_off(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

        def set(self, value: object) -> None:
            self._value = value

    shell_mode_var = _Var("detector")
    show_caked_var = _Var(True)
    invalidations: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(
            show_caked_2d_var=show_caked_var,
            show_qz_rods_var=_Var(False),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_overlay_actions_view_state",
        SimpleNamespace(show_geometry_overlays_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "app_shell_view_state",
        SimpleNamespace(view_mode_var=shell_mode_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(pick_armed=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(
            set_app_shell_view_mode=lambda view_state, mode: view_state.view_mode_var.set(mode)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_caked_roi_preview_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"integrate_qz_rods": False, "qr_half_width": 0.0},
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_qr_cylinder_overlay_view_state",
        lambda **kwargs: invalidations.append(dict(kwargs)),
    )
    monkeypatch.setattr(runtime_session, "_apply_main_caked_view_toggle", lambda: None)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)

    runtime_session.toggle_caked_2d()

    assert shell_mode_var.get() == "caked"
    assert invalidations == []


def test_toggle_caked_2d_reselects_current_hkl_peak_for_new_view(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

        def set(self, value: object) -> None:
            self._value = value

    shell_mode_var = _Var("detector")
    show_caked_var = _Var(True)
    reselect_calls: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(
            show_caked_2d_var=show_caked_var,
            show_qz_rods_var=_Var(False),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_overlay_actions_view_state",
        SimpleNamespace(show_geometry_overlays_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "app_shell_view_state",
        SimpleNamespace(view_mode_var=shell_mode_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(pick_armed=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "peak_selection_runtime_callbacks",
        SimpleNamespace(reselect_current_peak=lambda: reselect_calls.append("reselect") or True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(
            set_app_shell_view_mode=lambda view_state, mode: view_state.view_mode_var.set(mode)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_caked_roi_preview_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"integrate_qz_rods": False, "qr_half_width": 0.0},
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_qr_cylinder_overlay_view_state",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_apply_main_caked_view_toggle", lambda: None)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)

    runtime_session.toggle_caked_2d()

    assert shell_mode_var.get() == "caked"
    assert reselect_calls == ["reselect"]


def test_toggle_caked_2d_hides_selected_peak_when_reselect_fails(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

        def set(self, value: object) -> None:
            self._value = value

    class _Marker:
        def __init__(self) -> None:
            self.visible = True

        def set_visible(self, value: object) -> None:
            self.visible = bool(value)

    shell_mode_var = _Var("detector")
    show_caked_var = _Var(True)
    marker = _Marker()
    redraw_calls: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(
            show_caked_2d_var=show_caked_var,
            show_qz_rods_var=_Var(False),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_overlay_actions_view_state",
        SimpleNamespace(show_geometry_overlays_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "app_shell_view_state",
        SimpleNamespace(view_mode_var=shell_mode_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(pick_armed=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "peak_selection_runtime_callbacks",
        SimpleNamespace(reselect_current_peak=lambda: False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "selected_peak_marker",
        marker,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(
            set_app_shell_view_mode=lambda view_state, mode: view_state.view_mode_var.set(mode)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda *, force_matplotlib=False: redraw_calls.append(bool(force_matplotlib)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_caked_roi_preview_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"integrate_qz_rods": False, "qr_half_width": 0.0},
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_qr_cylinder_overlay_view_state",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_apply_main_caked_view_toggle", lambda: None)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)

    runtime_session.toggle_caked_2d()

    assert shell_mode_var.get() == "caked"
    assert marker.visible is False
    assert redraw_calls == [False]


def test_set_persistent_view_mode_keeps_q_space_when_enabling_caked_data(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

        def set(self, value: object) -> None:
            self._value = value

    shell_mode_var = _Var("detector")
    show_caked_var = _Var(False)
    schedule_calls: list[str] = []
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_caked_2d_var=show_caked_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "app_shell_view_state",
        SimpleNamespace(view_mode_var=shell_mode_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(pick_armed=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(unscaled_image=np.ones((1, 1), dtype=np.float64)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(
            set_app_shell_view_mode=lambda view_state, mode: view_state.view_mode_var.set(mode)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_qr_cylinder_overlay_view_state",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_refresh_run_status_bar", lambda: None)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session, "schedule_update", lambda: schedule_calls.append("schedule")
    )

    runtime_session._set_persistent_view_mode("q_space")

    assert show_caked_var.get() is True
    assert shell_mode_var.get() == "q_space"
    assert schedule_calls == ["schedule"]


def test_apply_scale_factor_to_existing_results_can_force_canvas_redraw(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_idle_main_figure_preview_state(monkeypatch, runtime_session)

    class _Var:
        def __init__(self, value: float | bool) -> None:
            self._value = value

        def get(self):
            return self._value

        def set(self, value) -> None:
            self._value = value

    redraw_calls: list[bool] = []
    synced_modes: list[str | None] = []
    stored_sources: list[np.ndarray] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            unscaled_image=np.ones((2, 2), dtype=np.float64),
            last_unscaled_image_signature=("sim", 1),
            last_q_space_image_unscaled=None,
            last_q_space_background_image_unscaled=None,
            last_caked_image_unscaled=np.full((2, 2), 2.0, dtype=np.float64),
            last_caked_background_image_unscaled=None,
            last_1d_integration_data={
                "radials_sim": None,
                "intensities_2theta_sim": None,
                "azimuths_sim": None,
                "intensities_azimuth_sim": None,
                "radials_bg": None,
                "intensities_2theta_bg": None,
                "azimuths_bg": None,
                "intensities_azimuth_bg": None,
            },
            caked_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(
            visible=False,
            current_background_index=0,
            backend_rotation_k=0,
            backend_flip_x=False,
            backend_flip_y=False,
            current_background_display=None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            simulation_min_var=_Var(0.0),
            simulation_max_var=_Var(10.0),
            background_min_var=_Var(0.0),
            background_max_var=_Var(1.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(
            simulation_limits_user_override=False,
            background_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_1d_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "vmin_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "vmax_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "colorbar_main",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "caked_colorbar",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    fake_image_artist = SimpleNamespace(set_visible=lambda *_args, **_kwargs: None)
    fake_background_artist = SimpleNamespace(set_visible=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime_session, "image_display", fake_image_artist, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        fake_background_artist,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "caked",
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: float(default),
    )
    monkeypatch.setattr(
        runtime_session,
        "_ensure_global_image_buffer_shape",
        lambda _image: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "global_image_buffer",
        np.zeros((2, 2), dtype=np.float64),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_roi_preview_display_sources",
        lambda **kwargs: (kwargs["simulation_image"], kwargs["background_image"]),
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_simulation_sliders_from_image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda _artist, source, **_kwargs: stored_sources.append(
            np.asarray(source, dtype=float).copy()
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        lambda **kwargs: synced_modes.append(kwargs.get("view_mode")),
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_intensity_display_range",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda *, force_matplotlib=False: redraw_calls.append(bool(force_matplotlib)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_update_chi_square_display", lambda: None)

    runtime_session.apply_scale_factor_to_existing_results(
        update_1d=False,
        force_canvas_redraw=True,
        update_chi_square=False,
    )

    assert synced_modes == ["caked"]
    assert redraw_calls == [True]
    assert len(stored_sources) >= 1


@pytest.mark.parametrize(
    ("view_mode", "payload_signature"),
    [
        ("caked", ("analysis", 3)),
        ("q_space", ("q-space", 7)),
    ],
)
def test_apply_scale_factor_to_existing_results_uses_semantic_analysis_source_signatures(
    monkeypatch,
    view_mode: str,
    payload_signature: tuple[str, int],
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_idle_main_figure_preview_state(monkeypatch, runtime_session)

    class _Var:
        def __init__(self, value: float | bool) -> None:
            self._value = value

        def get(self):
            return self._value

        def set(self, value) -> None:
            self._value = value

    stored_signatures: list[tuple[str, object | None]] = []
    fake_image_artist = SimpleNamespace(set_visible=lambda *_args, **_kwargs: None)
    fake_background_artist = SimpleNamespace(set_visible=lambda *_args, **_kwargs: None)
    scale = 2.0

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            unscaled_image=np.ones((2, 2), dtype=np.float64),
            last_unscaled_image_signature=("sim", 1),
            last_q_space_image_unscaled=np.full((2, 2), 3.0, dtype=np.float64),
            last_q_space_background_image_unscaled=np.full((2, 2), 4.0, dtype=np.float64),
            last_q_space_payload_signature=(
                payload_signature if view_mode == "q_space" else ("q-space", 99)
            ),
            last_caked_image_unscaled=np.full((2, 2), 2.0, dtype=np.float64),
            last_caked_background_image_unscaled=np.full((2, 2), 5.0, dtype=np.float64),
            last_analysis_cache_sig=(
                payload_signature if view_mode == "caked" else ("analysis", 99)
            ),
            last_1d_integration_data={
                "radials_sim": None,
                "intensities_2theta_sim": None,
                "azimuths_sim": None,
                "intensities_azimuth_sim": None,
                "radials_bg": None,
                "intensities_2theta_bg": None,
                "azimuths_bg": None,
                "intensities_azimuth_bg": None,
            },
            caked_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(
            visible=True,
            current_background_index=0,
            backend_rotation_k=0,
            backend_flip_x=False,
            backend_flip_y=False,
            current_background_display=None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            simulation_min_var=_Var(0.0),
            simulation_max_var=_Var(10.0),
            background_min_var=_Var(0.0),
            background_max_var=_Var(10.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(
            simulation_limits_user_override=False,
            background_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_1d_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "vmin_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "vmax_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "colorbar_main",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "caked_colorbar",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_display", fake_image_artist, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        fake_background_artist,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: view_mode,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: scale,
    )
    monkeypatch.setattr(
        runtime_session,
        "_ensure_global_image_buffer_shape",
        lambda _image: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "global_image_buffer",
        np.zeros((2, 2), dtype=np.float64),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_roi_preview_display_sources",
        lambda **kwargs: (kwargs["simulation_image"], kwargs["background_image"]),
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_simulation_sliders_from_image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda artist, _source, **kwargs: stored_signatures.append(
            (
                "primary" if artist is fake_image_artist else "background",
                kwargs.get("source_signature"),
            )
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_intensity_display_range",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_update_chi_square_display", lambda: None)

    runtime_session.apply_scale_factor_to_existing_results(
        update_1d=False,
        force_canvas_redraw=False,
        update_chi_square=False,
    )

    assert stored_signatures == [
        ("primary", (view_mode, "primary", payload_signature, scale)),
        ("background", (view_mode, "background", payload_signature, scale)),
    ]


def test_apply_scale_factor_to_existing_results_uses_runtime_image_signature_for_chi_square(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_idle_main_figure_preview_state(monkeypatch, runtime_session)

    class _Var:
        def __init__(self, value: float | bool) -> None:
            self._value = value

        def get(self):
            return self._value

        def set(self, value) -> None:
            self._value = value

    chi_state = {"buffer_sig": ("stale",)}
    dirty_calls: list[str] = []
    runtime_signature = ("runtime-sig", 42)

    monkeypatch.setattr(
        runtime_session, "last_unscaled_image_signature", ("wrong", 999), raising=False
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            unscaled_image=np.ones((2, 2), dtype=np.float64),
            last_unscaled_image_signature=runtime_signature,
            last_q_space_image_unscaled=None,
            last_q_space_background_image_unscaled=None,
            last_caked_image_unscaled=None,
            last_caked_background_image_unscaled=None,
            last_1d_integration_data={
                "radials_sim": None,
                "intensities_2theta_sim": None,
                "azimuths_sim": None,
                "intensities_azimuth_sim": None,
                "radials_bg": None,
                "intensities_2theta_bg": None,
                "azimuths_bg": None,
                "intensities_azimuth_bg": None,
            },
            caked_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(
            visible=False,
            current_background_index=0,
            backend_rotation_k=0,
            backend_flip_x=False,
            backend_flip_y=False,
            current_background_display=None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            simulation_min_var=_Var(0.0),
            simulation_max_var=_Var(10.0),
            background_min_var=_Var(0.0),
            background_max_var=_Var(1.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(
            simulation_limits_user_override=False,
            background_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_1d_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "vmin_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "vmax_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "colorbar_main",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "caked_colorbar",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "image_display",
        SimpleNamespace(set_visible=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        SimpleNamespace(set_visible=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: 1.5,
    )
    monkeypatch.setattr(
        runtime_session,
        "_ensure_global_image_buffer_shape",
        lambda _image: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "global_image_buffer",
        np.zeros((2, 2), dtype=np.float64),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_simulation_sliders_from_image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_intensity_display_range",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_update_chi_square_display", lambda: None)
    monkeypatch.setattr(runtime_session, "chi_square_state", chi_state, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_mark_chi_square_dirty",
        lambda: dirty_calls.append("dirty"),
        raising=False,
    )

    runtime_session.apply_scale_factor_to_existing_results(
        update_1d=False,
        force_canvas_redraw=False,
        update_chi_square=False,
    )

    assert chi_state["buffer_sig"][0] == runtime_signature
    assert chi_state["buffer_sig"][1] == 1.5
    assert dirty_calls == ["dirty"]


def test_apply_scale_factor_to_existing_results_clears_pending_preview_before_sync_and_redraw(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: float | bool) -> None:
            self._value = value

        def get(self):
            return self._value

        def set(self, value) -> None:
            self._value = value

    geometry_runtime_state = SimpleNamespace(
        _canvas_preview_limits=((4.0, 8.0), (9.0, 2.0)),
        _canvas_pan_session={"drag": True},
    )
    sync_states: list[tuple[object, object, str | None]] = []
    redraw_states: list[tuple[object, object, bool]] = []
    cleared_tokens: list[object] = []

    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        geometry_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_legacy_main_matplotlib_preview_view",
        lambda *, redraw=True: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            unscaled_image=np.ones((2, 2), dtype=np.float64),
            last_unscaled_image_signature=("sim", 1),
            last_q_space_image_unscaled=None,
            last_q_space_background_image_unscaled=None,
            last_caked_image_unscaled=np.full((2, 2), 2.0, dtype=np.float64),
            last_caked_background_image_unscaled=None,
            last_1d_integration_data={
                "radials_sim": None,
                "intensities_2theta_sim": None,
                "azimuths_sim": None,
                "intensities_azimuth_sim": None,
                "radials_bg": None,
                "intensities_2theta_bg": None,
                "azimuths_bg": None,
                "intensities_azimuth_bg": None,
            },
            caked_limits_user_override=False,
            interaction_drag_active=True,
            interaction_drag_requires_settled_update=True,
            interaction_settle_token="pending-preview",
            main_matplotlib_overlays_suspended=True,
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "root", object(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda _root, token: cleared_tokens.append(token),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_run_status_bar",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(
            visible=False,
            current_background_index=0,
            backend_rotation_k=0,
            backend_flip_x=False,
            backend_flip_y=False,
            current_background_display=None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            simulation_min_var=_Var(0.0),
            simulation_max_var=_Var(10.0),
            background_min_var=_Var(0.0),
            background_max_var=_Var(1.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(
            simulation_limits_user_override=False,
            background_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_1d_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "vmin_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "vmax_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "colorbar_main",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "caked_colorbar",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "image_display",
        SimpleNamespace(set_visible=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        SimpleNamespace(set_visible=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "caked",
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: float(default),
    )
    monkeypatch.setattr(
        runtime_session,
        "_ensure_global_image_buffer_shape",
        lambda _image: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "global_image_buffer",
        np.zeros((2, 2), dtype=np.float64),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_roi_preview_display_sources",
        lambda **kwargs: (kwargs["simulation_image"], kwargs["background_image"]),
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_simulation_sliders_from_image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        lambda **kwargs: sync_states.append(
            (
                geometry_runtime_state._canvas_preview_limits,
                geometry_runtime_state._canvas_pan_session,
                kwargs.get("view_mode"),
            )
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_intensity_display_range",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda *, force_matplotlib=False: redraw_states.append(
            (
                geometry_runtime_state._canvas_preview_limits,
                geometry_runtime_state._canvas_pan_session,
                bool(force_matplotlib),
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_update_chi_square_display", lambda: None)

    runtime_session.apply_scale_factor_to_existing_results(
        update_1d=False,
        force_canvas_redraw=True,
        update_chi_square=False,
    )

    assert cleared_tokens == ["pending-preview"]
    assert geometry_runtime_state._canvas_preview_limits is None
    assert geometry_runtime_state._canvas_pan_session is None
    assert sync_states == [(None, None, "caked")]
    assert redraw_states == [(None, None, True)]


def test_run_analysis_job_skips_q_space_for_caked_only_requests(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    q_space_calls: list[np.ndarray | None] = []

    monkeypatch.setattr(runtime_session, "_build_analysis_integrator", lambda _job: object())
    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda image, _ai, **_kwargs: {
            "shape": tuple(np.asarray(image).shape),
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda res2, **_kwargs: (
            None
            if res2 is None
            else {
                "image": np.ones((2, 2), dtype=np.float64),
                "radial": np.array([1.0, 2.0], dtype=np.float64),
                "azimuth": np.array([-1.0, 1.0], dtype=np.float64),
                "transform_bundle": None,
                "detector_shape": None,
                "extent": [0.0, 1.0, -1.0, 1.0],
            }
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_q_space_display_payload",
        lambda detector_image, **_kwargs: q_space_calls.append(
            None if detector_image is None else np.asarray(detector_image, dtype=np.float64)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runtime_session, "default_reserved_cpu_worker_count", lambda: 1)

    result = runtime_session._run_analysis_job(
        {
            "job_id": 8,
            "signature": ("analysis", 2),
            "epoch": 3,
            "image": np.eye(8, dtype=np.float64),
            "background_image": None,
            "npt_rad": 48,
            "npt_azim": 40,
            "is_preview": False,
            "cached_bg_res2": None,
            "cached_bg_caked": None,
            "intersection_cache": None,
            "sim_cache_sig": ("sim", 2),
            "bg_cache_sig": None,
            "sim_caking_sig": ("cake", 2),
            "bg_caking_sig": None,
            "distance_m": 0.5,
            "center": np.array([4.0, 4.0], dtype=np.float64),
            "pixel_size_m": 1.0e-4,
            "wavelength_m": 1.24e-10,
            "gamma_deg": 0.0,
            "Gamma_deg": 0.0,
            "chi_deg": 0.0,
            "psi_deg": 0.0,
            "psi_z_deg": 0.0,
            "theta_initial_deg": 0.0,
            "cor_angle_deg": 0.0,
            "zs": 0.0,
            "zb": 0.0,
            "q_space_requested": False,
        }
    )

    assert isinstance(result["sim_caked"], dict)
    assert result["bg_caked"] is None
    assert result["sim_q_space"] is None
    assert result["bg_q_space"] is None
    assert q_space_calls == []


def test_run_analysis_job_caked_only_request_ignores_q_space_failure(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(runtime_session, "_build_analysis_integrator", lambda _job: object())
    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda image, _ai, **_kwargs: {
            "shape": tuple(np.asarray(image).shape),
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda res2, **_kwargs: (
            None
            if res2 is None
            else {
                "image": np.ones((2, 2), dtype=np.float64),
                "radial": np.array([1.0, 2.0], dtype=np.float64),
                "azimuth": np.array([-1.0, 1.0], dtype=np.float64),
                "transform_bundle": None,
                "detector_shape": None,
                "extent": [0.0, 1.0, -1.0, 1.0],
            }
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_q_space_display_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("q-space failure")),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runtime_session, "default_reserved_cpu_worker_count", lambda: 1)

    result = runtime_session._run_analysis_job(
        {
            "job_id": 9,
            "signature": ("analysis", 3),
            "epoch": 3,
            "image": np.eye(8, dtype=np.float64),
            "background_image": None,
            "npt_rad": 48,
            "npt_azim": 40,
            "is_preview": False,
            "cached_bg_res2": None,
            "cached_bg_caked": None,
            "intersection_cache": None,
            "sim_cache_sig": ("sim", 3),
            "bg_cache_sig": None,
            "sim_caking_sig": ("cake", 3),
            "bg_caking_sig": None,
            "distance_m": 0.5,
            "center": np.array([4.0, 4.0], dtype=np.float64),
            "pixel_size_m": 1.0e-4,
            "wavelength_m": 1.24e-10,
            "gamma_deg": 0.0,
            "Gamma_deg": 0.0,
            "chi_deg": 0.0,
            "psi_deg": 0.0,
            "psi_z_deg": 0.0,
            "theta_initial_deg": 0.0,
            "cor_angle_deg": 0.0,
            "zs": 0.0,
            "zb": 0.0,
            "q_space_requested": False,
        }
    )

    assert isinstance(result["sim_caked"], dict)
    assert result["sim_q_space"] is None
    assert result["bg_q_space"] is None


@pytest.mark.parametrize(
    ("shape", "expected_method"),
    [
        ((128, 128), "lut"),
        ((513, 512), "exact"),
    ],
)
def test_prepare_q_space_display_payload_chooses_safe_conversion_method(
    monkeypatch,
    shape: tuple[int, int],
    expected_method: str,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    methods: list[str] = []

    def _convert_stub(_image, **kwargs):
        methods.append(str(kwargs["method"]))
        return SimpleNamespace(
            qr=np.array([0.0, 1.0], dtype=np.float64),
            qz=np.array([0.0, 1.0], dtype=np.float64),
            intensity=np.ones((2, 2), dtype=np.float64),
            sum_signal=np.ones((2, 2), dtype=np.float64),
            sum_normalization=np.ones((2, 2), dtype=np.float64),
            count=np.ones((2, 2), dtype=np.float64),
        )

    monkeypatch.setattr(runtime_session, "convert_image_to_q_space", _convert_stub)

    payload = runtime_session._prepare_q_space_display_payload(
        np.ones(shape, dtype=np.float64),
        npt_rad=8,
        npt_azim=6,
        distance_m=0.5,
        center=np.array([2.0, 2.0], dtype=np.float64),
        pixel_size_m=1.0e-4,
        wavelength_m=1.24e-10,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        theta_initial_deg=0.0,
        cor_angle_deg=0.0,
        zs=0.0,
        zb=0.0,
    )

    assert isinstance(payload, dict)
    assert methods == [expected_method]


def test_restore_caked_payload_rebuilds_q_space_from_live_q_space_mode(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: float) -> None:
            self._value = value

        def get(self) -> float:
            return float(self._value)

    background_native = np.arange(16, dtype=np.float64).reshape(4, 4)
    q_space_inputs: list[np.ndarray] = []
    stored_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            analysis_preview_bins=(8, 6),
            last_res2_sim=object(),
            last_res2_background=object(),
            ai_cache={"ai": None, "detector_shape": (4, 4)},
            stored_intersection_cache=(),
            unscaled_image=np.ones((4, 4), dtype=np.float64),
            last_caked_image_unscaled=None,
            last_caked_radial_values=None,
            last_caked_azimuth_values=None,
            last_caked_extent=None,
            last_caked_intersection_cache=None,
            last_caked_background_image_unscaled=None,
            last_q_space_image_unscaled=None,
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: {
            "image": np.ones((2, 2), dtype=np.float64),
            "radial": np.array([1.0, 2.0], dtype=np.float64),
            "azimuth": np.array([-1.0, 1.0], dtype=np.float64),
            "transform_bundle": None,
            "extent": [0.0, 1.0, 0.0, 1.0],
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_live_caked_transform_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_q_space_display_payload",
        lambda detector_image, **_kwargs: (
            q_space_inputs.append(np.asarray(detector_image, dtype=np.float64).copy())
            or {
                "image": np.asarray(detector_image, dtype=np.float64),
                "qr": np.array([0.0, 1.0], dtype=np.float64),
                "qz": np.array([0.0, 1.0], dtype=np.float64),
                "sum_signal": np.ones((2, 2), dtype=np.float64),
                "sum_normalization": np.ones((2, 2), dtype=np.float64),
                "count": np.ones((2, 2), dtype=np.float64),
                "extent": [0.0, 1.0, 0.0, 1.0],
            }
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_q_space_display_payload",
        lambda **kwargs: stored_payloads.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_current_background_backend",
        lambda: (_ for _ in ()).throw(AssertionError("restore should use native background")),
    )
    monkeypatch.setattr(
        runtime_session, "_get_current_background_native", lambda: background_native
    )
    monkeypatch.setattr(runtime_session, "center_x_var", _Var(2.0), raising=False)
    monkeypatch.setattr(runtime_session, "center_y_var", _Var(2.0), raising=False)
    monkeypatch.setattr(runtime_session, "corto_detector_var", _Var(0.5), raising=False)
    monkeypatch.setattr(runtime_session, "gamma_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "Gamma_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "chi_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "psi_z_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "cor_angle_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "zs_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "zb_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0e-4, raising=False)
    monkeypatch.setattr(runtime_session, "lambda_", 1.24, raising=False)
    monkeypatch.setattr(runtime_session, "psi", 0.0, raising=False)
    monkeypatch.setattr(
        runtime_session, "_current_effective_theta_initial", lambda strict_count=False: 0.0
    )
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "q_space")

    restored = runtime_session._restore_caked_display_payload_from_cached_results(
        background_visible=True,
        q_space_requested=False,
    )

    assert restored is True
    np.testing.assert_array_equal(
        runtime_session.simulation_runtime_state.last_caked_background_image_unscaled,
        np.ones((2, 2), dtype=np.float64),
    )
    assert len(q_space_inputs) == 2
    np.testing.assert_array_equal(q_space_inputs[1], background_native)
    assert len(stored_payloads) == 1


def test_restore_caked_payload_traps_live_q_space_rebuild_failures(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: float) -> None:
            self._value = value

        def get(self) -> float:
            return float(self._value)

    stored_payloads: list[dict[str, object]] = []
    progress_updates: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            analysis_preview_bins=(8, 6),
            last_analysis_cache_sig=("analysis", 4),
            last_res2_sim=object(),
            last_res2_background=object(),
            ai_cache={"ai": None, "detector_shape": (4, 4)},
            stored_intersection_cache=(),
            unscaled_image=np.ones((4, 4), dtype=np.float64),
            last_caked_image_unscaled=None,
            last_caked_radial_values=None,
            last_caked_azimuth_values=None,
            last_caked_extent=None,
            last_caked_intersection_cache=None,
            last_caked_background_image_unscaled=None,
            last_q_space_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_q_space_background_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_q_space_payload_signature=("stale", 1),
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: {
            "image": np.ones((2, 2), dtype=np.float64),
            "radial": np.array([1.0, 2.0], dtype=np.float64),
            "azimuth": np.array([-1.0, 1.0], dtype=np.float64),
            "transform_bundle": None,
            "extent": [0.0, 1.0, 0.0, 1.0],
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_live_caked_transform_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_q_space_display_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("q-space failure")),
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_q_space_display_payload",
        lambda **kwargs: stored_payloads.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label",
        SimpleNamespace(
            config=lambda **kwargs: progress_updates.append(str(kwargs.get("text", "")))
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "center_x_var", _Var(2.0), raising=False)
    monkeypatch.setattr(runtime_session, "center_y_var", _Var(2.0), raising=False)
    monkeypatch.setattr(runtime_session, "corto_detector_var", _Var(0.5), raising=False)
    monkeypatch.setattr(runtime_session, "gamma_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "Gamma_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "chi_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "psi_z_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "cor_angle_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "zs_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "zb_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0e-4, raising=False)
    monkeypatch.setattr(runtime_session, "lambda_", 1.24, raising=False)
    monkeypatch.setattr(runtime_session, "psi", 0.0, raising=False)
    monkeypatch.setattr(
        runtime_session, "_current_effective_theta_initial", lambda strict_count=False: 0.0
    )
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "q_space")

    restored = runtime_session._restore_caked_display_payload_from_cached_results(
        background_visible=True,
        q_space_requested=False,
    )

    assert restored is True
    assert stored_payloads == [{"sim_payload": None, "bg_payload": None}]
    assert runtime_session.simulation_runtime_state.last_q_space_payload_signature is None
    assert progress_updates == ["Q-space refresh failed: q-space failure"]


def test_analysis_display_payload_ready_requires_visible_background_payload() -> None:
    payload_ready = _load_runtime_session_function("_analysis_display_payload_ready")

    assert (
        payload_ready(
            show_caked_2d=True,
            q_space_requested=False,
            background_visible=True,
            caked_image=np.ones((2, 2), dtype=np.float64),
            caked_extent=[0.0, 1.0, 0.0, 1.0],
            caked_background_image=None,
            q_space_image=None,
            q_space_extent=None,
            q_space_background_image=None,
        )
        is False
    )
    assert (
        payload_ready(
            show_caked_2d=True,
            q_space_requested=False,
            background_visible=True,
            caked_image=np.ones((2, 2), dtype=np.float64),
            caked_extent=[0.0, 1.0, 0.0, 1.0],
            caked_background_image=np.ones((2, 2), dtype=np.float64),
            q_space_image=None,
            q_space_extent=None,
            q_space_background_image=None,
        )
        is True
    )
    assert (
        payload_ready(
            show_caked_2d=False,
            q_space_requested=True,
            background_visible=True,
            caked_image=None,
            caked_extent=None,
            caked_background_image=None,
            q_space_image=np.ones((2, 2), dtype=np.float64),
            q_space_extent=[0.0, 1.0, 0.0, 1.0],
            q_space_background_image=None,
        )
        is False
    )


def test_analysis_payload_ready_after_restore_attempt_restores_once_when_needed(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    restore_calls: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_extent=[0.0, 1.0, 0.0, 1.0],
            last_caked_background_image_unscaled=None,
            last_q_space_image_unscaled=None,
            last_q_space_extent=None,
            last_q_space_background_image_unscaled=None,
        ),
    )

    def _restore() -> None:
        restore_calls.append("restore")
        runtime_session.simulation_runtime_state.last_caked_background_image_unscaled = np.ones(
            (2, 2), dtype=np.float64
        )

    ready = runtime_session._analysis_payload_ready_after_restore_attempt(
        analysis_result_current=True,
        show_caked_2d=True,
        q_space_requested=False,
        background_visible=True,
        restore_payload=_restore,
    )

    assert ready is True
    assert restore_calls == ["restore"]


def test_analysis_payload_ready_after_restore_attempt_stays_false_when_restore_noops(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    restore_calls: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_extent=[0.0, 1.0, 0.0, 1.0],
            last_caked_background_image_unscaled=None,
            last_q_space_image_unscaled=None,
            last_q_space_extent=None,
            last_q_space_background_image_unscaled=None,
        ),
    )

    ready = runtime_session._analysis_payload_ready_after_restore_attempt(
        analysis_result_current=True,
        show_caked_2d=True,
        q_space_requested=False,
        background_visible=True,
        restore_payload=lambda: restore_calls.append("restore"),
    )

    assert ready is False
    assert restore_calls == ["restore"]


def test_analysis_payload_ready_after_restore_attempt_skips_restore_when_ready(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    restore_calls: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_extent=[0.0, 1.0, 0.0, 1.0],
            last_caked_background_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_q_space_image_unscaled=None,
            last_q_space_extent=None,
            last_q_space_background_image_unscaled=None,
        ),
    )

    ready = runtime_session._analysis_payload_ready_after_restore_attempt(
        analysis_result_current=True,
        show_caked_2d=True,
        q_space_requested=False,
        background_visible=True,
        restore_payload=lambda: restore_calls.append("restore"),
    )

    assert ready is True
    assert restore_calls == []


def test_restore_caked_payload_clears_stale_q_space_outside_live_q_space_mode(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    stored_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_res2_sim=object(),
            last_res2_background=object(),
            ai_cache={"ai": None, "detector_shape": (4, 4)},
            stored_intersection_cache=(),
            unscaled_image=np.ones((4, 4), dtype=np.float64),
            last_caked_image_unscaled=None,
            last_caked_radial_values=None,
            last_caked_azimuth_values=None,
            last_caked_extent=None,
            last_caked_intersection_cache=None,
            last_caked_background_image_unscaled=None,
            last_q_space_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_q_space_background_image_unscaled=np.ones((2, 2), dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: {
            "image": np.ones((2, 2), dtype=np.float64),
            "radial": np.array([1.0, 2.0], dtype=np.float64),
            "azimuth": np.array([-1.0, 1.0], dtype=np.float64),
            "transform_bundle": None,
            "extent": [0.0, 1.0, 0.0, 1.0],
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_live_caked_transform_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_q_space_display_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("restore should skip q-space rebuild outside q-space mode")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_q_space_display_payload",
        lambda **kwargs: stored_payloads.append(dict(kwargs)),
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0e-4, raising=False)
    monkeypatch.setattr(runtime_session, "corto_detector_var", _RuntimeVar(0.5), raising=False)
    monkeypatch.setattr(runtime_session, "center_x_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "center_y_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "detector")

    restored = runtime_session._restore_caked_display_payload_from_cached_results(
        background_visible=True,
        q_space_requested=True,
    )

    assert restored is True
    assert stored_payloads == [{"sim_payload": None, "bg_payload": None}]


def test_runtime_impl_hkl_pick_disarms_manual_geometry_and_preview_modes() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "deactivate_conflicting_modes_factory=lambda: (" in source
    assert "_set_geometry_manual_pick_mode(False)" in source
    assert "_set_geometry_preview_exclude_mode(False)" in source


def test_runtime_impl_hkl_pick_refreshes_mode_banner_without_viewport_hooks() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _handle_hkl_pick_mode_changed(_armed: bool) -> None:" in source
    assert 'globals().get("_refresh_fast_viewer_runtime_mode")' not in source
    assert "on_hkl_pick_mode_changed_factory=lambda: _handle_hkl_pick_mode_changed" in source


def test_runtime_impl_initializes_only_embedded_matplotlib_tk_primary_viewport() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "figure_canvas_cls = _get_tk_figure_canvas_cls()" in source
    assert "matplotlib_canvas = figure_canvas_cls(" in source
    assert "canvas = matplotlib_canvas" in source
    assert "_set_runtime_canvas(matplotlib_canvas)" in source
    assert "activate_runtime_primary_viewport(" not in source
    assert '"key": "primary_viewport_backend"' not in source
    assert "set_background_alpha=background_display.set_alpha" in source


def test_runtime_impl_shares_pick_hkl_live_cache_with_manual_qr_picker() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "geometry_manual_cache_workflow = (" in source
    assert "simulated_peaks_for_params=_geometry_manual_simulated_peaks_for_params" in source


def test_runtime_impl_allows_caked_preview_without_detector_accumulation() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "one_d_analysis_requested = bool(" in source
    assert "caked_analysis_requested = bool(" in source
    assert "accumulate_image_requested" in source
    assert "or one_d_analysis_requested" in source
    assert "ax.set_title('2D Caked Position Preview')" not in source
    assert "ax.set_title('2D Caked Position Preview (1D paused)')" not in source
    assert 'ax.set_title("")' in source


def test_runtime_impl_enables_async_preview_calculations_in_runtime_update_paths() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "FULL_UPDATE_DEBOUNCE_MS = 90" in source
    assert "PREVIEW_UPDATE_DEBOUNCE_MS = 24" in source
    assert "LIVE_DRAG_SETTLE_MS = 80" in source
    assert "PREVIEW_CALCULATIONS_ENABLED = True" in source
    assert "if not PREVIEW_CALCULATIONS_ENABLED:" in source
    assert "def _current_update_debounce_ms() -> int:" in source
    assert "_current_update_debounce_ms()," in source
    assert "simulation_runtime_state.interaction_drag_requires_settled_update = True" in source
    assert "simulation_runtime_state.interaction_settle_token = root.after(" in source
    assert 'preview_job["job_kind"] = "preview"' in source
    assert "request_status = _request_async_simulation_job(preview_job)" in source
    assert 'text="Computing preview simulation in background..."' in source
    assert "do_update_queue_live_preview" in source
    assert "preview_result = _run_simulation_generation_job" not in source
    assert "desired_analysis_preview = bool(" in source
    assert "PREVIEW_CALCULATIONS_ENABLED\n        and analysis_requested" in source
    assert (
        "PREVIEW_CALCULATIONS_ENABLED\n                and bool(live_geometry_preview_var.get())"
        in source
    )


def test_runtime_impl_gates_raw_hit_table_capture_by_job_kind() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    block_start = source.index("def _build_simulation_job(")
    block_end = source.index("def _build_scaled_primary_subset_payload(", block_start)
    block = source[block_start:block_end]

    assert "job_kind_value = str(job_kind)" in block
    assert (
        'capture_primary_raw = bool(run_primary_job and job_kind_value in {"full", "primary_fill"})'
        in block
    )
    assert (
        'capture_secondary_raw = bool(run_secondary_enabled and job_kind_value == "full")' in block
    )
    assert "build_intersection_cache_enabled = bool(" in block
    assert 'build_intersection_cache_for_job and job_kind_value == "full"' in block
    assert "(collect_hit_tables_enabled or build_intersection_cache_enabled)" in block
    assert "build_intersection_cache_enabled and collect_primary_hit_tables" in block
    assert "build_intersection_cache_enabled and collect_secondary_hit_tables" in block
    assert "selection_peak_cache_needed and collect_hit_tables_enabled" not in block
    assert "run_primary_job = bool(run_primary_job and primary_weight_active)" in block
    assert (
        "secondary_available_job = bool(secondary_available and secondary_weight_active)" in block
    )
    assert "active_peak_row_sides = _active_peak_row_sides_for_job(" in block
    assert '"secondary_available": bool(secondary_available_job)' in block
    assert '"active_peak_row_sides": tuple(active_peak_row_sides)' in block
    assert '"capture_primary_hit_tables_raw": capture_primary_raw' in block
    assert '"capture_secondary_hit_tables_raw": capture_secondary_raw' in block


def test_runtime_impl_blocks_startup_on_initial_simulation_with_overlay() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    block_start = source.index("has_cached_simulation = (")
    try:
        block_end = source.index(
            "elif ready_simulation_result is None and need_hit_table_refresh:",
            block_start,
        )
    except ValueError:
        block_end = source.index(
            "elif (\n        ready_simulation_result is None\n        and need_hit_table_refresh",
            block_start,
        )
    block = source[block_start:block_end]
    do_update_start = source.index("def do_update():")
    do_update_block = source[do_update_start:]
    startup_start = source.index("def _run_initial_startup_work():")
    startup_end = source.index("root.after_idle(_run_initial_startup_work)")
    startup_block = source[startup_start:startup_end]

    assert "def _schedule_post_idle_main_canvas_redraw() -> None:" in source
    assert "def _schedule_first_visible_simulation_settle_pass() -> None:" in source
    assert "def _show_initial_simulation_loading_overlay() -> None:" in source
    assert "Loading first simulation may take longer" in source
    assert "_show_initial_simulation_loading_overlay()" in startup_block
    assert "matplotlib_canvas.draw()" in startup_block
    assert "root.update_idletasks()" in startup_block
    assert "_request_main_canvas_redraw(force_matplotlib=True)" not in startup_block
    assert '"first_visible_simulation_settle_token"' in source
    assert '"first_visible_simulation_settled_signature"' in source
    assert "LIVE_DRAG_SETTLE_MS" in source
    assert (
        "detector_artist_signature_before = _current_detector_artist_source_signature()"
        in do_update_block
    )
    assert (
        "desired_detector_signature = _detector_display_raster_source_signature()"
        in do_update_block
    )
    assert (
        "detector_artist_signature_after = _current_detector_artist_source_signature()"
        in do_update_block
    )
    assert "_schedule_post_idle_main_canvas_redraw()" in do_update_block
    assert "_schedule_first_visible_simulation_settle_pass()" in do_update_block
    assert "and not bool(simulation_runtime_state.preview_active)" in do_update_block
    assert "and simulation_runtime_state.worker_active_job is None" in do_update_block
    assert "and simulation_runtime_state.worker_queued_job is None" in do_update_block
    assert "and not _live_interaction_active()" in do_update_block
    assert "update_canvas=False" in source
    assert "retry_allowed" in source
    assert "_request_legacy_main_matplotlib_redraw(force=True)" in source
    assert "start_exact_cake_numba_warmup_in_background()" not in startup_block
    assert "_run_simulation_generation_job(" in block
    assert "progress_label.config(text=_initial_simulation_progress_text())" in block


def test_runtime_impl_uses_numba_cache_heuristic_for_initial_loading_message() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "from ra_sim.user_paths import user_cache_root" in source
    assert "_NUMBA_CACHE_HAS_COMPILED_ARTIFACTS: bool | None = None" in source
    assert '_NUMBA_CACHE_WARM_MARKER_NAME = ".ra_sim_numba_cache_ready"' in source
    assert "def _numba_cache_dir() -> Path:" in source
    assert "NUMBA_CACHE_DIR" in source
    assert 'return user_cache_root() / "numba"' in source
    assert "def _numba_cache_warm_marker_path() -> Path:" in source
    assert "def _mark_numba_cache_compiled_artifacts_available() -> None:" in source
    assert "def _numba_cache_contains_compiled_artifacts() -> bool:" in source
    assert "if _NUMBA_CACHE_HAS_COMPILED_ARTIFACTS is not None:" in source
    assert "_NUMBA_CACHE_HAS_COMPILED_ARTIFACTS = True" in source
    assert "_NUMBA_CACHE_HAS_COMPILED_ARTIFACTS = bool(has_compiled_artifacts)" in source
    assert "def _numba_cache_has_compiled_artifact_files(cache_root: Path) -> bool:" not in source
    assert "marker_path.exists()" in source
    assert "marker_path.touch(exist_ok=True)" in source
    assert "def _initial_simulation_loading_message() -> str:" in source
    assert "Loading first simulation may take longer" in source
    assert "if this is the first run on this computer" in source
    assert "def _initial_simulation_progress_text() -> str:" in source
    assert '"Computing initial simulation..."' in source
    assert "First run on this computer may take longer." in source
    assert "_initial_simulation_loading_message()," in source
    assert "progress_label.config(text=_initial_simulation_progress_text())" in source


def test_runtime_session_geometry_manual_session_overlay_uses_projection_refresh(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    refresh_entry_geometry = lambda entry: {"refreshed": True, **dict(entry or {})}
    project_peaks_to_current_view = lambda entries: list(entries or [])
    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime_session, "_refresh_geometry_manual_pick_session", lambda: None)
    monkeypatch.setattr(
        runtime_session,
        "geometry_manual_projection_workflow",
        SimpleNamespace(
            refresh_entry_geometry=refresh_entry_geometry,
            project_peaks_to_current_view=project_peaks_to_current_view,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_manual_state",
        SimpleNamespace(
            pick_session={"group_key": ("q",), "group_entries": [], "pending_entries": []}
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pick_uses_caked_space",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_candidate_source_key",
        lambda entry: ("source", 1, 2) if entry is not None else None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_entry_display_coords",
        lambda _entry: (0.0, 0.0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_manual_geometry",
        SimpleNamespace(
            geometry_manual_session_initial_pairs_display=lambda *args, **kwargs: (
                captured.update(kwargs) or []
            )
        ),
        raising=False,
    )

    runtime_session._geometry_manual_session_initial_pairs_display()

    assert captured["refresh_entry_geometry"] is refresh_entry_geometry
    assert captured["project_peaks_to_current_view"] is project_peaks_to_current_view


def test_runtime_impl_defers_exact_cake_numba_warmup_until_after_startup_work() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "if analysis_sig is not None:" in source
    assert "if simulation_runtime_state.stored_sim_image is not None:" in source
    assert source.count("_schedule_exact_cake_numba_warmup_once()") >= 2
    assert source.count("_schedule_forward_simulation_numba_warmup_once()") >= 2
    assert source.count("_schedule_qr_rod_simulation_numba_warmup_once()") >= 2


def test_runtime_impl_distinguishes_preview_and_full_worker_jobs() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert 'str(payload.get("job_kind", "full")),' in source
    assert "def _simulation_result_matches_signature(" in source
    assert "def _simulation_result_superseded_by_queued_job(" in source
    assert "superseded = _simulation_result_superseded_by_queued_job(result, queued_job)" in source
    assert (
        "if isinstance(ready_result, dict) and not _simulation_result_matches_signature(" in source
    )
    assert '_promote_queued_simulation_job(reason="previous_ready_result_consumed")' in source


def test_runtime_impl_source_cache_build_ready_no_longer_inlines_caked_store() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "**dict(_store_worker_caked_view_for_background(bundle) or {})" not in source
    assert '"source_cache_rows_ready"' in source
    assert '"source_cache_caked_view_start"' in source
    assert '"source_cache_caked_view_timeout"' in source


def test_manual_caked_runtime_override_keeps_dynamic_exact_caked_path() -> None:
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")

    caked_cfg = geometry_fit.apply_manual_caked_point_geometry_fit_runtime_overrides(
        {"solver": {"dynamic_point_geometry_fit": False, "max_nfev": 99, "restarts": 4}},
        joint_background_mode=False,
    )
    caked_solver = caked_cfg["solver"]

    assert caked_solver["manual_point_fit_mode"] is True
    assert caked_solver["dynamic_point_geometry_fit"] is True
    assert caked_cfg["projection_view_mode"] == "caked"
    assert caked_solver["max_nfev"] == 30

    detector_cfg = geometry_fit.apply_manual_point_geometry_fit_runtime_overrides(
        {"solver": {"dynamic_point_geometry_fit": True, "max_nfev": 99}},
        joint_background_mode=False,
    )

    assert detector_cfg["solver"]["manual_point_fit_mode"] is True
    assert detector_cfg["solver"].get("dynamic_point_geometry_fit") is None
    assert detector_cfg.get("projection_view_mode") != "caked"


def test_manual_fit_space_classifier_rejects_mixed_backgrounds() -> None:
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")

    spaces = geometry_fit.geometry_manual_fit_space_by_background(
        [0, 1],
        {
            0: [{"pair_id": "caked", "background_two_theta_deg": 12.0, "background_phi_deg": 3.0}],
            1: [{"pair_id": "detector", "x": 10.0, "y": 20.0}],
        },
        pick_uses_caked_space=False,
        current_background_index=0,
    )

    assert spaces == {0: "caked", 1: "detector"}
    error = geometry_fit.manual_geometry_fit_space_preflight_error(
        spaces,
        osc_files=["caked.osc", "detector.osc"],
    )
    assert error is not None
    assert "mix detector-pixel and caked fit-space" in error
    assert "caked.osc=caked" in error
    assert "detector.osc=detector" in error


def test_prepare_caked_manual_fit_requires_exact_projector() -> None:
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")
    trace_pairs = [
        {
            "pair_id": "trace-0",
            "background_two_theta_deg": 1083.8179,
            "background_phi_deg": 1083.2697,
        },
        {
            "pair_id": "trace-1",
            "background_two_theta_deg": 1846.6204,
            "background_phi_deg": 1083.7344,
        },
    ]

    def _projector(cols, rows, **_kwargs):
        return {
            "two_theta_deg": np.asarray(cols, dtype=np.float64),
            "phi_deg": np.asarray(rows, dtype=np.float64),
            "valid": True,
            "fit_space_projector_kind": "exact_caked_bundle",
        }

    def _prepare_with_projector(projector, kind):
        events: list[str] = []
        measured = [
            dict(entry, fit_space_anchor_override=True, fit_space_anchor_source="test")
            for entry in trace_pairs
        ]

        def _build_dataset(background_index, **_kwargs):
            events.append("build")
            return {
                "dataset_index": int(background_index),
                "label": "trace.osc",
                "pair_count": len(measured),
                "resolved_source_pair_count": len(measured),
                "measured_for_fit": measured,
                "manual_point_pairs": measured,
                "spec": {
                    "dataset_index": int(background_index),
                    "label": "trace.osc",
                    "measured_peaks": measured,
                    "fit_space_projector": projector,
                    "fit_space_projector_kind": kind,
                    "fit_space_projector_unavailable_reason": (
                        None if kind == "exact_caked_bundle" else "missing_exact_caked_bundle"
                    ),
                },
            }

        def _ensure_caked_view() -> None:
            events.append("ensure")

        result = geometry_fit.prepare_geometry_fit_run(
            params={"theta_initial": 0.0},
            var_names=["zb"],
            fit_config={"geometry": {}},
            osc_files=["trace.osc"],
            current_background_index=0,
            theta_initial=0.0,
            preserve_live_theta=False,
            apply_geometry_fit_background_selection=lambda **_kwargs: True,
            current_geometry_fit_background_indices=lambda **_kwargs: [0],
            geometry_fit_uses_shared_theta_offset=lambda _indices: False,
            apply_background_theta_metadata=lambda **_kwargs: True,
            current_background_theta_values=lambda **_kwargs: [0.0],
            current_geometry_theta_offset=lambda **_kwargs: 0.0,
            geometry_manual_pairs_for_index=lambda _idx: trace_pairs,
            ensure_geometry_fit_caked_view=_ensure_caked_view,
            build_dataset=_build_dataset,
            build_runtime_config=lambda _params: {
                "solver": {"dynamic_point_geometry_fit": False, "max_nfev": 99}
            },
        )
        return result, events

    ok_result, ok_events = _prepare_with_projector(_projector, "exact_caked_bundle")

    assert ok_result.error_text is None
    assert ok_result.prepared_run is not None
    assert ok_events == ["ensure", "build"]
    ok_solver = ok_result.prepared_run.geometry_runtime_cfg["solver"]
    assert ok_solver["manual_point_fit_mode"] is True
    assert ok_solver["dynamic_point_geometry_fit"] is True
    assert ok_result.prepared_run.geometry_runtime_cfg["projection_view_mode"] == "caked"
    assert ok_result.prepared_run.dataset_specs[0]["fit_space_projector_kind"] == (
        "exact_caked_bundle"
    )
    assert geometry_fit.geometry_fit_datasets_use_caked_fit_space(
        ok_result.prepared_run.dataset_infos
    )

    missing_result, missing_events = _prepare_with_projector(None, None)

    assert missing_result.prepared_run is None
    assert missing_result.error_text is not None
    assert missing_events == ["ensure", "build"]
    assert "exact caked fit-space projector" in missing_result.error_text
    assert "Rebuild the caked/source cache" in missing_result.error_text


def test_prepare_caked_manual_fit_fails_without_exact_projector_no_recake() -> None:
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")
    caked_pairs = [
        {"pair_id": "caked", "background_two_theta_deg": 12.0, "background_phi_deg": 3.0}
    ]
    events: list[str] = []

    def _ensure_caked_view() -> None:
        events.append("ensure")
        raise RuntimeError("no exact payload")

    def _build_dataset(*_args, **_kwargs):
        pytest.fail("caked ensure failure must stop before dataset build")

    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 0.0},
        var_names=["zb"],
        fit_config={"geometry": {}},
        osc_files=["caked.osc"],
        current_background_index=0,
        theta_initial=0.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=lambda **_kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda _indices: False,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [0.0],
        current_geometry_theta_offset=lambda **_kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda _idx: caked_pairs,
        ensure_geometry_fit_caked_view=_ensure_caked_view,
        build_dataset=_build_dataset,
        build_runtime_config=lambda _params: {"solver": {}},
    )

    assert events == ["ensure"]
    assert result.prepared_run is None
    assert result.error_text is not None
    assert "exact caked fit-space projector could not be prepared" in result.error_text
    assert "Rebuild the caked/source cache" in result.error_text


def test_prepare_detector_manual_fit_ignores_projector_presence_for_routing() -> None:
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")
    detector_pairs = [
        {"pair_id": "detector-0", "x": 10.0, "y": 20.0},
        {"pair_id": "detector-1", "x": 30.0, "y": 40.0},
    ]

    def _projector(cols, rows, **_kwargs):
        return {
            "two_theta_deg": np.asarray(cols, dtype=np.float64),
            "phi_deg": np.asarray(rows, dtype=np.float64),
            "valid": True,
            "fit_space_projector_kind": "exact_caked_bundle",
        }

    def _build_dataset(background_index, **_kwargs):
        return {
            "dataset_index": int(background_index),
            "label": "detector.osc",
            "pair_count": len(detector_pairs),
            "resolved_source_pair_count": len(detector_pairs),
            "measured_for_fit": detector_pairs,
            "manual_point_pairs": detector_pairs,
            "spec": {
                "dataset_index": int(background_index),
                "label": "detector.osc",
                "measured_peaks": detector_pairs,
                "fit_space_projector": _projector,
                "fit_space_projector_kind": "exact_caked_bundle",
            },
        }

    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 0.0},
        var_names=["zb"],
        fit_config={"geometry": {}},
        osc_files=["detector.osc"],
        current_background_index=0,
        theta_initial=0.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=lambda **_kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda _indices: False,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [0.0],
        current_geometry_theta_offset=lambda **_kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda _idx: detector_pairs,
        ensure_geometry_fit_caked_view=lambda: None,
        build_dataset=_build_dataset,
        build_runtime_config=lambda _params: {
            "projection_view_mode": "caked",
            "solver": {"dynamic_point_geometry_fit": True, "max_nfev": 99},
        },
    )

    assert result.error_text is None
    assert result.prepared_run is not None
    detector_solver = result.prepared_run.geometry_runtime_cfg["solver"]
    assert detector_solver["manual_point_fit_mode"] is True
    assert detector_solver.get("dynamic_point_geometry_fit") is None
    assert result.prepared_run.geometry_runtime_cfg.get("projection_view_mode") != "caked"


def test_prepare_mixed_manual_fit_spaces_fails_before_dataset_build() -> None:
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")
    caked_pairs = [
        {"pair_id": "caked", "background_two_theta_deg": 12.0, "background_phi_deg": 3.0}
    ]
    detector_pairs = [{"pair_id": "detector", "x": 10.0, "y": 20.0}]

    def _pairs_for_index(index: int):
        return caked_pairs if int(index) == 0 else detector_pairs

    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 0.0},
        var_names=["zb"],
        fit_config={"geometry": {}},
        osc_files=["caked.osc", "detector.osc"],
        current_background_index=0,
        theta_initial=0.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=lambda **_kwargs: [0, 1],
        geometry_fit_uses_shared_theta_offset=lambda _indices: True,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [0.0, 0.0],
        current_geometry_theta_offset=lambda **_kwargs: 0.0,
        geometry_manual_pairs_for_index=_pairs_for_index,
        ensure_geometry_fit_caked_view=lambda: pytest.fail("mixed spaces must fail before ensure"),
        build_dataset=lambda *_args, **_kwargs: pytest.fail(
            "mixed spaces must fail before dataset build"
        ),
        build_runtime_config=lambda _params: {"solver": {}},
    )

    assert result.prepared_run is None
    assert result.error_text is not None
    assert "mix detector-pixel and caked fit-space" in result.error_text


def test_prepare_same_background_mixed_manual_fit_space_fails_before_dataset_build() -> None:
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")
    mixed_pairs = [
        {"pair_id": "caked", "background_two_theta_deg": 12.0, "background_phi_deg": 3.0},
        {"pair_id": "detector", "x": 10.0, "y": 20.0},
    ]

    spaces = geometry_fit.geometry_manual_fit_space_by_background(
        [0],
        {0: mixed_pairs},
        pick_uses_caked_space=False,
        current_background_index=0,
    )
    assert spaces == {0: "mixed"}

    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 0.0},
        var_names=["zb"],
        fit_config={"geometry": {}},
        osc_files=["mixed.osc"],
        current_background_index=0,
        theta_initial=0.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=lambda **_kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda _indices: False,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [0.0],
        current_geometry_theta_offset=lambda **_kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda _idx: mixed_pairs,
        ensure_geometry_fit_caked_view=lambda: pytest.fail(
            "same-background mixed spaces must fail before ensure"
        ),
        build_dataset=lambda *_args, **_kwargs: pytest.fail(
            "same-background mixed spaces must fail before dataset build"
        ),
        build_runtime_config=lambda _params: {"solver": {}},
    )

    assert result.prepared_run is None
    assert result.error_text is not None
    assert "within the same background" in result.error_text
    assert "mixed.osc=mixed" in result.error_text


def test_async_geometry_fit_job_preserves_caked_runtime_fields(monkeypatch, tmp_path) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_fit = runtime_session.gui_geometry_fit
    ensure_calls: list[str] = []
    radial_axis = np.asarray([10.0, 11.0], dtype=np.float64)
    azimuth_axis = np.asarray([1.0, 2.0], dtype=np.float64)
    raw_azimuth_axis = np.asarray(
        geometry_fit.gui_phi_to_raw_phi(azimuth_axis),
        dtype=np.float64,
    )
    transform_bundle = geometry_fit.CakeTransformBundle(
        detector_shape=(4, 4),
        radial_deg=radial_axis,
        raw_azimuth_deg=raw_azimuth_axis,
        gui_azimuth_deg=azimuth_axis,
        lut=object(),
    )
    caked_payload = {
        "background": np.ones((4, 4), dtype=np.float64),
        "radial_axis": radial_axis,
        "azimuth_axis": azimuth_axis,
        "raw_azimuth_axis": raw_azimuth_axis,
        "raw_to_gui_row_permutation": np.asarray([0, 1], dtype=np.int32),
        "transform_bundle": transform_bundle,
        "detector_shape": (4, 4),
    }
    manual_pairs = [
        {
            "pair_id": "caked-0",
            "background_two_theta_deg": 10.0,
            "background_phi_deg": 1.0,
        }
    ]
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["bg0.osc"],
        current_background_index=0,
        image_size=4,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda _idx: list(manual_pairs),
        load_background_by_index=lambda _idx: (
            np.ones((4, 4), dtype=np.float64),
            np.ones((4, 4), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        geometry_manual_simulated_lookup=lambda _rows: {},
        geometry_manual_entry_display_coords=lambda _entry: None,
        unrotate_display_peaks=lambda entries, shape, *, k: list(entries),
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda *_args, **_kwargs: ({}, {"pairs": 0}),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: False,
        geometry_manual_caked_view_for_index=lambda _idx: dict(caked_payload),
    )
    prepare_bindings = geometry_fit.GeometryFitRuntimePreparationBindings(
        fit_config={"geometry": {"solver": {"dynamic_point_geometry_fit": False}}},
        theta_initial=0.0,
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=lambda **_kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda _indices: False,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [0.0],
        current_geometry_theta_offset=lambda **_kwargs: 0.0,
        ensure_geometry_fit_caked_view=lambda: ensure_calls.append("ensure"),
        manual_dataset_bindings=manual_dataset_bindings,
        build_runtime_config=lambda _params: {"solver": {"dynamic_point_geometry_fit": False}},
    )
    execution_bindings = SimpleNamespace(
        downloads_dir=tmp_path,
        log_dir=tmp_path,
        simulation_runtime_state=SimpleNamespace(
            geometry_fit_job_counter=0,
            geometry_fit_event_queue=runtime_session.queue.Queue(),
            source_row_snapshots={},
            last_simulation_signature=("sig",),
        ),
        solver_inputs=SimpleNamespace(miller=[], intensities=[], image_size=4),
        background_runtime_state=SimpleNamespace(current_background_index=0),
    )
    bindings = SimpleNamespace(
        value_callbacks=geometry_fit.GeometryFitRuntimeValueCallbacks(
            current_var_names=lambda: ["zb"],
            current_params=lambda: {
                "theta_initial": 0.0,
                "center": [1.0, 1.0],
                "corto_detector": 0.5,
                "lambda": 1.54e-10,
            },
            current_ui_params=lambda: {},
            var_map={},
            build_mosaic_params=lambda **_kwargs: {},
        ),
        prepare_bindings_factory=lambda _var_names: prepare_bindings,
        execution_bindings=execution_bindings,
        solve_fit=lambda *_args, **_kwargs: None,
        stamp_factory=lambda: "20260422_000000",
    )

    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_background_label",
        lambda idx: f"bg{int(idx)}.osc",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_for_background",
        lambda idx, params: ("sig", int(idx)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_signature_summary",
        lambda signature: repr(signature),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_view_for_index",
        lambda _idx: dict(caked_payload),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda background_index, **kwargs: {
            "background_index": int(background_index),
            "mode": kwargs.get("mode_override") or "detector",
            "available": True,
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_build_live_preview_simulated_peaks_from_cache",
        lambda: [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_last_live_preview_cache_metadata",
        lambda: {},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_cache_inventory_snapshot",
        lambda: {"source_snapshot_count": 0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_last_source_snapshot_diagnostics",
        lambda: {},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_last_simulation_diagnostics",
        lambda: {},
        raising=False,
    )

    job = runtime_session._build_geometry_fit_async_job(bindings)

    assert ensure_calls == ["ensure"]
    assert job["manual_fit_space_by_background"] == {0: "caked"}
    assert job["pick_uses_caked_space"] is True
    assert job["projection_view_mode"] == "caked"
    assert job["geometry_runtime_cfg"]["solver"]["dynamic_point_geometry_fit"] is True
    assert (
        job["caked_views_by_background"][0]["transform_bundle"] is caked_payload["transform_bundle"]
    )

    manual_pairs[:] = [{"pair_id": "detector-0", "x": 10.0, "y": 20.0}]
    ensure_calls.clear()
    execution_bindings.simulation_runtime_state.geometry_fit_job_counter = 0
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_view_for_index",
        lambda _idx: pytest.fail("detector manual fit must not load caked payload"),
        raising=False,
    )

    detector_job = runtime_session._build_geometry_fit_async_job(bindings)

    assert ensure_calls == []
    assert detector_job["manual_fit_space_by_background"] == {0: "detector"}
    assert detector_job["projection_view_mode"] == "detector"
    assert detector_job["caked_views_by_background"] == {}
    assert detector_job["geometry_runtime_cfg"]["solver"].get("dynamic_point_geometry_fit") is None

    manual_pairs[:] = [
        {
            "pair_id": "caked-0",
            "background_two_theta_deg": 10.0,
            "background_phi_deg": 1.0,
        },
        {"pair_id": "detector-0", "x": 10.0, "y": 20.0},
    ]
    ensure_calls.clear()

    with pytest.raises(RuntimeError, match="within the same background"):
        runtime_session._build_geometry_fit_async_job(bindings)

    assert ensure_calls == []


def test_geometry_fit_worker_rejects_stored_mixed_manual_fit_space_before_prepare(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _make_geometry_fit_worker_job(runtime_session)
    prepare_calls: list[dict[str, object]] = []

    job["manual_fit_space_by_background"] = {0: "mixed"}
    job["manual_pairs_by_background"] = {
        0: [
            {
                "pair_id": "caked",
                "background_two_theta_deg": 10.0,
                "background_phi_deg": 1.0,
            },
            {"pair_id": "detector", "x": 1.0, "y": 2.0},
        ]
    }

    def _prepare_geometry_fit_run(**kwargs):
        prepare_calls.append(dict(kwargs))
        pytest.fail("mixed worker fit space must fail before prepare")

    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "prepare_geometry_fit_run",
        _prepare_geometry_fit_run,
    )

    result = runtime_session._run_async_geometry_fit_worker_job(job)
    events = _drain_geometry_fit_worker_events(job["event_queue"])
    failure_events = [event for event in events if event.get("kind") == "preflight_failure"]

    assert prepare_calls == []
    assert result.error_text is not None
    assert failure_events
    assert "mixed detector/caked manual fit spaces are not supported" in str(
        dict(failure_events[-1].get("payload") or {}).get("message")
    )


def test_headless_caked_manual_payload_uses_shared_exact_caked_builder(monkeypatch) -> None:
    headless = importlib.import_module("ra_sim.headless_geometry_fit")
    calls: list[dict[str, object]] = []

    def _fake_shared_builder(detector_image, *, ai, detector_shape, npt_rad, npt_azim):
        calls.append(
            {
                "detector_shape": tuple(detector_shape),
                "image_shape": tuple(np.asarray(detector_image).shape),
                "ai": ai,
                "npt_rad": int(npt_rad),
                "npt_azim": int(npt_azim),
            }
        )
        return {
            "background": np.ones((8, 8), dtype=np.float64),
            "radial_axis": np.linspace(0.0, 1.0, 8),
            "azimuth_axis": np.linspace(-1.0, 1.0, 8),
            "raw_azimuth_axis": np.linspace(-180.0, 180.0, 8),
            "raw_to_gui_row_permutation": np.arange(8, dtype=np.int32),
            "transform_bundle": object(),
            "detector_shape": tuple(detector_shape),
        }

    monkeypatch.setattr(
        headless.gui_geometry_fit,
        "build_geometry_fit_exact_caked_view_payload",
        _fake_shared_builder,
    )

    payload = headless._build_headless_geometry_fit_caked_view_payload(
        np.ones((8, 8), dtype=np.float64),
        params={
            "center": [4.0, 4.0],
            "center_x": 4.0,
            "center_y": 4.0,
            "corto_detector": 0.5,
            "lambda": 1.54,
        },
        pixel_size_m=1.0e-4,
        npt_rad=8,
        npt_azim=8,
    )

    assert calls and calls[0]["detector_shape"] == (8, 8)
    assert calls[0]["image_shape"] == (8, 8)
    assert calls[0]["npt_rad"] == 8
    assert calls[0]["npt_azim"] == 8
    assert isinstance(payload, dict)
    assert payload["ai"] is calls[0]["ai"]
    assert payload["transform_bundle"] is not None
    assert np.asarray(payload["background"]).shape == (8, 8)
    assert np.asarray(payload["background_image"]).shape == (8, 8)
    assert np.asarray(payload["radial_axis"]).size == 8
    assert np.asarray(payload["azimuth_axis"]).size == 8
    assert tuple(payload["detector_shape"]) == (8, 8)


def test_headless_hydrated_caked_payload_projects_rows_per_background(
    monkeypatch,
    tmp_path,
) -> None:
    headless = importlib.import_module("ra_sim.headless_geometry_fit")
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")
    projected_rows: list[dict[str, object]] = []
    captured_bundles: list[object] = []

    class _StopAfterProjection(RuntimeError):
        pass

    def _defaults() -> dict[str, object]:
        return {
            "zb": 0.0,
            "zs": 0.0,
            "theta_initial": 0.0,
            "psi_z": 0.0,
            "chi": 0.0,
            "cor_angle": 0.0,
            "sample_width_m": 0.0,
            "sample_length_m": 0.0,
            "sample_depth_m": 0.0,
            "gamma": 0.0,
            "Gamma": 0.0,
            "corto_detector": 0.5,
            "a": 4.0,
            "c": 7.0,
            "center_x": 1.0,
            "center_y": 1.0,
            "debye_x": 0.0,
            "debye_y": 0.0,
            "sigma_mosaic_deg": 0.8,
            "gamma_mosaic_deg": 0.7,
            "eta": 0.0,
            "bandwidth_percent": 0.0,
            "solve_q_steps": 100,
            "solve_q_rel_tol": 1.0e-3,
            "solve_q_mode": 0,
            "optics_mode": "fast",
            "p0": 0.5,
            "p1": 0.5,
            "p2": 0.5,
            "w0": 1.0,
            "w1": 0.0,
            "w2": 0.0,
            "finite_stack": False,
            "stack_layers": 1,
            "phase_delta_expression": "0.0",
            "phi_l_divisor": 1.0,
            "weight1": 1.0,
            "weight2": 0.0,
        }

    exact_bundles = {
        0: geometry_fit.CakeTransformBundle(
            detector_shape=(4, 4),
            radial_deg=np.asarray([10.0, 11.0], dtype=np.float64),
            raw_azimuth_deg=np.asarray(
                geometry_fit.gui_phi_to_raw_phi([1.0, 2.0]),
                dtype=np.float64,
            ),
            gui_azimuth_deg=np.asarray([1.0, 2.0], dtype=np.float64),
            lut=object(),
        ),
        1: geometry_fit.CakeTransformBundle(
            detector_shape=(4, 4),
            radial_deg=np.asarray([101.0, 102.0], dtype=np.float64),
            raw_azimuth_deg=np.asarray(
                geometry_fit.gui_phi_to_raw_phi([3.0, 4.0]),
                dtype=np.float64,
            ),
            gui_azimuth_deg=np.asarray([3.0, 4.0], dtype=np.float64),
            lut=object(),
        ),
    }

    monkeypatch.setattr(
        headless,
        "_build_runtime_defaults",
        lambda _state: headless._RuntimeDefaults(
            primary_cif_path="fake.cif",
            secondary_cif_path=None,
            osc_files=["bg0.osc", "bg1.osc"],
            current_background_index=0,
            image_size=4,
            pixel_size_m=1.0,
            lambda_angstrom=1.54,
            psi_deg=0.0,
            defaults=_defaults(),
            fit_config={"geometry": {}},
            intensity_threshold=0.0,
            include_rods_flag=False,
            two_theta_range=(0.0, 1.0),
            mx=1,
            background_flags={
                "backend_rotation_k": 0,
                "backend_flip_x": False,
                "backend_flip_y": False,
            },
        ),
    )
    monkeypatch.setattr(
        headless,
        "_restore_manual_pairs",
        lambda *_args, **_kwargs: {
            0: [{"background_two_theta_deg": 10.0, "background_phi_deg": 1.0}],
            1: [{"background_two_theta_deg": 101.0, "background_phi_deg": 3.0}],
        },
    )
    monkeypatch.setattr(
        headless,
        "_load_structure_model",
        lambda *_args, **_kwargs: (
            SimpleNamespace(miller=[], intensities=[]),
            SimpleNamespace(),
            "fake.cif",
            1.0 + 0.0j,
        ),
    )
    monkeypatch.setattr(
        headless.gui_background_theta,
        "default_geometry_fit_background_selection",
        lambda *, osc_files: "all",
    )
    monkeypatch.setattr(
        headless.gui_background_theta,
        "format_background_theta_values",
        lambda values: ",".join(str(float(value)) for value in values),
    )
    monkeypatch.setattr(
        headless.gui_manual_geometry,
        "geometry_manual_pairs_for_index",
        lambda idx, *, pairs_by_background: list(pairs_by_background.get(int(idx), ())),
    )
    monkeypatch.setattr(
        headless.gui_background,
        "apply_background_backend_orientation",
        lambda image, **_kwargs: image,
    )

    def _load_background_image_by_index(
        index,
        *,
        background_images,
        background_images_native,
        background_images_display,
        **_kwargs,
    ):
        idx = int(index)
        native = np.full((4, 4), float(idx), dtype=np.float64)
        display = native.copy()
        images = list(background_images)
        natives = list(background_images_native)
        displays = list(background_images_display)
        while len(images) <= idx:
            images.append(None)
            natives.append(None)
            displays.append(None)
        images[idx] = native
        natives[idx] = native
        displays[idx] = display
        return {
            "background_images": images,
            "background_images_native": natives,
            "background_images_display": displays,
            "background_image": native,
            "background_display": display,
        }

    monkeypatch.setattr(
        headless.gui_background,
        "load_background_image_by_index",
        _load_background_image_by_index,
    )
    monkeypatch.setattr(
        headless.gui_geometry_q_group_manager,
        "make_runtime_geometry_fit_simulation_callbacks",
        lambda **_kwargs: SimpleNamespace(
            simulated_peaks_for_params=lambda *_args, **_kwargs: [],
            simulated_lookup=lambda _rows: {},
            last_simulation_diagnostics=lambda: {},
        ),
    )

    def _payload(detector_image, **_kwargs):
        marker = int(float(np.asarray(detector_image, dtype=np.float64)[0, 0]))
        bundle = exact_bundles[int(marker)]
        return {
            "background": np.full(
                (bundle.gui_azimuth_deg.size, bundle.radial_deg.size),
                float(marker),
                dtype=np.float64,
            ),
            "background_image": np.full(
                (bundle.gui_azimuth_deg.size, bundle.radial_deg.size),
                float(marker),
                dtype=np.float64,
            ),
            "radial_axis": np.asarray(bundle.radial_deg, dtype=np.float64),
            "azimuth_axis": np.asarray(bundle.gui_azimuth_deg, dtype=np.float64),
            "raw_azimuth_axis": np.asarray(bundle.raw_azimuth_deg, dtype=np.float64),
            "raw_to_gui_row_permutation": np.arange(
                int(bundle.gui_azimuth_deg.size),
                dtype=np.int32,
            ),
            "detector_shape": (4, 4),
            "projection_view_mode": "caked",
            "transform_bundle": "json-safe-bundle",
        }

    monkeypatch.setattr(headless, "_build_headless_geometry_fit_caked_view_payload", _payload)

    def _hydrate(payload, **_kwargs):
        marker = int(float(np.asarray(payload["background"], dtype=np.float64)[0, 0]))
        hydrated = dict(payload)
        hydrated["transform_bundle"] = exact_bundles[marker]
        return hydrated

    monkeypatch.setattr(
        geometry_fit,
        "_geometry_fit_hydrate_exact_caked_payload",
        _hydrate,
    )

    def _make_projection_callbacks(**kwargs):
        def _project(rows):
            bundle = kwargs["caked_transform_bundle"]()
            captured_bundles.append(bundle)
            marker = int(float(np.asarray(kwargs["current_background_native"]())[0, 0]))
            return [
                dict(entry, projected_background=marker, used_exact_bundle=bundle)
                for entry in rows or ()
                if isinstance(entry, Mapping)
            ]

        return SimpleNamespace(
            project_peaks_to_current_view=_project,
            simulated_peaks_for_params=lambda *_args, **_kwargs: [],
            simulated_lookup=lambda _rows: {},
            entry_display_coords=lambda _entry: None,
            refresh_entry_geometry=lambda entry: dict(entry),
        )

    monkeypatch.setattr(
        headless.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        _make_projection_callbacks,
    )

    def _prepare_runtime_geometry_fit_run(*, bindings, **_kwargs):
        dataset_bindings = bindings.manual_dataset_bindings
        for background_idx in (0, 1):
            cached = dataset_bindings.geometry_manual_caked_view_for_index(background_idx)
            assert cached["transform_bundle"] is exact_bundles[background_idx]
            cached["transform_bundle"] = "json-safe-bundle"
        rows = [
            {"background_index": 0, "row_id": "first"},
            {"background_index": 1, "row_id": "second"},
            {"background_index": 0, "row_id": "third"},
        ]
        for row in rows:
            projected_rows.extend(
                dataset_bindings.geometry_manual_project_peaks_for_background_view(
                    int(row["background_index"]),
                    [row],
                )
            )
        raise _StopAfterProjection()

    monkeypatch.setattr(
        geometry_fit,
        "prepare_runtime_geometry_fit_run",
        _prepare_runtime_geometry_fit_run,
    )

    with pytest.raises(_StopAfterProjection):
        headless.run_headless_geometry_fit(
            {"geometry": {"manual_pairs": []}},
            state_path=tmp_path / "state.json",
            downloads_dir=tmp_path,
        )

    assert [(row["row_id"], row["projected_background"]) for row in projected_rows] == [
        ("first", 0),
        ("second", 1),
        ("third", 0),
    ]
    assert captured_bundles == [exact_bundles[0], exact_bundles[1], exact_bundles[0]]
    assert [row["used_exact_bundle"] for row in projected_rows] == [
        exact_bundles[0],
        exact_bundles[1],
        exact_bundles[0],
    ]


def test_cli_hydrated_caked_payload_projects_rows_per_background(
    monkeypatch,
    tmp_path,
) -> None:
    cli = importlib.import_module("ra_sim.cli")
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")
    projected_rows: list[dict[str, object]] = []
    captured_bundles: list[object] = []

    class _StopAfterProjection(RuntimeError):
        pass

    primary_cif_path = tmp_path / "primary.cif"
    primary_cif_path.write_text("data_test\n", encoding="utf-8")
    exact_bundles = {
        0: geometry_fit.CakeTransformBundle(
            detector_shape=(4, 4),
            radial_deg=np.asarray([10.0, 11.0], dtype=np.float64),
            raw_azimuth_deg=np.asarray(
                geometry_fit.gui_phi_to_raw_phi([1.0, 2.0]),
                dtype=np.float64,
            ),
            gui_azimuth_deg=np.asarray([1.0, 2.0], dtype=np.float64),
            lut=object(),
        ),
        1: geometry_fit.CakeTransformBundle(
            detector_shape=(4, 4),
            radial_deg=np.asarray([101.0, 102.0], dtype=np.float64),
            raw_azimuth_deg=np.asarray(
                geometry_fit.gui_phi_to_raw_phi([3.0, 4.0]),
                dtype=np.float64,
            ),
            gui_azimuth_deg=np.asarray([3.0, 4.0], dtype=np.float64),
            lut=object(),
        ),
    }

    class _SimulationRuntimeState(SimpleNamespace):
        def __init__(self, *args, **kwargs):
            super().__init__(**kwargs)
            self.geometry_fit_caking_ai_cache = {}
            self.profile_cache = dict(getattr(self, "profile_cache", {}) or {})
            self.source_row_snapshots = {}

    simulation_defaults = cli.HeadlessSimulationDefaults(
        out_path=str(tmp_path / "out.json"),
        image_size=4,
        samples=1,
        vmax=1.0,
        cif_file=str(primary_cif_path),
        geometry=SimpleNamespace(
            pixel_size_m=1.0,
            lambda_angstrom=1.54,
            theta_initial_deg=0.0,
            cor_angle_deg=0.0,
            chi_deg=0.0,
            psi_z_deg=0.0,
            zs=0.0,
            zb=0.0,
            sample_width_m=0.0,
            sample_length_m=0.0,
            Gamma_deg=0.0,
            gamma_deg=0.0,
            distance_m=0.5,
            center=[1.0, 1.0],
        ),
        mosaic=SimpleNamespace(),
        debye_waller=SimpleNamespace(x=0.0, y=0.0),
        occ=(1.0,),
        p_values=(0.5,),
        weights=None,
        two_theta_max=1.0,
        ht_max_miller_index=1,
        ht_phase_delta_expression="0.0",
        ht_phi_l_divisor=1.0,
        ht_finite_stack=False,
        ht_stack_layers=1,
        divergence_sigma_rad=0.0,
        bandwidth_sigma=0.0,
        bandwidth_fraction=0.0,
        sample_depth_m=0.0,
    )

    def _load_background_image_by_index(
        index,
        *,
        background_images,
        background_images_native,
        background_images_display,
        **_kwargs,
    ):
        idx = int(index)
        native = np.full((4, 4), float(idx), dtype=np.float64)
        display = native.copy()
        images = list(background_images)
        natives = list(background_images_native)
        displays = list(background_images_display)
        while len(images) <= idx:
            images.append(None)
            natives.append(None)
            displays.append(None)
        images[idx] = native
        natives[idx] = native
        displays[idx] = display
        return {
            "background_images": images,
            "background_images_native": natives,
            "background_images_display": displays,
            "background_image": native,
            "background_display": display,
        }

    def _make_projection_callbacks(**kwargs):
        def _project(rows):
            bundle = kwargs["caked_transform_bundle"]()
            captured_bundles.append(bundle)
            marker = int(float(np.asarray(kwargs["current_background_native"]())[0, 0]))
            return [
                dict(entry, projected_background=marker, used_exact_bundle=bundle)
                for entry in rows or ()
                if isinstance(entry, Mapping)
            ]

        return SimpleNamespace(
            project_peaks_to_current_view=_project,
            simulated_peaks_for_params=lambda *_args, **_kwargs: [],
            simulated_lookup=lambda _rows: {},
            entry_display_coords=lambda _entry: None,
            refresh_entry_geometry=lambda entry: dict(entry),
        )

    def _payload(detector_image, **_kwargs):
        marker = int(float(np.asarray(detector_image, dtype=np.float64)[0, 0]))
        bundle = exact_bundles[int(marker)]
        return {
            "background": np.full(
                (bundle.gui_azimuth_deg.size, bundle.radial_deg.size),
                float(marker),
                dtype=np.float64,
            ),
            "background_image": np.full(
                (bundle.gui_azimuth_deg.size, bundle.radial_deg.size),
                float(marker),
                dtype=np.float64,
            ),
            "radial_axis": np.asarray(bundle.radial_deg, dtype=np.float64),
            "azimuth_axis": np.asarray(bundle.gui_azimuth_deg, dtype=np.float64),
            "raw_azimuth_axis": np.asarray(bundle.raw_azimuth_deg, dtype=np.float64),
            "raw_to_gui_row_permutation": np.arange(
                int(bundle.gui_azimuth_deg.size),
                dtype=np.int32,
            ),
            "detector_shape": (4, 4),
            "projection_view_mode": "caked",
            "transform_bundle": "json-safe-bundle",
        }

    geometry_modules = SimpleNamespace(
        gui_background=SimpleNamespace(
            load_background_image_by_index=_load_background_image_by_index,
            apply_background_backend_orientation=lambda image, **_kwargs: image,
        ),
        gui_background_theta=SimpleNamespace(
            default_geometry_fit_background_selection=lambda *, osc_files: "all",
            format_background_theta_values=lambda values: ",".join(
                str(float(value)) for value in values
            ),
            apply_background_theta_metadata=lambda **_kwargs: True,
            apply_geometry_fit_background_selection=lambda **_kwargs: True,
            current_geometry_fit_background_indices=lambda **_kwargs: [0, 1],
            geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: True,
            current_geometry_theta_offset=lambda **_kwargs: 0.0,
            current_background_theta_values=lambda **_kwargs: [0.0, 0.0],
            background_theta_for_index=lambda index, **_kwargs: 0.0,
        ),
        gui_controllers=SimpleNamespace(
            clamp_site_occupancy_values=lambda values, fallback_values=None: list(
                values or fallback_values or [1.0]
            ),
            combine_cif_weighted_intensities=lambda *args, **kwargs: None,
            normalize_stacking_weight_values=lambda values: list(values),
        ),
        gui_geometry_fit=geometry_fit,
        gui_geometry_overlay=SimpleNamespace(
            rotate_point_for_display=lambda col, row, *_args, **_kwargs: (
                float(col),
                float(row),
            ),
            display_to_native_sim_coords=lambda col, row, *_args, **_kwargs: (
                float(col),
                float(row),
            ),
            native_sim_to_display_coords=lambda col, row, *_args, **_kwargs: (
                float(col),
                float(row),
            ),
            unrotate_display_peaks=lambda entries, *_args, **_kwargs: list(entries),
            select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
                {"label": "identity"},
                {"pairs": len(sim_pts)},
            ),
            apply_orientation_to_entries=lambda entries, *_args, **_kwargs: list(entries),
            orient_image_for_fit=lambda image, **_kwargs: image,
        ),
        gui_geometry_q_group_manager=SimpleNamespace(
            make_runtime_geometry_fit_simulation_callbacks=lambda **_kwargs: SimpleNamespace(
                simulated_peaks_for_params=lambda *_args, **_kwargs: [],
                simulated_lookup=lambda _rows: {},
                last_simulation_diagnostics=lambda: {},
            )
        ),
        gui_manual_geometry=SimpleNamespace(
            apply_geometry_manual_pairs_rows=lambda *_args, replace_pairs_by_background, **_kwargs: (
                replace_pairs_by_background(
                    {
                        0: [
                            {
                                "background_two_theta_deg": 10.0,
                                "background_phi_deg": 1.0,
                            }
                        ],
                        1: [
                            {
                                "background_two_theta_deg": 101.0,
                                "background_phi_deg": 3.0,
                            }
                        ],
                    }
                )
            ),
            geometry_manual_pairs_for_index=lambda idx, *, pairs_by_background: list(
                pairs_by_background.get(int(idx), ())
            ),
            make_runtime_geometry_manual_projection_callbacks=_make_projection_callbacks,
        ),
        gui_structure_model=SimpleNamespace(
            parse_cif_num=lambda value: float(value),
            extract_occupancy_site_metadata=lambda *_args, **_kwargs: ([], {}),
            extract_atom_site_fractional_metadata=lambda *_args, **_kwargs: [],
            build_initial_structure_model_state=lambda **_kwargs: SimpleNamespace(
                miller=[],
                intensities=[],
            ),
            active_primary_cif_path=lambda *_args, **_kwargs: str(primary_cif_path),
            current_iodine_z=lambda *_args, **_kwargs: 0.0,
            rebuild_diffraction_inputs=lambda *args, **kwargs: None,
        ),
        AtomSiteOverrideState=lambda: SimpleNamespace(),
        SimulationRuntimeState=_SimulationRuntimeState,
    )

    monkeypatch.setattr(cli, "_load_cli_geometry_modules", lambda: geometry_modules)
    monkeypatch.setattr(
        cli, "build_headless_simulation_defaults", lambda *, out_path: simulation_defaults
    )
    monkeypatch.setattr(
        cli,
        "get_instrument_config",
        lambda: {
            "instrument": {
                "detector": {"image_size": 4, "pixel_size_m": 1.0, "intensity_threshold": 1.0},
                "beam": {"sigma_mosaic_fwhm_deg": 0.8, "gamma_mosaic_fwhm_deg": 0.7, "eta": 0.0},
                "sample_orientation": {
                    "theta_initial_deg": 0.0,
                    "cor_deg": 0.0,
                    "chi_deg": 0.0,
                    "psi_deg": 0.0,
                    "psi_z_deg": 0.0,
                    "zb": 0.0,
                    "zs": 0.0,
                    "width_m": 0.0,
                    "length_m": 0.0,
                    "depth_m": 0.0,
                },
                "debye_waller": {"x": 0.0, "y": 0.0},
                "hendricks_teller": {
                    "default_p": [0.5],
                    "default_w": [1.0],
                    "finite_stack": False,
                    "stack_layers": 1,
                    "max_miller_index": 1,
                    "include_rods": False,
                },
                "fit": {},
                "occupancies": {"default": [1.0]},
            }
        },
    )
    monkeypatch.setattr(
        cli, "_load_cif_snapshot", lambda path: ({}, {"_cell_length_a": 4.0, "_cell_length_c": 7.0})
    )
    monkeypatch.setattr(cli, "resolve_index_of_refraction", lambda *_args, **_kwargs: 1.0 + 0.0j)
    monkeypatch.setattr(
        cli,
        "_build_headless_geometry_mosaic_params",
        lambda **_kwargs: (
            {"solve_q_steps": 100, "solve_q_rel_tol": 1.0e-3, "solve_q_mode": 0},
            1,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_fitting_optimization",
        lambda: SimpleNamespace(
            fit_geometry_parameters=lambda *args, **kwargs: None,
            fit_mosaic_shape_parameters=lambda *args, **kwargs: None,
            simulate_and_compare_hkl=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_stacking_fault_module",
        lambda: SimpleNamespace(
            DEFAULT_PHI_L_DIVISOR=1.0,
            DEFAULT_PHASE_DELTA_EXPRESSION="0.0",
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_diffraction_tools_module",
        lambda: SimpleNamespace(detector_two_theta_max=lambda *args, **kwargs: 1.0),
    )
    monkeypatch.setattr(
        cli,
        "_load_simulation_modules",
        lambda: SimpleNamespace(
            diffraction=SimpleNamespace(
                OPTICS_MODE_FAST=0,
                OPTICS_MODE_EXACT=1,
                hit_tables_to_max_positions=lambda *args, **kwargs: [],
                process_peaks_parallel=lambda *args, **kwargs: [],
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_tools_module",
        lambda: SimpleNamespace(
            build_intensity_dataframes=lambda *args, **kwargs: None,
            inject_fractional_reflections=lambda *args, **kwargs: None,
            miller_generator=lambda *args, **kwargs: [],
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_shared_headless_geometry_fit",
        lambda: SimpleNamespace(
            _build_headless_geometry_fit_caked_view_payload=_payload,
            _load_exact_cake_portable_module=lambda: SimpleNamespace(
                raw_phi_to_gui_phi=lambda value: value
            ),
        ),
    )

    def _hydrate(payload, **_kwargs):
        marker = int(float(np.asarray(payload["background"], dtype=np.float64)[0, 0]))
        hydrated = dict(payload)
        hydrated["transform_bundle"] = exact_bundles[marker]
        return hydrated

    monkeypatch.setattr(
        geometry_fit,
        "_geometry_fit_hydrate_exact_caked_payload",
        _hydrate,
    )

    def _prepare_runtime_geometry_fit_run(*, bindings, **_kwargs):
        dataset_bindings = bindings.manual_dataset_bindings
        for background_idx in (0, 1):
            cached = dataset_bindings.geometry_manual_caked_view_for_index(background_idx)
            assert cached["transform_bundle"] is exact_bundles[background_idx]
            cached["transform_bundle"] = "json-safe-bundle"
        rows = [
            {"background_index": 0, "row_id": "first"},
            {"background_index": 1, "row_id": "second"},
            {"background_index": 0, "row_id": "third"},
        ]
        for row in rows:
            projected_rows.extend(
                dataset_bindings.geometry_manual_project_peaks_for_background_view(
                    int(row["background_index"]),
                    [row],
                )
            )
        raise _StopAfterProjection()

    monkeypatch.setattr(
        geometry_fit,
        "prepare_runtime_geometry_fit_run",
        _prepare_runtime_geometry_fit_run,
    )

    with pytest.raises(_StopAfterProjection):
        cli.run_headless_geometry_fit(
            {
                "state": {
                    "files": {
                        "background_files": ["bg0.osc", "bg1.osc"],
                        "primary_cif_path": str(primary_cif_path),
                    },
                    "geometry": {"manual_pairs": []},
                }
            },
            source_path=tmp_path / "state.json",
            output_dir=tmp_path,
            run_mosaic_shape_fit=True,
        )

    assert [(row["row_id"], row["projected_background"]) for row in projected_rows] == [
        ("first", 0),
        ("second", 1),
        ("third", 0),
    ]
    assert captured_bundles == [exact_bundles[0], exact_bundles[1], exact_bundles[0]]
    assert [row["used_exact_bundle"] for row in projected_rows] == [
        exact_bundles[0],
        exact_bundles[1],
        exact_bundles[0],
    ]


@pytest.mark.parametrize(
    "bad_image",
    [
        np.asarray(1.0, dtype=np.float64),
        np.ones((8,), dtype=np.float64),
        np.ones((4, 4, 1), dtype=np.float64),
    ],
)
def test_headless_caked_manual_payload_rejects_malformed_images(
    monkeypatch,
    bad_image,
) -> None:
    headless = importlib.import_module("ra_sim.headless_geometry_fit")

    monkeypatch.setattr(
        headless.gui_geometry_fit,
        "build_geometry_fit_exact_caked_view_payload",
        lambda *_args, **_kwargs: pytest.fail("malformed images must fail before caked build"),
    )

    payload = headless._build_headless_geometry_fit_caked_view_payload(
        bad_image,
        params={
            "center": [4.0, 4.0],
            "center_x": 4.0,
            "center_y": 4.0,
            "corto_detector": 0.5,
            "lambda": 1.54,
        },
        pixel_size_m=1.0e-4,
        npt_rad=8,
        npt_azim=8,
    )

    assert payload is None


def test_manual_caked_objective_stays_in_fit_space_and_improves(monkeypatch) -> None:
    opt = importlib.import_module("ra_sim.fitting.optimization")

    def fake_process(*args, **_kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
        rows_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
        gamma = float(local_params.get("gamma", 0.0))
        return {
            "two_theta_deg": cols_arr + gamma,
            "phi_deg": rows_arr + gamma,
            "fit_space_source": "dataset_fit_space_projector",
            "input_frame": str(input_frame),
            "fit_space_projector_kind": "exact_caked_bundle",
            "cake_bundle_signature": f"sig-{anchor_kind}-{gamma:.3f}",
            "fit_space_local_params_signature": f"lp-{gamma:.3f}",
            "valid": True,
            "invalid_reason": None,
            "native_frame_conversion_source": f"test-{input_frame}",
            "native_frame_conversion_count": 0 if input_frame == "native_detector" else 1,
            "native_cols": cols_arr,
            "native_rows": rows_arr,
            "caked_projection_source": "fit_space_projector_native_detector",
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(
        opt,
        "_detector_pixels_to_fit_space",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("manual caked objective must not use detector fallback")
        ),
    )

    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[
            {
                "pair_id": "trace-0",
                "label": "peak-0",
                "hkl": (1, 0, 0),
                "overlay_match_index": 0,
                "source_reflection_index": 0,
                "source_table_index": 0,
                "source_row_index": 0,
                "source_branch_index": 0,
                "source_peak_index": 0,
                "fit_source_identity_only": True,
                "sim_col": 12.0,
                "sim_row": 12.0,
                "background_two_theta_deg": 10.0,
                "background_phi_deg": 10.0,
                "fit_space_anchor_override": True,
            }
        ],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        fit_space_projector=projector,
        fit_space_projector_kind="exact_caked_bundle",
    )
    base_params = {
        "gamma": 0.0,
        "Gamma": 0.0,
        "theta_initial": 0.0,
        "cor_angle": 0.0,
        "zs": 0.0,
        "zb": 0.0,
        "chi": 0.0,
        "a": 4.0,
        "c": 7.0,
        "center": [16.0, 16.0],
        "pixel_size": 1.0,
        "corto_detector": 100.0,
        "lambda": 1.54,
        "n2": 1.0,
        "psi": 0.0,
        "psi_z": 0.0,
        "debye_x": 0.0,
        "debye_y": 0.0,
        "optics_mode": 0,
        "mosaic_params": {
            "beam_x_array": np.zeros(1, dtype=np.float64),
            "beam_y_array": np.zeros(1, dtype=np.float64),
            "theta_array": np.zeros(1, dtype=np.float64),
            "phi_array": np.zeros(1, dtype=np.float64),
            "sigma_mosaic_deg": 0.2,
            "gamma_mosaic_deg": 0.1,
            "eta": 0.05,
            "wavelength_array": np.ones(1, dtype=np.float64),
        },
    }
    improved_params = dict(base_params, gamma=-2.0)

    def _evaluate(params):
        residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
            params,
            dataset_ctx,
            image_size=32,
            missing_pair_penalty_deg=5.0,
            theta_value=0.0,
            collect_diagnostics=True,
        )
        return residual, diagnostics, summary

    initial_residual, initial_diag, initial_summary = _evaluate(base_params)
    final_residual, final_diag, final_summary = _evaluate(improved_params)

    assert initial_diag[0]["pair_id"] == final_diag[0]["pair_id"] == "trace-0"
    assert initial_diag[0]["measured_fit_space_source"] == "cached_fit_space_anchor"
    assert final_diag[0]["measured_fit_space_source"] == "cached_fit_space_anchor"
    assert final_diag[0]["simulated_fit_space_source"] == "dataset_fit_space_projector"
    assert final_diag[0]["fit_space_projector_kind"] == "exact_caked_bundle"
    assert final_diag[0]["measured_detector_input_frame"] == "explicit_override"
    assert final_summary["manual_caked_residual_row_count"] == 1
    assert final_summary["dataset_fit_space_projector_row_count"] == 1
    assert final_summary["cached_fit_space_anchor_row_count"] == 1
    assert final_summary["analytic_detector_fit_space_row_count"] == 0
    assert final_summary["invalid_dataset_fit_space_projector_row_count"] == 0
    assert np.linalg.norm(final_residual) <= np.linalg.norm(initial_residual)
    assert np.linalg.norm(final_residual) == pytest.approx(0.0)


def test_geometry_fit_trace_keeps_placement_error_out_of_optimizer_residual() -> None:
    geometry_fit = importlib.import_module("ra_sim.gui.geometry_fit")

    assert (
        geometry_fit._geometry_fit_trace_optimizer_residual_px({"placement_error_px": 14.45})
        is None
    )
    assert geometry_fit._geometry_fit_trace_optimizer_residual_px(
        {"weighted_dx_px": 3.0, "weighted_dy_px": 4.0}
    ) == pytest.approx(5.0)

    record = geometry_fit._geometry_fit_trace_pair_record(
        phase="preflight",
        dataset_info={"dataset_index": 0, "label": "trace.osc"},
        entry={
            "pair_id": "trace-0",
            "overlay_match_index": 0,
            "placement_error_px": 14.45,
            "measured_x": 1083.8179,
            "measured_y": 1083.2697,
            "simulated_x": 1083.2697,
            "simulated_y": 1915.1821,
        },
        fit_run_id="fit-1",
    )

    assert record["optimizer_residual_px"] is None
    assert record["placement_error_px"] == pytest.approx(14.45)


def test_runtime_logged_cache_loaders_disabled_skip_disk_lookup(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(
        runtime_session,
        "intersection_cache_logging_enabled",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache_metadata",
        lambda: pytest.fail("disabled logged cache should not scan disk metadata"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache",
        lambda: pytest.fail("disabled logged cache should not scan disk tables"),
        raising=False,
    )

    metadata_loader, cache_loader = (
        runtime_session._geometry_fit_logged_intersection_cache_loaders()
    )

    assert metadata_loader is None
    assert cache_loader is None


def test_runtime_logged_cache_loaders_enabled_return_disk_loaders(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    metadata_loader = object()
    cache_loader = object()

    monkeypatch.setattr(
        runtime_session,
        "intersection_cache_logging_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache_metadata",
        metadata_loader,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache",
        cache_loader,
        raising=False,
    )

    assert runtime_session._geometry_fit_logged_intersection_cache_loaders() == (
        metadata_loader,
        cache_loader,
    )


def test_runtime_session_source_cache_caked_failure_does_not_hide_row_cache_success(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    dataset_calls: list[int] = []
    job = _make_geometry_fit_worker_job(runtime_session)

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_caking_ai_cache={},
            analysis_preview_bins=(4, 4),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(runtime_session, "DISPLAY_ROTATE_K", 0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_build_analysis_integrator",
        lambda _cfg: object(),
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_worker_caked_projection_view",
        lambda **_kwargs: {
            "radial_axis": np.asarray([1.0, 2.0], dtype=float),
            "azimuth_axis": np.asarray([3.0, 4.0], dtype=float),
            "transform_bundle": object(),
        },
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_fit_caked_roi_selection",
        lambda *_args, **_kwargs: {
            "enabled": False,
            "valid": False,
            "pixel_count": 0,
            "fraction": 0.0,
            "half_width_px": 0.0,
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cake boom")),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: pytest.fail("failed cake should not build payload"),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_manual_fit_dataset",
        lambda background_index, **_kwargs: dataset_calls.append(int(background_index)) or {},
    )

    def _prepare_geometry_fit_run(**kwargs):
        kwargs["build_dataset"](
            0,
            theta_base=0.0,
            base_fit_params={"theta_initial": 0.0},
            orientation_cfg={},
            stage_callback=kwargs.get("stage_callback"),
        )
        return runtime_session.gui_geometry_fit.GeometryFitPreparationResult()

    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "prepare_geometry_fit_run",
        _prepare_geometry_fit_run,
    )

    runtime_session._run_async_geometry_fit_worker_job(job)
    events = _drain_geometry_fit_worker_events(job["event_queue"])
    kinds = [str(event.get("kind")) for event in events]

    assert dataset_calls == [0]
    assert "source_cache_rows_ready" in kinds
    assert "source_cache_build_ready" in kinds
    assert "source_cache_caked_view_failed" in kinds
    assert kinds.index("source_cache_rows_ready") < kinds.index("source_cache_build_ready")
    assert kinds.index("source_cache_build_ready") < kinds.index("source_cache_caked_view_failed")


def test_runtime_session_source_cache_caked_timeout_does_not_hide_row_cache_success(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    dataset_calls: list[int] = []
    job = _make_geometry_fit_worker_job(runtime_session)
    perf_lock = threading.Lock()
    perf_state = {"value": 0.0}

    def _fake_perf_counter() -> float:
        with perf_lock:
            perf_state["value"] += 0.25
            return float(perf_state["value"])

    monkeypatch.setattr(runtime_session, "perf_counter", _fake_perf_counter)
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_caking_ai_cache={},
            analysis_preview_bins=(4, 4),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(runtime_session, "DISPLAY_ROTATE_K", 0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_build_analysis_integrator",
        lambda _cfg: object(),
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_worker_caked_projection_view",
        lambda **_kwargs: {
            "radial_axis": np.asarray([1.0, 2.0], dtype=float),
            "azimuth_axis": np.asarray([3.0, 4.0], dtype=float),
            "transform_bundle": object(),
        },
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_fit_caked_roi_selection",
        lambda *_args, **_kwargs: {
            "enabled": False,
            "valid": False,
            "pixel_count": 0,
            "fraction": 0.0,
            "half_width_px": 0.0,
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda *_args, **_kwargs: threading.Event().wait(1.0) or object(),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: {
            "image": np.ones((2, 2), dtype=float),
            "radial": np.asarray([1.0, 2.0], dtype=float),
            "azimuth": np.asarray([3.0, 4.0], dtype=float),
            "raw_azimuth_axis": np.asarray([3.0, 4.0], dtype=float),
            "raw_to_gui_row_permutation": np.asarray([0, 1], dtype=np.int32),
            "transform_bundle": object(),
            "detector_shape": (4, 4),
        },
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_manual_fit_dataset",
        lambda background_index, **_kwargs: dataset_calls.append(int(background_index)) or {},
    )

    def _prepare_geometry_fit_run(**kwargs):
        kwargs["build_dataset"](
            0,
            theta_base=0.0,
            base_fit_params={"theta_initial": 0.0},
            orientation_cfg={},
            stage_callback=kwargs.get("stage_callback"),
        )
        return runtime_session.gui_geometry_fit.GeometryFitPreparationResult()

    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "prepare_geometry_fit_run",
        _prepare_geometry_fit_run,
    )

    runtime_session._run_async_geometry_fit_worker_job(job)
    threading.Event().wait(0.25)
    events = _drain_geometry_fit_worker_events(job["event_queue"])
    kinds = [str(event.get("kind")) for event in events]

    assert dataset_calls == [0]
    assert "source_cache_build_ready" in kinds
    assert "source_cache_caked_view_timeout" in kinds
    assert kinds.index("source_cache_build_ready") < kinds.index("source_cache_caked_view_timeout")


def test_worker_caked_manual_prepare_fails_closed_before_dataset_build(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _make_geometry_fit_worker_job(runtime_session)

    job["var_names"] = ["zb"]
    job["projection_view_mode"] = "caked"
    job["projection_payload_by_background"] = {}
    job["caked_views_by_background"] = {}
    job["pick_uses_caked_space"] = True
    job["manual_fit_space_by_background"] = {0: "caked"}

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_caking_ai_cache={},
            analysis_preview_bins=(4, 4),
            ai_cache={},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(runtime_session, "DISPLAY_ROTATE_K", 0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_build_analysis_integrator",
        lambda _cfg: object(),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_fit_caked_roi_selection",
        lambda *_args, **_kwargs: {
            "enabled": False,
            "valid": False,
            "pixel_count": 0,
            "fraction": 0.0,
            "half_width_px": 0.0,
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cake boom")),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_manual_fit_dataset",
        lambda *_args, **_kwargs: pytest.fail(
            "caked worker must fail before dataset build when exact payload is missing"
        ),
    )

    action_result = runtime_session._run_async_geometry_fit_worker_job(job)
    events = _drain_geometry_fit_worker_events(job["event_queue"])
    kinds = [str(event.get("kind")) for event in events]

    assert action_result.error_text is not None
    assert "exact caked fit-space projector could not be prepared" in action_result.error_text
    assert "exact caked projector unavailable for background 1" in action_result.error_text
    assert "source_cache_caked_view_failed" in kinds
    assert "preflight_failure" in kinds


def test_worker_caked_manual_rejects_axes_only_payload_before_dataset_build(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _make_geometry_fit_worker_job(runtime_session)

    job["var_names"] = ["zb"]
    job["projection_view_mode"] = "caked"
    job["projection_payload_by_background"] = {}
    job["caked_views_by_background"] = {}
    job["pick_uses_caked_space"] = True
    job["manual_fit_space_by_background"] = {0: "caked"}

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_caking_ai_cache={},
            analysis_preview_bins=(4, 4),
            ai_cache={},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(runtime_session, "DISPLAY_ROTATE_K", 0, raising=False)
    monkeypatch.setattr(runtime_session, "_build_analysis_integrator", lambda _cfg: object())
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_fit_caked_roi_selection",
        lambda *_args, **_kwargs: {
            "enabled": False,
            "valid": False,
            "pixel_count": 0,
            "fraction": 0.0,
            "half_width_px": 0.0,
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runtime_session, "caking", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: {
            "image": np.ones((2, 2), dtype=float),
            "radial": np.asarray([1.0, 2.0], dtype=float),
            "azimuth": np.asarray([3.0, 4.0], dtype=float),
            "raw_azimuth_axis": np.asarray([3.0, 4.0], dtype=float),
            "raw_to_gui_row_permutation": np.asarray([0, 1], dtype=np.int32),
            "transform_bundle": None,
            "detector_shape": (4, 4),
        },
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_manual_fit_dataset",
        lambda *_args, **_kwargs: pytest.fail(
            "axes-only caked payload must fail before dataset build"
        ),
    )

    action_result = runtime_session._run_async_geometry_fit_worker_job(job)

    assert action_result.error_text is not None
    assert "exact caked fit-space projector could not be prepared" in action_result.error_text
    assert "exact caked projector unavailable for background 1" in action_result.error_text


def test_runtime_session_logged_cache_params_mismatch_rejects_before_heavy_hit_table_load(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    job = _make_geometry_fit_worker_job(runtime_session)
    job["live_rows_by_background"] = {0: []}
    dataset_calls: list[int] = []

    monkeypatch.setattr(
        runtime_session,
        "intersection_cache_logging_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(runtime_session, "DISPLAY_ROTATE_K", 0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_build_analysis_integrator",
        lambda _cfg: object(),
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_worker_caked_projection_view",
        lambda **_kwargs: {
            "radial_axis": np.asarray([1.0, 2.0], dtype=float),
            "azimuth_axis": np.asarray([3.0, 4.0], dtype=float),
            "transform_bundle": object(),
        },
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_fit_caked_roi_selection",
        lambda *_args, **_kwargs: {
            "enabled": False,
            "valid": False,
            "pixel_count": 0,
            "fraction": 0.0,
            "half_width_px": 0.0,
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: {
            "image": np.ones((2, 2), dtype=float),
            "radial": np.asarray([1.0, 2.0], dtype=float),
            "azimuth": np.asarray([3.0, 4.0], dtype=float),
            "raw_azimuth_axis": np.asarray([3.0, 4.0], dtype=float),
            "raw_to_gui_row_permutation": np.asarray([0, 1], dtype=np.int32),
            "transform_bundle": object(),
            "detector_shape": (4, 4),
        },
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_manual_fit_dataset",
        lambda background_index, **_kwargs: dataset_calls.append(int(background_index)) or {},
    )

    def _prepare_geometry_fit_run(**kwargs):
        kwargs["build_dataset"](
            0,
            theta_base=0.0,
            base_fit_params={"theta_initial": 0.0},
            orientation_cfg={},
            stage_callback=kwargs.get("stage_callback"),
        )
        return runtime_session.gui_geometry_fit.GeometryFitPreparationResult()

    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "prepare_geometry_fit_run",
        _prepare_geometry_fit_run,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache_metadata",
        lambda: {
            "signature_digest": "logged-digest",
            "av": 9.999,
            "cv": 28.64,
            "wavelength_center": 1.54,
            "theta_center": 0.0,
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache",
        lambda: pytest.fail("params mismatch should reject before heavy hit-table load"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "is_intersection_cache_table",
        lambda _table: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_copy_hit_tables",
        lambda tables: list(tables or ()),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_filter_hit_tables_for_required_branch_groups",
        lambda hit_tables, required_branch_group_keys=None: (
            list(hit_tables or ()),
            {
                "total_hit_tables_available": int(len(hit_tables or ())),
                "hit_tables_considered_for_rebinding": int(len(hit_tables or ())),
                "hit_tables_expanded_for_rebinding": int(len(hit_tables or ())),
            },
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: [object()],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_build_source_rows_from_hit_tables",
        lambda *_args, **_kwargs: (
            [_geometry_fit_worker_live_row()],
            [],
            [],
            [],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda idx: (
            np.full((4, 4), float(idx), dtype=np.float64),
            np.full((4, 4), float(idx), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)

    runtime_session._run_async_geometry_fit_worker_job(job)
    events = _drain_geometry_fit_worker_events(job["event_queue"])
    miss_payload = next(
        dict(event.get("payload") or {})
        for event in events
        if str(event.get("kind")) == "source_cache_logged_intersection_cache_miss"
    )

    assert dataset_calls == []
    assert miss_payload["status"] == "params_mismatch"
    assert miss_payload["expected_signature_digest"]
    assert miss_payload["actual_signature_digest"] == "logged-digest"
    assert miss_payload["mismatch_reason"] == "params_mismatch"
    assert miss_payload["heavy_hit_table_load_attempted"] is False


def test_worker_prebuild_uses_background_specific_projection_signature() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _prebuild_background_cache_bundle_worker(")
    helper_end = source.index("def _emit_source_cache_caked_view_event(", helper_start)
    helper_source = source[helper_start:helper_end]

    assert (
        "projection_view_signature = _worker_projection_view_signature_for_background("
        in helper_source
    )
    assert "projection_view_signature=projection_view_signature" in helper_source


def test_worker_projection_signature_recomputes_detector_instead_of_reusing_stale_caked() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _worker_projection_view_signature_for_background(")
    helper_end = source.index("def _project_source_rows_for_background(", helper_start)
    helper_source = source[helper_start:helper_end]

    assert "stored_signature" not in helper_source
    assert "mode_override=normalized_mode" in helper_source
    assert "signature_map[background_idx]" in helper_source


def test_worker_prebuild_generates_caked_payload_for_noncurrent_backgrounds(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _make_geometry_fit_worker_job(runtime_session)
    resolve_calls: list[dict[str, object]] = []
    seen_projection_payloads: list[object] = []

    job["current_background_index"] = 0
    job["required_indices"] = [1]
    job["requested_signatures"] = {1: ("sig", 1)}
    job["requested_signature_summaries"] = {1: "sig-summary-1"}
    job["background_labels"] = {1: "bg1.osc"}
    job["manual_pairs_by_background"] = {1: [_geometry_fit_worker_required_pair()]}
    job["background_images"][1] = {
        "native": np.ones((4, 4), dtype=np.float64),
        "display": np.ones((4, 4), dtype=np.float64),
    }
    job["projection_view_mode"] = "caked"
    job["projection_payload_by_background"] = {}
    job["osc_files"] = ["bg0.osc", "bg1.osc"]

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_caking_ai_cache={},
            analysis_preview_bins=(4, 4),
            ai_cache={},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_build_analysis_integrator",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_resolve_targeted_caked_projection_payload",
        lambda background_index, **kwargs: (
            resolve_calls.append({"background_index": int(background_index), **dict(kwargs)})
            or {
                **_geometry_fit_worker_caked_payload(
                    runtime_session,
                    background_value=float(background_index),
                    radial_values=[101.0, 102.0],
                    azimuth_values=[3.0, 4.0],
                ),
                "payload_marker": int(background_index),
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_fit_caked_roi_selection",
        lambda *_args, **_kwargs: {
            "enabled": False,
            "valid": False,
            "pixel_count": 0,
            "fraction": 0.0,
            "half_width_px": 0.0,
        },
    )

    def _fake_rebuild_geometry_fit_source_rows(**kwargs):
        seen_projection_payloads.append(kwargs.get("projection_payload"))
        return runtime_session.gui_geometry_fit.GeometryFitSourceRowRebuildResult(
            background_index=int(kwargs["background_index"]),
            requested_signature=kwargs["requested_signature"],
            requested_signature_summary=kwargs["requested_signature_summary"],
            projected_rows=[],
            stored_rows=[dict(_geometry_fit_worker_live_row(), background_index=1)],
            rebuild_source="fresh_simulation",
            rebuild_attempts=["fresh_simulation"],
            diagnostics={
                "status": "fresh_simulation_ready",
                "projection_view_mode": "caked",
            },
        )

    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "rebuild_geometry_fit_source_rows",
        _fake_rebuild_geometry_fit_source_rows,
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "prepare_geometry_fit_run",
        lambda **_kwargs: runtime_session.gui_geometry_fit.GeometryFitPreparationResult(),
    )

    runtime_session._run_async_geometry_fit_worker_job(job)

    assert resolve_calls
    assert any(
        call["background_index"] == 1 and call["allow_generated_payload"] is True
        for call in resolve_calls
    )
    assert len(seen_projection_payloads) == 1
    assert isinstance(seen_projection_payloads[0], Mapping)
    assert seen_projection_payloads[0]["payload_marker"] == 1
    assert isinstance(job["projection_payload_by_background"], dict)
    assert job["projection_payload_by_background"][1]["payload_marker"] == 1


def test_worker_does_not_call_current_view_projector_for_noncurrent_background(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _make_geometry_fit_worker_job(runtime_session)
    projected_rows: list[dict[str, object]] = []

    job["current_background_index"] = 0
    job["required_indices"] = []
    job["background_images"][1] = {
        "native": np.ones((4, 4), dtype=np.float64),
        "display": np.ones((4, 4), dtype=np.float64),
    }
    job["projection_view_mode"] = "caked"
    job["projection_payload_by_background"] = {
        1: _geometry_fit_worker_caked_payload(
            runtime_session,
            background_value=1.0,
            radial_values=[101.0, 102.0],
            azimuth_values=[1.0, 2.0],
        )
    }
    job["osc_files"] = ["bg0.osc", "bg1.osc"]
    job["project_rows"] = lambda _rows: pytest.fail(
        "worker should not use live current-view projector for non-current background"
    )

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_caking_ai_cache={},
            analysis_preview_bins=(4, 4),
            ai_cache={},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(
        runtime_session, "_build_analysis_integrator", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_fit_caked_roi_selection",
        lambda *_args, **_kwargs: {
            "enabled": False,
            "valid": False,
            "pixel_count": 0,
            "fraction": 0.0,
            "half_width_px": 0.0,
        },
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda rows: [
                dict(
                    entry,
                    projected_background=int(
                        np.asarray(kwargs["current_background_native"]())[0, 0]
                    ),
                )
                for entry in rows or ()
                if isinstance(entry, Mapping)
            ]
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_manual_fit_dataset",
        lambda _background_index, *, manual_dataset_bindings, **_kwargs: (
            projected_rows.extend(
                manual_dataset_bindings.geometry_manual_project_peaks_to_current_view(
                    [dict(_geometry_fit_worker_live_row(), background_index=1)]
                )
            )
            or {}
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "prepare_geometry_fit_run",
        lambda **kwargs: (
            kwargs["build_dataset"](
                0,
                theta_base=0.0,
                base_fit_params={"theta_initial": 0.0},
                orientation_cfg={},
                stage_callback=kwargs.get("stage_callback"),
            )
            or runtime_session.gui_geometry_fit.GeometryFitPreparationResult()
        ),
    )

    runtime_session._run_async_geometry_fit_worker_job(job)

    assert projected_rows[0]["projected_background"] == 1


def test_worker_noncurrent_background_without_payload_fails_closed(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _make_geometry_fit_worker_job(runtime_session)
    projected_rows: list[dict[str, object]] = []
    resolve_calls: list[dict[str, object]] = []

    job["current_background_index"] = 0
    job["required_indices"] = []
    job["background_images"][1] = {
        "native": np.ones((4, 4), dtype=np.float64),
        "display": np.ones((4, 4), dtype=np.float64),
    }
    job["projection_view_mode"] = "caked"
    job["projection_payload_by_background"] = {}
    job["caked_views_by_background"] = {}
    job["osc_files"] = ["bg0.osc", "bg1.osc"]
    job["project_rows"] = lambda _rows: pytest.fail(
        "worker should not use live current-view projector for non-current background"
    )

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_caking_ai_cache={},
            analysis_preview_bins=(4, 4),
            ai_cache={},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_build_analysis_integrator",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_resolve_targeted_caked_projection_payload",
        lambda background_index, **kwargs: (
            resolve_calls.append({"background_index": int(background_index), **dict(kwargs)})
            or None
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_fit_caked_roi_selection",
        lambda *_args, **_kwargs: {
            "enabled": False,
            "valid": False,
            "pixel_count": 0,
            "fraction": 0.0,
            "half_width_px": 0.0,
        },
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **_kwargs: pytest.fail(
            "worker should not build non-current caked projector without payload"
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_manual_fit_dataset",
        lambda _background_index, *, manual_dataset_bindings, **_kwargs: (
            projected_rows.extend(
                manual_dataset_bindings.geometry_manual_project_peaks_to_current_view(
                    [dict(_geometry_fit_worker_live_row(), background_index=1)]
                )
            )
            or {}
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "prepare_geometry_fit_run",
        lambda **kwargs: (
            kwargs["build_dataset"](
                0,
                theta_base=0.0,
                base_fit_params={"theta_initial": 0.0},
                orientation_cfg={},
                stage_callback=kwargs.get("stage_callback"),
            )
            or runtime_session.gui_geometry_fit.GeometryFitPreparationResult()
        ),
    )

    action_result = runtime_session._run_async_geometry_fit_worker_job(job)

    assert projected_rows == []
    assert action_result.error_text is not None
    assert "exact caked projector unavailable for background 2" in action_result.error_text
    assert any(call["background_index"] == 1 for call in resolve_calls)
    assert all(
        call["allow_generated_payload"] is True
        for call in resolve_calls
        if int(call["background_index"]) == 1
    )


def test_worker_uses_background_specific_caked_payload_for_projected_rows(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _make_geometry_fit_worker_job(runtime_session)
    projected_rows: list[dict[str, object]] = []

    job["current_background_index"] = 0
    job["required_indices"] = []
    job["background_images"][1] = {
        "native": np.ones((4, 4), dtype=np.float64),
        "display": np.ones((4, 4), dtype=np.float64),
    }
    job["projection_view_mode"] = "caked"
    job["projection_payload_by_background"] = {
        0: _geometry_fit_worker_caked_payload(
            runtime_session,
            background_value=0.0,
            radial_values=[11.0, 12.0],
            azimuth_values=[1.0, 2.0],
        ),
        1: _geometry_fit_worker_caked_payload(
            runtime_session,
            background_value=1.0,
            radial_values=[101.0, 102.0],
            azimuth_values=[3.0, 4.0],
        ),
    }
    job["osc_files"] = ["bg0.osc", "bg1.osc"]

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_caking_ai_cache={},
            analysis_preview_bins=(4, 4),
            ai_cache={},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(
        runtime_session, "_build_analysis_integrator", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda rows: [
                dict(
                    entry,
                    projected_payload_marker=float(
                        np.asarray(kwargs["last_caked_radial_values"](), dtype=float)[0]
                    ),
                )
                for entry in rows or ()
                if isinstance(entry, Mapping)
            ]
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_manual_fit_dataset",
        lambda _background_index, *, manual_dataset_bindings, **_kwargs: (
            projected_rows.extend(
                manual_dataset_bindings.geometry_manual_project_peaks_to_current_view(
                    [dict(_geometry_fit_worker_live_row(), background_index=1)]
                )
            )
            or {}
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "prepare_geometry_fit_run",
        lambda **kwargs: (
            kwargs["build_dataset"](
                0,
                theta_base=0.0,
                base_fit_params={"theta_initial": 0.0},
                orientation_cfg={},
                stage_callback=kwargs.get("stage_callback"),
            )
            or runtime_session.gui_geometry_fit.GeometryFitPreparationResult()
        ),
    )

    runtime_session._run_async_geometry_fit_worker_job(job)

    assert projected_rows[0]["projected_payload_marker"] == 101.0


def test_worker_caked_projection_hydrates_json_safe_payload_before_projection(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _make_geometry_fit_worker_job(runtime_session)
    projected_rows: list[dict[str, object]] = []
    captured_bundles: list[object] = []

    radial_axis = np.asarray([101.0, 102.0], dtype=np.float64)
    azimuth_axis = np.asarray([3.0, 4.0], dtype=np.float64)
    raw_azimuth_axis = np.asarray(
        runtime_session.gui_geometry_fit.gui_phi_to_raw_phi(azimuth_axis),
        dtype=np.float64,
    )
    exact_bundle = runtime_session.CakeTransformBundle(
        detector_shape=(4, 4),
        radial_deg=radial_axis,
        raw_azimuth_deg=raw_azimuth_axis,
        gui_azimuth_deg=azimuth_axis,
        lut=object(),
    )
    stale_payload = _geometry_fit_worker_caked_payload(
        runtime_session,
        background_value=1.0,
        radial_values=radial_axis,
        azimuth_values=azimuth_axis,
    )
    stale_payload["transform_bundle"] = "json-safe-bundle"

    job["current_background_index"] = 0
    job["required_indices"] = []
    job["background_images"][1] = {
        "native": np.ones((4, 4), dtype=np.float64),
        "display": np.ones((4, 4), dtype=np.float64),
    }
    job["projection_view_mode"] = "caked"
    job["projection_payload_by_background"] = {1: stale_payload}
    job["osc_files"] = ["bg0.osc", "bg1.osc"]

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_caking_ai_cache={},
            analysis_preview_bins=(4, 4),
            ai_cache={},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_build_analysis_integrator",
        lambda *_args, **_kwargs: None,
    )

    def _hydrate(payload, **_kwargs):
        hydrated = dict(payload)
        hydrated["transform_bundle"] = exact_bundle
        return hydrated

    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "_geometry_fit_hydrate_exact_caked_payload",
        _hydrate,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda rows: (
                captured_bundles.append(kwargs["caked_transform_bundle"]())
                or [
                    dict(
                        entry,
                        projected_with_exact_bundle=(
                            kwargs["caked_transform_bundle"]() is exact_bundle
                        ),
                    )
                    for entry in rows or ()
                    if isinstance(entry, Mapping)
                ]
            )
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_manual_fit_dataset",
        lambda _background_index, *, manual_dataset_bindings, **_kwargs: (
            projected_rows.extend(
                manual_dataset_bindings.geometry_manual_project_peaks_to_current_view(
                    [dict(_geometry_fit_worker_live_row(), background_index=1)]
                )
            )
            or {}
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "prepare_geometry_fit_run",
        lambda **kwargs: (
            kwargs["build_dataset"](
                0,
                theta_base=0.0,
                base_fit_params={"theta_initial": 0.0},
                orientation_cfg={},
                stage_callback=kwargs.get("stage_callback"),
            )
            or runtime_session.gui_geometry_fit.GeometryFitPreparationResult()
        ),
    )

    runtime_session._run_async_geometry_fit_worker_job(job)

    assert captured_bundles == [exact_bundle]
    assert projected_rows[0]["projected_with_exact_bundle"] is True
    assert job["projection_payload_by_background"][1]["transform_bundle"] is exact_bundle


def test_worker_manual_dataset_projection_splits_mixed_background_rows(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _make_geometry_fit_worker_job(runtime_session)
    projected_rows: list[dict[str, object]] = []

    job["required_indices"] = []
    job["projection_view_mode"] = "caked"
    job["background_images"][0] = {
        "native": np.zeros((4, 4), dtype=np.float64),
        "display": np.zeros((4, 4), dtype=np.float64),
    }
    job["background_images"][1] = {
        "native": np.ones((4, 4), dtype=np.float64),
        "display": np.ones((4, 4), dtype=np.float64),
    }
    job["projection_payload_by_background"] = {
        0: _geometry_fit_worker_caked_payload(
            runtime_session,
            background_value=0.0,
            radial_values=[11.0, 12.0],
            azimuth_values=[1.0, 2.0],
        ),
        1: _geometry_fit_worker_caked_payload(
            runtime_session,
            background_value=1.0,
            radial_values=[101.0, 102.0],
            azimuth_values=[3.0, 4.0],
        ),
    }
    job["osc_files"] = ["bg0.osc", "bg1.osc"]

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_caking_ai_cache={},
            analysis_preview_bins=(4, 4),
            ai_cache={},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(
        runtime_session, "_build_analysis_integrator", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_fit_caked_roi_selection",
        lambda *_args, **_kwargs: {
            "enabled": False,
            "valid": False,
            "pixel_count": 0,
            "fraction": 0.0,
            "half_width_px": 0.0,
        },
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda rows: [
                dict(
                    entry,
                    projected_background=int(
                        np.asarray(kwargs["current_background_native"]())[0, 0]
                    ),
                )
                for entry in rows or ()
                if isinstance(entry, dict)
            ]
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_manual_fit_dataset",
        lambda _background_index, *, manual_dataset_bindings, **_kwargs: (
            projected_rows.extend(
                manual_dataset_bindings.geometry_manual_project_peaks_to_current_view(
                    [
                        dict(_geometry_fit_worker_live_row(), background_index=1, row_id="first"),
                        dict(_geometry_fit_worker_live_row(), background_index=0, row_id="second"),
                        dict(_geometry_fit_worker_live_row(), background_index=1, row_id="third"),
                    ]
                )
            )
            or {}
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "prepare_geometry_fit_run",
        lambda **kwargs: (
            kwargs["build_dataset"](
                0,
                theta_base=0.0,
                base_fit_params={"theta_initial": 0.0},
                orientation_cfg={},
                stage_callback=kwargs.get("stage_callback"),
            )
            or runtime_session.gui_geometry_fit.GeometryFitPreparationResult()
        ),
    )

    runtime_session._run_async_geometry_fit_worker_job(job)

    assert [
        (str(row.get("row_id")), int(row.get("projected_background", -1))) for row in projected_rows
    ] == [("first", 1), ("second", 0), ("third", 1)]


def test_noncurrent_caked_projection_without_payload_fails_closed(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    resolve_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(ai_cache={}),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda idx: (
            np.full((4, 4), float(idx), dtype=np.float64),
            np.full((4, 4), float(idx), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_resolve_targeted_caked_projection_payload",
        lambda background_index, **kwargs: (
            resolve_calls.append({"background_index": int(background_index), **dict(kwargs)})
            or None
        ),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="exact caked projector unavailable for background 2"):
        runtime_session._geometry_manual_project_peaks_for_background(
            1,
            [dict(_geometry_fit_worker_live_row(), background_index=1, row_id="row")],
            mode_override="caked",
        )

    assert resolve_calls
    assert resolve_calls[0]["background_index"] == 1
    assert resolve_calls[0]["allow_generated_payload"] is True


def test_current_caked_projection_skips_legacy_projector_and_hydrates_exact_bundle(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
def test_manual_pick_caked_projection_missing_payload_fails_open(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    resolve_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(ai_cache={}),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda idx: (
            np.full((4, 4), float(idx), dtype=np.float64),
            np.full((4, 4), float(idx), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_resolve_targeted_caked_projection_payload",
        lambda background_index, **kwargs: (
            resolve_calls.append({"background_index": int(background_index), **dict(kwargs)})
            or None
        ),
        raising=False,
    )

    rows = runtime_session._geometry_manual_project_peaks_for_background(
        1,
        [dict(_geometry_fit_worker_live_row(), background_index=1, row_id="row")],
        mode_override="caked",
        strict_caked_projection=False,
    )

    assert rows == []
    assert resolve_calls
    assert resolve_calls[0]["background_index"] == 1
    assert resolve_calls[0]["allow_generated_payload"] is True


def test_manual_pick_detector_override_bypasses_current_caked_projector(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    current_view_calls: list[object] = []
    detector_callback_modes: list[object] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(ai_cache={}, profile_cache={}),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        lambda rows: current_view_calls.append(list(rows or ())) or [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda _idx: (
            np.ones((4, 4), dtype=np.float64),
            np.ones((4, 4), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda rows: (
                detector_callback_modes.append(bool(kwargs["caked_view_enabled"]()))
                or [
                    dict(
                        entry,
                        display_col=float(entry.get("sim_col", entry.get("display_col", 0.0))),
                        display_row=float(entry.get("sim_row", entry.get("display_row", 0.0))),
                        display_frame="detector_display",
                    )
                    for entry in rows or ()
                    if isinstance(entry, Mapping)
                ]
            )
        ),
    )

    rows = runtime_session._geometry_manual_project_peaks_for_background(
        0,
        [dict(_geometry_fit_worker_live_row(), background_index=0, row_id="row")],
        mode_override="detector",
        strict_caked_projection=False,
    )

    assert current_view_calls == []
    assert detector_callback_modes == [False]
    assert len(rows) == 1
    assert rows[0]["row_id"] == "row"
    assert rows[0]["display_frame"] == "detector_display"
    assert rows[0]["background_index"] == 0


    payload = _geometry_fit_worker_caked_payload(
        runtime_session,
        background_value=1.0,
        radial_values=[101.0, 102.0],
        azimuth_values=[3.0, 4.0],
    )
    exact_bundle = payload["transform_bundle"]
    stale_payload = dict(payload, transform_bundle="json-safe-bundle")
    captured_bundles: list[object] = []
    live_bundles: list[object] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        lambda _rows: pytest.fail("caked current-background path must not use legacy projector"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda _idx: (
            np.ones((4, 4), dtype=np.float64),
            np.ones((4, 4), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_resolve_targeted_caked_projection_payload",
        lambda *_args, **_kwargs: dict(stale_payload),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "_geometry_fit_hydrate_exact_caked_payload",
        lambda payload, **_kwargs: dict(payload, transform_bundle=exact_bundle),
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_live_caked_transform_bundle",
        lambda bundle: live_bundles.append(bundle),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(ai_cache={}, profile_cache={}),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda rows: (
                captured_bundles.append(kwargs["caked_transform_bundle"]())
                or [
                    dict(entry, projected_with_exact_bundle=True)
                    for entry in rows or ()
                    if isinstance(entry, Mapping)
                ]
            )
        ),
    )

    rows = runtime_session._geometry_manual_project_peaks_for_background(
        0,
        [dict(_geometry_fit_worker_live_row(), row_id="row")],
        mode_override="caked",
    )

    assert captured_bundles == [exact_bundle]
    assert live_bundles == [exact_bundle]
    assert rows[0]["projected_with_exact_bundle"] is True


def test_current_caked_projection_error_fails_closed(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    payload = _geometry_fit_worker_caked_payload(
        runtime_session,
        background_value=1.0,
        radial_values=[101.0, 102.0],
        azimuth_values=[3.0, 4.0],
    )

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        lambda _rows: pytest.fail("caked current-background path must not use legacy projector"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda _idx: (
            np.ones((4, 4), dtype=np.float64),
            np.ones((4, 4), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_resolve_targeted_caked_projection_payload",
        lambda *_args, **_kwargs: dict(payload),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **_kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda _rows: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(ai_cache={}, profile_cache={}),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)

    with pytest.raises(RuntimeError, match="boom"):
        runtime_session._geometry_manual_project_peaks_for_background(
            0,
            [dict(_geometry_fit_worker_live_row(), row_id="row")],
            mode_override="caked",
        )


def test_noncurrent_caked_projection_error_fails_closed(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(ai_cache={}),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda idx: (
            np.full((4, 4), float(idx), dtype=np.float64),
            np.full((4, 4), float(idx), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_resolve_targeted_caked_projection_payload",
        lambda *_args, **_kwargs: _geometry_fit_worker_caked_payload(
            runtime_session,
            background_value=1.0,
            radial_values=[101.0, 102.0],
            azimuth_values=[3.0, 4.0],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **_kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda _rows: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
    )

    with pytest.raises(RuntimeError, match="boom"):
        runtime_session._geometry_manual_project_peaks_for_background(
            1,
            [dict(_geometry_fit_worker_live_row(), background_index=1, row_id="row")],
            mode_override="caked",
        )


def test_noncurrent_caked_sync_rebuild_injects_generated_projection_payload(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    resolve_calls: list[dict[str, object]] = []
    seen_projection_payloads: list[object] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0, osc_files=["bg0.osc", "bg1.osc"]),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_app_shell_view_mode",
        lambda: "caked",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda idx: (
            np.full((4, 4), float(idx), dtype=np.float64),
            np.full((4, 4), float(idx), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_resolve_targeted_caked_projection_payload",
        lambda background_index, **kwargs: (
            resolve_calls.append({"background_index": int(background_index), **dict(kwargs)})
            or {
                "payload_marker": int(background_index),
                "detector_shape": (4, 4),
                "radial_axis": np.asarray([101.0, 102.0], dtype=np.float64),
                "azimuth_axis": np.asarray([3.0, 4.0], dtype=np.float64),
                "raw_azimuth_axis": np.asarray([3.0, 4.0], dtype=np.float64),
                "raw_to_gui_row_permutation": np.asarray([0, 1], dtype=np.int32),
                "transform_bundle": object(),
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache_metadata",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache",
        lambda: pytest.fail("sync rebuild test should skip heavy logged cache"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: [object()],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_build_source_rows_from_hit_tables",
        lambda *_args, **_kwargs: (
            [dict(_geometry_fit_worker_live_row(), background_index=1)],
            [],
            [],
            [],
        ),
        raising=False,
    )

    def _fake_rebuild_geometry_fit_source_rows(**kwargs):
        seen_projection_payloads.append(kwargs.get("projection_payload"))
        return runtime_session.gui_geometry_fit.GeometryFitSourceRowRebuildResult(
            background_index=int(kwargs["background_index"]),
            requested_signature=kwargs["requested_signature"],
            requested_signature_summary=kwargs["requested_signature_summary"],
            projected_rows=[],
            stored_rows=[dict(_geometry_fit_worker_live_row(), background_index=1)],
            rebuild_source="fresh_simulation",
            rebuild_attempts=["fresh_simulation"],
            diagnostics={
                "status": "fresh_simulation_ready",
                "projection_view_mode": "caked",
            },
        )

    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "rebuild_geometry_fit_source_rows",
        _fake_rebuild_geometry_fit_source_rows,
    )

    rows = runtime_session._geometry_manual_rebuild_source_rows_for_background(
        1,
        consumer="geometry_fit_dataset",
    )

    assert rows == []
    assert resolve_calls
    assert all(call["background_index"] == 1 for call in resolve_calls)
    assert all(call["allow_generated_payload"] is True for call in resolve_calls)
    assert len(seen_projection_payloads) == 1
    assert isinstance(seen_projection_payloads[0], dict)
    assert seen_projection_payloads[0]["payload_marker"] == 1


def test_geometry_manual_project_peaks_by_row_background_preserves_input_order(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=99),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(ai_cache={}),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 4, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda idx: (
            np.full((4, 4), float(idx), dtype=np.float64),
            np.full((4, 4), float(idx), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda rows: [
                dict(
                    entry,
                    projected_background=int(
                        np.asarray(kwargs["current_background_native"]())[0, 0]
                    ),
                )
                for entry in rows or ()
                if isinstance(entry, dict)
            ]
        ),
    )

    rows = runtime_session._geometry_manual_project_peaks_by_row_background(
        [
            dict(_geometry_fit_worker_live_row(), background_index=1, row_id="first"),
            dict(_geometry_fit_worker_live_row(), background_index=0, row_id="second"),
            dict(_geometry_fit_worker_live_row(), background_index=1, row_id="third"),
        ]
    )

    assert [(str(row.get("row_id")), int(row.get("projected_background", -1))) for row in rows] == [
        ("first", 1),
        ("second", 0),
        ("third", 1),
    ]


def test_noncurrent_caked_signature_does_not_use_current_qr_context(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(ai_cache={}, analysis_preview_bins=(5, 7)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_view_for_index",
        lambda _idx: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_worker_caked_projection_view",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_qr_cylinder_caked_projection_context",
        lambda: {
            "detector_shape": (4, 4),
            "radial_axis": [999.0],
            "azimuth_axis": [111.0],
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda _idx: (
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ),
        raising=False,
    )

    signature = runtime_session._geometry_fit_targeted_projection_view_signature(
        1,
        mode_override="caked",
    )

    assert signature["mode"] == "caked"
    assert signature["background_index"] == 1
    assert signature["available"] is False
    assert signature["reason"] == "missing_background_caked_payload"
    assert signature["current_background_index"] == 0
    assert signature["analysis_bins"] == [5, 7]
    assert signature["projection_payload_digest"] is None
    assert "radial_axis" not in signature
    assert "azimuth_axis" not in signature
    assert "raw_azimuth_axis" not in signature
    assert "raw_to_gui_row_permutation" not in signature


def test_noncurrent_caked_signature_uses_generated_payload_when_available(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    resolve_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_resolve_targeted_caked_projection_payload",
        lambda background_index, **kwargs: (
            resolve_calls.append({"background_index": int(background_index), **dict(kwargs)})
            or {
                "detector_shape": (4, 4),
                "radial_axis": np.asarray([101.0, 102.0], dtype=np.float64),
                "azimuth_axis": np.asarray([3.0, 4.0], dtype=np.float64),
                "raw_azimuth_axis": np.asarray([3.0, 4.0], dtype=np.float64),
                "raw_to_gui_row_permutation": np.asarray([0, 1], dtype=np.int32),
                "transform_bundle": object(),
            }
        ),
        raising=False,
    )

    signature = runtime_session._geometry_fit_targeted_projection_view_signature(
        1,
        mode_override="caked",
    )

    assert resolve_calls
    assert resolve_calls[0]["background_index"] == 1
    assert resolve_calls[0]["allow_generated_payload"] is True
    assert signature["mode"] == "caked"
    assert signature["background_index"] == 1
    assert signature["available"] is True
    assert signature["detector_shape"] == [4, 4]
    assert signature["radial_axis"] == [101.0, 102.0]
    assert signature["azimuth_axis"] == [3.0, 4.0]
    assert signature["raw_azimuth_axis"] == [3.0, 4.0]
    assert signature["raw_to_gui_row_permutation"] == [0, 1]
    assert signature["projection_payload_digest"]


def test_detector_signature_omits_analysis_bins(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(analysis_preview_bins=(5, 7), ai_cache={}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda _idx: (
            np.zeros((4, 4), dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_current_background_native",
        lambda: np.zeros((4, 4), dtype=np.float64),
        raising=False,
    )

    signature = runtime_session._geometry_fit_targeted_projection_view_signature(
        0,
        mode_override="detector",
    )

    assert signature["mode"] == "detector"
    assert signature["background_index"] == 0
    assert signature["current_background_index"] == 0
    assert signature["detector_shape"] == [4, 4]
    assert signature["available"] is True
    assert "analysis_bins" not in signature


def test_unavailable_projection_signature_forces_targeted_projected_cache_miss(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)

    runtime_session._geometry_fit_store_targeted_projected_cache_entry(
        background_index=0,
        key_digest="digest",
        payload={
            "consumer": "geometry_fit_dataset",
            "projection_view_signature": {
                "mode": "caked",
                "background_index": 0,
                "available": False,
                "reason": "missing_background_caked_payload",
            },
            "projected_rows": [dict(_geometry_fit_worker_live_row(), background_index=0)],
        },
    )

    assert (
        runtime_session._geometry_fit_load_targeted_projected_cache_entry(
            background_index=0,
            key_digest="digest",
        )
        is None
    )


def test_targeted_projected_cache_rejects_signature_background_mismatch(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    simulation_state = _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)

    simulation_state.geometry_fit_targeted_projected_cache_by_background = {
        0: {
            "digest": {
                "consumer": "geometry_fit_dataset",
                "projection_view_signature": {
                    "mode": "detector",
                    "available": True,
                    "background_index": 1,
                },
                "projected_rows": [dict(_geometry_fit_worker_live_row(), background_index=0)],
            }
        }
    }

    assert (
        runtime_session._geometry_fit_load_targeted_projected_cache_entry(
            background_index=0,
            key_digest="digest",
        )
        is None
    )


def test_stored_rows_only_payload_is_never_projected_cache_hit(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    simulation_state = _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)

    simulation_state.geometry_fit_targeted_projected_cache_by_background = {
        0: {
            "digest": {
                "consumer": "geometry_fit_dataset",
                "projection_view_signature": {
                    "mode": "detector",
                    "available": True,
                    "background_index": 0,
                },
                "stored_rows": [dict(_geometry_fit_worker_live_row(), background_index=0)],
                "projected_rows": [],
            }
        }
    }

    assert (
        runtime_session._geometry_fit_load_targeted_projected_cache_entry(
            background_index=0,
            key_digest="digest",
        )
        is None
    )


def test_second_unchanged_preflight_reuses_targeted_projected_cache(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    simulation_state = _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    required_pairs = [_geometry_fit_worker_required_pair()]
    first_project_calls: list[int] = []

    monkeypatch.setattr(
        runtime_session,
        "_current_app_shell_view_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache_metadata",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache",
        lambda: pytest.fail("uncached first targeted preflight should skip heavy logged cache"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: [object()],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_build_source_rows_from_hit_tables",
        lambda *_args, **_kwargs: (
            [_geometry_fit_worker_live_row()],
            [],
            [],
            [],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda idx: (
            np.full((4, 4), float(idx), dtype=np.float64),
            np.full((4, 4), float(idx), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda rows: (
                first_project_calls.append(
                    int(np.asarray(kwargs["current_background_native"]())[0, 0])
                )
                or [
                    dict(
                        entry,
                        projected_background=int(
                            np.asarray(kwargs["current_background_native"]())[0, 0]
                        ),
                    )
                    for entry in rows or ()
                    if isinstance(entry, Mapping)
                ]
            )
        ),
    )

    first_rows = runtime_session._geometry_manual_rebuild_source_rows_for_background(
        0,
        consumer="geometry_fit_dataset",
        required_pairs=required_pairs,
    )
    first_diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert first_rows
    assert first_project_calls == [0]
    assert first_diagnostics["targeted_cache_hit"] is False
    assert first_diagnostics["targeted_simulation_used"] is True
    assert first_diagnostics["targeted_performance_gate"]["ok"] is True
    assert simulation_state.geometry_fit_targeted_projected_cache_by_background[0]

    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: pytest.fail(
            "unchanged second targeted preflight should reuse projected cache"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_build_source_rows_from_hit_tables",
        lambda *_args, **_kwargs: pytest.fail(
            "unchanged second targeted preflight should not rebuild source rows"
        ),
        raising=False,
    )
    second_project_calls: list[int] = []
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda rows: (
                second_project_calls.append(
                    int(np.asarray(kwargs["current_background_native"]())[0, 0])
                )
                or [
                    dict(
                        entry,
                        projected_background=int(
                            np.asarray(kwargs["current_background_native"]())[0, 0]
                        ),
                    )
                    for entry in rows or ()
                    if isinstance(entry, Mapping)
                ]
            )
        ),
    )

    second_rows = runtime_session._geometry_manual_rebuild_source_rows_for_background(
        0,
        consumer="geometry_fit_dataset",
        required_pairs=required_pairs,
    )
    second_diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert second_rows == first_rows
    assert second_project_calls == []
    assert second_diagnostics["targeted_cache_hit"] is True
    assert second_diagnostics["cache_source"] == "targeted_projected_cache"
    assert second_diagnostics["targeted_simulation_used"] is False
    assert second_diagnostics["full_source_rows_built_for_rebinding"] is False
    assert second_diagnostics["full_source_rows_projected_for_rebinding"] is False
    assert second_diagnostics["unrelated_projected_row_count_for_rebinding"] == 0
    assert second_diagnostics["unrelated_scored_row_count_for_rebinding"] == 0
    assert second_diagnostics["targeted_performance_gate"]["ok"] is True


def test_targeted_projected_cache_miss_reprojects_when_view_changes(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    simulation_state = _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    required_pairs = [_geometry_fit_worker_required_pair()]
    current_view = {"mode": "detector"}
    first_project_calls: list[int] = []
    second_project_calls: list[int] = []

    monkeypatch.setattr(
        runtime_session,
        "_current_app_shell_view_mode",
        lambda: str(current_view["mode"]),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache_metadata",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache",
        lambda: pytest.fail("targeted preflight should skip heavy logged cache"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: [object()],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_build_source_rows_from_hit_tables",
        lambda *_args, **_kwargs: (
            [_geometry_fit_worker_live_row()],
            [],
            [],
            [],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda idx: (
            np.full((4, 4), float(idx), dtype=np.float64),
            np.full((4, 4), float(idx), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_resolve_targeted_caked_projection_payload",
        lambda *_args, **_kwargs: _geometry_fit_worker_caked_payload(
            runtime_session,
            background_value=1.0,
            radial_values=[1.0, 2.0],
            azimuth_values=[3.0, 4.0],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda idx: (
            np.full((4, 4), float(idx), dtype=np.float64),
            np.full((4, 4), float(idx), dtype=np.float64),
        ),
        raising=False,
    )

    def _project_for_current_view(background_index, rows, **_kwargs):
        call_bucket = (
            first_project_calls if str(current_view["mode"]) == "detector" else second_project_calls
        )
        call_bucket.append(int(background_index))
        return [
            dict(
                entry,
                projected_view=str(current_view["mode"]),
                projected_background=int(background_index),
            )
            for entry in rows or ()
            if isinstance(entry, Mapping)
        ]

    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda rows: _project_for_current_view(
                int(np.asarray(kwargs["current_background_native"]())[0, 0]),
                rows,
            )
        ),
    )

    first_rows = runtime_session._geometry_manual_rebuild_source_rows_for_background(
        0,
        consumer="geometry_fit_dataset",
        required_pairs=required_pairs,
    )

    assert first_rows
    assert first_project_calls == [0]

    current_view["mode"] = "caked"
    second_simulation_calls: list[int] = []
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: second_simulation_calls.append(1) or [object()],
        raising=False,
    )
    second_build_calls: list[int] = []
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_build_source_rows_from_hit_tables",
        lambda *_args, **_kwargs: (
            second_build_calls.append(1)
            or (
                [_geometry_fit_worker_live_row()],
                [],
                [],
                [],
            )
        ),
        raising=False,
    )

    second_rows = runtime_session._geometry_manual_rebuild_source_rows_for_background(
        0,
        consumer="geometry_fit_dataset",
        required_pairs=required_pairs,
    )
    second_diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert second_simulation_calls == [1]
    assert second_build_calls == [1]
    assert second_project_calls == [0]
    assert second_diagnostics["targeted_cache_hit"] is False
    assert second_diagnostics["cache_source"] != "targeted_projected_cache"


def test_same_mode_changed_projection_signature_forces_cache_miss(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    simulation_state = _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    required_pairs = [_geometry_fit_worker_required_pair()]
    current_signature = {
        "mode": "caked",
        "detector_shape": [64, 64],
        "radial_axis": [1.0, 2.0],
        "azimuth_axis": [-10.0, 10.0],
    }
    first_project_calls: list[int] = []
    second_project_calls: list[int] = []

    monkeypatch.setattr(
        runtime_session,
        "_current_app_shell_view_mode",
        lambda: "caked",
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda *_args, **_kwargs: {
            **dict(current_signature),
            "background_index": 0,
            "current_background_index": 1,
            "available": True,
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache_metadata",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "load_most_recent_logged_intersection_cache",
        lambda: pytest.fail("targeted preflight should skip heavy logged cache"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: [object()],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_resolve_targeted_caked_projection_payload",
        lambda *_args, **_kwargs: _geometry_fit_worker_caked_payload(
            runtime_session,
            background_value=1.0,
            radial_values=[1.0, 2.0],
            azimuth_values=[3.0, 4.0],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_load_background_image_by_index",
        lambda idx: (
            np.full((4, 4), float(idx), dtype=np.float64),
            np.full((4, 4), float(idx), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_build_source_rows_from_hit_tables",
        lambda *_args, **_kwargs: (
            [_geometry_fit_worker_live_row()],
            [],
            [],
            [],
        ),
        raising=False,
    )

    def _project_for_signature(background_index, rows, **_kwargs):
        call_bucket = (
            first_project_calls
            if len(current_signature["radial_axis"]) == 2
            else second_project_calls
        )
        call_bucket.append(int(background_index))
        return [
            dict(
                entry,
                projected_signature=len(current_signature["radial_axis"]),
                projected_background=int(background_index),
            )
            for entry in rows or ()
            if isinstance(entry, Mapping)
        ]

    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            project_peaks_to_current_view=lambda rows: _project_for_signature(
                int(np.asarray(kwargs["current_background_native"]())[0, 0]),
                rows,
            )
        ),
    )

    first_rows = runtime_session._geometry_manual_rebuild_source_rows_for_background(
        0,
        consumer="geometry_fit_dataset",
        required_pairs=required_pairs,
    )
    first_diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert first_project_calls == [0]
    assert first_diagnostics["targeted_cache_hit"] is False

    current_signature["radial_axis"] = [1.0, 2.0, 3.0]
    second_simulation_calls: list[int] = []
    second_build_calls: list[int] = []
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: second_simulation_calls.append(1) or [object()],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_build_source_rows_from_hit_tables",
        lambda *_args, **_kwargs: (
            second_build_calls.append(1) or [_geometry_fit_worker_live_row()],
            [],
            [],
            [],
        ),
        raising=False,
    )

    second_rows = runtime_session._geometry_manual_rebuild_source_rows_for_background(
        0,
        consumer="geometry_fit_dataset",
        required_pairs=required_pairs,
    )
    second_diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert second_simulation_calls == [1]
    assert second_build_calls == [1]
    assert second_project_calls == [0]
    assert second_diagnostics["targeted_cache_hit"] is False
    assert second_diagnostics["cache_source"] != "targeted_projected_cache"


def test_targeted_projected_cache_stored_rows_only_does_not_count_as_hit(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    simulation_state = _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    required_pairs = [_geometry_fit_worker_required_pair()]
    rebuild_calls: list[int] = []
    projected_rows = [dict(_geometry_fit_worker_live_row(), projected_cache_refresh=True)]

    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_rebuild_source_rows_for_background",
        lambda *args, **kwargs: (
            rebuild_calls.append(1) or [dict(entry) for entry in projected_rows]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda *_args, **_kwargs: {"mode": "detector", "detector_shape": [64, 64]},
        raising=False,
    )

    background_idx = 0
    requested_signature = runtime_session._geometry_source_snapshot_signature_for_background(
        background_idx,
        {},
    )
    requested_signature_summary = runtime_session._live_cache_signature_summary(requested_signature)
    required_targets = (
        runtime_session.gui_geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=background_idx,
        )
    )
    required_keys = runtime_session.gui_geometry_fit._geometry_fit_required_branch_group_keys(
        required_targets
    )
    digest = runtime_session.gui_geometry_fit._geometry_fit_required_branch_group_keys_digest(
        required_keys,
        background_index=background_idx,
        requested_signature=runtime_session.gui_geometry_fit._geometry_fit_cache_jsonable(
            requested_signature
        ),
        requested_signature_summary=runtime_session.gui_geometry_fit._geometry_fit_cache_jsonable(
            requested_signature_summary
        ),
        preflight_mode="manual_geometry_targeted",
        consumer="geometry_fit_dataset",
        projection_view_mode="detector",
        projection_view_signature={"mode": "detector", "detector_shape": [64, 64]},
    )
    runtime_session._geometry_fit_store_targeted_projected_cache_entry(
        background_index=background_idx,
        key_digest=str(digest),
        payload={
            "background_index": background_idx,
            "requested_signature": runtime_session.gui_geometry_fit._geometry_fit_cache_jsonable(
                requested_signature
            ),
            "required_branch_group_keys_digest": str(digest),
            "preflight_mode": "manual_geometry_targeted",
            "consumer": "geometry_fit_dataset",
            "projection_view_mode": "detector",
            "projection_view_signature": {"mode": "detector", "detector_shape": [64, 64]},
            "stored_rows": [dict(_geometry_fit_worker_live_row())],
            "projected_rows": [],
            "cache_source": "broken_targeted_cache",
        },
    )

    rows = runtime_session._geometry_manual_source_rows_for_background(
        background_idx,
        consumer="geometry_fit_dataset",
        required_pairs=required_pairs,
    )
    diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert rows == projected_rows
    assert rebuild_calls == [1]
    assert bool(diagnostics.get("targeted_cache_hit", False)) is False
    assert diagnostics.get("cache_source") != "targeted_projected_cache"


def test_geometry_manual_source_rows_tolerates_list_projection_signature(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda *_args, **_kwargs: ["legacy", "detector"],
        raising=False,
    )

    rows = runtime_session._geometry_manual_source_rows_for_background(
        0,
        {"a": 4.143},
        consumer="initial_pairs_display",
    )
    diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert rows == []
    assert diagnostics["reason"] == "invalid_projection_view_signature_type:list"
    assert diagnostics["projection_view_signature"]["legacy_projection_view_signature"] == [
        "legacy",
        "detector",
    ]


def test_manual_pick_cache_source_rows_rebuild_allowed_for_manual_pick_cache(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    simulation_state = _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    runtime_session.background_runtime_state.current_background_index = 0
    simulation_state.source_row_snapshots = {}
    simulation_state.stored_max_positions_local = [np.zeros((1, 7), dtype=np.float64)]
    rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 10),
            "hkl": (-1, 0, 10),
            "source_table_index": 3,
            "source_row_index": 4,
            "source_branch_index": 0,
            "sim_col": 12.0,
            "sim_row": 14.0,
            "native_col": 12.0,
            "native_row": 14.0,
        }
    ]
    rebuild_calls: list[dict[str, object]] = []
    projection_mode_overrides: list[object] = []

    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pick_uses_caked_space",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda _idx, **kwargs: (
            projection_mode_overrides.append(kwargs.get("mode_override"))
            or {
                "mode": str(kwargs.get("mode_override") or "detector"),
                "detector_shape": [64, 64],
                "available": True,
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_rebuild_source_rows_for_background",
        lambda background_idx, param_set=None, **kwargs: (
            rebuild_calls.append(
                {
                    "background_idx": int(background_idx),
                    "consumer": kwargs.get("consumer"),
                    "param_set": dict(param_set or {}),
                }
            )
            or [dict(entry) for entry in rows]
        ),
        raising=False,
    )

    returned_rows = runtime_session._geometry_manual_source_rows_for_background(
        0,
        {"a": 4.143},
        consumer="manual_pick_cache",
    )
    diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert returned_rows == [dict(entry) for entry in rows]
    assert rebuild_calls == [
        {
            "background_idx": 0,
            "consumer": "manual_pick_cache",
            "param_set": {"a": 4.143},
        }
    ]
    assert projection_mode_overrides == ["detector"]
    assert diagnostics["status"] == "snapshot_rebuilt"
    assert diagnostics["rebuild_attempted"] is True
    assert diagnostics["rebuild_returned_row_count"] == 1
    assert returned_rows[0]["q_group_key"] == ("q_group", "primary", 1, 10)


def test_manual_pick_cache_source_rows_rebuilds_when_snapshot_projection_empty(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
def test_manual_pick_cache_caked_view_uses_detector_rows_when_projector_missing(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    simulation_state = _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    runtime_session.background_runtime_state.current_background_index = 0
    simulation_state.source_row_snapshots = {}
    simulation_state.stored_max_positions_local = [np.zeros((1, 7), dtype=np.float64)]
    rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 10),
            "hkl": (-1, 0, 10),
            "source_table_index": 3,
            "source_row_index": 4,
            "source_branch_index": 0,
            "display_col": 120.0,
            "display_row": 130.0,
            "native_col": 220.0,
            "native_row": 230.0,
        }
    ]
    rebuild_calls: list[dict[str, object]] = []
    projection_mode_overrides: list[object] = []

    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pick_uses_caked_space",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_view_for_index",
        lambda _idx: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda _idx, **kwargs: (
            projection_mode_overrides.append(kwargs.get("mode_override"))
            or {
                "mode": str(kwargs.get("mode_override") or "caked"),
                "detector_shape": [64, 64],
                "available": kwargs.get("mode_override") == "detector",
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_rebuild_source_rows_for_background",
        lambda background_idx, param_set=None, **kwargs: (
            rebuild_calls.append(
                {
                    "background_idx": int(background_idx),
                    "consumer": kwargs.get("consumer"),
                    "param_set": dict(param_set or {}),
                }
            )
            or [dict(entry) for entry in rows]
        ),
        raising=False,
    )

    returned_rows = runtime_session._geometry_manual_source_rows_for_background(
        0,
        {"a": 4.143},
        consumer="manual_pick_cache",
    )
    diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert returned_rows == [dict(entry) for entry in rows]
    assert rebuild_calls == [
        {
            "background_idx": 0,
            "consumer": "manual_pick_cache",
            "param_set": {"a": 4.143},
        }
    ]
    assert projection_mode_overrides == ["detector"]
    assert diagnostics["projection_view_mode"] == "detector"
    assert diagnostics["status"] == "snapshot_rebuilt"


def test_manual_pick_cache_coverage_rebuild_bypasses_partial_snapshot(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    simulation_state = _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    runtime_session.background_runtime_state.current_background_index = 0
    simulation_state.stored_max_positions_local = [np.zeros((1, 7), dtype=np.float64)]
    partial_row = {
        "q_group_key": ("q_group", "primary", 1, 0),
        "hkl": (-1, 0, 0),
        "source_reflection_index": 10,
        "source_row_index": 0,
        "display_col": 12.0,
        "display_row": 14.0,
        "background_index": 0,
    }
    full_rows = [
        dict(partial_row),
        {
            "q_group_key": ("q_group", "primary", 3, 2),
            "hkl": (-3, 0, 2),
            "source_reflection_index": 21,
            "source_row_index": 1,
            "best_sample_index": 4,
            "display_col": 42.0,
            "display_row": 44.0,
            "background_index": 0,
        },
    ]
    requested_signature = runtime_session._geometry_source_snapshot_signature_for_background(
        0,
        {"a": 4.143},
    )
    simulation_state.source_row_snapshots = {
        0: {
            "simulation_signature": requested_signature,
            "row_content_signature": None,
            "stored_rows": [dict(partial_row)],
            "rows": [dict(partial_row)],
            "projected_rows": [],
            "valid_for_picker": True,
            "valid_for_geometry_fit_dataset": True,
            "background_index": 0,
            "created_from": "partial_snapshot",
        }
    }
    rebuild_calls: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pick_uses_caked_space",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_view_for_index",
        lambda _idx: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda _idx, **kwargs: {
            "mode": str(kwargs.get("mode_override") or "detector"),
            "detector_shape": [64, 64],
            "available": True,
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_rebuild_source_rows_for_background",
        lambda _idx, _params=None, **kwargs: (
            rebuild_calls.append(str(kwargs.get("consumer")))
            or [dict(entry) for entry in full_rows]
        ),
        raising=False,
    )

    returned_rows = runtime_session._geometry_manual_source_rows_for_background(
        0,
        {"a": 4.143},
        consumer="manual_pick_cache_coverage",
    )
    diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert returned_rows == [dict(entry) for entry in full_rows]
    assert rebuild_calls == ["manual_pick_cache_coverage"]
    assert diagnostics["status"] == "snapshot_rebuilt_coverage"
    assert diagnostics["rebuild_attempted"] is True
    assert diagnostics["rebuild_returned_row_count"] == 2


def test_geometry_source_snapshot_signature_tracks_sf_picker_inventory(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class ValueVar:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    simulation_state = SimpleNamespace(
        sim_primary_qr={("base",): np.zeros((1, 7), dtype=np.float64)},
        sim_miller1=np.zeros((1, 3), dtype=np.float64),
        sim_miller2=np.zeros((1, 3), dtype=np.float64),
        sf_prune_stats={"qr_kept": 2, "hkl_primary_kept": 3},
        stored_q_group_content_signature=("rows", 1),
        primary_source_mode="qr",
        primary_requested_source_mode="qr",
        primary_active_contribution_keys=[("sf", "base")],
        primary_requested_contribution_keys=[("sf", "base")],
        primary_filter_signature=("filter", "base"),
        primary_requested_filter_signature=("filter", "base"),
    )
    mosaic_params = {
        "solve_q_steps": 1000,
        "solve_q_rel_tol": 5.0e-4,
        "solve_q_mode": 1,
        "events_per_beam_phase": 25,
        "_sampling_signature": ("sampling", 25),
        "beam_x_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(3, dtype=np.float64),
        "sigma_mosaic_deg": 0.0,
        "gamma_mosaic_deg": 0.0,
        "eta": 0.0,
    }
    p0 = ValueVar(0.01)
    p1 = ValueVar(0.0)
    p2 = ValueVar(0.5)
    w0 = ValueVar(1.0)
    w1 = ValueVar(1.0)
    w2 = ValueVar(1.0)

    monkeypatch.setattr(runtime_session, "simulation_runtime_state", simulation_state)
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 4.143, "c": 28.64, "theta_initial": 0.0},
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "build_mosaic_params", lambda: dict(mosaic_params))
    monkeypatch.setattr(runtime_session, "_current_optics_mode_flag", lambda: 0)
    monkeypatch.setattr(runtime_session, "current_sf_prune_bias", lambda: 2.0, raising=False)
    monkeypatch.setattr(runtime_session, "_current_ordered_structure_scale", lambda: 1.0)
    monkeypatch.setattr(runtime_session, "_qr_cylinder_replace_simulation_enabled", lambda: False)
    monkeypatch.setattr(runtime_session, "av2", None, raising=False)
    monkeypatch.setattr(runtime_session, "cv2", None, raising=False)
    monkeypatch.setattr(runtime_session, "p0_var", p0, raising=False)
    monkeypatch.setattr(runtime_session, "p1_var", p1, raising=False)
    monkeypatch.setattr(runtime_session, "p2_var", p2, raising=False)
    monkeypatch.setattr(runtime_session, "w0_var", w0, raising=False)
    monkeypatch.setattr(runtime_session, "w1_var", w1, raising=False)
    monkeypatch.setattr(runtime_session, "w2_var", w2, raising=False)

    baseline_signature = runtime_session._geometry_source_snapshot_signature_for_background(
        0,
        {"a": 4.143},
    )
    p1.set(0.99)
    p_signature = runtime_session._geometry_source_snapshot_signature_for_background(
        0,
        {"a": 4.143},
    )
    simulation_state.primary_active_contribution_keys = [("sf", "expanded")]
    key_signature = runtime_session._geometry_source_snapshot_signature_for_background(
        0,
        {"a": 4.143},
    )

    assert p_signature != baseline_signature
    assert key_signature != p_signature


    simulation_state = _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    runtime_session.background_runtime_state.current_background_index = 0
    simulation_state.stored_max_positions_local = [np.zeros((1, 7), dtype=np.float64)]
    requested_signature = runtime_session._geometry_source_snapshot_signature_for_background(
        0,
        {"a": 4.143},
    )
    caked_row = {
        "q_group_key": ("q_group", "primary", 1, 10),
        "hkl": (-1, 0, 10),
        "source_table_index": 3,
        "source_row_index": 4,
        "display_col": 2.5,
        "display_row": -17.0,
        "caked_x": 2.5,
        "caked_y": -17.0,
        "display_frame": "caked_display",
        "background_index": 0,
    }
    rebuilt_row = {
        "q_group_key": ("q_group", "primary", 1, 10),
        "hkl": (-1, 0, 10),
        "source_table_index": 3,
        "source_row_index": 4,
        "source_branch_index": 0,
        "sim_col": 12.0,
        "sim_row": 14.0,
        "native_col": 12.0,
        "native_row": 14.0,
        "background_index": 0,
    }
    simulation_state.source_row_snapshots = {
        0: {
            "background_index": 0,
            "simulation_signature": requested_signature,
            "stored_rows": [dict(caked_row)],
            "rows": [dict(caked_row)],
            "projected_rows": [],
            "valid_for_picker": True,
            "valid_for_geometry_fit_dataset": True,
            "row_content_signature": None,
            "projection_view_signature": {"mode": "caked", "available": True},
            "created_from": "caked_preview",
        }
    }
    rebuild_calls: list[dict[str, object]] = []
    projection_mode_overrides: list[object] = []

    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pick_uses_caked_space",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda _idx, **kwargs: (
            projection_mode_overrides.append(kwargs.get("mode_override"))
            or {
                "mode": str(kwargs.get("mode_override") or "detector"),
                "detector_shape": [64, 64],
                "available": True,
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_project_peaks_for_background",
        lambda *_args, **_kwargs: [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_rebuild_source_rows_for_background",
        lambda background_idx, param_set=None, **kwargs: (
            rebuild_calls.append(
                {
                    "background_idx": int(background_idx),
                    "consumer": kwargs.get("consumer"),
                    "param_set": dict(param_set or {}),
                }
            )
            or [dict(rebuilt_row)]
        ),
        raising=False,
    )

    returned_rows = runtime_session._geometry_manual_source_rows_for_background(
        0,
        {"a": 4.143},
        consumer="manual_pick_cache",
    )
    diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert returned_rows == [dict(rebuilt_row)]
    assert rebuild_calls == [
        {
            "background_idx": 0,
            "consumer": "manual_pick_cache",
            "param_set": {"a": 4.143},
        }
    ]
    assert projection_mode_overrides == ["detector"]
    assert diagnostics["status"] == "snapshot_rebuilt"
    assert diagnostics["rebuild_attempted"] is True
    assert diagnostics["rebuild_returned_row_count"] == 1


def test_source_snapshot_signature_survives_manual_pick_arming(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    manual_geometry = importlib.import_module("ra_sim.gui.manual_geometry")

    runtime_state = SimpleNamespace(
        sim_primary_qr={},
        sim_miller1=np.asarray([[1.0, 0.0, 2.0]], dtype=np.float64),
        sim_miller2=np.asarray([[0.0, 1.0, 1.0]], dtype=np.float64),
        sf_prune_stats={},
        stored_q_group_content_signature=("rows", 2),
        stored_hit_table_signature=("hit", 1),
    )
    geometry_state = SimpleNamespace(manual_pick_armed=False)
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        geometry_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 5.0, "c": 6.0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "build_mosaic_params",
        lambda: {"sigma_mosaic_deg": 0.1, "gamma_mosaic_deg": 0.2, "eta": 0.3},
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "current_sf_prune_bias", lambda: 0.0, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_current_ordered_structure_scale",
        lambda: 1.0,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_qr_cylinder_replace_simulation_enabled",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_current_optics_mode_flag", lambda: 0, raising=False)
    monkeypatch.setattr(runtime_session, "av2", None, raising=False)
    monkeypatch.setattr(runtime_session, "cv2", None, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_from_params",
        lambda params, **kwargs: (
            "image",
            bool(kwargs.get("include_mosaic_shape")),
            int(kwargs.get("optics_mode_component", -1)),
            float(params.get("a", 0.0)),
            float(params.get("c", 0.0)),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_should_collect_hit_tables_for_update",
        lambda: bool(geometry_state.manual_pick_armed),
        raising=False,
    )

    before_signature = runtime_session._geometry_source_snapshot_signature_for_background(0, {})
    geometry_state.manual_pick_armed = True
    after_signature = runtime_session._geometry_source_snapshot_signature_for_background(0, {})

    assert after_signature == before_signature

    rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 2),
            "hkl": (1, 0, 2),
            "qr": 2.0,
            "qz": 4.0,
            "sim_col": 42.0,
            "sim_row": 55.0,
            "source_branch_index": 0,
        }
    ]
    cache_state = {"signature": None, "data": {}}
    callbacks = manual_geometry.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 1),
        current_background_index=lambda: 0,
        current_background_image=lambda: np.ones((8, 8), dtype=float),
        use_caked_space=lambda: False,
        replace_cache_state=lambda signature, data: cache_state.update(
            {"signature": signature, "data": dict(data)}
        ),
        current_geometry_fit_params=lambda: {"a": 5.0, "c": 6.0},
        pairs_for_index=lambda _idx: [],
        source_rows_for_background=lambda *_args, **_kwargs: [dict(entry) for entry in rows],
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_grouped_candidates=lambda candidate_rows: {
            ("q_group", "primary", 1, 2): [dict(entry) for entry in candidate_rows or ()]
        },
        build_simulated_lookup=lambda candidate_rows: {
            (0, idx): dict(entry) for idx, entry in enumerate(candidate_rows or ())
        },
        entry_display_coords=lambda entry: (
            (float(entry["sim_col"]), float(entry["sim_row"])) if entry else None
        ),
        source_snapshot_signature_for_background=(
            runtime_session._geometry_source_snapshot_signature_for_background
        ),
    )

    cache_data = callbacks.get_pick_cache(param_set={"a": 5.0, "c": 6.0}, prefer_cache=True)

    assert cache_data["grouped_candidates"]


def test_runtime_targeted_projected_cache_reuses_detector_entry_when_current_background_drifts(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    required_pairs = [_geometry_fit_worker_required_pair()]
    rebuild_calls: list[int] = []
    signature_state = {"current_background_index": 0}
    projected_rows = [
        dict(
            _geometry_fit_worker_live_row(),
            background_index=0,
            projected_cache_refresh=True,
        )
    ]

    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_rebuild_source_rows_for_background",
        lambda *args, **kwargs: (
            rebuild_calls.append(1) or [dict(entry) for entry in projected_rows]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda *_args, **_kwargs: {
            "mode": "detector",
            "background_index": 0,
            "current_background_index": int(signature_state["current_background_index"]),
            "detector_shape": [64, 64],
            "analysis_bins": [5, 7],
            "available": True,
        },
        raising=False,
    )

    background_idx = 0
    requested_signature = runtime_session._geometry_source_snapshot_signature_for_background(
        background_idx,
        {},
    )
    requested_signature_summary = runtime_session._live_cache_signature_summary(requested_signature)
    required_targets = (
        runtime_session.gui_geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=background_idx,
        )
    )
    required_keys = runtime_session.gui_geometry_fit._geometry_fit_required_branch_group_keys(
        required_targets
    )
    first_signature = runtime_session._geometry_fit_targeted_projection_view_signature(
        background_idx
    )
    digest = runtime_session.gui_geometry_fit._geometry_fit_required_branch_group_keys_digest(
        required_keys,
        background_index=background_idx,
        requested_signature=runtime_session.gui_geometry_fit._geometry_fit_cache_jsonable(
            requested_signature
        ),
        requested_signature_summary=runtime_session.gui_geometry_fit._geometry_fit_cache_jsonable(
            requested_signature_summary
        ),
        preflight_mode="manual_geometry_targeted",
        consumer="geometry_fit_dataset",
        projection_view_mode="detector",
        projection_view_signature=first_signature,
    )
    runtime_session._geometry_fit_store_targeted_projected_cache_entry(
        background_index=background_idx,
        key_digest=str(digest),
        payload={
            "background_index": background_idx,
            "requested_signature": runtime_session.gui_geometry_fit._geometry_fit_cache_jsonable(
                requested_signature
            ),
            "requested_signature_summary": runtime_session.gui_geometry_fit._geometry_fit_cache_jsonable(
                requested_signature_summary
            ),
            "required_branch_group_keys_digest": str(digest),
            "preflight_mode": "manual_geometry_targeted",
            "consumer": "geometry_fit_dataset",
            "projection_view_mode": "detector",
            "projection_view_signature": first_signature,
            "projected_rows": [dict(entry) for entry in projected_rows],
            "cache_source": "targeted_projected_cache",
        },
    )

    signature_state["current_background_index"] = 1

    rows = runtime_session._geometry_manual_source_rows_for_background(
        background_idx,
        consumer="geometry_fit_dataset",
        required_pairs=required_pairs,
    )
    diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert rows == projected_rows
    assert rebuild_calls == []
    assert bool(diagnostics.get("targeted_cache_hit", False)) is True
    assert diagnostics.get("cache_source") == "targeted_projected_cache"


def test_runtime_targeted_projected_cache_reuses_detector_entry_when_analysis_bins_drift(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _patch_runtime_targeted_rebuild_env(monkeypatch, runtime_session)
    required_pairs = [_geometry_fit_worker_required_pair()]
    rebuild_calls: list[int] = []
    signature_state = {
        "current_background_index": 1,
        "analysis_bins": [5, 7],
    }
    projected_rows = [
        dict(
            _geometry_fit_worker_live_row(),
            background_index=0,
            projected_cache_refresh=True,
        )
    ]

    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_rebuild_source_rows_for_background",
        lambda *args, **kwargs: (
            rebuild_calls.append(1) or [dict(entry) for entry in projected_rows]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda *_args, **_kwargs: {
            "mode": "detector",
            "background_index": 0,
            "current_background_index": int(signature_state["current_background_index"]),
            "detector_shape": [64, 64],
            "analysis_bins": list(signature_state["analysis_bins"]),
            "available": True,
        },
        raising=False,
    )

    background_idx = 0
    requested_signature = runtime_session._geometry_source_snapshot_signature_for_background(
        background_idx,
        {},
    )
    requested_signature_summary = runtime_session._live_cache_signature_summary(requested_signature)
    required_targets = (
        runtime_session.gui_geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=background_idx,
        )
    )
    required_keys = runtime_session.gui_geometry_fit._geometry_fit_required_branch_group_keys(
        required_targets
    )
    first_signature = runtime_session._geometry_fit_targeted_projection_view_signature(
        background_idx
    )
    digest = runtime_session.gui_geometry_fit._geometry_fit_required_branch_group_keys_digest(
        required_keys,
        background_index=background_idx,
        requested_signature=runtime_session.gui_geometry_fit._geometry_fit_cache_jsonable(
            requested_signature
        ),
        requested_signature_summary=runtime_session.gui_geometry_fit._geometry_fit_cache_jsonable(
            requested_signature_summary
        ),
        preflight_mode="manual_geometry_targeted",
        consumer="geometry_fit_dataset",
        projection_view_mode="detector",
        projection_view_signature=first_signature,
    )
    runtime_session._geometry_fit_store_targeted_projected_cache_entry(
        background_index=background_idx,
        key_digest=str(digest),
        payload={
            "background_index": background_idx,
            "requested_signature": runtime_session.gui_geometry_fit._geometry_fit_cache_jsonable(
                requested_signature
            ),
            "requested_signature_summary": runtime_session.gui_geometry_fit._geometry_fit_cache_jsonable(
                requested_signature_summary
            ),
            "required_branch_group_keys_digest": str(digest),
            "preflight_mode": "manual_geometry_targeted",
            "consumer": "geometry_fit_dataset",
            "projection_view_mode": "detector",
            "projection_view_signature": first_signature,
            "projected_rows": [dict(entry) for entry in projected_rows],
            "cache_source": "targeted_projected_cache",
        },
    )

    signature_state["analysis_bins"] = [11, 13]

    rows = runtime_session._geometry_manual_source_rows_for_background(
        background_idx,
        consumer="geometry_fit_dataset",
        required_pairs=required_pairs,
    )
    diagnostics = runtime_session._geometry_manual_last_source_snapshot_diagnostics()

    assert rows == projected_rows
    assert rebuild_calls == []
    assert bool(diagnostics.get("targeted_cache_hit", False)) is True
    assert diagnostics.get("cache_source") == "targeted_projected_cache"


def test_runtime_session_late_caked_event_is_drained_after_worker_exit(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    cmd_lines: list[str] = []
    progress_messages: list[str] = []
    scheduled_callbacks: list[tuple[int, object]] = []

    class _Root:
        def after(self, delay_ms, callback) -> str:
            scheduled_callbacks.append((int(delay_ms), callback))
            return f"after-token-{len(scheduled_callbacks)}"

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_cmd_line",
        lambda text: cmd_lines.append(str(text)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label_geometry",
        SimpleNamespace(
            config=lambda **kwargs: progress_messages.append(str(kwargs.get("text", "")))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_event_queue=runtime_session.queue.Queue(),
            geometry_fit_active_job=None,
            geometry_fit_future=None,
            geometry_fit_late_event_poll_token=None,
            geometry_fit_pending_late_event_tokens=set(),
            geometry_fit_pending_late_event_deadlines={},
            geometry_fit_late_event_tail_generation=0,
        ),
        raising=False,
    )

    event_queue = runtime_session.simulation_runtime_state.geometry_fit_event_queue
    event_queue.put(
        {
            "job_id": 99,
            "kind": "source_cache_caked_view_timeout",
            "payload": {
                "background_index": 0,
                "source_cache_generation_id": 7,
                "status": "timeout",
                "elapsed_s": 5.0,
            },
        }
    )
    runtime_session._drain_geometry_fit_worker_events(job_id=99)

    assert runtime_session.simulation_runtime_state.geometry_fit_pending_late_event_tokens == {
        (99, 0, 7)
    }
    assert len(scheduled_callbacks) == 1

    event_queue.put(
        {
            "job_id": 99,
            "kind": "source_cache_caked_view_ready",
            "payload": {
                "background_index": 0,
                "source_cache_generation_id": 7,
                "caked_view_stored": True,
                "caked_view_status": "stored",
                "late": True,
            },
        }
    )
    _delay_ms, callback = scheduled_callbacks.pop(0)
    callback()

    assert runtime_session.simulation_runtime_state.geometry_fit_pending_late_event_tokens == set()
    assert any(
        "source_cache_caked_view_ready" in line and "late=true" in line for line in cmd_lines
    )
    assert len(progress_messages) == 1
    assert "source_cache_caked_view_timeout" in progress_messages[0]


def test_runtime_session_late_caked_events_do_not_overwrite_progress_text(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    cmd_lines: list[str] = []
    progress_messages: list[str] = []

    monkeypatch.setattr(runtime_session, "root", SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_cmd_line",
        lambda text: cmd_lines.append(str(text)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label_geometry",
        SimpleNamespace(
            config=lambda **kwargs: progress_messages.append(str(kwargs.get("text", "")))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_late_event_poll_token=None,
            geometry_fit_pending_late_event_tokens=set(),
            geometry_fit_pending_late_event_deadlines={},
            geometry_fit_late_event_tail_generation=0,
            geometry_fit_active_job=None,
            geometry_fit_future=None,
        ),
        raising=False,
    )

    runtime_session._handle_geometry_fit_worker_event(
        "source_cache_build_ready",
        {
            "job_id": 99,
            "background_index": 0,
            "source_cache_generation_id": 7,
            "row_count": 3,
        },
    )
    runtime_session._handle_geometry_fit_worker_event(
        "progress_text",
        {"text": "solver: iter 1"},
    )
    runtime_session._handle_geometry_fit_worker_event(
        "source_cache_caked_view_ready",
        {
            "job_id": 99,
            "background_index": 0,
            "source_cache_generation_id": 7,
            "caked_view_stored": True,
            "caked_view_status": "stored",
            "late": True,
        },
    )

    assert len(progress_messages) >= 2
    assert "source_cache_build_ready" in progress_messages[0]
    assert progress_messages[-1] == "solver: iter 1"
    assert any(
        "source_cache_caked_view_ready" in line and "late=true" in line for line in cmd_lines
    )


def test_runtime_session_stale_late_caked_event_does_not_surface_in_new_run(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    cmd_lines: list[str] = []
    scheduled_callbacks: list[tuple[int, object]] = []

    class _Root:
        def after(self, delay_ms, callback) -> str:
            scheduled_callbacks.append((int(delay_ms), callback))
            return f"after-token-{len(scheduled_callbacks)}"

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_cmd_line",
        lambda text: cmd_lines.append(str(text)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label_geometry",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_event_queue=runtime_session.queue.Queue(),
            geometry_fit_active_job=None,
            geometry_fit_future=None,
            geometry_fit_late_event_poll_token=None,
            geometry_fit_pending_late_event_tokens=set(),
            geometry_fit_pending_late_event_deadlines={},
            geometry_fit_late_event_tail_generation=0,
        ),
        raising=False,
    )

    event_queue = runtime_session.simulation_runtime_state.geometry_fit_event_queue
    event_queue.put(
        {
            "job_id": 99,
            "kind": "source_cache_caked_view_timeout",
            "payload": {
                "background_index": 0,
                "source_cache_generation_id": 7,
                "status": "timeout",
                "elapsed_s": 5.0,
            },
        }
    )
    runtime_session._drain_geometry_fit_worker_events(job_id=99)

    assert len(scheduled_callbacks) == 1
    cmd_lines.clear()

    runtime_session._clear_geometry_fit_late_event_tail_state()
    runtime_session.simulation_runtime_state.geometry_fit_active_job = {"job_id": 100}
    runtime_session.simulation_runtime_state.geometry_fit_future = object()

    event_queue.put(
        {
            "job_id": 99,
            "kind": "source_cache_caked_view_ready",
            "payload": {
                "background_index": 0,
                "source_cache_generation_id": 7,
                "caked_view_stored": True,
                "caked_view_status": "stored",
                "late": True,
            },
        }
    )

    _delay_ms, callback = scheduled_callbacks.pop(0)
    callback()

    assert runtime_session.simulation_runtime_state.geometry_fit_pending_late_event_tokens == set()
    assert cmd_lines == []

    runtime_session._drain_geometry_fit_worker_events(job_id=100)
    assert cmd_lines == []


def test_runtime_session_late_caked_tail_drain_expires_hung_tokens(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    scheduled_callbacks: list[tuple[int, object]] = []
    perf_state = {"value": 0.0}

    class _Root:
        def after(self, delay_ms, callback) -> str:
            scheduled_callbacks.append((int(delay_ms), callback))
            return f"after-token-{len(scheduled_callbacks)}"

    def _fake_perf_counter() -> float:
        return float(perf_state["value"])

    monkeypatch.setattr(runtime_session, "root", _Root(), raising=False)
    monkeypatch.setattr(runtime_session, "perf_counter", _fake_perf_counter)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label_geometry",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_cmd_line",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_event_queue=runtime_session.queue.Queue(),
            geometry_fit_active_job=None,
            geometry_fit_future=None,
            geometry_fit_late_event_poll_token=None,
            geometry_fit_pending_late_event_tokens={(99, 0, 7)},
            geometry_fit_pending_late_event_deadlines={(99, 0, 7): 1.0},
            geometry_fit_late_event_tail_generation=0,
        ),
        raising=False,
    )

    runtime_session._schedule_geometry_fit_late_event_tail_drain()

    assert len(scheduled_callbacks) == 1

    perf_state["value"] = 2.0
    _delay_ms, callback = scheduled_callbacks.pop(0)
    callback()

    assert runtime_session.simulation_runtime_state.geometry_fit_pending_late_event_tokens == set()
    assert runtime_session.simulation_runtime_state.geometry_fit_pending_late_event_deadlines == {}
    assert runtime_session.simulation_runtime_state.geometry_fit_late_event_poll_token is None
    assert scheduled_callbacks == []


def test_runtime_impl_keeps_1d_updates_gated_on_intensity_accumulation() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "if (\n                one_d_analysis_requested" in source


def test_runtime_impl_places_qr_cylinder_mode_in_quick_controls() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '"key": "qr_cylinder_mode"' in source
    assert '"key": "display_raster_size"' in source
    assert '"label": "display px"' in source
    assert "_refresh_main_display_raster_projection" in source
    assert "QR_CYLINDER_DISPLAY_MODE_REPLACE" in source
    assert "parent=app_shell_view_state.match_peak_tools_frame" in source


def test_runtime_impl_moves_analysis_view_options_and_auto_match_to_quick_controls() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '"key": "fast_viewer"' not in source
    assert '"key": "log_display"' in source
    assert '"key": "show_qz_rods"' not in source
    assert '"key": "clear_integration_region"' in source
    assert '"key": "auto_match_scale"' in source
    assert 'control_type": "check"' in source
    assert 'control_type": "button"' in source
    assert "parent=None" in source
    assert "RA_SIM_FAST_VIEWER" not in source
    assert "control_locked=True" not in source
    assert "display_controls_view_state.fast_viewer_checkbutton = (" not in source
    assert (
        "display_controls_view_state.simulation_controls_frame"
        not in source[
            source.index("def _auto_match_scale_factor_to_radial_peak(") : source.index(
                "def _update_chi_square_display("
            )
        ]
    )
    assert (
        "def _current_selected_qr_rod_caked_mask_payload() -> dict[str, object] | None:" in source
    )
    payload_start = source.index(
        "def _current_selected_qr_rod_caked_mask_payload() -> dict[str, object] | None:"
    )
    payload_end = source.index("def _current_selected_qr_rod_drag_context()", payload_start)
    payload_block = source[payload_start:payload_end]
    assert "build_selected_qr_rod_qz_caked_mask(" in payload_block
    assert "q_space_rect_mask_for_qr_centers" not in payload_block
    assert "caked_axes_to_qr_qz_maps" not in payload_block
    assert "def _sync_selected_qr_rod_controls_state() -> None:" in source
    assert "def _invalidate_qr_cylinder_band_cache() -> None:" in source


def test_runtime_impl_reset_to_defaults_resets_selected_qr_rod_controls() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    block_start = source.index("def reset_to_defaults():")
    block_end = source.index("def _initialize_runtime_controls_block_33() -> None:", block_start)
    block = source[block_start:block_end]

    assert "integrate_selected_qr_rod_var.set(False)" in block
    assert "mirror_selected_qr_phi_var.set(False)" in block
    assert 'caked_intensity_mode_var.set("density")' in block
    assert 'rod_profile_intensity_mode_var.set("density")' in block
    assert 'selected_qr_rod_key_var.set("")' in block
    assert "qz_extent = _current_caked_qz_extent()" in block
    assert "delta_qr_var.set(0.1)" in block
    assert "_sync_selected_qr_rod_controls_state()" in block


def test_runtime_impl_selected_qr_rod_toggle_disables_peak_pick_immediately() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "disable_peak_pick=lambda: _set_analysis_peak_pick_mode(False)" in source


def test_runtime_impl_syncs_selected_qr_rod_controls_before_1d_refresh() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    do_update_start = source.index("def do_update():")
    do_update_block = source[do_update_start:]

    sync_index = do_update_block.index("_sync_selected_qr_rod_controls_state()")
    update_1d_index = do_update_block.index("_update_1d_plots_from_caked(sim_res2, bg_res2)")

    assert sync_index < update_1d_index


def test_runtime_impl_gui_state_import_disables_peak_pick_when_selected_qr_rod_restores() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    block_start = source.index(
        "def _apply_full_gui_state_snapshot(snapshot: dict[str, object]) -> str:"
    )
    block_end = source.index("def _export_full_gui_state() -> None:", block_start)
    block = source[block_start:block_end]

    sync_index = block.index("_sync_selected_qr_rod_controls_state()")
    roi_index = block.index(
        '_current_analysis_roi_values().get("integrate_selected_qr_rod", False)'
    )
    disable_index = block.index("_set_analysis_peak_pick_mode(")

    assert "analysis_peak_selection_state.pick_armed" in block
    assert sync_index < roi_index < disable_index


def test_runtime_impl_gui_state_import_prepares_caked_view_before_manual_restore() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    block_start = source.index(
        "def _apply_full_gui_state_snapshot(snapshot: dict[str, object]) -> str:"
    )
    block_end = source.index("def _export_full_gui_state() -> None:", block_start)
    block = source[block_start:block_end]

    detect_index = block.index(
        "geometry_manual_pairs_rows_missing_caked_backfill_count("
    )
    ensure_index = block.index("_ensure_geometry_fit_caked_view(force_refresh=True)")
    restore_index = block.index("gui_state_io.apply_gui_state_geometry(")

    assert detect_index < ensure_index < restore_index


def test_runtime_impl_full_gui_state_export_includes_selected_qr_rod_fields() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    block_start = source.index("def _collect_full_gui_state_snapshot() -> dict[str, object]:")
    block_end = source.index("def _load_background_files_for_import_state(", block_start)
    block = source[block_start:block_end]

    assert 'snapshot["analysis_range"] = dict(_current_analysis_roi_values())' in block
    assert "_current_analysis_range_values()" not in block


def test_runtime_session_current_analysis_range_values_preserve_rod_controls(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(
        runtime_session,
        "integration_range_controls_view_state",
        SimpleNamespace(
            tth_min_value=1.5,
            tth_max_value=55.0,
            phi_min_value=-12.0,
            phi_max_value=18.0,
            integrate_selected_qr_rod_value=True,
            mirror_selected_qr_phi_value=False,
            caked_intensity_mode_value="raw_sum",
            rod_profile_intensity_mode_value="raw_sum",
            selected_qr_rod_key_value="phase-a|1",
            qz_min_value=-0.5,
            qz_max_value=1.5,
            delta_qr_value=0.03125,
        ),
        raising=False,
    )
    for name in (
        "tth_min_var",
        "tth_max_var",
        "phi_min_var",
        "phi_max_var",
        "integrate_selected_qr_rod_var",
        "mirror_selected_qr_phi_var",
        "caked_intensity_mode_var",
        "rod_profile_intensity_mode_var",
        "selected_qr_rod_key_var",
        "qz_min_var",
        "qz_max_var",
        "delta_qr_var",
    ):
        monkeypatch.setitem(runtime_session.__dict__, name, None)

    values = runtime_session._current_analysis_range_values()
    roi_values = runtime_session._current_analysis_roi_values()

    assert values == {
        "tth_min": 1.5,
        "tth_max": 55.0,
        "phi_min": -12.0,
        "phi_max": 18.0,
    }
    assert roi_values == {
        "tth_min": 1.5,
        "tth_max": 55.0,
        "phi_min": -12.0,
        "phi_max": 18.0,
        "integrate_selected_qr_rod": True,
        "mirror_selected_qr_phi": False,
        "caked_intensity_mode": "raw_sum",
        "rod_profile_intensity_mode": "raw_sum",
        "selected_qr_rod_key": "phase-a|1",
        "qz_min": -0.5,
        "qz_max": 1.5,
        "delta_qr": 0.03125,
    }


def test_runtime_session_caked_profiles_from_sum_fields_supports_raw_sum_mode() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    res2 = SimpleNamespace(
        intensity=np.array([[10.0, 10.0], [10.0, 10.0]]),
        sum_signal=np.array([[20.0, 40.0], [60.0, 80.0]]),
        sum_normalization=np.array([[2.0, 4.0], [3.0, 8.0]]),
        radial=np.array([10.0, 20.0]),
        azimuthal=np.array([90.0, 270.0]),
    )

    raw_i2t, raw_phi, raw_az, raw_rad = runtime_session._caked_profiles_from_sum_fields(
        res2,
        tth_min=0.0,
        tth_max=90.0,
        phi_min=-180.0,
        phi_max=180.0,
        use_rectangular_roi=False,
        intensity_mode="raw_sum",
    )
    density_i2t, density_phi, density_az, density_rad = (
        runtime_session._caked_profiles_from_sum_fields(
            res2,
            tth_min=0.0,
            tth_max=90.0,
            phi_min=-180.0,
            phi_max=180.0,
            use_rectangular_roi=False,
            intensity_mode="density",
        )
    )

    np.testing.assert_allclose(raw_rad, np.array([10.0, 20.0]))
    np.testing.assert_allclose(raw_az, np.array([-180.0, 0.0]))
    np.testing.assert_allclose(raw_i2t, np.array([80.0, 120.0]))
    np.testing.assert_allclose(raw_phi, np.array([60.0, 140.0]))
    np.testing.assert_allclose(density_rad, raw_rad)
    np.testing.assert_allclose(density_az, raw_az)
    np.testing.assert_allclose(density_i2t, np.array([16.0, 10.0]))
    np.testing.assert_allclose(density_phi, np.array([10.0, 140.0 / 11.0]))


def test_runtime_session_prepare_caked_display_payload_supports_raw_sum_mode() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    res2 = SimpleNamespace(
        intensity=np.array([[10.0, 10.0], [10.0, 10.0]]),
        sum_signal=np.array([[20.0, 40.0], [60.0, 80.0]]),
        sum_normalization=np.array([[2.0, 4.0], [3.0, 8.0]]),
        count=np.ones((2, 2), dtype=np.float64),
        radial=np.array([10.0, 20.0]),
        radial_deg=np.array([10.0, 20.0]),
        azimuthal=np.array([90.0, 270.0]),
        azimuthal_deg=np.array([90.0, 270.0]),
    )

    raw_payload = runtime_session._prepare_caked_display_payload(
        res2,
        detector_shape=(4, 4),
        intensity_mode="raw_sum",
    )
    density_payload = runtime_session._prepare_caked_display_payload(
        res2,
        detector_shape=(4, 4),
        intensity_mode="density",
    )

    assert raw_payload["caked_intensity_mode"] == "raw_sum"
    assert density_payload["caked_intensity_mode"] == "density"
    np.testing.assert_allclose(raw_payload["image"], np.array([[20.0, 40.0], [60.0, 80.0]]))
    np.testing.assert_allclose(density_payload["image"], np.array([[10.0, 10.0], [20.0, 10.0]]))


def test_runtime_session_caked_intensity_raster_signature_includes_mode(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    state = SimpleNamespace(
        last_analysis_cache_sig=(("sim", 1), ("bg", 2)),
        last_caked_intensity_mode="density",
    )
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", state, raising=False)

    density_signature = runtime_session._analysis_view_payload_signature("caked")
    state.last_caked_intensity_mode = "raw_sum"
    raw_signature = runtime_session._analysis_view_payload_signature("caked")

    assert density_signature == ((("sim", 1), ("bg", 2)), "density")
    assert raw_signature == ((("sim", 1), ("bg", 2)), "raw_sum")
    assert density_signature != raw_signature


def test_runtime_session_caked_display_source_signature_changes_with_mode(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    image = np.ones((2, 2), dtype=np.float64)
    state = SimpleNamespace(
        last_analysis_cache_sig=(("sim", 1), ("bg", 2)),
        last_caked_intensity_mode="density",
    )
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", state, raising=False)

    density_signature = runtime_session._analysis_display_raster_source_signature(
        view_mode="caked",
        role="primary",
        scale=1.0,
        display_source=image,
        raw_source=image,
    )
    state.last_caked_intensity_mode = "raw_sum"
    raw_signature = runtime_session._analysis_display_raster_source_signature(
        view_mode="caked",
        role="primary",
        scale=1.0,
        display_source=image,
        raw_source=image,
    )

    assert density_signature == ("caked", "primary", ((("sim", 1), ("bg", 2)), "density"), 1.0)
    assert raw_signature == ("caked", "primary", ((("sim", 1), ("bg", 2)), "raw_sum"), 1.0)
    assert density_signature != raw_signature


def test_runtime_session_caked_intensity_raw_visual_falls_back_to_count_weighted_sum() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    image = np.array([[2.0, 4.0], [6.0, 8.0]])
    payload = {
        "image": image,
        "count": np.array([[1.0, 3.0], [2.0, 4.0]]),
    }

    raw_image = runtime_session._caked_payload_image_for_intensity_mode(
        payload,
        "raw_sum",
    )
    density_image = runtime_session._caked_payload_image_for_intensity_mode(
        payload,
        "density",
    )

    np.testing.assert_allclose(raw_image, np.array([[2.0, 12.0], [12.0, 32.0]]))
    np.testing.assert_allclose(density_image, image)


def test_runtime_session_refresh_integration_repaints_caked_view_when_1d_hidden(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    restore_calls: list[dict[str, object]] = []
    display_calls: list[bool] = []
    primary_calls: list[tuple[object, object]] = []
    projection_calls: list[bool] = []
    main_redraw_calls: list[dict[str, object]] = []
    visuals_calls: list[bool] = []
    redraw_calls: list[bool] = []
    clear_calls: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            ai_cache={},
            last_res2_sim=object(),
            last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_intensity_mode="raw_sum",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(
            show_1d_var=_Var(False),
            show_caked_2d_var=_Var(True),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(visible=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "caked",
    )
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "caked")
    monkeypatch.setattr(
        runtime_session,
        "_current_caked_intensity_mode",
        lambda: "density",
    )
    monkeypatch.setattr(
        runtime_session,
        "_restore_caked_display_payload_from_cached_results",
        lambda **kwargs: restore_calls.append(dict(kwargs)) or True,
    )
    monkeypatch.setattr(
        runtime_session.gui_canvas_interactions,
        "capture_axis_limits",
        lambda axis: "limits",
    )
    monkeypatch.setattr(runtime_session, "ax", object(), raising=False)
    monkeypatch.setitem(
        runtime_session.__dict__,
        "_apply_primary_figure_display_from_cached_results",
        lambda previous, limits: primary_calls.append((previous, limits)) or "caked",
    )
    monkeypatch.setitem(
        runtime_session.__dict__,
        "_apply_current_primary_raster_projection",
        lambda: projection_calls.append(True),
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **kwargs: main_redraw_calls.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_1d_plot_cache_and_lines",
        lambda: clear_calls.append(True),
    )
    monkeypatch.setattr(
        runtime_session,
        "refresh_integration_region_visuals",
        lambda: visuals_calls.append(True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_overlay_canvas_redraw",
        lambda: redraw_calls.append(True),
        raising=False,
    )
    monkeypatch.setitem(
        runtime_session.__dict__,
        "_refresh_display_from_controls",
        lambda: display_calls.append(True),
    )

    assert runtime_session._refresh_integration_from_cached_results() is True
    assert restore_calls == [{"background_visible": False, "q_space_requested": False}]
    assert display_calls == []
    assert primary_calls == [("caked", "limits")]
    assert projection_calls == [True]
    assert main_redraw_calls == [{"force_matplotlib": True}]
    assert clear_calls == [True]
    assert visuals_calls == [True]
    assert redraw_calls == [True]


def test_runtime_session_refresh_integration_keeps_current_caked_display_scale(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    display_refresh_calls: list[bool] = []
    update_1d_calls: list[tuple[object, object]] = []
    visuals_calls: list[bool] = []
    redraw_calls: list[bool] = []
    sim_res2 = object()

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            ai_cache={"ai": object()},
            unscaled_image=np.ones((2, 2), dtype=np.float64),
            last_res2_sim=sim_res2,
            last_res2_background=None,
            last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_background_image_unscaled=None,
            last_caked_intensity_mode="density",
            last_q_space_image_unscaled=None,
            analysis_active_job=None,
            analysis_queued_job=None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(
            show_1d_var=_Var(True),
            show_caked_2d_var=_Var(True),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(visible=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "caked",
    )
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "caked")
    monkeypatch.setattr(
        runtime_session,
        "_current_caked_intensity_mode",
        lambda: "density",
    )
    monkeypatch.setattr(runtime_session, "_analysis_integration_outputs_visible", lambda: True)
    monkeypatch.setattr(
        runtime_session,
        "_refresh_visible_caked_display_from_cached_results",
        lambda: display_refresh_calls.append(True) or True,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_background_backend_for_comparison",
        lambda: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_1d_plots_from_caked",
        lambda sim, bg: update_1d_calls.append((sim, bg)),
    )
    monkeypatch.setattr(
        runtime_session,
        "refresh_integration_region_visuals",
        lambda: visuals_calls.append(True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_overlay_canvas_redraw",
        lambda: redraw_calls.append(True),
        raising=False,
    )

    assert runtime_session._refresh_integration_from_cached_results() is True
    assert display_refresh_calls == []
    assert update_1d_calls == [(sim_res2, None)]
    assert visuals_calls == [True]
    assert redraw_calls == [True]


def test_runtime_session_current_selected_qr_rod_caked_mask_payload_uses_cached_rod_controls(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    signature_calls: list[dict[str, object]] = []
    mask_calls: list[dict[str, object]] = []
    encoded_key = runtime_session.gui_controllers.encode_bragg_qr_group_key(("phase-a", 1))
    monkeypatch.setattr(runtime_session, "_active_caked_primary_view", lambda: True)

    monkeypatch.setattr(
        runtime_session,
        "integration_range_controls_view_state",
        SimpleNamespace(
            phi_min_value=72.5,
            phi_max_value=85.0,
            integrate_selected_qr_rod_value=True,
            mirror_selected_qr_phi_value=True,
            selected_qr_rod_key_value=encoded_key,
            qz_min_value=-1.0,
            qz_max_value=1.0,
            delta_qr_value=0.03125,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_radial_values=np.asarray([10.0, 20.0], dtype=float),
            last_caked_azimuth_values=np.asarray([-5.0, 5.0], dtype=float),
            ai_cache={},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        SimpleNamespace(qr_cylinder_band_cache={"signature": None, "result": None}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "active_qr_cylinder_overlay_entries_factory",
        lambda: [{"key": ("phase-a", 1), "qr": 1.25}],
        raising=False,
    )
    config_values = {"gamma": 1.5}

    def _render_config():
        return runtime_session.gui_qr_cylinder_overlay.build_qr_cylinder_overlay_render_config(
            render_in_caked_space=True,
            image_size=64,
            display_rotate_k=-1,
            center_col=10.0,
            center_row=11.0,
            distance_cor_to_detector=123.0,
            gamma_deg=config_values["gamma"],
            Gamma_deg=2.5,
            chi_deg=3.5,
            psi_deg=4.5,
            psi_z_deg=5.5,
            zs=6.5,
            zb=7.5,
            theta_initial_deg=8.5,
            cor_angle_deg=9.5,
            pixel_size_m=1.0e-4,
            wavelength=1.54,
            n2=1.1 + 0.0j,
        )

    monkeypatch.setattr(
        runtime_session,
        "qr_cylinder_overlay_render_config_factory",
        _render_config,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_qr_cylinder_caked_projection_context",
        lambda: {"projection": "context"},
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "caked_axes_to_qr_qz_maps",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("selected-Qr ROI must use projected cylinder geometry")
        ),
    )

    def _signature(**kwargs):
        signature_calls.append(dict(kwargs))
        return (
            "selected",
            round(float(kwargs["selected_entry"]["qr"]), 10),
            round(float(kwargs["delta_qr"]), 10),
            round(float(kwargs["qz_min"]), 10),
            round(float(kwargs["qz_max"]), 10),
            round(float(kwargs["phi_min"]), 10),
            round(float(kwargs["phi_max"]), 10),
            kwargs["phi_windows"],
            round(float(kwargs["config"].gamma_deg), 10),
        )

    monkeypatch.setattr(
        runtime_session.gui_qr_cylinder_overlay,
        "build_selected_qr_rod_qz_caked_mask_signature",
        _signature,
    )
    monkeypatch.setattr(
        runtime_session.gui_qr_cylinder_overlay,
        "build_selected_qr_rod_qz_caked_mask",
        lambda **kwargs: (
            mask_calls.append(dict(kwargs))
            or {
                "mask": np.asarray([[True, True], [True, False]], dtype=bool),
                "signature": _signature(**kwargs),
            }
        ),
    )
    for name in (
        "phi_min_var",
        "phi_max_var",
        "integrate_selected_qr_rod_var",
        "mirror_selected_qr_phi_var",
        "selected_qr_rod_key_var",
        "qz_min_var",
        "qz_max_var",
        "delta_qr_var",
    ):
        monkeypatch.setitem(runtime_session.__dict__, name, None)

    result = runtime_session._current_selected_qr_rod_caked_mask_payload()
    cached = runtime_session._current_selected_qr_rod_caked_mask_payload()
    config_values["gamma"] = 9.5
    changed = runtime_session._current_selected_qr_rod_caked_mask_payload()

    assert len(mask_calls) == 2
    assert len(signature_calls) >= 3
    assert cached is result
    assert result["selected_qr_rod_key"] == encoded_key
    assert mask_calls[0]["phi_windows"] == ((-85.0, -72.5), (72.5, 85.0))
    np.testing.assert_array_equal(
        result["mask"],
        np.asarray([[True, True], [True, False]], dtype=bool),
    )
    assert changed is not None
    assert changed["signature"] != result["signature"]


def test_runtime_session_current_selected_qr_rod_caked_mask_payload_returns_none_outside_caked_view(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    build_calls: list[str] = []

    monkeypatch.setattr(runtime_session, "_active_caked_primary_view", lambda: False)
    monkeypatch.setattr(
        runtime_session,
        "integration_range_controls_view_state",
        SimpleNamespace(
            integrate_selected_qr_rod_value=True,
            selected_qr_rod_key_value="phase-a|1",
            qz_min_value=-1.0,
            qz_max_value=1.0,
            delta_qr_value=0.03125,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "caked_axes_to_qr_qz_maps",
        lambda *args, **kwargs: build_calls.append("build"),
    )
    for name in (
        "integrate_selected_qr_rod_var",
        "selected_qr_rod_key_var",
        "qz_min_var",
        "qz_max_var",
        "delta_qr_var",
    ):
        monkeypatch.setitem(runtime_session.__dict__, name, None)

    assert runtime_session._current_selected_qr_rod_caked_mask_payload() is None
    assert build_calls == []


def test_runtime_session_selected_qr_rod_1d_uses_detector_qz_profile(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    sim_image = np.asarray([[1.0, 2.0]], dtype=float)
    bg_image = np.asarray([[10.0, 20.0]], dtype=float)
    context = {
        "qz_edges": np.asarray([0.0, 1.0, 2.0], dtype=float),
        "qr_center": 1.0,
        "delta_qr": 0.05,
        "qr_map": np.ones((1, 2), dtype=float),
        "qz_map": np.asarray([[0.5, 1.5]], dtype=float),
        "valid_q": np.ones((1, 2), dtype=bool),
        "detector_phi_deg": np.zeros((1, 2), dtype=float),
        "phi_windows": ((-180.0, 180.0),),
        "signature": ("rod", "detector"),
    }
    integration_calls: list[np.ndarray] = []

    class _Line:
        def __init__(self) -> None:
            self.data = None

        def set_data(self, x, y) -> None:
            self.data = (np.asarray(x), np.asarray(y))

    class _Axis:
        def __init__(self) -> None:
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.visible = True

        def set_xlabel(self, value) -> None:
            self.xlabel = value

        def set_ylabel(self, value) -> None:
            self.ylabel = value

        def set_title(self, value) -> None:
            self.title = value

        def set_yscale(self, _value) -> None:
            pass

        def set_visible(self, value) -> None:
            self.visible = bool(value)

        def get_visible(self) -> bool:
            return self.visible

        def relim(self) -> None:
            pass

        def autoscale_view(self, *args, **kwargs) -> None:
            pass

    def _integrate(**kwargs):
        image = np.asarray(kwargs["detector_image"], dtype=float)
        integration_calls.append(image.copy())
        scale = 1.0 if np.array_equal(image, sim_image) else 10.0
        return {
            "qz_center": np.asarray([0.5, 1.5], dtype=float),
            "pixel_count": np.asarray([1, 2], dtype=np.int64),
            "intensity_sum": np.asarray([2.0, 8.0], dtype=float) * scale,
            "intensity_mean": np.asarray([2.0, 4.0], dtype=float) * scale,
        }

    monkeypatch.setattr(runtime_session, "_ensure_analysis_figure", lambda: None)
    monkeypatch.setattr(
        runtime_session,
        "_clear_analysis_peak_fit_results",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_current_selected_qr_rod_detector_integration_context", lambda: context)
    monkeypatch.setattr(runtime_session, "_selected_qr_rod_detector_mode_requested", lambda: True)
    monkeypatch.setattr(runtime_session, "_current_rod_profile_intensity_mode", lambda: "density")
    monkeypatch.setattr(runtime_session, "_get_scale_factor_value", lambda default=1.0: 1.5)
    monkeypatch.setattr(runtime_session, "_current_background_backend_for_comparison", lambda: bg_image)
    monkeypatch.setattr(
        runtime_session.gui_qr_cylinder_overlay,
        "integrate_detector_qr_rod_qz_profile",
        _integrate,
    )
    monkeypatch.setattr(
        runtime_session,
        "_caked_profiles_from_sum_fields",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("selected-Qr rod numeric profile must not use caked bins")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(unscaled_image=sim_image, last_1d_integration_data={}),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "line_1d_rad", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_rad_bg", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_az", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_az_bg", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_radial", _Axis(), raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_azim", _Axis(), raising=False)
    monkeypatch.setattr(runtime_session, "canvas_1d", None, raising=False)

    runtime_session._update_1d_plots_from_caked(object(), object())

    assert len(integration_calls) == 2
    np.testing.assert_array_equal(integration_calls[0], sim_image)
    np.testing.assert_array_equal(integration_calls[1], bg_image)
    np.testing.assert_allclose(runtime_session.line_1d_rad.data[0], np.asarray([0.5, 1.5]))
    np.testing.assert_allclose(runtime_session.line_1d_rad.data[1], np.asarray([3.0, 6.0]))
    np.testing.assert_allclose(runtime_session.line_1d_rad_bg.data[1], np.asarray([20.0, 40.0]))
    assert runtime_session.line_1d_az.data[0].size == 0
    assert runtime_session.line_1d_az_bg.data[0].size == 0
    assert runtime_session.simulation_runtime_state.last_1d_integration_data["x_axis_kind"] == "qz"
    assert (
        runtime_session.simulation_runtime_state.last_1d_integration_data["x_axis_label"]
        == "Qz (A^-1)"
    )
    assert runtime_session.ax_1d_radial.xlabel == "Qz (A^-1)"
    assert runtime_session.ax_1d_azim.get_visible() is False


def test_runtime_session_selected_qr_rod_1d_prefers_caked_mask_profile(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    sim_res2 = object()
    bg_res2 = object()
    mask_payload = {
        "mask": np.ones((2, 2), dtype=bool),
        "qz_min": 0.0,
        "qz_max": 2.0,
        "signature": ("rod", "caked"),
    }

    class _Line:
        def __init__(self) -> None:
            self.data = None

        def set_data(self, x, y) -> None:
            self.data = (np.asarray(x), np.asarray(y))

    class _Axis:
        def __init__(self) -> None:
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.visible = True

        def set_xlabel(self, value) -> None:
            self.xlabel = value

        def set_ylabel(self, value) -> None:
            self.ylabel = value

        def set_title(self, value) -> None:
            self.title = value

        def set_yscale(self, _value) -> None:
            pass

        def set_visible(self, value) -> None:
            self.visible = bool(value)

        def get_visible(self) -> bool:
            return self.visible

        def relim(self) -> None:
            pass

        def autoscale_view(self, *args, **kwargs) -> None:
            pass

    def _payload(res2):
        image = (
            np.asarray([[2.0, 4.0], [6.0, 8.0]], dtype=float)
            if res2 is sim_res2
            else np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float)
        )
        return {
            "image": image,
            "radial": np.asarray([10.0, 20.0], dtype=float),
            "azimuth": np.asarray([-90.0, 90.0], dtype=float),
        }

    monkeypatch.setattr(runtime_session, "_ensure_analysis_figure", lambda: None)
    monkeypatch.setattr(
        runtime_session,
        "_clear_analysis_peak_fit_results",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_selected_qr_rod_detector_mode_requested", lambda: True)
    monkeypatch.setattr(runtime_session, "_current_selected_qr_rod_caked_mask_payload", lambda: mask_payload)
    monkeypatch.setattr(runtime_session, "_current_selected_qr_rod_detector_integration_context", lambda: (_ for _ in ()).throw(AssertionError("detector fallback should not run")))
    monkeypatch.setattr(runtime_session, "_current_rod_profile_intensity_mode", lambda: "density")
    monkeypatch.setattr(runtime_session, "_get_scale_factor_value", lambda default=1.0: 1.5)
    monkeypatch.setattr(runtime_session, "_caked_profile_payload_for_result", _payload)
    monkeypatch.setattr(
        runtime_session,
        "_caked_qz_map_for_profile_payload",
        lambda _payload: np.asarray([[0.25, 1.25], [0.25, 1.25]], dtype=float),
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(unscaled_image=np.ones((2, 2)), last_1d_integration_data={}),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "line_1d_rad", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_rad_bg", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_az", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_az_bg", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_radial", _Axis(), raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_azim", _Axis(), raising=False)
    monkeypatch.setattr(runtime_session, "canvas_1d", None, raising=False)

    runtime_session._update_1d_plots_from_caked(sim_res2, bg_res2)

    np.testing.assert_allclose(runtime_session.line_1d_rad.data[0], [0.5, 1.5])
    np.testing.assert_allclose(runtime_session.line_1d_rad.data[1], [6.0, 9.0])
    np.testing.assert_allclose(runtime_session.line_1d_rad_bg.data[1], [20.0, 30.0])
    assert runtime_session.line_1d_az.data[0].size == 0
    assert runtime_session.line_1d_az_bg.data[0].size == 0
    assert runtime_session.simulation_runtime_state.last_1d_integration_data["x_axis_kind"] == "qz"
    assert runtime_session.ax_1d_radial.xlabel == "Qz (A^-1)"
    assert runtime_session.ax_1d_azim.get_visible() is False


def test_runtime_session_caked_profiles_fall_back_when_sum_normalization_empty() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    res2 = SimpleNamespace(
        intensity=np.asarray([[2.0, 4.0], [6.0, 8.0]], dtype=float),
        sum_signal=np.zeros((2, 2), dtype=float),
        sum_normalization=np.zeros((2, 2), dtype=float),
        radial=np.asarray([10.0, 20.0], dtype=float),
        azimuthal=np.asarray([90.0, 270.0], dtype=float),
    )

    i2t, i_phi, _az, _rad = runtime_session._caked_profiles_from_sum_fields(
        res2,
        tth_min=0.0,
        tth_max=90.0,
        phi_min=-180.0,
        phi_max=180.0,
        intensity_mode="density",
    )

    np.testing.assert_allclose(i2t, [4.0, 6.0])
    np.testing.assert_allclose(i_phi, [3.0, 7.0])


def test_runtime_session_selected_qr_rod_1d_clear_hides_azimuth_plot(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Line:
        def __init__(self) -> None:
            self.data = None

        def set_data(self, x, y) -> None:
            self.data = (np.asarray(x), np.asarray(y))

    class _Axis:
        def __init__(self) -> None:
            self.visible = True

        def set_xlabel(self, _value) -> None:
            pass

        def set_ylabel(self, _value) -> None:
            pass

        def set_title(self, _value) -> None:
            pass

        def set_visible(self, value) -> None:
            self.visible = bool(value)

        def get_visible(self) -> bool:
            return self.visible

        def relim(self) -> None:
            pass

        def autoscale_view(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(last_1d_integration_data={}),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "line_1d_rad", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_rad_bg", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_az", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_az_bg", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_radial", _Axis(), raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_azim", _Axis(), raising=False)
    monkeypatch.setattr(runtime_session, "canvas_1d", None, raising=False)

    runtime_session._clear_selected_qr_rod_detector_1d_plot()

    assert runtime_session.ax_1d_azim.get_visible() is False
    assert runtime_session.line_1d_az.data[0].size == 0
    assert runtime_session.line_1d_az_bg.data[0].size == 0


def test_runtime_session_caked_profiles_from_sum_fields_standard_update_restores_azimuth_plot(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def get(self) -> float:
            return 0.0

    class _Line:
        def __init__(self) -> None:
            self.data = None

        def set_data(self, x, y) -> None:
            self.data = (np.asarray(x), np.asarray(y))

    class _Axis:
        def __init__(self, *, visible: bool = True) -> None:
            self.visible = bool(visible)
            self.xlabel = None
            self.ylabel = None
            self.title = None

        def set_xlabel(self, value) -> None:
            self.xlabel = value

        def set_ylabel(self, value) -> None:
            self.ylabel = value

        def set_title(self, value) -> None:
            self.title = value

        def set_yscale(self, _value) -> None:
            pass

        def set_visible(self, value) -> None:
            self.visible = bool(value)

        def get_visible(self) -> bool:
            return self.visible

        def relim(self) -> None:
            pass

        def autoscale_view(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(runtime_session, "_ensure_analysis_figure", lambda: None)
    monkeypatch.setattr(
        runtime_session,
        "_clear_analysis_peak_fit_results",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_selected_qr_rod_detector_mode_requested", lambda: False)
    monkeypatch.setattr(runtime_session, "_current_caked_intensity_mode", lambda: "density")
    monkeypatch.setattr(runtime_session, "_get_scale_factor_value", lambda default=1.0: 1.0)
    monkeypatch.setattr(
        runtime_session,
        "_caked_profiles_from_sum_fields",
        lambda *args, **kwargs: (
            np.asarray([10.0, 20.0], dtype=float),
            np.asarray([3.0, 4.0], dtype=float),
            np.asarray([-5.0, 5.0], dtype=float),
            np.asarray([1.0, 2.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(last_1d_integration_data={}),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "tth_min_var", _Var(), raising=False)
    monkeypatch.setattr(runtime_session, "tth_max_var", _Var(), raising=False)
    monkeypatch.setattr(runtime_session, "phi_min_var", _Var(), raising=False)
    monkeypatch.setattr(runtime_session, "phi_max_var", _Var(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_rad", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_rad_bg", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_az", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "line_1d_az_bg", _Line(), raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_radial", _Axis(), raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_azim", _Axis(visible=False), raising=False)
    monkeypatch.setattr(runtime_session, "canvas_1d", None, raising=False)

    runtime_session._update_1d_plots_from_caked(object(), None)

    assert runtime_session.ax_1d_azim.get_visible() is True
    np.testing.assert_allclose(runtime_session.line_1d_az.data[0], [-5.0, 5.0])
    np.testing.assert_allclose(runtime_session.line_1d_az.data[1], [3.0, 4.0])
    assert runtime_session.ax_1d_azim.title == "Azimuthal Integration (φ)"


def test_runtime_session_auto_match_selected_qr_rod_uses_detector_qz_profiles(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    sim_image = np.asarray([[1.0, 2.0]], dtype=float)
    bg_image = np.asarray([[10.0, 20.0]], dtype=float)
    context = {
        "qz_edges": np.asarray([0.0, 1.0, 2.0], dtype=float),
        "qr_center": 1.0,
        "delta_qr": 0.05,
        "qr_map": np.ones((1, 2), dtype=float),
        "qz_map": np.asarray([[0.5, 1.5]], dtype=float),
        "valid_q": np.ones((1, 2), dtype=bool),
        "detector_phi_deg": np.zeros((1, 2), dtype=float),
        "phi_windows": ((-180.0, 180.0),),
        "signature": ("rod", "detector"),
    }
    scales: list[float] = []
    statuses: list[str] = []

    def _integrate(**kwargs):
        image = np.asarray(kwargs["detector_image"], dtype=float)
        if np.array_equal(image, sim_image):
            sums = np.asarray([2.0, 4.0], dtype=float)
        else:
            sums = np.asarray([10.0, 20.0], dtype=float)
        return {
            "qz_center": np.asarray([0.5, 1.5], dtype=float),
            "pixel_count": np.asarray([1, 1], dtype=np.int64),
            "intensity_sum": sums,
            "intensity_mean": sums,
        }

    monkeypatch.setattr(runtime_session, "_current_selected_qr_rod_detector_integration_context", lambda: context)
    monkeypatch.setattr(runtime_session, "_selected_qr_rod_detector_mode_requested", lambda: True)
    monkeypatch.setattr(runtime_session, "_current_rod_profile_intensity_mode", lambda: "raw_sum")
    monkeypatch.setattr(runtime_session, "_current_background_backend_for_comparison", lambda: bg_image)
    monkeypatch.setattr(
        runtime_session.gui_qr_cylinder_overlay,
        "integrate_detector_qr_rod_qz_profile",
        _integrate,
    )
    monkeypatch.setattr(
        runtime_session,
        "_caked_profiles_from_sum_fields",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("selected-Qr rod auto-scale must not use caked bins")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            unscaled_image=sim_image,
            last_1d_integration_data={"simulated_2d_image": sim_image},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label_positions",
        SimpleNamespace(config=lambda **kwargs: statuses.append(kwargs["text"])),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_scale_factor_value",
        lambda value, **_kwargs: scales.append(float(value)),
    )
    monkeypatch.setattr(runtime_session, "apply_scale_factor_to_existing_results", lambda **_kwargs: None)

    runtime_session._auto_match_scale_factor_to_radial_peak()

    assert scales == [5.0]
    assert statuses and "Qz peak" in statuses[-1]
    assert runtime_session.simulation_runtime_state.last_1d_integration_data["x_axis_kind"] == "qz"


def test_runtime_session_sync_selected_qr_rod_controls_state_disables_non_caked_controls(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    encoded_key = runtime_session.gui_controllers.encode_bragg_qr_group_key(("phase-a", 1))

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

        def set(self, value: object) -> None:
            self._value = value

    class _Widget:
        def __init__(self) -> None:
            self.state_value = "normal"
            self.values = []
            self.bounds: dict[str, object] = {}

        def configure(self, **kwargs) -> None:
            self.state_value = str(kwargs.get("state", self.state_value))
            if "values" in kwargs:
                self.values = list(kwargs["values"])
            if "from_" in kwargs:
                self.bounds["from"] = kwargs["from_"]
            if "to" in kwargs:
                self.bounds["to"] = kwargs["to"]

        def config(self, **kwargs) -> None:
            self.configure(**kwargs)

    integrate_widget = _Widget()
    mirror_widget = _Widget()
    combobox = _Widget()
    qz_min_slider = _Widget()
    qz_min_entry = _Widget()
    qz_max_slider = _Widget()
    qz_max_entry = _Widget()
    delta_slider = _Widget()
    delta_entry = _Widget()
    tth_min_slider = _Widget()
    tth_min_entry = _Widget()
    tth_max_slider = _Widget()
    tth_max_entry = _Widget()
    phi_min_slider = _Widget()
    phi_min_entry = _Widget()
    phi_max_slider = _Widget()
    phi_max_entry = _Widget()
    integrate_var = _Var(True)
    monkeypatch.setattr(
        runtime_session,
        "integration_range_controls_view_state",
        SimpleNamespace(
            integrate_selected_qr_rod_var=integrate_var,
            integrate_selected_qr_rod_value=True,
            integrate_selected_qr_rod_checkbutton=integrate_widget,
            mirror_selected_qr_phi_checkbutton=mirror_widget,
            selected_qr_rod_combobox=combobox,
            selected_qr_rod_display_var=_Var(""),
            selected_qr_rod_key_var=_Var(""),
            qz_min_var=_Var(-2.0),
            qz_max_var=_Var(2.0),
            qz_min_value=-2.0,
            qz_max_value=2.0,
            qz_min_slider=qz_min_slider,
            qz_min_entry=qz_min_entry,
            qz_max_slider=qz_max_slider,
            qz_max_entry=qz_max_entry,
            delta_qr_slider=delta_slider,
            delta_qr_entry=delta_entry,
            tth_min_slider=tth_min_slider,
            tth_min_entry=tth_min_entry,
            tth_max_slider=tth_max_slider,
            tth_max_entry=tth_max_entry,
            phi_min_slider=phi_min_slider,
            phi_min_entry=phi_min_entry,
            phi_max_slider=phi_max_slider,
            phi_max_entry=phi_max_entry,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_analysis_selected_qr_rod_entries",
        lambda: [{"key": ("phase-a", 1), "qr": 1.25}],
    )
    monkeypatch.setattr(runtime_session, "_current_caked_qz_extent", lambda: (-0.5, 1.5))
    monkeypatch.setattr(
        runtime_session.gui_integration_range_drag,
        "_sync_runtime_range_text_vars",
        lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(runtime_session, "_active_caked_primary_view", lambda: False)
    runtime_session._sync_selected_qr_rod_controls_state()
    assert integrate_widget.state_value == "disabled"
    assert mirror_widget.state_value == "disabled"
    assert combobox.state_value == "disabled"
    assert qz_min_slider.state_value == "disabled"
    assert delta_entry.state_value == "disabled"
    assert tth_min_slider.state_value == "normal"
    assert phi_max_entry.state_value == "normal"

    integrate_var.set(False)
    monkeypatch.setattr(runtime_session, "_active_caked_primary_view", lambda: True)
    runtime_session._sync_selected_qr_rod_controls_state()
    assert integrate_widget.state_value == "normal"
    assert mirror_widget.state_value == "disabled"
    assert combobox.state_value == "disabled"
    assert qz_min_slider.state_value == "disabled"
    assert delta_entry.state_value == "disabled"
    assert tth_min_slider.state_value == "normal"
    assert (
        runtime_session.integration_range_controls_view_state.selected_qr_rod_key_var.get()
        == encoded_key
    )
    assert qz_min_slider.bounds == {"from": 0.0, "to": 1.5}
    assert qz_max_slider.bounds == {"from": 0.0, "to": 1.5}
    assert runtime_session.integration_range_controls_view_state.qz_min_var.get() == 0.0
    assert runtime_session.integration_range_controls_view_state.qz_max_var.get() == 1.5

    integrate_var.set(True)
    runtime_session._sync_selected_qr_rod_controls_state()
    assert combobox.state_value == "normal"
    assert mirror_widget.state_value == "normal"
    assert qz_min_slider.state_value == "normal"
    assert delta_entry.state_value == "normal"
    assert tth_min_slider.state_value == "disabled"
    assert phi_min_slider.state_value == "normal"
    assert phi_max_entry.state_value == "normal"


def test_runtime_session_current_qr_cylinder_caked_projection_context_prefers_live_bundle_shape(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        def __init__(self):
            self.detector_shape = (9, 7)
            self.raw_azimuth_deg = np.asarray([-90.0, 0.0], dtype=float)

    bundle = FakeBundle()

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(runtime_session, "_current_live_caked_transform_bundle", lambda: bundle)
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_radial_values=np.asarray([10.0, 20.0], dtype=float),
            last_caked_azimuth_values=np.asarray([-5.0, 5.0], dtype=float),
            ai_cache={"ai": None, "detector_shape": (99, 88)},
            unscaled_image=np.ones((4, 4), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_normalize_geometry_fit_caked_view_payload",
        lambda payload, **kwargs: dict(payload),
    )

    context = runtime_session._current_qr_cylinder_caked_projection_context()

    assert context["detector_shape"] == (9, 7)
    np.testing.assert_allclose(context["raw_azimuth_axis"], [-90.0, 0.0])


def test_runtime_impl_keeps_runtime_executors_in_latest_request_wins_mode() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert source.count("max_workers=default_reserved_cpu_worker_count(),") >= 2
    assert "Keep GUI simulation updates in latest-request-wins mode" in source
    assert "Keep GUI analysis updates in latest-request-wins mode" in source
    assert "def _replace_queued_simulation_job(job: dict[str, object]) -> None:" in source
    assert "def _replace_queued_analysis_job(job: dict[str, object]) -> None:" in source
    assert "simulation_runtime_state.worker_ready_result = None" in source
    assert "simulation_runtime_state.analysis_ready_result = None" in source
    assert '"job_requested"' in source
    assert '"job_submitted"' in source
    assert '"job_queued"' in source
    assert '"job_replaced"' in source
    assert '"job_started"' in source
    assert '"job_finished"' in source
    assert '"job_promoted_from_queued"' in source
    assert '"job_discarded_as_stale"' in source


def test_runtime_impl_removes_display_intensity_toggle_from_ui() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "accumulate_intensity_enabled=True" not in source


def test_runtime_impl_gates_1d_analysis_on_analyze_visibility() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _analysis_integration_outputs_visible() -> bool:" in source
    assert "app_shell_view_state.control_tab_var" in source
    assert "gui_views.analysis_popout_window_open(analysis_popout_view_state)" in source
    assert "and _analysis_integration_outputs_visible()" in source
    assert "if not _analysis_integration_outputs_visible():" in source
    assert (
        '_analysis_tab_trace_add("write", _handle_analysis_integration_visibility_change)' in source
    )


def test_runtime_impl_stages_structure_bootstrap_after_first_paint() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "build_lightweight_structure_model_state(" in source
    assert "def _bootstrap_structure_model_state_for_startup() -> None:" in source
    assert 'progress_label.config(text="Initializing structure model...")' in source
    assert "_bootstrap_structure_model_state_for_startup()" in source
    assert "_set_structure_bootstrap_controls_enabled(True)" in source
    assert "root.after_idle(_run_initial_startup_work)" in source


def test_runtime_impl_disables_simulation_tab_during_structure_bootstrap() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index(
        "def _set_structure_bootstrap_controls_enabled(enabled: bool) -> None:"
    )
    helper_end = source.index("def _reset_structure_model_control_vars(", helper_start)
    helper_block = source[helper_start:helper_end]

    assert "app_shell_view_state.simulation_body" in helper_block
    assert "app_shell_view_state.right_col" in helper_block
    assert "app_shell_view_state.fit_body" in helper_block


def test_runtime_impl_keeps_simulation_panel_always_expanded() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    panel_start = source.index("sampling_pruning_frame = CollapsibleFrame(")
    panel_end = source.index("sampling_pruning_frame.pack(", panel_start)
    panel_block = source[panel_start:panel_end]

    assert "expanded=True" in panel_block
    assert "collapsible=False" in panel_block


def test_runtime_impl_builds_shell_before_deferred_runtime_state_load() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    main_start = source.index(
        'def main(write_excel_flag=None, startup_mode="prompt", calibrant_bundle=None):'
    )
    startup_start = source.index("def _run_initial_startup_work():", main_start)
    startup_end = source.index("root.after_idle(_run_initial_startup_work)", startup_start)
    pre_startup_block = source[main_start:startup_start]
    startup_block = source[startup_start:startup_end]

    assert "runtime_context = RuntimeContext(" in pre_startup_block
    assert "runtime_context = build_runtime_window_context(runtime_context)" in pre_startup_block
    assert "build_runtime_state_context()" not in pre_startup_block
    assert "root.deiconify()" in pre_startup_block
    assert "runtime_context = build_runtime_state_context()" in startup_block
    assert "runtime_context = build_runtime_plot_context(runtime_context)" in startup_block
    assert "runtime_context = build_runtime_controls_context(runtime_context)" in startup_block


def test_runtime_impl_lazy_builds_excel_dataframes() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    apply_start = source.index("def _apply_structure_model_runtime_cache_state() -> None:")
    bootstrap_start = source.index("def _bootstrap_structure_model_state_for_startup() -> None:")
    apply_block = source[apply_start:bootstrap_start]
    export_start = source.index("def export_initial_excel():")
    export_end = source.index("app_shell_view_state = app_state.app_shell_view")
    export_block = source[export_start:export_end]

    assert "def _ensure_structure_model_dataframes() -> tuple[object, object]:" in source
    assert "def _build_intensity_dataframes(" in source
    assert "build_intensity_dataframes(" not in apply_block
    assert "df_summary, df_details = _ensure_structure_model_dataframes()" in export_block


def test_runtime_impl_lazy_mounts_analysis_surfaces() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    lazy_block_start = source.index("analysis_surfaces_initialized = False")
    lazy_helper_start = source.index("def _ensure_analysis_figure() -> None:")
    lazy_block = source[lazy_block_start:lazy_helper_start]

    assert "analysis_surfaces_initialized = False" in source
    assert "fig_1d = None" in source
    assert "def _ensure_analysis_figure() -> None:" in source
    assert "def ensure_analysis_surfaces_initialized() -> None:" in source
    assert "def _render_analysis_integration_range_controls(" in source
    peak_tools_start = source.index("def _render_analysis_peak_tools_controls(")
    export_controls_start = source.index("def _render_analysis_export_controls(", peak_tools_start)
    peak_tools_block = source[peak_tools_start:export_controls_start]
    plot_controls_start = source.index("def _render_analysis_plot_controls(")
    restore_axis_start = source.index("def _restore_analysis_peak_axis_view(", plot_controls_start)
    plot_controls_block = source[plot_controls_start:restore_axis_start]
    assert "_show_analysis_tab_lazy_placeholders()" in source
    assert "plt.subplots(2, 1, figsize=(5, 8))" not in lazy_block
    assert "_mount_analysis_figure(app_shell_view_state.plot_frame_1d)" not in source
    assert "_render_analysis_integration_range_controls(" in peak_tools_block
    assert "create_runtime_integration_range_controls(" not in plot_controls_block
    assert "ensure_analysis_surfaces_initialized()" in source


def test_runtime_impl_trims_tools_imports_from_startup_path() -> None:
    runtime_source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    file_parsing_source = FILE_PARSING_SOURCE_PATH.read_text(encoding="utf-8")

    import_block_start = runtime_source.index(
        "from ra_sim.io.file_parsing import parse_poni_file, Open_ASC"
    )
    import_block_end = runtime_source.index(
        "from ra_sim.io.data_loading import (", import_block_start
    )
    import_block = runtime_source[import_block_start:import_block_end]
    dataframe_helper_start = runtime_source.index("def _build_intensity_dataframes(")
    dataframe_helper_end = runtime_source.index(
        "def _current_primary_cif_path()", dataframe_helper_start
    )
    dataframe_helper = runtime_source[dataframe_helper_start:dataframe_helper_end]
    azimuthal_helper_start = runtime_source.index("def _show_azimuthal_radial_plot_demo() -> None:")
    azimuthal_helper_end = runtime_source.index("app_shell_view_state = app_state.app_shell_view")
    azimuthal_helper = runtime_source[azimuthal_helper_start:azimuthal_helper_end]

    assert "from ra_sim.utils.tools import (" not in import_block
    assert "from ra_sim.utils.diffraction_tools import (" in import_block
    assert "build_intensity_dataframes," not in import_block
    assert "view_azimuthal_radial," not in import_block
    assert "detect_blobs," not in import_block
    assert (
        "from ra_sim.utils.diffraction_tools import build_intensity_dataframes" in dataframe_helper
    )
    assert "from ra_sim.utils.diffraction_tools import view_azimuthal_radial" in azimuthal_helper
    assert "from ra_sim.utils.tools import detect_blobs" not in file_parsing_source


def _mapping_field_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return str(node.value)
    return None


def _subscript_field_name(node: ast.Subscript) -> str | None:
    return _mapping_field_name(node.slice)


def _mapping_call_field_name(node: ast.Call) -> tuple[str | None, str | None]:
    if not isinstance(node.func, ast.Attribute) or not node.args:
        return None, None
    return _mapping_field_name(node.args[0]), str(node.func.attr)


def _python_sources(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def test_runtime_session_hkl_pick_cache_rebuilds_when_grouped_row_content_changes() -> None:
    peak_selection = importlib.import_module("ra_sim.gui.peak_selection")
    payload_factory = _load_runtime_session_function(
        "_hkl_pick_simulation_points_from_qr_picker_cache"
    )

    row = {
        "source_table_index": 3,
        "source_row_index": 4,
        "source_branch_index": 0,
        "q_group_key": ("primary", 1, 0, 2),
        "hkl": (1, 0, 2),
        "hkl_raw": (1.0, 0.0, 2.0),
        "native_col": 150.0,
        "native_row": 160.0,
        "sim_col_raw": 50.0,
        "sim_row_raw": 60.0,
        "sim_col": 50.0,
        "sim_row": 60.0,
        "display_col": 50.0,
        "display_row": 60.0,
        "caked_x": 10.0,
        "caked_y": 20.0,
        "raw_caked_x": 10.0,
        "raw_caked_y": 20.0,
        "two_theta_deg": 10.0,
        "phi_deg": 20.0,
        "intensity": 77.0,
    }
    grouped_entries = [row]
    grouped_cache = {
        "signature": ("grouped-cache", 1),
        "grouped_candidates": {"primary": grouped_entries},
    }

    payload_factory.__globals__["gui_peak_selection"] = peak_selection
    payload_factory.__globals__["_current_geometry_fit_params"] = lambda: {"a": 5.0}
    payload_factory.__globals__["_get_geometry_manual_pick_cache"] = lambda **_kwargs: grouped_cache
    payload_factory.__globals__["_hkl_pick_simulation_points_payload_cache"] = {}

    first_payload = payload_factory()

    row["caked_x"] = 30.0
    row["caked_y"] = 40.0
    row["raw_caked_x"] = 30.0
    row["raw_caked_y"] = 40.0
    row["two_theta_deg"] = 30.0
    row["phi_deg"] = 40.0

    second_payload = payload_factory()

    assert first_payload is not second_payload
    assert first_payload["caked_index"] is not second_payload["caked_index"]
    assert tuple(first_payload["caked_index"]["points"]) == ((10.0, 20.0),)
    assert tuple(second_payload["caked_index"]["points"]) == ((30.0, 40.0),)
    assert second_payload["candidates"][0]["caked_x"] == 30.0
    assert second_payload["candidates"][0]["caked_y"] == 40.0
    assert second_payload["source_signature"] != first_payload["source_signature"]


def test_runtime_session_hkl_pick_cache_rebuilds_when_provider_row_content_changes() -> None:
    peak_selection = importlib.import_module("ra_sim.gui.peak_selection")
    payload_factory = _load_runtime_session_function(
        "_hkl_pick_simulation_points_from_qr_picker_cache"
    )

    row = {
        "source_table_index": 3,
        "source_row_index": 4,
        "source_branch_index": 0,
        "q_group_key": ("primary", 1, 0, 2),
        "hkl": (1, 0, 2),
        "hkl_raw": (1.0, 0.0, 2.0),
        "native_col": 150.0,
        "native_row": 160.0,
        "sim_col_raw": 50.0,
        "sim_row_raw": 60.0,
        "sim_col": 50.0,
        "sim_row": 60.0,
        "display_col": 50.0,
        "display_row": 60.0,
        "caked_x": 10.0,
        "caked_y": 20.0,
        "raw_caked_x": 10.0,
        "raw_caked_y": 20.0,
        "two_theta_deg": 10.0,
        "phi_deg": 20.0,
        "intensity": 77.0,
    }
    provider_rows = [row]

    payload_factory.__globals__["gui_peak_selection"] = peak_selection
    payload_factory.__globals__["_current_geometry_fit_params"] = lambda: {"a": 5.0}
    payload_factory.__globals__["_get_geometry_manual_pick_cache"] = lambda **_kwargs: {}
    payload_factory.__globals__["_geometry_manual_simulated_peaks_for_params"] = (
        lambda *_args, **_kwargs: provider_rows
    )
    payload_factory.__globals__["_geometry_manual_pick_candidates"] = lambda raw_rows: {
        ("q_group", "primary", 1, 2): [dict(entry) for entry in raw_rows if isinstance(entry, dict)]
    }
    payload_factory.__globals__["_hkl_pick_simulation_points_payload_cache"] = {}

    first_payload = payload_factory()

    row["caked_x"] = 30.0
    row["caked_y"] = 40.0
    row["raw_caked_x"] = 30.0
    row["raw_caked_y"] = 40.0
    row["two_theta_deg"] = 30.0
    row["phi_deg"] = 40.0

    second_payload = payload_factory()

    assert first_payload is not second_payload
    assert first_payload["caked_index"] is not second_payload["caked_index"]
    assert tuple(first_payload["caked_index"]["points"]) == ((10.0, 20.0),)
    assert tuple(second_payload["caked_index"]["points"]) == ((30.0, 40.0),)
    assert second_payload["candidates"][0]["caked_x"] == 30.0
    assert second_payload["candidates"][0]["caked_y"] == 40.0
    assert second_payload["source_signature"] != first_payload["source_signature"]


def test_runtime_session_hkl_pick_uses_grouped_qr_picker_cache_without_intersection_cache() -> None:
    peak_selection = importlib.import_module("ra_sim.gui.peak_selection")
    payload_factory = _load_runtime_session_function(
        "_hkl_pick_simulation_points_from_qr_picker_cache"
    )
    cache_calls: list[dict[str, object]] = []
    grouped_cache = {
        "signature": ("grouped-cache", 2),
        "grouped_candidates": {
            "primary": [
                {
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "source_branch_index": 0,
                    "q_group_key": ("primary", 1, 0, 2),
                    "hkl": (1, 0, 2),
                    "hkl_raw": (1.0, 0.0, 2.0),
                    "native_col": 150.0,
                    "native_row": 160.0,
                    "sim_col_raw": 50.0,
                    "sim_row_raw": 60.0,
                    "sim_col": 50.0,
                    "sim_row": 60.0,
                    "display_col": 50.0,
                    "display_row": 60.0,
                    "caked_x": 10.0,
                    "caked_y": 20.0,
                    "raw_caked_x": 10.0,
                    "raw_caked_y": 20.0,
                    "two_theta_deg": 10.0,
                    "phi_deg": 20.0,
                    "intensity": 77.0,
                }
            ]
        },
        "simulated_lookup": {("primary", 1, 0, 2): {"peak_count": 1}},
    }

    payload_factory.__globals__["gui_peak_selection"] = peak_selection
    payload_factory.__globals__["_current_geometry_fit_params"] = lambda: {"a": 5.0}
    payload_factory.__globals__["_get_geometry_manual_pick_cache"] = lambda **kwargs: (
        cache_calls.append(dict(kwargs)) or grouped_cache
    )
    payload_factory.__globals__["_geometry_manual_simulated_peaks_for_params"] = (
        lambda *_args, **_kwargs: pytest.fail("provider fallback should stay unused")
    )
    payload_factory.__globals__["_geometry_manual_pick_candidates"] = lambda *_args, **_kwargs: (
        pytest.fail("provider regrouping should stay unused")
    )
    payload_factory.__globals__["_hkl_pick_simulation_points_payload_cache"] = {}

    payload = payload_factory()

    assert cache_calls == [{"param_set": {"a": 5.0}, "prefer_cache": True}]
    assert payload["candidates"]
    assert payload["candidates"][0]["hkl"] == (1, 0, 2)
    assert payload["source_signature"][0] == "grouped"


def test_runtime_session_hkl_pick_builds_grouped_cache_from_stored_raw_peak_rows() -> None:
    peak_selection = importlib.import_module("ra_sim.gui.peak_selection")
    manual_geometry = importlib.import_module("ra_sim.gui.manual_geometry")
    payload_factory = _load_runtime_session_function(
        "_hkl_pick_simulation_points_from_qr_picker_cache"
    )

    primary_row = {
        "source_table_index": 1,
        "source_row_index": 2,
        "source_branch_index": 0,
        "q_group_key": ("primary", 1, 0, 2),
        "hkl": (1, 0, 2),
        "hkl_raw": (1.0, 0.0, 2.0),
        "native_col": 150.0,
        "native_row": 160.0,
        "sim_col_raw": 50.0,
        "sim_row_raw": 60.0,
        "sim_col": 50.0,
        "sim_row": 60.0,
        "display_col": 50.0,
        "display_row": 60.0,
        "caked_x": 10.0,
        "caked_y": 20.0,
        "raw_caked_x": 10.0,
        "raw_caked_y": 20.0,
        "two_theta_deg": 10.0,
        "phi_deg": 20.0,
        "intensity": 77.0,
    }
    secondary_row = {
        "source_table_index": 2,
        "source_row_index": 3,
        "source_branch_index": 0,
        "q_group_key": ("secondary", 0, 1, 1),
        "hkl": (0, 1, 1),
        "hkl_raw": (0.0, 1.0, 1.0),
        "native_col": 170.0,
        "native_row": 180.0,
        "sim_col_raw": 70.0,
        "sim_row_raw": 80.0,
        "sim_col": 70.0,
        "sim_row": 80.0,
        "display_col": 70.0,
        "display_row": 80.0,
        "caked_x": 30.0,
        "caked_y": 40.0,
        "raw_caked_x": 30.0,
        "raw_caked_y": 40.0,
        "two_theta_deg": 30.0,
        "phi_deg": 40.0,
        "intensity": 88.0,
    }
    stored_rows = [dict(primary_row), dict(secondary_row)]
    cache_state = {"signature": None, "data": {}}

    def _replace_cache_state(signature, data) -> None:
        cache_state["signature"] = signature
        cache_state["data"] = dict(data)

    projection_callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.ones((256, 256), dtype=float),
        current_background_native=lambda: np.ones((256, 256), dtype=float),
        image_size=lambda: 256,
        display_to_native_sim_coords=lambda col, row, _shape: (
            float(col),
            float(row),
        ),
        native_sim_to_display_coords=lambda col, row, _shape: (
            float(col),
            float(row),
        ),
        filter_simulated_peaks=lambda rows: (list(rows or []), None, None),
        collapse_simulated_peaks=lambda rows, merge_radius_px=6.0: (
            list(rows or []),
            None,
        ),
        build_live_preview_simulated_peaks_from_cache=lambda: [
            dict(entry) for entry in stored_rows
        ],
    )
    cache_callbacks = manual_geometry.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 1),
        current_background_index=lambda: 0,
        current_background_image=lambda: np.ones((256, 256), dtype=float),
        use_caked_space=projection_callbacks.pick_uses_caked_space,
        replace_cache_state=_replace_cache_state,
        current_geometry_fit_params=lambda: {"a": 5.0},
        pairs_for_index=lambda _idx: [],
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=projection_callbacks.simulated_peaks_for_params,
        build_grouped_candidates=projection_callbacks.pick_candidates,
        build_simulated_lookup=projection_callbacks.simulated_lookup,
        project_peaks_to_current_view=projection_callbacks.project_peaks_to_current_view,
        entry_display_coords=projection_callbacks.entry_display_coords,
        peak_records=lambda: [],
    )

    cache_data = cache_callbacks.get_pick_cache(param_set={"a": 5.0}, prefer_cache=True)

    assert set(cache_data["grouped_candidates"]) == {
        ("primary", 1, 0, 2),
        ("secondary", 0, 1, 1),
    }
    assert set(cache_data["simulated_lookup"]) == {
        ("source_branch", 1, 0),
        ("source_branch", 2, 0),
    }
    assert cache_data["cache_metadata"]["cache_source"] == (
        "geometry_manual_simulated_peaks_for_params(prefer_cache=True)"
    )
    assert cache_state["signature"] == cache_data["signature"]

    payload_factory.__globals__["gui_peak_selection"] = peak_selection
    payload_factory.__globals__["_current_geometry_fit_params"] = lambda: {"a": 5.0}
    payload_factory.__globals__["_get_geometry_manual_pick_cache"] = cache_callbacks.get_pick_cache
    payload_factory.__globals__["_geometry_manual_simulated_peaks_for_params"] = (
        projection_callbacks.simulated_peaks_for_params
    )
    payload_factory.__globals__["_geometry_manual_pick_candidates"] = (
        projection_callbacks.pick_candidates
    )
    payload_factory.__globals__["_hkl_pick_simulation_points_payload_cache"] = {}

    payload = payload_factory()

    assert payload["source_signature"][0] == "grouped"
    assert len(payload["candidates"]) == 2
    assert {tuple(candidate["hkl"]) for candidate in payload["candidates"]} == {
        (1, 0, 2),
        (0, 1, 1),
    }


def _load_runtime_session_function(function_name: str):
    tree = ast.parse(RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {"np": np, "Sequence": Sequence}
            exec(compile(module, str(RUNTIME_SESSION_SOURCE_PATH), "exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"Function {function_name!r} not found in runtime_session.py")


def _raw_field_reads(path: Path, *, field_name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    line_numbers: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            mapped_field, method_name = _mapping_call_field_name(node)
            if mapped_field == field_name and method_name == "get":
                line_numbers.add(int(node.lineno))
        elif isinstance(node, ast.Subscript):
            mapped_field = _subscript_field_name(node)
            if mapped_field == field_name and isinstance(node.ctx, ast.Load):
                line_numbers.add(int(node.lineno))
    return sorted(line_numbers)


def _expr_contains_len_range(expr: ast.AST | None) -> bool:
    if expr is None:
        return False
    for node in ast.walk(expr):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
        ):
            for arg in node.args:
                for nested in ast.walk(arg):
                    if (
                        isinstance(nested, ast.Call)
                        and isinstance(nested.func, ast.Name)
                        and nested.func.id == "len"
                    ):
                        return True
    return False


def _assigned_names(node: ast.AST | None) -> list[str]:
    if isinstance(node, ast.Name):
        return [str(node.id)]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in node.elts:
            names.extend(_assigned_names(element))
        return names
    return []


def _trust_field_assignment_lines(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    field_names = {
        "source_reflection_index",
        "source_reflection_namespace",
        "source_reflection_is_full",
    }
    hits: set[tuple[str, int]] = set()
    pending: list[ast.AST] = [tree]
    while pending:
        node = pending.pop()
        pending.extend(ast.iter_child_nodes(node))
        target_nodes: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            target_nodes = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            target_nodes = [node.target]
        for target in target_nodes:
            if isinstance(target, ast.Subscript):
                mapped_field = _subscript_field_name(target)
                if mapped_field in field_names and isinstance(target.ctx, ast.Store):
                    hits.add((str(mapped_field), int(target.lineno)))
    return sorted(hits, key=lambda item: (item[1], item[0]))


def test_runtime_source_peak_alias_reads_stay_in_boundary_modules() -> None:
    offenders: list[str] = []
    inspected_paths = _python_sources(GUI_SOURCE_ROOT) + [OPTIMIZATION_MOSAIC_PROFILES_SOURCE_PATH]
    for path in inspected_paths:
        if path in RAW_SOURCE_PEAK_READ_ALLOWLIST:
            continue
        line_numbers = _raw_field_reads(path, field_name="source_peak_index")
        if line_numbers:
            rel_path = path.relative_to(REPO_ROOT)
            offenders.append(f"{rel_path}:{','.join(str(line) for line in line_numbers)}")

    assert offenders == []


def test_source_reflection_index_arrays_are_not_minted_from_len_ranges() -> None:
    offenders: list[str] = []
    for path in _python_sources(RA_SIM_SOURCE_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[ast.AST] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            if not _expr_contains_len_range(value):
                continue
            assigned_names = {name for target in targets for name in _assigned_names(target)}
            if not any("source_reflection_indices" in name for name in assigned_names):
                continue
            rel_path = path.relative_to(REPO_ROOT)
            offenders.append(f"{rel_path}:{int(node.lineno)}")

    assert offenders == []


def test_runtime_trust_field_assignments_stay_in_manual_geometry() -> None:
    offenders: list[str] = []
    for path in _python_sources(GUI_SOURCE_ROOT):
        if path in TRUST_FIELD_ASSIGNMENT_ALLOWLIST:
            continue
        hits = _trust_field_assignment_lines(path)
        if hits:
            rel_path = path.relative_to(REPO_ROOT)
            offenders.extend(f"{rel_path}:{line}:{field_name}" for field_name, line in hits)

    assert offenders == []


def test_runtime_session_logged_measurement_match_uses_lookup_point_before_fit() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    picked_frames = [
        {
            "label": "1,0,0",
            "fit": (5.0, 6.0),
            "lookup": (90.0, 91.0),
        },
        {
            "label": "1,0,0",
            "fit": (50.0, 60.0),
            "lookup": (5.0, 6.0),
            "native": (7.0, 8.0),
        },
    ]

    matched = runtime_session._find_picked_frame_for_logged_measurement(
        picked_frames,
        label="1,0,0",
        measured_point=(5.0, 6.0),
    )

    assert matched is picked_frames[1]


def test_runtime_session_builds_logged_lookup_point_from_detector_anchor() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    lookup = runtime_session._build_picked_frame_lookup_point(
        {
            "label": "1,0,0",
            "x": 50.0,
            "y": 60.0,
            "background_detector_x": 5.0,
            "background_detector_y": 6.0,
        }
    )

    assert lookup == pytest.approx((5.0, 6.0))


def test_runtime_session_formats_logged_measurement_context_without_mislabeling_lookup() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    text = runtime_session._format_logged_measurement_context(
        {
            "display": (1.0, 2.0),
            "native": (3.0, 4.0),
            "fit": (50.0, 60.0),
        },
        measured_point=(5.0, 6.0),
    )

    assert "display=(1.000, 2.000)" in text
    assert "native=(3.000, 4.000)" in text
    assert "lookup=(5.000, 6.000)" in text
    assert "fit=(50.000, 60.000)" in text
    assert "fit=(5.000, 6.000)" not in text


def test_runtime_session_formats_unmatched_logged_measurement_as_lookup_only() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    text = runtime_session._format_logged_measurement_context(
        None,
        measured_point=(5.0, 6.0),
    )

    assert text == "lookup=(5.000, 6.000)"


def test_render_analysis_peak_overlays_prefers_x_fit_and_skips_duplicate_groups(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Axis:
        def __init__(self) -> None:
            self.plot_calls: list[tuple[np.ndarray, np.ndarray | None, dict[str, object]]] = []
            self.vline_calls: list[tuple[float, dict[str, object]]] = []
            self.text_calls: list[tuple[float, float, str, dict[str, object]]] = []

        def plot(self, x_values, y_values=None, **kwargs):
            x_arr = np.asarray(x_values, dtype=float)
            y_arr = None if y_values is None else np.asarray(y_values, dtype=float)
            self.plot_calls.append((x_arr, y_arr, dict(kwargs)))
            return [SimpleNamespace(remove=lambda: None)]

        def axvline(self, x_value, **kwargs):
            self.vline_calls.append((float(x_value), dict(kwargs)))
            return SimpleNamespace(remove=lambda: None)

        def text(self, x_value, y_value, text, **kwargs):
            self.text_calls.append((float(x_value), float(y_value), str(text), dict(kwargs)))
            return SimpleNamespace(remove=lambda: None)

    full_x_fit = np.linspace(1.0, 2.0, 25)
    full_y_fit = np.linspace(5.0, 8.0, 25)
    short_x_window = np.linspace(1.3, 1.5, 5)
    radial_axis = _Axis()
    azimuth_axis = _Axis()
    caked_axis = _Axis()

    monkeypatch.setattr(runtime_session, "_resolved_primary_analysis_display_mode", lambda: "caked")
    monkeypatch.setattr(
        runtime_session, "_clear_analysis_peak_overlay_artists", lambda **_kwargs: None
    )
    monkeypatch.setattr(runtime_session, "_analysis_cache_overlay_tables", lambda _show_caked: [])
    monkeypatch.setattr(
        runtime_session,
        "_analysis_curve_data",
        lambda *_args, **_kwargs: (
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            "simulated",
        ),
    )
    monkeypatch.setattr(
        runtime_session, "_request_overlay_canvas_redraw", lambda **_kwargs: None, raising=False
    )
    monkeypatch.setattr(
        runtime_session, "canvas_1d", SimpleNamespace(draw_idle=lambda: None), raising=False
    )
    monkeypatch.setattr(runtime_session, "ax", caked_axis, raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_radial", radial_axis, raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_azim", azimuth_axis, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_ANALYSIS_PEAK_MODEL_COLORS",
        {"gaussian": "#2a9d8f"},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(
            selected_peaks=[{"two_theta_deg": 1.4, "phi_deg": 5.0, "source": "simulated"}],
            caked_peak_artists=[],
            radial_peak_artists=[],
            azimuth_peak_artists=[],
            radial_fit_artists=[],
            azimuth_fit_artists=[],
            radial_fit_results=[
                {
                    "success": True,
                    "model": "gaussian",
                    "fit_group_id": "radial:simulated:gaussian",
                    "plot_fit": True,
                    "x_fit": full_x_fit,
                    "y_fit": full_y_fit,
                    "x_window": short_x_window,
                },
                {
                    "success": True,
                    "model": "gaussian",
                    "fit_group_id": "radial:simulated:gaussian",
                    "plot_fit": False,
                    "x_fit": full_x_fit,
                    "y_fit": full_y_fit,
                    "x_window": short_x_window,
                },
            ],
            azimuth_fit_results=[],
        ),
        raising=False,
    )

    runtime_session._render_analysis_peak_overlays(redraw=False)

    assert len(radial_axis.plot_calls) == 1
    plotted_x, plotted_y, _kwargs = radial_axis.plot_calls[0]
    assert np.allclose(plotted_x, full_x_fit)
    assert np.allclose(plotted_y, full_y_fit)


def test_render_analysis_peak_overlays_breaks_wrapped_fit_curve_gap(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Axis:
        def __init__(self) -> None:
            self.plot_calls: list[tuple[np.ndarray, np.ndarray | None, dict[str, object]]] = []

        def plot(self, x_values, y_values=None, **kwargs):
            x_arr = np.asarray(x_values, dtype=float)
            y_arr = None if y_values is None else np.asarray(y_values, dtype=float)
            self.plot_calls.append((x_arr, y_arr, dict(kwargs)))
            return [SimpleNamespace(remove=lambda: None)]

        def axvline(self, x_value, **kwargs):
            return SimpleNamespace(remove=lambda: None)

        def text(self, x_value, y_value, text, **kwargs):
            return SimpleNamespace(remove=lambda: None)

    wrapped_x_fit = np.concatenate(
        (
            np.linspace(-179.0, -170.0, 8),
            np.linspace(170.0, 179.0, 8),
        )
    )
    wrapped_y_fit = np.linspace(2.0, 5.0, wrapped_x_fit.size)
    radial_axis = _Axis()
    azimuth_axis = _Axis()
    caked_axis = _Axis()

    monkeypatch.setattr(runtime_session, "_resolved_primary_analysis_display_mode", lambda: "caked")
    monkeypatch.setattr(runtime_session, "_analysis_cache_overlay_tables", lambda _show_caked: [])
    monkeypatch.setattr(
        runtime_session,
        "_analysis_curve_data",
        lambda *_args, **_kwargs: (
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            "simulated",
        ),
    )
    monkeypatch.setattr(
        runtime_session, "_request_overlay_canvas_redraw", lambda **_kwargs: None, raising=False
    )
    monkeypatch.setattr(
        runtime_session, "canvas_1d", SimpleNamespace(draw_idle=lambda: None), raising=False
    )
    monkeypatch.setattr(runtime_session, "ax", caked_axis, raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_radial", radial_axis, raising=False)
    monkeypatch.setattr(runtime_session, "ax_1d_azim", azimuth_axis, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_ANALYSIS_PEAK_MODEL_COLORS",
        {"gaussian": "#2a9d8f"},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(
            radial_fit_results=[],
            azimuth_fit_results=[
                {
                    "success": True,
                    "model": "gaussian",
                    "fit_group_id": "azimuth:simulated:gaussian",
                    "plot_fit": True,
                    "x_fit": wrapped_x_fit,
                    "y_fit": wrapped_y_fit,
                }
            ],
        ),
        raising=False,
    )

    runtime_session._render_analysis_peak_overlays(redraw=False)

    assert len(azimuth_axis.plot_calls) == 1
    plotted_x, plotted_y, _kwargs = azimuth_axis.plot_calls[0]
    assert np.count_nonzero(~np.isfinite(plotted_x)) == 1
    assert np.count_nonzero(~np.isfinite(plotted_y)) == 1
    assert np.allclose(plotted_x[np.isfinite(plotted_x)], wrapped_x_fit)
    assert np.allclose(plotted_y[np.isfinite(plotted_y)], wrapped_y_fit)


def test_fit_selected_analysis_peaks_uses_selected_integration_window_only(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

    x_curve = np.linspace(10.0, 12.0, 41)
    y_curve = np.linspace(2.0, 4.0, 41)
    fit_calls: list[dict[str, object]] = []

    def _fit_composite_peak_profile(x_values, y_values, center_guesses, *, model, max_nfev=1000):
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        fit_calls.append(
            {
                "x_values": x_arr,
                "y_values": y_arr,
                "center_guesses": list(center_guesses),
                "model": model,
                "max_nfev": max_nfev,
            }
        )
        return {
            "success": True,
            "model": model,
            "label": "Gaussian",
            "baseline": 0.5,
            "components": [
                {
                    "component_index": 0,
                    "selected_axis_value": 11.2,
                    "amplitude": 3.5,
                    "center": 11.2,
                    "fwhm": 0.3,
                    "sigma": 0.127,
                }
            ],
            "component_groups": [{"component_index": 0, "center_guess_indices": [0]}],
            "rmse": 0.02,
            "rss": 0.0164,
            "x_fit": x_arr,
            "y_fit": y_arr,
            "x_window": x_arr,
            "y_window": y_arr,
            "nfev": 7,
        }

    fake_peak_tools = SimpleNamespace(
        fit_composite_peak_profile=_fit_composite_peak_profile,
        recommended_peak_window_half_width=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Analyze fit must not use local-window width helper.")
        ),
        profile_model_label=lambda model: "Gaussian" if model == "gaussian" else str(model),
    )

    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(
            selected_peaks=[{"two_theta_deg": 11.2, "phi_deg": 5.0, "source": "simulated"}],
            radial_fit_results=[],
            azimuth_fit_results=[],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_tools_view_state",
        SimpleNamespace(
            fit_gaussian_var=_Var(True),
            fit_lorentzian_var=_Var(False),
            fit_pseudo_voigt_var=_Var(False),
            fit_radial_var=_Var(True),
            fit_azimuth_var=_Var(False),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_get_analysis_peak_tools_module", lambda: fake_peak_tools)
    monkeypatch.setattr(
        runtime_session,
        "_analysis_curve_data",
        lambda axis_kind, source_preference, **_kwargs: (
            x_curve.copy(),
            y_curve.copy(),
            "simulated",
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_analysis_peak_axis_value",
        lambda peak_entry, *, axis_kind, axis_values: float(peak_entry["two_theta_deg"]),
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"tth_min": 10.0, "tth_max": 12.0, "phi_min": -10.0, "phi_max": 10.0},
    )
    monkeypatch.setattr(runtime_session, "_clear_analysis_peak_fit_results", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_set_analysis_peak_fit_results_text", lambda _text: None)
    monkeypatch.setattr(runtime_session, "_analysis_peak_fit_results_text", lambda: "")
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "progress_label_positions",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )

    runtime_session._fit_selected_analysis_peaks()

    assert len(fit_calls) == 1
    assert np.allclose(np.asarray(fit_calls[0]["x_values"], dtype=float), x_curve)
    assert fit_calls[0]["center_guesses"] == [11.2]
    assert len(runtime_session.analysis_peak_selection_state.radial_fit_results) == 1
    fit_entry = runtime_session.analysis_peak_selection_state.radial_fit_results[0]
    assert fit_entry["fit_group_id"] == "radial:simulated:gaussian"
    assert fit_entry["plot_fit"] is True
    assert np.all(np.asarray(fit_entry["x_fit"], dtype=float) >= 10.0)
    assert np.all(np.asarray(fit_entry["x_fit"], dtype=float) <= 12.0)


def test_analysis_caked_peak_sources_returns_background_and_simulated_when_both_available(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_radial_values=np.array([10.0, 11.0, 12.0], dtype=float),
            last_caked_azimuth_values=np.array([-5.0, 5.0], dtype=float),
            last_caked_background_image_unscaled=np.ones((2, 3), dtype=float),
            last_caked_image_unscaled=np.full((2, 3), 2.0, dtype=float),
            peak_records=[],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(visible=True),
        raising=False,
    )

    sources = runtime_session._analysis_caked_peak_sources()

    assert [entry["source"] for entry in sources] == ["background", "simulated"]
    assert all("radial_axis" in entry for entry in sources)
    assert all("azimuth_axis" in entry for entry in sources)


def test_analysis_caked_peak_sources_includes_simulated_records_without_caked_image(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_radial_values=np.array([10.0, 11.0, 12.0], dtype=float),
            last_caked_azimuth_values=np.array([-5.0, 5.0], dtype=float),
            last_caked_background_image_unscaled=None,
            last_caked_image_unscaled=None,
            peak_records=[
                {
                    "two_theta_deg": 11.0,
                    "phi_deg": 0.0,
                    "intensity": 7.5,
                    "hkl": (1, 1, 1),
                }
            ],
        ),
        raising=False,
    )

    sources = runtime_session._analysis_caked_peak_sources()

    assert len(sources) == 1
    assert sources[0]["source"] == "simulated"
    assert sources[0].get("image") is None
    assert len(sources[0]["peak_records"]) == 1


def test_find_analysis_peaks_in_selected_box_keeps_background_and_simulated_overlap(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    state = SimpleNamespace(
        selected_peaks=[],
        radial_fit_results=[{"stale": True}],
        azimuth_fit_results=[{"stale": True}],
    )
    status_messages: list[str] = []
    progress_messages: list[str] = []

    def _clear_results(**_kwargs) -> None:
        state.radial_fit_results = []
        state.azimuth_fit_results = []

    monkeypatch.setattr(runtime_session, "analysis_peak_selection_state", state, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_analysis_caked_peak_sources",
        lambda: [
            {"source": "background", "image": np.ones((2, 2), dtype=float)},
            {"source": "simulated", "peak_records": [{"two_theta_deg": 11.0, "phi_deg": 2.0}]},
        ],
    )
    monkeypatch.setattr(
        runtime_session,
        "_discover_background_peaks_in_selected_box",
        lambda _payload, *, max_peaks: [
            {
                "two_theta_deg": 11.0,
                "phi_deg": 2.0,
                "source": "background",
                "raw_two_theta_deg": 11.0,
                "raw_phi_deg": 2.0,
                "discovery_method": "background_local_max",
                "prominence_sigma": 8.0,
                "background_intensity": 25.0,
            }
        ][:max_peaks],
    )
    monkeypatch.setattr(
        runtime_session,
        "_discover_simulated_peaks_in_selected_box",
        lambda _payload, *, max_peaks: [
            {
                "two_theta_deg": 11.01,
                "phi_deg": 2.01,
                "source": "simulated",
                "raw_two_theta_deg": 11.01,
                "raw_phi_deg": 2.01,
                "discovery_method": "simulation_peak_record",
                "intensity": 50.0,
            }
        ][:max_peaks],
    )
    monkeypatch.setattr(
        runtime_session, "_analysis_peak_selection_status_text", lambda: "Selected peaks: 2"
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_analysis_peak_selection_status_text",
        lambda text: status_messages.append(str(text)),
    )
    monkeypatch.setattr(runtime_session, "_clear_analysis_peak_fit_results", _clear_results)
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "progress_label_positions",
        SimpleNamespace(
            config=lambda **kwargs: progress_messages.append(str(kwargs.get("text", "")))
        ),
        raising=False,
    )

    count = runtime_session._find_analysis_peaks_in_selected_box(replace_existing=True)

    assert count == 2
    assert len(state.selected_peaks) == 2
    assert {entry["source"] for entry in state.selected_peaks} == {"background", "simulated"}
    assert state.radial_fit_results == []
    assert state.azimuth_fit_results == []
    assert status_messages == ["Selected peaks: 2"]
    assert progress_messages[-1] == "Found 1 background and 1 simulated peaks in selected box."


def test_find_analysis_peaks_in_selected_box_does_not_include_outside_box_peaks(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    state = SimpleNamespace(selected_peaks=[], radial_fit_results=[], azimuth_fit_results=[])

    monkeypatch.setattr(runtime_session, "analysis_peak_selection_state", state, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"tth_min": 10.5, "tth_max": 11.5, "phi_min": -1.0, "phi_max": 1.0},
    )
    monkeypatch.setattr(
        runtime_session,
        "_analysis_caked_peak_sources",
        lambda: [
            {
                "source": "simulated",
                "radial_axis": np.array([10.0, 11.0, 12.0], dtype=float),
                "azimuth_axis": np.array([-5.0, 0.0, 5.0], dtype=float),
                "peak_records": [
                    {"two_theta_deg": 11.0, "phi_deg": 0.0, "intensity": 5.0},
                    {"two_theta_deg": 12.0, "phi_deg": 0.0, "intensity": 50.0},
                ],
            }
        ],
    )
    monkeypatch.setattr(
        runtime_session,
        "_discover_background_peaks_in_selected_box",
        lambda _payload, *, max_peaks: [],
    )
    monkeypatch.setattr(
        runtime_session, "_analysis_peak_selection_status_text", lambda: "Selected peaks: 1"
    )
    monkeypatch.setattr(
        runtime_session, "_set_analysis_peak_selection_status_text", lambda _text: None
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_analysis_peak_fit_results",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "progress_label_positions",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )

    count = runtime_session._find_analysis_peaks_in_selected_box(replace_existing=True)

    assert count == 1
    assert state.selected_peaks == [
        {
            "two_theta_deg": 11.0,
            "phi_deg": 0.0,
            "source": "simulated",
            "raw_two_theta_deg": 11.0,
            "raw_phi_deg": 0.0,
            "discovery_method": "simulation_peak_record",
            "intensity": 5.0,
        }
    ]


def test_fit_selected_analysis_peaks_fits_both_sources_for_both_axes(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

    curve_calls: list[tuple[str, str, bool]] = []

    def _fit_composite_peak_profile(x_values, y_values, center_guesses, *, model, max_nfev=1000):
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        center = float(center_guesses[0])
        return {
            "success": True,
            "model": model,
            "label": "Gaussian",
            "baseline": 0.5,
            "components": [
                {
                    "component_index": 0,
                    "selected_axis_value": center,
                    "amplitude": 3.0,
                    "center": center,
                    "fwhm": 0.25,
                    "sigma": 0.106,
                }
            ],
            "component_groups": [{"component_index": 0, "center_guess_indices": [0]}],
            "rmse": 0.01,
            "rss": 0.001,
            "x_fit": x_arr,
            "y_fit": y_arr,
            "x_window": x_arr,
            "y_window": y_arr,
            "nfev": min(int(max_nfev), 5),
        }

    fake_peak_tools = SimpleNamespace(
        fit_composite_peak_profile=_fit_composite_peak_profile,
        profile_model_label=lambda model: "Gaussian" if model == "gaussian" else str(model),
        align_angle_to_axis=lambda value, _axis_values: float(value),
    )

    x_radial = np.linspace(10.0, 12.0, 41)
    y_radial = np.linspace(1.0, 4.0, 41)
    x_azimuth = np.linspace(-10.0, 10.0, 41)
    y_azimuth = np.linspace(4.0, 1.0, 41)

    def _curve_data(axis_kind, source_preference, *, allow_fallback=True):
        curve_calls.append((str(axis_kind), str(source_preference), bool(allow_fallback)))
        if str(axis_kind) == "radial":
            return x_radial.copy(), y_radial.copy(), str(source_preference)
        return x_azimuth.copy(), y_azimuth.copy(), str(source_preference)

    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(
            selected_peaks=[
                {"two_theta_deg": 10.8, "phi_deg": -2.0, "source": "background"},
                {"two_theta_deg": 11.2, "phi_deg": 3.0, "source": "simulated"},
            ],
            radial_fit_results=[],
            azimuth_fit_results=[],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_tools_view_state",
        SimpleNamespace(
            fit_gaussian_var=_Var(True),
            fit_lorentzian_var=_Var(False),
            fit_pseudo_voigt_var=_Var(False),
            fit_radial_var=_Var(True),
            fit_azimuth_var=_Var(True),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_get_analysis_peak_tools_module", lambda: fake_peak_tools)
    monkeypatch.setattr(runtime_session, "_analysis_curve_data", _curve_data)
    monkeypatch.setattr(
        runtime_session,
        "_analysis_peak_axis_value",
        lambda peak_entry, *, axis_kind, axis_values: (
            float(peak_entry["two_theta_deg"])
            if str(axis_kind) == "radial"
            else float(peak_entry["phi_deg"])
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"tth_min": 10.0, "tth_max": 12.0, "phi_min": -10.0, "phi_max": 10.0},
    )
    monkeypatch.setattr(runtime_session, "_clear_analysis_peak_fit_results", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_set_analysis_peak_fit_results_text", lambda _text: None)
    monkeypatch.setattr(runtime_session, "_analysis_peak_fit_results_text", lambda: "")
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "progress_label_positions",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )

    runtime_session._fit_selected_analysis_peaks()

    assert set(curve_calls) == {
        ("radial", "background", False),
        ("radial", "simulated", False),
        ("azimuth", "background", False),
        ("azimuth", "simulated", False),
    }
    assert {
        entry["curve_source"]
        for entry in runtime_session.analysis_peak_selection_state.radial_fit_results
        if entry.get("success")
    } == {"background", "simulated"}
    assert {
        entry["curve_source"]
        for entry in runtime_session.analysis_peak_selection_state.azimuth_fit_results
        if entry.get("success")
    } == {"background", "simulated"}


def test_fit_selected_analysis_peaks_does_not_fallback_between_sources(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

    fit_calls: list[str] = []
    curve_calls: list[tuple[str, str, bool]] = []
    fake_peak_tools = SimpleNamespace(
        fit_composite_peak_profile=lambda *_args, **_kwargs: fit_calls.append("fit"),
        profile_model_label=lambda model: "Gaussian" if model == "gaussian" else str(model),
        align_angle_to_axis=lambda value, _axis_values: float(value),
    )

    def _curve_data(axis_kind, source_preference, *, allow_fallback=True):
        curve_calls.append((str(axis_kind), str(source_preference), bool(allow_fallback)))
        if str(source_preference) == "simulated":
            return np.empty((0,), dtype=float), np.empty((0,), dtype=float), "simulated"
        return np.linspace(10.0, 12.0, 41), np.linspace(1.0, 2.0, 41), "background"

    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(
            selected_peaks=[{"two_theta_deg": 11.2, "phi_deg": 5.0, "source": "simulated"}],
            radial_fit_results=[],
            azimuth_fit_results=[],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_tools_view_state",
        SimpleNamespace(
            fit_gaussian_var=_Var(True),
            fit_lorentzian_var=_Var(False),
            fit_pseudo_voigt_var=_Var(False),
            fit_radial_var=_Var(True),
            fit_azimuth_var=_Var(False),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_get_analysis_peak_tools_module", lambda: fake_peak_tools)
    monkeypatch.setattr(runtime_session, "_analysis_curve_data", _curve_data)
    monkeypatch.setattr(
        runtime_session,
        "_analysis_peak_axis_value",
        lambda peak_entry, *, axis_kind, axis_values: float(peak_entry["two_theta_deg"]),
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"tth_min": 10.0, "tth_max": 12.0, "phi_min": -10.0, "phi_max": 10.0},
    )
    monkeypatch.setattr(runtime_session, "_clear_analysis_peak_fit_results", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_set_analysis_peak_fit_results_text", lambda _text: None)
    monkeypatch.setattr(runtime_session, "_analysis_peak_fit_results_text", lambda: "")
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "progress_label_positions",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )

    runtime_session._fit_selected_analysis_peaks()

    assert curve_calls == [("radial", "simulated", False)]
    assert fit_calls == []
    assert len(runtime_session.analysis_peak_selection_state.radial_fit_results) == 1
    failure = runtime_session.analysis_peak_selection_state.radial_fit_results[0]
    assert failure["success"] is False
    assert failure["curve_source"] == "simulated"
    assert "No selected-window 1D data were available for source 'simulated'." in failure["error"]


def test_fit_selected_analysis_peaks_auto_discovers_both_sources_when_none_selected(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

    discovery_calls: list[bool] = []
    curve_calls: list[tuple[str, str, bool]] = []
    state = SimpleNamespace(selected_peaks=[], radial_fit_results=[], azimuth_fit_results=[])

    def _discover(*, replace_existing=True):
        discovery_calls.append(bool(replace_existing))
        state.selected_peaks = [
            {"two_theta_deg": 10.8, "phi_deg": -2.0, "source": "background"},
            {"two_theta_deg": 11.2, "phi_deg": 3.0, "source": "simulated"},
        ]
        return 2

    def _fit_composite_peak_profile(x_values, y_values, center_guesses, *, model, max_nfev=1000):
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        center = float(center_guesses[0])
        return {
            "success": True,
            "model": model,
            "label": "Gaussian",
            "baseline": 0.0,
            "components": [
                {
                    "component_index": 0,
                    "selected_axis_value": center,
                    "amplitude": 3.0,
                    "center": center,
                    "fwhm": 0.2,
                    "sigma": 0.085,
                }
            ],
            "component_groups": [{"component_index": 0, "center_guess_indices": [0]}],
            "rmse": 0.01,
            "rss": 0.001,
            "x_fit": x_arr,
            "y_fit": y_arr,
            "x_window": x_arr,
            "y_window": y_arr,
            "nfev": min(int(max_nfev), 5),
        }

    fake_peak_tools = SimpleNamespace(
        fit_composite_peak_profile=_fit_composite_peak_profile,
        profile_model_label=lambda model: "Gaussian" if model == "gaussian" else str(model),
        align_angle_to_axis=lambda value, _axis_values: float(value),
    )

    def _curve_data(axis_kind, source_preference, *, allow_fallback=True):
        curve_calls.append((str(axis_kind), str(source_preference), bool(allow_fallback)))
        if str(axis_kind) == "radial":
            return np.linspace(10.0, 12.0, 41), np.linspace(1.0, 4.0, 41), str(source_preference)
        return np.linspace(-10.0, 10.0, 41), np.linspace(4.0, 1.0, 41), str(source_preference)

    monkeypatch.setattr(runtime_session, "analysis_peak_selection_state", state, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_tools_view_state",
        SimpleNamespace(
            fit_gaussian_var=_Var(True),
            fit_lorentzian_var=_Var(False),
            fit_pseudo_voigt_var=_Var(False),
            fit_radial_var=_Var(True),
            fit_azimuth_var=_Var(True),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_find_analysis_peaks_in_selected_box", _discover)
    monkeypatch.setattr(runtime_session, "_get_analysis_peak_tools_module", lambda: fake_peak_tools)
    monkeypatch.setattr(runtime_session, "_analysis_curve_data", _curve_data)
    monkeypatch.setattr(
        runtime_session,
        "_analysis_peak_axis_value",
        lambda peak_entry, *, axis_kind, axis_values: (
            float(peak_entry["two_theta_deg"])
            if str(axis_kind) == "radial"
            else float(peak_entry["phi_deg"])
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"tth_min": 10.0, "tth_max": 12.0, "phi_min": -10.0, "phi_max": 10.0},
    )
    monkeypatch.setattr(runtime_session, "_clear_analysis_peak_fit_results", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_set_analysis_peak_fit_results_text", lambda _text: None)
    monkeypatch.setattr(runtime_session, "_analysis_peak_fit_results_text", lambda: "")
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "progress_label_positions",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )

    runtime_session._fit_selected_analysis_peaks()

    assert discovery_calls == [True]
    assert set(curve_calls) == {
        ("radial", "background", False),
        ("radial", "simulated", False),
        ("azimuth", "background", False),
        ("azimuth", "simulated", False),
    }


def test_fit_selected_analysis_peaks_fails_visible_for_unknown_source(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

    fake_peak_tools = SimpleNamespace(
        fit_composite_peak_profile=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unknown-source peaks should not reach composite fitting")
        ),
        profile_model_label=lambda model: "Gaussian" if model == "gaussian" else str(model),
        align_angle_to_axis=lambda value, _axis_values: float(value),
    )

    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(
            selected_peaks=[{"two_theta_deg": 11.2, "phi_deg": 5.0, "source": "mystery"}],
            radial_fit_results=[],
            azimuth_fit_results=[],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_tools_view_state",
        SimpleNamespace(
            fit_gaussian_var=_Var(True),
            fit_lorentzian_var=_Var(False),
            fit_pseudo_voigt_var=_Var(False),
            fit_radial_var=_Var(True),
            fit_azimuth_var=_Var(False),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_get_analysis_peak_tools_module", lambda: fake_peak_tools)
    monkeypatch.setattr(
        runtime_session,
        "_analysis_curve_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unknown-source peaks should not request curve data")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_analysis_peak_axis_value",
        lambda peak_entry, *, axis_kind, axis_values: float(peak_entry["two_theta_deg"]),
    )
    monkeypatch.setattr(runtime_session, "_clear_analysis_peak_fit_results", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_set_analysis_peak_fit_results_text", lambda _text: None)
    monkeypatch.setattr(runtime_session, "_analysis_peak_fit_results_text", lambda: "")
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "progress_label_positions",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )

    runtime_session._fit_selected_analysis_peaks()

    assert len(runtime_session.analysis_peak_selection_state.radial_fit_results) == 1
    failure = runtime_session.analysis_peak_selection_state.radial_fit_results[0]
    assert failure["success"] is False
    assert failure["source"] == "unknown"
    assert failure["fit_group_id"] == "radial:unknown:gaussian"
    assert "could not be resolved" in failure["error"]


def test_fit_group_ids_include_source(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

    def _fit_composite_peak_profile(x_values, y_values, center_guesses, *, model, max_nfev=1000):
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        center = float(center_guesses[0])
        return {
            "success": True,
            "model": model,
            "label": "Gaussian",
            "baseline": 0.0,
            "components": [
                {
                    "component_index": 0,
                    "selected_axis_value": center,
                    "amplitude": 2.0,
                    "center": center,
                    "fwhm": 0.2,
                    "sigma": 0.085,
                }
            ],
            "component_groups": [{"component_index": 0, "center_guess_indices": [0]}],
            "rmse": 0.01,
            "rss": 0.001,
            "x_fit": x_arr,
            "y_fit": y_arr,
            "x_window": x_arr,
            "y_window": y_arr,
            "nfev": min(int(max_nfev), 5),
        }

    fake_peak_tools = SimpleNamespace(
        fit_composite_peak_profile=_fit_composite_peak_profile,
        profile_model_label=lambda model: "Gaussian" if model == "gaussian" else str(model),
        align_angle_to_axis=lambda value, _axis_values: float(value),
    )

    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(
            selected_peaks=[
                {"two_theta_deg": 10.8, "phi_deg": -2.0, "source": "background"},
                {"two_theta_deg": 11.2, "phi_deg": 3.0, "source": "simulated"},
            ],
            radial_fit_results=[],
            azimuth_fit_results=[],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_tools_view_state",
        SimpleNamespace(
            fit_gaussian_var=_Var(True),
            fit_lorentzian_var=_Var(False),
            fit_pseudo_voigt_var=_Var(False),
            fit_radial_var=_Var(True),
            fit_azimuth_var=_Var(True),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_get_analysis_peak_tools_module", lambda: fake_peak_tools)
    monkeypatch.setattr(
        runtime_session,
        "_analysis_curve_data",
        lambda axis_kind, source_preference, **_kwargs: (
            (np.linspace(10.0, 12.0, 41), np.linspace(1.0, 4.0, 41), str(source_preference))
            if str(axis_kind) == "radial"
            else (np.linspace(-10.0, 10.0, 41), np.linspace(4.0, 1.0, 41), str(source_preference))
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_analysis_peak_axis_value",
        lambda peak_entry, *, axis_kind, axis_values: (
            float(peak_entry["two_theta_deg"])
            if str(axis_kind) == "radial"
            else float(peak_entry["phi_deg"])
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"tth_min": 10.0, "tth_max": 12.0, "phi_min": -10.0, "phi_max": 10.0},
    )
    monkeypatch.setattr(runtime_session, "_clear_analysis_peak_fit_results", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_set_analysis_peak_fit_results_text", lambda _text: None)
    monkeypatch.setattr(runtime_session, "_analysis_peak_fit_results_text", lambda: "")
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "progress_label_positions",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )

    runtime_session._fit_selected_analysis_peaks()

    fit_group_ids = {
        entry["fit_group_id"]
        for entry in (
            list(runtime_session.analysis_peak_selection_state.radial_fit_results)
            + list(runtime_session.analysis_peak_selection_state.azimuth_fit_results)
        )
    }
    assert {
        "radial:background:gaussian",
        "radial:simulated:gaussian",
        "azimuth:background:gaussian",
        "azimuth:simulated:gaussian",
    } <= fit_group_ids


def test_analysis_curve_selected_window_returns_empty_when_selection_excludes_curve(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setitem(
        runtime_session._analysis_curve_selected_window.__globals__,
        "_current_analysis_range_values",
        lambda: {
            "tth_min": 20.0,
            "tth_max": 22.0,
            "phi_min": -30.0,
            "phi_max": 30.0,
        },
    )
    monkeypatch.setitem(
        runtime_session._analysis_curve_selected_window.__globals__,
        "_get_analysis_peak_tools_module",
        lambda: SimpleNamespace(
            align_angle_to_axis=lambda value, axis_values: float(value),
        ),
    )

    x_values = np.linspace(10.0, 12.0, 21)
    y_values = np.linspace(1.0, 2.0, 21)

    selected_x, selected_y = runtime_session._analysis_curve_selected_window(
        "radial",
        x_values,
        y_values,
    )

    assert selected_x.size == 0
    assert selected_y.size == 0


def test_clear_selected_analysis_peaks_clears_stale_fit_results(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Artist:
        def __init__(self, label: str) -> None:
            self.label = label
            self.removed = False

        def remove(self) -> None:
            self.removed = True

    caked_artist = _Artist("caked")
    radial_peak_artist = _Artist("radial-peak")
    azimuth_peak_artist = _Artist("azimuth-peak")
    radial_fit_artist = _Artist("radial-fit")
    azimuth_fit_artist = _Artist("azimuth-fit")
    fit_results_text: list[str] = []
    selection_status_text: list[str] = []
    overlay_redraw_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(
            selected_peaks=[{"two_theta_deg": 11.2, "phi_deg": 5.0, "source": "simulated"}],
            radial_fit_results=[{"success": True, "fit_group_id": "radial:simulated:gaussian"}],
            azimuth_fit_results=[{"success": True, "fit_group_id": "azimuth:simulated:gaussian"}],
            caked_peak_artists=[caked_artist],
            radial_peak_artists=[radial_peak_artist],
            azimuth_peak_artists=[azimuth_peak_artist],
            radial_fit_artists=[radial_fit_artist],
            azimuth_fit_artists=[azimuth_fit_artist],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_analysis_peak_fit_results_text",
        lambda text: fit_results_text.append(str(text)),
    )
    monkeypatch.setattr(
        runtime_session,
        "_analysis_peak_selection_status_text",
        lambda: "No analysis peaks selected.",
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_analysis_peak_selection_status_text",
        lambda text: selection_status_text.append(str(text)),
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "caked",
    )
    monkeypatch.setattr(
        runtime_session,
        "_analysis_cache_overlay_tables",
        lambda _show_caked: [],
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_overlay_canvas_redraw",
        lambda **kwargs: overlay_redraw_calls.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "canvas_1d",
        SimpleNamespace(draw_idle=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label_positions",
        SimpleNamespace(config=lambda **_kwargs: None),
        raising=False,
    )

    runtime_session._clear_selected_analysis_peaks()

    assert runtime_session.analysis_peak_selection_state.selected_peaks == []
    assert runtime_session.analysis_peak_selection_state.radial_fit_results == []
    assert runtime_session.analysis_peak_selection_state.azimuth_fit_results == []
    assert runtime_session.analysis_peak_selection_state.caked_peak_artists == []
    assert runtime_session.analysis_peak_selection_state.radial_peak_artists == []
    assert runtime_session.analysis_peak_selection_state.azimuth_peak_artists == []
    assert runtime_session.analysis_peak_selection_state.radial_fit_artists == []
    assert runtime_session.analysis_peak_selection_state.azimuth_fit_artists == []
    assert caked_artist.removed is True
    assert radial_peak_artist.removed is True
    assert azimuth_peak_artist.removed is True
    assert radial_fit_artist.removed is True
    assert azimuth_fit_artist.removed is True
    assert fit_results_text[-1] == "Fit results will appear here."
    assert selection_status_text[-1] == "No analysis peaks selected."
    assert overlay_redraw_calls == [{"force": True}]
