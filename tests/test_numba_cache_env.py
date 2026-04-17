import importlib
import importlib.abc
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _snapshot_ra_sim_modules() -> dict[str, Any]:
    return {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "ra_sim" or name.startswith("ra_sim.")
    }


def _clear_ra_sim_modules() -> None:
    for name in list(sys.modules):
        if name == "ra_sim" or name.startswith("ra_sim."):
            sys.modules.pop(name, None)


def _restore_modules(modules: dict[str, Any]) -> None:
    for name in list(sys.modules):
        if name == "ra_sim" or name.startswith("ra_sim."):
            sys.modules.pop(name, None)
    sys.modules.update(modules)


def _fresh_import_and_restore(module_name: str) -> None:
    previous = _snapshot_ra_sim_modules()
    try:
        _clear_ra_sim_modules()
        importlib.import_module(module_name)
    finally:
        _restore_modules(previous)


def _default_numba_cache_path(home: str) -> str:
    return str((Path(home) / ".cache" / "ra_sim" / "numba").resolve())


_CLI_HEAVY_PREFIXES = (
    "ra_sim.headless_geometry_fit",
    "ra_sim.gui",
    "ra_sim.fitting.optimization",
    "ra_sim.utils.stacking_fault",
    "ra_sim.utils.diffraction_tools",
    "ra_sim.utils.calculations",
    "ra_sim.utils.tools",
    "ra_sim.simulation",
)


def _assert_cli_heavy_modules_unloaded() -> None:
    for prefix in _CLI_HEAVY_PREFIXES:
        assert prefix not in sys.modules
        assert not any(name.startswith(prefix + ".") for name in sys.modules)


_HEADLESS_LAZY_PREFIXES = (
    "ra_sim.gui",
    "ra_sim.fitting.optimization",
    "ra_sim.simulation",
    "ra_sim.utils.stacking_fault",
    "ra_sim.utils.diffraction_tools",
    "ra_sim.utils.tools",
)


def _assert_module_prefix_absent(prefix: str) -> None:
    assert prefix not in sys.modules
    assert not any(name.startswith(prefix + ".") for name in sys.modules)


def _assert_module_prefix_present(prefix: str) -> None:
    assert prefix in sys.modules or any(name.startswith(prefix + ".") for name in sys.modules)


def test_numba_cache_dir_defaults_to_stable_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    _fresh_import_and_restore("ra_sim")

    expected = _default_numba_cache_path(str(tmp_path))
    assert os.environ["NUMBA_CACHE_DIR"] == expected


def test_numba_cache_dir_preserves_existing_env(monkeypatch, tmp_path):
    existing = str(tmp_path / "existing-numba-cache")
    monkeypatch.setenv("NUMBA_CACHE_DIR", existing)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "unused")

    _fresh_import_and_restore("ra_sim")

    assert os.environ["NUMBA_CACHE_DIR"] == existing


def test_numba_cache_dir_bootstraps_before_main_entrypoint_cli_forward(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    previous = _snapshot_ra_sim_modules()

    try:
        _clear_ra_sim_modules()
        importlib.import_module("ra_sim.__main__")
        expected = _default_numba_cache_path(str(tmp_path))
        assert os.environ["NUMBA_CACHE_DIR"] == expected
        assert "ra_sim.cli" not in sys.modules
    finally:
        _restore_modules(previous)


def test_numba_cache_dir_present_before_cli_jit_modules_import(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    previous = _snapshot_ra_sim_modules()

    try:
        _clear_ra_sim_modules()
        importlib.import_module("ra_sim.cli")
        expected = _default_numba_cache_path(str(tmp_path))
        assert os.environ["NUMBA_CACHE_DIR"] == expected
    finally:
        _restore_modules(previous)


def test_cli_import_keeps_geometry_and_simulation_stack_lazy(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    previous = _snapshot_ra_sim_modules()

    try:
        _clear_ra_sim_modules()
        cli = importlib.import_module("ra_sim.cli")
        expected = _default_numba_cache_path(str(tmp_path))
        assert os.environ["NUMBA_CACHE_DIR"] == expected
        _assert_cli_heavy_modules_unloaded()

        a_val, c_val = cli._parse_cif_cell_a_c(str(Path("tests/local_test.cif")))

        assert a_val == 4.0
        assert c_val == 10.0
        _assert_cli_heavy_modules_unloaded()
    finally:
        _restore_modules(previous)


def test_headless_geometry_fit_import_keeps_heavy_runtime_modules_lazy(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    previous = _snapshot_ra_sim_modules()

    try:
        _clear_ra_sim_modules()
        headless_geometry_fit = importlib.import_module("ra_sim.headless_geometry_fit")

        for prefix in _HEADLESS_LAZY_PREFIXES:
            _assert_module_prefix_absent(prefix)

        assert headless_geometry_fit.gui_background.__name__ == "ra_sim.gui.background"
        _assert_module_prefix_present("ra_sim.gui")
        for prefix in _HEADLESS_LAZY_PREFIXES:
            if prefix != "ra_sim.gui":
                _assert_module_prefix_absent(prefix)

        assert (
            headless_geometry_fit._load_simulation_diffraction().__name__
            == "ra_sim.simulation.diffraction"
        )
        _assert_module_prefix_present("ra_sim.simulation")
        for prefix in (
            "ra_sim.fitting.optimization",
            "ra_sim.utils.stacking_fault",
            "ra_sim.utils.diffraction_tools",
            "ra_sim.utils.tools",
        ):
            _assert_module_prefix_absent(prefix)

        assert headless_geometry_fit._load_stacking_fault_runtime().__name__ == (
            "ra_sim.utils.stacking_fault"
        )
        _assert_module_prefix_present("ra_sim.utils.stacking_fault")
        for prefix in (
            "ra_sim.fitting.optimization",
            "ra_sim.utils.diffraction_tools",
            "ra_sim.utils.tools",
        ):
            _assert_module_prefix_absent(prefix)

        assert headless_geometry_fit._load_diffraction_tools().__name__ == (
            "ra_sim.utils.diffraction_tools"
        )
        _assert_module_prefix_present("ra_sim.utils.diffraction_tools")
        _assert_module_prefix_absent("ra_sim.fitting.optimization")
        _assert_module_prefix_absent("ra_sim.utils.tools")

        assert headless_geometry_fit._load_tools_runtime().__name__ == "ra_sim.utils.tools"
        _assert_module_prefix_present("ra_sim.utils.tools")
        _assert_module_prefix_absent("ra_sim.fitting.optimization")

        assert (
            headless_geometry_fit._load_fitting_runtime().__name__ == "ra_sim.fitting.optimization"
        )
        _assert_module_prefix_present("ra_sim.fitting.optimization")
    finally:
        _restore_modules(previous)


def test_headless_geometry_fit_public_gui_proxies_follow_current_loader(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    previous = _snapshot_ra_sim_modules()

    try:
        _clear_ra_sim_modules()
        headless_geometry_fit = importlib.import_module("ra_sim.headless_geometry_fit")
        original_loader = headless_geometry_fit._load_gui_modules

        for prefix in _HEADLESS_LAZY_PREFIXES:
            _assert_module_prefix_absent(prefix)

        synthetic_background = SimpleNamespace(
            __name__="synthetic.gui.background",
            sentinel="synthetic-background",
        )
        monkeypatch.setattr(
            headless_geometry_fit,
            "_load_gui_modules",
            lambda: SimpleNamespace(gui_background=synthetic_background),
        )

        assert original_loader is not None
        assert headless_geometry_fit.gui_background.__name__ == "synthetic.gui.background"
        assert headless_geometry_fit.gui_background.sentinel == "synthetic-background"
        for prefix in _HEADLESS_LAZY_PREFIXES:
            _assert_module_prefix_absent(prefix)
    finally:
        _restore_modules(previous)


def test_runtime_session_import_stays_helper_only_without_optional_cif_deps(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    class _BlockCifFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "CifFile":
                raise ModuleNotFoundError("No module named 'CifFile'")
            if fullname == "Dans_Diffraction":
                raise ModuleNotFoundError("No module named 'Dans_Diffraction'")
            return None

    previous = _snapshot_ra_sim_modules()
    finder = _BlockCifFinder()
    cif_path = Path(__file__).resolve().parent / "local_test.cif"

    try:
        _clear_ra_sim_modules()
        sys.modules.pop("CifFile", None)
        sys.modules.pop("Dans_Diffraction", None)
        sys.meta_path.insert(0, finder)

        runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

        assert runtime_session._STRUCTURE_MODEL_MODULE is None
        assert "ra_sim.gui.structure_model" not in sys.modules
        assert runtime_session.parse_cif_num("2.5(1)") == 2.5
        assert runtime_session._normalize_occupancy_label("", 0) == "site_1"
        labels, expanded_map = runtime_session._extract_occupancy_site_metadata(
            {"_atom_site_label": ["I1", "Nb1", "I1"]},
            "unused.cif",
        )
        assert labels == ["I1", "Nb1"]
        assert expanded_map == []
        rows = runtime_session._extract_atom_site_fractional_metadata(
            {
                "_atom_site_label": ["I", "I"],
                "_atom_site_fract_x": ["1/2", "0.125(2)"],
                "_atom_site_fract_y": ["0.25", "0.5"],
                "_atom_site_fract_z": ["0.0", "1/4"],
            }
        )
        assert [row["label"] for row in rows] == ["I #1", "I #2"]
        assert rows[0]["x"] == 0.5
        assert rows[1]["z"] == 0.25
        assert runtime_session._STRUCTURE_MODEL_MODULE is not None
        assert "ra_sim.gui.structure_model" in sys.modules

        with pytest.raises(ModuleNotFoundError, match="CifFile"):
            runtime_session._read_cif_block(str(cif_path))

        assert runtime_session._STRUCTURE_MODEL_MODULE is not None
        assert "ra_sim.gui.structure_model" in sys.modules
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        _restore_modules(previous)


def test_runtime_session_numba_cache_heuristic_uses_warm_marker_without_scanning(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    previous = _snapshot_ra_sim_modules()

    try:
        _clear_ra_sim_modules()
        runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
        marker_path = runtime_session._numba_cache_warm_marker_path()

        monkeypatch.setattr(
            runtime_session,
            "_NUMBA_CACHE_HAS_COMPILED_ARTIFACTS",
            None,
            raising=False,
        )
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.touch()

        def _unexpected_rglob(self, pattern):
            raise AssertionError(f"unexpected recursive scan: {self} {pattern}")

        monkeypatch.setattr(Path, "rglob", _unexpected_rglob, raising=True)

        assert runtime_session._numba_cache_contains_compiled_artifacts() is True
    finally:
        _restore_modules(previous)


def test_runtime_session_numba_cache_heuristic_returns_false_without_marker_or_scan(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    previous = _snapshot_ra_sim_modules()

    try:
        _clear_ra_sim_modules()
        runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
        marker_path = runtime_session._numba_cache_warm_marker_path()

        if marker_path.exists():
            marker_path.unlink()
        monkeypatch.setattr(
            runtime_session,
            "_NUMBA_CACHE_HAS_COMPILED_ARTIFACTS",
            None,
            raising=False,
        )

        def _unexpected_rglob(self, pattern):
            raise AssertionError(f"unexpected recursive scan: {self} {pattern}")

        monkeypatch.setattr(Path, "rglob", _unexpected_rglob, raising=True)

        assert runtime_session._numba_cache_contains_compiled_artifacts() is False
        assert not marker_path.exists()
    finally:
        _restore_modules(previous)


def test_runtime_session_numba_cache_heuristic_marks_warm_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    previous = _snapshot_ra_sim_modules()

    try:
        _clear_ra_sim_modules()
        runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
        marker_path = runtime_session._numba_cache_warm_marker_path()

        monkeypatch.setattr(
            runtime_session,
            "_NUMBA_CACHE_HAS_COMPILED_ARTIFACTS",
            None,
            raising=False,
        )
        runtime_session._mark_numba_cache_compiled_artifacts_available()
        assert marker_path.exists()
        assert runtime_session._NUMBA_CACHE_HAS_COMPILED_ARTIFACTS is True
    finally:
        _restore_modules(previous)


@pytest.mark.parametrize(
    ("compiled_artifacts_available", "expected_mark_calls"),
    [
        (False, []),
        (True, ["mark"]),
    ],
)
def test_runtime_session_apply_ready_result_marks_warm_cache_only_for_compiled_runs(
    monkeypatch,
    tmp_path,
    compiled_artifacts_available,
    expected_mark_calls,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    previous = _snapshot_ra_sim_modules()

    try:
        _clear_ra_sim_modules()
        runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
        mark_calls: list[str] = []

        monkeypatch.setattr(
            runtime_session,
            "_copy_intersection_cache_tables",
            lambda tables: list(tables or []),
            raising=False,
        )
        monkeypatch.setattr(
            runtime_session,
            "_resolved_peak_table_payload",
            lambda cache_tables, fallback_tables: list(cache_tables or fallback_tables or []),
            raising=False,
        )
        monkeypatch.setattr(
            runtime_session.gui_geometry_q_group_manager,
            "audited_full_order_source_reflection_index_groups",
            lambda *_args, **_kwargs: ([], []),
            raising=False,
        )
        monkeypatch.setattr(
            runtime_session,
            "_store_primary_cache_payload",
            lambda **_kwargs: None,
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
            "_mark_numba_cache_compiled_artifacts_available",
            lambda: mark_calls.append("mark"),
            raising=False,
        )

        runtime_session._apply_ready_simulation_result(
            {
                "primary_image": np.zeros((2, 2), dtype=np.float64),
                "secondary_image": np.zeros((2, 2), dtype=np.float64),
                "primary_intersection_cache": [],
                "secondary_intersection_cache": [],
                "primary_max_positions": [],
                "secondary_max_positions": [],
                "primary_peak_table_lattice": [],
                "secondary_peak_table_lattice": [],
                "image_generation_elapsed_ms": 1.25,
                "numba_cache_compiled_artifacts_available": compiled_artifacts_available,
            }
        )

        assert mark_calls == expected_mark_calls
    finally:
        _restore_modules(previous)


def test_structure_model_import_keeps_pure_helpers_available_without_optional_cif_deps(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    class _BlockCifFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "CifFile":
                raise ModuleNotFoundError("No module named 'CifFile'")
            if fullname == "Dans_Diffraction":
                raise ModuleNotFoundError("No module named 'Dans_Diffraction'")
            return None

    previous = _snapshot_ra_sim_modules()
    finder = _BlockCifFinder()
    cif_path = Path(__file__).resolve().parent / "local_test.cif"

    try:
        _clear_ra_sim_modules()
        sys.modules.pop("CifFile", None)
        sys.modules.pop("Dans_Diffraction", None)
        sys.meta_path.insert(0, finder)

        structure_model = importlib.import_module("ra_sim.gui.structure_model")

        assert structure_model.parse_cif_num("2.5(1)") == 2.5
        labels, expanded_map = structure_model.extract_occupancy_site_metadata(
            {"_atom_site_label": ["I1", "Nb1", "I1"]},
            "unused.cif",
        )
        assert labels == ["I1", "Nb1"]
        assert expanded_map == []
        rows = structure_model.extract_atom_site_fractional_metadata(
            {
                "_atom_site_label": ["I", "I"],
                "_atom_site_fract_x": ["1/2", "0.125(2)"],
                "_atom_site_fract_y": ["0.25", "0.5"],
                "_atom_site_fract_z": ["0.0", "1/4"],
            }
        )
        assert [row["label"] for row in rows] == ["I #1", "I #2"]
        assert rows[0]["x"] == 0.5
        assert rows[1]["z"] == 0.25

        with pytest.raises(ModuleNotFoundError, match="Dans_Diffraction"):
            structure_model.extract_occupancy_site_metadata(
                {"_atom_site_label": ["I1", "Nb1", "I1"]},
                "unused.cif",
                expand_structure=True,
            )

        with pytest.raises(ModuleNotFoundError, match="CifFile"):
            structure_model._read_cif_block(str(cif_path))
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        _restore_modules(previous)


def test_calculations_optional_optics_imports_retry_after_late_dependency_availability(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    class _BlockOptionalOpticsFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "Dans_Diffraction":
                raise ModuleNotFoundError("No module named 'Dans_Diffraction'")
            if fullname == "xraydb":
                raise ModuleNotFoundError("No module named 'xraydb'")
            return None

    previous = _snapshot_ra_sim_modules()
    finder = _BlockOptionalOpticsFinder()

    try:
        _clear_ra_sim_modules()
        sys.modules.pop("Dans_Diffraction", None)
        sys.modules.pop("xraydb", None)
        sys.meta_path.insert(0, finder)

        calculations = importlib.import_module("ra_sim.utils.calculations")

        assert "Dans_Diffraction" not in sys.modules
        assert "xraydb" not in sys.modules
        assert calculations.d_spacing(1, 0, 0, 2.0, 3.0) == 1.7320508075688774
        assert calculations._get_dans_diffraction_module() is None
        assert calculations._DANS_DIFFRACTION_MODULE is calculations._OPTIONAL_IMPORT_UNSET
        assert calculations._get_xraydb_module() is None
        assert calculations._XRAYDB_MODULE is calculations._OPTIONAL_IMPORT_UNSET

        with pytest.raises(
            RuntimeError,
            match="CIF-based optics require Dans_Diffraction and xraydb.",
        ):
            calculations._cif_optics_properties("unused.cif")

        if finder in sys.meta_path:
            sys.meta_path.remove(finder)

        fake_dans_diffraction = ModuleType("Dans_Diffraction")

        class _FakeCrystal:
            def __init__(self, _path):
                self.Symmetry = SimpleNamespace(generate_matrices=lambda: None)
                self.Structure = SimpleNamespace(
                    type=["I", "Nb"],
                    occupancy=np.array([1.0, 1.0], dtype=np.float64),
                )
                self.Cell = SimpleNamespace(volume=lambda: 100.0)

            def generate_structure(self):
                return None

        fake_dans_diffraction.Crystal = _FakeCrystal

        fake_xraydb = ModuleType("xraydb")
        fake_xraydb.atomic_mass = lambda symbol: {"I": 126.90447, "Nb": 92.90637}[symbol]
        fake_xraydb.atomic_number = lambda symbol: {"I": 53.0, "Nb": 41.0}[symbol]

        previous_dans = sys.modules.get("Dans_Diffraction")
        previous_xraydb = sys.modules.get("xraydb")
        with monkeypatch.context() as module_patch:
            module_patch.setitem(sys.modules, "Dans_Diffraction", fake_dans_diffraction)
            module_patch.setitem(sys.modules, "xraydb", fake_xraydb)

            assert calculations._get_dans_diffraction_module() is fake_dans_diffraction
            assert calculations._get_xraydb_module() is fake_xraydb

            calculations._cif_optics_properties.cache_clear()
            props = calculations._cif_optics_properties("unused.cif")

            assert props["path"].endswith("unused.cif")
            assert props["element_symbols"] == ("I", "Nb")
            assert props["density_g_cm3"] > 0.0
            assert props["electron_density_m3"] > 0.0

        assert sys.modules.get("Dans_Diffraction") is previous_dans
        assert sys.modules.get("xraydb") is previous_xraydb
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        _restore_modules(previous)
