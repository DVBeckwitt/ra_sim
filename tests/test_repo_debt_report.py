import builtins
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "tools" / "repo_debt_report.py"
HEAVY_MODULE_NAMES = (
    "ra_sim.gui._runtime.runtime_session",
    "ra_sim.fitting.optimization",
)


def _is_heavy_module_name(name: str) -> bool:
    return any(name == module or name.startswith(f"{module}.") for module in HEAVY_MODULE_NAMES)


def _assert_not_heavy_import(name: str, fromlist: object = ()) -> None:
    if _is_heavy_module_name(name):
        raise AssertionError(f"repo debt report imported heavy module {name}")
    if not fromlist:
        return
    for item in fromlist:
        if isinstance(item, str) and _is_heavy_module_name(f"{name}.{item}"):
            raise AssertionError(f"repo debt report imported heavy module {name}.{item}")


def _load_report_module():
    spec = importlib.util.spec_from_file_location("repo_debt_report", REPORT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repo_debt_report_json_contract() -> None:
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if level == 0:
            _assert_not_heavy_import(name, fromlist)
        return original_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name: str, package: str | None = None) -> object:
        _assert_not_heavy_import(name)
        return original_import_module(name, package)

    try:
        builtins.__import__ = guarded_import
        importlib.import_module = guarded_import_module
        before_modules = set(sys.modules)
        report = _load_report_module()
        payload = report.build_report(REPO_ROOT)
        imported_modules = set(sys.modules) - before_modules
    finally:
        builtins.__import__ = original_import
        importlib.import_module = original_import_module

    assert not (set(HEAVY_MODULE_NAMES) & imported_modules)
    assert sorted(payload) == [
        "diagnostics",
        "duplicate_diagnostic_functions",
        "env_flags",
        "test_tiers",
        "top_files",
        "top_functions",
    ]
    assert payload["top_files"]
    assert payload["top_functions"]
    tier_member_count = sum(len(names) for names in payload["test_tiers"]["tiers"].values())
    assert payload["test_tiers"]["total"] == tier_member_count
    assert payload["test_tiers"]["classified"] == tier_member_count
    assert payload["test_tiers"]["unclassified"] == []
    assert payload["test_tiers"]["duplicates"] == {}
    assert payload["diagnostics"]["unclassified"] == []
    assert payload["diagnostics"]["duplicates"] == {}
    assert isinstance(payload["duplicate_diagnostic_functions"], list)
    assert "observed" in payload["env_flags"]
    assert "declared" in payload["env_flags"]
    assert "undeclared" in payload["env_flags"]


def test_repo_debt_report_cli_emits_json_without_heavy_imports() -> None:
    child_script = f"""
import builtins
import importlib
import importlib.util
import json
import pathlib
import sys

report_path = pathlib.Path({str(REPORT_PATH)!r})
repo_root = pathlib.Path({str(REPO_ROOT)!r})
heavy_modules = {HEAVY_MODULE_NAMES!r}
original_import = builtins.__import__
original_import_module = importlib.import_module

def is_heavy_module_name(name):
    return any(name == module or name.startswith(f"{{module}}.") for module in heavy_modules)

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and is_heavy_module_name(name):
        raise AssertionError(f"repo debt report imported heavy module {{name}}")
    if level == 0:
        for item in fromlist or ():
            if isinstance(item, str) and is_heavy_module_name(f"{{name}}.{{item}}"):
                raise AssertionError(f"repo debt report imported heavy module {{name}}.{{item}}")
    return original_import(name, globals, locals, fromlist, level)

def guarded_import_module(name, package=None):
    if is_heavy_module_name(name):
        raise AssertionError(f"repo debt report imported heavy module {{name}}")
    return original_import_module(name, package)

spec = importlib.util.spec_from_file_location("repo_debt_report_child", report_path)
module = importlib.util.module_from_spec(spec)
assert spec is not None
assert spec.loader is not None
builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module
try:
    spec.loader.exec_module(module)
    payload = module.build_report(repo_root)
finally:
    builtins.__import__ = original_import
    importlib.import_module = original_import_module
print(json.dumps(payload, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["top_files"]
