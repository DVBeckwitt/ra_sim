import sys
from pathlib import Path

import pytest

# Allow importing ra_sim from repository root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ra_sim.debug_controls import reset_run_bundle_state, reset_startup_debug_log_path
from ra_sim.test_tiers import FAST_TEST_FILES, INTEGRATION_TEST_FILES


def _item_path(item: pytest.Item) -> Path:
    raw_path = getattr(item, "path", None)
    if raw_path is not None:
        return Path(raw_path)
    return Path(str(item.fspath))


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    del config

    fast_marker = pytest.mark.fast
    integration_marker = pytest.mark.integration
    benchmark_marker = pytest.mark.benchmark

    for item in items:
        path = _item_path(item)
        if "benchmarks" in path.parts:
            item.add_marker(benchmark_marker)
            continue
        if path.name in INTEGRATION_TEST_FILES:
            item.add_marker(integration_marker)
            continue
        if path.name in FAST_TEST_FILES:
            item.add_marker(fast_marker)


@pytest.fixture(autouse=True)
def _reset_startup_log_path() -> None:
    reset_startup_debug_log_path()
    reset_run_bundle_state()
    yield
    reset_startup_debug_log_path()
    reset_run_bundle_state()
