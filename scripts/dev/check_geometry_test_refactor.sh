#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

FILES=(
  tests/test_manual_geometry_selection_helpers.py
  tests/test_gui_geometry_fit_workflow.py
  tests/test_gui_runtime_import_safe.py
  tests/test_geometry_fitting.py
)

collect_after="${TMPDIR:-/tmp}/ra_sim_collect_after.txt"
baseline_normalized="${TMPDIR:-/tmp}/ra_sim_collect_baseline_normalized.txt"
collect_after_normalized="${TMPDIR:-/tmp}/ra_sim_collect_after_normalized.txt"

python -m pytest --collect-only -q "${FILES[@]}" > "${collect_after}"

if [ -f artifacts/test_refactor_baseline/collect.txt ]; then
  sed -E 's/^([0-9]+ tests collected) in .*$/\1/' \
    artifacts/test_refactor_baseline/collect.txt > "${baseline_normalized}"
  sed -E 's/^([0-9]+ tests collected) in .*$/\1/' \
    "${collect_after}" > "${collect_after_normalized}"
  diff -u "${baseline_normalized}" "${collect_after_normalized}"
fi

python -m pytest -q --tb=short "${FILES[@]}"
