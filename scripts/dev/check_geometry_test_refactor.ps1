$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "../..")
Set-Location $repoRoot

$files = @(
    "tests/test_manual_geometry_selection_helpers.py",
    "tests/test_gui_geometry_fit_workflow.py",
    "tests/test_gui_runtime_import_safe.py",
    "tests/test_geometry_fitting.py"
)

$collectAfter = Join-Path ([System.IO.Path]::GetTempPath()) "ra_sim_collect_after.txt"
$baselineNormalized = Join-Path ([System.IO.Path]::GetTempPath()) "ra_sim_collect_baseline_normalized.txt"
$collectAfterNormalized = Join-Path ([System.IO.Path]::GetTempPath()) "ra_sim_collect_after_normalized.txt"
$baseline = "artifacts/test_refactor_baseline/collect.txt"

python -m pytest --collect-only -q @files > $collectAfter
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (Test-Path $baseline) {
    (Get-Content $baseline) -replace '^([0-9]+ tests collected) in .*$', '$1' |
        Set-Content -Path $baselineNormalized
    (Get-Content $collectAfter) -replace '^([0-9]+ tests collected) in .*$', '$1' |
        Set-Content -Path $collectAfterNormalized
    $diff = git diff --no-index -- $baselineNormalized $collectAfterNormalized
    if ($LASTEXITCODE -ne 0) {
        $diff
        exit $LASTEXITCODE
    }
}

python -m pytest -q --tb=short @files
exit $LASTEXITCODE
