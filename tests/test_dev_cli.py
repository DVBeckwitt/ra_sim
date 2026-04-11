from ra_sim import dev


def test_typecheck_targets_include_refactored_modules() -> None:
    targets = dev.typecheck_targets()

    assert "ra_sim/fitting/optimization_mosaic_profiles.py" in targets
    assert "ra_sim/fitting/optimization_runtime.py" in targets
    assert "ra_sim/gui/_runtime/live_cache_helpers.py" in targets
    assert "ra_sim/gui/_runtime/primary_cache_helpers.py" in targets
    assert "ra_sim/gui/runtime_primary_cache.py" in targets
    assert "ra_sim/config/loader.py" in targets
    assert "ra_sim/dev.py" in targets


def test_format_targets_cover_large_split_frontier() -> None:
    targets = dev.format_targets()

    assert "ra_sim/fitting/optimization.py" in targets
    assert "ra_sim/gui/_runtime/runtime_session.py" in targets
    assert "ra_sim/fitting/optimization_mosaic_profiles.py" in targets
    assert "ra_sim/gui/_runtime/primary_cache_helpers.py" in targets


def test_pip_install_command_variants_keep_group_then_extra_fallback() -> None:
    variants = dev.pip_install_command_variants()

    assert variants[0][-4:] == ["--group", "dev", "-e", "."]
    assert variants[1][-2:] == ["-e", ".[dev]"]


def test_pip_lock_command_variants_keep_group_then_extra_fallback() -> None:
    variants = dev.pip_lock_command_variants(output="custom.lock")

    assert variants[0][-6:] == ["-e", ".", "--group", "dev", "-o", "custom.lock"]
    assert variants[1][-4:] == ["-e", ".[dev]", "-o", "custom.lock"]


def test_pre_commit_install_command_uses_module_entrypoint() -> None:
    command = dev.pre_commit_install_command()

    assert command == [dev.sys.executable, "-m", "pre_commit", "install"]


def test_pytest_tiers_map_to_expected_markers() -> None:
    fast_command = dev.pytest_command_for_tier("fast")
    assert fast_command[0:3] == [dev.sys.executable, "-m", "pytest"]
    assert any(path.endswith("test_dev_cli.py") for path in fast_command)
    assert fast_command[-2:] == ["-m", dev.FAST_MARKER]
    assert dev.pytest_command_for_tier("integration")[-2:] == [
        "-m",
        dev.INTEGRATION_MARKER,
    ]


def test_parser_exposes_format_and_hook_commands() -> None:
    parser = dev.build_parser()

    for command in ("format", "format-check", "hooks"):
        args = parser.parse_args([command])
        assert args.command == command
