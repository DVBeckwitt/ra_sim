from __future__ import annotations

from scripts.debug import run_new4_geometry_fit_ladder as ladder


def _rung2_payload(
    *,
    active_params: tuple[str, ...],
    near_zero_params: tuple[str, ...] = (),
    fixed_params: tuple[str, ...] = ("c",),
) -> dict[str, object]:
    return {
        "status": "ok",
        "pass": True,
        "active_params": list(active_params),
        "active_parameters": list(active_params),
        "active_param_count": len(active_params),
        "near_zero_params": list(near_zero_params),
        "near_zero_parameters": list(near_zero_params),
        "near_zero_param_count": len(near_zero_params),
        "fixed_params": list(fixed_params),
        "fixed_parameters": list(fixed_params),
        "fixed_param_count": len(fixed_params),
        "excluded_params": list(fixed_params),
        "excluded_parameters": list(fixed_params),
        "excluded_param_count": len(fixed_params),
        "unsafe_params": [],
        "unsafe_parameters": [],
        "unsafe_param_count": 0,
        "non_finite_params": [],
        "non_finite_parameters": [],
        "non_finite_param_count": 0,
        "residual_probe_called": True,
        "least_squares_called": False,
        "optimizer_solve_called": False,
        "state_hash_unchanged": True,
        "provider_pair_count": 7,
        "dataset_pair_count": 7,
        "optimizer_request_pair_count": 7,
        "fixed_source_pair_count": 7,
        "fixed_source_resolved_count": 7,
        "fallback_entry_count": 0,
        "matched_pair_count": 7,
        "missing_pair_count": 0,
        "fallback_row_count": 0,
        "fixed_source_resolution_fallback_count": 0,
        "missing_fixed_source_count": 0,
        "branch_mismatch_count": 0,
        "provider_to_optimizer_identity_match": True,
        "provider_to_optimizer_point_match": True,
        "params": [
            {
                "param_name": str(name),
                "provider_pair_count": 7,
                "dataset_pair_count": 7,
                "optimizer_request_pair_count": 7,
                "fixed_source_pair_count": 7,
                "fixed_source_resolved_count": 7,
                "fallback_entry_count": 0,
                "matched_pair_count": 7,
                "missing_pair_count": 0,
                "fallback_row_count": 0,
                "fixed_source_resolution_fallback_count": 0,
                "missing_fixed_source_count": 0,
                "branch_mismatch_count": 0,
                "provider_to_optimizer_identity_match": True,
                "provider_to_optimizer_point_match": True,
            }
            for name in active_params
        ],
    }


def test_rung2_accepts_gamma_Gamma_active_with_c_fixed() -> None:
    payload = _rung2_payload(
        active_params=tuple(ladder.EXPECTED_RUNG2_ACTIVE_PARAMS),
        near_zero_params=(),
        fixed_params=("c",),
    )

    assert ladder._rung2_green_failures(payload) == []


def test_rung2_rejects_gamma_Gamma_near_zero_under_free_tilt_contract() -> None:
    active = tuple(
        name
        for name in ladder.EXPECTED_RUNG2_ACTIVE_PARAMS
        if name not in {"gamma", "Gamma"}
    )
    payload = _rung2_payload(
        active_params=active,
        near_zero_params=("gamma", "Gamma"),
        fixed_params=("c",),
    )

    failures = ladder._rung2_green_failures(payload)

    assert "active_params_mismatch" in failures
    assert "near_zero_params_mismatch" in failures


def test_rung2_rejects_c_near_zero_when_c_is_fixed() -> None:
    payload = _rung2_payload(
        active_params=tuple(ladder.EXPECTED_RUNG2_ACTIVE_PARAMS),
        near_zero_params=("c",),
        fixed_params=("c",),
    )

    failures = ladder._rung2_green_failures(payload)

    assert "near_zero_params_mismatch" in failures
    assert "fixed_param_leaked_into_probe_results" in failures


def test_rung2_rejects_c_active_when_c_is_fixed() -> None:
    payload = _rung2_payload(
        active_params=tuple(ladder.EXPECTED_RUNG2_ACTIVE_PARAMS | {"c"}),
        near_zero_params=(),
        fixed_params=("c",),
    )

    failures = ladder._rung2_green_failures(payload)

    assert "active_params_mismatch" in failures
    assert "fixed_param_leaked_into_probe_results" in failures
