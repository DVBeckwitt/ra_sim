from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ra_sim import hbn
from ra_sim.hbn import (
    build_hbn_geometry_debug_trace,
    convert_hbn_bundle_geometry_to_simulation,
    format_hbn_geometry_debug_trace,
    load_bundle_npz,
    load_tilt_hint,
)


def _rotate_point_by_k(
    col: float, row: float, shape: tuple[int, int], k: int
) -> tuple[float, float]:
    h, w = int(shape[0]), int(shape[1])
    c = float(col)
    r = float(row)
    for _ in range(int(k) % 4):
        r, c, h, w = w - 1 - c, r, w, h
    return c, r


def _rotate_components(gamma_src: float, Gamma_src: float, k_delta: int) -> tuple[float, float]:
    alpha_rad = np.deg2rad(90.0 * float(k_delta))
    c = float(np.cos(alpha_rad))
    s = float(np.sin(alpha_rad))
    gamma_tgt = c * float(gamma_src) + s * float(Gamma_src)
    Gamma_tgt = -s * float(gamma_src) + c * float(Gamma_src)
    return float(gamma_tgt), float(Gamma_tgt)


def _write_bundle(
    path: Path,
    *,
    tilt_x_deg: float,
    tilt_y_deg: float,
    center: tuple[float, float],
    include_canonical_metadata: bool,
    source_rotate_k: int = 0,
    gamma_sign_from_tilt_x: int = 1,
    gamma_sign_from_tilt_y: int = 1,
) -> None:
    payload = {
        "img_bgsub": np.zeros((32, 32), dtype=np.float32),
        "img_log": np.zeros((32, 32), dtype=np.float32),
        "ell_points_ds": np.array(
            [np.array([[1.0, 2.0]], dtype=np.float32)],
            dtype=object,
        ),
        "ellipse_params": np.array([[16.0, 16.0, 8.0, 7.0, 0.0]], dtype=np.float32),
        "tilt_x_deg": np.array(float(tilt_x_deg), dtype=np.float64),
        "tilt_y_deg": np.array(float(tilt_y_deg), dtype=np.float64),
        "tilt_correction": {
            "tilt_x_deg": float(tilt_x_deg),
            "tilt_y_deg": float(tilt_y_deg),
        },
        "tilt_hint": {
            "rot1_rad": float(np.deg2rad(tilt_x_deg)),
            "rot2_rad": float(np.deg2rad(tilt_y_deg)),
            "tilt_rad": float(
                np.hypot(np.deg2rad(tilt_x_deg), np.deg2rad(tilt_y_deg))
            ),
        },
        "distance_estimate_m": {"mean_m": 0.075},
        "center": np.asarray(center, dtype=np.float64),
    }
    if include_canonical_metadata:
        payload.update(
            {
                "sim_background_rotate_k": np.array(int(source_rotate_k), dtype=np.int32),
                "tilt_correction_kind": np.array("to_flat"),
                "tilt_model": np.array("RzRx"),
                "tilt_frame": np.array("simulation_background_display"),
                "simulation_gamma_sign_from_tilt_x": np.array(
                    int(gamma_sign_from_tilt_x), dtype=np.int32
                ),
                "simulation_Gamma_sign_from_tilt_y": np.array(
                    int(gamma_sign_from_tilt_y), dtype=np.int32
                ),
            }
        )
    np.savez(path, **payload)


def test_identity_conversion_uses_signed_components() -> None:
    tx = 2.1
    ty = -1.4
    converted = convert_hbn_bundle_geometry_to_simulation(
        tilt_x_deg=tx,
        tilt_y_deg=ty,
        center_xy=None,
        source_rotate_k=0,
        target_rotate_k=0,
        image_size=32,
        simulation_gamma_sign_from_tilt_x=1,
        simulation_Gamma_sign_from_tilt_y=1,
    )
    assert np.isclose(converted["gamma_deg"], tx, atol=1e-10)
    assert np.isclose(converted["Gamma_deg"], ty, atol=1e-10)

    flipped = convert_hbn_bundle_geometry_to_simulation(
        tilt_x_deg=tx,
        tilt_y_deg=ty,
        center_xy=None,
        source_rotate_k=0,
        target_rotate_k=0,
        image_size=32,
        simulation_gamma_sign_from_tilt_x=-1,
        simulation_Gamma_sign_from_tilt_y=-1,
    )
    assert np.isclose(flipped["gamma_deg"], -tx, atol=1e-10)
    assert np.isclose(flipped["Gamma_deg"], -ty, atol=1e-10)


def test_frame_rotation_k_delta_applied_to_tilt_and_center() -> None:
    tx = 1.7
    ty = -0.9
    center_src = (30.0, 10.0)  # (col, row)
    shape = (100, 100)

    base = convert_hbn_bundle_geometry_to_simulation(
        tilt_x_deg=tx,
        tilt_y_deg=ty,
        center_xy=center_src,
        source_rotate_k=0,
        target_rotate_k=0,
        image_size=shape,
        simulation_gamma_sign_from_tilt_x=1,
        simulation_Gamma_sign_from_tilt_y=1,
    )
    rot = convert_hbn_bundle_geometry_to_simulation(
        tilt_x_deg=tx,
        tilt_y_deg=ty,
        center_xy=center_src,
        source_rotate_k=0,
        target_rotate_k=1,
        image_size=shape,
        simulation_gamma_sign_from_tilt_x=1,
        simulation_Gamma_sign_from_tilt_y=1,
    )

    expected_gamma, expected_Gamma = _rotate_components(tx, ty, k_delta=1)
    assert np.isclose(base["gamma_deg"], tx, atol=1e-10)
    assert np.isclose(base["Gamma_deg"], ty, atol=1e-10)
    assert np.isclose(rot["gamma_deg"], expected_gamma, atol=1e-10)
    assert np.isclose(rot["Gamma_deg"], expected_Gamma, atol=1e-10)

    expected_col, expected_row = _rotate_point_by_k(center_src[0], center_src[1], shape, 1)
    assert rot["k_delta"] == 1
    assert np.isclose(rot["center_col"], expected_col)
    assert np.isclose(rot["center_row"], expected_row)


def test_geometry_debug_trace_roundtrips_center_and_inverse_rotation() -> None:
    npz_center = (1593.129, 1539.099)  # (col, row)
    image_shape = (3000, 3000)
    source_k = 0
    target_k = -1

    converted = convert_hbn_bundle_geometry_to_simulation(
        tilt_x_deg=8.0,
        tilt_y_deg=9.0,
        center_xy=npz_center,
        source_rotate_k=source_k,
        target_rotate_k=target_k,
        image_size=image_shape,
    )
    sim_row = float(converted["center_row"])
    sim_col = float(converted["center_col"])

    trace = build_hbn_geometry_debug_trace(
        npz_center_xy=npz_center,
        source_rotate_k=source_k,
        target_rotate_k=target_k,
        image_size=image_shape,
        tilt_x_deg=8.0,
        tilt_y_deg=9.0,
        simulation_center_row=sim_row,
        simulation_center_col=sim_col,
    )

    expected_sim = trace["expected_sim_center"]
    assert isinstance(expected_sim, dict)
    assert np.isclose(expected_sim["row"], sim_row)
    assert np.isclose(expected_sim["col"], sim_col)

    back_err = trace["applied_back_to_npz_error"]
    assert isinstance(back_err, dict)
    assert np.isclose(back_err["d_col"], 0.0, atol=1e-10)
    assert np.isclose(back_err["d_row"], 0.0, atol=1e-10)

    text = format_hbn_geometry_debug_trace(trace)
    assert "k_delta=-1" in text
    assert "rotating simulation by +90 deg CCW should recover hBN frame" in text


def test_legacy_bundle_without_canonical_metadata_raises_keyerror(tmp_path: Path) -> None:
    bundle = tmp_path / "legacy_bundle.npz"
    _write_bundle(
        bundle,
        tilt_x_deg=1.2,
        tilt_y_deg=-0.7,
        center=(15.0, 14.0),
        include_canonical_metadata=False,
    )

    with pytest.raises(KeyError, match="required canonical metadata keys"):
        load_bundle_npz(bundle, verbose=False)


def test_main_and_app_use_same_conversion_helper() -> None:
    main_text = Path("main.py").read_text(encoding="utf-8")
    app_text = Path("ra_sim/gui/app.py").read_text(encoding="utf-8")
    assert "convert_hbn_bundle_geometry_to_simulation(" in main_text
    assert "convert_hbn_bundle_geometry_to_simulation(" in app_text
    assert "SIMULATION_GEOMETRY_ROTATE_K" in main_text
    assert "SIMULATION_GEOMETRY_ROTATE_K" in app_text
    assert "HBN_FITTER_ROTATE_K" in main_text
    assert "HBN_FITTER_ROTATE_K" in app_text
    assert "estimate_detector_tilt(" not in main_text
    assert "estimate_detector_tilt(" not in app_text


def test_explicit_sign_metadata_controls_detector_rotation_signs(tmp_path: Path) -> None:
    tx = 2.4
    ty = -1.6
    bundle = tmp_path / "signed_bundle.npz"
    _write_bundle(
        bundle,
        tilt_x_deg=tx,
        tilt_y_deg=ty,
        center=(12.0, 10.0),
        include_canonical_metadata=True,
        source_rotate_k=0,
        gamma_sign_from_tilt_x=-1,
        gamma_sign_from_tilt_y=-1,
    )

    loaded = load_bundle_npz(bundle, verbose=False)
    tilt_correction = loaded[5]
    center = loaded[8]
    assert isinstance(tilt_correction, dict)
    assert tilt_correction["simulation_gamma_sign_from_tilt_x"] == -1
    assert tilt_correction["simulation_Gamma_sign_from_tilt_y"] == -1

    converted = convert_hbn_bundle_geometry_to_simulation(
        tilt_x_deg=float(tilt_correction["tilt_x_deg"]),
        tilt_y_deg=float(tilt_correction["tilt_y_deg"]),
        center_xy=center,
        source_rotate_k=int(tilt_correction["sim_background_rotate_k"]),
        target_rotate_k=int(tilt_correction["sim_background_rotate_k"]),
        image_size=(32, 32),
        simulation_gamma_sign_from_tilt_x=int(
            tilt_correction["simulation_gamma_sign_from_tilt_x"]
        ),
        simulation_Gamma_sign_from_tilt_y=int(
            tilt_correction["simulation_Gamma_sign_from_tilt_y"]
        ),
    )
    assert np.isclose(converted["gamma_deg"], -tx, atol=1e-10)
    assert np.isclose(converted["Gamma_deg"], -ty, atol=1e-10)


def test_startup_hint_and_import_button_same_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "canonical_bundle.npz"
    _write_bundle(
        bundle,
        tilt_x_deg=2.8,
        tilt_y_deg=-1.9,
        center=(20.0, 12.0),
        include_canonical_metadata=True,
        source_rotate_k=0,
        gamma_sign_from_tilt_x=1,
        gamma_sign_from_tilt_y=1,
    )

    monkeypatch.setattr(
        hbn,
        "resolve_hbn_paths",
        lambda paths_file=None: {"bundle": str(bundle)},
    )
    hint = load_tilt_hint(paths_file="unused")
    assert hint is not None

    loaded = load_bundle_npz(bundle, verbose=False)
    tilt_correction = loaded[5]
    center = loaded[8]
    assert isinstance(tilt_correction, dict)

    converted = convert_hbn_bundle_geometry_to_simulation(
        tilt_x_deg=float(tilt_correction["tilt_x_deg"]),
        tilt_y_deg=float(tilt_correction["tilt_y_deg"]),
        center_xy=center,
        source_rotate_k=int(tilt_correction["sim_background_rotate_k"]),
        target_rotate_k=int(hbn.SIM_BACKGROUND_ROTATE_K),
        image_size=(32, 32),
        simulation_gamma_sign_from_tilt_x=-int(
            tilt_correction["simulation_gamma_sign_from_tilt_x"]
        ),
        simulation_Gamma_sign_from_tilt_y=int(
            tilt_correction["simulation_Gamma_sign_from_tilt_y"]
        ),
    )

    assert np.isclose(hint["gamma_deg"], converted["gamma_deg"])
    assert np.isclose(hint["Gamma_deg"], converted["Gamma_deg"])
    assert np.isclose(hint["center_row"], converted["center_row"])
    assert np.isclose(hint["center_col"], converted["center_col"])
