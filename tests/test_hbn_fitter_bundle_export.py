from __future__ import annotations

from pathlib import Path

import numpy as np

from hbn_fitter import fitter as hbn_fitter


class _DummyVar:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _DummyFitter:
    def __init__(self) -> None:
        self.img_bgsub = np.arange(16, dtype=np.float32).reshape(4, 4)
        self.optim = {
            "tilt_x_deg": 2.75,
            "tilt_y_deg": -1.5,
            "cost_zero": 0.12,
            "cost_final": 0.03,
        }
        self.fit_quality = {}
        self.points_ds = [[(1.0, 2.0), (3.0, 4.0)]]
        self.points_raw_ds = [[(1.5, 2.5), (3.5, 4.5)]]
        self.points_sigma_ds = [[0.2, 0.3]]
        self.down = 1
        self.ellipses = [
            {
                "xc": 12.0,
                "yc": 10.0,
                "a": 8.0,
                "b": 7.5,
                "theta": 0.25,
                "ring_index": 0,
            }
        ]
        self.hbn_path = _DummyVar("hbn.osc")
        self.dark_path = _DummyVar("dark.osc")

    def _get_fullres_log_image(self):
        return np.ones((4, 4), dtype=np.float32)

    def _get_center(self, strict=False):
        return (12.0, 10.0), "ui_entry"

    def _points_for_fit(self, points, downsample=1):
        return points

    def _sigma_for_fit(self, values, downsample=1):
        return values


def test_build_hbn_fitter_bundle_payload_exports_opposite_sign_detector_tilts() -> None:
    fitter = _DummyFitter()
    center, center_src = fitter._get_center(strict=False)

    bundle = hbn_fitter.build_hbn_fitter_bundle_payload(
        img_bgsub=fitter.img_bgsub,
        img_log_full=fitter._get_fullres_log_image(),
        downsample_factor=fitter.down,
        center=center,
        center_source=center_src,
        optim=fitter.optim,
        fit_quality=fitter.fit_quality,
        points_ds=fitter._points_for_fit(fitter.points_ds, downsample=fitter.down),
        points_raw_ds=fitter._points_for_fit(fitter.points_raw_ds, downsample=fitter.down),
        points_sigma_ds=fitter._sigma_for_fit(fitter.points_sigma_ds, downsample=fitter.down),
        ellipses=fitter.ellipses,
        input_hbn_path=fitter.hbn_path.get(),
        input_dark_path=fitter.dark_path.get(),
        created_utc="2026-03-28T12:00:00Z",
    )

    tilt_correction = bundle["tilt_correction"]
    tilt_hint = bundle["tilt_hint"]

    assert np.isclose(float(bundle["tilt_x_deg"]), -2.75)
    assert np.isclose(float(bundle["tilt_y_deg"]), 1.5)
    assert np.isclose(float(bundle["tilt_x_deg_internal"]), 2.75)
    assert np.isclose(float(bundle["tilt_y_deg_internal"]), -1.5)

    assert np.isclose(float(tilt_correction["tilt_x_deg"]), -2.75)
    assert np.isclose(float(tilt_correction["tilt_y_deg"]), 1.5)
    assert np.isclose(float(tilt_hint["rot1_rad"]), np.deg2rad(-2.75))
    assert np.isclose(float(tilt_hint["rot2_rad"]), np.deg2rad(1.5))
    assert str(np.asarray(bundle["created_utc"]).reshape(-1)[0]) == "2026-03-28T12:00:00Z"


def test_hbn_fitter_save_bundle_writes_payload_with_expected_tilt_signs(tmp_path: Path) -> None:
    fitter = _DummyFitter()
    out_path = tmp_path / "bundle.npz"

    hbn_fitter.HBNFitterGUI.save_bundle(fitter, out_path)

    data = np.load(out_path, allow_pickle=True)
    tilt_correction = data["tilt_correction"].item()
    tilt_hint = data["tilt_hint"].item()

    assert np.isclose(float(data["tilt_x_deg"]), -2.75)
    assert np.isclose(float(data["tilt_y_deg"]), 1.5)
    assert np.isclose(float(data["tilt_x_deg_internal"]), 2.75)
    assert np.isclose(float(data["tilt_y_deg_internal"]), -1.5)

    assert np.isclose(float(tilt_correction["tilt_x_deg"]), -2.75)
    assert np.isclose(float(tilt_correction["tilt_y_deg"]), 1.5)
    assert np.isclose(float(tilt_hint["rot1_rad"]), np.deg2rad(-2.75))
    assert np.isclose(float(tilt_hint["rot2_rad"]), np.deg2rad(1.5))
