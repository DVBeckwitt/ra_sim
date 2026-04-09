from types import SimpleNamespace

import numpy as np

from ra_sim.utils import diffraction_tools


def test_view_azimuthal_radial_preserves_full_cake_azimuth(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeIntegrator:
        def __init__(self, **kwargs) -> None:
            captured["integrator_kwargs"] = dict(kwargs)

        def integrate2d(self, simulated_image, **kwargs):
            captured["integrate2d_image"] = np.asarray(simulated_image, dtype=float)
            captured["integrate2d_kwargs"] = dict(kwargs)
            return SimpleNamespace(
                intensity=np.asarray(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                    ],
                    dtype=float,
                ),
                radial=np.asarray([10.0, 20.0], dtype=float),
                azimuthal=np.asarray([-170.0, 10.0, 100.0], dtype=float),
            )

    monkeypatch.setattr(diffraction_tools, "AzimuthalIntegrator", _FakeIntegrator)
    monkeypatch.setattr(diffraction_tools.plt, "figure", lambda *args, **kwargs: None)
    monkeypatch.setattr(diffraction_tools.plt, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(diffraction_tools.plt, "xlabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(diffraction_tools.plt, "ylabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(diffraction_tools.plt, "colorbar", lambda *args, **kwargs: None)
    monkeypatch.setattr(diffraction_tools.plt, "show", lambda *args, **kwargs: None)

    def _capture_imshow(image, **kwargs):
        captured["imshow_image"] = np.asarray(image, dtype=float)
        captured["imshow_kwargs"] = dict(kwargs)

    monkeypatch.setattr(diffraction_tools.plt, "imshow", _capture_imshow)

    diffraction_tools.view_azimuthal_radial(
        simulated_image=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        center=(12.0, 13.0),
        detector_params={
            "pixel_size": 1.0e-4,
            "poni1": 0.01,
            "poni2": 0.02,
            "dist": 0.3,
            "rot1": 0.0,
            "rot2": 0.0,
            "rot3": 0.0,
            "wavelength": 1.54e-10,
        },
    )

    imshow_image = np.asarray(captured["imshow_image"], dtype=float)
    imshow_kwargs = dict(captured["imshow_kwargs"])
    extent = list(imshow_kwargs["extent"])

    assert imshow_image.shape == (3, 2)
    assert np.allclose(imshow_image, [[3.0, 4.0], [1.0, 2.0], [5.0, 6.0]])
    assert extent[0:2] == [10.0, 20.0]
    assert extent[2] < -90.0
    assert extent[3] > 90.0
