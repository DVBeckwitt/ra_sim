"""Portable flat-detector wrapper for the fast exact detector-to-cake splitter.

Example
-------
```python
import numpy as np

from ra_sim.simulation.exact_cake_portable import convert_image_to_angle_space

image = np.load("detector.npy")
result = convert_image_to_angle_space(
    image,
    pixel_size_m=1.0e-4,
    distance_m=0.075,
    poni1_m=0.1596422,
    poni2_m=0.1453120,
    npt_rad=1000,
    npt_azim=720,
    tth_min_deg=0.0,
    tth_max_deg=90.0,
    correct_solid_angle=True,
    engine="numba",
    workers=24,
)

cake = result.intensity
two_theta_deg = result.radial_deg
azimuth_deg = result.azimuthal_deg
```
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from ra_sim.simulation.exact_cake import (
    DetectorCakeGeometry,
    DetectorCakeLUT,
    DetectorCakeResult,
    build_detector_to_cake_lut,
    integrate_detector_to_cake_exact,
    integrate_detector_to_cake_lut,
)


PHI_ZERO_OFFSET_DEGREES = -90.0


@dataclass(frozen=True)
class PortableGeometry:
    pixel_size_m: float
    distance_m: float
    center_row_px: float
    center_col_px: float

def _parse_poni_file_simple(poni_path: str | Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in Path(poni_path).expanduser().read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            values[key] = float(value)
        except ValueError:
            continue
    return values


def build_geometry(
    *,
    pixel_size_m: float,
    distance_m: float | None = None,
    center_row_px: float | None = None,
    center_col_px: float | None = None,
    poni1_m: float | None = None,
    poni2_m: float | None = None,
    poni_path: str | Path | None = None,
) -> PortableGeometry:
    """Build portable detector geometry from either center pixels or PONI values.

    Zero-rotation, flat-detector convention:
    - ``center_row_px = poni1_m / pixel_size_m``
    - ``center_col_px = poni2_m / pixel_size_m``
    """

    pixel_size = float(pixel_size_m)
    if pixel_size <= 0.0:
        raise ValueError("pixel_size_m must be > 0.")
    parsed_poni: dict[str, float] = {}
    if poni_path is not None:
        parsed_poni = _parse_poni_file_simple(poni_path)
    dist = distance_m if distance_m is not None else parsed_poni.get("Dist")
    if dist is None:
        raise ValueError("distance_m or poni_path with Dist is required.")
    row_px = center_row_px
    col_px = center_col_px
    if row_px is None:
        if poni1_m is None:
            poni1_m = parsed_poni.get("Poni1")
        if poni1_m is None:
            raise ValueError("Provide center_row_px or poni1_m/poni_path.")
        row_px = float(poni1_m) / pixel_size
    if col_px is None:
        if poni2_m is None:
            poni2_m = parsed_poni.get("Poni2")
        if poni2_m is None:
            raise ValueError("Provide center_col_px or poni2_m/poni_path.")
        col_px = float(poni2_m) / pixel_size
    return PortableGeometry(
        pixel_size_m=pixel_size,
        distance_m=float(dist),
        center_row_px=float(row_px),
        center_col_px=float(col_px),
    )


def build_angle_axes(
    *,
    npt_rad: int = 1000,
    npt_azim: int = 720,
    tth_min_deg: float = 0.0,
    tth_max_deg: float = 90.0,
    azimuth_min_deg: float = -180.0,
    azimuth_max_deg: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uniformly spaced 2theta and azimuth bin centers."""

    radial_edges = np.linspace(float(tth_min_deg), float(tth_max_deg), int(max(2, npt_rad)) + 1, dtype=np.float64)
    azimuth_edges = np.linspace(
        float(azimuth_min_deg),
        float(azimuth_max_deg),
        int(max(2, npt_azim)) + 1,
        dtype=np.float64,
    )
    radial_deg = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    azimuthal_deg = 0.5 * (azimuth_edges[:-1] + azimuth_edges[1:])
    return radial_deg, azimuthal_deg


def flat_solid_angle_normalization(
    image_shape: tuple[int, int],
    geometry: PortableGeometry,
) -> np.ndarray:
    """Return flat-detector solid-angle normalization for the given geometry."""

    row_coords = ((np.arange(int(image_shape[0]), dtype=np.float64) + 0.5) - float(geometry.center_row_px)) * float(geometry.pixel_size_m)
    col_coords = ((np.arange(int(image_shape[1]), dtype=np.float64) + 0.5) - float(geometry.center_col_px)) * float(geometry.pixel_size_m)
    yy = row_coords[:, None]
    xx = col_coords[None, :]
    path = np.sqrt(xx * xx + yy * yy + float(geometry.distance_m) ** 2)
    return np.asarray((float(geometry.distance_m) / path) ** 3, dtype=np.float32)


def detector_pixel_angular_maps(
    image_shape: tuple[int, int],
    geometry: PortableGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel flat-detector ``(2theta_deg, raw_azimuth_deg)`` maps."""

    row_coords = (
        (np.arange(int(image_shape[0]), dtype=np.float64) + 0.5) - float(geometry.center_row_px)
    ) * float(geometry.pixel_size_m)
    col_coords = (
        (np.arange(int(image_shape[1]), dtype=np.float64) + 0.5) - float(geometry.center_col_px)
    ) * float(geometry.pixel_size_m)
    yy = row_coords[:, None]
    xx = col_coords[None, :]
    two_theta_deg = np.degrees(np.arctan2(np.hypot(xx, yy), float(geometry.distance_m)))
    raw_azimuth_deg = np.degrees(np.arctan2(yy, xx))
    return np.asarray(two_theta_deg, dtype=np.float64), np.asarray(raw_azimuth_deg, dtype=np.float64)


def detector_points_to_angles(
    cols: np.ndarray | list[float] | tuple[float, ...],
    rows: np.ndarray | list[float] | tuple[float, ...],
    geometry: PortableGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat-detector GUI-style ``(2theta_deg, phi_deg)`` for detector points."""

    cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
    rows_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
    two_theta = np.full(cols_arr.shape, np.nan, dtype=np.float64)
    phi = np.full(cols_arr.shape, np.nan, dtype=np.float64)

    if cols_arr.shape != rows_arr.shape:
        return two_theta, phi
    if not np.isfinite(float(geometry.distance_m)) or float(geometry.distance_m) <= 0.0:
        return two_theta, phi
    if not np.isfinite(float(geometry.pixel_size_m)) or float(geometry.pixel_size_m) <= 0.0:
        return two_theta, phi

    center_row = float(geometry.center_row_px)
    center_col = float(geometry.center_col_px)
    if not np.isfinite(center_row) or not np.isfinite(center_col):
        return two_theta, phi

    valid_mask = np.isfinite(cols_arr) & np.isfinite(rows_arr)
    if not np.any(valid_mask):
        return two_theta, phi

    dx = (cols_arr[valid_mask] - center_col) * float(geometry.pixel_size_m)
    dy = (center_row - rows_arr[valid_mask]) * float(geometry.pixel_size_m)
    radius = np.hypot(dx, dy)
    two_theta_valid = np.degrees(np.arctan2(radius, float(geometry.distance_m)))
    phi_valid = np.degrees(np.arctan2(dx, dy))
    phi_valid = (phi_valid + 180.0) % 360.0 - 180.0

    two_theta[valid_mask] = two_theta_valid
    phi[valid_mask] = phi_valid
    return two_theta, phi


def detector_two_theta_max_deg(
    image_shape: tuple[int, int],
    geometry: PortableGeometry,
) -> float:
    """Return the largest detector-corner 2theta spanned by the flat detector."""

    height = int(image_shape[0])
    width = int(image_shape[1])
    row_edges = (
        np.arange(height + 1, dtype=np.float64) - float(geometry.center_row_px)
    ) * float(geometry.pixel_size_m)
    col_edges = (
        np.arange(width + 1, dtype=np.float64) - float(geometry.center_col_px)
    ) * float(geometry.pixel_size_m)
    max_two_theta = 0.0
    for row_edge in (float(row_edges[0]), float(row_edges[-1])):
        for col_edge in (float(col_edges[0]), float(col_edges[-1])):
            two_theta_deg = math.degrees(
                math.atan2(
                    math.hypot(float(col_edge), float(row_edge)),
                    float(geometry.distance_m),
                )
            )
            if two_theta_deg > max_two_theta:
                max_two_theta = float(two_theta_deg)
    return float(max_two_theta)


def _normalize_two_theta_unit(unit: str | None) -> str:
    if unit is None:
        return "rad"
    text = str(unit).strip().lower()
    if text in {"2th_deg", "deg", "degree", "degrees"}:
        return "deg"
    if text in {"2th_rad", "rad", "radian", "radians"}:
        return "rad"
    raise ValueError("Unsupported two-theta unit.")


def _normalize_angle_unit(unit: str | None) -> str:
    if unit is None:
        return "rad"
    text = str(unit).strip().lower()
    if text in {"deg", "degree", "degrees"}:
        return "deg"
    if text in {"rad", "radian", "radians"}:
        return "rad"
    raise ValueError("Unsupported angle unit.")


class FastAzimuthalIntegrator:
    """Small flat-detector integrator wrapper for the GUI exact-cake path."""

    def __init__(
        self,
        *,
        dist: float,
        poni1: float,
        poni2: float,
        pixel1: float,
        pixel2: float,
        rot1: float = 0.0,
        rot2: float = 0.0,
        rot3: float = 0.0,
        wavelength: float | None = None,
    ) -> None:
        del rot1, rot2, rot3, wavelength
        pixel_row = float(pixel1)
        pixel_col = float(pixel2)
        if not np.isfinite(pixel_row) or pixel_row <= 0.0:
            raise ValueError("pixel1 must be > 0.")
        if not np.isfinite(pixel_col) or pixel_col <= 0.0:
            raise ValueError("pixel2 must be > 0.")
        if not np.isclose(pixel_row, pixel_col, atol=1.0e-15, rtol=0.0):
            raise ValueError("FastAzimuthalIntegrator currently requires square pixels.")
        self.geometry = PortableGeometry(
            pixel_size_m=float(pixel_row),
            distance_m=float(dist),
            center_row_px=float(poni1) / float(pixel_row),
            center_col_px=float(poni2) / float(pixel_col),
        )
        self._detector_map_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._cake_lut_cache: dict[
            tuple[tuple[int, int], int, float, float, int, float, float],
            DetectorCakeLUT,
        ] = {}

    def twoThetaArray(self, *, shape: tuple[int, int], unit: str | None = None) -> np.ndarray:
        two_theta_deg, _raw_azimuth_deg = self._detector_maps_for_shape(shape)
        if _normalize_two_theta_unit(unit) == "deg":
            return np.array(two_theta_deg, copy=True)
        return np.deg2rad(two_theta_deg)

    def chiArray(self, *, shape: tuple[int, int], unit: str | None = None) -> np.ndarray:
        _two_theta_deg, raw_azimuth_deg = self._detector_maps_for_shape(shape)
        if _normalize_angle_unit(unit) == "deg":
            return np.array(raw_azimuth_deg, copy=True)
        return np.deg2rad(raw_azimuth_deg)

    def integrate2d(
        self,
        image: np.ndarray,
        *,
        npt_rad: int = 1000,
        npt_azim: int = 720,
        correctSolidAngle: bool = False,
        method: str | None = None,
        unit: str = "2th_deg",
        normalization: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
        engine: str = "auto",
        workers: int | str | None = "auto",
    ) -> DetectorCakeResult:
        if _normalize_two_theta_unit(unit) != "deg":
            raise ValueError("FastAzimuthalIntegrator.integrate2d only supports degree output.")

        image_arr = np.asarray(image)
        radial_deg, azimuthal_deg = build_angle_axes(
            npt_rad=int(max(1, npt_rad)),
            npt_azim=int(max(1, npt_azim)),
            tth_min_deg=0.0,
            tth_max_deg=detector_two_theta_max_deg(image_arr.shape, self.geometry),
            azimuth_min_deg=-180.0,
            azimuth_max_deg=180.0,
        )
        norm = normalization
        if norm is None and bool(correctSolidAngle):
            norm = flat_solid_angle_normalization(image_arr.shape, self.geometry)
        exact_geometry = DetectorCakeGeometry(
            pixel_size_m=float(self.geometry.pixel_size_m),
            distance_m=float(self.geometry.distance_m),
            center_row_px=float(self.geometry.center_row_px),
            center_col_px=float(self.geometry.center_col_px),
        )
        method_name = "lut" if method is None else str(method).strip().lower()
        if method_name == "lut" and rows is None and cols is None:
            lut = self._cached_cake_lut(
                image_arr.shape,
                radial_deg,
                azimuthal_deg,
                exact_geometry,
                engine=engine,
                workers=workers,
            )
            if lut is not None:
                return integrate_detector_to_cake_lut(
                    image_arr,
                    radial_deg,
                    azimuthal_deg,
                    lut,
                    normalization=norm,
                    mask=mask,
                )
        return integrate_detector_to_cake_exact(
            image_arr,
            radial_deg,
            azimuthal_deg,
            exact_geometry,
            normalization=norm,
            mask=mask,
            rows=rows,
            cols=cols,
            engine=engine,
            workers=workers,
        )

    def _detector_maps_for_shape(
        self,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        detector_shape = tuple(int(v) for v in tuple(shape)[:2])
        cached = self._detector_map_cache.get(detector_shape)
        if cached is None:
            cached = detector_pixel_angular_maps(detector_shape, self.geometry)
            self._detector_map_cache[detector_shape] = cached
        return cached

    def _cached_cake_lut(
        self,
        image_shape: tuple[int, ...],
        radial_deg: np.ndarray,
        azimuthal_deg: np.ndarray,
        geometry: DetectorCakeGeometry,
        *,
        engine: str,
        workers: int | str | None,
    ) -> DetectorCakeLUT | None:
        engine_name = str(engine).strip().lower()
        if engine_name not in {"", "auto", "numba"}:
            return None
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        key = (
            detector_shape,
            int(radial_deg.size),
            float(radial_deg[0]),
            float(radial_deg[-1]),
            int(azimuthal_deg.size),
            float(azimuthal_deg[0]),
            float(azimuthal_deg[-1]),
        )
        cached = self._cake_lut_cache.get(key)
        if cached is not None:
            return cached
        try:
            built = build_detector_to_cake_lut(
                detector_shape,
                radial_deg,
                azimuthal_deg,
                geometry,
                workers=workers,
            )
        except (MemoryError, RuntimeError):
            return None
        self._cake_lut_cache[key] = built
        return built


def convert_image_to_angle_space(
    image: np.ndarray,
    *,
    pixel_size_m: float,
    distance_m: float | None = None,
    center_row_px: float | None = None,
    center_col_px: float | None = None,
    poni1_m: float | None = None,
    poni2_m: float | None = None,
    poni_path: str | Path | None = None,
    npt_rad: int = 1000,
    npt_azim: int = 720,
    tth_min_deg: float = 0.0,
    tth_max_deg: float = 90.0,
    azimuth_min_deg: float = -180.0,
    azimuth_max_deg: float = 180.0,
    normalization: np.ndarray | None = None,
    correct_solid_angle: bool = False,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    engine: str = "auto",
    workers: int | str | None = "auto",
) -> DetectorCakeResult:
    """Convert one detector image directly into angle space.

    Inputs needed:
    - image
    - pixel size
    - distance
    - either center pixels or PONI values
    - optional ``rows``/``cols`` detector coordinates to limit integration
    """

    portable_geometry = build_geometry(
        pixel_size_m=pixel_size_m,
        distance_m=distance_m,
        center_row_px=center_row_px,
        center_col_px=center_col_px,
        poni1_m=poni1_m,
        poni2_m=poni2_m,
        poni_path=poni_path,
    )
    radial_deg, azimuthal_deg = build_angle_axes(
        npt_rad=npt_rad,
        npt_azim=npt_azim,
        tth_min_deg=tth_min_deg,
        tth_max_deg=tth_max_deg,
        azimuth_min_deg=azimuth_min_deg,
        azimuth_max_deg=azimuth_max_deg,
    )
    norm = normalization
    if norm is None and correct_solid_angle:
        norm = flat_solid_angle_normalization(np.asarray(image).shape, portable_geometry)
    return integrate_detector_to_cake_exact(
        image,
        radial_deg,
        azimuthal_deg,
        DetectorCakeGeometry(
            pixel_size_m=float(portable_geometry.pixel_size_m),
            distance_m=float(portable_geometry.distance_m),
            center_row_px=float(portable_geometry.center_row_px),
            center_col_px=float(portable_geometry.center_col_px),
        ),
        normalization=norm,
        rows=rows,
        cols=cols,
        engine=engine,
        workers=workers,
    )


def prepare_gui_phi_display(
    result: DetectorCakeResult,
    *,
    phi_min_deg: float = -180.0,
    phi_max_deg: float = 180.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cake_image, radial_deg, gui_phi_deg)`` for GUI-style plotting."""

    gui_phi = ((PHI_ZERO_OFFSET_DEGREES - np.asarray(result.azimuthal_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0
    order = np.argsort(gui_phi)
    gui_phi = gui_phi[order]
    cake = np.asarray(result.intensity, dtype=np.float64)[order, :]
    if float(phi_min_deg) <= float(phi_max_deg):
        mask = (gui_phi >= float(phi_min_deg)) & (gui_phi <= float(phi_max_deg))
    else:
        mask = (gui_phi >= float(phi_min_deg)) | (gui_phi <= float(phi_max_deg))
    return cake[mask, :], np.asarray(result.radial_deg, dtype=np.float64), gui_phi[mask]


__all__ = [
    "FastAzimuthalIntegrator",
    "PortableGeometry",
    "build_angle_axes",
    "build_geometry",
    "convert_image_to_angle_space",
    "detector_points_to_angles",
    "detector_pixel_angular_maps",
    "detector_two_theta_max_deg",
    "flat_solid_angle_normalization",
    "prepare_gui_phi_display",
]
