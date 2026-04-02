"""Temporary viewer for the first background image referenced by a GUI-state file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ra_sim.gui import background as gui_background
from ra_sim.gui import background_manager as gui_background_manager
from ra_sim.io.file_parsing import parse_poni_file
from ra_sim.io.data_loading import load_gui_state_file
from ra_sim.io.osc_reader import read_osc
from ra_sim.config.loader import get_instrument_config, get_path

try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError("pyFAI is required to build the phi-vs-2theta view.") from exc


DEFAULT_STATE_PATH = Path(r"C:\Users\Kenpo\.local\share\ra_sim\init.json")
DISPLAY_ROTATE_K = -1
DEFAULT_CAKED_RADIAL_BINS = 1000
DEFAULT_CAKED_AZIMUTH_BINS = 720
PHI_ZERO_OFFSET_DEGREES = -90.0


def _finite_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(parsed) if np.isfinite(parsed) else float(default)


def _valid_range(min_value: object, max_value: object) -> tuple[float, float]:
    min_val = _finite_float(min_value, 0.0)
    max_val = _finite_float(max_value, max(min_val + 1.0, 1.0))
    if max_val <= min_val:
        max_val = min_val + max(abs(min_val) * 1.0e-3, 1.0)
    return float(min_val), float(max_val)


def _adjust_phi_zero(phi_values) -> np.ndarray:
    return PHI_ZERO_OFFSET_DEGREES - np.asarray(phi_values)


def _wrap_phi_range(phi_values) -> np.ndarray:
    wrapped = ((np.asarray(phi_values) + 180.0) % 360.0) - 180.0
    return wrapped


def _auto_caked_limits(image) -> tuple[float, float]:
    if image is None:
        return 0.0, 1.0

    finite_mask = np.isfinite(image)
    if not np.any(finite_mask):
        return 0.0, 1.0

    finite_vals = np.asarray(image, dtype=float)[finite_mask]
    vmin = float(np.nanmin(finite_vals))
    vmax = float(np.nanmax(finite_vals))
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return 0.0, 1.0
    if np.isclose(vmin, vmax):
        if vmin == 0.0:
            vmax = 1.0
        else:
            spread = abs(vmax) * 1.0e-3 or 1.0
            vmin -= spread
            vmax += spread
    return float(vmin), float(vmax)


def _build_azimuthal_integrator(
    *,
    center_x: float,
    center_y: float,
    distance_m: float,
    wavelength_m: float,
    pixel_size_m: float,
) -> AzimuthalIntegrator:
    return AzimuthalIntegrator(
        dist=float(distance_m),
        poni1=float(center_x) * float(pixel_size_m),
        poni2=float(center_y) * float(pixel_size_m),
        rot1=0.0,
        rot2=0.0,
        rot3=0.0,
        wavelength=float(wavelength_m),
        pixel1=float(pixel_size_m),
        pixel2=float(pixel_size_m),
    )


def _prepare_caked_display_payload(res2) -> dict[str, object] | None:
    if res2 is None:
        return None

    caked_img = np.asarray(res2.intensity, dtype=float)
    radial_vals = np.asarray(res2.radial, dtype=float)
    azimuth_vals = _wrap_phi_range(_adjust_phi_zero(res2.azimuthal))

    if azimuth_vals.size:
        azimuth_order = np.argsort(azimuth_vals)
        azimuth_vals = azimuth_vals[azimuth_order]
        caked_img = caked_img[azimuth_order, :]

    radial_mask = (radial_vals >= 0.0) & (radial_vals <= 90.0)
    if np.any(radial_mask):
        radial_vals = radial_vals[radial_mask]
        caked_img = caked_img[:, radial_mask]

    if radial_vals.size:
        radial_min = float(np.min(radial_vals))
        radial_max = float(np.max(radial_vals))
    else:
        radial_min, radial_max = 0.0, 90.0

    if azimuth_vals.size:
        azimuth_min = float(np.min(azimuth_vals))
        azimuth_max = float(np.max(azimuth_vals))
    else:
        azimuth_min, azimuth_max = -180.0, 180.0

    return {
        "image": np.asarray(caked_img, dtype=float),
        "radial": np.asarray(radial_vals, dtype=float),
        "azimuth": np.asarray(azimuth_vals, dtype=float),
        "extent": [radial_min, radial_max, azimuth_min, azimuth_max],
    }


def _build_background_caked_payload(
    *,
    backend_image: np.ndarray,
    center_x: float,
    center_y: float,
    distance_m: float,
    wavelength_m: float,
    pixel_size_m: float,
) -> dict[str, object] | None:
    ai = _build_azimuthal_integrator(
        center_x=center_x,
        center_y=center_y,
        distance_m=distance_m,
        wavelength_m=wavelength_m,
        pixel_size_m=pixel_size_m,
    )
    res2 = ai.integrate2d(
        np.asarray(backend_image, dtype=np.float64),
        npt_rad=DEFAULT_CAKED_RADIAL_BINS,
        npt_azim=DEFAULT_CAKED_AZIMUTH_BINS,
        correctSolidAngle=True,
        method="lut",
        unit="2th_deg",
    )
    return _prepare_caked_display_payload(res2)


def _get_detector_angular_maps(
    ai: AzimuthalIntegrator,
    *,
    detector_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if ai is None:
        return None, None
    if len(detector_shape) < 2 or detector_shape[0] <= 0 or detector_shape[1] <= 0:
        return None, None

    try:
        two_theta = ai.twoThetaArray(shape=detector_shape, unit="2th_deg")
    except TypeError:
        two_theta = np.rad2deg(ai.twoThetaArray(shape=detector_shape))

    try:
        phi_vals = ai.chiArray(shape=detector_shape, unit="deg")
    except TypeError:
        phi_vals = np.rad2deg(ai.chiArray(shape=detector_shape))

    return np.asarray(two_theta, dtype=float), _adjust_phi_zero(phi_vals)


def _get_detector_solid_angle_map(
    ai: AzimuthalIntegrator,
    *,
    detector_shape: tuple[int, int],
) -> np.ndarray | None:
    if ai is None:
        return None
    if len(detector_shape) < 2 or detector_shape[0] <= 0 or detector_shape[1] <= 0:
        return None

    try:
        solid_angle = ai.solidAngleArray(shape=detector_shape)
    except TypeError:
        solid_angle = ai.solidAngleArray(detector_shape)
    return np.asarray(solid_angle, dtype=float)


def _axis_values_to_fractional_indices(
    values,
    axis_values,
) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    mapped = np.full(values_arr.shape, np.nan, dtype=float)

    axis_arr = np.asarray(axis_values, dtype=float).reshape(-1)
    if axis_arr.size == 0:
        return mapped

    finite_axis_mask = np.isfinite(axis_arr)
    if not np.any(finite_axis_mask):
        return mapped

    axis_arr = axis_arr[finite_axis_mask]
    axis_idx = np.flatnonzero(finite_axis_mask).astype(float)

    unique_axis, unique_positions = np.unique(axis_arr, return_index=True)
    axis_used = unique_axis
    idx_used = axis_idx[unique_positions]
    if axis_used.size == 0:
        return mapped
    if axis_used.size == 1:
        close_mask = np.isfinite(values_arr) & np.isclose(values_arr, axis_used[0])
        mapped[close_mask] = idx_used[0]
        return mapped

    value_mask = (
        np.isfinite(values_arr)
        & (values_arr >= float(axis_used[0]))
        & (values_arr <= float(axis_used[-1]))
    )
    if not np.any(value_mask):
        return mapped

    mapped[value_mask] = np.interp(values_arr[value_mask], axis_used, idx_used)
    return mapped


def _sample_image_bilinear(
    image: np.ndarray,
    row_coords,
    col_coords,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    rows = np.asarray(row_coords, dtype=float)
    cols = np.asarray(col_coords, dtype=float)
    sampled = np.full(rows.shape, float(fill_value), dtype=float)
    if img.ndim != 2 or img.size == 0:
        return sampled

    height, width = img.shape[:2]
    valid = (
        np.isfinite(rows)
        & np.isfinite(cols)
        & (rows >= 0.0)
        & (cols >= 0.0)
        & (rows <= float(height - 1))
        & (cols <= float(width - 1))
    )
    if not np.any(valid):
        return sampled

    row_vals = rows[valid]
    col_vals = cols[valid]
    row0 = np.floor(row_vals).astype(int)
    col0 = np.floor(col_vals).astype(int)
    row1 = np.clip(row0 + 1, 0, height - 1)
    col1 = np.clip(col0 + 1, 0, width - 1)

    row_frac = row_vals - row0
    col_frac = col_vals - col0
    v00 = img[row0, col0]
    v10 = img[row1, col0]
    v01 = img[row0, col1]
    v11 = img[row1, col1]

    weights = np.stack(
        [
            (1.0 - row_frac) * (1.0 - col_frac),
            row_frac * (1.0 - col_frac),
            (1.0 - row_frac) * col_frac,
            row_frac * col_frac,
        ],
        axis=0,
    )
    values = np.stack([v00, v10, v01, v11], axis=0)
    finite = np.isfinite(values)
    weighted_sum = np.sum(np.where(finite, weights * values, 0.0), axis=0)
    weight_sum = np.sum(np.where(finite, weights, 0.0), axis=0)

    valid_samples = np.full(row_vals.shape, float(fill_value), dtype=float)
    np.divide(
        weighted_sum,
        weight_sum,
        out=valid_samples,
        where=weight_sum > 0.0,
    )
    sampled[valid] = valid_samples
    return sampled


def _apply_inverse_background_backend_orientation(
    image: np.ndarray | None,
    *,
    flip_x: bool,
    flip_y: bool,
    rotation_k: int,
) -> np.ndarray | None:
    if image is None:
        return None
    oriented = np.asarray(image)
    k_mod = int(rotation_k) % 4
    if k_mod:
        oriented = np.rot90(oriented, -k_mod)
    if bool(flip_x):
        oriented = np.flip(oriented, axis=1)
    if bool(flip_y):
        oriented = np.flip(oriented, axis=0)
    return oriented


def _reconstruct_detector_from_caked(
    *,
    caked_image: np.ndarray,
    radial_axis,
    azimuth_axis,
    ai: AzimuthalIntegrator,
    detector_shape: tuple[int, int],
) -> np.ndarray:
    two_theta_map, phi_map = _get_detector_angular_maps(
        ai,
        detector_shape=detector_shape,
    )
    if two_theta_map is None or phi_map is None:
        raise RuntimeError("Failed to build detector angular maps for inverse caking.")

    caked_cols = _axis_values_to_fractional_indices(two_theta_map, radial_axis)
    caked_rows = _axis_values_to_fractional_indices(
        _wrap_phi_range(phi_map),
        azimuth_axis,
    )
    return _sample_image_bilinear(
        np.asarray(caked_image, dtype=float),
        caked_rows,
        caked_cols,
    )


def _load_first_background_view(state_path: Path) -> dict[str, object]:
    payload = load_gui_state_file(state_path)
    state = payload.get("state", {})
    if not isinstance(state, dict):
        raise ValueError(f"State payload is invalid: {state_path}")

    files = state.get("files", {})
    variables = state.get("variables", {})
    flags = state.get("flags", {})
    if not isinstance(files, dict):
        raise ValueError(f"State file has no valid 'files' section: {state_path}")
    if not isinstance(variables, dict):
        variables = {}
    if not isinstance(flags, dict):
        flags = {}

    background_files = files.get("background_files", [])
    if not isinstance(background_files, list) or not background_files:
        raise ValueError(f"No background files were found in: {state_path}")

    first_background_path = Path(str(background_files[0])).expanduser()
    if not first_background_path.is_file():
        raise FileNotFoundError(f"Background file not found: {first_background_path}")

    native_image = np.asarray(read_osc(str(first_background_path)))
    cache_state = gui_background.initialize_background_cache(
        str(first_background_path),
        total_count=len(background_files),
        display_rotate_k=DISPLAY_ROTATE_K,
        read_osc=read_osc,
    )
    display_image = np.asarray(cache_state["current_background_display"])
    backend_flip_x = bool(flags.get("background_backend_flip_x", False))
    backend_flip_y = bool(flags.get("background_backend_flip_y", False))
    backend_rotation_k = int(flags.get("background_backend_rotation_k", 3))
    backend_image = np.asarray(
        gui_background.apply_background_backend_orientation(
            native_image,
            flip_x=backend_flip_x,
            flip_y=backend_flip_y,
            rotation_k=backend_rotation_k,
        )
    )
    defaults = gui_background_manager.resolve_background_display_defaults(display_image)

    background_vmin, background_vmax = _valid_range(
        variables.get("background_min_var"),
        variables.get("background_max_var"),
    )
    if not bool(flags.get("background_limits_user_override", False)):
        background_vmin, background_vmax = _valid_range(
            defaults.vmin_default,
            defaults.vmax_default,
        )

    instrument = get_instrument_config().get("instrument", {})
    detector_config = instrument.get("detector", {})
    geometry_defaults = instrument.get("geometry_defaults", {})
    pixel_size_m = _finite_float(
        detector_config.get("pixel_size_m"),
        1.0e-4,
    )
    image_size = int(native_image.shape[0])

    poni_path = Path(str(get_path("geometry_poni"))).expanduser()
    poni = parse_poni_file(poni_path)
    poni1 = _finite_float(poni.get("Poni1"), geometry_defaults.get("poni1_m", 0.0))
    poni2 = _finite_float(poni.get("Poni2"), geometry_defaults.get("poni2_m", 0.0))
    default_center_x = _finite_float(
        variables.get("center_x_var"),
        poni2 / pixel_size_m,
    )
    default_center_y = _finite_float(
        variables.get("center_y_var"),
        image_size - (poni1 / pixel_size_m),
    )
    distance_m = _finite_float(
        variables.get("corto_detector_var"),
        poni.get("Dist", geometry_defaults.get("distance_m", 0.075)),
    )
    wavelength_m = _finite_float(
        poni.get("Wavelength"),
        geometry_defaults.get("wavelength_m", 1.0e-10),
    )
    ai = _build_azimuthal_integrator(
        center_x=default_center_x,
        center_y=default_center_y,
        distance_m=distance_m,
        wavelength_m=wavelength_m,
        pixel_size_m=pixel_size_m,
    )

    caked_payload = _build_background_caked_payload(
        backend_image=backend_image,
        center_x=default_center_x,
        center_y=default_center_y,
        distance_m=distance_m,
        wavelength_m=wavelength_m,
        pixel_size_m=pixel_size_m,
    )
    if caked_payload is None:
        raise RuntimeError("Failed to build the caked phi-vs-2theta view.")

    caked_defaults = _auto_caked_limits(caked_payload["image"])
    simulation_override = bool(flags.get("simulation_limits_user_override", False))
    if simulation_override:
        caked_vmin, caked_vmax = _valid_range(
            variables.get("simulation_min_var"),
            variables.get("simulation_max_var"),
        )
    else:
        caked_vmin, caked_vmax = _valid_range(*caked_defaults)

    reconstructed_backend = _reconstruct_detector_from_caked(
        caked_image=np.asarray(caked_payload["image"], dtype=float),
        radial_axis=caked_payload["radial"],
        azimuth_axis=caked_payload["azimuth"],
        ai=ai,
        detector_shape=tuple(int(v) for v in backend_image.shape[:2]),
    )
    solid_angle_map = _get_detector_solid_angle_map(
        ai,
        detector_shape=tuple(int(v) for v in backend_image.shape[:2]),
    )
    if solid_angle_map is not None:
        # ``integrate2d(..., correctSolidAngle=True)`` stores intensities in a
        # solid-angle-normalized space. Multiply by the detector solid angle to
        # compare the inverse-mapped image back in detector-count space.
        reconstructed_backend = np.asarray(reconstructed_backend, dtype=float) * np.asarray(
            solid_angle_map,
            dtype=float,
        )
    reconstructed_native = _apply_inverse_background_backend_orientation(
        reconstructed_backend,
        flip_x=backend_flip_x,
        flip_y=backend_flip_y,
        rotation_k=backend_rotation_k,
    )
    reconstructed_display = np.rot90(
        np.asarray(reconstructed_native, dtype=float),
        DISPLAY_ROTATE_K,
    )
    reconstructed_vmin, reconstructed_vmax = _valid_range(
        background_vmin,
        background_vmax,
    )

    transparency = _finite_float(variables.get("background_transparency_var"), 0.0)
    transparency = min(1.0, max(0.0, transparency))
    alpha = 1.0 - transparency

    height, width = display_image.shape[:2]
    return {
        "background_path": first_background_path,
        "display_image": display_image,
        "width": int(width),
        "height": int(height),
        "vmin": float(background_vmin),
        "vmax": float(background_vmax),
        "alpha": float(alpha),
        "caked_image": np.asarray(caked_payload["image"], dtype=float),
        "caked_radial": np.asarray(caked_payload["radial"], dtype=float),
        "caked_azimuth": np.asarray(caked_payload["azimuth"], dtype=float),
        "caked_extent": list(caked_payload["extent"]),
        "caked_vmin": float(caked_vmin),
        "caked_vmax": float(caked_vmax),
        "reconstructed_display_image": np.asarray(reconstructed_display, dtype=float),
        "reconstructed_vmin": float(reconstructed_vmin),
        "reconstructed_vmax": float(reconstructed_vmax),
    }


def _build_figure(view_data: dict[str, object]):
    detector_image = np.asarray(view_data["display_image"])
    caked_image = np.asarray(view_data["caked_image"])
    reconstructed_image = np.asarray(view_data["reconstructed_display_image"])
    width = int(view_data["width"])
    height = int(view_data["height"])
    radial_min, radial_max, azimuth_min, azimuth_max = [
        float(value) for value in view_data["caked_extent"]
    ]

    fig, (detector_ax, caked_ax, reconstructed_ax) = plt.subplots(
        1,
        3,
        num="RA-SIM Background Preview",
        figsize=(21, 6),
        constrained_layout=True,
    )

    detector_ax.set_aspect("auto")
    detector_ax.imshow(
        detector_image,
        cmap="turbo",
        origin="upper",
        aspect="auto",
        zorder=0,
        alpha=float(view_data["alpha"]),
        vmin=float(view_data["vmin"]),
        vmax=float(view_data["vmax"]),
        extent=[0, width, height, 0],
    )
    detector_ax.set_xlim(0, width)
    detector_ax.set_ylim(height, 0)
    detector_ax.set_title("Simulated Diffraction Pattern")
    detector_ax.set_xlabel("X (pixels)")
    detector_ax.set_ylabel("Y (pixels)")

    caked_ax.set_aspect("auto")
    caked_ax.imshow(
        caked_image,
        cmap="turbo",
        origin="lower",
        aspect="auto",
        zorder=0,
        vmin=float(view_data["caked_vmin"]),
        vmax=float(view_data["caked_vmax"]),
        extent=[radial_min, radial_max, azimuth_min, azimuth_max],
    )
    caked_ax.set_xlim(radial_min, radial_max)
    caked_ax.set_ylim(azimuth_min, azimuth_max)
    caked_ax.set_title("2D Caked Integration")
    caked_ax.set_xlabel("2θ (degrees)")
    caked_ax.set_ylabel("φ (degrees)")

    reconstructed_ax.set_aspect("auto")
    reconstructed_ax.imshow(
        np.ma.masked_invalid(reconstructed_image),
        cmap="turbo",
        origin="upper",
        aspect="auto",
        zorder=0,
        vmin=float(view_data["reconstructed_vmin"]),
        vmax=float(view_data["reconstructed_vmax"]),
        extent=[0, width, height, 0],
    )
    reconstructed_ax.set_xlim(0, width)
    reconstructed_ax.set_ylim(height, 0)
    reconstructed_ax.set_title("Inverse From 2D Caked")
    reconstructed_ax.set_xlabel("X (pixels)")
    reconstructed_ax.set_ylabel("Y (pixels)")

    return fig, (detector_ax, caked_ax, reconstructed_ax)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Display the first background image from a saved RA-SIM GUI-state file "
            "using the same detector-view presentation as the GUI, alongside "
            "the corresponding phi-vs-2theta caked view and its inverse detector "
            "reconstruction."
        )
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help=f"GUI-state JSON to read (default: {DEFAULT_STATE_PATH})",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path to save before showing.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the figure and optionally save it without opening the viewer window.",
    )
    args = parser.parse_args()

    state_path = args.state.expanduser()
    view_data = _load_first_background_view(state_path)
    fig, _ax = _build_figure(view_data)

    if args.save is not None:
        output_path = args.save.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved preview to: {output_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
