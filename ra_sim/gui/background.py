"""Background image cache and orientation helpers for the GUI runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import numpy as np

from ra_sim.debug_controls import retain_optional_cache
from ra_sim.gui.geometry_overlay import rotate_point_for_display as _rotate_point


def _read_background_image(
    file_path: str,
    *,
    read_osc: Callable[[str], object],
) -> np.ndarray:
    """Read one OSC background image and return it as a 2D numpy array."""

    raw_image = read_osc(file_path)
    image_array = np.asarray(raw_image)
    if image_array.ndim < 2:
        raise ValueError(
            f"Background '{Path(file_path).name}' is not a 2D image."
        )
    return image_array


def _retain_background_history(*, total_count: int) -> bool:
    return retain_optional_cache(
        "background_history",
        feature_needed=int(total_count) > 1,
    )


def _prune_background_history_entries(
    *,
    selected_index: int,
    background_images: list[object | None],
    background_images_native: list[object | None],
    background_images_display: list[object | None],
) -> None:
    total_count = len(background_images_native)
    if _retain_background_history(total_count=total_count):
        return
    for idx in range(total_count):
        if idx == int(selected_index):
            continue
        if idx < len(background_images):
            background_images[idx] = None
        background_images_native[idx] = None
        if idx < len(background_images_display):
            background_images_display[idx] = None


def initialize_background_cache(
    first_path: str,
    *,
    total_count: int,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
) -> dict[str, object]:
    """Initialize lazy background caches by loading only the first image."""

    if int(total_count) <= 0:
        raise ValueError("total_count must be positive")

    initial_native = _read_background_image(
        str(first_path),
        read_osc=read_osc,
    )
    background_images: list[object | None] = [None] * int(total_count)
    background_images_native: list[object | None] = [None] * int(total_count)
    background_images_display: list[object | None] = [None] * int(total_count)

    native_copy = np.array(initial_native)
    display_copy = np.rot90(native_copy, int(display_rotate_k))
    background_images[0] = native_copy
    background_images_native[0] = native_copy
    background_images_display[0] = display_copy
    _prune_background_history_entries(
        selected_index=0,
        background_images=background_images,
        background_images_native=background_images_native,
        background_images_display=background_images_display,
    )

    return {
        "background_images": background_images,
        "background_images_native": background_images_native,
        "background_images_display": background_images_display,
        "current_background_index": 0,
        "current_background_image": native_copy,
        "current_background_display": display_copy,
    }


def load_background_image_by_index(
    index: int,
    *,
    osc_files: Sequence[object],
    background_images: Sequence[object | None],
    background_images_native: Sequence[object | None],
    background_images_display: Sequence[object | None],
    display_rotate_k: int,
    read_osc: Callable[[str], object],
) -> dict[str, object]:
    """Return background arrays for *index*, loading from disk if needed."""

    idx = int(index)
    if idx < 0 or idx >= len(osc_files):
        raise IndexError(f"Background index out of range: {idx}")

    updated_background_images = list(background_images)
    updated_background_images_native = list(background_images_native)
    updated_background_images_display = list(background_images_display)

    native_cached = updated_background_images_native[idx]
    display_cached = updated_background_images_display[idx]
    if native_cached is not None and display_cached is not None:
        return {
            "background_images": updated_background_images,
            "background_images_native": updated_background_images_native,
            "background_images_display": updated_background_images_display,
            "background_image": np.asarray(native_cached),
            "background_display": np.asarray(display_cached),
        }

    first_native = updated_background_images_native[0]
    if first_native is None:
        first_path = str(osc_files[0])
        first_native_array = np.array(
            _read_background_image(first_path, read_osc=read_osc)
        )
        updated_background_images[0] = first_native_array
        updated_background_images_native[0] = first_native_array
        updated_background_images_display[0] = np.rot90(
            first_native_array,
            int(display_rotate_k),
        )
        first_native = first_native_array

    file_path_local = str(osc_files[idx])
    image_array = _read_background_image(file_path_local, read_osc=read_osc)
    expected_shape = tuple(int(v) for v in np.asarray(first_native).shape[:2])
    image_shape = tuple(int(v) for v in image_array.shape[:2])
    if image_shape != expected_shape:
        raise ValueError(
            f"Background '{Path(file_path_local).name}' has shape {image_shape}, "
            f"expected {expected_shape}."
        )

    native = np.array(image_array)
    display = np.rot90(native, int(display_rotate_k))
    updated_background_images[idx] = native
    updated_background_images_native[idx] = native
    updated_background_images_display[idx] = display
    _prune_background_history_entries(
        selected_index=idx,
        background_images=updated_background_images,
        background_images_native=updated_background_images_native,
        background_images_display=updated_background_images_display,
    )

    return {
        "background_images": updated_background_images,
        "background_images_native": updated_background_images_native,
        "background_images_display": updated_background_images_display,
        "background_image": native,
        "background_display": display,
    }


def get_current_background_native(
    *,
    osc_files: Sequence[object],
    background_images: Sequence[object | None],
    background_images_native: Sequence[object | None],
    background_images_display: Sequence[object | None],
    current_background_index: int,
    current_background_image: object,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
) -> dict[str, object]:
    """Return the current native background and any updated lazy caches."""

    if 0 <= int(current_background_index) < len(background_images_native):
        try:
            return load_background_image_by_index(
                int(current_background_index),
                osc_files=osc_files,
                background_images=background_images,
                background_images_native=background_images_native,
                background_images_display=background_images_display,
                display_rotate_k=display_rotate_k,
                read_osc=read_osc,
            )
        except Exception:
            return {
                "background_images": list(background_images),
                "background_images_native": list(background_images_native),
                "background_images_display": list(background_images_display),
                "background_image": np.asarray(current_background_image),
                "background_display": None,
            }
    return {
        "background_images": list(background_images),
        "background_images_native": list(background_images_native),
        "background_images_display": list(background_images_display),
        "background_image": current_background_image,
        "background_display": None,
    }


def get_current_background_display(
    *,
    osc_files: Sequence[object],
    background_images: Sequence[object | None],
    background_images_native: Sequence[object | None],
    background_images_display: Sequence[object | None],
    current_background_index: int,
    current_background_image: object,
    current_background_display: object,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
) -> dict[str, object]:
    """Return the current display background and any updated lazy caches."""

    if 0 <= int(current_background_index) < len(background_images_display):
        try:
            return load_background_image_by_index(
                int(current_background_index),
                osc_files=osc_files,
                background_images=background_images,
                background_images_native=background_images_native,
                background_images_display=background_images_display,
                display_rotate_k=display_rotate_k,
                read_osc=read_osc,
            )
        except Exception:
            return {
                "background_images": list(background_images),
                "background_images_native": list(background_images_native),
                "background_images_display": list(background_images_display),
                "background_image": None,
                "background_display": np.asarray(current_background_display),
            }
    return {
        "background_images": list(background_images),
        "background_images_native": list(background_images_native),
        "background_images_display": list(background_images_display),
        "background_image": np.asarray(current_background_image),
        "background_display": np.rot90(
            np.asarray(current_background_image),
            int(display_rotate_k),
        ),
    }


def apply_background_backend_orientation(
    image: np.ndarray | None,
    *,
    flip_x: bool,
    flip_y: bool,
    rotation_k: int,
) -> np.ndarray | None:
    """Apply backend-only debug orientation transforms to a background image."""

    if image is None:
        return None
    oriented = np.asarray(image)
    if bool(flip_y):
        oriented = np.flip(oriented, axis=0)
    if bool(flip_x):
        oriented = np.flip(oriented, axis=1)
    k_mod = int(rotation_k) % 4
    if k_mod:
        oriented = np.rot90(oriented, k_mod)
    return oriented


def background_backend_point_to_native_coords(
    col: float,
    row: float,
    *,
    native_shape: Sequence[int] | None,
    flip_x: bool,
    flip_y: bool,
    rotation_k: int,
) -> tuple[float | None, float | None]:
    """Map one backend-oriented detector point back into native detector space.

    This helper is coordinate-only. If a future inverse path needs to move
    caked intensities or a reconstructed detector image back through the same
    backend orientation, restore detector-space intensity weighting first
    (for example, undo solid-angle correction before reorienting the array).
    """

    if native_shape is None:
        return None, None
    try:
        height = int(native_shape[0])
        width = int(native_shape[1])
    except Exception:
        return None, None
    if height <= 0 or width <= 0:
        return None, None
    if not (np.isfinite(col) and np.isfinite(row)):
        return None, None

    k_mod = int(rotation_k) % 4
    backend_shape = (width, height) if (k_mod % 2) else (height, width)
    try:
        native_col, native_row = _rotate_point(
            float(col),
            float(row),
            backend_shape,
            -k_mod,
        )
    except Exception:
        return None, None

    if bool(flip_x):
        native_col = float(width - 1.0 - native_col)
    if bool(flip_y):
        native_row = float(height - 1.0 - native_row)
    if not (np.isfinite(native_col) and np.isfinite(native_row)):
        return None, None
    return float(native_col), float(native_row)


def load_background_files(
    file_paths: Sequence[object],
    *,
    image_size: int,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
    select_index: int = 0,
) -> dict[str, object]:
    """Replace the loaded background set from selected OSC file paths."""

    normalized_paths: list[str] = []
    for raw_path in file_paths:
        text_path = str(raw_path).strip().strip('"').strip("'")
        if not text_path:
            continue
        candidate = Path(text_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"File not found: {candidate}")
        normalized_paths.append(str(candidate))

    if not normalized_paths:
        raise ValueError("No background files selected.")

    expected_shape = (int(image_size), int(image_size))
    loaded_native: list[np.ndarray] = []
    for file_path_local in normalized_paths:
        try:
            image_array = _read_background_image(
                file_path_local,
                read_osc=read_osc,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read background file '{file_path_local}': {exc}"
            ) from exc

        image_shape = tuple(int(v) for v in image_array.shape[:2])
        if image_shape != expected_shape:
            raise ValueError(
                f"Background '{Path(file_path_local).name}' has shape {image_shape}, "
                f"expected {expected_shape}."
            )
        loaded_native.append(np.array(image_array))

    background_images = [np.array(img) for img in loaded_native]
    background_images_native = [np.array(img) for img in loaded_native]
    background_images_display = [
        np.rot90(img, int(display_rotate_k)) for img in background_images_native
    ]

    index = max(0, min(int(select_index), len(background_images_native) - 1))
    current_background_image = background_images_native[index]
    current_background_display = background_images_display[index]
    _prune_background_history_entries(
        selected_index=index,
        background_images=background_images,
        background_images_native=background_images_native,
        background_images_display=background_images_display,
    )
    return {
        "osc_files": list(normalized_paths),
        "background_images": background_images,
        "background_images_native": background_images_native,
        "background_images_display": background_images_display,
        "current_background_index": index,
        "current_background_image": current_background_image,
        "current_background_display": current_background_display,
    }


def switch_background(
    *,
    osc_files: Sequence[object],
    background_images: Sequence[object | None],
    background_images_native: Sequence[object | None],
    background_images_display: Sequence[object | None],
    current_background_index: int,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
) -> dict[str, object]:
    """Advance to the next background image, loading it lazily if needed."""

    if not osc_files:
        raise ValueError("No background images loaded.")

    next_index = (int(current_background_index) + 1) % len(osc_files)
    updated = load_background_image_by_index(
        next_index,
        osc_files=osc_files,
        background_images=background_images,
        background_images_native=background_images_native,
        background_images_display=background_images_display,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
    )
    return {
        "background_images": list(updated["background_images"]),
        "background_images_native": list(updated["background_images_native"]),
        "background_images_display": list(updated["background_images_display"]),
        "current_background_index": next_index,
        "current_background_image": updated["background_image"],
        "current_background_display": updated["background_display"],
    }
