"""Helpers for GUI state persistence and background image switching."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np


def canonicalize_gui_state_background_path(path: object) -> str:
    """Return a stable normalized path key for background files."""

    return os.path.normcase(str(Path(str(path)).expanduser().resolve(strict=False)))


def background_files_match_loaded_state(
    file_paths: list[str],
    *,
    osc_files: list[str],
    background_images_native: list[object],
    background_images_display: list[object],
) -> bool:
    """Return whether the requested backgrounds already match the loaded state."""

    if not file_paths:
        return False
    if len(file_paths) != len(osc_files):
        return False
    if len(background_images_native) != len(file_paths):
        return False
    if len(background_images_display) != len(file_paths):
        return False
    requested_paths = [
        canonicalize_gui_state_background_path(path) for path in file_paths
    ]
    current_paths = [
        canonicalize_gui_state_background_path(path) for path in osc_files
    ]
    return requested_paths == current_paths


def load_background_files_for_state(
    file_paths: list[str],
    *,
    osc_files: list[str],
    background_images: list[object],
    background_images_native: list[object],
    background_images_display: list[object],
    select_index: int = 0,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
    set_background_display: Callable[[object], None] | None = None,
) -> dict[str, object] | None:
    """Load or reuse background images and return the updated background state."""

    normalized_paths = [
        str(Path(str(path)).expanduser())
        for path in file_paths
        if path is not None
    ]
    if not normalized_paths:
        return None

    if background_files_match_loaded_state(
        normalized_paths,
        osc_files=osc_files,
        background_images_native=background_images_native,
        background_images_display=background_images_display,
    ):
        index = max(0, min(int(select_index), len(background_images_native) - 1))
        current_background_image = background_images_native[index]
        current_background_display = background_images_display[index]
        if set_background_display is not None:
            set_background_display(current_background_display)
        return {
            "osc_files": list(normalized_paths),
            "background_images": list(background_images),
            "background_images_native": list(background_images_native),
            "background_images_display": list(background_images_display),
            "current_background_index": index,
            "current_background_image": current_background_image,
            "current_background_display": current_background_display,
        }

    loaded_native = [np.asarray(read_osc(path)) for path in normalized_paths]
    if not loaded_native:
        return None

    new_background_images = [np.array(img) for img in loaded_native]
    new_background_images_native = [np.array(img) for img in loaded_native]
    new_background_images_display = [
        np.rot90(img, display_rotate_k) for img in new_background_images_native
    ]
    index = max(0, min(int(select_index), len(new_background_images_native) - 1))
    current_background_image = new_background_images_native[index]
    current_background_display = new_background_images_display[index]
    if set_background_display is not None:
        set_background_display(current_background_display)
    return {
        "osc_files": list(normalized_paths),
        "background_images": new_background_images,
        "background_images_native": new_background_images_native,
        "background_images_display": new_background_images_display,
        "current_background_index": index,
        "current_background_image": current_background_image,
        "current_background_display": current_background_display,
    }
