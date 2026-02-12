"""Shared GUI state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AppState:
    """Minimal mutable state container for GUI controller/view coordination."""

    instrument_config: dict[str, Any] = field(default_factory=dict)
    file_paths: dict[str, Any] = field(default_factory=dict)
    image_size: int = 3000
    background_images_native: list[np.ndarray] = field(default_factory=list)
    background_images_display: list[np.ndarray] = field(default_factory=list)
    current_background_index: int = 0
