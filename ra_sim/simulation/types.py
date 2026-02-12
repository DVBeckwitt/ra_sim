"""Typed request/response models for diffraction simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DetectorGeometry:
    image_size: int
    av: float
    cv: float
    lambda_angstrom: float
    distance_m: float
    gamma_deg: float
    Gamma_deg: float
    chi_deg: float
    psi_deg: float
    psi_z_deg: float
    zs: float
    zb: float
    center: np.ndarray
    theta_initial_deg: float
    cor_angle_deg: float
    unit_x: np.ndarray
    n_detector: np.ndarray


@dataclass
class BeamSamples:
    beam_x_array: np.ndarray
    beam_y_array: np.ndarray
    theta_array: np.ndarray
    phi_array: np.ndarray
    wavelength_array: np.ndarray


@dataclass
class MosaicParams:
    sigma_mosaic_deg: float
    gamma_mosaic_deg: float
    eta: float


@dataclass
class DebyeWallerParams:
    x: float
    y: float


@dataclass
class SimulationRequest:
    miller: np.ndarray
    intensities: np.ndarray
    geometry: DetectorGeometry
    beam: BeamSamples
    mosaic: MosaicParams
    debye_waller: DebyeWallerParams
    n2: complex
    image_buffer: np.ndarray | None = None
    save_flag: int = 0
    record_status: bool = False
    thickness: float = 0.0


@dataclass
class SimulationResult:
    image: np.ndarray
    hit_tables: list[Any]
    q_data: np.ndarray
    q_count: np.ndarray
    all_status: np.ndarray
    miss_tables: list[Any]
    degeneracy: np.ndarray | None = None
