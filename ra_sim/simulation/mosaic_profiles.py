
"""Random mosaic block profile generation for diffraction simulations."""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.special import ndtri


_PROFILE_DIMENSIONS = 5
_UNIT_CLIP_EPS = np.finfo(np.float64).eps
_AUTO_CLUSTER_MIN_SAMPLES = 96
_AUTO_CLUSTER_MIN_CLUSTERS = 16
_AUTO_CLUSTER_MAX_CLUSTERS = 64
_AUTO_CLUSTER_MULTIPLIER = 3.0
_AUTO_CLUSTER_MIN_REDUCTION = 0.75
_AUTO_CLUSTER_KMEANS_ITERS = 12
RANDOM_GAUSSIAN_SAMPLING = "random_gaussian"
STRATIFIED_GAUSSIAN_SAMPLING = "stratified_gaussian"


def _as_profile_matrix(
    beam_x_array: np.ndarray,
    beam_y_array: np.ndarray,
    theta_array: np.ndarray,
    phi_array: np.ndarray,
    wavelength_array: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        (
            np.asarray(beam_x_array, dtype=np.float64).reshape(-1),
            np.asarray(beam_y_array, dtype=np.float64).reshape(-1),
            np.asarray(theta_array, dtype=np.float64).reshape(-1),
            np.asarray(phi_array, dtype=np.float64).reshape(-1),
            np.asarray(wavelength_array, dtype=np.float64).reshape(-1),
        )
    )


def _normalized_profile_features(profile_matrix: np.ndarray) -> np.ndarray:
    centered = profile_matrix - np.mean(profile_matrix, axis=0, keepdims=True)
    scales = np.std(centered, axis=0, keepdims=True)
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    return centered / scales


def _choose_cluster_count(
    sample_count: int,
    *,
    max_clusters: int,
) -> int:
    if sample_count < _AUTO_CLUSTER_MIN_SAMPLES:
        return sample_count
    target = int(round(np.sqrt(float(sample_count)) * _AUTO_CLUSTER_MULTIPLIER))
    target = max(_AUTO_CLUSTER_MIN_CLUSTERS, target)
    target = min(int(max_clusters), sample_count)
    if target >= int(np.ceil(_AUTO_CLUSTER_MIN_REDUCTION * float(sample_count))):
        return sample_count
    return max(target, 1)


def _fallback_cluster_labels(features: np.ndarray, cluster_count: int) -> np.ndarray:
    centered = features - np.mean(features, axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        components = centered @ vh[: min(2, vh.shape[0])].T
    except np.linalg.LinAlgError:
        components = centered[:, : min(2, centered.shape[1])]

    if components.ndim == 1:
        components = components[:, None]
    if components.shape[1] > 1:
        order = np.lexsort((components[:, 1], components[:, 0]))
    else:
        order = np.argsort(components[:, 0], kind="mergesort")

    labels = np.empty(features.shape[0], dtype=np.int64)
    for label, members in enumerate(np.array_split(order, cluster_count)):
        labels[members] = label
    return labels


def cluster_beam_profiles(
    beam_x_array: np.ndarray,
    beam_y_array: np.ndarray,
    theta_array: np.ndarray,
    phi_array: np.ndarray,
    wavelength_array: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    max_clusters: int = _AUTO_CLUSTER_MAX_CLUSTERS,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compress beam samples into weighted representative clusters.

    Returns clustered beam arrays, per-cluster weights, a raw-to-cluster map,
    and representative raw indices for each cluster.
    """

    profiles = _as_profile_matrix(
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        wavelength_array,
    )
    sample_count = int(profiles.shape[0])
    if sample_count == 0:
        empty = np.empty((0,), dtype=np.float64)
        empty_i = np.empty((0,), dtype=np.int64)
        return empty, empty, empty, empty, empty, empty, empty_i, empty_i

    raw_weights = np.ones(sample_count, dtype=np.float64)
    if sample_weights is not None:
        raw_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if raw_weights.shape[0] != sample_count:
            raise ValueError("sample_weights must match the beam sample count")
        raw_weights = np.where(np.isfinite(raw_weights) & (raw_weights > 0.0), raw_weights, 0.0)

    cluster_count = _choose_cluster_count(sample_count, max_clusters=max_clusters)
    if cluster_count >= sample_count:
        identity = np.arange(sample_count, dtype=np.int64)
        return (
            profiles[:, 0].copy(),
            profiles[:, 1].copy(),
            profiles[:, 2].copy(),
            profiles[:, 3].copy(),
            profiles[:, 4].copy(),
            raw_weights.copy(),
            identity,
            identity,
        )

    features = _normalized_profile_features(profiles)
    try:
        _, labels = kmeans2(
            features,
            cluster_count,
            iter=_AUTO_CLUSTER_KMEANS_ITERS,
            minit="points",
            seed=0,
        )
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    except Exception:
        labels = _fallback_cluster_labels(features, cluster_count)

    unique_labels = np.unique(labels)
    compressed_count = int(unique_labels.size)
    clustered_profiles = np.zeros((compressed_count, profiles.shape[1]), dtype=np.float64)
    clustered_weights = np.zeros(compressed_count, dtype=np.float64)
    raw_to_cluster = np.empty(sample_count, dtype=np.int64)
    cluster_to_rep = np.empty(compressed_count, dtype=np.int64)

    for cluster_idx, label in enumerate(unique_labels):
        members = np.flatnonzero(labels == label)
        raw_to_cluster[members] = cluster_idx
        member_weights = raw_weights[members]
        total_weight = float(np.sum(member_weights))
        if total_weight <= 0.0:
            total_weight = float(members.size)
            member_weights = np.ones(members.size, dtype=np.float64)
        clustered_weights[cluster_idx] = total_weight
        clustered_profiles[cluster_idx, :] = np.sum(
            profiles[members] * member_weights[:, None],
            axis=0,
        ) / total_weight
        deltas = features[members] - np.mean(features[members], axis=0, keepdims=True)
        dist_sq = np.sum(deltas * deltas, axis=1)
        cluster_to_rep[cluster_idx] = int(members[int(np.argmin(dist_sq))])

    return (
        clustered_profiles[:, 0],
        clustered_profiles[:, 1],
        clustered_profiles[:, 2],
        clustered_profiles[:, 3],
        clustered_profiles[:, 4],
        clustered_weights,
        raw_to_cluster,
        cluster_to_rep,
    )


def _coerce_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _latin_hypercube_unit_samples(
    num_points: int,
    dim: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``num_points`` Latin-hypercube samples in ``[0, 1)^dim``."""

    if num_points <= 0:
        return np.empty((0, dim), dtype=np.float64)

    unit = np.empty((num_points, dim), dtype=np.float64)
    for axis in range(dim):
        perm = rng.permutation(num_points).astype(np.float64)
        unit[:, axis] = (perm + rng.random(num_points)) / float(num_points)
    return unit


def _gaussian_offsets_from_unit(unit: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(unit, dtype=np.float64), _UNIT_CLIP_EPS, 1.0 - _UNIT_CLIP_EPS)
    return ndtri(clipped)


def sample_stratified_gaussian_1d(
    num_samples: int,
    *,
    mean: float,
    sigma: float,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Return one equal-weight 1D stratified Gaussian sample array."""

    sample_count = max(int(num_samples), 0)
    if sample_count == 0:
        return np.empty((0,), dtype=np.float64)

    mean_value = float(mean)
    sigma_value = float(sigma)
    if not np.isfinite(mean_value):
        raise ValueError("mean must be finite")
    if not np.isfinite(sigma_value):
        raise ValueError("sigma must be finite")
    if sigma_value < 0.0:
        raise ValueError("sigma must be non-negative")
    if sigma_value == 0.0:
        return np.full(sample_count, mean_value, dtype=np.float64)

    rng_obj = _coerce_rng(rng)
    unit = (
        np.arange(sample_count, dtype=np.float64) + rng_obj.random(sample_count)
    ) / float(sample_count)
    unit = rng_obj.permutation(unit)
    gaussian = _gaussian_offsets_from_unit(unit)
    return mean_value + sigma_value * gaussian


def _cartesian_product_axis(values: np.ndarray, counts: tuple[int, ...], axis: int) -> np.ndarray:
    """Broadcast one 1D coordinate sample array across the full Cartesian product."""

    if not counts or any(count <= 0 for count in counts):
        return np.empty((0,), dtype=np.float64)

    axis_values = np.asarray(values, dtype=np.float64).reshape(-1)
    inner = int(np.prod(counts[axis + 1 :], dtype=np.int64)) if axis + 1 < len(counts) else 1
    outer = int(np.prod(counts[:axis], dtype=np.int64)) if axis > 0 else 1
    return np.tile(np.repeat(axis_values, inner), outer)


def generate_random_profiles(
    num_samples,
    divergence_sigma,
    bw_sigma,
    lambda0,
    bandwidth,
    *,
    rng: np.random.Generator | int | None = None,
):
    """Generate low-discrepancy Gaussian beam profiles with antithetic pairing."""

    sample_count = max(int(num_samples), 0)
    if sample_count == 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty, empty, empty, empty

    rng_obj = _coerce_rng(rng)
    pair_count = sample_count // 2

    if pair_count > 0:
        base_unit = _latin_hypercube_unit_samples(
            pair_count,
            _PROFILE_DIMENSIONS,
            rng=rng_obj,
        )
        anti_unit = 1.0 - base_unit
        unit = np.empty((sample_count, _PROFILE_DIMENSIONS), dtype=np.float64)
        unit[0 : 2 * pair_count : 2, :] = base_unit
        unit[1 : 2 * pair_count : 2, :] = anti_unit
    else:
        unit = np.empty((sample_count, _PROFILE_DIMENSIONS), dtype=np.float64)

    if sample_count % 2 == 1:
        unit[-1, :] = 0.5

    gaussian = _gaussian_offsets_from_unit(unit)

    theta_array = divergence_sigma * gaussian[:, 0]
    phi_array = divergence_sigma * gaussian[:, 1]
    beam_x_array = bw_sigma * gaussian[:, 2]
    beam_y_array = bw_sigma * gaussian[:, 3]
    wavelength_array = lambda0 + (lambda0 * bandwidth) * gaussian[:, 4]

    return (
        np.asarray(beam_x_array, dtype=np.float64),
        np.asarray(beam_y_array, dtype=np.float64),
        np.asarray(theta_array, dtype=np.float64),
        np.asarray(phi_array, dtype=np.float64),
        np.asarray(wavelength_array, dtype=np.float64),
    )


def generate_stratified_profiles(
    *,
    x_mean: float,
    x_sigma: float,
    x_samples: int,
    y_mean: float,
    y_sigma: float,
    y_samples: int,
    dx_mean: float,
    dx_sigma: float,
    dx_samples: int,
    dz_mean: float,
    dz_sigma: float,
    dz_samples: int,
    lambda_mean: float,
    lambda_sigma: float,
    lambda_samples: int,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return equal-weight Cartesian-product Gaussian beam samples.

    User-facing axes are ``(x, y, dx, dz, lambda)``. Internally ``dx`` maps to
    ``phi_array`` and ``dz`` maps to ``theta_array``.
    """

    counts = (
        max(int(x_samples), 0),
        max(int(y_samples), 0),
        max(int(dx_samples), 0),
        max(int(dz_samples), 0),
        max(int(lambda_samples), 0),
    )
    if any(count <= 0 for count in counts):
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty, empty, empty, empty, empty

    rng_obj = _coerce_rng(rng)
    x_values = sample_stratified_gaussian_1d(
        counts[0],
        mean=x_mean,
        sigma=x_sigma,
        rng=rng_obj,
    )
    y_values = sample_stratified_gaussian_1d(
        counts[1],
        mean=y_mean,
        sigma=y_sigma,
        rng=rng_obj,
    )
    dx_values = sample_stratified_gaussian_1d(
        counts[2],
        mean=dx_mean,
        sigma=dx_sigma,
        rng=rng_obj,
    )
    dz_values = sample_stratified_gaussian_1d(
        counts[3],
        mean=dz_mean,
        sigma=dz_sigma,
        rng=rng_obj,
    )
    lambda_values = sample_stratified_gaussian_1d(
        counts[4],
        mean=lambda_mean,
        sigma=lambda_sigma,
        rng=rng_obj,
    )

    total_count = int(np.prod(counts, dtype=np.int64))
    weights = np.full(total_count, 1.0 / float(total_count), dtype=np.float64)

    return (
        _cartesian_product_axis(x_values, counts, 0),
        _cartesian_product_axis(y_values, counts, 1),
        _cartesian_product_axis(dz_values, counts, 3),
        _cartesian_product_axis(dx_values, counts, 2),
        _cartesian_product_axis(lambda_values, counts, 4),
        weights,
    )

