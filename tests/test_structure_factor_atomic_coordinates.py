import numpy as np

from ra_sim.StructureFactor.AtomicCoordinates import get_atomic_coordinates


def _assert_coords_in_unit_cell(atoms):
    coords = np.array([(x, y, z) for _, x, y, z in atoms], dtype=np.float64)
    assert np.all(coords >= 0.0)
    assert np.all(coords < 1.0)
    return coords


def test_get_atomic_coordinates_wraps_fractional_sites_and_deduplicates_symmetry_images():
    positions = np.array(
        [
            [0.2, 0.3, 0.4],
            [0.8, 0.1, 0.2],
        ],
        dtype=np.float64,
    )
    operations = {
        "rotations": np.array([np.eye(3, dtype=int)] * 4),
        "translations": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1e-12, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }

    cell, atoms = get_atomic_coordinates(
        lattice=None,
        positions=positions,
        numbers=None,
        space_group_operations=operations,
        atomic_labels=["Bi", "Se1"],
        cell_params=(4.0, 4.0, 10.0, 90.0, 90.0, 120.0),
    )

    assert cell == (4.0, 4.0, 10.0, 90.0, 90.0, 120.0)
    assert [label for label, *_ in atoms] == ["Bi", "Bi", "Se1", "Se1"]
    np.testing.assert_allclose(
        np.array([(x, y, z) for _, x, y, z in atoms], dtype=np.float64),
        [
            (0.2, 0.3, 0.4),
            (0.7, 0.3, 0.4),
            (0.8, 0.1, 0.2),
            (0.3, 0.1, 0.2),
        ],
    )

    coords = _assert_coords_in_unit_cell(atoms)

    unique_coords = []
    for coord in coords:
        if not unique_coords:
            unique_coords.append(coord)
            continue
        deltas = np.abs(np.asarray(unique_coords) - coord)
        periodic_deltas = np.minimum(deltas, 1.0 - deltas)
        if not np.any(np.all(periodic_deltas <= 1e-8, axis=1)):
            unique_coords.append(coord)

    assert len(unique_coords) == len(atoms)


def test_get_atomic_coordinates_keeps_distinct_labels_on_same_wrapped_site():
    operations = {
        "rotations": np.array([np.eye(3, dtype=int)] * 3),
        "translations": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1e-12, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }

    _, atoms = get_atomic_coordinates(
        lattice=None,
        positions=np.array(
            [
                [0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4],
            ],
            dtype=np.float64,
        ),
        numbers=None,
        space_group_operations=operations,
        atomic_labels=["Bi", "Se1"],
        cell_params=(4.0, 4.0, 10.0, 90.0, 90.0, 120.0),
    )

    assert [label for label, *_ in atoms] == ["Bi", "Se1"]
    np.testing.assert_allclose(
        np.array([(x, y, z) for _, x, y, z in atoms], dtype=np.float64),
        [
            (0.2, 0.3, 0.4),
            (0.2, 0.3, 0.4),
        ],
    )
    _assert_coords_in_unit_cell(atoms)
