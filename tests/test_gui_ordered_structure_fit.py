import math

import numpy as np

from ra_sim.gui import ordered_structure_fit
from ra_sim.gui.state import OrderedStructureFitSnapshot


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


def test_solve_positive_weighted_scale_returns_expected_value() -> None:
    primary = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
    fixed = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    measured = 2.5 * primary + fixed
    weights = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64)

    scale = ordered_structure_fit.solve_positive_weighted_scale(
        measured,
        primary,
        fixed_component=fixed,
        weights=weights,
    )

    assert math.isclose(scale, 2.5, rel_tol=1.0e-10, abs_tol=1.0e-10)


def test_build_hybrid_ordered_structure_mask_includes_bragg_and_specular_rois() -> None:
    non_specular = np.array(
        [
            [8.0, 6.0, 5.0, 0.0, 1.0, 0.0, 1.0],
            [7.0, 7.0, 5.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    specular = np.array(
        [
            [10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 2.0],
            [9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )

    mask = ordered_structure_fit.build_hybrid_ordered_structure_mask(
        image_shape=(24, 24),
        primary_hit_tables=[non_specular, specular],
        max_reflections=24,
        tube_width_scale=1.0,
        specular_width_scale=2.5,
        equal_peak_weights=True,
    )

    assert mask.roi_count == 2
    assert mask.bragg_roi_count == 1
    assert mask.specular_roi_count == 1
    assert (1, 0, 1) in mask.selected_hkls
    assert (0, 0, 2) in mask.selected_hkls
    assert bool(mask.pixel_mask[5, 6])
    assert bool(mask.pixel_mask[10, 10])
    assert bool(mask.pixel_mask[0, 0]) is False


def test_fit_ordered_structure_parameters_recovers_synthetic_solution() -> None:
    image_shape = (6, 6)
    yy, xx = np.mgrid[0:image_shape[0], 0:image_shape[1]]
    occ_pattern = (xx + 1.0) / 10.0
    z_pattern = (yy + 1.0) / 12.0
    debye_pattern = ((xx - yy) ** 2 + 1.0) / 20.0
    fixed = np.full(image_shape, 0.15, dtype=np.float64)

    true_occ = 0.62
    true_z = 0.31
    true_debye = 0.24
    true_scale = 1.7
    measured = true_scale * (
        true_occ * occ_pattern + true_z * z_pattern + true_debye * debye_pattern
    ) + fixed

    mask = ordered_structure_fit.OrderedStructureMask(
        pixel_mask=np.ones(image_shape, dtype=bool),
        weight_map=np.ones(image_shape, dtype=np.float64),
        roi_count=3,
        bragg_roi_count=2,
        specular_roi_count=1,
    )
    specs = [
        ordered_structure_fit.OrderedStructureParameterSpec(
            name="occ:A",
            value=0.85,
            lower=0.0,
            upper=1.0,
        ),
        ordered_structure_fit.OrderedStructureParameterSpec(
            name="atom:A:z",
            value=0.12,
            lower=0.0,
            upper=0.5,
        ),
        ordered_structure_fit.OrderedStructureParameterSpec(
            name="debye_x",
            value=0.05,
            lower=0.0,
            upper=1.0,
        ),
    ]

    def _simulate(parameter_values: dict[str, float]):
        primary = (
            parameter_values["occ:A"] * occ_pattern
            + parameter_values["atom:A:z"] * z_pattern
            + parameter_values["debye_x"] * debye_pattern
        )
        return primary, fixed

    result = ordered_structure_fit.fit_ordered_structure_parameters(
        measured_image=measured,
        mask=mask,
        parameter_specs=specs,
        simulate_components=_simulate,
        coarse_downsample_factor=2,
        loss="soft_l1",
        f_scale=2.0,
        coarse_max_nfev=20,
        polish_max_nfev=20,
        restarts=1,
    )

    assert result.acceptance_passed is True
    assert result.final_objective < result.initial_objective
    assert math.isclose(result.scale, true_scale, rel_tol=0.08, abs_tol=0.08)
    assert math.isclose(result.parameter_values["occ:A"], true_occ, rel_tol=0.08, abs_tol=0.08)
    assert math.isclose(result.parameter_values["atom:A:z"], true_z, rel_tol=0.08, abs_tol=0.08)
    assert math.isclose(result.parameter_values["debye_x"], true_debye, rel_tol=0.1, abs_tol=0.1)


def test_apply_and_restore_ordered_structure_values_update_runtime_vars() -> None:
    occ_vars = [_Var(1.0), _Var(0.5)]
    atom_vars = [{"x": _Var(0.1), "y": _Var(0.2), "z": _Var(0.3)}]
    debye_x_var = _Var(0.0)
    debye_y_var = _Var(0.0)
    scale_var = _Var(1.0)

    ordered_structure_fit.apply_ordered_structure_values(
        {
            "occ:site_1": 0.75,
            "occ:site_2": 0.25,
            "atom:site_1:x": 0.11,
            "atom:site_1:z": 0.33,
            "debye_x": 0.2,
            "debye_y": 0.4,
        },
        occupancy_vars=occ_vars,
        occupancy_param_names=["occ:site_1", "occ:site_2"],
        atom_site_vars=atom_vars,
        atom_param_names=[{"x": "atom:site_1:x", "y": "atom:site_1:y", "z": "atom:site_1:z"}],
        debye_x_var=debye_x_var,
        debye_y_var=debye_y_var,
        ordered_scale_var=scale_var,
        scale_value=1.8,
    )

    assert occ_vars[0].get() == 0.75
    assert occ_vars[1].get() == 0.25
    assert atom_vars[0]["x"].get() == 0.11
    assert atom_vars[0]["y"].get() == 0.2
    assert atom_vars[0]["z"].get() == 0.33
    assert debye_x_var.get() == 0.2
    assert debye_y_var.get() == 0.4
    assert scale_var.get() == 1.8

    snapshot = OrderedStructureFitSnapshot(
        occupancy_values=[1.0, 0.5],
        atom_site_values=[(0.1, 0.2, 0.3)],
        debye_x=0.0,
        debye_y=0.0,
        ordered_scale=1.0,
    )
    restored = ordered_structure_fit.restore_ordered_structure_snapshot(
        snapshot,
        occupancy_vars=occ_vars,
        atom_site_vars=atom_vars,
        debye_x_var=debye_x_var,
        debye_y_var=debye_y_var,
        ordered_scale_var=scale_var,
    )

    assert restored is True
    assert occ_vars[0].get() == 1.0
    assert occ_vars[1].get() == 0.5
    assert atom_vars[0]["x"].get() == 0.1
    assert atom_vars[0]["y"].get() == 0.2
    assert atom_vars[0]["z"].get() == 0.3
    assert debye_x_var.get() == 0.0
    assert debye_y_var.get() == 0.0
    assert scale_var.get() == 1.0
