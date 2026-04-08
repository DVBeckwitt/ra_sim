import numpy as np

from ra_sim.io.data_loading import load_parameters, save_all_parameters


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


def _required_parameter_vars():
    return [
        _Var(1.0),
        _Var(2.0),
        _Var(3.0),
        _Var(4.0),
        _Var(5.0),
        _Var(6.0),
        _Var(7.0),
        _Var(8.0),
        _Var(9.0),
        _Var(10.0),
        _Var(11.0),
        _Var(12.0),
        _Var(13.0),
        _Var(14.0),
        _Var(15.0),
        _Var(16.0),
        _Var(17.0),
        _Var(18.0),
        _Var(19.0),
        _Var(20.0),
    ]


def test_save_all_parameters_omits_removed_stratified_sampling_fields(tmp_path) -> None:
    file_path = tmp_path / "params.npy"
    vars_ = _required_parameter_vars()

    save_all_parameters(
        file_path,
        *vars_,
        resolution_var=_Var("Custom"),
        custom_samples_var=_Var("64"),
        rod_points_per_gz_var=_Var("250"),
        bandwidth_percent_var=_Var("0.7"),
        beam_sampling_method_var=_Var("stratified_gaussian"),
        beam_sampling_seed_var=_Var("42"),
        stratified_sampling_vars={"x_mean": _Var("0.1"), "x_samples": _Var("3")},
    )

    saved = np.load(file_path, allow_pickle=True).item()

    assert saved["sampling_count"] == 64
    assert saved["rod_points_per_gz"] == 250
    assert saved["bandwidth_percent"] == 0.7
    assert "beam_sampling_method" not in saved
    assert "beam_sampling_seed" not in saved
    assert "x_mean" not in saved
    assert "x_samples" not in saved


def test_load_parameters_ignores_removed_stratified_sampling_fields(tmp_path) -> None:
    file_path = tmp_path / "params.npy"
    np.save(
        file_path,
        {
            "sampling_resolution": "Custom",
            "sampling_count": 48,
            "rod_points_per_gz": 275,
            "bandwidth_percent": 0.6,
            "beam_sampling_method": "stratified_gaussian",
            "beam_sampling_seed": 99,
            "x_mean": 0.1,
            "x_samples": 4,
        },
    )

    vars_ = _required_parameter_vars()
    resolution_var = _Var("Low")
    custom_samples_var = _Var("12")
    rod_points_var = _Var(100)
    bandwidth_var = _Var(0.2)
    sampling_method_var = _Var("random_gaussian")
    sampling_seed_var = _Var("0")
    legacy_stratified_vars = {"x_mean": _Var("0"), "x_samples": _Var("1")}

    message = load_parameters(
        file_path,
        *vars_,
        resolution_var=resolution_var,
        custom_samples_var=custom_samples_var,
        rod_points_per_gz_var=rod_points_var,
        bandwidth_percent_var=bandwidth_var,
        beam_sampling_method_var=sampling_method_var,
        beam_sampling_seed_var=sampling_seed_var,
        stratified_sampling_vars=legacy_stratified_vars,
    )

    assert message == "Parameters loaded from parameters.npy"
    assert resolution_var.get() == "Custom"
    assert custom_samples_var.get() == "48"
    assert rod_points_var.get() == 275
    assert bandwidth_var.get() == 0.6
    assert sampling_method_var.get() == "random_gaussian"
    assert sampling_seed_var.get() == "0"
    assert legacy_stratified_vars["x_mean"].get() == "0"
    assert legacy_stratified_vars["x_samples"].get() == "1"
