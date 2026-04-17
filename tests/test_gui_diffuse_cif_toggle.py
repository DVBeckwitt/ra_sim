from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ra_sim.gui import diffuse_cif_toggle


DIFFUSE_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ra_sim"
    / "gui"
    / "diffuse_cif_toggle.py"
)


class _FakeTkWidget:
    def __init__(self) -> None:
        self.manager = ""
        self.pack_calls: list[dict[str, object]] = []
        self.destroyed = False

    def winfo_manager(self) -> str:
        return self.manager

    def pack(self, **kwargs) -> None:
        self.pack_calls.append(dict(kwargs))
        self.manager = "pack"

    def destroy(self) -> None:
        self.destroyed = True


class _FakeWindow:
    def __init__(self) -> None:
        self.parent = None
        self.title_text = None
        self.transient_parent = None
        self.protocols: dict[str, object] = {}
        self.destroyed = False

    def title(self, text: str) -> None:
        self.title_text = text

    def transient(self, parent) -> None:
        self.transient_parent = parent

    def protocol(self, name: str, callback) -> None:
        self.protocols[name] = callback

    def destroy(self) -> None:
        self.destroyed = True


class _FakeFrame:
    created: list["_FakeFrame"] = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.pack_calls: list[dict[str, object]] = []
        self.__class__.created.append(self)

    def pack(self, **kwargs) -> None:
        self.pack_calls.append(dict(kwargs))


class _EmbeddedCanvas(FigureCanvasAgg):
    created: list["_EmbeddedCanvas"] = []

    def __init__(self, figure, master=None) -> None:
        self.master = master
        self.widget = _FakeTkWidget()
        self.__class__.created.append(self)
        super().__init__(figure)

    def get_tk_widget(self):
        return self.widget


def test_diffuse_viewer_source_uses_embedded_tk_canvas_only() -> None:
    source = DIFFUSE_SOURCE_PATH.read_text(encoding="utf-8")

    assert "manager.show" not in source
    assert "matplotlib.pyplot as plt" not in source
    assert "FigureCanvasTkAgg" in source


def test_open_diffuse_cif_toggle_algebraic_embeds_canvas_in_toplevel(
    monkeypatch,
    tmp_path,
) -> None:
    cif_path = tmp_path / "test.cif"
    cif_path.write_text("data_test", encoding="utf-8")
    window = _FakeWindow()
    _FakeFrame.created = []
    _EmbeddedCanvas.created = []

    monkeypatch.setattr(
        diffuse_cif_toggle,
        "_read_lattice_from_cif",
        lambda _path: (4.557, 7.0),
        raising=False,
    )
    monkeypatch.setattr(
        diffuse_cif_toggle,
        "_infer_iodine_z_like_diffuse",
        lambda _path: 0.25,
        raising=False,
    )
    monkeypatch.setattr(
        diffuse_cif_toggle,
        "_get_base_curves",
        lambda **_kwargs: {
            (0, 0): {
                "L": np.array([0.0, 1.0, 2.0], dtype=float),
                "F2": np.array([1.0, 2.0, 3.0], dtype=float),
            },
            (1, 0): {
                "L": np.array([0.0, 1.0, 2.0], dtype=float),
                "F2": np.array([2.0, 3.0, 4.0], dtype=float),
            },
        },
        raising=False,
    )
    monkeypatch.setattr(
        diffuse_cif_toggle,
        "analytical_ht_intensity_for_pair",
        lambda L_vals, F2_vals, *_args, **_kwargs: np.asarray(F2_vals, dtype=float),
        raising=False,
    )

    fig = diffuse_cif_toggle.open_diffuse_cif_toggle_algebraic(
        cif_path=str(cif_path),
        occ=[1.0],
        p_values=[0.1, 0.2, 0.3],
        w_values=[40.0, 30.0, 30.0],
        lambda_angstrom=1.5406,
        mx=2,
        parent=object(),
        tk_module=SimpleNamespace(
            Toplevel=lambda parent: setattr(window, "parent", parent) or window,
            BOTH="both",
        ),
        ttk_module=SimpleNamespace(Frame=_FakeFrame),
        figure_canvas_cls=_EmbeddedCanvas,
    )

    ui = fig._ra_sim_diffuse_ui
    assert isinstance(ui["canvas"], _EmbeddedCanvas)
    assert ui["canvas_widget"].pack_calls == [{"fill": "both", "expand": True}]
    assert _FakeFrame.created[0].pack_calls == [{"fill": "both", "expand": True}]
    assert window.title_text == f"Diffuse HT (algebraic) - {cif_path.name}"
    assert window.protocols["WM_DELETE_WINDOW"] is ui["close"]
    assert window.destroyed is False
    assert ui["canvas_widget"].destroyed is False

    window.protocols["WM_DELETE_WINDOW"]()

    assert window.destroyed is True
    assert ui["canvas_widget"].destroyed is True


def test_open_diffuse_cif_toggle_algebraic_cleans_up_partial_window_on_init_error(
    monkeypatch,
    tmp_path,
) -> None:
    cif_path = tmp_path / "test.cif"
    cif_path.write_text("data_test", encoding="utf-8")
    window = _FakeWindow()
    _FakeFrame.created = []
    _EmbeddedCanvas.created = []

    monkeypatch.setattr(
        diffuse_cif_toggle,
        "_read_lattice_from_cif",
        lambda _path: (4.557, 7.0),
        raising=False,
    )
    monkeypatch.setattr(
        diffuse_cif_toggle,
        "_infer_iodine_z_like_diffuse",
        lambda _path: 0.25,
        raising=False,
    )
    monkeypatch.setattr(
        diffuse_cif_toggle,
        "_get_base_curves",
        lambda **_kwargs: {
            (0, 0): {
                "L": np.array([0.0, 1.0, 2.0], dtype=float),
                "F2": np.array([1.0, 2.0, 3.0], dtype=float),
            },
            (1, 0): {
                "L": np.array([0.0, 1.0, 2.0], dtype=float),
                "F2": np.array([2.0, 3.0, 4.0], dtype=float),
            },
        },
        raising=False,
    )
    monkeypatch.setattr(
        diffuse_cif_toggle,
        "analytical_ht_intensity_for_pair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("viewer boom")),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="viewer boom"):
        diffuse_cif_toggle.open_diffuse_cif_toggle_algebraic(
            cif_path=str(cif_path),
            occ=[1.0],
            p_values=[0.1, 0.2, 0.3],
            w_values=[40.0, 30.0, 30.0],
            lambda_angstrom=1.5406,
            mx=2,
            parent=object(),
            tk_module=SimpleNamespace(
                Toplevel=lambda parent: setattr(window, "parent", parent) or window,
                BOTH="both",
            ),
            ttk_module=SimpleNamespace(Frame=_FakeFrame),
            figure_canvas_cls=_EmbeddedCanvas,
        )

    assert len(_FakeFrame.created) == 1
    assert len(_EmbeddedCanvas.created) == 1
    assert window.destroyed is True
    assert _EmbeddedCanvas.created[0].widget.destroyed is True
    assert _FakeFrame.created[0].pack_calls == [{"fill": "both", "expand": True}]
