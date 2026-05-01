from __future__ import annotations

from ra_sim.gui import controllers
from ra_sim.gui import manual_geometry
from ra_sim.gui import peak_selection
from ra_sim.utils.pbi2_ht_shift_cif import DISORDERED_PHASE_SOURCE_LABEL


def test_disordered_phase_label_survives_controller_normalization():
    assert (
        controllers.normalize_bragg_qr_source_label(DISORDERED_PHASE_SOURCE_LABEL)
        == DISORDERED_PHASE_SOURCE_LABEL
    )
    assert controllers.normalize_bragg_qr_source_label("DISORDERED_PHASE") == (
        DISORDERED_PHASE_SOURCE_LABEL
    )


def test_disordered_phase_label_survives_manual_geometry_normalization():
    assert (
        manual_geometry.normalize_bragg_qr_source_label(DISORDERED_PHASE_SOURCE_LABEL)
        == DISORDERED_PHASE_SOURCE_LABEL
    )
    assert manual_geometry.normalize_bragg_qr_source_label("DISORDERED_PHASE") == (
        DISORDERED_PHASE_SOURCE_LABEL
    )


def test_disordered_phase_label_survives_peak_selection_normalization():
    assert (
        peak_selection._normalize_peak_source_label(DISORDERED_PHASE_SOURCE_LABEL)
        == DISORDERED_PHASE_SOURCE_LABEL
    )
    assert peak_selection._peak_overlay_source_label(DISORDERED_PHASE_SOURCE_LABEL) == (
        DISORDERED_PHASE_SOURCE_LABEL
    )
