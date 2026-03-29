from hbn_fitter import fitter as legacy_fitter
from ra_sim.hbn_fitter import fitter as packaged_fitter


def test_legacy_hbn_fitter_wrapper_reexports_packaged_surface() -> None:
    assert legacy_fitter.HBNFitterGUI is packaged_fitter.HBNFitterGUI
    assert (
        legacy_fitter.build_hbn_fitter_bundle_payload
        is packaged_fitter.build_hbn_fitter_bundle_payload
    )
