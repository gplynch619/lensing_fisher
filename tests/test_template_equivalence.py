"""The Fisher and the A_template chain must describe the same model.

``TemplateLensingCAMB._get_template_lensed_cmb_spectra`` lenses with
``A_template * <frozen template>`` via ``get_lensed_cls_with_spectrum``, holding
the lensing kernel fixed while cosmology varies the unlensed spectra. The binned
parametrization is the same operation with per-bin amplitudes, so a *uniform*
``q_i = a`` across bins spanning the whole ell range must be indistinguishable
from ``A_template = 1 + a``.

If this test fails, L_eff is describing a different measurement than the one the
chain reports, and the comparison is not apples-to-apples. This is exactly the
correspondence the pre-2026 code broke: it lensed with ``sum q_i B_i`` and no
fiducial at all, so ``q = 0`` was an unlensed CMB rather than the template.
"""

import numpy as np
import pytest

camb = pytest.importorskip("camb", reason="CAMB not installed")

from lensing_fisher import clpp_template  # noqa: E402
from lensing_fisher.local_lens import BinnedLensingTheory, basis_matrix  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.37, ombh2=0.02233, omch2=0.1198, tau=0.054, mnu=0.06, omk=0)
    pars.InitPower.set_params(As=1e-10 * np.exp(3.043), ns=0.9652)
    pars.set_for_lmax(1500, lens_potential_accuracy=1, lens_margin=300)
    pars.WantTensors = False
    fid = {"H0": 67.37, "ombh2": 0.02233, "omch2": 0.1198,
           "logA": 3.043, "ns": 0.9652, "tau": 0.054}
    clpp_fid = clpp_template.clpp_fid_from_camb(pars, pars.max_l)
    return pars, fid, clpp_fid


@pytest.mark.parametrize("a", [0.0, 0.05, -0.05, 0.2])
def test_uniform_q_equals_A_template(setup, a):
    """Uniform q = a reproduces lensing by (1 + a) * template, spectrum by spectrum."""
    pars, fid, clpp_fid = setup

    # Bins must span the full CAMB ell range for the correspondence to be exact:
    # anything above the top edge stays at C_fid and would not be scaled.
    edges = np.concatenate([np.geomspace(2.0, 1500.0, 20), [float(pars.max_l)]])
    f = BinnedLensingTheory(pars, edges, clpp_fid, steepness=2.0)

    n_bins = len(edges) - 1
    pars_q = dict(fid, **{f"clpp_{i+1}": a for i in range(n_bins)})
    got = f(pars_q, 1400, 2)

    # What TemplateLensingCAMB does for A_template = 1 + a.
    results = camb.get_results(pars)
    expected = results.get_lensed_cls_with_spectrum(
        (1.0 + a) * clpp_fid, CMB_unit="muK"
    )

    for spec, ix in (("TT", 0), ("EE", 1), ("BB", 2), ("TE", 3)):
        ref = expected[2:1401, ix]
        assert np.allclose(got[spec], ref, rtol=2e-3, atol=1e-4 * np.abs(ref).max()), (
            f"{spec} mismatch at a={a}"
        )


def test_basis_tiles_exactly_including_endpoints(setup):
    """sum_i B_i == 1 across the whole range, endpoints included.

    This is what makes uniform q exactly A_template. Plain sigmoid top-hats give
    0.5 at the outermost edges, which is why the end bins are one-sided.
    """
    pars, _, _ = setup
    edges = np.concatenate([np.geomspace(2.0, 1500.0, 20), [float(pars.max_l)]])
    B = basis_matrix(edges, np.arange(pars.max_l + 1), steepness=2.0)
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-6)


def test_cosmology_moves_unlensed_not_the_kernel(setup):
    """Same structure as the chain: theta moves the unlensed spectra only."""
    pars, fid, clpp_fid = setup

    edges = np.concatenate([np.geomspace(2.0, 1500.0, 20), [float(pars.max_l)]])
    f = BinnedLensingTheory(pars, edges, clpp_fid, steepness=2.0)
    n_bins = len(edges) - 1

    base = dict(fid, **{f"clpp_{i+1}": 0.0 for i in range(n_bins)})
    moved = dict(base, logA=base["logA"] + 0.02)

    d0 = f(base, 1400, 2)
    d1 = f(moved, 1400, 2)

    assert np.array_equal(f.clpp_fid, clpp_fid)          # kernel frozen
    assert not np.allclose(d0["TT"], d1["TT"], rtol=1e-8)  # spectra respond
