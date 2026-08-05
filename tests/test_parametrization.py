"""Tests for the binned Cl_pp parametrization and the CAMB results cache.

Uses a stub in place of CAMB where possible so the bulk of the suite runs
without a cosmology environment; the CAMB-dependent tests are skipped if camb
is not importable.
"""

import numpy as np
import pytest

from lensing_fisher import clpp_template
from lensing_fisher.local_lens import BinnedLensingTheory, basis_matrix, smooth_indicator


# --------------------------------------------------------------------------
# Bin basis
# --------------------------------------------------------------------------

def test_basis_matrix_shape_and_localization():
    edges = np.array([2.0, 20.0, 60.0, 200.0, 1000.0])
    ells = np.arange(0, 1200)
    B = basis_matrix(edges, ells, steepness=2.0)

    assert B.shape == (ells.size, edges.size - 1)
    # Each basis function is ~1 well inside its own bin and ~0 well inside others.
    for j in range(edges.size - 1):
        mid = 0.5 * (edges[j] + edges[j + 1])
        i_mid = int(mid)
        assert B[i_mid, j] > 0.99
        for k in range(edges.size - 1):
            if k != j:
                assert B[i_mid, k] < 0.01


def test_basis_matrix_partition_of_unity():
    """Non-uniform contiguous bins tile to 1.

    The one-sided end bins make this hold everywhere, endpoints included, which
    is what makes a uniform q exactly A_template.
    """
    edges = np.array([2.0, 10.0, 25.0, 60.0, 150.0, 400.0, 2000.0])
    ells = np.arange(0, 2500)

    total = basis_matrix(edges, ells, steepness=2.0).sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-6)


def test_basis_matrix_rejects_bad_edges():
    ells = np.arange(100)
    with pytest.raises(ValueError, match="strictly increasing"):
        basis_matrix([10.0, 5.0, 20.0], ells, 2.0)
    with pytest.raises(ValueError, match="at least 2 bins"):
        basis_matrix([10.0, 20.0], ells, 2.0)


def test_smooth_indicator_width():
    f = smooth_indicator(100.0, 50.0, steepness=5.0)
    assert f(100.0) > 0.99          # centre
    assert f(80.0) > 0.9            # inside
    assert f(120.0) > 0.9           # inside
    assert f(60.0) < 0.01           # outside
    assert f(140.0) < 0.01          # outside


# --------------------------------------------------------------------------
# Template loading / interpolation
# --------------------------------------------------------------------------

def _write_template(tmp_path, L, CL, name="tmpl.pkl"):
    import pickle
    p = tmp_path / name
    with open(p, "wb") as f:
        pickle.dump({"L": L, "CL_pp_fid": CL}, f)
    return p


def test_template_roundtrip(tmp_path):
    L = np.arange(2, 3000, dtype=float)
    CL = 1e-7 * (L / 100.0) ** -1.5
    path = _write_template(tmp_path, L, CL)

    L2, CL2 = clpp_template.load_template(path)
    got = clpp_template.interpolate_to_ells(L2, CL2, np.arange(2, 3000))
    assert np.allclose(got, CL, rtol=1e-6)


def test_template_extrapolation_is_a_power_law(tmp_path):
    """Beyond the template range the continuation must stay a sane power law.

    CAMB needs C_fid out to lmax + lens_margin, typically ~1000 past where a
    template pickle stops; an unconstrained cubic can swing wildly there.
    """
    L = np.arange(2, 2000, dtype=float)
    CL = 1e-7 * (L / 100.0) ** -2.0
    path = _write_template(tmp_path, L, CL)
    L2, CL2 = clpp_template.load_template(path)

    ells = np.arange(0, 4600)
    got = clpp_template.interpolate_to_ells(L2, CL2, ells)

    assert got[0] == 0.0
    assert np.all(got[1:] > 0)                      # never negative
    assert np.all(np.isfinite(got))
    # Exact power law in, so extrapolation should recover it closely.
    expected_tail = 1e-7 * (ells[3000:] / 100.0) ** -2.0
    assert np.allclose(got[3000:], expected_tail, rtol=1e-3)


def test_template_rejects_malformed(tmp_path):
    import pickle
    bad = tmp_path / "bad.pkl"
    with open(bad, "wb") as f:
        pickle.dump({"L": np.arange(10)}, f)
    with pytest.raises(ValueError, match="must be a dict with keys"):
        clpp_template.load_template(bad)


# --------------------------------------------------------------------------
# Frozen C_fid semantics + cache (needs CAMB)
# --------------------------------------------------------------------------

camb = pytest.importorskip("camb", reason="CAMB not installed")


@pytest.fixture(scope="module")
def camb_setup():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.37, ombh2=0.02233, omch2=0.1198, tau=0.054, mnu=0.06, omk=0)
    pars.InitPower.set_params(As=1e-10 * np.exp(3.043), ns=0.9652)
    pars.set_for_lmax(1000, lens_potential_accuracy=1, lens_margin=200)
    pars.WantTensors = False
    fid = {"H0": 67.37, "ombh2": 0.02233, "omch2": 0.1198,
           "logA": 3.043, "ns": 0.9652, "tau": 0.054}
    clpp_fid = clpp_template.clpp_fid_from_camb(pars, pars.max_l)
    return pars, fid, clpp_fid


def test_q_zero_returns_c_fid(camb_setup):
    """q = 0 must lens with exactly C_fid, not with zero and not with CAMB's own."""
    pars, fid, clpp_fid = camb_setup
    edges = np.array([2.0, 50.0, 200.0, 600.0, float(pars.max_l)])
    f = BinnedLensingTheory(pars, edges, clpp_fid, steepness=2.0)

    p = dict(fid, **{f"clpp_{i+1}": 0.0 for i in range(len(edges) - 1)})
    f(p, 800, 2)
    # Reach into the closure's stored fiducial: it is what q=0 reproduces.
    assert np.array_equal(f.clpp_fid, clpp_fid)
    assert np.all(clpp_fid[2:] > 0)


def test_single_bin_perturbation_is_local_and_fractional(camb_setup):
    pars, fid, clpp_fid = camb_setup
    edges = np.array([2.0, 50.0, 200.0, 600.0, float(pars.max_l)])
    f = BinnedLensingTheory(pars, edges, clpp_fid, steepness=2.0)

    q = {f"clpp_{i+1}": 0.0 for i in range(len(edges) - 1)}
    q["clpp_2"] = 0.1                                     # bin [50, 200)
    B, fidarr = f.basis, f.clpp_fid
    qv = np.array([q[f"clpp_{i+1}"] for i in range(len(edges) - 1)])
    clpp = fidarr * (1.0 + B @ qv)

    ells = np.arange(len(clpp))
    inner = (ells > 60) & (ells < 190)
    outside = (ells > 250) & (ells < 590)
    # 10% up inside the perturbed bin ...
    assert np.allclose(clpp[inner] / fidarr[inner], 1.1, atol=1e-3)
    # ... and untouched elsewhere.
    assert np.allclose(clpp[outside] / fidarr[outside], 1.0, atol=1e-3)


def test_cfid_is_frozen_under_cosmology_variation(camb_setup):
    """The guard on the whole design: varying cosmology must not move C_fid.

    The lensing kernel is held fixed while the unlensed spectra respond, which
    is exactly what TemplateLensingCAMB does in the A_template chain.
    """
    pars, fid, clpp_fid = camb_setup
    edges = np.array([2.0, 50.0, 200.0, 600.0, float(pars.max_l)])
    f = BinnedLensingTheory(pars, edges, clpp_fid, steepness=2.0)

    base = dict(fid, **{f"clpp_{i+1}": 0.0 for i in range(len(edges) - 1)})
    shifted = dict(base, omch2=base["omch2"] * 1.05)

    d0 = f(base, 800, 2)
    fid_after_first = f.clpp_fid.copy()
    d1 = f(shifted, 800, 2)

    # C_fid untouched by the cosmology change ...
    assert np.array_equal(f.clpp_fid, fid_after_first)
    # ... while the lensed spectra genuinely responded.
    assert not np.allclose(d0["TT"], d1["TT"], rtol=1e-6)


def test_camb_cache_counts_distinct_cosmologies(camb_setup):
    pars, fid, clpp_fid = camb_setup
    edges = np.array([2.0, 50.0, 200.0, 600.0, float(pars.max_l)])
    f = BinnedLensingTheory(pars, edges, clpp_fid, steepness=2.0, cache_size=4)

    base = dict(fid, **{f"clpp_{i+1}": 0.0 for i in range(len(edges) - 1)})

    # Ten calls varying only bin amplitudes: one cosmology, so one get_results.
    for v in np.linspace(-0.1, 0.1, 10):
        f(dict(base, clpp_3=float(v)), 800, 2)
    assert f.cache_stats["misses"] == 1
    assert f.cache_stats["hits"] == 9

    # A nuisance-style extra key must not invalidate it either.
    f(dict(base, A_planck=1.01), 800, 2)
    assert f.cache_stats["misses"] == 1

    # A genuine cosmology change must.
    f(dict(base, H0=68.0), 800, 2)
    assert f.cache_stats["misses"] == 2


def test_bbn_predictor_survives_every_cache_miss(camb_setup):
    """set_cosmology rederives YHe, so the BBN predictor must be re-supplied.

    The regression: _camb_results called set_cosmology without the config's
    camb.set_cosmology block, so the first cache miss silently dropped
    PArthENoPE for CAMB's PRIMAT default and every evaluation after it — the
    Fisher and the Cobaya chains would then describe different damping tails.
    """
    pars, fid, clpp_fid = camb_setup
    edges = np.array([2.0, 50.0, 200.0, 600.0, float(pars.max_l)])
    extras = {"mnu": 0.06, "omk": 0.0,
              "bbn_predictor": "PArthENoPE_880.2_standard.dat",
              "nnu": 3.044, "num_massive_neutrinos": 1}

    f = BinnedLensingTheory(pars, edges, clpp_fid, steepness=2.0,
                            cosmology_kwargs=extras)
    base = dict(fid, **{f"clpp_{i+1}": 0.0 for i in range(len(edges) - 1)})

    reference = camb.CAMBparams()
    reference.set_cosmology(H0=fid["H0"], ombh2=fid["ombh2"], omch2=fid["omch2"],
                            tau=fid["tau"], **extras)

    for shift in (1.0, 1.02, 0.98):                 # three separate cache misses
        f._camb_results(dict(base, omch2=base["omch2"] * shift))
        assert f.camb_pars.YHe == pytest.approx(reference.YHe, rel=1e-12)


def test_free_cosmology_parameter_overrides_its_fixed_value(camb_setup):
    """A parameter promoted into the Fisher vector must not stay pinned."""
    pars, fid, clpp_fid = camb_setup
    edges = np.array([2.0, 50.0, 200.0, 600.0, float(pars.max_l)])
    f = BinnedLensingTheory(pars, edges, clpp_fid, steepness=2.0,
                            cosmology_kwargs={"mnu": 0.06, "omk": 0.0})

    base = dict(fid, **{f"clpp_{i+1}": 0.0 for i in range(len(edges) - 1)})
    f._camb_results(dict(base, mnu=0.12))
    assert f.camb_pars.omnuh2 == pytest.approx(0.12 / 93.14, rel=2e-2)
