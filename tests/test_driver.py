"""The driver's individual steps. Pure functions; no CAMB or likelihoods needed."""

import numpy as np
import pytest

from lensing_fisher import driver


# ----------------------------------------------------------------------
# Bin layout
# ----------------------------------------------------------------------

def test_resolve_explicit_edges():
    edges = driver.resolve_bin_edges({"edges": [2, 10, 40, 200]})
    assert np.allclose(edges, [2, 10, 40, 200])


def test_resolve_generated_edges():
    log = driver.resolve_bin_edges({"edges": {"start": 2, "stop": 2000, "n": 50}})
    assert log.size == 51
    assert np.isclose(log[0], 2) and np.isclose(log[-1], 2000)
    # log spacing: constant ratio between successive edges
    assert np.allclose(np.diff(np.log(log)), np.log(log[1] / log[0]))

    lin = driver.resolve_bin_edges(
        {"edges": {"start": 0, "stop": 100, "n": 4, "spacing": "linear"}}
    )
    assert np.allclose(lin, [0, 25, 50, 75, 100])


def test_resolve_rejects_unknown_spacing():
    with pytest.raises(ValueError, match="log' or 'linear"):
        driver.resolve_bin_edges({"edges": {"start": 2, "stop": 10, "n": 3, "spacing": "sqrt"}})


def test_catchall_bin_extends_to_lmax():
    edges = driver.add_catchall_bin(np.array([2.0, 100.0, 2000.0]), camb_lmax=4550)
    assert np.isclose(edges[-1], 4550)
    assert edges.size == 4                      # one extra bin appended


def test_catchall_bin_is_a_noop_when_already_at_lmax():
    edges = np.array([2.0, 100.0, 4550.0])
    assert np.array_equal(driver.add_catchall_bin(edges, camb_lmax=4550), edges)


# ----------------------------------------------------------------------
# Parameter assembly
# ----------------------------------------------------------------------

COSMO_CFG = {
    "fiducial": {"H0": 67.37, "ombh2": 0.02233, "omch2": 0.1198,
                 "logA": 3.043, "ns": 0.9652, "tau": 0.054},
    "step_sizes_relative": {k: 0.01 for k in
                            ("H0", "ombh2", "omch2", "logA", "ns", "tau")},
    "nuisance_step_relative": 0.01,
}


def test_assemble_parameters_orders_cosmology_nuisance_bins():
    edges = np.array([2.0, 100.0, 1000.0, 4550.0])
    params = driver.assemble_parameters(
        COSMO_CFG, {"step_size": 0.05}, {"A_planck": 1.0, "Tcal": 1.0}, edges
    )

    assert params.names == (
        ["H0", "ombh2", "omch2", "logA", "ns", "tau"]
        + ["A_planck", "Tcal"]
        + ["clpp_1", "clpp_2", "clpp_3"]
    )
    assert params.bin_names == ["clpp_1", "clpp_2", "clpp_3"]
    assert all(params.fiducial[b] == 0.0 for b in params.bin_names)
    assert params.steps.size == len(params.names)
    # relative steps applied to the fiducial value
    assert np.isclose(params.steps[0], 0.01 * 67.37)
    # bins take the fractional step as-is
    assert np.allclose(params.steps[-3:], 0.05)


def test_assemble_parameters_requires_a_cosmology_step():
    cfg = {"fiducial": {"H0": 67.0, "ombh2": 0.022, "omch2": 0.12,
                        "logA": 3.0, "ns": 0.96, "tau": 0.054},
           "step_sizes_relative": {"H0": 0.01}}
    with pytest.raises(ValueError, match="no entry in step_sizes"):
        driver.assemble_parameters(cfg, {}, {}, np.array([2.0, 100.0, 1000.0]))


def test_nuisance_never_duplicates_a_cosmology_name():
    edges = np.array([2.0, 100.0, 1000.0])
    params = driver.assemble_parameters(
        COSMO_CFG, {}, {"tau": 0.05, "A_planck": 1.0}, edges
    )
    assert params.nuisance_names == ["A_planck"]
    assert params.names.count("tau") == 1


# ----------------------------------------------------------------------
# Adaptive bin step sizes
# ----------------------------------------------------------------------

def test_clpp_steps_scalar():
    steps = driver.clpp_step_sizes(0.07, np.array([2.0, 100.0, 1000.0]))
    assert np.allclose(steps, 0.07)


def test_clpp_steps_track_sigma_from_a_previous_fisher(tmp_path):
    """Narrow, weakly-constrained bins should get proportionally larger steps."""
    import pickle

    prev_edges = np.array([2.0, 100.0, 1000.0])
    names = ["H0", "clpp_1", "clpp_2"]
    # bin 2 ten times more weakly constrained than bin 1
    F = np.diag([1.0, 1.0 / 0.01**2, 1.0 / 0.1**2])
    path = tmp_path / "prev.pkl"
    pickle.dump({"fisher_matrix": F, "param_names": names, "bin_edges": prev_edges},
                open(path, "wb"))

    steps = driver.clpp_step_sizes(
        {"from_fisher": str(path), "target_sigma_frac": 0.5, "min": 0.0, "max": 10.0},
        prev_edges,
    )
    assert np.isclose(steps[0], 0.5 * 0.01)
    assert np.isclose(steps[1], 0.5 * 0.1)


def test_clpp_steps_interpolate_onto_a_changed_grid(tmp_path):
    """The grid moves every iteration, so sigma must be re-mapped in L."""
    import pickle

    prev_edges = np.geomspace(2.0, 2000.0, 6)
    names = ["H0"] + [f"clpp_{i+1}" for i in range(5)]
    F = np.diag([1.0] + list(1.0 / np.linspace(0.05, 0.5, 5) ** 2))
    path = tmp_path / "prev.pkl"
    pickle.dump({"fisher_matrix": F, "param_names": names, "bin_edges": prev_edges},
                open(path, "wb"))

    new_edges = np.geomspace(2.0, 2000.0, 9)          # 8 bins, different grid
    steps = driver.clpp_step_sizes(
        {"from_fisher": str(path), "target_sigma_frac": 0.3, "min": 0.0, "max": 10.0},
        new_edges,
    )
    assert steps.size == 8
    assert np.all(np.diff(steps) > 0)                 # still rising with L


def test_theory_lmax_guard_uses_requested_lmax_not_max_l():
    """camb_pars.max_l is lmax + lens_margin, where lensed spectra are unreliable.

    Checking window support against max_l would accept bandpowers sitting in the
    lens_margin region. CAMB returns numbers there, so the error would be silent.
    """
    import inspect
    from lensing_fisher import driver

    src = inspect.getsource(driver.run)
    call = src[src.index("check_theory_lmax"):src.index("bin_edges = resolve")]
    assert "set_for_lmax" in call and "max_l" not in call


def test_camb_cosmology_extras_reach_set_cosmology():
    """bbn_predictor must be settable: CAMB's default differs from the chains'."""
    camb = pytest.importorskip("camb")
    from lensing_fisher.driver import build_camb_params

    cosmology = {"H0": 67.37, "ombh2": 0.02233, "omch2": 0.1198,
                 "tau": 0.054, "logA": 3.043, "ns": 0.9652, "mnu": 0.06}
    cfg = {"set_for_lmax": {"lmax": 500, "lens_potential_accuracy": 1},
           "matter_power": {"kmax": 2}, "accuracy": {"AccuracyBoost": 1.0}}

    default = build_camb_params(cfg, cosmology)
    parthenope = build_camb_params(
        {**cfg, "set_cosmology": {"bbn_predictor": "PArthENoPE_880.2_standard.dat"}},
        cosmology,
    )
    assert default.YHe != parthenope.YHe


@pytest.mark.parametrize("epsilon", [0.0, 1e-9], ids=["exact", "near"])
def test_clpp_steps_refuse_a_rank_deficient_fisher(tmp_path, epsilon):
    """Per-bin sigma does not exist when adjacent bins are degenerate.

    The lensed CMB resolves only a handful of C_L^pp modes, so on a fine grid the
    marginalized block is singular and sqrt(diag(inv(F))) is NaN. Feeding those
    NaNs forward produced a Fisher matrix of quiet garbage, so this must raise.

    Both spellings occur: an exactly flat direction makes numpy raise, while the
    real thing is a zero that finite differences have blurred to the *negative*
    side, which inverts happily and returns a negative variance. `epsilon` is
    that blur — the run this guards against had 23 such eigenvalues.
    """
    import pickle

    prev_edges = np.array([2.0, 100.0, 1000.0, 2000.0])
    names = ["H0"] + [f"clpp_{i+1}" for i in range(3)]
    # Bins 2 and 3 enter only through their sum: the (0, 1, -1) direction is flat.
    F = np.zeros((4, 4))
    F[0, 0] = 1.0
    F[1, 1] = 100.0
    F[2:, 2:] = np.array([[1.0, 1.0 + epsilon], [1.0 + epsilon, 1.0]])
    path = tmp_path / "prev.pkl"
    pickle.dump({"fisher_matrix": F, "param_names": names, "bin_edges": prev_edges},
                open(path, "wb"))

    with pytest.raises(ValueError, match="rank-deficient"):
        driver.clpp_step_sizes(
            {"from_fisher": str(path), "target_sigma_frac": 0.3}, prev_edges
        )
