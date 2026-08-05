"""Tests for the Fisher -> L_eff -> next-bin-edges chain. No CAMB required."""

import numpy as np
import pytest

from lensing_fisher import analysis

_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------
# Parameter blocks and marginalization
# --------------------------------------------------------------------------

def test_clpp_indices_sort_numerically_not_lexically():
    names = ["H0", "tau"] + [f"clpp_{i}" for i in range(1, 13)]
    idx = analysis.clpp_indices(names)
    assert [names[i] for i in idx] == [f"clpp_{i}" for i in range(1, 13)]
    # The trap: lexical order would put clpp_10 before clpp_2.
    assert names[idx[1]] == "clpp_2"
    assert names[idx[9]] == "clpp_10"


def test_marginalize_is_schur_not_slice():
    """Marginalizing must invert-slice-invert, not slice."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    F = A @ A.T + 5 * np.eye(5)          # SPD
    names = ["H0", "A_planck", "clpp_1", "clpp_2", "clpp_3"]

    got = analysis.marginalize(F, names)
    cov = np.linalg.inv(F)
    expected = np.linalg.inv(cov[2:, 2:])

    assert np.allclose(got, expected)
    # And it genuinely differs from conditioning (a plain slice).
    assert not np.allclose(got, F[2:, 2:])


def test_marginalize_equals_slice_when_uncorrelated():
    F = np.diag([3.0, 4.0, 1.0, 2.0])
    names = ["H0", "A_planck", "clpp_1", "clpp_2"]
    assert np.allclose(analysis.marginalize(F, names), np.diag([1.0, 2.0]))


def test_clpp_indices_require_clpp_params():
    with pytest.raises(ValueError, match="no clpp_"):
        analysis.clpp_indices(["H0", "tau"])


# --------------------------------------------------------------------------
# Jacobian and weights
# --------------------------------------------------------------------------

def test_bin_averaged_ratio_is_one_when_template_is_fiducial():
    edges = np.array([2.0, 50.0, 200.0, 1000.0])
    L = np.arange(2, 1001, dtype=float)
    c = 1e-7 * (L / 100.0) ** -2
    r = analysis.bin_averaged_ratio(edges, c, c, L)
    assert np.allclose(r, 1.0)


def test_bin_averaged_ratio_weights_by_cfid_across_wide_bins():
    """A wide bin's r must be the C_fid-weighted mean, not the centre value."""
    edges = np.array([100.0, 1000.0])
    L = np.arange(100, 1001, dtype=float)
    fid = (L / 100.0) ** -3.0                 # steeply falling: low L dominates
    tem = fid * (1.0 + L / 1000.0)            # ratio rises across the bin

    r = analysis.bin_averaged_ratio(edges, tem, fid, L)[0]
    centre_value = 1.0 + 550.0 / 1000.0

    # C_fid weighting pulls r toward the low-L end, well below the centre value.
    assert r < centre_value
    # Bins are half-open [lo, hi), so compare against the same support.
    m = (L >= edges[0]) & (L < edges[1])
    assert np.isclose(r, _trapz(tem[m], L[m]) / _trapz(fid[m], L[m]))


def test_information_per_bin_matches_total():
    F = np.array([[4.0, 1.0], [1.0, 3.0]])
    r = np.array([1.0, 2.0])
    w = analysis.information_per_bin(F, r)
    assert np.isclose(w.sum(), r @ F @ r)


def test_weight_density_divides_by_bin_width():
    """The correction the notebook did not need: per-bin totals -> a density.

    Two bins carrying equal information but of different widths must give equal
    integrated weight, so the wider one has the lower density.
    """
    edges = np.array([0.0, 100.0, 300.0])
    L = np.arange(0, 300, dtype=float)
    w_bins = np.array([1.0, 1.0])

    w = analysis.weight_density(w_bins, edges, L)
    narrow = w[(L >= 0) & (L < 100)]
    wide = w[(L >= 100) & (L < 300)]

    assert np.allclose(narrow, 1.0 / 100.0)
    assert np.allclose(wide, 1.0 / 200.0)
    assert np.isclose(_trapz(narrow, L[:100]) * 100 / 99, _trapz(wide, L[100:]) * 200 / 199, rtol=1e-2)


def test_weight_density_uniform_bins_reduces_to_constant_rescaling():
    """For uniform bins the width division is a constant, so L_eff is unchanged.

    This is why the notebook's formula gave the right answer for the old
    uniform 40-bin grid despite omitting the division.
    """
    edges = np.arange(0.0, 1001.0, 100.0)
    L = np.arange(0, 1000, dtype=float)
    w_bins = np.array([5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.4])

    density = analysis.weight_density(w_bins, edges, L)
    naive = density * 100.0                      # what "assign w_j to every L" gives

    assert np.isclose(*[analysis.effective_L(x, L)[0] for x in (density, naive)])


# --------------------------------------------------------------------------
# Effective L
# --------------------------------------------------------------------------

def test_effective_L_of_a_uniform_distribution():
    L = np.linspace(0.0, 100.0, 10001)
    w = np.ones_like(L)
    L_eff, lo, hi = analysis.effective_L(w, L)
    assert np.isclose(L_eff, 50.0, atol=1e-6)
    assert np.isclose(lo, 16.0, atol=0.05)
    assert np.isclose(hi, 84.0, atol=0.05)


def test_effective_L_of_a_gaussian():
    L = np.linspace(0.0, 400.0, 40001)
    w = np.exp(-0.5 * ((L - 160.0) / 40.0) ** 2)
    L_eff, lo, hi = analysis.effective_L(w, L)
    assert np.isclose(L_eff, 160.0, atol=0.1)
    assert np.isclose(hi - L_eff, 40.0, atol=0.5)     # ~1 sigma each side
    assert np.isclose(L_eff - lo, 40.0, atol=0.5)


def test_effective_L_rejects_nonpositive_total():
    L = np.linspace(1.0, 10.0, 100)
    with pytest.raises(ValueError, match="non-positive"):
        analysis.effective_L(np.zeros_like(L), L)


# --------------------------------------------------------------------------
# Bin placement
# --------------------------------------------------------------------------

def _bin_information(w, L, edges):
    return np.array([
        _trapz(w[(L >= a) & (L <= b)], L[(L >= a) & (L <= b)])
        for a, b in zip(edges[:-1], edges[1:])
    ])


def test_next_bin_edges_equalizes_information_when_unconstrained():
    L = np.linspace(2.0, 2000.0, 20000)
    w = np.exp(-0.5 * ((np.log(L) - np.log(80.0)) / 0.8) ** 2) / L   # peaked near L~80

    edges = analysis.next_bin_edges(w, L, n_bins=20, min_width=1.0)
    info = _bin_information(w, L, edges)

    assert edges.size == 21
    assert np.isclose(edges[0], 2.0) and np.isclose(edges[-1], 2000.0)
    assert info.std() / info.mean() < 0.02


def test_min_width_constraint_binds_gracefully():
    """When min_width binds, bins stop being equal-information — by design.

    Near a sharply peaked kernel the equal-information solution wants bins
    narrower than min_width. The forward/backward passes then widen them, which
    necessarily redistributes information. The contract is that the grid stays
    valid (contiguous, min-width respected, spanning the range), not that
    equalization survives.
    """
    L = np.linspace(2.0, 2000.0, 20000)
    w = np.exp(-0.5 * ((np.log(L) - np.log(80.0)) / 0.8) ** 2) / L

    edges = analysis.next_bin_edges(w, L, n_bins=20, min_width=8.0)
    info = _bin_information(w, L, edges)

    assert np.all(np.diff(edges) >= 8.0 - 1e-9)
    assert np.isclose(edges[0], 2.0) and np.isclose(edges[-1], 2000.0)
    # Still far more even than a uniform grid would be.
    uniform = _bin_information(w, L, np.linspace(2.0, 2000.0, 21))
    assert info.std() / info.mean() < 0.3 * (uniform.std() / uniform.mean())


def test_next_bin_edges_puts_narrow_bins_where_the_weight_is():
    L = np.linspace(2.0, 2000.0, 20000)
    w = np.exp(-0.5 * ((np.log(L) - np.log(80.0)) / 0.8) ** 2) / L

    edges = analysis.next_bin_edges(w, L, n_bins=20, min_width=8.0)
    widths = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    near_peak = widths[np.argmin(np.abs(centres - 80.0))]
    at_tail = widths[-1]
    assert near_peak < at_tail / 5.0


def test_next_bin_edges_respects_min_width_even_for_a_sharp_peak():
    L = np.linspace(0.0, 1000.0, 10001)
    w = np.exp(-0.5 * ((L - 500.0) / 2.0) ** 2)      # nearly a delta function

    edges = analysis.next_bin_edges(w, L, n_bins=25, min_width=10.0)
    assert np.all(np.diff(edges) >= 10.0 - 1e-9)
    assert np.isclose(edges[0], 0.0) and np.isclose(edges[-1], 1000.0)


def test_next_bin_edges_rejects_infeasible_request():
    L = np.linspace(0.0, 100.0, 1001)
    w = np.ones_like(L)
    with pytest.raises(ValueError, match="cannot fit"):
        analysis.next_bin_edges(w, L, n_bins=20, min_width=10.0)


def test_next_bin_edges_is_a_fixed_point_for_already_equal_bins():
    """Feeding back a converged grid should return (nearly) the same grid."""
    L = np.linspace(2.0, 2000.0, 20000)
    w = np.exp(-0.5 * ((np.log(L) - np.log(80.0)) / 0.8) ** 2) / L

    first = analysis.next_bin_edges(w, L, n_bins=20, min_width=8.0)
    second = analysis.next_bin_edges(w, L, n_bins=20, min_width=8.0)
    assert analysis.edges_converged(first, second, tol=1e-6)


def test_edges_converged():
    a = np.array([1.0, 2.0, 3.0])
    assert analysis.edges_converged(a, a + 0.5, tol=1.0)
    assert not analysis.edges_converged(a, a + 2.0, tol=1.0)
    assert not analysis.edges_converged(a, np.array([1.0, 2.0]), tol=1.0)


# --------------------------------------------------------------------------
# End-to-end on a synthetic Fisher pickle
# --------------------------------------------------------------------------

def test_summarize_roundtrip(tmp_path):
    import pickle

    n_bins = 12
    edges = np.geomspace(2.0, 2000.0, n_bins + 1)
    names = ["H0", "tau", "A_planck"] + [f"clpp_{i+1}" for i in range(n_bins)]

    # Information concentrated in the bins around L ~ 80.
    centres = 0.5 * (edges[:-1] + edges[1:])
    diag = np.exp(-0.5 * ((np.log(centres) - np.log(80.0)) / 0.7) ** 2) + 1e-3
    F = np.eye(len(names))
    F[3:, 3:] = np.diag(diag)

    payload = {
        "fisher_matrix": F,
        "param_names": names,
        "bin_edges": edges,
        "clpp_fid": 1e-7 * (np.arange(2500) / 100.0 + 1) ** -2,
        "parametrization": "local_lens",
    }
    path = tmp_path / "f.pkl"
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)

    out = analysis.summarize(path)
    assert out["r"].shape == (n_bins,)
    assert np.allclose(out["r"], 1.0)             # no template given => r == 1
    assert 20.0 < out["L_eff"] < 400.0
    assert out["L_minus"] < out["L_eff"] < out["L_plus"]
    assert out["sigma_A_template"] > 0


def test_bin_averaged_ratio_rejects_underspecified_bins():
    """One sample integrates to zero, so r_j would be a silent NaN."""
    edges = np.array([2.0, 2.5, 3.0, 100.0])
    L = np.arange(2, 101, dtype=float)          # nothing inside [2.5, 3.0)
    c = 1e-7 * (L / 100.0) ** -2
    with pytest.raises(ValueError, match="at least 2"):
        analysis.bin_averaged_ratio(edges, c, c, L)


def test_summarize_handles_bins_narrower_than_a_multipole(tmp_path):
    """The iteration-0 grid: log-spaced from L=2, several bins hold no integer.

    r_j is a quadrature over smooth functions, so summarize must sample the bins
    rather than rely on the integer grid.
    """
    import pickle

    edges = np.concatenate([np.geomspace(2.0, 200.0, 21), [400.0]])
    n_bins = edges.size - 1
    assert min(int(np.floor(edges[j + 1]) - np.ceil(edges[j]) + 1)
               for j in range(n_bins)) <= 0          # some bin holds no integer

    names = ["H0"] + [f"clpp_{i+1}" for i in range(n_bins)]
    F = np.diag(np.concatenate([[1.0], np.linspace(1.0, 5.0, n_bins)]))
    clpp_fid = np.concatenate([[0.0, 0.0], 1e-7 * (np.arange(2, 501) / 100.0) ** -2])

    template = tmp_path / "tem.pkl"
    L = np.arange(clpp_fid.size)
    pickle.dump({"L": L, "CL_pp_fid": clpp_fid * 0.99}, open(template, "wb"))

    fisher = {"fisher_matrix": F, "param_names": names, "bin_edges": edges,
              "clpp_fid": clpp_fid, "steepness": 2.0}
    out = analysis.summarize(fisher, template_file=str(template))

    assert np.all(np.isfinite(out["r"]))
    assert np.allclose(out["r"], 0.99, rtol=2e-3)    # uniform rescaling -> r = 0.99
    assert np.isfinite(out["L_eff"])


def _summarize_fixture(last_edge, tail_fraction):
    """A peaked weight distribution plus a catch-all bin of adjustable width.

    ``tail_fraction`` is the catch-all's share of the total information; the real
    SPA iteration 1 had 2.1% of it spread over a bin 7551 wide.
    """
    edges = np.concatenate([np.linspace(2.0, 200.0, 20), [last_edge]])
    n = edges.size - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    diag = np.exp(-0.5 * ((centres - 100.0) / 40.0) ** 2) + 1e-3
    diag[-1] = tail_fraction * diag[:-1].sum()
    F = np.diag(np.concatenate([[1.0], diag]))
    names = ["H0"] + [f"clpp_{i+1}" for i in range(n)]
    return {"fisher_matrix": F, "param_names": names, "bin_edges": edges,
            "clpp_fid": np.ones(int(last_edge) + 1), "steepness": 2.0}


def test_catchall_bin_is_excluded_from_the_moments():
    """A wide, near-empty final bin must not drag the mean.

    weight_density spreads a bin's information uniformly, so a bin running to
    CAMB's lmax contributes an enormous lever arm to <L>. Iteration 1 of the SPA
    run moved L_eff from 164 to 261 on this alone.
    """
    narrow = analysis.summarize(_summarize_fixture(400.0, 0.02))
    wide = analysis.summarize(_summarize_fixture(8550.0, 0.02))

    # Same physics, catch-all 20x wider: the moments must not notice.
    assert np.isclose(narrow["L_eff"], wide["L_eff"], rtol=1e-9)
    assert np.isclose(narrow["L_median"], wide["L_median"], rtol=1e-9)
    assert wide["excluded_catchall"] and wide["moment_L_max"] == 200.0

    # ... whereas including it is exactly the failure mode being guarded against.
    # The real run inflated L_eff by 1.59x (164.2 -> 261.2) on a 2.1% tail.
    leaky = analysis.summarize(_summarize_fixture(8550.0, 0.02),
                               exclude_catchall=False)
    assert leaky["L_eff"] > 1.25 * wide["L_eff"]
    assert wide["excluded_information"] < 0.05      # a 2% tail, reported


def test_median_is_reported_and_robust_to_the_tail():
    wide = analysis.summarize(_summarize_fixture(8550.0, 0.02), exclude_catchall=False)
    # The mean is dragged past its own 68% upper bound; the median is not.
    assert wide["L_eff"] > wide["L_plus"]
    assert wide["L_minus"] < wide["L_median"] < wide["L_plus"]


def test_weight_quantile_matches_a_known_cdf():
    L = np.linspace(0.0, 100.0, 1001)
    w = np.ones_like(L)                       # uniform -> quantile is linear in L
    assert np.isclose(analysis.weight_quantile(w, L, 0.5), 50.0, atol=1e-6)
    assert np.allclose(analysis.weight_quantile(w, L, [0.25, 0.75]), [25.0, 75.0],
                       atol=1e-6)
