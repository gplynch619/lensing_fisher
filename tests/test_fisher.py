"""FisherMatrix against analytically known answers, serial and under MPI."""

import numpy as np
import pytest

from lensing_fisher import FisherMatrix


def _gaussian_loglike(A, names):
    """logL = -1/2 x^T A x, whose Fisher matrix is exactly A."""
    def loglike(p):
        x = np.array([p[n] for n in names])
        return -0.5 * float(x @ A @ x)
    return loglike


def test_recovers_a_known_hessian():
    names = ["a", "b", "c"]
    A = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 3.0]])
    fm = FisherMatrix(_gaussian_loglike(A, names), {n: 0.0 for n in names}, np.full(3, 0.01))
    assert np.allclose(fm.compute(), A, atol=1e-8)


def test_diagonal_matches_inverse_variance():
    sigmas = np.array([1.0, 2.0, 0.5])
    names = ["a", "b", "c"]
    A = np.diag(1.0 / sigmas**2)
    fm = FisherMatrix(_gaussian_loglike(A, names), {n: 0.0 for n in names}, np.full(3, 0.01))
    fm.compute()
    assert np.allclose(fm.parameter_errors(), sigmas, rtol=1e-6)


def test_fiducial_likelihood_evaluated_once():
    """Every diagonal element needs logL at the fiducial; it must be cached."""
    names = ["a", "b", "c", "d"]
    A = np.eye(4)
    calls = {"n": 0}
    base = _gaussian_loglike(A, names)

    def counting(p):
        if all(p[n] == 0.0 for n in names):
            calls["n"] += 1
        return base(p)

    FisherMatrix(counting, {n: 0.0 for n in names}, np.full(4, 0.01)).compute()
    assert calls["n"] == 1


def test_task_bucketing_covers_every_element_exactly_once():
    """Whatever the rank count, the buckets must tile the upper triangle."""
    names = [f"p{i}" for i in range(7)]
    A = np.eye(7)

    for size in (1, 2, 3, 5, 8):
        seen = []
        for rank in range(size):
            fm = FisherMatrix(_gaussian_loglike(A, names), {n: 0.0 for n in names},
                              np.full(7, 0.01), cached_params=["p0", "p3"])
            fm.size, fm.rank, fm.use_mpi = size, rank, size > 1
            seen += fm._tasks()
        assert len(seen) == len(set(seen)) == 7 * 8 // 2


def test_cached_params_are_grouped_in_execution_order():
    """A rank must walk runs of elements sharing a cached-parameter set."""
    names = [f"p{i}" for i in range(6)]
    fm = FisherMatrix(_gaussian_loglike(np.eye(6), names), {n: 0.0 for n in names},
                      np.full(6, 0.01), cached_params=["p0"])
    keys = [frozenset(i for i in t if i == 0) for t in fm._tasks()]
    # The no-cosmology bucket comes first and is contiguous.
    first_nonempty = next(i for i, k in enumerate(keys) if k)
    assert all(not k for k in keys[:first_nonempty])
    assert all(k for k in keys[first_nonempty:])


def test_refuses_to_save_all_zeros(tmp_path):
    names = ["a", "b"]
    fm = FisherMatrix(lambda p: 0.0, {n: 0.0 for n in names}, np.full(2, 0.01))
    fm.compute()
    with pytest.raises(RuntimeError, match="all-zero"):
        fm.save(str(tmp_path), "f.pkl")


def test_metadata_may_not_shadow_core_keys(tmp_path):
    names = ["a", "b"]
    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    fm = FisherMatrix(_gaussian_loglike(A, names), {n: 0.0 for n in names}, np.full(2, 0.01))
    fm.compute()
    with pytest.raises(ValueError, match="shadow core keys"):
        fm.save(str(tmp_path), "f.pkl", metadata={"fisher_matrix": None})


def test_save_roundtrip_carries_provenance(tmp_path):
    import pickle
    names = ["a", "b"]
    A = np.array([[2.0, 0.5], [0.5, 3.0]])
    fm = FisherMatrix(_gaussian_loglike(A, names), {n: 0.0 for n in names}, np.full(2, 0.01))
    fm.compute()
    path = fm.save(str(tmp_path), "f", metadata={"bin_edges": np.array([1.0, 2.0, 3.0])})

    saved = pickle.load(open(path, "rb"))
    assert np.allclose(saved["fisher_matrix"], A, atol=1e-8)
    assert saved["param_names"] == names
    assert np.allclose(saved["bin_edges"], [1.0, 2.0, 3.0])
