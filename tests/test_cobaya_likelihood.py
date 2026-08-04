"""The Cobaya wrapper's wiring to build_likelihoods.

The chain and the Fisher share one dataset definition, so this class is the
single point where that promise can silently break. It did: the wrapper called
``build_likelihoods(cfg)`` with the whole config instead of the ``likelihoods:``
block, and unpacked a 3-tuple into two names. Neither is caught by anything else
in the suite, because nothing else constructs the wrapper.

A stub backend keeps this free of candl, clipy and CAMB.
"""

import numpy as np
import pytest

from lensing_fisher.likelihoods import BACKENDS, build_likelihoods

cobaya = pytest.importorskip("cobaya")

from lensing_fisher.cobaya_likelihood import CandlClipyCombined  # noqa: E402


class _StubLike:
    """The bits of a candl likelihood the wrapper actually touches."""

    def __init__(self, ell_min, ell_max, nuisance, logl=-1.0):
        self.ell_min, self.ell_max = ell_min, ell_max
        self.required_nuisance_parameters = list(nuisance)
        self.priors = []
        self._logl = logl
        self.seen = None

    def log_like(self, pars):
        self.seen = pars
        return self._logl


@pytest.fixture
def stub_backend(monkeypatch):
    built = {}

    def make_stub(*, ell_min=2, ell_max=1000, nuisance=(), logl=-1.0):
        like = _StubLike(ell_min, ell_max, nuisance, logl)
        built[id(like)] = like
        return like

    monkeypatch.setitem(BACKENDS, "stub", make_stub)
    monkeypatch.setitem(
        __import__("lensing_fisher.likelihoods", fromlist=["_NUISANCE_EXTRACTORS"])
        ._NUISANCE_EXTRACTORS,
        "stub",
        lambda like: {n: 1.0 for n in like.required_nuisance_parameters},
    )
    return built


SPEC = {
    "planck_like": {"backend": "stub", "ell_max": 1000, "nuisance": ["A_planck"], "logl": -3.0},
    "act_like": {"backend": "stub", "ell_min": 600, "ell_max": 2500,
                 "nuisance": ["A_act", "P_act"], "logl": -5.0},
}


def test_build_likelihoods_returns_three(stub_backend):
    """Guards the contract the wrapper got wrong."""
    names, likes, nuisance = build_likelihoods(SPEC)
    assert names == ["planck_like", "act_like"]
    assert len(likes) == 2
    assert set(nuisance) == {"A_planck", "A_act", "P_act"}


def test_initialize_wires_names_likes_and_nuisance(stub_backend, monkeypatch):
    monkeypatch.setattr("lensing_fisher.cobaya_likelihood.effective_ell_max",
                        lambda like, tol: int(like.ell_max))

    like = CandlClipyCombined({"likelihoods": SPEC, "stop_at_error": True})

    assert like._names == ["planck_like", "act_like"]
    assert [type(x).__name__ for x in like._likes] == ["_StubLike", "_StubLike"]
    assert set(like._nuisance_fid) == {"A_planck", "A_act", "P_act"}
    # ell_min is the minimum over likelihoods, ell_max the maximum
    assert (like._ell_min, like._ell_max) == (2, 2500)
    assert like._theory_lmax == 2500
    assert set(like.get_can_support_params()) == {"A_planck", "A_act", "P_act"}
    assert like.get_requirements() == {"Cl": {s: 2500 for s in ("tt", "te", "ee", "bb")}}


def test_logp_sums_and_slices_per_likelihood(stub_backend, monkeypatch):
    """Each likelihood gets its own Dl dict over its own ell range."""
    monkeypatch.setattr("lensing_fisher.cobaya_likelihood.effective_ell_max",
                        lambda like, tol: int(like.ell_max))
    wrapper = CandlClipyCombined({"likelihoods": SPEC, "stop_at_error": True})

    ells = np.arange(0, 2501)
    cl = {"ell": ells, "tt": ells * 1.0, "te": ells * 2.0, "ee": ells * 3.0, "bb": ells * 0.0}

    class _Provider:
        def get_Cl(self, **kwargs):
            return cl

    wrapper.provider = _Provider()
    assert wrapper.logp(A_planck=1.0, A_act=1.0, P_act=1.0) == pytest.approx(-8.0)

    planck, act = wrapper._likes
    assert planck.seen["Dl"]["ell"][0] == 2 and planck.seen["Dl"]["ell"][-1] == 1000
    assert act.seen["Dl"]["ell"][0] == 600 and act.seen["Dl"]["ell"][-1] == 2500
    # sliced from the right offset, not from zero
    assert act.seen["Dl"]["TT"][0] == pytest.approx(600.0)
    # and the dicts are independent objects
    assert planck.seen["Dl"] is not act.seen["Dl"]


def test_missing_dataset_and_likelihoods_raises(stub_backend):
    with pytest.raises(ValueError, match="needs either 'dataset_file'"):
        CandlClipyCombined({"stop_at_error": True})
